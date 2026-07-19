import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional, Tuple,Union,List
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from nets.transformer import PreNormTransformer
from nets.encoder import PointEncoder
from diffusers.models.attention import BasicTransformerBlock

class DiffusionTransformer(nn.Module):
    def __init__(self,
                 data_dim: int,
                 model_dim: int = 128,
                 depth: int = 8,
                 num_heads: int = 8,
                 # ---- 条件与时间步相关：对外透出 ----
                 cond_in_dim: Optional[int] = None,   # 条件序列原始维度；为 None 表示不投影或与 model_dim 相同
                 cond_dim: int = 128,                # 时间条件维度，用于 time-attn-bias/token-shift
                 max_steps: int = 1000,
                 use_cross_attn: bool = False,       # 是否在骨干中启用交叉注意力
                 use_time_attn_bias: bool = True,    # 是否启用按头 logits 偏置
                 use_token_shift: bool = False,      # 是否对 token 表征做加性偏置
                 # ---- 其他 ----
                 scheduler: Optional[object] = None,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.0,
                 use_pos_emb: bool = True,
                 pos_emb_type: str = "sine",
                 max_len: int = 1024,
                 in_proj_type: str = "linear",       # 新增参数: "linear", "mlp", "siren"
                 in_proj_hidden: int = 128           # MLP/SIREN隐藏层维度
                 ):
        super().__init__()
        self.data_dim = data_dim
        self.model_dim = model_dim
        self.max_steps = max_steps
        self.use_pos_emb = use_pos_emb
        self.pos_emb_type = pos_emb_type
        self.max_len = max_len
        self.use_cross_attn = bool(use_cross_attn)

        # 根据 in_proj_type 选择不同的投影方式
        if in_proj_type == "linear":
            self.in_proj = nn.Linear(data_dim, model_dim)
        elif in_proj_type == "mlp":
            self.in_proj = nn.Sequential(
                nn.Linear(data_dim, in_proj_hidden),
                nn.GELU(),
                nn.Linear(in_proj_hidden, model_dim)
            )
        elif in_proj_type == "siren":
            self.in_proj = PointEncoder(data_dim,out_dim=model_dim)
        else:
            raise ValueError(f"Unknown in_proj_type: {in_proj_type}")

        # 可选：条件序列投影到 model_dim，只有在需要交叉注意力时才用得到
        need_cond_proj = (cond_in_dim is not None) and (cond_in_dim != model_dim)
        self.cond_proj = nn.Linear(cond_in_dim, model_dim) if need_cond_proj else None

        if use_pos_emb:
            if pos_emb_type == "learned":
                self.pos_emb = nn.Parameter(torch.zeros(1, max_len, model_dim))
                nn.init.trunc_normal_(self.pos_emb, std=0.02)
            elif pos_emb_type == "sine":
                self.register_buffer("pos_emb", self._build_sine_pos_emb(max_len, model_dim), persistent=False)
            else:
                raise ValueError(f"Unknown pos_emb_type: {pos_emb_type}")
        else:
            self.pos_emb = None

        self.backbone = PreNormTransformer(
            dim=model_dim, depth=depth, num_heads=num_heads,
            cond_dim=cond_dim, max_steps=max_steps,
            use_time_attn_bias=use_time_attn_bias,
            use_token_shift=use_token_shift,
            use_cross_attn=use_cross_attn,
            mlp_ratio=mlp_ratio, dropout=dropout,
        )
        self.out_proj = nn.Linear(model_dim, data_dim)

        if scheduler is None:
            scheduler = DDPMScheduler(num_train_timesteps=max_steps, prediction_type='epsilon',beta_schedule="squaredcos_cap_v2", clip_sample=False)
        self.set_scheduler(scheduler)

    def _build_sine_pos_emb(self, max_len, model_dim):
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / model_dim))
        pe = torch.zeros(max_len, model_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, max_len, D]

    def forward(self, noisy_x: torch.Tensor, t: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                # ---- 新增：条件序列与其掩码 ----
                cond: Optional[torch.Tensor] = None,                 # (B, Lc, Dc or D)
                cond_attn_mask: Optional[torch.Tensor] = None,       # (Lq,Lk)|(B,Lq,Lk)|(B,H,Lq,Lk)
                cond_key_padding_mask: Optional[torch.Tensor] = None # (B, Lc)
                ) -> torch.Tensor:
        return self.predict(noisy_x, t,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            cond=cond,
                            cond_attn_mask=cond_attn_mask,
                            cond_key_padding_mask=cond_key_padding_mask)

    def predict(self, noisy_x: torch.Tensor, t: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                cond: Optional[torch.Tensor] = None,
                cond_attn_mask: Optional[torch.Tensor] = None,
                cond_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if t.dtype != torch.long:
            t = t.long()

        h = self.in_proj(noisy_x)
        if self.use_pos_emb and self.pos_emb is not None:
            L = h.shape[1]
            h = h + self.pos_emb[:, :L, :]

        # 条件序列可选投影
        if cond is not None:
            if self.cond_proj is not None:
                cond = self.cond_proj(cond)
            else:
                # 若未显式提供 cond_in_dim 且不开启投影，则要求与 model_dim 一致
                assert cond.shape[-1] == self.model_dim, \
                    f"cond dim mismatch: got {cond.shape[-1]}, expect {self.model_dim}"

        h = self.backbone(h, t,
                          attn_mask=attn_mask,
                          key_padding_mask=key_padding_mask,
                          cond=cond if self.use_cross_attn else None,
                          cond_attn_mask=cond_attn_mask,
                          cond_key_padding_mask=cond_key_padding_mask)
        return self.out_proj(h)

    @torch.no_grad()
    def sample(self,
            shape: Tuple[int, int, int],
            attn_mask: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
            num_steps: Optional[int] = None,
            generator: Optional[torch.Generator] = None,
            # 采样期同样支持条件输入
            cond: Optional[torch.Tensor] = None,
            cond_attn_mask: Optional[torch.Tensor] = None,
            cond_key_padding_mask: Optional[torch.Tensor] = None,
            # 返回 n 等分的中间结果（None 则只返回最终样本）
            return_n_intermediate: Optional[int] = None,
            # 新增：选择中间结果的类型："state"=常规迭代状态，"x0"=一步直跳到 t0 的还原结果
            intermediate_mode: str = "state") -> Union[torch.Tensor, List[torch.Tensor]]:

        device = next(self.parameters()).device
        B, L, C = shape
        assert C == self.data_dim, f"data_dim mismatch: got {C}, expect {self.data_dim}"
        if num_steps is None:
            num_steps = self.max_steps

        self.scheduler.set_timesteps(num_steps, device=device)

        x = torch.randn((B, L, C), device=device) if generator is None \
            else torch.randn((B, L, C), device=device, generator=generator)

        # 计算需要记录的迭代索引
        collect_intermediates = False
        target_indices = set()
        if return_n_intermediate is not None:
            n = int(return_n_intermediate)
            if n <= 0:
                raise ValueError("return_n_intermediate must be a positive integer or None")
            timesteps_len = len(self.scheduler.timesteps)
            if n == 1:
                indices = [timesteps_len - 1]
            else:
                indices = [int(round(i * (timesteps_len - 1) / (n - 1))) for i in range(n)]
            target_indices = set(indices)
            collect_intermediates = True
            intermediates: List[torch.Tensor] = []

        # 主循环
        for step_idx, t in enumerate(self.scheduler.timesteps):
            t_batch = torch.full((B,), int(t), device=device, dtype=torch.long)

            model_out = self.predict(
                x, t_batch,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                cond=cond if self.use_cross_attn else None,
                cond_attn_mask=cond_attn_mask,
                cond_key_padding_mask=cond_key_padding_mask
            )

            step_out = self.scheduler.step(model_out, t, x)
            x = step_out.prev_sample

            if collect_intermediates and step_idx in target_indices:
                if intermediate_mode == "x0":
                    # 优先使用调度器提供的 pred_original_sample
                    if hasattr(step_out, "pred_original_sample") and step_out.pred_original_sample is not None:
                        intermediates.append(step_out.pred_original_sample.clone())
                    else:
                        # 无法可靠从 eps/velocity 自行复原 x0，明确报错以避免静默错误
                        raise RuntimeError(
                            "intermediate_mode='x0' 需要调度器在 step(...) 返回 pred_original_sample；"
                            "当前 scheduler 未提供该字段，无法安全地一步还原到 t0。"
                        )
                else:
                    # 默认返回常规去噪轨迹上的状态
                    intermediates.append(x.clone())

        if collect_intermediates:
            return intermediates
        return x

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
        return self