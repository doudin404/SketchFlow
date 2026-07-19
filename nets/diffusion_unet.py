import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional, Tuple,Union,List
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from nets.transformer_unet import TransformerUNet1D
from nets.encoder import PointEncoder
from diffusers.models.attention import BasicTransformerBlock

class DiffusionTransformer(nn.Module):
    def __init__(self,
                 data_dim: int,
                 model_dim: int = 128,
                 depth: int = 8,
                 num_heads: int = 8,
                 cond_in_dim: Optional[int] = None,
                 cond_dim: int = 128,
                 max_steps: int = 1000,
                 use_cross_attn: bool = False,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.0,
                 use_pos_emb=None,
                 in_proj_type=None,
                 # passthrough UNet params
                 dim_mults=(1.0, 1.25, 1.5, 2.0),
                 depths=(2,2,2,2), 
                 heads=(8,8,8,8)
                 ):
        super().__init__()
        self.data_dim = data_dim
        self.model_dim = model_dim
        self.max_steps = max_steps
        self.use_cross_attn = bool(use_cross_attn)

        # === 保留 cond 投影: 仅在 cross-attn 打开时用于 cond_tokens 维度对齐 ===
        need_cond_proj = (cond_in_dim is not None) and (cond_in_dim != model_dim)
        self.cond_proj = nn.Linear(cond_in_dim, model_dim) if need_cond_proj else None


        # === 替换骨干为 TransformerUNet1D（内部自带输入/时间条件处理） ===
        self.backbone = TransformerUNet1D(
            in_channels=data_dim,
            out_channels=data_dim,
            base_dim=model_dim,
            dim_mults=dim_mults,
            depths=depths,
            heads=heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            cond_vec_dim=cond_dim,                       # 这里的 cond_vec_dim 仅决定 AdaLN 向量维度
            cond_tok_dim=(model_dim if use_cross_attn else None),
            use_cross=use_cross_attn
        )

        # 由 UNet 已输出 data_dim，故设为恒等
        self.out_proj = nn.Identity()

    def _build_sine_pos_emb(self, max_len, model_dim):
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / model_dim))
        pe = torch.zeros(max_len, model_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, noisy_x: torch.Tensor, t: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                cond: Optional[torch.Tensor] = None,
                cond_attn_mask: Optional[torch.Tensor] = None,
                cond_key_padding_mask: Optional[torch.Tensor] = None,
                cond_vec: Optional[torch.Tensor] = None                 # NEW
                ) -> torch.Tensor:
        return self.predict(noisy_x, t,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            cond=cond,
                            cond_attn_mask=cond_attn_mask,
                            cond_key_padding_mask=cond_key_padding_mask,
                            cond_vec=cond_vec)                          # NEW


    def predict(self, noisy_x: torch.Tensor, t: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                cond: Optional[torch.Tensor] = None,
                cond_attn_mask: Optional[torch.Tensor] = None,
                cond_key_padding_mask: Optional[torch.Tensor] = None,
                cond_vec: Optional[torch.Tensor] = None) -> torch.Tensor:
        if t.dtype != torch.long:
            t = t.long()

        cond_tokens = None
        if cond is not None and self.use_cross_attn:
            cond_tokens = self.cond_proj(cond) if self.cond_proj is not None else cond

        h = self.backbone(
            noisy_x, t,
            cond_vec=cond_vec,                                      # CHANGED
            cond_tokens=cond_tokens
        )
        return self.out_proj(h)