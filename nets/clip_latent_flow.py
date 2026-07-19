import math
import torch
import torch.nn as nn
import torch.nn.functional as F


"""CLIP latent flow conditioner for SketchFlow.

This module learns a flow-matching velocity field that transports noisy text
CLIP embeddings toward the corresponding sketch/image CLIP embedding manifold.
"""

# ---------- utils ----------
def l2_normalize(x, eps=1e-8):
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def sinusoidal_time_embedding(t, dim=128):
    """
    t: [B,1] in [0,1] or any continuous values (like log_sigma)
    return: [B, dim]
    """
    device = t.device
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(math.log(1.0), math.log(1000.0), half, device=device)
    )  # geometric
    ang = 2 * math.pi * t * freqs  # [B,1]*[half] -> broadcast
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb

# ---------- FiLM 残差块 ----------
class FiLMBlock(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        # 由条件生成 gamma/beta
        self.to_gamma = nn.Linear(cond_dim, dim)
        self.to_beta  = nn.Linear(cond_dim, dim)

    def forward(self, x, cond_vec):
        # cond_vec: [B, cond_dim]
        h = self.norm(x)
        gamma = self.to_gamma(cond_vec)
        beta  = self.to_beta(cond_vec)
        h = h * (1 + gamma) + beta
        h = F.silu(self.fc1(h))
        h = self.fc2(h)
        return x + h

# ---------- 速度场网络 ----------
class ClipLatentFlowNet(nn.Module):
    """
    v_theta(x_t, t, c, sigma_txt) -> 速度向量 (dim)
    """
    def __init__(self, dim=512, width=1024, depth=10, t_dim=128, c_dim=512, sigma_dim=128):
        super().__init__()
        self.inp = nn.Linear(dim, width)
        self.time_proj = nn.Sequential(
            nn.Linear(t_dim, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(c_dim, width), nn.SiLU(), nn.Linear(width, width)
        )
        # [新增]: sigma 的位置编码投影
        self.sigma_proj = nn.Sequential(
            nn.Linear(sigma_dim, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.blocks = nn.ModuleList([FiLMBlock(width, cond_dim=width) for _ in range(depth)])
        self.outp = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, dim))

    def forward(self, x, t, c, sigma_txt):
        # x: [B, D]  t: [B,1]  c: [B, C], sigma_txt: [B, 1]
        te = sinusoidal_time_embedding(t, dim=128)
        te = self.time_proj(te)         # [B, W]
        ce = self.cond_proj(c)          # [B, W]
        
        # [新增]: 对数位置编码逻辑
        # 1. 计算对数 (加上 clamp 防止取对数出现 -inf)
        log_sigma = torch.log(sigma_txt.clamp(min=1e-8))
        # 2. 复用正余弦编码 (充当 log 位置编码)
        se = sinusoidal_time_embedding(log_sigma, dim=128)
        se = self.sigma_proj(se)        # [B, W]

        h = self.inp(x) + te
        # [修改]: 融合 t, c, 和 sigma 的编码作为最终的 FiLM 条件
        cond_vec = te + ce + se         
        
        for blk in self.blocks:
            h = blk(h, cond_vec)
        v = self.outp(h)                # [B, D]
        return v

# ---------- 顶层模块：训练 forward 与推理 shift ----------
class ClipLatentFlowConditioner(nn.Module):
    def __init__(
        self,
        dim=512,
        width=1024,
        depth=10,
        t_eps=0.02,
        sigma_txt=0.01,
        sigma_img=1e-3,
        flow_l2=1e-6,
        sigma_perturb_std=0.0, # [新增]: 决定扰动强度的参数，0为关闭扰动
    ):
        super().__init__()
        self.dim = dim
        self.t_eps = t_eps
        self.sigma_txt = sigma_txt
        self.sigma_img = sigma_img
        self.flow_l2 = flow_l2
        self.sigma_perturb_std = sigma_perturb_std

        # [修改]: 传入 sigma_dim
        self.flow = ClipLatentFlowNet(dim=dim, width=width, depth=depth, t_dim=128, c_dim=dim, sigma_dim=128)

    @torch.no_grad()
    def _noisify(self, e, sigma):
        if sigma <= 0:
            return e
        noise = torch.randn_like(e) * sigma
        return e + noise

    def forward(
        self,
        e_img,                 # [B, D] 
        e_txt,                 # [B, D] 
        teacher_forcing_prob=1,
        cond_vec_std=None,     # [B, D] or float    
    ):
        e_img = l2_normalize(e_img)
        e_txt = l2_normalize(e_txt)
        B, D = e_img.shape
        device = e_img.device

        # [新增]: 计算基础的 sigma_txt 尺度
        if cond_vec_std is not None:
            base_sigma = cond_vec_std
            if not isinstance(base_sigma, torch.Tensor):
                base_sigma = torch.full((B, 1), float(base_sigma), device=device)
        else:
            base_sigma = torch.full((B, 1), self.sigma_txt, device=device)

        # 确保 base_sigma 维度对齐
        if base_sigma.dim() == 1 and base_sigma.numel() == B:
            base_sigma = base_sigma.unsqueeze(1) # [B, 1]

        # [新增]: 对数正态扰动逻辑
        if self.sigma_perturb_std > 0.0:
            # x ~ N(0, std^2) -> multiplier = e^x
            x_log = torch.randn(B, 1, device=device) * self.sigma_perturb_std
            multiplier = torch.exp(x_log)
            sigma_txt_eff = base_sigma * multiplier
        else:
            # 行为与原始代码一致
            sigma_txt_eff = base_sigma

        # 实际注入的噪声
        txt_noise = sigma_txt_eff * torch.randn_like(e_img)

        x0 = e_txt + txt_noise
        x1 = e_img + self.sigma_img * torch.randn_like(e_img)

        # 采样时间并避免端点
        t = torch.rand(B, 1, device=device)
        t = t.clamp(self.t_eps, 1 - self.t_eps)

        # 线性路径 + 常速度目标
        x_t = (1 - t) * x0 + t * x1
        v_star = x1 - x0

        # [新增]: 提取 [B, 1] 形状的 sigma 特征送入网络进行位置编码
        sigma_cond = sigma_txt_eff.mean(dim=-1, keepdim=True) if sigma_txt_eff.dim() > 1 and sigma_txt_eff.shape[-1] > 1 else sigma_txt_eff.view(B, 1)

        # [修改]: 预测速度，传入带扰动的 sigma 
        v_pred = self.flow(x_t, t, c=x_t, sigma_txt=sigma_cond)

        # 损失
        loss_mse = F.mse_loss(v_pred, v_star)
        loss_reg = self.flow_l2 * (v_pred.pow(2).mean())
        loss = loss_mse + loss_reg

        # 条件输出
        cond = x1
        if teacher_forcing_prob > 0.0 and self.training and False:
            with torch.no_grad():
                use_tf = t > 0.5
                if use_tf.any():
                    x_hat = x_t + (1.0 - t) * v_pred.detach()
                    cond = torch.where(use_tf, x_hat, cond)
        cond = l2_normalize(cond)
        return cond, loss

    @torch.no_grad()
    def shift(self, e_txt, steps=30, variance_scale=0.1, flow_alpha=1.0, cond_vec_std=None, sigma_txt_override=None):
        """
        [修改]: 增加了 sigma_txt_override 参数，保证推理时的灵活性。
        """
        e_txt = l2_normalize(e_txt)
        B, D = e_txt.shape
        device = e_txt.device

        # 计算基础 sigma
        if sigma_txt_override is not None:
            base_sigma = torch.full((B, 1), float(sigma_txt_override), device=device)
        elif cond_vec_std is not None:
            base_sigma = cond_vec_std
            if not isinstance(base_sigma, torch.Tensor):
                base_sigma = torch.full((B, 1), float(base_sigma), device=device)
        else:
            base_sigma = torch.full((B, 1), self.sigma_txt, device=device)
            
        if base_sigma.dim() == 1 and base_sigma.numel() == B:
            base_sigma = base_sigma.unsqueeze(1)

        sigma_txt_eff = base_sigma

        eps = torch.randn_like(e_txt)
        
        # [修改]: 考虑到 variance_scale，结合实际的 sigma 生成初始噪声
        noise = sigma_txt_eff * eps * variance_scale
        x0 = e_txt + noise
        
        if flow_alpha <= 0:
            return l2_normalize(x0)
        elif flow_alpha >= 1:
            # [修改]: 将最终生效的 sigma 传入 ODE Solver
            x1 = self._ode_solve_heun(x0, sigma_txt_eff, steps=steps)
            return l2_normalize(x1)
        else:
            x1_flow = self._ode_solve_heun(x0, sigma_txt_eff, steps=steps)
            x1 = (1 - flow_alpha) * x0 + flow_alpha * x1_flow
            return l2_normalize(x1)

    # -------- ODE: Heun (改进欧拉) --------
    @torch.no_grad()
    def _ode_solve_heun(self, x0, sigma_txt_eff, steps=60):
        # [修改]: 接收 sigma_txt_eff 参数并送入网络
        B, D = x0.shape
        device = x0.device
        
        x = x0
        t = torch.zeros(B, 1, device=device)
        dt = 1.0 / steps

        # 将 sigma 展平为网络所需的 [B, 1]
        sigma_cond = sigma_txt_eff.mean(dim=-1, keepdim=True) if sigma_txt_eff.dim() > 1 and sigma_txt_eff.shape[-1] > 1 else sigma_txt_eff.view(B, 1)

        for _ in range(steps):
            t_mid = (t + dt).clamp(0, 1)
            
            # [修改]: 传入 sigma_cond
            v1 = self.flow(x, t.clamp(self.t_eps, 1 - self.t_eps), c=x, sigma_txt=sigma_cond)
            x_euler = x + dt * v1
            
            v2 = self.flow(x_euler, t_mid.clamp(self.t_eps, 1 - self.t_eps), c=x_euler, sigma_txt=sigma_cond)
            x = x + 0.5 * dt * (v1 + v2)
            
            t = t_mid

        return x
