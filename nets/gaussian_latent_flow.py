import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""Gaussian latent-flow conditioner used as a SketchFlow ablation."""

# ---------- utils ----------
def l2_normalize(x, eps=1e-8):
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def sinusoidal_time_embedding(t, dim=128):
    """
    t: [B,1] in [0,1]
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

# ---------- FiLM residual block ----------
class FiLMBlock(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.to_gamma = nn.Linear(cond_dim, dim)
        self.to_beta  = nn.Linear(cond_dim, dim)

    def forward(self, x, cond_vec):
        h = self.norm(x)
        gamma = self.to_gamma(cond_vec)
        beta  = self.to_beta(cond_vec)
        h = h * (1 + gamma) + beta
        h = F.silu(self.fc1(h))
        h = self.fc2(h)
        return x + h

# ---------- Velocity field network ----------
class GaussianLatentFlowNet(nn.Module):
    """
    v_theta(x_t, t, c) -> 速度向量 (dim)
    条件 c 来自文本嵌入 e_txt，经线性映射；时间 t 用正余弦嵌入。
    """
    def __init__(self, dim=512, width=1024, depth=10, t_dim=128, c_dim=512):
        super().__init__()
        self.inp = nn.Linear(dim, width)
        self.time_proj = nn.Sequential(
            nn.Linear(t_dim, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(c_dim, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.blocks = nn.ModuleList([FiLMBlock(width, cond_dim=width) for _ in range(depth)])
        self.outp = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, dim))

    def forward(self, x, t, c):
        # x: [B, D]  t: [B,1]  c: [B, C]
        te = sinusoidal_time_embedding(t, dim=128)
        te = self.time_proj(te)
        ce = self.cond_proj(c)
        h = self.inp(x) + te
        cond_vec = te + ce
        for blk in self.blocks:
            h = blk(h, cond_vec)
        v = self.outp(h)
        return v

# ---------- Gaussian source conditioner ----------
class GaussianLatentFlowConditioner(nn.Module):
    """
    New conditioner for ablation study.
    - Training: Learns a conditional flow from N(0,I) to image embeddings, conditioned on text embeddings.
    - Inference: Generates image embeddings from N(0,I) conditioned on text embeddings.
    """
    def __init__(
        self,
        dim=512,
        width=1024,
        depth=10,
        t_eps=0.02,
        sigma_txt=0.025, # For optional noising of condition
        flow_l2=1e-6,
        **kwargs, # Ignore unused arguments like sigma_img from the old config
    ):
        super().__init__()
        self.dim = dim
        self.t_eps = t_eps
        self.sigma_txt = sigma_txt
        self.flow_l2 = flow_l2
        self.flow = GaussianLatentFlowNet(dim=dim, width=width, depth=depth, t_dim=128, c_dim=dim)

    def forward(
        self,
        e_img,                 # [B, D] Target image embeddings
        e_txt,                 # [B, D] Conditional text embeddings
        cond_vec_std=None,     # [B, D] Optional dynamic standard deviation
    ):
        """
        Returns:
          cond: The target image embedding to be used as condition for the downstream model.
          loss: The flow-matching MSE loss.
        """
        e_img = l2_normalize(e_img)
        e_txt = l2_normalize(e_txt)
        B = e_img.size(0)

        # 在训练阶段对 e_txt 加入一定规模的噪声干扰，增强条件鲁棒性
        if self.training:
            if cond_vec_std is not None:
                noise = torch.randn_like(e_txt) * cond_vec_std
                e_txt = l2_normalize(e_txt + noise)
            elif self.sigma_txt > 0:
                noise = torch.randn_like(e_txt) * self.sigma_txt
                e_txt = l2_normalize(e_txt + noise)


        # Source distribution is standard normal
        x0 = torch.randn_like(e_img)
        # Target distribution is the image embedding
        x1 = e_img

        # Sample time and avoid endpoints
        t = torch.rand(B, 1, device=e_img.device).clamp(self.t_eps, 1 - self.t_eps)

        # Linear path and constant velocity target
        x_t = (1 - t) * x0 + t * x1
        v_star = x1 - x0

        # Predict velocity, conditioned on the text embedding
        v_pred = self.flow(x_t, t, c=e_txt)

        # Loss calculation
        loss_mse = F.mse_loss(v_pred, v_star)
        loss_reg = self.flow_l2 * (v_pred.pow(2).mean())
        loss = loss_mse + loss_reg

        # The condition for the main U-Net is always the target image embedding
        cond = x1
        return cond, loss

    @torch.no_grad()
    def shift(self, e_txt, steps=30, variance_scale=0.0, cond_vec_std=None, **kwargs):
        """
        Generate image embedding from noise, conditioned on text embedding.
        The `variance_scale` parameter controls the noise level on the text condition.
        `flow_alpha` is ignored for API compatibility with SketchFlowModel.sample.
        """
        e_txt_cond = l2_normalize(e_txt)
        if variance_scale > 0:
            if cond_vec_std is not None:
                noise = torch.randn_like(e_txt_cond) * cond_vec_std * variance_scale
            else:
                noise = torch.randn_like(e_txt_cond) * self.sigma_txt * variance_scale
            e_txt_cond = e_txt_cond + noise
            e_txt_cond = l2_normalize(e_txt_cond)

        # ODE solve from N(0, I) to the target, conditioned on e_txt_cond
        x0 = torch.randn_like(e_txt)
        x1 = self._ode_solve_heun(x0, steps=steps, cond_c=e_txt_cond)
        
        return l2_normalize(x1)

    @torch.no_grad()
    def _ode_solve_heun(self, x0, steps, cond_c):
        B, D = x0.shape
        device = x0.device
        
        x = x0
        t = torch.zeros(B, 1, device=device)
        dt = 1.0 / steps

        for _ in range(steps):
            t_mid = (t + dt).clamp(0, 1)
            # Predict velocity conditioned on `cond_c`
            v1 = self.flow(x, t.clamp(self.t_eps, 1 - self.t_eps), c=cond_c)
            x_euler = x + dt * v1
            v2 = self.flow(x_euler, t_mid.clamp(self.t_eps, 1 - self.t_eps), c=cond_c)
            x = x + 0.5 * dt * (v1 + v2)
            t = t_mid

        return x
