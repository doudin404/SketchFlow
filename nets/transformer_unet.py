# pip install torch transformers diffusers accelerate
import math, torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler

# ---------------- utils ----------------
def sinusoidal_embedding(t: torch.Tensor, dim: int = 256):
    device = t.device
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / (half - 1))
    t = t.float().unsqueeze(-1) * freqs
    emb = torch.cat([torch.cos(t), torch.sin(t)], dim=-1)
    if dim % 2: emb = F.pad(emb, (0,1))
    return emb

class AdaLN(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(cond_dim, dim * 2)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)
    def forward(self, x, c):
        s, b = self.proj(c).chunk(2, -1)
        return (1 + s).unsqueeze(1) * self.norm(x) + b.unsqueeze(1)

class SelfAttn(nn.Module):
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
    def forward(self, x, attn_mask=None, key_padding_mask=None):
        y,_ = self.attn(x,x,x,attn_mask=attn_mask,key_padding_mask=key_padding_mask,need_weights=False)
        return y

class CrossAttn(nn.Module):
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
    def forward(self, x, kv, attn_mask=None, key_padding_mask=None):
        y,_ = self.attn(x,kv,kv,attn_mask=attn_mask,key_padding_mask=key_padding_mask,need_weights=False)
        return y

class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, cond_dim, dropout=0.0, use_cross=False):
        super().__init__()
        self.ada1 = AdaLN(dim, cond_dim)
        self.sa   = SelfAttn(dim, heads, dropout)
        self.ada2 = AdaLN(dim, cond_dim)
        self.mlp  = nn.Sequential(nn.Linear(dim, int(dim*mlp_ratio)), nn.GELU(), nn.Dropout(dropout), nn.Linear(int(dim*mlp_ratio), dim))
        self.use_cross = use_cross
        if use_cross:
            self.adax = AdaLN(dim, cond_dim); self.ca = CrossAttn(dim, heads, dropout)
    def forward(self, x, c, tokens=None):
        x = x + self.sa(self.ada1(x, c))
        if self.use_cross and tokens is not None:
            x = x + self.ca(self.adax(x, c), tokens)
        x = x + self.mlp(self.ada2(x, c))
        return x

class Down1D(nn.Module):
    def __init__(self, c_in, c_out): super().__init__(); self.conv = nn.Conv1d(c_in, c_out, 3, 2, 1)
    def forward(self, x): return self.conv(x.transpose(1,2)).transpose(1,2)

class Up1D(nn.Module):
    def __init__(self, c_in, c_out): super().__init__(); self.proj = nn.Conv1d(c_in, c_out, 1)
    def forward(self, x, Lout):
        x = F.interpolate(x.transpose(1,2), size=Lout, mode="linear", align_corners=False)
        return self.proj(x).transpose(1,2)

# ---------------- UNet1D with AdaLN & optional cross-attn ----------------
class TransformerUNet1D(nn.Module):
    def __init__(self,
        in_channels=6, out_channels=None,
        base_dim=128, dim_mults=(1, 1.5, 2, 4),
        depths=(2,2,2,2), heads=(4,4,8,8), mlp_ratio=4.0, dropout=0.0,
        cond_vec_dim=256, cond_tok_dim: Optional[int]=None, use_cross=False
    ):
        super().__init__()
        self.use_cross = use_cross
        self.in_proj  = nn.Linear(in_channels, base_dim)
        self.t_mlp    = nn.Sequential(nn.Linear(256, cond_vec_dim), nn.SiLU(), nn.Linear(cond_vec_dim, cond_vec_dim))
        self.fuse     = nn.Sequential(nn.Linear(cond_vec_dim*2, cond_vec_dim), nn.SiLU(), nn.Linear(cond_vec_dim, cond_vec_dim))
        self.cond_proj = nn.ModuleList()
        dims = [int(base_dim*m) for m in dim_mults]
        for d in dims:
            self.cond_proj.append(nn.Linear(cond_tok_dim, d) if (use_cross and cond_tok_dim) else nn.Identity())


        # Encoder per-scale blocks + downs
        self.encs, self.downs = nn.ModuleList(), nn.ModuleList()
        for i in range(len(dims) - 1):                      # 0 .. n-2
            d, dep, h = dims[i], depths[i], heads[i]
            self.encs.append(nn.ModuleList([
                Block(d, h, mlp_ratio, cond_vec_dim, dropout, use_cross) for _ in range(dep)
            ]))
            self.downs.append(Down1D(dims[i], dims[i+1]))   # 明确 d_i → d_{i+1}

        # ---------- Bottleneck：在最深尺度（仅这里使用 512） ----------
        d_bn, h_bn = dims[-1], heads[-1]
        self.bottleneck = nn.ModuleList([
            Block(d_bn, h_bn, mlp_ratio, cond_vec_dim, dropout, use_cross) for _ in range(depths[-1])
        ])

        # ---------- Decoder：从最深回到最浅 ----------
        self.ups  = nn.ModuleList()
        self.decs = nn.ModuleList()
        self.dec_scale_idx = []
        for i in reversed(range(len(dims) - 1)):            # 上采样目标尺度索引 i
            self.ups.append(Up1D(dims[i+1], dims[i]))       # d_{i+1} → d_i
            self.decs.append(nn.ModuleList([
                Block(dims[i], heads[i], mlp_ratio, cond_vec_dim, dropout, use_cross) for _ in range(depths[i])
            ]))
            self.dec_scale_idx.append(i)

        self.out = nn.Linear(dims[0], out_channels or in_channels)

    def build_cvec(self, t, cvec=None):
        te  = self.t_mlp(sinusoidal_embedding(t, 256))
        return te if cvec is None else self.fuse(torch.cat([te, cvec], -1))

    def forward(self, x, t, cond_vec=None, cond_tokens=None):
        B,L,_ = x.shape
        cvec = self.build_cvec(t, cond_vec)
        x = self.in_proj(x)
        skips: List[torch.Tensor] = []

        # ----- Encoder（对称：使用尺度 i）-----
        for i, blocks in enumerate(self.encs):
            tokens = self.cond_proj[i](cond_tokens) if (self.use_cross and cond_tokens is not None) else None
            for blk in blocks:
                x = blk(x, cvec, tokens)
            skips.append(x)
            if not isinstance(self.downs[i], nn.Identity):
                x = self.downs[i](x)

        # ----- Bottleneck -----
        tokens = self.cond_proj[-1](cond_tokens) if (self.use_cross and cond_tokens is not None) else None
        for blk in self.bottleneck:
            x = blk(x, cvec, tokens)

        # ----- Decoder（对称：使用 dec_scale_idx[k]）-----
        for up, blocks, si in zip(self.ups, self.decs, self.dec_scale_idx):
            x = up(x, skips[-1].shape[1])
            x = x + skips.pop()
            tokens = self.cond_proj[si](cond_tokens) if (self.use_cross and cond_tokens is not None) else None
            for blk in blocks:
                x = blk(x, cvec, tokens)

        return self.out(x)