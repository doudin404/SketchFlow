import torch
import torch.nn as nn
import torch.nn.functional as F

def _as_bhqlk_mask(mask, B, H, Lq, Lk, device, dtype):
    if mask is None:
        return None
    if mask.dtype == torch.bool:
        mask = torch.where(mask, torch.finfo(dtype).min, torch.zeros((), device=mask.device, dtype=dtype))

    if mask.dim() == 2:          # (Lq,Lk)
        mask = mask.unsqueeze(0).unsqueeze(0).expand(B, H, Lq, Lk)
    elif mask.dim() == 3:        # (B,Lq,Lk)
        mask = mask.unsqueeze(1).expand(B, H, Lq, Lk)
    elif mask.dim() == 4:        # (B,H,Lq,Lk)
        pass
    else:
        raise ValueError("attn_mask 支持形状: (Lq,Lk),(B,Lq,Lk),(B,H,Lq,Lk)")
    return mask.to(device=device, dtype=dtype)


class CondTimeEmbedding(nn.Module):
    def __init__(self, num_heads, cond_dim, max_steps: int | None = None, hidden=256):
        super().__init__()
        self.embed = nn.Embedding(max_steps, cond_dim) if max_steps is not None else None
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.SiLU(), nn.Linear(hidden, num_heads)
        )

    def forward(self, t):  # t: (B,) int 或 (B,cond_dim) float
        if t.dtype in (torch.int32, torch.int64):
            if self.embed is None:
                raise ValueError("传入整型时间步需设置 max_steps 以启用 nn.Embedding")
            h = self.embed(t)
        else:
            h = t
        return self.mlp(h).unsqueeze(-1).unsqueeze(-1)  # (B,H,1,1)


# ========= 注意力实现 =========

class SDPAMultihead(nn.Module):
    """自注意力；支持按头掩码与按头时间步偏置。"""
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.h = num_heads
        self.d = embed_dim // num_heads
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = dropout

    def forward(self, x, attn_mask=None, cond_bias=None, key_padding_mask=None):
        B, L, D = x.shape
        device, dtype = x.device, x.dtype

        qkv = self.in_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(t):  # (B,L,D)->(B,H,L,d)
            return t.view(B, L, self.h, self.d).transpose(1, 2)
        q, k, v = map(split_heads, (q, k, v))

        m = _as_bhqlk_mask(attn_mask, B, self.h, L, L, device, dtype)
        if key_padding_mask is not None:
            # 仅屏蔽K轴
            kpm = key_padding_mask[:, None, None, :].expand(B, self.h, L, L)
            kpm = torch.where(kpm, torch.finfo(dtype).min, 0.).to(dtype)
            m = kpm if m is None else (m + kpm)

        if cond_bias is not None:
            assert cond_bias.shape == (B, self.h, 1, 1)
            cb = cond_bias.to(dtype).expand(B, self.h, L, L)
            m = cb if m is None else (m + cb)

        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=m,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False
            )
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(y)


class SDPACrossAttn(nn.Module):
    """交叉注意力；Q 来自 x，K/V 来自 cond。"""
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.h = num_heads
        self.d = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = dropout

    def forward(self, x, cond, attn_mask=None, key_padding_mask=None, cond_bias=None):
        B, Lq, D = x.shape
        Bc, Lk, Dc = cond.shape
        assert B == Bc and D == Dc
        device, dtype = x.device, x.dtype

        q = self.q_proj(x).view(B, Lq, self.h, self.d).transpose(1, 2)
        k = self.k_proj(cond).view(B, Lk, self.h, self.d).transpose(1, 2)
        v = self.v_proj(cond).view(B, Lk, self.h, self.d).transpose(1, 2)

        m = _as_bhqlk_mask(attn_mask, B, self.h, Lq, Lk, device, dtype)
        if key_padding_mask is not None:
            # 屏蔽 cond 的 padding（K轴）
            kpm = key_padding_mask[:, None, None, :].expand(B, self.h, Lq, Lk)
            kpm = torch.where(kpm, torch.finfo(dtype).min, 0.).to(dtype)
            m = kpm if m is None else (m + kpm)

        if cond_bias is not None:
            # 复用按头偏置；对所有 Q-K 位置加同一偏置
            assert cond_bias.shape == (B, self.h, 1, 1)
            cb = cond_bias.to(dtype).expand(B, self.h, Lq, Lk)
            m = cb if m is None else (m + cb)

        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=m,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False
            )
        y = y.transpose(1, 2).contiguous().view(B, Lq, D)
        return self.out_proj(y)


# ========= Transformer 堆叠 =========

class PreNormBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0,
                 use_cross_attn: bool = False):
        super().__init__()
        self.use_cross_attn = bool(use_cross_attn)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = SDPAMultihead(dim, num_heads, dropout=dropout)

        if self.use_cross_attn:
            self.norm_x = nn.LayerNorm(dim)
            self.norm_c = nn.LayerNorm(dim)
            self.cross_attn = SDPACrossAttn(dim, num_heads, dropout=dropout)

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x,
                attn_mask=None, cond_bias=None, key_padding_mask=None,
                cond=None, cond_attn_mask=None, cond_key_padding_mask=None,
                cross_cond_bias=None):
        # 自注意力
        x = x + self.attn(self.norm1(x),
                               attn_mask=attn_mask,
                               cond_bias=cond_bias,
                               key_padding_mask=key_padding_mask)
        # 交叉注意力（可选）
        if self.use_cross_attn and (cond is not None):
            x = x + self.cross_attn(self.norm_x(x),
                                    self.norm_c(cond),
                                    attn_mask=cond_attn_mask,
                                    key_padding_mask=cond_key_padding_mask,
                                    cond_bias=cross_cond_bias)
        # 前馈
        x = x + self.mlp(self.norm2(x))
        return x


class PreNormTransformer(nn.Module):
    """
    Pre-Norm Encoder，支持：
      1) 按头logits时间偏置（use_time_attn_bias）
      2) token加性偏置（use_token_shift）
      3) 条件序列与交叉注意力（use_cross_attn）
    是否开启由初始化参数决定；未提供条件时自动跳过交叉注意力。
    """
    def __init__(self, dim, depth, num_heads, mlp_ratio=4.0, dropout=0.0,
                 cond_dim: int | None = 128,
                 max_steps: int | None = None,
                 use_time_attn_bias: bool = True,
                 use_token_shift: bool = False,
                 use_cross_attn: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([
            PreNormBlock(dim, num_heads, mlp_ratio, dropout, use_cross_attn)
            for _ in range(depth)
        ])

        self.use_time_attn_bias = bool(use_time_attn_bias)
        self.use_token_shift = bool(use_token_shift)
        self.use_cross_attn = bool(use_cross_attn)

        if self.use_time_attn_bias:
            if cond_dim is None:
                raise ValueError("use_time_attn_bias=True 需要提供 cond_dim")
            # 为自注意力与交叉注意力分别建立各自的 MLP（如果启用了交叉注意力）
            self.time_to_attn = CondTimeEmbedding(num_heads, cond_dim, max_steps=max_steps)
            self.time_to_cross_attn = CondTimeEmbedding(num_heads, cond_dim, max_steps=max_steps) \
            if self.use_cross_attn else None
        else:
            self.time_to_attn = None
            self.time_to_cross_attn = None

        if self.use_token_shift:
            if cond_dim is None:
                raise ValueError("use_token_shift=True 需要提供 cond_dim")
            self.time_to_token = nn.Sequential(
            nn.Linear(cond_dim, dim), nn.SiLU(), nn.Linear(dim, dim),
            )
        else:
            self.time_to_token = None

    def forward(self, x, t=None,
        attn_mask=None, key_padding_mask=None,
        cond=None, cond_attn_mask=None, cond_key_padding_mask=None):
        """
        x: (B,L,D)
        t: (B,) int 或 (B,cond_dim) float；None 则不注入时间条件
        cond: 条件序列 (B,Lc,D)；None 则跳过交叉注意力
        *_mask: bool 或 float；形状分别为
            attn_mask: (L,L)|(B,L,L)|(B,H,L,L)
            cond_attn_mask: (Lq,Lk)|(B,Lq,Lk)|(B,H,Lq,Lk)
            key_padding_mask: (B,L)
            cond_key_padding_mask: (B,Lc)
        """
        cond_logits_bias = None
        cross_cond_bias = None
        if t is not None:
            if self.use_time_attn_bias and self.time_to_attn is not None:
            # 自注意力按头偏置
                cond_logits_bias = self.time_to_attn(t)
                # 交叉注意力使用单独的 MLP（若启用）
                if self.use_cross_attn and self.time_to_cross_attn is not None:
                    cross_cond_bias = self.time_to_cross_attn(t)
                else:
                    cross_cond_bias = None
            if self.use_token_shift and self.time_to_token is not None:
                x = x + self.time_to_token(t).unsqueeze(1)

        for layer in self.layers:
            x = layer(x,
                      attn_mask=attn_mask,
                      cond_bias=cond_logits_bias,
                      key_padding_mask=key_padding_mask,
                      cond=cond if self.use_cross_attn else None,
                      cond_attn_mask=cond_attn_mask,
                      cond_key_padding_mask=cond_key_padding_mask,
                      cross_cond_bias=cross_cond_bias)
        return x
 