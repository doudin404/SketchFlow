import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

def pairwise_cosine_similarity(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """z: (B,L,C) -> sim: (B,L,L) in [-1,1]"""
    z_norm = F.normalize(z, p=2, dim=-1, eps=eps)
    return torch.einsum('bid,bjd->bij', z_norm, z_norm)

def pairwise_l2_distance(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    diff = z.unsqueeze(2) - z.unsqueeze(1)  # (B,L,L,C)
    dist = diff.pow(2).sum(-1).clamp_min(eps).sqrt()  # (B,L,L)
    return dist


def pairwise_rbf_similarity(z: torch.Tensor, gamma: float = None) -> torch.Tensor:
    diff = z.unsqueeze(2) - z.unsqueeze(1)
    dist2 = diff.pow(2).sum(-1)
    if gamma is None:
        gamma = 1.0 / (z.size(-1))
    return torch.exp(-gamma * dist2)




class SequentialIndexer:
    """
    三种“序号”方案，输入:
      points: (N, D) 张量，仅用于取设备和 dtype
      conn_matrix: (N, N) 邻接或连通矩阵，若点 i-1 与 i 连则 conn_matrix[i-1, i] 为真
    方案:
      - 'weighted': 原函数等价方案，连/断增量不同，然后前缀和并均值中心化
      - 'binary':   直接赋值，i>0 时 indices[i] = 1(连) 或 0(断)，indices[0]=0
      - 'stroke':   以断笔为分隔标注笔画序号，同一笔画内值相等，序号步长可配(默认 0.1)
    """
    def __init__(self,
                 fect_connected: float = 1.0,
                 fect_disconnected: float = 2.0,
                 stroke_step: float = 0.1,
                 mean_center_weighted: bool = True,
                 mode: str = 'weighted'):
        self.fect_connected = float(fect_connected)
        self.fect_disconnected = float(fect_disconnected)
        self.stroke_step = float(stroke_step)
        self.mean_center_weighted = bool(mean_center_weighted)
        self.mode = mode

    @staticmethod
    def _diag_conn_bool(conn_matrix: torch.Tensor) -> torch.Tensor:
        # 取超对角线 conn_matrix[i-1, i]，形状 (N-1,) 的 bool
        return torch.diagonal(conn_matrix, offset=1).to(dtype=torch.bool)

    def compute(self,
                points: torch.Tensor,
                conn_matrix: torch.Tensor,
                ) -> torch.Tensor:
        N = points.size(0)
        device = points.device
        dtype = points.dtype
        if N <= 1:
            return torch.zeros((N, 1), dtype=dtype, device=device)

        conn = self._diag_conn_bool(conn_matrix)           # (N-1,) bool

        if self.mode == 'weighted':
            inc_c = 2.0 / N * self.fect_connected
            inc_d = 2.0 / N * self.fect_disconnected
            incs = torch.where(
                conn,
                torch.as_tensor(inc_c, dtype=dtype, device=device),
                torch.as_tensor(inc_d, dtype=dtype, device=device),
            )                                             # (N-1,)
            indices = torch.cat([
                torch.zeros(1, dtype=dtype, device=device),
                torch.cumsum(incs, dim=0)
            ], dim=0)                                     # (N,)
            if self.mean_center_weighted:
                indices = indices - indices.mean()
            return indices.unsqueeze(1)

        elif self.mode == 'binary':
            # indices[0]=0，之后直接由是否连线决定 1/0
            vals = conn.to(dtype).to(device)              # (N-1,)
            indices = torch.cat([
                torch.zeros(1, dtype=dtype, device=device),
                vals
            ], dim=0)                                     # (N,)
            return indices.unsqueeze(1)

        elif self.mode == 'stroke':
            # 遇到断笔(~conn)则笔画序号+1，同一笔画内相同；起始为 0
            breaks = (~conn).to(dtype).to(device)         # (N-1,)
            stroke_ids = torch.cat([
                torch.zeros(1, dtype=dtype, device=device),
                torch.cumsum(breaks, dim=0)
            ], dim=0)                                     # (N,)
            indices = stroke_ids * self.stroke_step
            return indices.unsqueeze(1)

        else:
            raise ValueError(f"Unsupported mode: {mode}")

# 使用示例
# indexer = SequentialIndexer(fect_connected=1.0, fect_disconnected=2.0, stroke_step=0.1)
# idx_weighted = indexer.compute(points, conn_matrix, mode='weighted')
# idx_binary   = indexer.compute(points, conn_matrix, mode='binary')
# idx_stroke   = indexer.compute(points, conn_matrix, mode='stroke')
