import torch



# --------- 公共：Chamfer 距离(对集合/点云，L 维无序) ----------
def chamfer_l2(a: torch.Tensor,
               b: torch.Tensor,
               normalize: bool = False,
               reduce_batch: bool = True) -> torch.Tensor:
    """
    a,b: (B, L, C)
    返回 Chamfer L2 距离
      - reduce_batch=True  -> 标量，整个 batch 的平均值
      - reduce_batch=False -> (B,)，每个样本一个值
    如果 normalize=True，则在一一对应时数值与 F.mse_loss(a,b) 等价
    """
    # pairwise 距离: (B, L, L)
    d = torch.cdist(a, b, p=2) ** 2
    a2b = d.min(dim=2).values   # (B, L)
    b2a = d.min(dim=1).values   # (B, L)
    cd_per = a2b.mean(dim=1) + b2a.mean(dim=1)   # (B,)

    if normalize:
        C = a.size(-1)
        cd_per = cd_per / (2 * C)

    if reduce_batch:
        return cd_per.mean()
    else:
        return cd_per