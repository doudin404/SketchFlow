import numpy as np
from typing import List, Tuple

def _polyline_length(pl: np.ndarray) -> float:
    if pl is None or len(pl) < 2:
        return 0.0
    seg = np.diff(pl, axis=0)
    return float(np.linalg.norm(seg, axis=1).sum())

def _resample_polyline_equal_arclen(pl: np.ndarray, k: int) -> np.ndarray:
    """
    将一条折线按自身弧长等距重采样为 k 个点（包含两端）。
    若原始长度为 0，则返回重复点。
    """
    if k <= 0:
        return np.zeros((0, 2), dtype=float)
    if pl.shape[0] == 0:
        return np.zeros((k, 2), dtype=float)
    if pl.shape[0] == 1:
        return np.repeat(pl.astype(float), k, axis=0)

    seg = np.diff(pl, axis=0).astype(float)
    seg_len = np.linalg.norm(seg, axis=1)
    total = seg_len.sum()

    if total <= 1e-12:
        # 所有点重合
        return np.repeat(pl[:1].astype(float), k, axis=0)

    # 累积弧长，长度为 M（最后一个是总长）
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    # 目标采样位置（含起终点）
    target = np.linspace(0.0, total, k)

    # 找到每个 target 落在哪个线段 [cum[i], cum[i+1]] 上
    idx = np.searchsorted(cum, target, side="right") - 1
    idx = np.clip(idx, 0, len(seg_len) - 1)

    # 段内线性插值参数 t \in [0,1]
    local = target - cum[idx]
    denom = seg_len[idx]
    # 避免除零
    denom = np.where(denom < 1e-12, 1.0, denom)
    t = (local / denom)[:, None]

    p0 = pl[idx]
    p1 = pl[idx + 1]
    pts = p0 + t * (p1 - p0)
    # 确保首末点精确对齐
    pts[0] = pl[0]
    pts[-1] = pl[-1]
    return pts

def _largest_remainder_allocation(
    lengths: np.ndarray, total_points: int, min_per_stroke: int
) -> np.ndarray:
    """
    按长度比例分配整数点数，总和为 total_points。
    先给每条折线 min_per_stroke 基数（若长度>0），其余按最大余数法分配。
    对于长度为 0 的折线，若需保留也给 min_per_stroke=1，否则为 0。
    """
    n = len(lengths)
    alloc = np.zeros(n, dtype=int)

    # 哪些折线需要至少一个点：长度>0 的优先
    pos = lengths > 0
    m = pos.sum()

    # 情况A：总点数不够覆盖所有正长折线的最小需求
    if m * min_per_stroke > total_points and m > 0:
        # 降级为每条至少 1 点（退一步保证不超额）
        min_per_stroke = 1
        if m * min_per_stroke > total_points:
            # 仍不够，则给最长的前 total_points 条各 1 点，其余 0
            order = np.argsort(-lengths)  # 长到短
            alloc[order[:total_points]] = 1
            return alloc

    # 先发放基数
    alloc[pos] = min_per_stroke
    used = int(alloc.sum())
    remain = total_points - used
    if remain <= 0:
        return alloc

    total_len = float(lengths[pos].sum())
    if total_len <= 1e-12:
        # 全是零长，平均分配剩余
        # 从头开始轮流+1直到发完
        order = np.arange(n)
        i = 0
        while remain > 0:
            alloc[order[i % n]] += 1
            remain -= 1
            i += 1
        return alloc

    # 理想值（去掉已分配的基数后部分）
    ideal = np.zeros(n, dtype=float)
    ideal[pos] = (lengths[pos] / total_len) * remain
    base = np.floor(ideal).astype(int)
    alloc += base
    remain2 = remain - int(base.sum())
    if remain2 > 0:
        frac = ideal - base
        order = np.argsort(-frac)  # 余数大的优先
        for i in range(remain2):
            alloc[order[i]] += 1
    return alloc

def _allocate_with_min_segments(lengths: np.ndarray, total_points: int, min_per_stroke: int) -> np.ndarray:
    """
    先尝试“等距优先”：每条正长度轨迹先按最低 1 点分配；
    若产生了段数 < min_per_stroke 的轨迹，则按“每条至少 min_per_stroke+1 点”重新分配。

    注意：min_per_stroke 表示“最少段数”，而底层分配器的参数表示“最少点数”，
    因此需要做 +1 的映射（对正长度轨迹）。
    """
    lengths = np.asarray(lengths, dtype=float)
    n = len(lengths)
    pos = lengths > 0

    # 第一次尝试：每条至少 1 点（0 段）
    alloc_try = _largest_remainder_allocation(lengths, total_points, min_per_stroke=1)

    # 计算段数：正长度轨迹 seg_i = points_i - 1；零长轨迹按 0 段计
    seg = np.zeros(n, dtype=int)
    seg[pos] = np.maximum(alloc_try[pos] - 1, 0)

    # 若所有正长度轨迹都满足最少段数，直接返回第一次结果
    if np.all(seg[pos] >= min_per_stroke):
        return alloc_try

    # 否则回退到“每条至少 min_per_stroke+1 点”再分配
    min_points_needed = min_per_stroke + 1
    return _largest_remainder_allocation(lengths, total_points, min_per_stroke=min_points_needed)

def resample_polylines(
    polylines: List[np.ndarray], n_samples: int, keep_empty=False
) -> List[np.ndarray]:
    """
    将一组折线按总弧长比例分配采样点，使整体点间距尽量均匀。
    - polylines: List[Ndarray(N_i,2)]
    - n_samples: 目标总点数（包含所有折线采样点之和）
    - keep_empty: 若某些折线被分配 0 点，是否以空折线保留；否则丢弃。
    返回同结构的折线列表（数量可能变化，点数之和为 n_samples）。
    """
    if n_samples <= 0 or len(polylines) == 0:
        return []

    lengths = np.array([_polyline_length(pl) for pl in polylines], dtype=float)
    has_pos = (lengths > 1e-12).sum() > 0

    # 常见期望：正长度折线至少 2 点，若点数太少则退化为至少 1 点
    min_per_stroke = 2 if n_samples >= 2 * (lengths > 0).sum() else 1 if has_pos else 1

    alloc = _allocate_with_min_segments(lengths, n_samples, min_per_stroke)

    out: List[np.ndarray] = []
    for pl, k in zip(polylines, alloc):
        if k <= 0:
            if keep_empty:
                out.append(np.zeros((0, 2), dtype=float))
            continue
        out.append(_resample_polyline_equal_arclen(pl, int(k)))
    return out

# -------- 示例 --------
if __name__ == "__main__":
    # 构造两条折线
    pl1 = np.array([[0,0],[10,0],[20,0]], dtype=float)       # 长度 20
    pl2 = np.array([[0,0],[0,10],[0,30],[0,40]], dtype=float) # 长度 40
    polys = [pl1, pl2]

    res = resample_polylines(polys, n_samples=256)
    # 检查总点数
    total = sum(len(p) for p in res)
    print("total points:", total)
    for i, p in enumerate(res):
        print(f"stroke {i}: {len(p)} points, first={p[0]}, last={p[-1]}")

def resample_and_flatten(polylines: List[np.ndarray], K: int) -> Tuple[np.ndarray, np.ndarray]:
    # 用你已有的 resample_polylines 做分配，然后扁平化
    pls = resample_polylines(polylines, K, keep_empty=False)      # List[(m_i,2)], sum m_i == K
    if not pls:
        pts = np.zeros((K, 2), np.float32)
        biny = np.zeros((K,), np.uint8)
        return pts, biny
    pts = np.concatenate(pls, axis=0).astype(np.float32, copy=False)  # (K,2)
    biny = np.zeros((K,), np.uint8)
    off = 0
    for pl in pls:
        m = len(pl)
        if m > 1:
            biny[off+1:off+m] = 1
        off += m
    return pts, biny