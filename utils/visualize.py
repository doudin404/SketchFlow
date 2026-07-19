from dataclasses import dataclass
from typing import Tuple, Optional, Literal, Union
import io

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image

@dataclass
class RendererConfig:
    img_size: Tuple[int, int] = (224, 224)   # CLIP 输入尺寸
    padding: int = 8             # 轻微边距，避免触边
    line_width: float = 2        # 适中线宽，保证224像素下清晰
    marker_size: float = 2.5
    min_alpha: float = 0
    color: str = "black"
    marker: Optional[str] = None  # "o"
    stroke_alpha: float = 0.2    # 断笔状态（indices=0）时连线的不透明度
    draw_strokes: bool = False   # 是否绘制断笔连线
    use_gradient: bool = False   # 是否启用笔画顺序颜色渐变
    start_color: str = "#87CEEB" # 渐变起始颜色（浅蓝）
    end_color: str = "#1E3A8A"   # 渐变结束颜色（深蓝）
    transparent: bool = False    # 是否输出透明背景（SVG/PNG 均生效）


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """
    将 hex 颜色字符串转换为 RGB 元组
    
    Args:
        hex_color: hex 颜色字符串，如 "#87CEEB" 或 "87CEEB"
    
    Returns:
        (r, g, b) 元组，每个分量范围 0-255
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """
    将 RGB 元组转换为 hex 颜色字符串
    
    Args:
        r, g, b: RGB 分量，范围 0-255
    
    Returns:
        hex 颜色字符串，如 "#87CEEB"
    """
    return f"#{r:02x}{g:02x}{b:02x}"


def interpolate_color(start_color: str, end_color: str, ratio: float) -> str:
    """
    在两个颜色之间进行线性插值
    
    Args:
        start_color: 起始颜色的 hex 字符串
        end_color: 结束颜色的 hex 字符串
        ratio: 插值比例，范围 0.0-1.0
    
    Returns:
        插值后的 hex 颜色字符串
    """
    start_rgb = hex_to_rgb(start_color)
    end_rgb = hex_to_rgb(end_color)
    
    # 线性插值每个 RGB 分量
    r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * ratio)
    g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * ratio)
    b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * ratio)
    
    # 确保值在有效范围内
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    
    return rgb_to_hex(r, g, b)


class StrokeRenderer:
    """
    仅支持 'binary' 模式。
    indices 表示每个点的不透明度，范围 [0, 1]。
    """

    def __init__(self, cfg: RendererConfig = RendererConfig(), idx_mode=None):
        self.cfg = cfg

    @torch.no_grad()
    def normalize_points(self, pts: torch.Tensor) -> torch.Tensor:
        if pts.ndim != 2 or pts.size(1) != 2:
            raise ValueError("points 需为形状 [N, 2]")

        W, H = self.cfg.img_size
        pad = self.cfg.padding
        mins, _ = torch.min(pts, dim=0)
        maxs, _ = torch.max(pts, dim=0)
        span = torch.clamp(maxs - mins, min=1e-6)

        scale = min(W - 2 * pad, H - 2 * pad) / torch.max(span)

        center_pts = (maxs + mins) / 2
        center_canvas = torch.tensor([W / 2, H / 2], dtype=pts.dtype, device=pts.device)

        pts0 = (pts - center_pts) * scale + center_canvas
        return pts0
    
    @torch.no_grad()
    def normalize_unit_points(self, pts):
        W, H = self.cfg.img_size
        scale = min(W, H) / 2 - self.cfg.padding
        center = torch.tensor([W / 2, H / 2], device=pts.device)
        return pts * scale + center

    @torch.no_grad()
    def render(
        self,
        points,
        indices,
        invert_y: bool = True,
        output: Literal["tensor", "np", "svg","pil"] = "tensor",
        out_path: Optional[str] = None,
        svg_dpi: int = 96,
        return_tensor: Optional[bool] = None,
        extra_scale:float=1.0
    ) -> Union[torch.Tensor, np.ndarray, str, None]:
        """
        output:
          - "tensor": 返回 torch.FloatTensor [H,W,3] in [0,1]
          - "np":     返回 np.uint8 RGB 图 [H,W,3]
          - "svg":    返回 SVG 字符串（若提供 out_path 则写文件并返回字符串副本）
        """
        # 兼容旧签名
        if return_tensor is not None:
            output = "tensor" if return_tensor else "np"

        if isinstance(points, np.ndarray):
            pts = torch.from_numpy(points.copy()).to(dtype=torch.float32)
        else:  # Assumes torch.Tensor
            pts = points.clone().to(dtype=torch.float32)

        seq = torch.as_tensor(indices, dtype=torch.float32).squeeze(-1) / extra_scale
        if pts.ndim != 2 or pts.size(1) != 2:
            raise ValueError("points 需为形状 [N, 2]")
        if seq.ndim != 1 or seq.size(0) != pts.size(0):
            raise ValueError("indices 需为形状 [N] 或 [N,1]")

        pts = self.normalize_unit_points(pts)
        if invert_y:
            H = self.cfg.img_size[1]
            pts[:, 1] = H - pts[:, 1]
        pts = pts.cpu() # 将 Tensor 移动到 CPU，因为 matplotlib 无法直接处理 GPU Tensor

        N = pts.size(0)
        alpha = torch.clamp(seq[1:], self.cfg.min_alpha, 1.0) if N >= 2 else torch.tensor([])

        # 图像尺寸与 DPI
        Wpx, Hpx = self.cfg.img_size
        dpi = svg_dpi if output == "svg" else 100
        fig_w_in, fig_h_in = Wpx / dpi, Hpx / dpi

        fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)
        ax.set_xlim(0, Wpx)
        ax.set_ylim(0, Hpx)
        ax.axis("off")
        ax.set_aspect("equal")

        if self.cfg.transparent:
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)

        # 渐变颜色获取函数：按段序 i 映射到 [0,1]
        def seg_color(i: int) -> str:
            if not self.cfg.use_gradient:
                return self.cfg.color
            # N 段为 N-1，i ∈ [0, N-2]
            if N <= 2:
                ratio = 0.0
            else:
                ratio = i / float(N - 2)
            return interpolate_color(self.cfg.start_color, self.cfg.end_color, ratio)

        if N < 2:
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                marker=self.cfg.marker,
                linestyle="None",
                markersize=self.cfg.marker_size,
                color=self.cfg.start_color if self.cfg.use_gradient else self.cfg.color,
                solid_capstyle="round",
                solid_joinstyle="round",
                antialiased=True,
            )
        else:
            # 分段绘制：颜色随段序渐变，不透明度根据 indices 和配置决定
            for i in range(N - 1):
                alpha_val = float(alpha[i].item())
                
                # 根据配置决定线段的不透明度
                if self.cfg.draw_strokes:
                    # 绘制所有连线，断笔连线使用 stroke_alpha
                    if alpha_val > self.cfg.min_alpha:
                        # 落笔状态：完全不透明
                        line_alpha = self.cfg.min_alpha
                    else:
                        # 断笔状态：使用配置的 stroke_alpha
                        line_alpha = self.cfg.stroke_alpha
                else:
                    # 只绘制落笔状态的连线
                    # if alpha_val <= self.cfg.min_alpha:
                    #     continue  # 跳过断笔连线
                    line_alpha = max(alpha_val, self.cfg.min_alpha)
                
                ax.plot(
                    [pts[i, 0], pts[i + 1, 0]],
                    [pts[i, 1], pts[i + 1, 1]],
                    linewidth=self.cfg.line_width,
                    marker=self.cfg.marker,
                    markersize=self.cfg.marker_size,
                    color=seg_color(i),
                    alpha=line_alpha,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                    antialiased=True,
                )

        if output == "svg":
            buf = io.StringIO()
            fig.savefig(
                buf, format="svg", dpi=dpi, bbox_inches="tight", pad_inches=0,
                transparent=self.cfg.transparent,
            )
            svg_text = buf.getvalue()
            buf.close()
            plt.close(fig)
            if out_path is not None:
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(svg_text)
            return svg_text

        # 位图路径
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())  # [H, W, 4], uint8
        if self.cfg.transparent:
            img_rgba = buf.copy()
            img = img_rgba[..., :3].copy()
        else:
            img = buf[..., :3].copy()
            img_rgba = None
        plt.close(fig)

        if out_path and output in ("tensor", "np"):
            if self.cfg.transparent and img_rgba is not None:
                Image.fromarray(img_rgba, mode="RGBA").save(out_path)
            else:
                Image.fromarray(img).save(out_path)

        if output == "tensor":
            return torch.from_numpy(img.astype(np.float32) / 255.0)
        elif output == "np":
            return img
        elif output == "pil":
            if self.cfg.transparent and img_rgba is not None:
                return Image.fromarray(img_rgba, mode="RGBA")
            return Image.fromarray(img, mode="RGB")
        else:
            print(f"Unsupported output format: {output}")
            return None

    @torch.no_grad()
    def render_gif(
        self,
        points,
        indices,
        out_path: str,
        invert_y: bool = True,
        fps: int = 20,
        extra_scale: float = 1.0,
        step: int = 1,
        hold_factor: int = 15,
    ) -> None:
        """
        渲染并保存为展示绘制过程的 GIF 动图

        Args:
            points: 笔画坐标点集合
            indices: 落笔状态集合
            out_path: GIF 文件保存路径
            invert_y: 是否反转Y轴
            fps: GIF 帧率
            extra_scale: 额外缩放因子
            step: 每隔多少"可见"段捕获一帧（不可见的断笔段不计入）
            hold_factor: 末帧停留时长相对单帧 duration 的倍数（>=1）
        """
        if isinstance(points, np.ndarray):
            pts = torch.from_numpy(points.copy()).to(dtype=torch.float32)
        else:
            pts = points.clone().to(dtype=torch.float32)

        seq = torch.tensor(indices, dtype=torch.float32).squeeze(-1) / extra_scale
        if pts.ndim != 2 or pts.size(1) != 2:
            raise ValueError("points 需为形状 [N, 2]")
        if seq.ndim != 1 or seq.size(0) != pts.size(0):
            raise ValueError("indices 需为形状 [N] 或 [N,1]")

        pts = self.normalize_unit_points(pts)
        if invert_y:
            H = self.cfg.img_size[1]
            pts[:, 1] = H - pts[:, 1]
        pts = pts.cpu()

        N = pts.size(0)
        alpha = torch.clamp(seq[1:], self.cfg.min_alpha, 1.0) if N >= 2 else torch.tensor([])

        Wpx, Hpx = self.cfg.img_size
        dpi = 100
        fig_w_in, fig_h_in = Wpx / dpi, Hpx / dpi

        # 显式使用 Agg 后端（headless 环境安全；不影响全局 pyplot 后端）。
        # 不通过 pyplot 创建，因此不会被 pyplot 的图表注册表追踪，也无需 plt.close。
        fig = Figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ax.set_xlim(0, Wpx)
        ax.set_ylim(0, Hpx)
        ax.axis("off")
        ax.set_aspect("equal")
        # 让 axes 充满 figure，避免与 plt.subplots 默认 margin 不同导致的尺寸差异
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        if self.cfg.transparent:
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)

        def seg_color(i: int) -> str:
            if not self.cfg.use_gradient:
                return self.cfg.color
            if N <= 2:
                ratio = 0.0
            else:
                ratio = i / float(N - 2)
            return interpolate_color(self.cfg.start_color, self.cfg.end_color, ratio)

        def grab_rgba() -> np.ndarray:
            buf = np.asarray(canvas.buffer_rgba())
            if self.cfg.transparent:
                return buf.copy()
            return buf[..., :3].copy()

        frames: list = []
        step = max(1, int(step))

        # 初次完整绘制 → 缓存空白背景 → 抓首帧
        canvas.draw()
        bg = canvas.copy_from_bbox(fig.bbox)
        frames.append(grab_rgba())

        if N < 2:
            ax.plot(
                pts[:, 0], pts[:, 1],
                marker=self.cfg.marker, linestyle="None", markersize=self.cfg.marker_size,
                color=self.cfg.start_color if self.cfg.use_gradient else self.cfg.color,
                solid_capstyle="round", solid_joinstyle="round", antialiased=True,
            )
            canvas.draw()
            frames[-1] = grab_rgba()  # 用绘制完成后的状态替换首帧
        else:
            drawn_since_capture = 0
            for i in range(N - 1):
                alpha_val = float(alpha[i].item())
                if self.cfg.draw_strokes:
                    if alpha_val > self.cfg.min_alpha:
                        line_alpha = self.cfg.min_alpha
                    else:
                        line_alpha = self.cfg.stroke_alpha
                else:
                    if alpha_val <= self.cfg.min_alpha:
                        continue
                    line_alpha = max(alpha_val, self.cfg.min_alpha)

                (line,) = ax.plot(
                    [pts[i, 0], pts[i + 1, 0]],
                    [pts[i, 1], pts[i + 1, 1]],
                    linewidth=self.cfg.line_width,
                    marker=self.cfg.marker,
                    markersize=self.cfg.marker_size,
                    color=seg_color(i),
                    alpha=line_alpha,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                    antialiased=True,
                )

                # 每个"chunk"的第一个段：先把画布还原为上一次保存的背景
                if drawn_since_capture == 0:
                    canvas.restore_region(bg)
                # 增量绘制：只 rasterize 这条新加入的 Line2D
                ax.draw_artist(line)
                drawn_since_capture += 1

                if drawn_since_capture >= step:
                    # 抓帧并把当前状态作为新的背景缓存
                    frames.append(grab_rgba())
                    bg = canvas.copy_from_bbox(fig.bbox)
                    drawn_since_capture = 0

            # 兜底：循环结束时尚有未抓的 chunk
            if drawn_since_capture > 0:
                frames.append(grab_rgba())
                bg = canvas.copy_from_bbox(fig.bbox)  # 不严格需要，但保持状态一致

        # fig 不在 pyplot 管理下，无需 plt.close；本地引用结束后由 GC 回收

        if not frames:
            return

        duration = max(1, int(1000 / max(1, fps)))
        hold = max(1, int(hold_factor))
        durations = [duration] * (len(frames) - 1) + [duration * hold]

        self._save_frames_as_gif(frames, out_path, durations)

    def _save_frames_as_gif(self, frames, out_path: str, durations) -> None:
        """
        将帧序列保存为 GIF。

        - 非透明：把 RGB 帧直接交给 PIL 量化保存。
        - 透明：把 alpha<128 的像素替换为 sentinel 颜色，按共享调色板量化，
          通过 GIF 的 transparency + disposal=2 模拟透明背景。
        """
        if not self.cfg.transparent:
            pil_frames = [Image.fromarray(f, mode="RGB") for f in frames]
            pil_frames[0].save(
                out_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=durations,
                loop=0,
            )
            return

        sentinel = np.array([255, 0, 255], dtype=np.uint8)
        prepared_rgb = []
        for arr in frames:
            rgb = arr[..., :3].copy()
            mask = arr[..., 3] < 128
            rgb[mask] = sentinel
            prepared_rgb.append(rgb)

        # PIL 整数常量在新旧版本都有效：MEDIANCUT=0, Dither.NONE=0
        palette_src = Image.fromarray(prepared_rgb[-1], mode="RGB").quantize(
            colors=255, method=0
        )
        palette = palette_src.getpalette() or []

        transparent_idx = 0
        best_dist = float("inf")
        for i in range(0, min(len(palette), 256 * 3), 3):
            r, g, b = palette[i], palette[i + 1], palette[i + 2]
            dist = (r - int(sentinel[0])) ** 2 \
                + (g - int(sentinel[1])) ** 2 \
                + (b - int(sentinel[2])) ** 2
            if dist < best_dist:
                best_dist = dist
                transparent_idx = i // 3

        quantized = []
        for rgb in prepared_rgb:
            pil = Image.fromarray(rgb, mode="RGB")
            q = pil.quantize(palette=palette_src, dither=0)
            quantized.append(q)

        quantized[0].save(
            out_path,
            save_all=True,
            append_images=quantized[1:],
            duration=durations,
            loop=0,
            transparency=transparent_idx,
            disposal=2,
        )
