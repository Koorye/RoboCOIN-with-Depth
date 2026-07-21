"""
da3.py — Depth Anything V3 深度修复（优化版）

- chunk_size=0: 所有帧一次推理，GPU 批量 resize
- chunk_size>0, overlap=0: 简单分批（无对齐）
- chunk_size>0, overlap>0: VDA 风格 depth scale+shift 帧间对齐
"""

from __future__ import annotations

import numpy as np

from src.depth_repair.base import BaseDepthRepair


class DA3DepthRepair(BaseDepthRepair):
    """DA3 深度修复。

    Parameters
    ----------
    model_id, process_res, chunk_size, overlap, temporal_alpha, device
    """

    def __init__(
        self,
        model_id: str = "depth-anything/DA3METRIC-LARGE",
        process_res: int = 504,
        chunk_size: int = 0,
        overlap: int = 0,
        temporal_alpha: float = 0.0,
        device: str = "auto",
    ) -> None:
        super().__init__(device=device)
        self._model_id = model_id
        self._process_res = process_res
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._temporal_alpha = temporal_alpha

    # ------------------------------------------------------------------
    # 模型
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        from depth_anything_3.api import DepthAnything3
        self._model = DepthAnything3.from_pretrained(self._model_id).to(self._device)

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------

    def _repair_impl(
        self, rgb: list[np.ndarray], depth: list[np.ndarray]
    ) -> list[np.ndarray]:
        n = len(rgb)
        H_orig, W_orig = rgb[0].shape[:2]
        cs = self._chunk_size if self._chunk_size > 0 else n
        ol = self._overlap if cs < n else 0

        if ol <= 0 or cs >= n:
            return self._infer_simple(rgb, cs, H_orig, W_orig)
        return self._infer_streaming(rgb, n, cs, ol, H_orig, W_orig)

    # ------------------------------------------------------------------
    # 简单推理 — GPU 批量 resize，零拷贝
    # ------------------------------------------------------------------

    def _infer_simple(
        self, rgb: list[np.ndarray],
        cs: int, H_orig: int, W_orig: int,
    ) -> list[np.ndarray]:
        import torch
        import torch.nn.functional as F

        n = len(rgb)
        results: list[np.ndarray | None] = [None] * n

        for start in range(0, n, cs):
            end = min(start + cs, n)
            batch_rgb = rgb[start:end]

            # 绕过 inference()，跳过 cuda.synchronize ×2 + add_processed_images
            imgs_cpu, _, _ = self._model._preprocess_inputs(
                batch_rgb, None, None,
                self._process_res, "upper_bound_resize")
            imgs, _, _ = self._model._prepare_model_inputs(
                imgs_cpu, None, None)
            raw = self._model.forward(imgs, None, None,
                                      export_feat_layers=[])
            pred = self._model._convert_to_prediction(raw)

            # GPU 上：float → mm → 批量 resize → 一次性搬回 CPU
            depth = torch.as_tensor(
                pred.depth, device=self._device).float() * 1000.0  # (B, H', W')
            depth = F.interpolate(
                depth.unsqueeze(1), size=(H_orig, W_orig),
                mode='bilinear', align_corners=False,
            ).squeeze(1)                                        # (B, H, W)
            batch = depth.cpu().numpy()

            for j in range(len(batch)):
                results[start + j] = batch[j]

        return results  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Streaming 推理 — CPU 对齐 + 批量 resize
    # ------------------------------------------------------------------

    def _infer_streaming(
        self, rgb: list[np.ndarray], n: int,
        cs: int, ol: int, H_orig: int, W_orig: int,
    ) -> list[np.ndarray]:
        import cv2

        chunks = _make_chunks_streaming(n, cs, ol)
        num_chunks = len(chunks)

        # 逐块推理（绕过 inference() 避免 cuda.synchronize）
        chunk_depths: list[np.ndarray] = []
        for start, end in chunks:
            batch_rgb = rgb[start:end]
            imgs_cpu, _, _ = self._model._preprocess_inputs(
                batch_rgb, None, None,
                self._process_res, "upper_bound_resize")
            imgs, _, _ = self._model._prepare_model_inputs(
                imgs_cpu, None, None)
            raw = self._model.forward(imgs, None, None,
                                      export_feat_layers=[])
            pred = self._model._convert_to_prediction(raw)
            depth = np.asarray(pred.depth, dtype=np.float32)
            chunk_depths.append(depth)

        # VDA 风格帧间对齐（CPU）
        for ci in range(1, num_chunks):
            d_prev = chunk_depths[ci - 1][-ol:]
            d_curr = chunk_depths[ci][:ol]
            mask = (d_prev > 0) & np.isfinite(d_prev) & (d_curr > 0) & np.isfinite(d_curr)
            if mask.sum() < 100:
                continue
            p, t = d_curr[mask].ravel(), d_prev[mask].ravel()
            A = np.stack([p, np.ones_like(p)], axis=1)
            scale, shift = np.linalg.lstsq(A, t, rcond=None)[0]
            scale = float(np.clip(scale, 0.5, 2.0))
            chunk_depths[ci] = np.maximum(chunk_depths[ci] * scale + shift, 0)
            w = np.linspace(0, 1, ol, dtype=np.float32).reshape(ol, 1, 1)
            chunk_depths[ci][:ol] = (1 - w) * d_prev + w * chunk_depths[ci][:ol]

        # 组装：cv2.resize 写入预分配数组，避免额外分配
        results = [np.zeros((H_orig, W_orig), dtype=np.float32) for _ in range(n)]
        for ci, (start, end) in enumerate(chunks):
            d_mm = chunk_depths[ci] * 1000.0
            C_local = len(d_mm)
            ks, ke = 0, C_local if ci == num_chunks - 1 else C_local - ol
            for j in range(ks, ke):
                cv2.resize(d_mm[j], (W_orig, H_orig),
                          dst=results[start + j],
                          interpolation=cv2.INTER_LINEAR)

        return results

    # ------------------------------------------------------------------
    # 时序平滑（CPU）
    # ------------------------------------------------------------------

    def smooth_depth(
        self, depth_frames: list[np.ndarray],
    ) -> list[np.ndarray]:
        """对深度帧应用时序均值对齐（CPU）。"""
        if not depth_frames or self._temporal_alpha <= 0:
            return depth_frames
        frames = np.stack(depth_frames, axis=0)
        return _temporal_mean_align(frames, self._temporal_alpha)


# ============================================================================
# helpers
# ============================================================================

def _make_chunks_streaming(n: int, cs: int, ol: int) -> list[tuple[int, int]]:
    if n <= cs:
        return [(0, n)]
    step = cs - ol
    num = (n - ol + step - 1) // step
    return [(i * step, min(i * step + cs, n)) for i in range(num)]


def _temporal_mean_align(
    frames: np.ndarray, alpha: float,
) -> list[np.ndarray]:
    """CPU 时序均值对齐（向量化优化）。

    Parameters
    ----------
    frames : (N, H, W) float32
    alpha : EMA 平滑系数

    Returns
    -------
    list of (H, W) float32
    """
    n = len(frames)

    # 向量化：一次性计算所有帧的有效像素均值
    valid = (frames > 0) & np.isfinite(frames)          # (N, H, W)
    valid_counts = valid.sum(axis=(1, 2))                # (N,)
    means = (frames * valid).sum(axis=(1, 2)) / np.maximum(valid_counts, 1)

    # EMA 平滑（标量序列）
    smooth = means.copy()
    for i in range(1, n):
        smooth[i] = alpha * means[i] + (1 - alpha) * smooth[i - 1]

    # 向量化：一次性缩放所有帧
    scale = smooth / np.maximum(means, 1e-6)             # (N,)
    return list((frames * scale.reshape(n, 1, 1)).astype(np.float32))
