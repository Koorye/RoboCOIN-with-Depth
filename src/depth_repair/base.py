"""
base.py — 深度修复抽象基类
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ============================================================================
# shared helpers
# ============================================================================

def decode_rgb_frames(h5_path: str | Path, camera: str) -> list[np.ndarray]:
    """从 H5 解码 RGB 帧 → ``(H, W, 3)`` uint8 列表。"""
    import h5py

    frames: list[np.ndarray] = []
    with h5py.File(h5_path, "r") as f:
        ds = f[f"observations/camera/rgb/{camera}/images"]
        for i in range(len(ds)):
            buf = np.frombuffer(ds[i], dtype=np.uint8)
            frames.append(cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
    return frames


def decode_depth_frames(h5_path: str | Path, camera: str) -> list[np.ndarray]:
    """从 H5 解码深度帧 → ``(H, W)`` float32 (mm) 列表。"""
    import h5py

    frames: list[np.ndarray] = []
    with h5py.File(h5_path, "r") as f:
        ds = f[f"observations/camera/depth/{camera}/images"]
        for i in range(len(ds)):
            buf = np.frombuffer(ds[i], dtype=np.uint8)
            frames.append(cv2.imdecode(buf, cv2.IMREAD_UNCHANGED).astype(np.float32))
    return frames


def encode_depth_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    """将深度帧编码为 PNG bytes (uint16 mm)。"""
    encoded: list[np.ndarray] = []
    for d in frames:
        d16 = np.clip(np.round(d), 0, 65535).astype(np.uint16)
        ok, buf = cv2.imencode(".png", d16)
        encoded.append(np.frombuffer(buf, dtype=np.uint8) if ok else d16)
    return encoded


# ============================================================================
# BaseDepthRepair
# ============================================================================


class BaseDepthRepair(ABC):
    """深度修复抽象基类。

    子类需实现:
    - ``_load_model()``: 加载模型到 ``self._device``
    - ``_repair_impl(rgb, depth)``: 核心推理逻辑
    """

    def __init__(self, device: str = "auto") -> None:
        import torch

        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)
        self._model: Any = None

    # -- 子类必须实现 ---------------------------------------------------------

    @abstractmethod
    def _load_model(self) -> None:
        """加载模型，赋值 ``self._model``。"""
        ...

    @abstractmethod
    def _repair_impl(
        self, rgb: list[np.ndarray], depth: list[np.ndarray]
    ) -> list[np.ndarray]:
        """核心推理。

        Parameters
        ----------
        rgb : list[np.ndarray]
            RGB 帧 (H, W, 3) uint8。
        depth : list[np.ndarray]
            原始深度帧 (H, W) float32 (mm)；可能为空。

        Returns
        -------
        list[np.ndarray]
            修复后深度帧 (H, W) float32 (mm)。
        """
        ...

    # -- 统计对齐（子类共用） -------------------------------------------------

    @staticmethod
    def _align_with_stats(
        pred: np.ndarray, target: list[np.ndarray],
    ) -> np.ndarray:
        """用 95% 分位数对齐，排除零值离群点。"""
        t = np.stack(target, axis=0)
        valid = (t > 0) & np.isfinite(t)
        if valid.sum() < 100:
            return pred

        t_p95 = np.percentile(t[valid], 95)
        p_p95 = np.percentile(pred, 95)

        if p_p95 < 1e-6:
            return pred

        scale = t_p95 / p_p95
        shift = np.median(t[valid]) - np.median(pred) * scale
        return np.maximum(pred * scale + shift, 0)

    # -- 时序平滑（子类可覆盖）----------------------------------------------

    def smooth_depth(
        self, depth_frames: list[np.ndarray],
    ) -> list[np.ndarray]:
        """对深度帧应用时序平滑。默认无操作。"""
        return depth_frames

    # -- 公开 API -------------------------------------------------------------

    def repair_h5(
        self, h5_path: str | Path, camera: str = "head",
    ) -> list[np.ndarray]:
        """修复单个 H5 文件中指定相机的深度序列。"""
        if self._model is None:
            self._load_model()
        rgb = decode_rgb_frames(h5_path, camera)
        depth = decode_depth_frames(h5_path, camera)
        return self._repair_impl(rgb, depth)

    def repair_frames(
        self,
        rgb: list[np.ndarray],
        depth: list[np.ndarray] | None = None,
    ) -> list[np.ndarray]:
        """修复深度序列（直接传入帧数据）。"""
        if self._model is None:
            self._load_model()
        if depth is None:
            depth = []
        return self._repair_impl(rgb, depth)
