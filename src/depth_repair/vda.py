"""
vda.py — Video Depth Anything 深度预测
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_VDA_DIR = str(Path(__file__).resolve().parents[2] / "third_party" / "video_depth_anything")
sys.path.insert(0, _VDA_DIR)

from src.depth_repair.base import BaseDepthRepair


class VDADepthRepair(BaseDepthRepair):
    """Video Depth Anything — 从 RGB 序列预测 metric 深度，再统计对齐。

    Parameters
    ----------
    checkpoint : str
        VDA checkpoint 路径。
    encoder : ``"vits"`` | ``"vitb"`` | ``"vitl"``
        编码器大小。
    max_frames : int
        单次推理最大帧数（超出则等距采样）。
    align_stats : bool
        是否用原始深度统计 (median/MAD) 对齐预测。
    device : str
        计算设备。
    """

    _CONFIGS = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }

    def __init__(
        self,
        checkpoint: str = "checkpoints/video_depth_anything_vits.pth",
        encoder: str = "vits",
        input_size: int = 518,
        metric: bool = True,
        invert: bool = False,
        fp32: bool = False,
        device: str = "auto",
    ) -> None:
        super().__init__(device=device)
        self._ckpt = checkpoint
        self._encoder = encoder
        self._input_size = input_size
        self._metric = metric
        self._invert = invert
        self._fp32 = fp32

        if self._metric:
            self._ckpt = self._ckpt.replace("video_depth_anything", "metric_video_depth_anything")

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        import torch
        from video_depth_anything.video_depth import VideoDepthAnything

        cfg = self._CONFIGS[self._encoder]
        model = VideoDepthAnything(**cfg, metric=self._metric)
        state = torch.load(self._ckpt, map_location="cpu")
        model.load_state_dict(state, strict=True)
        self._model = model.to(self._device).eval()

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------

    def _repair_impl(
        self, rgb: list[np.ndarray], depth: list[np.ndarray]
    ) -> list[np.ndarray]:
        import torch

        n = len(rgb)
        input_frames = np.stack(rgb, axis=0)

        torch.cuda.empty_cache()
        device_str = str(self._device)
        try:
            with torch.inference_mode():
                pred, _ = self._model.infer_video_depth(
                    input_frames, target_fps=-1, input_size=self._input_size,
                    device=device_str, fp32=self._fp32,
                )
            if isinstance(pred, torch.Tensor):
                pred = pred.detach().cpu().numpy()
            pred = np.squeeze(pred)
        except Exception as e:
            torch.cuda.empty_cache()
            raise RuntimeError(
                f"VDA infer_video_depth failed (metric={self._metric}, "
                f"frames={len(rgb)}, input_size={self._input_size}): {e}"
            ) from e
        torch.cuda.empty_cache()

        if pred is None or pred.size == 0:
            return [np.zeros_like(rgb[0], dtype=np.float32) for _ in rgb]

        if self._metric:
            depths_mm = pred * 1000.0  # m → mm
        else:
            # relative 模型输出任意尺度，直接当 mm 用
            depths_mm = pred.astype(np.float32)
            if self._invert and depths_mm.size > 0:
                depths_mm = depths_mm.max() - depths_mm + depths_mm.min()

        return [d.astype(np.float32) for d in depths_mm]
