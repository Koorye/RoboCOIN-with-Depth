"""
lingbot.py — LingBot-Depth 深度补全 / 精化
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party" / "lingbot-depth"))

from src.depth_repair.base import BaseDepthRepair


class LingBotDepthRepair(BaseDepthRepair):
    """LingBot-Depth — 联合 RGB + 原始深度做深度补全。

    Parameters
    ----------
    model_id : str
        HuggingFace 模型 ID 或本地路径。
    intrinsics : tuple
        相机内参 (fx, fy, cx, cy) 像素单位。
    apply_mask : bool
        是否应用模型输出的有效性 mask。
    device : str
        计算设备。
    """

    def __init__(
        self,
        model_id: str = "robbyant/lingbot-depth-pretrain-vitl-14-v0.5",
        intrinsics: tuple[float, float, float, float] = (247.0, 247.0, 128.0, 128.0),
        apply_mask: bool = True,
        device: str = "auto",
    ) -> None:
        super().__init__(device=device)
        self._model_id = model_id
        self._intrinsics = intrinsics
        self._apply_mask = apply_mask

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        from mdm.model.v2 import MDMModel

        self._model = MDMModel.from_pretrained(self._model_id).to(self._device)

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------

    def _repair_impl(
        self, rgb: list[np.ndarray], depth: list[np.ndarray]
    ) -> list[np.ndarray]:
        import torch

        if not depth:
            raise ValueError("LingBot 策略需要原始深度帧")

        h, w = rgb[0].shape[:2]
        K = self._normalize_intrinsics(h, w)
        K_t = torch.tensor(K, dtype=torch.float32, device=self._device).unsqueeze(0)

        repaired: list[np.ndarray] = []
        for r, d in zip(rgb, depth):
            r_t = (
                torch.tensor(r / 255.0, dtype=torch.float32, device=self._device)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )  # (1, 3, H, W)

            d_m = d / 1000.0  # mm → m
            d_m[~np.isfinite(d_m)] = 0.0
            d_t = torch.tensor(d_m, dtype=torch.float32, device=self._device).unsqueeze(0)

            with torch.no_grad():
                out = self._model.infer(
                    r_t, depth_in=d_t, intrinsics=K_t, apply_mask=self._apply_mask,
                )
            d_refined = out["depth"].squeeze().cpu().numpy()  # (H, W) meter
            repaired.append((d_refined * 1000.0).astype(np.float32))

        return repaired

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _normalize_intrinsics(self, h: int, w: int) -> np.ndarray:
        fx, fy, cx, cy = self._intrinsics
        return np.array([
            [fx / w, 0.0,    cx / w],
            [0.0,    fy / h, cy / h],
            [0.0,    0.0,    1.0],
        ], dtype=np.float32)
