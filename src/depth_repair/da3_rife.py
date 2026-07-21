"""
da3_rife.py — DA3 + RIFE 插帧深度修复

继承 DA3DepthRepair，均匀采样关键帧用 DA3 推理，RIFE 插值补全中间帧。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from src.depth_repair.da3 import DA3DepthRepair

# RIFE 路径
_RIFE_HOME = Path(__file__).resolve().parents[2] / "third_party" / "Practical-RIFE"


class DA3RIFEDepthRepair(DA3DepthRepair):
    """DA3 + RIFE 插帧。

    采样关键帧 → DA3 推理 → RIFE 插值补帧。

    Parameters
    ----------
    sample_stride : int
        采样步长。2 = 每 2 帧取 1 帧（省一半 GPU），默认 1 = 不采样。
    rife_checkpoint : str | None
        RIFE 权重目录（train_log），默认自动查找。
    其他参数同 DA3DepthRepair。
    """

    def __init__(
        self,
        sample_stride: int = 1,
        rife_home: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._sample_stride = sample_stride
        self._rife_home = Path(rife_home) if rife_home else _RIFE_HOME
        self._rife_model = None

    # ------------------------------------------------------------------
    # RIFE 模型加载
    # ------------------------------------------------------------------

    def _load_rife(self):
        """延迟加载 RIFE 模型。"""
        if self._rife_model is not None:
            return

        home = str(self._rife_home)
        if home not in sys.path:
            sys.path.insert(0, home)

        from train_log.RIFE_HDv3 import Model
        self._rife_model = Model()
        self._rife_model.load_model(
            str(self._rife_home / "train_log"), rank=-1)
        self._rife_model.eval()
        self._rife_model.device()

    # ------------------------------------------------------------------
    # 推理（覆盖父类）
    # ------------------------------------------------------------------

    def _repair_impl(
        self, rgb: list[np.ndarray], depth: list[np.ndarray]
    ) -> list[np.ndarray]:
        n = len(rgb)
        stride = self._sample_stride

        if stride <= 1 or n <= stride:
            return super()._repair_impl(rgb, depth)

        # 1. 采样关键帧（始终包含最后一帧）
        key_idx = sorted(set(range(0, n, stride)) | {n - 1})

        # 如果采样后帧数不变，直接走父类
        if len(key_idx) == n:
            return super()._repair_impl(rgb, depth)

        key_rgb = [rgb[i] for i in key_idx]

        # 2. DA3 推理关键帧
        key_depth = super()._repair_impl(key_rgb, depth)

        # 3. RIFE 插值补全（RGB 算光流 → warp 深度）
        self._load_rife()
        H, W = rgb[0].shape[:2]
        result = self._interpolate(key_rgb, key_depth, key_idx, n, H, W)
        assert len(result) == n, f"frame count mismatch: {len(result)} != {n}"
        return result

    # ------------------------------------------------------------------
    # RIFE 插值
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _interpolate(
        self,
        key_rgb: list[np.ndarray],
        key_depth: list[np.ndarray],
        key_idx: list[int],
        n: int,
        H: int, W: int,
    ) -> list[np.ndarray]:
        """RGB 算光流 → warp 深度。

        RIFE 在 RGB 上估算光流（纹理丰富），然后将流应用到深度图 warp。
        深度图不需要归一化——warp 只做空间变换，值不变。
        """
        import torch.nn.functional as F
        from model.warplayer import warp

        all_depth: list[np.ndarray | None] = [None] * n
        for i, d in zip(key_idx, key_depth):
            all_depth[i] = d

        # pad 尺寸
        max_h = max(d.shape[0] for d in key_depth)
        max_w = max(d.shape[1] for d in key_depth)
        PAD = 64
        ph = ((max_h - 1) // PAD + 1) * PAD
        pw = ((max_w - 1) // PAD + 1) * PAD

        # 预转换所有关键帧为 tensor
        rgb_t = {}   # idx → (1, 3, ph, pw) on GPU
        depth_t = {} # idx → (1, 1, ph, pw) on GPU
        for i, (rgb, d) in enumerate(zip(key_rgb, key_depth)):
            idx = key_idx[i]
            rgb_t[idx] = self._rgb_to_tensor(rgb, ph, pw)
            depth_t[idx] = self._depth_to_warp_tensor(d, ph, pw)

        scale_list = [16, 8, 4, 2, 1]

        for k in range(len(key_idx) - 1):
            i0, i1 = key_idx[k], key_idx[k + 1]
            gap = i1 - i0
            if gap <= 1:
                continue

            # RGB 拼接 → flownet 输入
            imgs = torch.cat((rgb_t[i0], rgb_t[i1]), 1)  # (1, 6, ph, pw)
            d0 = depth_t[i0]
            d1 = depth_t[i1]

            for t in range(1, gap):
                timestep = t / gap
                flow_list, mask, _ = self._rife_model.flownet(
                    imgs, timestep, scale_list=scale_list)

                flow = flow_list[-1]          # (1, 4, ph, pw)
                m = torch.sigmoid(mask)       # occlusion mask

                # warp 深度
                w0 = warp(d0, flow[:, :2])
                w1 = warp(d1, flow[:, 2:4])
                merged = w0 * m + w1 * (1 - m)

                # 轻量平滑：去除 RIFE 粗尺度流场的网格伪影
                import torchvision.transforms.functional as TF
                merged = TF.gaussian_blur(merged, kernel_size=5, sigma=1.0)

                all_depth[i0 + t] = merged[0, 0, :H, :W].cpu().numpy()

        return _fill_none(all_depth, H, W)

    # ------------------------------------------------------------------
    # tensor 转换
    # ------------------------------------------------------------------

    @staticmethod
    def _rgb_to_tensor(rgb: np.ndarray, ph: int, pw: int):
        """(H, W, 3) uint8 → (1, 3, ph, pw) float32 [0,1] on GPU."""
        import torch.nn.functional as F

        h, w = rgb.shape[:2]
        t = torch.from_numpy(rgb).float().cuda() / 255.0
        t = t.permute(2, 0, 1)[None, :, :, :]     # (1, 3, H, W)
        if ph > h or pw > w:
            t = F.pad(t, (0, pw - w, 0, ph - h), mode='replicate')
        return t

    @staticmethod
    def _depth_to_warp_tensor(d: np.ndarray, ph: int, pw: int):
        """(H, W) float32 → (1, 1, ph, pw) on GPU（不做归一化）。"""
        import torch.nn.functional as F

        h, w = d.shape
        t = torch.from_numpy(d).float().cuda()
        t = t[None, None, :, :]                    # (1, 1, H, W)
        if ph > h or pw > w:
            t = F.pad(t, (0, pw - w, 0, ph - h), mode='replicate')
        return t


# ============================================================================
# helpers
# ============================================================================

def _fill_none(
    results: list, H_orig: int, W_orig: int,
) -> list[np.ndarray]:
    """将 None 填充为零数组。"""
    return [d if d is not None else np.zeros((H_orig, W_orig), dtype=np.float32)
            for d in results]
