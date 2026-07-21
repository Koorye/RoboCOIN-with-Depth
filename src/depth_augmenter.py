"""
depth_augmenter.py — 为已有 LeRobot 数据集增量添加深度和点云数据

用法:
    from src.depth_augmenter import DepthAugmenter

    augmenter = DepthAugmenter("your_org/dataset", root="data/lerobot")
    augmenter.run(strategy="da3", model_id="depth-anything/DA3METRIC-LARGE")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.depth_encoder import DepthVideoEncoder

CAMERA_INTRINSICS = {
    "head":  (605, 605, 323, 252, 225),
    "right": (434, 434, 314, 239, 45),
    "left":  (434, 434, 314, 239, 45),
    "chest": (605, 605, 323, 252, 0),
}


class DepthAugmenter:
    """读取已有 LeRobot 数据集，创建带深度的新数据集。

    对每个 episode 逐帧推理深度，可选生成点云。通过 add_frame
    将 RGB + depth + pcd 写入新数据集。
    """

    def __init__(
        self, repo_id: str, root: str = "data/lerobot",
        strategy: str = "da3",
        fps: int = 30, crf: int = 6, preset: str = "slow",
        **repair_kwargs,
    ):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.repo_id = repo_id
        self.root = Path(root)
        self.src = LeRobotDataset(repo_id, root=root)
        self._rgb_keys = [k for k in self.src.features
                          if "rgb" in k and "observation.images" in k]
        if not self._rgb_keys:
            raise ValueError(f"No RGB cameras found in {repo_id}")

        from src.depth_repair import create_depth_repair
        self._repair = create_depth_repair(strategy, **repair_kwargs)
        self._encoder = DepthVideoEncoder(fps=fps, crf=crf, preset=preset)
        self._fps = fps

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def run(
        self,
        output_repo_id: str | None = None,
        output_root: str | None = None,
        normalize: bool = False,
        save_pcd: bool = False,
    ):
        """创建带深度的新数据集。

        Parameters
        ----------
        output_repo_id : str | None
            输出 repo_id，默认 "{src_repo_id}_with_depth"。
        output_root : str | None
            输出 root，默认同源数据集 root。
        normalize : bool
            per-episode min-max 归一化。
        save_pcd : bool
            逐帧保存点云数据。
        """
        out_id = output_repo_id or f"{self.repo_id}_with_depth"
        out_root = output_root or str(self.root)
        features = dict(self.src.features)
        depth_keys = {}
        pcd_keys = {}

        # 深度 + 点云 features
        for rgb_key in self._rgb_keys:
            depth_key = rgb_key.replace("rgb", "depth")
            depth_keys[rgb_key] = depth_key
            feat = features[rgb_key]
            features[depth_key] = {
                "dtype": "video",
                "shape": feat["shape"][:2],  # (H, W) 无 channel
                "names": ["height", "width"],
            }
            if save_pcd:
                pcd_key = depth_key.replace(".images.", ".pointcloud.")
                pcd_keys[rgb_key] = pcd_key
                H, W = feat["shape"][:2]
                N = pcd_num_points(H, W, stride=7)
                features[pcd_key] = {
                    "dtype": "float32",
                    "shape": (N, 3),
                    "names": ["index", "xyz"],
                }

        # 创建输出数据集
        from src.lerobot_with_depth_dataset import LeRobotWithDepthDataset
        dst = LeRobotWithDepthDataset.create(
            repo_id=out_id, root=out_root, fps=self._fps,
            features=features, normalize_depth=normalize,
        )

        # 获取源数据集 episode 边界
        ep_from = self.src.episode_data_index["from"].tolist()
        ep_to = self.src.episode_data_index["to"].tolist()

        for ep_idx in tqdm(range(len(ep_from)), desc="episodes"):
            fi0, fi1 = ep_from[ep_idx], ep_to[ep_idx]

            # 按 episode 读取 RGB 帧
            rgb_ep: dict[str, list] = {k: [] for k in self._rgb_keys}
            for fi in tqdm(range(fi0, fi1), desc="frames"):
                frame = self.src[fi]
                for k in self._rgb_keys:
                    rgb_ep[k].append(np.uint8(frame[k].permute(1, 2, 0).numpy() * 255))  # (C,H,W) → (H,W,C)
            
            tasks = [self.src[frame_idx]["task"] for frame_idx in range(fi0, fi1)]

            # 推理深度
            depth_ep: dict[str, list] = {}
            pcd_ep: dict[str, list] = {}
            for rgb_key in self._rgb_keys:
                rgbs = rgb_ep[rgb_key]
                depths = self._repair.repair_frames(rgbs, [])
                depth_ep[rgb_key] = depths

                if save_pcd:
                    cam = _cam_from_key(rgb_key)
                    intr = CAMERA_INTRINSICS.get(cam, (605, 605, 323, 252, 0))
                    N = pcd_num_points(rgbs[0].shape[0], rgbs[0].shape[1], stride=7)
                    pcds = []
                    for d, r in zip(depths, rgbs):
                        pts, _ = _depth_to_pointcloud(
                            d, r, intr[0], intr[1], intr[2], intr[3], 5000, stride=7,
                            camera_pitch_deg=intr[4],
                        )
                        # 填充到固定长度
                        padded = np.zeros((N, 3), dtype=np.float32)
                        n_valid = min(len(pts), N)
                        padded[:n_valid] = pts[:n_valid]
                        pcds.append(padded)
                    pcd_ep[rgb_key] = pcds

            # add_frame
            n = fi1 - fi0
            for i in range(n):
                frame_dict = {}
                fi = fi0 + i
                src_frame = self.src[fi]

                # RGB
                for k in self._rgb_keys:
                    frame_dict[k] = rgb_ep[k][i]
                # Depth
                for rgb_key, depth_key in depth_keys.items():
                    frame_dict[depth_key] = depth_ep[rgb_key][i]
                # PCD
                for rgb_key, pcd_key in pcd_keys.items():
                    frame_dict[pcd_key] = pcd_ep[rgb_key][i]
                # State / Action
                if "observation.state" in src_frame:
                    frame_dict["observation.state"] = src_frame["observation.state"].numpy()
                if "action" in src_frame:
                    frame_dict["action"] = src_frame["action"].numpy()

                dst.add_frame(frame_dict, task=tasks[i])

            dst.save_episode()

        dst.finalize()
        print(f"\nDone → {out_id}")


# ============================================================================
# 点云工具
# ============================================================================

def pcd_num_points(H: int, W: int, stride: int = 3) -> int:
    """根据视频尺寸和采样步长计算点云最大点数。"""
    return (H // stride) * (W // stride)


def _cam_from_key(key: str) -> str:
    """observation.images.rgb.head → head"""
    return key.rsplit(".", 1)[-1]


def _depth_to_pointcloud(
    depth: np.ndarray, rgb: np.ndarray | None,
    fx: float, fy: float, cx: float, cy: float,
    max_depth_mm: float = 5000, stride: int = 1,
    camera_pitch_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    H, W = depth.shape
    u = np.arange(0, W, stride)
    v = np.arange(0, H, stride)
    uu, vv = np.meshgrid(u, v)
    d = depth[vv, uu].astype(np.float32)
    valid = (d > 0) & np.isfinite(d) & (d < max_depth_mm)
    z = d[valid] / 1000.0
    x = (uu[valid] - cx) * z / fx
    y = (vv[valid] - cy) * z / fy
    pts = np.stack([x, y, z], axis=-1).astype(np.float32)
    if camera_pitch_deg != 0:
        pts = _rotate_x(pts, np.deg2rad(camera_pitch_deg))
    cols = rgb[vv, uu][valid] if rgb is not None else None
    return pts, cols


def _rotate_x(points: np.ndarray, angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
    return points @ R.T
