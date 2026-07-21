import json
import numpy as np
import os
import shutil

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from .video_utils import decode_video_frames

from src.depth_encoder import DepthVideoEncoder

_DEFAULT_K = np.array([[605, 0, 323], [0, 605, 252], [0, 0, 1]], dtype=np.float32)


class LeRobotWithDepthDataset(LeRobotDataset):
    """继承 LeRobotDataset，增加深度视频编码和元数据注入能力。

    用法:
        dataset = LeRobotWithDepthDataset.create(
            root=..., repo_id=..., fps=..., features=...,
        )
        for frame_data in episode_frames:
            dataset.add_frame(frame_data, task='...')
        dataset.save_episode()
        dataset.finalize()

    深度字段自动检测:
        features 中以 ``"observation"`` 开头且路径中包含 ``"depth"`` 的
        key 会被自动识别为深度相机，无需额外配置。

    深度精度: 每条流水线独立归一化并编码为12-bit H.265。
    """

    # ------------------------------------------------------------------
    # 深度字段自动检测
    # ------------------------------------------------------------------

    @staticmethod
    def _split_depth_features(features: dict) -> tuple[dict, dict, list[str]]:
        """从 features 中分离深度字段。

        Returns
        -------
        non_depth_features : dict
            去除深度字段后的 features，传给 ``super().create()``。
        depth_features : dict
            被剥离的深度字段，供后续 ``_inject_depth_info()`` 注入。
        depth_keys : list[str]
            深度字段的 key 列表。
        """
        depth_keys = sorted(
            key for key in features
            if key.startswith("observation.images") and "depth" in key
        )
        depth_features = {k: features[k] for k in depth_keys}
        non_depth_features = {k: v for k, v in features.items() if k not in depth_keys}
        return non_depth_features, depth_features, depth_keys

    @property
    def depth_keys(self) -> list[str]:
        """自动检测到的深度特征 key 列表。"""
        return self._depth_keys

    def _has_depth(self) -> bool:
        return len(self.depth_keys) > 0

    # ------------------------------------------------------------------
    # 创建
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, *, root, repo_id, fps, features, normalize_depth: bool = False,
               enable_pcd: bool = False, pcd_with_rgb: bool = False,
               pcd_cameras: dict | None = None, pcd_stride: int = 3, **kwargs):
        non_depth_features, depth_features, depth_keys = cls._split_depth_features(features)
        instance = super().create(
            root=root, repo_id=repo_id, fps=fps, features=non_depth_features, **kwargs,
        )
        instance._depth_features = depth_features
        instance._depth_keys = depth_keys
        instance._normalize_depth = normalize_depth
        instance._enable_pcd = enable_pcd
        instance._pcd_with_rgb = pcd_with_rgb
        instance._pcd_cameras = pcd_cameras or {}
        instance._pcd_stride = pcd_stride
        instance._depth_encoder = DepthVideoEncoder(fps=fps)
        instance._pending_depths: dict[str, list] = {}
        instance._depth_stats: dict[str, list] = {}
        instance._episode_counter = 0
        return instance

    # ------------------------------------------------------------------
    # 帧操作
    # ------------------------------------------------------------------

    def add_frame(self, frame: dict, task: str, timestamp: float | None = None) -> None:
        """添加一帧数据。深度帧被分离暂存，其余传入父类。"""
        if self._has_depth():
            non_depth: dict = {}
            for key, val in frame.items():
                if key in self.depth_keys:
                    self._pending_depths.setdefault(key, []).append(val)
                else:
                    non_depth[key] = val
            super().add_frame(non_depth, task, timestamp=timestamp)
        else:
            super().add_frame(frame, task, timestamp=timestamp)

    # ------------------------------------------------------------------
    # __getitem__ — 深度取 channel 0, 可选点云
    # ------------------------------------------------------------------

    def __init__(self, *args,
                 enable_pcd: bool = False,
                 pcd_with_rgb: bool = False,
                 pcd_cameras: dict | None = None,
                 pcd_stride: int = 3,
                 **kwargs):
        """
        Parameters
        ----------
        enable_pcd : bool
            是否在线生成点云。
        pcd_with_rgb : bool
            点云是否附带 RGB 颜色 (N,6) = xyz+RGB。
        pcd_cameras : dict | None
            相机参数 {cam_name: {K: (3,3), extrinsics: (4,4) | None}}。
        pcd_stride : int
            点云采样步长。
        """
        super().__init__(*args, **kwargs)
        self._enable_pcd = enable_pcd
        self._pcd_with_rgb = pcd_with_rgb
        self._pcd_cameras = pcd_cameras or {}
        self._pcd_stride = pcd_stride

    def __getitem__(self, idx):
        import torch
        sample = super().__getitem__(idx)
        for key in list(sample.keys()):
            if "depth" in key and "images" in key:
                val = sample[key]
                if val.ndim == 3 and val.shape[0] in (1, 3, 4):
                    val = val[0]
                sample[key] = val

        if self._enable_pcd:
            for key in list(sample.keys()):
                if "rgb" in key and "images" in key:
                    depth_key = key.replace("rgb", "depth")
                    pcd_key = key.replace(".images.", ".pointcloud.").replace("rgb", "pcd")
                    if depth_key not in sample:
                        continue
                    cam = key.rsplit(".", 1)[-1].replace("_rgb", "").replace("cam_", "")
                    cfg = self._pcd_cameras.get(cam, {})
                    K = torch.as_tensor(cfg.get("K", _DEFAULT_K), dtype=torch.float32)
                    E = cfg.get("extrinsics")
                    if E is not None:
                        E = torch.as_tensor(E, dtype=torch.float32)
                    rgb = sample[key] if self._pcd_with_rgb else None
                    pts = self._depth_to_pointcloud(
                        sample[depth_key], K, E, rgb=rgb, stride=self._pcd_stride,
                    )
                    sample[pcd_key] = pts
        return sample

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int):
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault. This probably happens because a memory reference to the video loader is created in
        the main process and a subprocess fails to access it.
        """
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            frames = decode_video_frames(video_path, query_ts, self.tolerance_s, self.video_backend)
            item[vid_key] = frames.squeeze(0)

        return item

    @staticmethod
    def _depth_to_pointcloud(
        depth, K, extrinsics=None, rgb=None, stride: int = 3,
    ):
        """深度图 → 点云 tensor。rgb 不为空时返回 (N,6) = xyz+rgb。"""
        import torch
        depth = torch.as_tensor(depth, dtype=torch.float32)
        K = torch.as_tensor(K, dtype=torch.float32)
        H, W = depth.shape
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        d_strided = depth[::stride, ::stride]
        Hs, Ws = d_strided.shape
        vv, uu = torch.meshgrid(
            torch.arange(Hs), torch.arange(Ws), indexing="ij",
        )
        d = d_strided
        valid = (d > 0) & torch.isfinite(d)
        z = d[valid] / 1000.0
        x = (uu[valid].float() * stride - cx) * z / fx
        y = (vv[valid].float() * stride - cy) * z / fy
        pts = torch.stack([x, y, z], dim=-1)
        if extrinsics is not None:
            extrinsics = torch.as_tensor(extrinsics, dtype=torch.float32)
            ones = torch.ones(len(pts), 1, dtype=torch.float32)
            cam_pts = torch.cat([pts, ones], dim=1)
            pts = (cam_pts @ extrinsics.T)[:, :3]
        if rgb is not None:
            rgb = torch.as_tensor(rgb, dtype=torch.float32)
            if rgb.max() > 1.0:
                rgb = rgb / 255.0
            # handle CHW vs HWC
            if rgb.ndim == 3 and rgb.shape[0] == 3:
                rgb = rgb.permute(1, 2, 0)  # CHW → HWC
            rgb_strided = rgb[::stride, ::stride, :].reshape(-1, 3)
            colors = rgb_strided[valid.flatten()]
            pts = torch.cat([pts, colors], dim=-1)  # (N, 6)
        return pts

    # ------------------------------------------------------------------
    # Episode 保存
    # ------------------------------------------------------------------

    def save_episode(self, episode_data: dict | None = None) -> None:
        """保存一个episode。父类写入RGB视频+parquet后编码深度视频。"""
        episode_index = self._episode_counter
        super().save_episode(episode_data=episode_data)
        if self._pending_depths:
            self._encode_depth(episode_index)
            self._pending_depths.clear()
        self._episode_counter += 1

    # ------------------------------------------------------------------
    # 后处理
    # ------------------------------------------------------------------

    def finalize(self):
        self._rename_tmp_files()
        if self._has_depth():
            self._inject_depth_stats()
            self._inject_depth_info()

    # ==================================================================
    # 内部实现
    # ==================================================================

    def _encode_depth(self, episode_index: int):
        chunk = self.meta.get_episode_chunk(episode_index)
        chunk_name = f"chunk-{chunk:03d}"

        for depth_key in self.depth_keys:
            depths = self._pending_depths[depth_key]
            video_dir = os.path.join(self.root, "videos", chunk_name, depth_key)
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(
                video_dir, f"episode_{int(episode_index):06d}.mp4"
            )

            # 统计（基于原始深度）
            self._depth_stats.setdefault(depth_key, []).append(
                self._depth_encoder.compute_stats(depths)
            )

            if self._normalize_depth:
                dmin, dmax = self._depth_encoder.normalize_params(depths, percentile=95)
                normalize = (dmin, dmax)
            else:
                normalize = None
                # 截断 >4095 的值到 12-bit 范围
                depths = [np.clip(np.round(d), 0, 4095).astype(np.float32) for d in depths]

            self._depth_encoder.encode_from_arrays(depths, video_path, normalize=normalize)

            shutil.move(video_path, video_path + ".tmp")

    def _rename_tmp_files(self):
        for dirpath, _, filenames in os.walk(self.root):
            for fn in filenames:
                if fn.endswith(".mp4.tmp"):
                    src = os.path.join(dirpath, fn)
                    shutil.move(src, src[:-4])

    def _inject_depth_stats(self):
        stat_path = os.path.join(self.root, "meta", "episodes_stats.jsonl")
        with open(stat_path) as f:
            stats = [json.loads(line.strip()) for line in f]

        for depth_key in self.depth_keys:
            depth_stats = self._depth_stats[depth_key]
            assert len(stats) == len(depth_stats), (
                f"Episode count mismatch for {depth_key}"
            )
            for i, ds in enumerate(depth_stats):
                stats[i]["stats"][depth_key] = ds

        with open(stat_path, "w") as f:
            for stat in stats:
                f.write(json.dumps(stat) + "\n")

    def _inject_depth_info(self):
        info_path = os.path.join(self.root, "meta", "info.json")
        with open(info_path) as f:
            info = json.load(f)

        # 从已有 RGB 视频 feature 推断 video info（fps 等），深度覆写 codec/pix_fmt
        video_info = self._build_depth_video_info(info.get("features", {}))

        for depth_key in self.depth_keys:
            feat = self._depth_features.get(depth_key, {})
            depth_entry = {
                "dtype": feat.get("dtype", "video"),
                "shape": feat.get("shape", [480, 640]),
                "names": feat.get("names", ["height", "width"]),
                "info": dict(video_info),
            }
            info.setdefault("features", {})[depth_key] = depth_entry

        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)

    def _build_depth_video_info(self, features: dict) -> dict:
        """从已有 features 推断深度视频的 video info。

        优先复用已有 video feature 的 fps/base path 等字段，
        深度特化: codec=libx265, pix_fmt=gray12le。
        """
        # 尝试从任意已有 video feature 复制通用字段
        for _key, feat in features.items():
            if isinstance(feat, dict) and "info" in feat:
                base = dict(feat["info"])
                base["video.codec"] = "libx265"
                base["video.pix_fmt"] = "gray12le"
                return base

        # 回退: 纯默认
        return {
            "video.fps": 30,
            "video.codec": "libx265",
            "video.pix_fmt": "gray12le",
        }
