#!/usr/bin/env python3
"""
将 RoboCOIN HDF5 轨迹转换为 LeRobot 格式数据集（含深度）。

- 保存所有 RGB 相机（chest / head / left / right）
- 保存所有深度相机（chest / head / left / right），编码为 12-bit H.265
- arm left + effector left + arm right + effector right 拼接为
  ``observation.state``，同时作为 ``action``

用法:
    python scripts/create_lerobot_dataset_with_depth.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.load_hdf5 import load_h5
from src.lerobot_with_depth_dataset import LeRobotWithDepthDataset


_RGB_CAMERAS: dict[str, str] = {
    "observations/camera/rgb/chest/images": "observations.images.rgb.chest",
    "observations/camera/rgb/head/images":  "observations.images.rgb.head",
    "observations/camera/rgb/left/images":  "observations.images.rgb.left",
    "observations/camera/rgb/right/images": "observations.images.rgb.right",
}

_DEPTH_CAMERAS: dict[str, str] = {
    "observations/camera/depth/chest/images": "observations.images.depth.chest",
    "observations/camera/depth/head/images":  "observations.images.depth.head",
    "observations/camera/depth/left/images":  "observations.images.depth.left",
    "observations/camera/depth/right/images": "observations.images.depth.right",
}

# depth_h5_path → 对应 rgb_h5_path（用于 repair）
_DEPTH_TO_RGB: dict[str, str] = {
    "observations/camera/depth/chest/images": "observations/camera/rgb/chest/images",
    "observations/camera/depth/head/images":  "observations/camera/rgb/head/images",
    "observations/camera/depth/left/images":  "observations/camera/rgb/left/images",
    "observations/camera/depth/right/images": "observations/camera/rgb/right/images",
}

_STATE_PLAN: list[tuple[str, list[str]]] = [
    ("observations/arm/left/joints", [
        "left_joint_1", "left_joint_2", "left_joint_3", "left_joint_4",
        "left_joint_5", "left_joint_6", "left_joint_7",
    ]),
    ("observations/effector/left/position", ["left_gripper"]),
    ("observations/arm/right/joints", [
        "right_joint_1", "right_joint_2", "right_joint_3", "right_joint_4",
        "right_joint_5", "right_joint_6", "right_joint_7",
    ]),
    ("observations/effector/right/position", ["right_gripper"]),
]

_STATE_NAMES: list[str] = [n for _, names in _STATE_PLAN for n in names]
_STATE_DIM: int = len(_STATE_NAMES)
DEFAULT_FPS = 30


def _resolve(data: dict, path: str):
    for seg in path.split("/"):
        data = data[seg]
    return data


def _subsample(data: dict, indices: np.ndarray):
    """原地对 data 中所有沿时间轴的 list 做均匀采样（图像帧）。"""
    for key in list(data.keys()):
        val = data[key]
        if isinstance(val, list) and len(val) > len(indices):
            data[key] = [val[i] for i in indices]
        elif isinstance(val, dict):
            _subsample(val, indices)
        elif isinstance(val, np.ndarray) and val.ndim >= 1 and len(val) > len(indices):
            data[key] = val[indices]


def main():
    parser = argparse.ArgumentParser(description="RoboCOIN HDF5 → LeRobot (含深度)")
    
    parser.add_argument("--h5-dir", type=str)
    parser.add_argument("--repo-id", type=str)
    parser.add_argument("--root", type=str, default="data/lerobot")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--task", type=str, default="arrange_teaset")
    parser.add_argument("--image-writer-threads", type=int, default=16)
    parser.add_argument("--normalize-depth", action="store_true", help="启用 per-episode min-max 归一化（默认关闭，直接存mm）")
    parser.add_argument("--sample-frames", type=int, default=0, help="均匀采样帧数（0=全部，用于快速测试）")

    # depth repair
    parser.add_argument("--repair", action="store_true", help="启用深度修复")
    parser.add_argument("--repair-strategy", choices=["da3", "vda", "lingbot"], default="da3")

    # da3
    parser.add_argument("--da3-model", default="depth-anything/DA3METRIC-LARGE")
    parser.add_argument("--da3-process-res", type=int, default=504)
    parser.add_argument("--da3-chunk-size", type=int, default=0)
    parser.add_argument("--da3-overlap", type=int, default=0)
    parser.add_argument("--da3-temporal-alpha", type=float, default=0.0)

    # vda
    parser.add_argument("--vda-encoder", choices=["vits", "vitb", "vitl"], default="vitl")
    parser.add_argument("--vda-checkpoint", default="checkpoints/video_depth_anything_vitl.pth")
    parser.add_argument("--vda-input-size", type=int, default=378)
    parser.add_argument("--vda-metric", action="store_true", default=True)
    parser.add_argument("--vda-invert", action="store_true", default=False)
    parser.add_argument("--vda-fp32", action="store_true", default=False)

    # lingbot v1
    parser.add_argument("--lingbot-model", default="robbyant/lingbot-depth-pretrain-vitl-14-v0.5")
    parser.add_argument("--lingbot-intrinsics", type=float, nargs=4,
                        default=[247.0, 247.0, 128.0, 128.0])

    args = parser.parse_args()

    h5_files = sorted(Path(args.h5_dir).rglob("*.h5"))

    first = load_h5(h5_files[0])
    rgb_shape = _resolve(first, next(iter(_RGB_CAMERAS)))[0].shape  # (H, W, C)
    depth_shape = _resolve(first, next(iter(_DEPTH_CAMERAS)))[0].shape  # (H, W)
    depth_shape = (*depth_shape, 3)  # (H, W) -> (H, W, 3)

    features: dict = {}
    for key in _RGB_CAMERAS.values():
        features[key] = {"dtype": "video", "shape": rgb_shape, "names": ["height", "width", "channel"]}
    for key in _DEPTH_CAMERAS.values():
        features[key] = {"dtype": "video", "shape": depth_shape, "names": ["height", "width", "channel"]}
    features["observation.state"] = {"dtype": "float32", "shape": (_STATE_DIM,), "names": _STATE_NAMES}
    features["action"] = {"dtype": "float32", "shape": (_STATE_DIM,), "names": _STATE_NAMES}

    repair = None
    if args.repair:
        from src.depth_repair import create_depth_repair

        strategy = args.repair_strategy
        if strategy == "da3":
            repair = create_depth_repair(
                "da3",
                model_id=args.da3_model,
                process_res=args.da3_process_res,
                chunk_size=args.da3_chunk_size,
                overlap=args.da3_overlap,
                temporal_alpha=args.da3_temporal_alpha,
            )
        elif strategy == "vda":
            repair = create_depth_repair(
                "vda",
                checkpoint=args.vda_checkpoint,
                encoder=args.vda_encoder,
                input_size=args.vda_input_size,
                metric=args.vda_metric,
                invert=args.vda_invert,
                fp32=args.vda_fp32,
            )
        elif strategy == "lingbot":
            repair = create_depth_repair(
                "lingbot",
                model_id=args.lingbot_model,
                intrinsics=tuple(args.lingbot_intrinsics),
            )

    dataset = LeRobotWithDepthDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=args.root,
        use_videos=True,
        image_writer_threads=args.image_writer_threads,
        normalize_depth=args.normalize_depth,
    )

    for i_file, h5_path in enumerate(h5_files):
        print(f"[{i_file + 1}/{len(h5_files)}] {h5_path.name}")
        data = load_h5(h5_path)
        n = len(_resolve(data, "observations/timestamp"))

        # 均匀采样（快速测试）
        if args.sample_frames > 0 and args.sample_frames < n:
            idx = np.linspace(0, n - 1, args.sample_frames, dtype=int)
            # 对所有时序数组做采样
            _subsample(data, idx)
            n = args.sample_frames
            print(f"  Sampled {n} frames")

        if repair is not None:
            for depth_h5_path, rgb_h5_path in _DEPTH_TO_RGB.items():
                rgb_frames = _resolve(data, rgb_h5_path)
                depth_frames = _resolve(data, depth_h5_path)
                depth_frames[:] = repair.repair_frames(rgb_frames, depth_frames)

        for i in tqdm(range(n), leave=False):
            state = np.concatenate(
                [_resolve(data, p)[i].ravel().astype(np.float32) for p, _ in _STATE_PLAN]
            )
            frame: dict = {"observation.state": state, "action": state}

            for h5_key, lr_key in _RGB_CAMERAS.items():
                frame[lr_key] = _resolve(data, h5_key)[i]
            for h5_key, lr_key in _DEPTH_CAMERAS.items():
                frame[lr_key] = _resolve(data, h5_key)[i]

            dataset.add_frame(frame, task=args.task)

        dataset.save_episode()

    dataset.finalize()


if __name__ == "__main__":
    main()
