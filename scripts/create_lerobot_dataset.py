#!/usr/bin/env python3
"""
将 RoboCOIN HDF5 轨迹转换为 LeRobot 格式数据集（不含深度）。

- 保存所有 RGB 相机（chest / head / left / right）
- arm left + effector left + arm right + effector right 拼接为
  ``observation.state``，同时作为 ``action``

用法:
    python scripts/create_lerobot_dataset.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.load_hdf5 import load_h5
from lerobot.datasets.lerobot_dataset import LeRobotDataset


_RGB_CAMERAS: dict[str, str] = {
    "observations/camera/rgb/chest/images": "observation.images.cam_chest_rgb",
    "observations/camera/rgb/head/images":  "observation.images.cam_head_rgb",
    "observations/camera/rgb/left/images":  "observation.images.cam_left_rgb",
    "observations/camera/rgb/right/images": "observation.images.cam_right_rgb",
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


def main():
    parser = argparse.ArgumentParser(description="RoboCOIN HDF5 → LeRobot (不含深度)")
    parser.add_argument("--h5-dir", type=str)
    parser.add_argument("--repo-id", type=str)
    parser.add_argument("--root", type=str, default="data/lerobot")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--task", type=str, default="arrange_teaset")
    parser.add_argument("--image-writer-threads", type=int, default=16)
    args = parser.parse_args()
    
    h5_files = sorted(Path(args.h5_dir).rglob("*.h5"))

    first = load_h5(h5_files[0])
    rgb_shape = _resolve(first, next(iter(_RGB_CAMERAS)))[0].shape  # (H, W, C)

    features: dict = {}
    for key in _RGB_CAMERAS.values():
        features[key] = {"dtype": "video", "shape": rgb_shape, "names": ["height", "width", "channel"]}
    features["observation.state"] = {"dtype": "float32", "shape": (_STATE_DIM,), "names": _STATE_NAMES}
    features["action"] = {"dtype": "float32", "shape": (_STATE_DIM,), "names": _STATE_NAMES}

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id, 
        fps=args.fps, 
        features=features,
        root=args.root, 
        use_videos=True,
        image_writer_threads=args.image_writer_threads,
    )

    for i, h5_path in enumerate(h5_files):
        print(f"Processing {h5_path} ({i + 1}/{len(h5_files)})...")
        data = load_h5(h5_path)
        n = len(_resolve(data, "observations/timestamp"))
        for i in tqdm(range(n)):
            state = np.concatenate(
                [_resolve(data, p)[i].ravel().astype(np.float32) for p, _ in _STATE_PLAN]
            )
            frame: dict = {"observation.state": state, "action": state}

            for h5_key, lr_key in _RGB_CAMERAS.items():
                frame[lr_key] = _resolve(data, h5_key)[i]

            dataset.add_frame(frame, task=args.task)
        
        dataset.save_episode()


if __name__ == "__main__":
    main()
