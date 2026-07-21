"""
concat_videos.py — 拼接深度/点云视频为对比网格

用法:
    python docs/concat_videos.py                    # 全部 4 个网格
    python docs/concat_videos.py --type pcd         # 只拼点云
    python docs/concat_videos.py --type depth       # 只拼深度
    python docs/concat_videos.py --camera head      # 只拼 head

输出:
    docs/images/grid_depth_head.mp4    # 深度图网格
    docs/images/grid_depth_right.mp4
    docs/images/grid_pcd_head.mp4      # 点云网格
    docs/images/grid_pcd_right.mp4
"""

from __future__ import annotations

import argparse
import subprocess as sp
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "tests" / "output"
OUT_DIR = ROOT / "docs" / "images"

# 视频文件顺序与显示名映射（按策略逐个列出）
STRATEGIES = [
    ("original",    "Original"),
    ("mono",        "DA3 MONO"),
    ("metric",      "DA3 METRIC"),
    ("metric_temporal", "DA3 METRIC+Temporal"),
    ("nested", "DA3NESTED GIANT LARGE-1.1"),
    ("large11",     "DA3 LARGE-1.1"),
    ("vda",         "VDA (relative)"),
    ("vda_metric",  "VDA Metric"),
    ("lingbot",     "LingBot v1"),
]

COLS = 3


def find_video(prefix: str, camera: str, suffix: str) -> Path | None:
    """查找匹配的视频文件。"""
    name = f"{prefix}_{camera}{suffix}.mp4"
    path = INPUT_DIR / name
    return path if path.exists() else None


def concat_grid(videos: list[Path], labels: list[str], out_path: Path) -> None:
    """Python 逐帧拼接网格视频（带标题，中心裁剪 50%）。"""
    if out_path.exists():
        print(f"  skip (exists): {out_path}")
        return

    caps = [cv2.VideoCapture(str(v)) for v in videos]
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    W = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 中心裁剪 50%
    crop_W, crop_H = W // 2, H // 2
    cx0, cy0 = (W - crop_W) // 2, (H - crop_H) // 2

    n = len(videos)
    n_rows = (n + COLS - 1) // COLS
    title_h = 28
    cell_H = crop_H + title_h
    grid_W, grid_H = COLS * crop_W, n_rows * cell_H

    placeholder = np.full((crop_H, crop_W, 3), 255, dtype=np.uint8)

    tmp_avi = out_path.with_suffix(".avi")
    writer = cv2.VideoWriter(str(tmp_avi), cv2.VideoWriter_fourcc(*"MJPG"), fps, (grid_W, grid_H))

    while True:
        grid = np.full((grid_H, grid_W, 3), 255, dtype=np.uint8)
        all_done = True
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                cropped = placeholder
            else:
                all_done = False
                cropped = frame[cy0:cy0 + crop_H, cx0:cx0 + crop_W]
            ri, ci = i // COLS, i % COLS
            y0, x0 = ri * cell_H, ci * crop_W
            cv2.rectangle(grid, (x0, y0), (x0 + crop_W, y0 + title_h), (240, 240, 240), -1)
            cv2.putText(grid, labels[i], (x0 + 6, y0 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 2)
            grid[y0 + title_h:y0 + cell_H, x0:x0 + crop_W] = cropped
        if all_done:
            break
        writer.write(grid)

    for cap in caps:
        cap.release()
    writer.release()

    sp.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(tmp_avi), "-c:v", "libx265", "-preset", "slow", "-crf", "12",
            "-pix_fmt", "yuv420p", str(out_path)], check=True)
    tmp_avi.unlink()
    print(f"  -> {out_path}")


def build_grid(suffix: str, camera: str, out_name: str) -> None:
    """为指定 suffix/camera 构建网格视频。"""
    videos = []
    labels = []
    for prefix, label in STRATEGIES:
        v = find_video(prefix, camera, suffix)
        if v:
            videos.append(v)
            labels.append(label)
        else:
            print(f"  [warn] missing: {prefix}_{camera}{suffix}.mp4")

    if not videos:
        print(f"  no videos found for {camera}{suffix}")
        return

    print(f"\nBuilding {out_name} ({len(videos)} videos):")
    out_path = OUT_DIR / f"{out_name}.mp4"
    concat_grid(videos, labels, out_path)


def main():
    parser = argparse.ArgumentParser(description="拼接深度/点云对比网格")
    parser.add_argument("--type", choices=["depth", "pcd", "all"], default="all")
    parser.add_argument("--camera", choices=["head", "right", "all"], default="all")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    do_depth = args.type in ("depth", "all")
    do_pcd = args.type in ("pcd", "all")

    for cam in (["head", "right"] if args.camera == "all" else [args.camera]):
        if do_depth:
            build_grid("", cam, f"grid_depth_{cam}")
        if do_pcd:
            build_grid("_pcd", cam, f"grid_pcd_{cam}")

    print("\nDone.")


if __name__ == "__main__":
    main()
