#!/usr/bin/env python3
"""
深度视频 → 点云 PCD。

用法:
    python scripts/depth_to_pcd.py --input depth.mp4 --output out_dir \
        --fx 605 --fy 605 --cx 323 --cy 252
"""

import argparse
import subprocess as sp
from pathlib import Path

import numpy as np


def decode_depth_video(path: str) -> list[np.ndarray]:
    """解码 gray12le 深度视频 → (H, W) float32 mm 帧列表。"""
    probe = sp.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {probe.stderr}")
    W, H = map(int, probe.stdout.strip().split(","))

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", path,
        "-f", "rawvideo", "-pix_fmt", "gray12le",
        "-",
    ]
    proc = sp.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {proc.stderr}")

    raw = np.frombuffer(proc.stdout, dtype=np.uint16)
    frame_bytes = W * H
    n_frames = len(raw) // frame_bytes
    frames = [raw[i * frame_bytes:(i + 1) * frame_bytes]
              .reshape(H, W).astype(np.float32)
              for i in range(n_frames)]
    return frames


def depth_to_pointcloud(
    depth: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    max_depth_mm: float = 5000,
    stride: int = 1,
) -> np.ndarray:
    """深度图 → (N, 3) 点云 (m)。"""
    H, W = depth.shape
    u = np.arange(0, W, stride)
    v = np.arange(0, H, stride)
    uu, vv = np.meshgrid(u, v)
    d = depth[vv, uu].astype(np.float32)
    valid = (d > 0) & np.isfinite(d) & (d < max_depth_mm)
    z = d[valid] / 1000.0
    x = (uu[valid] - cx) * z / fx
    y = (vv[valid] - cy) * z / fy
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def save_pcd(path: Path, pts: np.ndarray):
    """保存点云为 binary PCD 文件。"""
    n = len(pts)
    header = (
        f"# .PCD v0.7 - Point Cloud Data file format\n"
        f"VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
        f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\nDATA binary\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(pts.tobytes())


def main():
    parser = argparse.ArgumentParser(description="深度视频 → PCD")
    parser.add_argument("--input", required=True, help="深度视频路径")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--fx", type=float, default=605)
    parser.add_argument("--fy", type=float, default=605)
    parser.add_argument("--cx", type=float, default=323)
    parser.add_argument("--cy", type=float, default=252)
    parser.add_argument("--max-depth", type=float, default=5000)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Decoding {args.input} ...")
    frames = decode_depth_video(args.input)
    n = len(frames)
    print(f"  {n} frames, {frames[0].shape}")

    indices = {"start": 0, "middle": n // 2, "end": n - 1}
    for label, idx in indices.items():
        pts = depth_to_pointcloud(
            frames[idx], args.fx, args.fy, args.cx, args.cy,
            max_depth_mm=args.max_depth, stride=args.stride,
        )
        fname = out / f"{label}_{idx}.pcd"
        save_pcd(fname, pts)
        print(f"  {fname}: {len(pts)} points")


if __name__ == "__main__":
    main()
