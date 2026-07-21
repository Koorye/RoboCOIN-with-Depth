"""
render_pcd_video.py — 从 PCD 文件重新渲染旋转点云视频

用法:
    python docs/render_pcd_video.py                          # 所有 head middle PCD
    python docs/render_pcd_video.py --camera right            # right 视角
    python docs/render_pcd_video.py --prefix metric           # 只渲染 metric_*
    python docs/render_pcd_video.py --grid                    # 渲染后拼成大视频
"""

from __future__ import annotations

import argparse
import subprocess as sp
import tempfile
from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PCD_DIR = ROOT / "tests" / "output"
OUT_DIR = ROOT / "docs" / "images"


def load_pcd(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """加载 ASCII PCD 文件，返回 (points, colors)。"""
    with open(path) as f:
        lines = f.readlines()

    header_end = 0
    n_points = 0
    has_rgb = False
    for i, line in enumerate(lines):
        if line.startswith("FIELDS"):
            has_rgb = "rgb" in line
        elif line.startswith("POINTS"):
            n_points = int(line.split()[1])
        elif line.startswith("DATA"):
            header_end = i + 1
            break

    pts = np.zeros((n_points, 3), dtype=np.float32)
    cols = np.zeros((n_points, 3), dtype=np.uint8) if has_rgb else None

    for i, line in enumerate(lines[header_end:]):
        parts = line.split()
        if len(parts) < 3:
            continue
        pts[i] = [float(parts[0]), float(parts[1]), float(parts[2])]
        if has_rgb and len(parts) >= 4:
            rgb = int(float(parts[3]))
            cols[i] = [(rgb >> 16) & 0xFF, (rgb >> 8) & 0xFF, rgb & 0xFF]

    return pts, cols


def render_rotating_video(
    pcd_path: Path,
    out_path: Path,
    n_frames: int = 300,
    fps: float = 30,
    elev: float = 25,
    dist: float = 2,
):
    """围绕 Z 轴旋转 360° 渲染点云视频（每种点云独立缩放填满屏幕）。"""
    if out_path.exists():
        print(f"  -> skip (exists): {out_path}")
        return

    pts, cols = load_pcd(pcd_path)
    if len(pts) == 0:
        raise ValueError(f"PCD empty: {pcd_path}")

    # 居中
    centroid = pts.mean(axis=0)
    pts[:, 0] -= centroid[0]
    pts[:, 1] -= centroid[1]
    pts[:, 2] -= centroid[2]

    # 独立归一化填满
    half = max(pts[:, 0].ptp(), pts[:, 1].ptp(), pts[:, 2].ptp()) / 2 * 1.05
    x_lim, y_lim, z_lim = (-half, half), (-half, half), (-half, half)

    if len(pts) > 200000:
        idx = np.random.RandomState(0).choice(len(pts), 200000, replace=False)
        pts, cols = pts[idx], cols[idx] if cols is not None else None

    tmpdir = tempfile.mkdtemp(prefix="pcd_render_")
    try:
        for fi in range(n_frames):
            azim = fi * 120.0 / n_frames  # 1/3 圈，视频循环 3 次即完整一周
            fig = plt.figure(figsize=(8, 8), facecolor="#f0f0f0")
            ax = fig.add_subplot(projection="3d", facecolor="#f0f0f0")
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       c=cols.astype(np.float32) / 255.0 if cols is not None else None,
                       s=1.0, marker=".", alpha=0.8)
            ax.set_xlim(*x_lim); ax.set_ylim(*y_lim); ax.set_zlim(*z_lim)
            ax.set_axis_off()
            ax.grid(False)
            ax.view_init(elev=elev, azim=azim)
            ax.dist = dist
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
            fig.savefig(f"{tmpdir}/{fi:06d}.png", dpi=100,
                        facecolor="#f0f0f0", pad_inches=0)
            plt.close(fig)

        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-r", str(fps), "-i", f"{tmpdir}/%06d.png",
            "-c:v", "libx265", "-preset", "slow", "-crf", "12",
            "-pix_fmt", "yuv420p", str(out_path),
        ]
        sp.run(cmd, check=True)
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"  -> {out_path}")


def concat_grid(videos: list[Path], out_path: Path, cols: int = 3):
    """Python 逐帧拼接网格视频（带标题，中心裁剪 50%）。"""
    if out_path.exists():
        print(f"Grid skip (exists): {out_path}")
        return

    caps = [cv2.VideoCapture(str(v)) for v in videos]
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    W = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 中心裁剪 50% 区域
    crop_W, crop_H = W // 2, H // 2
    cx0, cy0 = (W - crop_W) // 2, (H - crop_H) // 2

    n = len(videos)
    n_rows = (n + cols - 1) // cols
    title_h = 28
    cell_H = crop_H + title_h
    grid_W, grid_H = cols * crop_W, n_rows * cell_H

    placeholder = np.full((crop_H, crop_W, 3), 255, dtype=np.uint8)

    # 提取标题名
    labels = [v.stem.replace("_head_middle_rot360", "").replace("_right_middle_rot360", "")
              for v in videos]

    tmp_avi = out_path.with_suffix(".avi")
    writer = cv2.VideoWriter(tmp_avi, cv2.VideoWriter_fourcc(*"MJPG"), fps, (grid_W, grid_H))

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
            ri, ci = i // cols, i % cols
            y0 = ri * cell_H
            x0 = ci * crop_W
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
    # ffmpeg 转 H.265
    sp.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", tmp_avi, "-c:v", "libx265", "-preset", "slow",
            "-crf", "12", "-pix_fmt", "yuv420p", str(out_path)], check=True)
    tmp_avi.unlink()
    print(f"Grid video: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="PCD → 旋转点云视频")
    parser.add_argument("--camera", default="head", help="head / right")
    parser.add_argument("--prefix", default=None, help="只渲染匹配前缀的 PCD")
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--dist", type=float, default=2, help="相机距离")
    parser.add_argument("--grid", action="store_true", help="渲染后拼成大视频")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pattern = f"*_{args.camera}_middle.pcd"
    if args.prefix:
        pattern = f"{args.prefix}*_{args.camera}_middle.pcd"

    pcd_files = sorted(PCD_DIR.glob(pattern))
    if not pcd_files:
        print(f"No PCD files matching: {pattern}")
        return

    rendered = []
    for pcd_path in pcd_files:
        name = pcd_path.stem
        out_path = OUT_DIR / f"{name}_rot360.mp4"
        print(f"Rendering {pcd_path.name} ...")
        render_rotating_video(pcd_path, out_path, fps=args.fps, dist=args.dist)
        rendered.append(out_path)

    if args.grid and rendered:
        grid_path = OUT_DIR / f"pcd_grid_{args.camera}.mp4"
        concat_grid(rendered, grid_path)

    print("Done.")


if __name__ == "__main__":
    main()
