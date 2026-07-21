"""
utils/depth_to_pointcloud.py — 深度+RGB视频 → 点云可视化

用法:
    python utils/depth_to_pointcloud.py \
        --depth-video depth.mp4 --rgb-video rgb.mp4 \
        --fx 605 --fy 605 --cx 323 --cy 252 \
        --vis -o scene.ply
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from tqdm import tqdm

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================================
# 视频解码
# ============================================================================

def read_depth_video(path: str, max_frames: int = 0) -> list[np.ndarray]:
    """读取 12-bit HEVC 深度视频，返回 float32 mm 帧列表。"""
    w, h = _video_resolution(path)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", path,
        "-f", "rawvideo", "-pix_fmt", "gray12le",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed:\n{proc.stderr.decode()}")

    raw = np.frombuffer(proc.stdout, dtype=np.uint16)
    frame_pixels = w * h
    n_frames = len(raw) // frame_pixels

    frames: list[np.ndarray] = []
    for i in range(n_frames):
        start = i * frame_pixels
        end = start + frame_pixels
        if end > len(raw):
            break
        frames.append(raw[start:end].reshape(h, w).astype(np.float32))
        if max_frames and len(frames) >= max_frames:
            break
    return frames


def read_rgb_video(path: str, max_frames: int = 0) -> list[np.ndarray]:
    """读取 RGB 视频，返回 (H, W, 3) uint8 帧列表（用 ffmpeg 解码避免 codec 问题）。"""
    w, h = _video_resolution(path)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", path,
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg RGB decode failed:\n{proc.stderr.decode()}")

    raw = np.frombuffer(proc.stdout, dtype=np.uint8)
    frame_bytes = w * h * 3
    n_frames = len(raw) // frame_bytes

    frames: list[np.ndarray] = []
    for i in range(n_frames):
        start = i * frame_bytes
        end = start + frame_bytes
        if end > len(raw):
            break
        frames.append(raw[start:end].reshape(h, w, 3))
        if max_frames and len(frames) >= max_frames:
            break
    return frames


def _video_resolution(path: str) -> tuple[int, int]:
    """用 ffprobe 获取视频宽高。"""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        path,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    w, h = map(int, out.stdout.strip().split(","))
    return w, h


# ============================================================================
# 深度 → 点云
# ============================================================================

def _rotate_x(points: np.ndarray, angle_rad: float) -> np.ndarray:
    """绕 X 轴旋转点云（俯仰补偿：正角=相机朝下转成水平）。"""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
    return points @ R.T


def depth_to_pointcloud(
    depth: np.ndarray,         # (H, W) float32 (mm)
    rgb: np.ndarray | None = None,
    fx: float = 605.4,
    fy: float = 605.2,
    cx: float = 323.3,
    cy: float = 252.0,
    max_depth_mm: float = 5000,
    stride: int = 1,
    camera_pitch_deg: float = 0.0,
    depth_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    H, W = depth.shape
    u = np.arange(0, W, stride)
    v = np.arange(0, H, stride)
    uu, vv = np.meshgrid(u, v)

    d = depth[vv, uu].astype(np.float32) * depth_scale
    valid = (d > 0) & np.isfinite(d) & (d < max_depth_mm * depth_scale)

    z = d[valid] / 1000.0  # mm → m
    x = (uu[valid] - cx) * z / fx
    y = (vv[valid] - cy) * z / fy

    points = np.stack([x, y, z], axis=-1).astype(np.float32)
    if camera_pitch_deg != 0:
        points = _rotate_x(points, np.deg2rad(camera_pitch_deg))
    colors = rgb[vv, uu][valid].astype(np.uint8) if rgb is not None else None
    return points, colors


def frames_to_pointcloud(
    depths: list[np.ndarray],
    rgbs: list[np.ndarray],
    fx: float, fy: float, cx: float, cy: float,
    max_depth_mm: float = 5000,
    stride: int = 2,
    camera_pitch_deg: float = 0.0,
    depth_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    all_pts, all_cols = [], []
    for d, r in zip(depths, rgbs):
        pts, cols = depth_to_pointcloud(d, r, fx, fy, cx, cy, max_depth_mm, stride, camera_pitch_deg, depth_scale)
        all_pts.append(pts)
        if cols is not None:
            all_cols.append(cols)
    points = np.concatenate(all_pts, axis=0)
    colors = np.concatenate(all_cols, axis=0) if all_cols else None
    return points, colors


# ============================================================================
# 输出
# ============================================================================

def save_ply(path: str, points: np.ndarray, colors: np.ndarray | None = None):
    n = len(points)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            p = points[i]
            if colors is not None:
                c = colors[i]
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n")
            else:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    print(f"Saved {n} points to {path}")


def visualize_pointcloud(points: np.ndarray, colors: np.ndarray | None = None):
    """静态点云可视化（合并所有帧）。"""
    _static_plot(points, colors)


def play_pointcloud(
    depths: list[np.ndarray],
    rgbs: list[np.ndarray],
    fx: float, fy: float, cx: float, cy: float,
    max_depth_mm: float = 5000,
    stride: int = 2,
    sample: int = 200000,
    camera_pitch_deg: float = 0.0,
    depth_scale: float = 1.0,
):
    """逐帧播放点云动画（matplotlib 3D，← → 键控制）。"""
    import matplotlib.pyplot as plt

    print(f"Precomputing {len(depths)} frame point clouds...")
    frame_pts: list[np.ndarray] = []
    frame_cols: list[np.ndarray] = []
    for d, r in tqdm(list(zip(depths, rgbs))[:10]):
        pts, cols = depth_to_pointcloud(d, r, fx, fy, cx, cy, max_depth_mm, stride, camera_pitch_deg, depth_scale)
        if len(pts) > sample:
            idx = np.random.choice(len(pts), sample, replace=False)
            pts, cols = pts[idx], cols[idx] if cols is not None else None
        frame_pts.append(pts)
        frame_cols.append(cols if cols is not None else np.full((len(pts), 3), 128, dtype=np.uint8))

    n_frames = len(frame_pts)
    idx = [0]  # mutable counter for callback

    # 等轴尺度：所有轴取相同范围，避免物体拉伸
    all_pts = np.concatenate(frame_pts, axis=0)
    x_mid = (float(all_pts[:, 0].min()) + float(all_pts[:, 0].max())) / 2
    y_mid = (float(all_pts[:, 1].min()) + float(all_pts[:, 1].max())) / 2
    z_mid = (float(all_pts[:, 2].min()) + float(all_pts[:, 2].max())) / 2
    half = max(
        float(all_pts[:, 0].max()) - x_mid,
        float(all_pts[:, 1].max()) - y_mid,
        float(all_pts[:, 2].max()) - z_mid,
    ) * 1.1

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")
    scat = ax.scatter([], [], [], s=0.5, marker=".")
    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)
    ax.set_zlim(z_mid - half, z_mid + half)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")

    def _redraw():
        i = idx[0]
        p, c = frame_pts[i], frame_cols[i].astype(np.float32) / 255.0
        scat._offsets3d = (p[:, 0], p[:, 1], p[:, 2])
        scat.set_color(c)
        ax.set_title(f"Frame {i + 1}/{n_frames}  [{n_frames} total]")
        fig.canvas.draw_idle()

    playing = [False]

    def _on_key(event):
        if event.key == "right":
            idx[0] = (idx[0] + 1) % n_frames
        elif event.key == "left":
            idx[0] = (idx[0] - 1) % n_frames
        elif event.key == " ":
            playing[0] = not playing[0]
        _redraw()

    fig.canvas.mpl_connect("key_press_event", _on_key)

    # 动画循环
    import time
    _redraw()
    print("Controls: ← → 切换帧, Space 播放/暂停, Q 关闭")
    try:
        while plt.fignum_exists(fig.number):
            if playing[0]:
                idx[0] = (idx[0] + 1) % n_frames
                _redraw()
                time.sleep(0.05)
            fig.canvas.flush_events()
            plt.pause(0.03)
    except (KeyboardInterrupt, Exception):
        pass
    finally:
        plt.close(fig)


def _static_plot(points: np.ndarray, colors: np.ndarray | None = None):
    try:
        import matplotlib.pyplot as plt
        n = min(len(points), 200000)
        idx = np.random.choice(len(points), n, replace=False)
        p = points[idx]
        c = colors[idx].astype(np.float32) / 255.0 if colors is not None else None

        # 等轴尺度
        x_mid = (p[:, 0].min() + p[:, 0].max()) / 2
        y_mid = (p[:, 1].min() + p[:, 1].max()) / 2
        z_mid = (p[:, 2].min() + p[:, 2].max()) / 2
        half = max(p[:, 0].max() - x_mid, p[:, 1].max() - y_mid, p[:, 2].max() - z_mid) * 1.1

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(projection="3d")
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=c, s=0.5, marker=".")
        ax.set_xlim(x_mid - half, x_mid + half)
        ax.set_ylim(y_mid - half, y_mid + half)
        ax.set_zlim(z_mid - half, z_mid + half)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
        ax.set_title(f"Point Cloud ({n} points, sampled)")
        plt.show()
    except Exception as e:
        print(f"[WARN] 可视化失败: {e}")
        print("请用 MeshLab / CloudCompare 打开 PLY 文件查看。")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="深度+RGB视频 → 点云")
    parser.add_argument("--depth-video", required=True, help="深度 MP4 视频 (12-bit HEVC)")
    parser.add_argument("--rgb-video", required=True, help="RGB MP4 视频")
    parser.add_argument("--fx", type=float, default=605.4, help="焦距 fx，默认 head RGB")
    parser.add_argument("--fy", type=float, default=605.2, help="焦距 fy")
    parser.add_argument("--cx", type=float, default=323.3, help="主点 cx")
    parser.add_argument("--cy", type=float, default=252.0, help="主点 cy")
    parser.add_argument("-o", "--output", default=None, help="输出 PLY 路径")
    parser.add_argument("--vis", action="store_true", help="逐帧播放点云动画")
    parser.add_argument("--sample", type=int, default=200000, help="每帧采样点数")
    parser.add_argument("--stride", type=int, default=2, help="采样步长 (1=最密, 越大越稀疏)")
    parser.add_argument("--max-depth", type=float, default=5000, help="深度上限 mm")
    parser.add_argument("--max-frames", type=int, default=0, help="最大帧数 (0=全部)")
    parser.add_argument("--camera-pitch", type=float, default=0.0, help="相机俯仰角补偿 (度)，正=相机朝下")
    parser.add_argument("--depth-scale", type=float, default=1.0, help="深度缩放因子 (默认 1.0)")
    args = parser.parse_args()

    print(f"Loading depth: {args.depth_video}")
    depths = read_depth_video(args.depth_video, args.max_frames)
    print(f"  {len(depths)} frames, {depths[0].shape}")

    print(f"Loading RGB: {args.rgb_video}")
    rgbs = read_rgb_video(args.rgb_video, args.max_frames)
    print(f"  {len(rgbs)} frames, {rgbs[0].shape}")

    n = min(len(depths), len(rgbs))
    depths, rgbs = depths[:n], rgbs[:n]
    pitch = args.camera_pitch
    scale = args.depth_scale
    print(f"intrinsics: fx={args.fx:.1f} fy={args.fy:.1f} cx={args.cx:.1f} cy={args.cy:.1f} pitch={pitch}° scale={scale}")

    completer = None

    if args.vis:
        play_pointcloud(
            depths, rgbs, args.fx, args.fy, args.cx, args.cy,
            args.max_depth, args.stride, sample=args.sample,
            camera_pitch_deg=pitch, depth_scale=scale,
        )
    if args.output:
        pts, cols = frames_to_pointcloud(
            depths, rgbs, args.fx, args.fy, args.cx, args.cy,
            args.max_depth, args.stride, camera_pitch_deg=pitch, depth_scale=scale,
        )
        save_ply(args.output, pts, cols)


if __name__ == "__main__":
    main()
