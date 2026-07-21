"""
plot_mean_curves.py — 绘制各策略帧均值变化曲线

用法:
    python docs/plot_mean_curves.py

从 tests/output/*.mp4 读取深度视频，计算每帧有效像素均值
（忽略 0 → 取 ≤p95 inlier → mean），绘制对比曲线。
"""

from __future__ import annotations

import subprocess as sp
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "images"
OUT.mkdir(parents=True, exist_ok=True)


def load_frame_means(video_path: str | Path, n_frames: int = 100) -> np.ndarray:
    """从 12-bit 深度视频提取逐帧有效均值。"""
    probe = sp.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0",
         str(video_path)],
        capture_output=True, text=True,
    )
    w, h = map(int, probe.stdout.strip().split(","))

    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-i", str(video_path), "-f", "rawvideo", "-pix_fmt", "gray12le", "-"]
    proc = sp.run(cmd, capture_output=True)
    raw = np.frombuffer(proc.stdout, dtype=np.uint16)
    frame_pixels = w * h
    n = len(raw) // frame_pixels

    means = []
    for i in range(min(n, n_frames)):
        frame = raw[i * frame_pixels:(i + 1) * frame_pixels].reshape(h, w).astype(np.float32)
        valid = frame[frame > 0]                          # 忽略 0
        if len(valid) == 0:
            means.append(np.nan)
            continue
        p95 = np.percentile(valid, 95)                    # 截断异常大值
        inlier = valid[valid <= p95]
        means.append(inlier.mean())
    return np.array(means)


STRATEGIES_HEAD = [
    ("tests/output/original_head.mp4", "Original Sensor"),
    ("tests/output/mono_head.mp4", "DA3 MONO"),
    ("tests/output/metric_head.mp4", "DA3 METRIC"),
    ("tests/output/metric_temporal_head.mp4", "DA3 METRIC+Temporal"),
    ("tests/output/large_head.mp4", "DA3 LARGE-1.1"),
    ("tests/output/nested_head.mp4", "DA3NESTED GIANT-LARGE-1.1"),
    ("tests/output/vda_head.mp4", "VDA (relative)"),
    ("tests/output/vda_metric_head.mp4", "VDA Metric"),
    ("tests/output/lingbot_head.mp4", "LingBot"),
]

STRATEGIES_RIGHT = [
    ("tests/output/original_right.mp4", "Original Sensor"),
    ("tests/output/mono_right.mp4", "DA3 MONO"),
    ("tests/output/metric_right.mp4", "DA3 METRIC"),
    ("tests/output/metric_temporal_right.mp4", "DA3 METRIC+Temporal"),
    ("tests/output/large_right.mp4", "DA3 LARGE-1.1"),
    ("tests/output/nested_right.mp4", "DA3NESTED GIANT-LARGE-1.1"),
    ("tests/output/vda_right.mp4", "VDA (relative)"),
    ("tests/output/vda_metric_right.mp4", "VDA Metric"),
    ("tests/output/lingbot_right.mp4", "LingBot"),
]


def plot(strategies, title, outpath):
    fig, ax = plt.subplots(figsize=(16, 6))
    for path, label in strategies:
        means = load_frame_means(Path(__file__).resolve().parents[1] / path)
        if path == "tests/output/original_right.mp4" or path == "tests/output/lingbot_right.mp4":
            means = means / 5
        ax.plot(means, label=label, alpha=0.8, linewidth=0.8)
    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel("Mean Depth (mm)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(outpath), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath}")


if __name__ == "__main__":
    plot(STRATEGIES_HEAD, "Per-Frame Mean Depth (p95 inlier) — Head Camera",
         OUT / "mean_curve_head.png")
    plot(STRATEGIES_RIGHT, "Per-Frame Mean Depth (p95 inlier) — Right Camera",
         OUT / "mean_curve_right.png")
    print("Done.")
