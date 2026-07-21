"""
conftest.py — 共享 fixtures

H5 测试数据: data/raw/AI2_Alphabot_2_arrange_teaset_0/20260318_104215_528.h5
- 1149 帧, 4 个相机视角: head / right / left / chest
- 一个文件 = 一个 episode (episode 0 = 全部帧)
"""

from __future__ import annotations

import subprocess as sp
import sys
from pathlib import Path

import numpy as np
import pytest

# 确保项目根在 sys.path
_PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJ_ROOT))

from src.depth_repair.base import decode_rgb_frames, decode_depth_frames

# ---------------------------------------------------------------------------
# 性能统计
# ---------------------------------------------------------------------------

import os
import time
import torch as _torch

_BENCH_FILE = _PROJ_ROOT / "tests" / "output" / "bench_results.txt"
_BENCH_RESULTS: list[dict] = []


@pytest.fixture(autouse=True)
def _cleanup_gpu():
    """每个 test 结束后释放 GPU 显存。"""
    yield
    if _torch.cuda.is_available():
        import gc
        gc.collect()
        for d in range(_torch.cuda.device_count()):
            with _torch.cuda.device(d):
                _torch.cuda.empty_cache()
                _torch.cuda.reset_peak_memory_stats(d)


def bench(fn, *, warmup: int = 1, name: str = "bench"):
    """运行 fn()，记录显存占用和耗时，追加到 bench_results.txt。"""
    for _ in range(warmup):
        _ = fn()

    if _torch.cuda.is_available():
        for d in range(_torch.cuda.device_count()):
            with _torch.cuda.device(d):
                _torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn()
    if _torch.cuda.is_available():
        for d in range(_torch.cuda.device_count()):
            with _torch.cuda.device(d):
                _torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    mem_torch_peak = 0
    mem_torch_now = 0
    mem_nvsmi = 0
    if _torch.cuda.is_available():
        # 多卡汇总
        mem_torch_peak = sum(_torch.cuda.max_memory_allocated(d)
                             for d in range(_torch.cuda.device_count())) / 1024**2
        mem_torch_now = sum(_torch.cuda.memory_allocated(d)
                            for d in range(_torch.cuda.device_count())) / 1024**2
        # nvidia-smi 真实显存（多卡汇总）
        import subprocess as _sp
        _torch.cuda.synchronize()
        r = _sp.run(["nvidia-smi", "--id=0", "--query-compute-apps=used_memory",
                        "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
        mem_nvsmi = sum(int(x) for x in r.stdout.strip().split("\n") if x.strip())

    entry = {"name": name, "time_s": round(elapsed, 1),
             "mem_peak_mib": round(mem_nvsmi),
             "mem_pytorch_mib": round(mem_torch_peak)}
    _BENCH_RESULTS.append(entry)

    line = (f"{name:40s}  time={elapsed:6.1f}s  "
            f"mem(nvsmi)={mem_nvsmi:6.0f}MiB  mem(torch)={mem_torch_peak:6.0f}MiB")
    print(f"  {line}")

    _BENCH_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_BENCH_FILE, "a") as f:
        f.write(line + "\n")
    return result


# ---------------------------------------------------------------------------
# 共享相机参数
# ---------------------------------------------------------------------------

CAMERAS = {
    "head":  {"fx": 605, "fy": 605, "cx": 323, "cy": 252, "pitch": 225},
    "right": {"fx": 434, "fy": 434, "cx": 314, "cy": 239, "pitch": 45},
}


# ---------------------------------------------------------------------------
# 共享 test helper
# ---------------------------------------------------------------------------


def run_and_save(repair, rgbs, depths, prefix, cam, *, save=True):
    """bench 推理 + 可选保存深度/点云/PCD + 释放显存。"""
    repaired = bench(lambda: repair.repair_frames(rgbs, depths),
                     name=f"{prefix}_{cam}")

    assert len(repaired) == len(rgbs)
    for d in repaired:
        assert d.dtype == np.float32

    if save:
        c = CAMERAS[cam]
        save_depth_video(repaired, get_output_dir() / f"{prefix}_{cam}")
        save_pointcloud_video(repaired, rgbs, get_output_dir() / f"{prefix}_{cam}_pcd",
                              fx=c["fx"], fy=c["fy"], cx=c["cx"], cy=c["cy"],
                              stride=3, sample=20000, camera_pitch_deg=c["pitch"])
        for label, idx in [("start", 0), ("middle", len(repaired) // 2), ("end", -1)]:
            pts, cols = _depth_to_pointcloud(repaired[idx], rgbs[idx],
                                             c["fx"], c["fy"], c["cx"], c["cy"],
                                             max_depth_mm=5000, stride=3,
                                             camera_pitch_deg=c["pitch"])
            save_pcd(get_output_dir() / f"{prefix}_{cam}_{label}", pts, cols)

    import torch, gc
    del repair._model; del repair
    gc.collect(); torch.cuda.empty_cache()
    return repaired


def save_bench_summary():
    """追加汇总到 bench_results.txt。"""
    if not _BENCH_RESULTS:
        return
    _BENCH_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_BENCH_FILE, "a") as f:
        f.write(f"\n{'='*75}\n")
        f.write(f"  Session {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*75}\n")
        f.write(f"{'Strategy':40s} {'Time(s)':>8s} {'VRAM(MiB)':>10s} {'PyTorch(MiB)':>13s}\n")
        f.write("-" * 75 + "\n")
        for e in _BENCH_RESULTS:
            f.write(f"{e['name']:40s} {e['time_s']:8.1f} {e['mem_peak_mib']:10.0f} {e['mem_pytorch_mib']:13.0f}\n")
    print(f"\nBench results appended to {_BENCH_FILE}")


# ---------------------------------------------------------------------------
# 输出目录
# ---------------------------------------------------------------------------

_OUTPUT_DIR = _PROJ_ROOT / "tests" / "output"


def get_output_dir() -> Path:
    """返回测试输出目录（自动创建）。"""
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return _OUTPUT_DIR


# ---------------------------------------------------------------------------
# 视频保存
# ---------------------------------------------------------------------------


def save_depth_video(
    frames: list[np.ndarray],
    path: str | Path,
    fps: float = 30.0,
    max_val: int = 4095,
    crf: int = 9,
    preset: str = "slow",
) -> Path:
    """将深度帧保存为 12-bit 灰度 H.265 MP4。

    深度值 clip(0, max_val) → uint16, gray12le 输入 libx265。

    Parameters
    ----------
    frames : list[np.ndarray]
        (H, W) float32 深度帧 (mm)。
    path : str | Path
        输出路径（自动加 .mp4 后缀）。
    fps : float
        帧率。
    max_val : int
        截断上限（>=max_val 截断为 max_val），不做归一化。
    crf : int
        x265 CRF 质量 (0-51, 默认 28)。
    preset : str
        x265 preset (ultrafast/fast/medium/slow)。

    Returns
    -------
    Path
        输出文件路径。
    """
    out = Path(path).with_suffix(".mp4")
    if not frames:
        raise ValueError("frames 为空")

    H, W = frames[0].shape[:2]
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{W}x{H}", "-pix_fmt", "gray12le",
        "-r", str(fps),
        "-i", "-",
        "-r", str(fps),
        "-fps_mode", "cfr",
        "-video_track_timescale", "90000",
        "-c:v", "libx265",
        "-crf", str(crf),
        "-preset", preset,
        "-pix_fmt", "gray12le",
        "-g", str(int(fps)),
        "-keyint_min", str(int(fps)),
        "-x265-params", "bframes=0",
        str(out),
    ]
    proc = sp.Popen(cmd, stdin=sp.PIPE)
    from tqdm import tqdm as _tqdm
    for d in _tqdm(frames, desc=f"depth → {out.name}", unit="f"):
        clipped = np.clip(d, 0, max_val).astype(np.uint16)
        proc.stdin.write(clipped.tobytes())  # type: ignore[union-attr]
    proc.stdin.close()  # type: ignore[union-attr]
    proc.wait()
    return out


# ---------------------------------------------------------------------------
# RGB 视频
# ---------------------------------------------------------------------------


def save_rgb_video(
    frames: list[np.ndarray],
    path: str | Path,
    fps: float = 30.0,
    crf: int = 23,
    preset: str = "fast",
) -> Path:
    """将 RGB 帧保存为 H.265 MP4。"""
    out = Path(path).with_suffix(".mp4")
    if not frames:
        raise ValueError("frames 为空")

    H, W = frames[0].shape[:2]
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{W}x{H}", "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx265", "-crf", str(crf), "-preset", preset,
        "-pix_fmt", "yuv420p",
        str(out),
    ]
    proc = sp.Popen(cmd, stdin=sp.PIPE)
    from tqdm import tqdm as _tqdm
    for f in _tqdm(frames, desc=f"rgb → {out.name}", unit="f"):
        proc.stdin.write(f.tobytes())  # type: ignore[union-attr]
    proc.stdin.close()  # type: ignore[union-attr]
    proc.wait()
    return out


# ---------------------------------------------------------------------------
# 点云视频
# ---------------------------------------------------------------------------


def save_pointcloud_video(
    depths: list[np.ndarray],
    rgbs: list[np.ndarray],
    path: str | Path,
    fx: float = 605.4, fy: float = 605.2,
    cx: float = 323.3, cy: float = 252.0,
    fps: float = 30.0,
    max_depth_mm: float = 5000,
    sample: int = 20000,
    stride: int = 1,
    camera_pitch_deg: float = 0.0,
) -> Path:
    """将深度+RGB 帧渲染为 matplotlib 3D 点云视频。

    Parameters
    ----------
    depths : list[np.ndarray]
        (H, W) float32 深度帧 (mm)。
    rgbs : list[np.ndarray]
        (H, W, 3) uint8 RGB 帧。
    path : str | Path
        输出路径（自动加 .mp4 后缀）。
    fx, fy, cx, cy : float
        相机内参。
    fps : float
        帧率。
    max_depth_mm : float
        深度上限 mm。
    sample : int
        每帧随机采样点数。
    stride : int
        像素采样步长。

    Returns
    -------
    Path
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    out = Path(path).with_suffix(".mp4")
    if not depths:
        raise ValueError("frames 为空")

    # 预计算全局范围 + 固定采样 mask（防抖动）
    H, W = depths[0].shape
    u = np.arange(0, W, stride)
    v = np.arange(0, H, stride)
    uu, vv = np.meshgrid(u, v)
    # 以第一帧修复后深度为准，固定有效像素 mask（避免 sensor 空洞）
    d0 = depths[0][vv, uu].astype(np.float32)
    mask_fixed = (d0 > 0) & np.isfinite(d0) & (d0 < max_depth_mm)
    if mask_fixed.sum() < sample:
        # sensor 空洞太多，改取所有非零像素（不限制 max_depth）
        mask_fixed = (d0 > 0) & np.isfinite(d0)
    mask_flat = mask_fixed.ravel()
    if mask_flat.sum() > sample:
        keep = np.sort(np.random.RandomState(0).choice(
            mask_flat.sum(), sample, replace=False))
        full_idx = np.where(mask_flat)[0]
        mask_flat[:] = False
        mask_flat[full_idx[keep]] = True
    mask_fixed = mask_flat.reshape(mask_fixed.shape)

    # 基于第一帧确定质心和尺度，后续所有帧统一使用（防抖动）
    from tqdm import tqdm as _tqdm
    p0, _ = _depth_to_pointcloud_fixed(depths[0], rgbs[0], fx, fy, cx, cy, max_depth_mm,
                                        uu, vv, mask_fixed, camera_pitch_deg)
    p0[:, 0] = -p0[:, 0]
    p0[:, 1] = -p0[:, 1]
    centroid = p0.mean(axis=0)
    dists0 = np.linalg.norm(p0 - centroid, axis=1)
    half = np.percentile(dists0, 95) * 1.05
    x_lim = (-half, half)
    y_lim = (-half, half)
    z_lim = (-half, half)

    n_frames = len(depths)

    # 3 段视频：0-120°, 120-240°, 240-360° → concat 成完整一周
    import tempfile, os, shutil
    tmpdir = tempfile.mkdtemp(prefix="pcd_video_")
    seg_paths = []
    from tqdm import tqdm as _tqdm
    try:
        for seg in range(3):
            azim_start = seg * 120.0
            seg_dir = os.path.join(tmpdir, f"seg{seg}")
            os.makedirs(seg_dir)
            for fi, (d, r) in enumerate(_tqdm(list(zip(depths, rgbs)),
                                              desc=f"pcd seg{seg} → {out.name}", unit="f")):
                pts, cols = _depth_to_pointcloud_fixed(d, r, fx, fy, cx, cy, max_depth_mm,
                                                        uu, vv, mask_fixed, camera_pitch_deg)
                pts[:, 0] = -pts[:, 0]; pts[:, 1] = -pts[:, 1]
                pts[:, 0] -= centroid[0]; pts[:, 1] -= centroid[1]; pts[:, 2] -= centroid[2]

                azim = azim_start + fi * 120.0 / n_frames

                fig = plt.figure(figsize=(8, 8), facecolor="#f0f0f0")
                ax = fig.add_subplot(projection="3d", facecolor="#f0f0f0")
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                           c=cols.astype(np.float32) / 255.0 if cols is not None else None,
                           s=1.0, marker=".", alpha=0.8)
                ax.set_xlim(*x_lim); ax.set_ylim(*y_lim); ax.set_zlim(*z_lim)
                ax.set_axis_off(); ax.grid(False)
                ax.view_init(elev=25, azim=azim); ax.dist = 2
                fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
                fig.savefig(os.path.join(seg_dir, f"{fi:06d}.png"), dpi=100,
                            facecolor="#f0f0f0", bbox_inches="tight", pad_inches=0)
                plt.close(fig)

            seg_path = os.path.join(tmpdir, f"seg{seg}.mp4")
            sp.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-r", str(fps), "-i", os.path.join(seg_dir, "%06d.png"),
                    "-c:v", "libx265", "-preset", "slow", "-crf", "12",
                    "-pix_fmt", "yuv420p", seg_path], check=True)
            seg_paths.append(seg_path)
            shutil.rmtree(seg_dir)

        concat_list = os.path.join(tmpdir, "concat.txt")
        with open(concat_list, "w") as f:
            for sp_path in seg_paths:
                f.write(f"file '{sp_path}'\n")
        sp.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", concat_list,
                "-c:v", "libx265", "-preset", "slow", "-crf", "12",
                "-pix_fmt", "yuv420p", str(out)], check=True)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return out


def _depth_to_pointcloud(
    depth: np.ndarray,
    rgb: np.ndarray | None,
    fx: float, fy: float, cx: float, cy: float,
    max_depth_mm: float = 5000,
    stride: int = 1,
    camera_pitch_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """单帧深度 → 点云 (m)，支持相机俯仰补偿。"""
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
    cols = rgb[vv, uu][valid].astype(np.uint8) if rgb is not None else None
    return pts, cols


def _depth_to_pointcloud_fixed(
    depth: np.ndarray,
    rgb: np.ndarray | None,
    fx: float, fy: float, cx: float, cy: float,
    max_depth_mm: float,
    uu: np.ndarray, vv: np.ndarray, mask: np.ndarray,
    camera_pitch_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """使用预计算像素网格 + 固定 mask 生成点云（防抖）。"""
    d = depth[vv, uu].astype(np.float32)
    valid = mask & (d > 0) & np.isfinite(d) & (d < max_depth_mm)
    z = d[valid] / 1000.0
    x = (uu[valid] - cx) * z / fx
    y = (vv[valid] - cy) * z / fy
    pts = np.stack([x, y, z], axis=-1).astype(np.float32)
    if camera_pitch_deg != 0:
        pts = _rotate_x(pts, np.deg2rad(camera_pitch_deg))
    cols = rgb[vv, uu][valid].astype(np.uint8) if rgb is not None else None
    return pts, cols


def _rotate_x(points: np.ndarray, angle_rad: float) -> np.ndarray:
    """绕 X 轴旋转点云。"""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
    return points @ R.T


# ---------------------------------------------------------------------------
# PCD 保存
# ---------------------------------------------------------------------------


def save_pcd(
    path: str | Path,
    points: np.ndarray,
    colors: np.ndarray | None = None,
):
    """保存点云为 PCD 文件（ASCII 格式）。"""
    out = Path(path).with_suffix(".pcd")
    n = len(points)
    has_rgb = colors is not None
    with open(out, "w") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z" + (" rgb" if has_rgb else "") + "\n")
        f.write("SIZE 4 4 4" + (" 4" if has_rgb else "") + "\n")
        f.write("TYPE F F F" + (" F" if has_rgb else "") + "\n")
        f.write("COUNT 1 1 1" + (" 1" if has_rgb else "") + "\n")
        f.write(f"WIDTH {n}\nHEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {n}\n")
        f.write("DATA ascii\n")
        for i in range(n):
            x, y, z = points[i]
            if has_rgb:
                r, g, b = int(colors[i][0]), int(colors[i][1]), int(colors[i][2])
                rgb = (r << 16) | (g << 8) | b
                f.write(f"{x} {y} {z} {rgb}\n")
            else:
                f.write(f"{x} {y} {z}\n")
    return out


# ---------------------------------------------------------------------------
# CLI 选项
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False,
                     help="运行模型推理测试（需 GPU + 模型）")
    parser.addoption("--bench-only", action="store_true", default=False,
                     help="仅测试推理开销，不保存视频和点云")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: 标记为需要 GPU/模型的慢测试")


def pytest_sessionfinish(session, exitstatus):
    save_bench_summary()


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="需要 --run-slow 选项")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_H5 = _PROJ_ROOT / "data" / "raw" / "AI2_Alphabot_2_arrange_teaset_0" / "20260318_104215_528.h5"
_DEFAULT_SAMPLE = 100


def _uniform_sample(frames: list, n: int) -> list:
    """从帧列表中均匀采样 n 帧。"""
    total = len(frames)
    if n >= total:
        return frames
    idx = np.linspace(0, total - 1, n, dtype=int)
    return [frames[i] for i in idx]


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def h5_path() -> Path:
    """测试 H5 文件路径。"""
    assert _H5.exists(), f"测试数据不存在: {_H5}"
    return _H5


@pytest.fixture(scope="session")
def total_frames(h5_path: Path) -> int:
    """返回 H5 总帧数。"""
    import h5py
    with h5py.File(h5_path, "r") as f:
        return len(f["observations/camera/rgb/head/images"])


# --- head camera -----------------------------------------------------------


@pytest.fixture(scope="module")
def head_rgb(h5_path: Path) -> list:
    """Episode 0, head 相机 RGB 帧（均匀采样 200 帧）。"""
    frames = decode_rgb_frames(h5_path, "head")
    return _uniform_sample(frames, _DEFAULT_SAMPLE)


@pytest.fixture(scope="module")
def head_depth(h5_path: Path) -> list:
    """Episode 0, head 相机深度帧 mm（均匀采样 200 帧）。"""
    frames = decode_depth_frames(h5_path, "head")
    return _uniform_sample(frames, _DEFAULT_SAMPLE)


# --- right camera ----------------------------------------------------------


@pytest.fixture(scope="module")
def right_rgb(h5_path: Path) -> list:
    """Episode 0, right 相机 RGB 帧（均匀采样 200 帧）。"""
    frames = decode_rgb_frames(h5_path, "right")
    return _uniform_sample(frames, _DEFAULT_SAMPLE)


@pytest.fixture(scope="module")
def right_depth(h5_path: Path) -> list:
    """Episode 0, right 相机深度帧 mm（均匀采样 200 帧）。"""
    frames = decode_depth_frames(h5_path, "right")
    return _uniform_sample(frames, _DEFAULT_SAMPLE)