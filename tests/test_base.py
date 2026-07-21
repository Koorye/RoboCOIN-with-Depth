"""
test_base.py — 解码 / 编码 / 对齐辅助函数测试（无需模型，始终运行）
"""

from __future__ import annotations

import numpy as np

from src.depth_repair.base import (
    decode_rgb_frames,
    decode_depth_frames,
    encode_depth_frames,
)
from tests.conftest import get_output_dir, save_depth_video, save_rgb_video, save_pointcloud_video, save_pcd, _depth_to_pointcloud


# ============================================================================
# decode_rgb_frames
# ============================================================================


def test_decode_rgb_frames_head(head_rgb):
    """head 相机 RGB 帧解析正确。"""
    assert isinstance(head_rgb, list)
    assert len(head_rgb) == 100
    for frame in head_rgb:
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        assert frame.shape[2] == 3          # (H, W, 3)
        assert frame.dtype == np.uint8
        assert frame.min() >= 0
        assert frame.max() <= 255


def test_decode_rgb_frames_right(right_rgb):
    """right 相机 RGB 帧解析正确。"""
    assert isinstance(right_rgb, list)
    assert len(right_rgb) == 100
    for frame in right_rgb:
        assert frame.ndim == 3
        assert frame.shape[2] == 3
        assert frame.dtype == np.uint8


def test_rgb_head_right_same_count(head_rgb, right_rgb):
    """head 和 right RGB 帧数一致。"""
    assert len(head_rgb) == len(right_rgb)


# ============================================================================
# decode_depth_frames
# ============================================================================


def test_decode_depth_frames_head(head_depth):
    """head 相机深度帧解析正确。"""
    assert isinstance(head_depth, list)
    assert len(head_depth) == 100
    for frame in head_depth:
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 2                    # (H, W)
        assert frame.dtype == np.float32
        valid = frame[frame > 0]
        if len(valid) > 0:
            assert np.all(np.isfinite(valid))


def test_decode_depth_frames_right(right_depth):
    """right 相机深度帧解析正确。"""
    assert isinstance(right_depth, list)
    assert len(right_depth) == 100
    for frame in right_depth:
        assert frame.ndim == 2
        assert frame.dtype == np.float32


# ============================================================================
# RGB + depth 一致性
# ============================================================================


def test_rgb_depth_match(head_rgb, head_depth):
    """同一视角 RGB 和深度帧数一致。"""
    assert len(head_rgb) == len(head_depth)


def test_rgb_depth_same_shape(head_rgb, head_depth):
    """同一帧 RGB 和 depth 空间尺寸一致。"""
    for r, d in zip(head_rgb, head_depth):
        assert r.shape[:2] == d.shape[:2], f"shape mismatch: rgb={r.shape}, depth={d.shape}"


# ============================================================================
# encode_depth_frames
# ============================================================================


def test_encode_depth_frames_roundtrip(head_depth):
    """深度帧编码后可重新解码，值接近。"""
    import cv2

    encoded = encode_depth_frames(head_depth)
    assert len(encoded) == len(head_depth)

    for orig, enc in zip(head_depth, encoded):
        assert isinstance(enc, np.ndarray)
        assert enc.dtype == np.uint8

        decoded = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED).astype(np.float32)
        assert decoded.shape == orig.shape

        mask = (orig > 0) & np.isfinite(orig)
        if mask.sum() > 100:
            diff = np.abs(orig[mask] - decoded[mask])
            assert diff.max() <= 2, f"encode/decode roundtrip error: max diff={diff.max():.1f}mm"


# ============================================================================
# H5 整体结构
# ============================================================================


def test_h5_has_expected_cameras(h5_path):
    """H5 包含预期的相机视角。"""
    import h5py
    with h5py.File(h5_path, "r") as f:
        rgb_cams = set(f["observations/camera/rgb"].keys())
        depth_cams = set(f["observations/camera/depth"].keys())
    expected = {"head", "right", "left", "chest"}
    assert rgb_cams == expected, f"RGB cameras: {rgb_cams}"
    assert depth_cams == expected, f"depth cameras: {depth_cams}"


def test_all_frames_count(h5_path, total_frames):
    """原始 H5 共 1149 帧。"""
    assert total_frames == 1149, f"expected 1149 frames, got {total_frames}"


# ============================================================================
# 原始深度视频保存
# ============================================================================


def test_save_original_depth_videos(head_depth, right_depth):
    """保存原始深度视频（head + right）作为 baseline 参考。"""
    for camera, frames in [("head", head_depth), ("right", right_depth)]:
        out = save_depth_video(frames, get_output_dir() / f"original_{camera}")
        assert out.exists()
        size_mb = out.stat().st_size / 1e6
        print(f"\n  -> original_{camera}: {out} ({size_mb:.1f} MB)")


def test_save_original_rgb_videos(head_rgb, right_rgb):
    """保存原始 RGB 视频（head + right）作为参考。"""
    for camera, frames in [("head", head_rgb), ("right", right_rgb)]:
        out = save_rgb_video(frames, get_output_dir() / f"original_{camera}_rgb")
        assert out.exists()
        size_mb = out.stat().st_size / 1e6
        print(f"\n  -> original_{camera}_rgb: {out} ({size_mb:.1f} MB)")


def test_save_original_pcd_videos(head_rgb, head_depth, right_rgb, right_depth):
    """保存原始深度点云视频（head + right）。"""
    for cam, rgbs, depths, intr in [
        ("head", head_rgb, head_depth, {"fx": 605, "fy": 605, "cx": 323, "cy": 252, "pitch": 225}),
        ("right", right_rgb, right_depth, {"fx": 434, "fy": 434, "cx": 314, "cy": 239, "pitch": 225}),
    ]:
        out = save_pointcloud_video(depths, rgbs, get_output_dir() / f"original_{cam}_pcd",
                                    fx=intr["fx"], fy=intr["fy"], cx=intr["cx"], cy=intr["cy"],
                                    stride=3, sample=20000, camera_pitch_deg=intr["pitch"])
        print(f"\n  -> {out}")


def test_save_original_pcd(head_rgb, head_depth, right_rgb, right_depth):
    """保存原始深度点云（start/middle/end，去掉空洞）。"""
    for cam, rgbs, depths, intr in [
        ("head", head_rgb, head_depth, {"fx": 605, "fy": 605, "cx": 323, "cy": 252, "pitch": 225}),
        ("right", right_rgb, right_depth, {"fx": 434, "fy": 434, "cx": 314, "cy": 239, "pitch": 225}),
    ]:
        for label, idx in [("start", 0), ("middle", len(depths) // 2), ("end", -1)]:
            d, r = depths[idx], rgbs[idx]
            # 去掉空洞：只保留有效深度像素
            mask = (d > 0) & np.isfinite(d)
            if mask.sum() < 100:
                print(f"  -> skip {cam}_{label}: too few valid points ({mask.sum()})")
                continue
            # stride=1 取所有有效点
            pts, cols = _depth_to_pointcloud(d, r, intr["fx"], intr["fy"], intr["cx"], intr["cy"],
                                             max_depth_mm=5000, stride=3,
                                             camera_pitch_deg=intr["pitch"])
            out = save_pcd(get_output_dir() / f"original_{cam}_{label}", pts, cols)
            print(f"\n  -> {out}")
