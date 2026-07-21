"""
test_da3_rife.py — DA3 + RIFE 插帧深度修复测试
"""

from __future__ import annotations

import numpy as np
import pytest

from src.depth_repair.da3_rife import DA3RIFEDepthRepair
from tests.conftest import run_and_save


@pytest.mark.slow
def test_rife_stride2_head(head_rgb, head_depth, request):
    """stride=2: 每 2 帧取 1 帧推理，RIFE 插值补 1 帧。"""
    repair = DA3RIFEDepthRepair(
        model_id="depth-anything/DA3METRIC-LARGE",
        process_res=504, chunk_size=16, overlap=0,
        sample_stride=2,
    )
    repaired = run_and_save(repair, head_rgb, head_depth,
                            "rife_s2", "head",
                            save=not request.config.getoption("--bench-only"))
    assert len(repaired) == len(head_rgb)
    for d in repaired:
        assert d.dtype == np.float32


@pytest.mark.slow
def test_rife_stride3_head(head_rgb, head_depth, request):
    """stride=3: 每 3 帧取 1 帧推理，RIFE 插值补 2 帧。"""
    repair = DA3RIFEDepthRepair(
        model_id="depth-anything/DA3METRIC-LARGE",
        process_res=504, chunk_size=16, overlap=0,
        sample_stride=3,
    )
    repaired = run_and_save(repair, head_rgb, head_depth,
                            "rife_s3", "head",
                            save=not request.config.getoption("--bench-only"))
    assert len(repaired) == len(head_rgb)


@pytest.mark.slow
def test_rife_stride5_head(head_rgb, head_depth, request):
    """stride=5: 每 5 帧取 1 帧推理，RIFE 插值补 4 帧。"""
    repair = DA3RIFEDepthRepair(
        model_id="depth-anything/DA3METRIC-LARGE",
        process_res=504, chunk_size=16, overlap=0,
        sample_stride=5,
    )
    repaired = run_and_save(repair, head_rgb, head_depth,
                            "rife_s5", "head",
                            save=not request.config.getoption("--bench-only"))
    assert len(repaired) == len(head_rgb)


@pytest.mark.slow
def test_rife_stride2_right(right_rgb, right_depth, request):
    """stride=2, right 相机。"""
    repair = DA3RIFEDepthRepair(
        model_id="depth-anything/DA3METRIC-LARGE",
        process_res=504, chunk_size=16, overlap=0,
        sample_stride=2,
    )
    repaired = run_and_save(repair, right_rgb, right_depth,
                            "rife_s2", "right",
                            save=not request.config.getoption("--bench-only"))
    assert len(repaired) == len(right_rgb)


@pytest.mark.slow
def test_rife_stride3_right(right_rgb, right_depth, request):
    """stride=3, right 相机。"""
    repair = DA3RIFEDepthRepair(
        model_id="depth-anything/DA3METRIC-LARGE",
        process_res=504, chunk_size=16, overlap=0,
        sample_stride=3,
    )
    repaired = run_and_save(repair, right_rgb, right_depth,
                            "rife_s3", "right",
                            save=not request.config.getoption("--bench-only"))
    assert len(repaired) == len(right_rgb)


@pytest.mark.slow
def test_rife_stride5_right(right_rgb, right_depth, request):
    """stride=5, right 相机。"""
    repair = DA3RIFEDepthRepair(
        model_id="depth-anything/DA3METRIC-LARGE",
        process_res=504, chunk_size=16, overlap=0,
        sample_stride=5,
    )
    repaired = run_and_save(repair, right_rgb, right_depth,
                            "rife_s5", "right",
                            save=not request.config.getoption("--bench-only"))
    assert len(repaired) == len(right_rgb)