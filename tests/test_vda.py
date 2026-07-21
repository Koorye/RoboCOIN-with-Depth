"""
test_vda.py — Video Depth Anything 深度预测测试
"""

from __future__ import annotations

import pytest

from src.depth_repair.vda import VDADepthRepair
from tests.conftest import run_and_save


TEST_CASES = [
    ("vda",        {"metric": False, "invert": True, "fp32": False}),
    ("vda_metric", {"metric": True,  "fp32": False}),
]


def _make_test(prefix, extra):
    @pytest.mark.slow
    def _test_head(h5_path, head_rgb, head_depth, right_rgb, right_depth, request):
        repair = VDADepthRepair(input_size=504, encoder="vitl",
                                checkpoint="checkpoints/video_depth_anything_vitl.pth",
                                **extra)
        run_and_save(repair, head_rgb, head_depth, prefix, "head",
                     save=not request.config.getoption("--bench-only"))

    @pytest.mark.slow
    def _test_right(h5_path, head_rgb, head_depth, right_rgb, right_depth, request):
        repair = VDADepthRepair(input_size=504, encoder="vitl",
                                checkpoint="checkpoints/video_depth_anything_vitl.pth",
                                **extra)
        run_and_save(repair, right_rgb, right_depth, prefix, "right",
                     save=not request.config.getoption("--bench-only"))

    return _test_head, _test_right


for _prefix, _extra in TEST_CASES:
    h, r = _make_test(_prefix, _extra)
    globals()[f"test_{_prefix}_head"] = h
    globals()[f"test_{_prefix}_right"] = r
