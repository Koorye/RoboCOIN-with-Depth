"""
test_lingbot.py — LingBot-Depth 深度补全测试
"""

from __future__ import annotations

import pytest

from src.depth_repair.lingbot import LingBotDepthRepair
from tests.conftest import run_and_save


INTRINSICS = {
    "head":  (605, 605, 323, 252),
    "right": (434, 434, 314, 239),
}


def _make_test(cam):
    @pytest.mark.slow
    def _test(h5_path, head_rgb, head_depth, right_rgb, right_depth, request):
        rgbs = head_rgb if cam == "head" else right_rgb
        depths = head_depth if cam == "head" else right_depth
        repair = LingBotDepthRepair(device="cuda", intrinsics=INTRINSICS[cam])
        run_and_save(repair, rgbs, depths, "lingbot", cam,
                     save=not request.config.getoption("--bench-only"))
    return _test


for _cam in ("head", "right"):
    globals()[f"test_lingbot_{_cam}"] = _make_test(_cam)
