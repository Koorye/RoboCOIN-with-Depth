"""
test_da3.py — Depth Anything V3 深度修复测试
"""

from __future__ import annotations

import pytest

from src.depth_repair.da3 import DA3DepthRepair
from tests.conftest import run_and_save


MODELS = {
    "nested":  {"id": "depth-anything/DA3NESTED-GIANT-LARGE-1.1", "chunk": 16, "overlap": 8},
    "large":   {"id": "depth-anything/DA3-LARGE-1.1",            "chunk": 16, "overlap": 8},
    "metric":  {"id": "depth-anything/DA3METRIC-LARGE",           "chunk": 16, "overlap": 0},
    "mono":    {"id": "depth-anything/DA3MONO-LARGE",             "chunk": 16, "overlap": 0},
}

EXTRA = {
    "nested":           {},
    "large":            {},
    "metric":           {},
    "mono":             {},
    "metric_temporal":  {"temporal_alpha": 0.7},
}

TEST_CASES = [
    ("nested", "head"),  ("nested", "right"),
    ("large",  "head"),  ("large",  "right"),
    ("mono",   "head"),  ("mono",   "right"),
    ("metric", "head"),  ("metric", "right"),
    ("metric_temporal", "head"), ("metric_temporal", "right"),
]


def _make_test(model_key, cam):
    cfg = MODELS.get(model_key, MODELS["metric"])
    extra = EXTRA.get(model_key, {})

    @pytest.mark.slow
    def _test(h5_path, head_rgb, head_depth, right_rgb, right_depth, request):
        rgbs = head_rgb if cam == "head" else right_rgb
        depths = head_depth if cam == "head" else right_depth
        repair = DA3DepthRepair(model_id=cfg["id"], process_res=504,
                                chunk_size=cfg["chunk"], overlap=cfg["overlap"], **extra)
        run_and_save(repair, rgbs, depths, model_key, cam,
                     save=not request.config.getoption("--bench-only"))
    return _test


# 动态生成 test 函数
for _model, _cam in TEST_CASES:
    globals()[f"test_{_model}_{_cam}"] = _make_test(_model, _cam)
