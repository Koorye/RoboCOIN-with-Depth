"""
src/depth_repair — 深度修复工具包

用法::

    from src.depth_repair import create_depth_repair

    repair = create_depth_repair("da3")
    repaired = repair.repair_h5("data/raw/xxx.h5", camera="head")

    # 或直接实例化
    from src.depth_repair import VDADepthRepair, LingBotDepthRepair, DA3DepthRepair
"""

from src.depth_repair.base import BaseDepthRepair
from src.depth_repair.vda import VDADepthRepair
from src.depth_repair.lingbot import LingBotDepthRepair
from src.depth_repair.da3 import DA3DepthRepair
from src.depth_repair.da3_rife import DA3RIFEDepthRepair


def create_depth_repair(strategy: str = "da3", **kwargs) -> BaseDepthRepair:
    registry = {
        "da3": DA3DepthRepair,
        "da3_rife": DA3RIFEDepthRepair,
        "vda": VDADepthRepair,
        "lingbot": LingBotDepthRepair,
    }
    if strategy not in registry:
        raise ValueError(
            f"Unknown strategy: {strategy!r}. Available: {list(registry.keys())}"
        )
    return registry[strategy](**kwargs)
