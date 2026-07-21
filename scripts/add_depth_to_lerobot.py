#!/usr/bin/env python3
"""
为已有 LeRobot 数据集新增深度数据。

用法:
    python scripts/add_depth_to_lerobot.py \
        --repo-id your_org/dataset_name \
        --cameras head right \
        --strategy da3 --da3-model depth-anything/DA3METRIC-LARGE
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.depth_augmenter import DepthAugmenter


def main():
    parser = argparse.ArgumentParser(description="为 LeRobot 数据集新增深度")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", default="data/lerobot")
    parser.add_argument("--output-repo-id", default=None, help="输出 repo_id")
    parser.add_argument("--output-root", default=None, help="输出 root 目录")
    parser.add_argument("--cameras", nargs="+", default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--crf", type=int, default=6)
    parser.add_argument("--preset", default="slow")
    parser.add_argument("--normalize", action="store_true", default=False,
                        help="per-episode min-max 归一化到 [0,4095]")
    parser.add_argument("--save-pcd", action="store_true", default=False,
                        help="逐帧保存 PCD 点云")

    # repair strategy
    parser.add_argument("--strategy", default="da3", choices=["da3", "vda"])

    # da3
    parser.add_argument("--da3-model", default="depth-anything/DA3METRIC-LARGE")
    parser.add_argument("--da3-process-res", type=int, default=504)
    parser.add_argument("--da3-chunk-size", type=int, default=4)
    parser.add_argument("--da3-overlap", type=int, default=0)
    parser.add_argument("--da3-temporal-alpha", type=float, default=0.0)

    # vda
    parser.add_argument("--vda-encoder", default="vitl")
    parser.add_argument("--vda-checkpoint", default="checkpoints/video_depth_anything_vitl.pth")
    parser.add_argument("--vda-input-size", type=int, default=378)
    parser.add_argument("--vda-metric", action="store_true", default=True)
    parser.add_argument("--vda-invert", action="store_true", default=False)
    parser.add_argument("--vda-fp32", action="store_true", default=False)

    args = parser.parse_args()

    # 构建 repair kwargs
    repair_kwargs = {}
    if args.strategy == "da3":
        repair_kwargs = dict(
            model_id=args.da3_model, process_res=args.da3_process_res,
            chunk_size=args.da3_chunk_size, overlap=args.da3_overlap,
            temporal_alpha=args.da3_temporal_alpha,
        )
    elif args.strategy == "vda":
        repair_kwargs = dict(
            checkpoint=args.vda_checkpoint, encoder=args.vda_encoder,
            input_size=args.vda_input_size, metric=args.vda_metric,
            invert=args.vda_invert, fp32=args.vda_fp32,
        )
    augmenter = DepthAugmenter(
        args.repo_id, root=args.root,
        strategy=args.strategy,
        fps=args.fps, crf=args.crf, preset=args.preset,
        **repair_kwargs,
    )
    augmenter.run(
        output_repo_id=args.output_repo_id,
        output_root=args.output_root,
        normalize=args.normalize,
        save_pcd=args.save_pcd,
    )


if __name__ == "__main__":
    main()
