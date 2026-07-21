#!/usr/bin/env python3
"""
为 LeRobot 数据集（文件系统级别）新增深度数据。

用法:
    python scripts/add_depth_fs.py --dataset-dir data/lerobot/your_dataset \
        --strategy da3 --da3-model depth-anything/DA3METRIC-LARGE
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.depth_augmenter_fs import DepthAugmenterFS


def main():
    parser = argparse.ArgumentParser(description="FS-level LeRobot 深度增强")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--crf", type=int, default=6)
    parser.add_argument("--preset", default="medium", choices=[
        "ultrafast", "superfast", "veryfast", "faster", 
        "fast", "medium", "slow", "slower", "veryslow"
    ])
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--log-file", default=None,
                        help="日志文件路径（默认仅输出到终端）")

    parser.add_argument("--strategy", default="da3",
                        choices=["da3", "da3_rife", "vda"])
    parser.add_argument("--sample-stride", type=int, default=1,
                        help="RIFE 采样步长（da3_rife 专用，默认 1=不采样）")
    parser.add_argument("--rife-home", default=None,
                        help="Practical-RIFE 目录（默认 third_party/Practical-RIFE）")
    
    parser.add_argument("--da3-model", default="depth-anything/DA3METRIC-LARGE")
    parser.add_argument("--da3-process-res", type=int, default=504)
    parser.add_argument("--da3-chunk-size", type=int, default=0)
    parser.add_argument("--da3-overlap", type=int, default=0)
    parser.add_argument("--da3-temporal-alpha", type=float, default=0.0)

    parser.add_argument("--vda-encoder", default="vitl")
    parser.add_argument("--vda-checkpoint", default="checkpoints/video_depth_anything_vitl.pth")
    parser.add_argument("--vda-input-size", type=int, default=518)
    parser.add_argument("--vda-metric", action="store_true", default=True)
    parser.add_argument("--vda-invert", action="store_true", default=False)
    parser.add_argument("--vda-fp32", action="store_true", default=False)

    args = parser.parse_args()

    repair_kwargs = {}
    if args.strategy in ("da3", "da3_rife"):
        repair_kwargs = dict(
            model_id=args.da3_model, process_res=args.da3_process_res,
            chunk_size=args.da3_chunk_size, overlap=args.da3_overlap,
            temporal_alpha=args.da3_temporal_alpha,
        )
        if args.strategy == "da3_rife":
            repair_kwargs["sample_stride"] = args.sample_stride
            if args.rife_home:
                repair_kwargs["rife_home"] = args.rife_home
    elif args.strategy == "vda":
        repair_kwargs = dict(
            checkpoint=args.vda_checkpoint, encoder=args.vda_encoder,
            input_size=args.vda_input_size, metric=args.vda_metric,
            invert=args.vda_invert, fp32=args.vda_fp32,
        )

    augmenter = DepthAugmenterFS(
        args.dataset_dir, strategy=args.strategy,
        fps=args.fps, crf=args.crf, preset=args.preset,
        **repair_kwargs,
    )
    augmenter.run(normalize=args.normalize, log_file=args.log_file)


if __name__ == "__main__":
    main()
