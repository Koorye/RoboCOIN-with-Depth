"""
depth_augmenter_fs_parallel.py — 带预解码和异步后处理的深度增强器

继承 DepthAugmenterFS，三个优化：
1. 预解码：后台线程持续解码后续 RGB 帧到队列，GPU 不等 I/O
2. GPU 推理后立即释放 GPU，smooth+stats+normalize+encode 打包到后台线程
3. ffmpeg 编码异步执行

用法:
    from src.depth_augmenter_fs_parallel import ParallelDepthAugmenterFS

    augmenter = ParallelDepthAugmenterFS(dataset_dir, strategy="da3")
    augmenter.run()
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
from loguru import logger
from tqdm import tqdm

from src.depth_augmenter_fs import DepthAugmenterFS, _decode_video


class ParallelDepthAugmenterFS(DepthAugmenterFS):
    """带预解码和异步后处理的深度增强器。

    继承 DepthAugmenterFS，覆盖 run()。

    2 队列流水线：
    - q1 (prefetch→GPU):  预读取下一视频，满时阻塞，GPU 不等 I/O
    - q3 (GPU→post):      GPU 完成后推结果，满时阻塞，背压限速

    GPU 拿到帧立即推理，不等待上一轮 ffmpeg。
    q3 size=1 保证最多 1 个 ffmpeg 在跑，不会 OOM。
    """

    # ------------------------------------------------------------------
    # 覆盖父类 run()
    # ------------------------------------------------------------------

    def run(self, normalize: bool = False, log_file: str | None = None):
        # ---- 配置 loguru ----
        logger.remove()
        logger.add(
            sys.stdout,
            format=(
                "<green>{time:HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<level>{message}</level>"
            ),
            level="INFO",
            colorize=True,
        )
        if log_file:
            logger.add(
                log_file,
                format=(
                    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                    "{level: <8} | {message}"
                ),
                level="DEBUG",
                rotation="100 MB",
                retention="7 days",
            )
            logger.info(f"Log file: {log_file}")

        videos_dir = self.root / "videos"
        if not videos_dir.is_dir():
            raise FileNotFoundError(str(videos_dir))

        rgb_dirs = sorted(videos_dir.glob("chunk-*/observation.images.*rgb*"))
        if not rgb_dirs:
            raise FileNotFoundError(f"No RGB video dirs under {videos_dir}")

        info_path = self.root / "meta" / "info.json"
        stats_path = self.root / "meta" / "episodes_stats.jsonl"

        with open(info_path) as f:
            info = json.load(f)
        with open(stats_path) as f:
            stats_lines = f.readlines()

        video_params = self._get_video_params(info.get("features", {}))

        # ---- 断点续跑 ----
        completed = self._scan_progress(rgb_dirs)

        for rgb_dir in rgb_dirs:
            depth_dir_name = rgb_dir.name.replace("rgb", "depth")
            depth_dir = rgb_dir.parent / depth_dir_name
            depth_dir.mkdir(parents=True, exist_ok=True)
            depth_key = depth_dir_name
            done_set = completed.get(depth_key, set())

            mp4_files = sorted(rgb_dir.glob("episode_*.mp4"))
            total = len(mp4_files)

            # 过滤已完成 episode
            pending = [p for p in mp4_files
                       if int(p.stem.split("_")[-1]) not in done_set]
            n_done = total - len(pending)

            # 确保 info 有 depth key
            self._inject_depth_to_info(info, depth_key, video_params)

            if not pending:
                logger.info(
                    f"[resume] {depth_key}: {n_done}/{total} done, skip")
                continue

            n = len(pending)
            mp4_files = pending
            logger.info("=" * 60)
            logger.info(
                f"Chunk: {rgb_dir.name}  episodes: {n}/{total}")
            logger.info("=" * 60)

            all_ep_stats = []

            # ================================================================
            # 3 队列流水线
            #
            #   预读取 ──q1──▶ GPU推理 ──q3──▶ 后处理/ffmpeg
            #
            # q1 (size=1): 预读取最多领先 1 个视频，满时阻塞
            # q3 (size=1): GPU 完成后推结果，满时阻塞 → 自然限制 ffmpeg 并发
            #
            # GPU 拿到帧立即推理，不等待上一轮 ffmpeg。
            # 背压来自 q3: 若后处理慢，GPU 推结果时阻塞，自然限速。
            # ================================================================

            # 队列1: prefetch → GPU
            q1_prefetch: queue.Queue = queue.Queue(maxsize=1)

            # 队列3: GPU → 后处理
            q3_depth: queue.Queue = queue.Queue(maxsize=1)

            # ---- Worker 1: 预读取视频 ----
            def _worker_prefetch():
                for mp4_path in mp4_files:
                    fname = mp4_path.name
                    logger.info(f"[prefetch] decoding {fname} ...")
                    t0 = time.time()
                    frames = _decode_video(mp4_path)
                    dt = time.time() - t0
                    if frames:
                        logger.info(
                            f"[prefetch] {fname}: {len(frames)} frames "
                            f"in {dt:.1f}s"
                        )
                    else:
                        logger.warning(
                            f"[prefetch] {fname}: decode FAILED ({dt:.1f}s)"
                        )
                    # size=1: 队列满则阻塞，等 GPU 取走
                    q1_prefetch.put((mp4_path, frames))
                # 哨兵通知下游结束
                q1_prefetch.put(None)

            # ---- Worker 2: GPU 深度推理 ----
            def _worker_gpu():
                while True:
                    item = q1_prefetch.get()
                    if item is None:                # 哨兵：所有视频已处理
                        q3_depth.put(None)          # 传递哨兵给后处理
                        break

                    mp4_path, rgb_frames = item
                    fname = mp4_path.name

                    if not rgb_frames:
                        logger.warning(f"[gpu] {fname}: skip (decode failed)")
                        pbar.update(1)
                        continue

                    logger.info(
                        f"[gpu] inference start: {fname} "
                        f"({len(rgb_frames)} frames)"
                    )
                    t0 = time.time()
                    depth_frames = self._repair.repair_frames(rgb_frames, [])
                    dt = time.time() - t0
                    logger.info(
                        f"[gpu] inference done: {fname} ({dt:.1f}s)"
                    )

                    # size=1: 若后处理未完成则阻塞，自然限速
                    q3_depth.put((mp4_path, depth_frames))

            # ---- Worker 3: 后处理 + ffmpeg 编码 ----
            def _worker_postprocess():
                while True:
                    item = q3_depth.get()
                    if item is None:                # 哨兵：全部完成
                        break

                    mp4_path, depth_frames = item
                    fname = mp4_path.name

                    try:
                        # 时序平滑
                        t0 = time.time()
                        depth_frames = self._smooth_depth(depth_frames)
                        dt_smooth = time.time() - t0

                        # 统计量
                        all_ep_stats.append(
                            self._encoder.compute_stats(depth_frames))

                        # 归一化或钳位
                        if normalize:
                            dmin, dmax = self._encoder.normalize_params(
                                depth_frames, percentile=95)
                            norm = (dmin, dmax)
                        else:
                            norm = None
                            depth_frames = [np.clip(f, 0, 4095)
                                            for f in depth_frames]

                        # ffmpeg 编码
                        depth_path = str(depth_dir / mp4_path.name)
                        t0 = time.time()
                        self._encoder.encode_from_arrays(
                            depth_frames, depth_path, normalize=norm)
                        os.rename(depth_path, depth_path + ".tmp")
                        dt_encode = time.time() - t0

                        logger.info(
                            f"[post] {fname}: smooth={dt_smooth:.1f}s, "
                            f"encode={dt_encode:.1f}s"
                        )
                    except Exception:
                        logger.exception(f"[post] {fname} FAILED")

                    pbar.update(1)

            # ---- 启动流水线 ----
            t_prefetch = threading.Thread(
                target=_worker_prefetch, name="prefetch", daemon=True)
            t_gpu = threading.Thread(
                target=_worker_gpu, name="gpu", daemon=True)
            t_post = threading.Thread(
                target=_worker_postprocess, name="post", daemon=True)

            with tqdm(total=n, desc=f"inference {rgb_dir.name}") as pbar:
                t_prefetch.start()
                t_gpu.start()
                t_post.start()

                t_prefetch.join()
                t_gpu.join()
                t_post.join()

            logger.info("Pipeline done.")

            # ---- 注入 meta ----
            depth_key = depth_dir.name.replace("rgb", "depth")
            self._inject_depth_to_info(info, depth_key, video_params)
            self._inject_depth_to_stats(
                stats_path, stats_lines, depth_key,
                all_ep_stats, mp4_files,
            )
            self._rename_tmp_files(depth_dir)

        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)
        logger.info("Done.")

