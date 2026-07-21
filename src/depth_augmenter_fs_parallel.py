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

    继承 DepthAugmenterFS，覆盖 run()：
    - 后台线程持续解码，GPU 推理从队列取帧，不等 I/O
    - 推理完成后，smooth + stats + normalize + encode 打包到后台线程
    - 主线程立即处理下一个 episode
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
            post_threads: list[threading.Thread] = []

            # 预解码队列
            rgb_queue: queue.Queue = queue.Queue(maxsize=2)

            def _prefetch_worker():
                for idx, mp4_path in enumerate(mp4_files):
                    fname = mp4_path.name
                    logger.debug(f"[prefetch] decoding {fname} ...")
                    t0 = time.time()
                    frames = _decode_video(mp4_path)
                    dt = time.time() - t0
                    if frames:
                        logger.info(
                            f"[prefetch] {fname} decoded: "
                            f"{len(frames)} frames in {dt:.1f}s"
                        )
                    else:
                        logger.warning(
                            f"[prefetch] {fname} decode FAILED ({dt:.1f}s)"
                        )
                    rgb_queue.put((idx, mp4_path, frames))
                logger.debug("[prefetch] all done, sentinel out")
                rgb_queue.put(None)

            prefetch_thread = threading.Thread(
                target=_prefetch_worker, daemon=True)
            prefetch_thread.start()

            # ---- 主线程：取帧 → GPU 推理 → 启动后处理线程 ----
            for _ in tqdm(range(n), desc="inference"):
                t0 = time.time()
                item = rgb_queue.get()
                wait = time.time() - t0
                if item is None:
                    logger.info("[main] got sentinel, loop ends")
                    break
                idx, mp4_path, rgb_frames = item
                fname = mp4_path.name

                if wait > 0.1:
                    logger.warning(
                        f"[main] waited {wait:.1f}s for {fname}"
                    )
                else:
                    logger.debug(
                        f"[main] got {fname} immediately (wait={wait:.3f}s)"
                    )

                if not rgb_frames:
                    logger.warning(f"[main] {fname}: empty frames, skip")
                    continue

                # GPU 深度推理（无 temporal，快速释放 GPU）
                logger.info(
                    f"[main] GPU inference start: {fname} "
                    f"({len(rgb_frames)} frames)"
                )
                t0 = time.time()
                depth_frames = self._repair.repair_frames(rgb_frames, [])
                print(depth_frames[0])
                dt_infer = time.time() - t0
                logger.info(
                    f"[main] GPU inference done:  {fname} "
                    f"({dt_infer:.1f}s), handing off to postprocess"
                )

                # 后处理（smooth + stats + normalize + encode）→ 后台线程
                depth_path = str(depth_dir / mp4_path.name)
                t = threading.Thread(
                    target=self._postprocess_and_encode,
                    args=(depth_frames, depth_path, normalize,
                          fname, all_ep_stats),
                    daemon=True,
                )
                t.start()
                post_threads.append(t)
                alive = sum(1 for t in post_threads if t.is_alive())
                logger.info(
                    f"[main] postprocess launched: {fname} "
                    f"(active: {alive})"
                )

            prefetch_thread.join()

            # ---- 等待所有后处理完成 ----
            n_post = len(post_threads)
            logger.info(f"Waiting for {n_post} postprocess threads ...")
            for t in tqdm(post_threads, desc="postprocess"):
                t.join()
            logger.info("All postprocess done.")

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

    # ------------------------------------------------------------------
    # 后台后处理线程（smooth → stats → normalize → encode）
    # ------------------------------------------------------------------

    def _postprocess_and_encode(
        self,
        depth_frames: list[np.ndarray],
        depth_path: str,
        normalize: bool,
        fname: str,
        all_ep_stats: list,
    ) -> None:
        """后台线程：smooth → stats → normalize/clip → encode → rename。

        在线程间共享的 ``all_ep_stats`` 通过 append 操作修改，
        Python list.append 是线程安全的。
        """
        try:
            # 1. 时序平滑（CPU）
            t0 = time.time()
            depth_frames = self._smooth_depth(depth_frames)
            dt_smooth = time.time() - t0
            logger.info(
                f"[post] smooth done: {fname} ({dt_smooth:.1f}s)"
            )

            # 2. 统计量
            all_ep_stats.append(
                self._encoder.compute_stats(depth_frames))

            # 3. 归一化或钳位
            if normalize:
                dmin, dmax = self._encoder.normalize_params(
                    depth_frames, percentile=95)
                norm = (dmin, dmax)
            else:
                norm = None
                depth_frames = [np.clip(f, 0, 4095)
                                for f in depth_frames]

            # 4. 编码 + 重命名
            t0 = time.time()
            self._encoder.encode_from_arrays(
                depth_frames, depth_path, normalize=norm)
            os.rename(depth_path, depth_path + ".tmp")
            dt_encode = time.time() - t0
            logger.info(
                f"[post] encode done: {fname} ({dt_encode:.1f}s)"
            )

        except Exception as e:
            logger.error(f"[post] {fname} FAILED: {e}")
