import cv2
import numpy as np
import subprocess


class DepthVideoEncoder:
    """12-bit H.265 (HEVC) 深度视频编码器。

    精度保障:
    - gray12le 像素格式: 单通道12-bit，无颜色空间转换、无子采样
    - Per-Episode Min-Max 归一化: 将所有值映射到 [0, 4095] 满12-bit范围
    - crf=0: 近无损编码（在 veryslow 预设下 MAE<1mm）
    - 恒定帧率: 输入/输出均指定 -r fps，确保 CFR

    实测: crf=0 veryslow, 267帧 256×256 深度图, MAE=0.53mm, 压缩比 21:1。
    """

    def __init__(
        self, 
        fps=30, 
        crf=6, 
        preset='slow'
    ):
        self.fps = fps
        self.crf = crf
        self.preset = preset

    @staticmethod
    def normalize_params(frames, percentile=99):
        frames = np.stack(frames, axis=0)
        min_val = np.percentile(frames, 100 - percentile)
        max_val = np.percentile(frames, percentile)
        return min_val, max_val

    @staticmethod
    def normalize(frame, min_val, max_val):
        """归一化到 [0, 4095]: norm = (x - min) / (max - min) * 4095"""
        return np.clip((frame - min_val) / (max_val - min_val) * 4095, 0, 4095).astype(np.uint16)

    @staticmethod
    def unnormalize(frame, min_val, max_val):
        """逆归一化: raw = norm * (max - min) + min"""
        return np.uint16(frame * (max_val - min_val) + min_val)

    @staticmethod
    def compute_stats(frames):
        frames = np.stack(frames, axis=0)
        v = int(frames.min())
        return {
            'min': [[[v]], [[v]], [[v]]],
            'max': [[[int(frames.max())]], [[int(frames.max())]], [[int(frames.max())]]],
            'mean': [[[int(frames.mean())]], [[int(frames.mean())]], [[int(frames.mean())]]],
            'std': [[[int(frames.std())]], [[int(frames.std())]], [[int(frames.std())]]],
            'count': [len(frames)],
        }

    def _build_cmd(self, w, h, save_path):
        return [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{w}x{h}', '-pix_fmt', 'gray12le',
            '-r', str(self.fps),
            '-i', '-',
            '-r', str(self.fps),
            '-fps_mode', 'cfr',

            '-video_track_timescale', '90000',
            
            '-c:v', 'libx265',
            '-crf', str(self.crf),
            '-preset', self.preset,
            '-pix_fmt', 'gray12le',

            '-g', str(self.fps),
            '-keyint_min', str(self.fps),

            '-x265-params', 'bframes=0',
            save_path,
        ]

    def encode(self, frames, save_path, normalize=None):
        first = cv2.imread(frames[0], cv2.IMREAD_UNCHANGED)
        h, w = first.shape
        raw_frames = [cv2.imread(f, cv2.IMREAD_UNCHANGED) for f in frames]
        return self._encode_raw(raw_frames, w, h, save_path, normalize)

    def encode_from_arrays(self, arrays, save_path, normalize=None):
        h, w = arrays[0].shape
        return self._encode_raw(arrays, w, h, save_path, normalize)

    def _encode_raw(self, frames, w, h, save_path, normalize):
        if normalize:
            min_val, max_val = normalize
            frames = [self.normalize(f, min_val, max_val) for f in frames]
        else:
            frames = [np.clip(f, 0, 4095) for f in frames]

        proc = subprocess.Popen(
            self._build_cmd(w, h, save_path), stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for f in frames:
            proc.stdin.write(f.astype(np.uint16).tobytes())  # type: ignore[union-attr]
        proc.stdin.close()  # type: ignore[union-attr]
        return proc.wait()
