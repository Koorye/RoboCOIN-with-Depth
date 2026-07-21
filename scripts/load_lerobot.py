#!/usr/bin/env python3
"""
LeRobot 数据集可视化：RGB + Depth + Point Cloud（方向键切换帧）。

用法:
    python scripts/load_lerobot.py --repo-id test --root data/lerobot/dataset
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _to_hwc(rgb):
    if hasattr(rgb, "numpy"):
        rgb = rgb.numpy()
    if rgb.ndim == 3 and rgb.shape[0] == 3:
        rgb = rgb.transpose(1, 2, 0)
    if rgb.max() <= 1.0:
        rgb = (rgb * 255).astype(np.uint8)
    return np.ascontiguousarray(rgb.astype(np.uint8))


def _depth_to_color(depth):
    import cv2
    d = np.asarray(depth, dtype=np.float32)
    valid = (d > 0) & np.isfinite(d)
    if valid.any():
        vmax = np.percentile(d[valid], 98)
        d = np.clip(d, 0, vmax) / max(vmax, 1)
    d = (d * 255).astype(np.uint8)
    return cv2.applyColorMap(d, cv2.COLORMAP_TURBO)


def _make_pcd_renderer(H=480, W=640):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import io, cv2

    dpi = 100
    fig = Figure(figsize=(W / dpi, H / dpi), facecolor="#f0f0f0", dpi=dpi)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(projection="3d", facecolor="#f0f0f0")
    ax.set_axis_off(); ax.grid(False)
    ax.dist = 2
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    _scat = None

    def render(pts, azim=45, elev=25):
        nonlocal _scat
        if hasattr(pts, "numpy"):
            pts = pts.numpy()
        pts = np.asarray(pts, dtype=np.float32)

        has_rgb = pts.shape[1] >= 6
        if has_rgb:
            xyz = pts[:, :3]; colors = np.clip(pts[:, 3:6], 0, 1)
        else:
            xyz = pts[:, :3]; colors = None

        valid_mask = (xyz[:, 2] > 0) & np.isfinite(xyz[:, 2])
        if valid_mask.sum() < 10:
            return np.full((H, W, 3), 240, dtype=np.uint8)

        valid = xyz[valid_mask]
        valid_colors = colors[valid_mask] if colors is not None else None
        if len(valid) > 20000:
            idx = np.random.RandomState(0).choice(len(valid), 20000, replace=False)
            valid = valid[idx]
            valid_colors = valid_colors[idx] if valid_colors is not None else None

        centroid = valid.mean(axis=0); valid = valid - centroid
        half = max(abs(valid[:, 0]).max(), abs(valid[:, 1]).max(),
                   abs(valid[:, 2]).max()) * 1.05

        ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_zlim(-half, half)
        ax.view_init(elev=elev, azim=azim)
        if _scat is not None:
            _scat.remove()
        _scat = ax.scatter(valid[:, 0], valid[:, 1], valid[:, 2],
                           s=1.0, marker=".", alpha=0.8,
                           c=valid_colors, depthshade=False)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor="#f0f0f0",
                    bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        img = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (W, H))
        else:
            img = np.full((H, W, 3), 240, dtype=np.uint8)
        return img
    return render


def _save_pcd_all(sample, cameras, state, out_dir):
    """保存所有相机的点云为 PCD 文件。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for cam in cameras:
        pcd_key = f"observation.pointcloud.{cam}_pcd"
        if pcd_key not in sample:
            continue
        pts = sample[pcd_key]
        if hasattr(pts, "numpy"):
            pts = pts.numpy()
        pts = np.asarray(pts, dtype=np.float32)
        valid = (pts[:, 2] > 0) & np.isfinite(pts[:, 2])
        pts = pts[valid]

        fname = out / f"ep{state['ep']:04d}_frame{state['fi']:06d}_{cam}.pcd"
        _write_pcd(fname, pts[:, :3],
                   pts[:, 3:6] if pts.shape[1] >= 6 else None)
        print(f"  -> {fname}")


def _write_pcd(path: Path, xyz: np.ndarray, rgb: np.ndarray | None = None):
    """写入 PCD 文件（binary format, xyz + optional rgb）。"""
    n = len(xyz)
    has_rgb = rgb is not None and rgb.shape[0] == n
    fields = "x y z" + (" rgb" if has_rgb else "")
    sizes = "4 4 4" + (" 4" if has_rgb else "")
    types = "F F F" + (" F" if has_rgb else "")
    counts = "1 1 1" + (" 1" if has_rgb else "")

    header = (
        f"# .PCD v0.7 - Point Cloud Data file format\n"
        f"VERSION 0.7\n"
        f"FIELDS {fields}\n"
        f"SIZE {sizes}\n"
        f"TYPE {types}\n"
        f"COUNT {counts}\n"
        f"WIDTH {n}\n"
        f"HEIGHT 1\n"
        f"VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        f"DATA binary\n"
    )

    if has_rgb:
        rgb_u8 = np.clip(rgb * 255, 0, 255).astype(np.uint32)
        rgb_packed = ((rgb_u8[:, 0].astype(np.uint32) << 16) |
                      (rgb_u8[:, 1].astype(np.uint32) << 8) |
                      rgb_u8[:, 2].astype(np.uint32))
        data = np.column_stack([xyz, rgb_packed.view(np.float32).reshape(-1, 1)])
    else:
        data = xyz

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.astype(np.float32).tobytes())


def _cam_from_key(key: str) -> str:
    return key.rsplit(".", 1)[-1].replace("_rgb", "")


def main():
    parser = argparse.ArgumentParser(description="LeRobot 数据集可视化")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", default="data/lerobot")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--camera", default="all")
    parser.add_argument("--native", action="store_true",
                        help="使用原生 LeRobotDataset（无 PCD）")
    parser.add_argument("--save-pcd", default=None,
                        help="PCD 保存目录（按 's' 键保存当前帧）")
    args = parser.parse_args()

    if args.native:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        ds = LeRobotDataset(args.repo_id, root=args.root)
        has_pcd = False
        print("Using native LeRobotDataset (no PCD)")
    else:
        from src.lerobot_with_depth_dataset import LeRobotWithDepthDataset as LeRobotDataset
        ds = LeRobotDataset(args.repo_id, root=args.root, enable_pcd=True, pcd_with_rgb=True)
        has_pcd = True

    print(f"Episodes: {ds.num_episodes}  Frames: {ds.num_frames}")
    print("Keys: ← → ±1 frame, Shift ±10, Ctrl ±100, ↑ ↓ episode")
    if has_pcd:
        print("PCD: w/s elev, a/d azim, p save PCD")

    rgb_keys = sorted(k for k in ds.features if "rgb" in k and "images" in k)
    cameras = [_cam_from_key(k) for k in rgb_keys]
    if args.camera != "all":
        cameras = [c for c in cameras if c == args.camera]

    ep_from = int(ds.episode_data_index["from"][args.episode])
    ep_to = int(ds.episode_data_index["to"][args.episode])
    state = {"ep": args.episode, "fi": min(ep_from + args.frame, ep_to - 1)}

    import matplotlib.pyplot as plt

    n_rows = len(cameras)
    n_cols = 3 if has_pcd else 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_rows == 1:
        axes = [axes]

    imgs = {}
    for ri, cam in enumerate(cameras):
        imgs[(ri, 0)] = axes[ri][0].imshow(np.zeros((480, 640, 3), dtype=np.uint8))
        axes[ri][0].set_title(f"{cam} RGB", fontsize=10)
        axes[ri][0].set_axis_off()
        imgs[(ri, 1)] = axes[ri][1].imshow(np.zeros((480, 640, 3), dtype=np.uint8))
        axes[ri][1].set_title(f"{cam} Depth", fontsize=10)
        axes[ri][1].set_axis_off()
        if has_pcd:
            imgs[(ri, 2)] = axes[ri][2].imshow(np.zeros((480, 640, 3), dtype=np.uint8))
            axes[ri][2].set_title(f"{cam} PCD", fontsize=10)
            axes[ri][2].set_axis_off()

    title = plt.suptitle("", fontsize=14)
    render_pcd = _make_pcd_renderer() if has_pcd else None
    pcd_view = {"azim": 45, "elev": 25}

    def _redraw():
        ep = state["ep"]
        ep0 = int(ds.episode_data_index["from"][ep])
        ep1 = int(ds.episode_data_index["to"][ep])
        fi = state["fi"]
        sample = ds[fi]
        state["_sample"] = sample

        for ri, cam in enumerate(cameras):
            rgb_key = f"observation.images.{cam}_rgb"
            depth_key = f"observation.images.{cam}_depth"
            pcd_key = f"observation.pointcloud.{cam}_pcd"

            if rgb_key in sample:
                imgs[(ri, 0)].set_data(_to_hwc(sample[rgb_key]))
            if depth_key in sample:
                depth = sample[depth_key]
                if hasattr(depth, "numpy"):
                    depth = depth.numpy()
                if depth.ndim == 3 and depth.shape[0] in (1, 3, 4):
                    depth = depth[0]
                print(depth)
                imgs[(ri, 1)].set_data(_depth_to_color(depth))
            if has_pcd and pcd_key in sample:
                imgs[(ri, 2)].set_data(render_pcd(sample[pcd_key],
                    azim=pcd_view["azim"], elev=pcd_view["elev"]))

        title.set_text(f"Episode {ep}  Frame {fi - ep0}/{ep1 - ep0 - 1}  (global {fi})")
        fig.canvas.draw_idle()

    def _on_key(event):
        ep = state["ep"]
        ep0 = int(ds.episode_data_index["from"][ep])
        ep1 = int(ds.episode_data_index["to"][ep])

        key = (event.key or "").lower()
        if "ctrl" in key:       step = 100
        elif "shift" in key:    step = 10
        else:                   step = 1

        if "right" in key:
            state["fi"] = min(state["fi"] + step, ep1 - 1)
        elif "left" in key:
            state["fi"] = max(state["fi"] - step, ep0)
        elif key == "up":
            if ep < ds.num_episodes - 1:
                state["ep"] += 1
                state["fi"] = int(ds.episode_data_index["from"][state["ep"]])
        elif event.key == "down":
            if ep > 0:
                state["ep"] -= 1
                state["fi"] = int(ds.episode_data_index["from"][state["ep"]])
        elif has_pcd and args.save_pcd and key == "p":
            _save_pcd_all(state["_sample"], cameras, state, args.save_pcd)
            print(f"  [saved] ep{state['ep']} frame{state['fi']}")
        elif has_pcd and key == "a":
            pcd_view["azim"] = (pcd_view["azim"] + 30) % 360
        elif has_pcd and key == "d":
            pcd_view["azim"] = (pcd_view["azim"] - 30) % 360
        elif has_pcd and key == "w":
            pcd_view["elev"] += 30
        elif has_pcd and key == "s":
            pcd_view["elev"] -= 30
        elif event.key in ("q", "escape"):
            plt.close(fig)
            return
        _redraw()

    fig.canvas.mpl_connect("key_press_event", _on_key)
    _redraw()
    plt.show()


if __name__ == "__main__":
    main()
