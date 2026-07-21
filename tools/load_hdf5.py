import h5py
import json
import numpy as np
from pathlib import Path
from typing import Any


_JPEG_SIG = b"\xff\xd8\xff"
_PNG_SIG = b"\x89PNG"


def _check_image_magic(buf: bytes | np.ndarray) -> str | None:
    """Return ``'rgb'``, ``'depth'``, or ``None`` by inspecting leading bytes."""
    if isinstance(buf, np.ndarray):
        if buf.dtype != np.uint8 or len(buf) < 4:
            return None
        head = buf[:4].tobytes()
    elif isinstance(buf, bytes):
        head = buf[:4]
    else:
        return None
    if head.startswith(_JPEG_SIG):
        return "rgb"
    if head.startswith(_PNG_SIG):
        return "depth"
    return None


def _decode_image(buf: np.ndarray) -> np.ndarray:
    """Decode a single encoded image buffer (JPEG or PNG) to a pixel array.

    Returns ``(H, W, 3)`` uint8 for JPEG, ``(H, W)`` uint16 for PNG depth.
    """
    import cv2

    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        return buf
    if img.ndim == 3:
        return img[..., ::-1]  # BGR -> RGB
    return img


def _decode_dataset(value: Any) -> Any:
    """Convert an HDF5 dataset value to an appropriate Python type.

    Image bytes (JPEG / PNG stored as uint8 arrays) are auto-decoded.
    """
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            scalar = value[()]
            if isinstance(scalar, bytes):
                try:
                    s = scalar.decode("utf-8")
                except UnicodeDecodeError:
                    return scalar
                try:
                    return json.loads(s)
                except (json.JSONDecodeError, TypeError):
                    return s
            if isinstance(scalar, np.integer):
                return int(scalar)
            if isinstance(scalar, np.floating):
                return float(scalar)
            if isinstance(scalar, np.bool_):
                return bool(scalar)
            return scalar

        if value.ndim == 1:
            # array of bytes -> decode each to string
            if value.dtype.kind == "O" and len(value) > 0 and isinstance(value[0], bytes):
                return [b.decode("utf-8") if isinstance(b, bytes) else b for b in value]

            # array of encoded images -> decode each
            if len(value) > 0:
                first = value[0]
                if isinstance(first, np.ndarray) and first.dtype == np.uint8:
                    kind = _check_image_magic(first)
                    if kind is not None:
                        return [_decode_image(v) for v in value]

            # other object arrays -> keep as numpy
            if value.dtype.kind == "O":
                return value
            if len(value) <= 100:
                return value.tolist()
            return value

        return value

    if isinstance(value, bytes):
        try:
            s = value.decode("utf-8")
        except UnicodeDecodeError:
            return value
        try:
            return json.loads(s)
        except (json.JSONDecodeError, TypeError):
            return s
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def load_h5(path: str | Path, max_depth: int = 50) -> dict[str, Any]:
    """Read an HDF5 file and return its contents as a nested dict."""
    path = Path(path)

    def _visit(group: h5py.Group, depth: int = 0) -> dict[str, Any]:
        if depth > max_depth:
            raise RecursionError(
                f"Maximum nesting depth ({max_depth}) exceeded at {group.name}"
            )
        result: dict[str, Any] = {}
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Group):
                result[key] = _visit(item, depth + 1)
            else:
                result[key] = _decode_dataset(item[()])
        return result

    with h5py.File(path, "r") as f:
        return _visit(f)


if __name__ == "__main__":
    import sys
    import pprint

    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        candidates = sorted(Path("data/raw").rglob("*.h5"))
        if not candidates:
            print("No .h5 files found under data/raw/. Pass a path as argument.")
            sys.exit(1)
        target = str(candidates[0])

    print(f"Loading (dict): {target}")
    data = load_h5(target)

    def _brief(obj, max_len=120):
        if isinstance(obj, dict):
            return {k: _brief(v) for k, v in obj.items()}
        if isinstance(obj, np.ndarray):
            return f"ndarray(shape={obj.shape}, dtype={obj.dtype})"
        if isinstance(obj, list):
            if len(obj) > 6:
                return f"list(len={len(obj)})[{_brief(obj[0])}, ...]"
            return [_brief(x) for x in obj]
        if isinstance(obj, str) and len(obj) > max_len:
            return obj[:max_len] + "..."
        return obj

    pprint.pprint(_brief(data), width=140, depth=5)

    print(data['observations']['camera']['rgb']['chest']['images'][0].shape)
    print(data['observations']['camera']['depth']['chest']['images'][0].shape)