from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image


def load_mask(path: str | Path, *, positive_threshold: float = 127) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    arr = np.asarray(Image.open(p).convert("L"))
    return arr > positive_threshold


def ensure_binary(mask: np.ndarray) -> np.ndarray:
    if mask.dtype == bool:
        return mask
    return mask > 0


def _window_stack(mask: np.ndarray, radius: int, *, pad_value: bool) -> np.ndarray:
    if radius < 0:
        raise ValueError("radius must be >= 0")
    mask = ensure_binary(mask)
    if radius == 0:
        return mask[None, ...]
    padded = np.pad(mask, radius, mode="constant", constant_values=pad_value)
    views = []
    h, w = mask.shape
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            views.append(padded[dy:dy + h, dx:dx + w])
    return np.stack(views, axis=0)


def binary_dilation(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    return _window_stack(mask, radius, pad_value=False).any(axis=0)


def binary_erosion(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    return _window_stack(mask, radius, pad_value=False).all(axis=0)


def boundary_band(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    """Return a binary boundary band using dilation minus erosion.

    This avoids adding OpenCV/SciPy as mandatory dependencies and keeps the
    metric implementation auditable. Radius must be fixed before evaluation.
    """
    mask = ensure_binary(mask)
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    dilated = binary_dilation(mask, radius)
    eroded = binary_erosion(mask, radius)
    return np.logical_and(dilated, np.logical_not(eroded))


def connected_components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    """Return component masks for a binary 2D mask."""
    mask = ensure_binary(mask)
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    if connectivity == 4:
        neigh = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neigh = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]
    ys, xs = np.where(mask)
    for y0, x0 in zip(ys, xs):
        if visited[y0, x0]:
            continue
        stack = [(int(y0), int(x0))]
        visited[y0, x0] = True
        coords = []
        while stack:
            y, x = stack.pop()
            coords.append((y, x))
            for dy, dx in neigh:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not visited[yy, xx]:
                    visited[yy, xx] = True
                    stack.append((yy, xx))
        comp = np.zeros_like(mask, dtype=bool)
        if coords:
            yy, xx = zip(*coords)
            comp[np.array(yy), np.array(xx)] = True
            components.append(comp)
    return components
