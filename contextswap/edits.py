"""Manual corrections of the matte (human-in-the-loop).

    {"op": "add" | "remove", "kind": "wand",    "x": 120, "y": 340, "tol": 30}
    {"op": "add" | "remove", "kind": "polygon", "points": [[x, y], [x, y], ...]}
"""
from __future__ import annotations

import cv2
import numpy as np


def wand_mask(rgb, x, y, tol=30):
    """Magic wand: connected region whose colour is within +-tol of the clicked pixel."""
    h, w = rgb.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return np.zeros((h, w), bool)
    img = cv2.GaussianBlur(np.ascontiguousarray(rgb, dtype=np.uint8), (5, 5), 0)
    mask = np.zeros((h + 2, w + 2), np.uint8)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
    cv2.floodFill(img, mask, (int(x), int(y)), (0, 0, 0), (tol,) * 3, (tol,) * 3, flags)
    region = mask[1:-1, 1:-1]
    region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return region > 0


def polygon_mask(shape, points):
    m = np.zeros(shape, np.uint8)
    pts = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
    if len(pts) >= 3:
        cv2.fillPoly(m, [pts], 1)
    return m > 0


def apply_edits(rgb, alpha, edits):
    """Return a new alpha with every edit applied in order. The input is not modified."""
    a = alpha.astype(np.float32).copy()
    for e in edits:
        if e["kind"] == "wand":
            region = wand_mask(rgb, e["x"], e["y"], e.get("tol", 30))
        elif e["kind"] == "polygon":
            region = polygon_mask(a.shape, e["points"])
        else:
            raise ValueError(f"unknown edit kind {e['kind']!r}")
        soft = cv2.GaussianBlur(region.astype(np.float32), (0, 0), 1.0)
        a = np.maximum(a, soft) if e["op"] == "add" else a * (1.0 - soft)
    return np.clip(a, 0.0, 1.0)