import string

import numpy as np


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    """Parse '#RRGGBB', 'RRGGBB', '#RGB' or 'RGB' into an (r, g, b) tuple."""
    v = value.strip().lstrip("#")
    if len(v) == 3:
        v = "".join(c * 2 for c in v)
    if len(v) != 6 or any(c not in string.hexdigits for c in v):
        raise ValueError(f"Invalid hex colour: {value!r} (expected e.g. '#FF5733')")
    return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec.601 luma of an (..., 3) array, returned as float32."""
    rgb = rgb.astype(np.float32)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
