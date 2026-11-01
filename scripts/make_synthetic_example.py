"""Regenerate examples/synthetic_*.png (synthetic scene with a known matte, no model needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from contextswap.pipeline import RenderParams, render
from tests.synthetic import make_scene

out = Path(__file__).resolve().parents[1] / "examples"
rgb, alpha, _ = make_scene()
img = Image.fromarray(rgb)
img.save(out / "synthetic_input.png")
for name, keep in [("no_shadow", False), ("with_shadow", True)]:
    render(img, alpha, RenderParams(bg_hex="#2E86DE", keep_shadows=keep)).image.save(out / f"synthetic_{name}.png")
print("written to", out)
