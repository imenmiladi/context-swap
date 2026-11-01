import argparse
import sys
from pathlib import Path

from PIL import Image

from .pipeline import RenderParams, process
from .segment import ENGINES
from .shadow import BG_MODELS
from .utils import hex_to_rgb


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="contextswap",
        description="Put a person/product photo on a solid colour background.",
    )
    p.add_argument("input", help="input image (jpg/png/webp...)")
    p.add_argument("--bg", required=True, help='background colour, hex, e.g. "#FF5733" (quote it in a shell!)')
    p.add_argument("-o", "--output", help="output PNG (default: <input>_<hex>.png)")
    p.add_argument("--engine", choices=list(ENGINES), default="birefnet", help="segmentation engine")
    p.add_argument("--max-side", type=int, default=2048, help="downscale if longest side is larger (0 = no cap)")

    s = p.add_argument_group("shadows")
    s.add_argument("--keep-shadows", action="store_true", help="keep the subject's cast shadow, tinted by the new colour")
    s.add_argument("--shadow-strength", type=float, default=0.8, help="0..1 (default 0.8)")
    s.add_argument("--bg-model", choices=BG_MODELS, default="uniform", help="original background: flat or gradient")
    s.add_argument("--shadow-threshold", type=float, default=0.04, help="ignore darkening below this (default 0.04)")
    s.add_argument("--shadow-smooth", type=float, default=2.0, help="shadow map blur sigma, px (default 2)")
    s.add_argument("--shadow-hue-tolerance", type=float, default=0.25,
                   help="reject non-shadow dark objects whose hue differs from the backdrop; 0 = off (default 0.25)")
    e = p.add_argument_group("edges")
    e.add_argument("--no-decontaminate", action="store_true", help="skip edge colour decontamination")
    e.add_argument("--edge-shrink", type=int, default=0, help="erode the matte by N px to remove halos")
    e.add_argument("--feather", type=float, default=0.0, help="blur the matte, sigma in px")

    p.add_argument("--save-debug", action="store_true", help="also save <output>_alpha.png and _shadow.png")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    hex_to_rgb(args.bg)  # fail fast on a bad colour
    inp = Path(args.input)
    out = Path(args.output) if args.output else inp.with_name(f"{inp.stem}_{args.bg.lstrip('#').upper()}.png")

    params = RenderParams(
        bg_hex=args.bg,
        keep_shadows=args.keep_shadows,
        shadow_strength=args.shadow_strength,
        bg_model=args.bg_model,
        shadow_threshold=args.shadow_threshold,
        shadow_smooth=args.shadow_smooth,
        shadow_hue_tolerance=args.shadow_hue_tolerance,
        decontaminate=not args.no_decontaminate,
        edge_shrink=args.edge_shrink,
        feather=args.feather,
    )
    res = process(inp, params, engine=args.engine, max_side=args.max_side)
    res.image.save(out)
    if args.save_debug:
        Image.fromarray((res.alpha * 255).astype("uint8")).save(out.with_name(out.stem + "_alpha.png"))
        if res.shadow_map is not None:
            Image.fromarray((res.shadow_map * 255).astype("uint8")).save(out.with_name(out.stem + "_shadow.png"))
    timing = ", ".join(f"{k}={v:.1f}s" for k, v in res.timings.items())
    print(f"saved {out}  ({timing})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
