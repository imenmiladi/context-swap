"""Streamlit UI:  streamlit run app.py

Heavy stages are cached separately, so changing the background colour is instant,
and tweaking shadow settings does not re-run the segmentation model.
"""
import hashlib
import io
import time

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from contextswap.edits import apply_edits
from contextswap.pipeline import composite, estimate_foreground, load_image, refine_alpha
from contextswap.segment import ENGINES, compute_alpha
from contextswap.shadow import BG_MODELS, extract_shadow_map

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:  # the app still works, only the manual selection tool is disabled
    streamlit_image_coordinates = None

st.set_page_config(page_title="Context swap", page_icon="🎨", layout="wide")


# ----------------------------------------------------------------- cached stages
@st.cache_data(show_spinner=False, max_entries=8)
def cached_alpha(image_bytes: bytes, engine: str, max_side: int) -> np.ndarray:
    return compute_alpha(load_image(image_bytes, max_side), engine)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_foreground(rgb: np.ndarray, alpha: np.ndarray, decontaminate: bool) -> np.ndarray:
    return estimate_foreground(rgb, alpha, decontaminate)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_shadow(rgb, alpha, bg_model, threshold, smooth, hue_tol) -> np.ndarray:
    return extract_shadow_map(rgb, alpha, bg_model, threshold, smooth, hue_tol)[0]


def make_overlay(rgb: np.ndarray, alpha: np.ndarray, points: list) -> Image.Image:
    """Original image with the current subject tinted green (+ pending polygon points)."""
    a = alpha[..., None]
    tint = np.array([0, 220, 120], np.float32)
    img = Image.fromarray((rgb * (1 - 0.40 * a) + tint * 0.40 * a).astype(np.uint8))
    if points:
        d = ImageDraw.Draw(img)
        r = max(3, img.width // 150)
        pts = [tuple(p) for p in points]
        if len(pts) > 1:
            d.line(pts, fill=(255, 255, 0), width=max(1, r // 2))
        for x, y in pts:
            d.ellipse([x - r, y - r, x + r, y + r], outline=(255, 0, 0), width=2)
    return img


# ------------------------------------------------------------------- callbacks
# Callbacks run BEFORE the script reruns, so no st.rerun() and no component re-creation are
# needed: the page layout stays identical and the browser keeps the scroll position.
def on_image_click():
    ss = st.session_state
    click, size = ss.get("img_click"), ss.get("img_size")
    if not click or not size:
        return
    x = int(click["x"] * size[0] / click["width"])   # displayed px -> image px
    y = int(click["y"] * size[1] / click["height"])
    op = "add" if ss.action.startswith("Add") else "remove"
    if ss.tool == "Magic wand":
        ss.edits.append({"op": op, "kind": "wand", "x": x, "y": y, "tol": ss.wand_tol})
    elif ss.tool == "Polygon":
        ss.poly_points.append([x, y])


def apply_polygon():
    ss = st.session_state
    if len(ss.poly_points) >= 3:
        op = "add" if ss.action.startswith("Add") else "remove"
        ss.edits.append({"op": op, "kind": "polygon", "points": ss.poly_points})
        ss.poly_points = []


def clear_points():
    st.session_state.poly_points = []


# ---------------------------------------------------------------------- sidebar
st.title("🎨 Context swap")
st.caption("Put a person or product on any solid colour, optionally keeping the cast shadow.")

with st.sidebar:
    st.header("Settings")
    uploaded = st.file_uploader("Image", type=["jpg", "jpeg", "png", "webp"])
    bg_hex = st.color_picker("Background colour", "#FF5733")

    st.subheader("Segmentation")
    engine = st.selectbox(
        "Engine", list(ENGINES), index=0,
        help="BiRefNet = best quality (downloads ~900 MB the first time). "
             "Lite = faster on CPU. GrabCut = no model, rough, for quick tests.",
    )
    max_side = st.select_slider("Max image side (px)", [512, 768, 1024, 1536, 2048, 3072, 4096], value=1536,
                                help="Larger = sharper output but slower edge decontamination.")

    st.subheader("Manual selection")
    tool = st.radio("Tool", ["Off", "Magic wand", "Polygon"], horizontal=True, key="tool",
                    disabled=streamlit_image_coordinates is None,
                    help="Fix the cut-out by hand: click on the image on the left.")
    if streamlit_image_coordinates is None:
        st.warning("Click component missing: run  python -m pip install streamlit-image-coordinates  "
                   "then restart the app.")
    action = st.radio("Action", ["Add to subject", "Remove from subject"], horizontal=True, key="action",
                      disabled=tool == "Off")
    wand_tol = st.slider("Wand tolerance", 5, 80, 30, key="wand_tol", disabled=tool != "Magic wand",
                         help="How different in colour a neighbouring pixel may be and still be selected.")
    b1, b2 = st.columns(2)
    undo = b1.button("↩ Undo", disabled=tool == "Off")
    reset = b2.button("Reset edits", disabled=tool == "Off")

    st.subheader("Shadows")
    keep_shadows = st.checkbox("Keep subject shadows", value=True)
    shadow_strength = st.slider("Shadow strength", 0.0, 1.0, 0.8, 0.05, disabled=not keep_shadows)
    bg_model = st.radio("Original background", BG_MODELS, horizontal=True, disabled=not keep_shadows,
                        help="'uniform' = flat backdrop. 'gradient' = fits a plane (studio gradients).")
    with st.expander("Advanced shadow settings"):
        shadow_threshold = st.slider("Noise threshold", 0.0, 0.2, 0.04, 0.005, disabled=not keep_shadows)
        shadow_smooth = st.slider("Smoothing (px)", 1.0, 10.0, 2.0, 0.5, disabled=not keep_shadows)
        hue_tol = st.slider("Hue tolerance", 0.0, 0.6, 0.25, 0.05, disabled=not keep_shadows,
                            help="Rejects dark objects whose colour differs from the backdrop "
                                 "(e.g. a wooden stool on a white wall). 0 = off.")

    st.subheader("Edges")
    decontaminate = st.checkbox("Remove old-background halo", value=True)
    edge_shrink = st.slider("Shrink edge (px)", 0, 6, 0)
    feather = st.slider("Feather (px)", 0.0, 5.0, 0.0, 0.5)

# ------------------------------------------------------------------------- main
if uploaded is None:
    st.info("⬅️ Upload an image in the sidebar to start.")
    st.stop()

data = uploaded.getvalue()
image = load_image(data, max_side)
rgb = np.asarray(image)

# edits belong to one (image, engine, size): reset them when any of these changes
signature = (hashlib.md5(data).hexdigest(), engine, max_side)
if st.session_state.get("signature") != signature:
    st.session_state.update(signature=signature, edits=[], poly_points=[])
st.session_state.img_size = (image.width, image.height)
if undo and st.session_state.edits:
    st.session_state.edits.pop()
if reset:
    st.session_state.edits = []
    st.session_state.poly_points = []

t0 = time.perf_counter()
try:
    with st.spinner(f"Segmenting with {engine}… (first run downloads the model)"):
        alpha_raw = cached_alpha(data, engine, max_side)
except Exception as exc:  # model download / torch problems must not crash the UI
    st.error(f"Segmentation failed with '{engine}': {exc}")
    st.caption("Try the 'grabcut' engine to check the rest of the pipeline.")
    st.stop()
t_seg = time.perf_counter() - t0

t0 = time.perf_counter()
alpha_edit = apply_edits(rgb, alpha_raw, st.session_state.edits) if st.session_state.edits else alpha_raw
alpha = refine_alpha(alpha_edit, edge_shrink, feather)
with st.spinner("Cleaning edges…"):
    fg = cached_foreground(rgb, alpha, decontaminate)
# shadows are measured with the EDITED matte: an object added to the subject can't be a shadow
shadow = (cached_shadow(rgb, alpha_edit, bg_model, shadow_threshold, shadow_smooth, hue_tol)
          if keep_shadows else None)
out = Image.fromarray(composite(fg, alpha, bg_hex, shadow, shadow_strength))
t_render = time.perf_counter() - t0

left, right = st.columns(2)
with left:
    if tool == "Off" or streamlit_image_coordinates is None:
        st.subheader("Original")
        st.image(image)
        if tool != "Off":
            st.warning("Install the click component:  pip install streamlit-image-coordinates")
    else:
        st.subheader("Click to edit the selection")
        streamlit_image_coordinates(
            make_overlay(rgb, alpha_edit, st.session_state.poly_points),
            width=min(image.width, 700),
            key="img_click",
            on_click=on_image_click,
        )
        st.caption("Green = current subject. "
                   + ("Click an object to add/remove it." if tool == "Magic wand"
                      else "Click points around an area, then press 'Apply polygon'."))
        if tool == "Polygon":
            p1, p2 = st.columns(2)
            p1.button("✅ Apply polygon", on_click=apply_polygon,
                      disabled=len(st.session_state.poly_points) < 3)
            p2.button("Clear points", on_click=clear_points,
                      disabled=not st.session_state.poly_points)
with right:
    st.subheader("Result")
    st.image(out)
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    st.download_button("⬇️ Download PNG", buf.getvalue(), file_name="context_swap.png", mime="image/png")

n_edits = len(st.session_state.edits)
st.caption(f"{image.width}×{image.height}px · segmentation {t_seg:.1f}s (cached after first run) · "
           f"render {t_render:.1f}s · {n_edits} manual edit{'s' if n_edits != 1 else ''}")

with st.expander("🔍 Debug views (matte, shadow map, decontaminated subject)"):
    c1, c2, c3 = st.columns(3)
    c1.image((alpha * 255).astype(np.uint8), caption="Alpha matte")
    if shadow is not None:
        c2.image((shadow * 255).astype(np.uint8), caption="Shadow map (white = darkest)")
    c3.image(np.clip(fg, 0, 255).astype(np.uint8), caption="Foreground colours")