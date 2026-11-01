import numpy as np
from PIL import Image

from contextswap.pipeline import RenderParams, load_image, render
from contextswap.utils import hex_to_rgb
from tests.synthetic import make_scene


def _render(**kw):
    rgb, alpha, region = make_scene()
    res = render(Image.fromarray(rgb), alpha, RenderParams(**kw))
    return np.asarray(res.image).astype(int), alpha, region


def test_background_is_exact_colour_far_from_subject():
    out, alpha, _ = _render(bg_hex="#2E86DE", decontaminate=False)
    corner = out[:30, :30].reshape(-1, 3)
    assert np.abs(corner - np.array(hex_to_rgb("#2E86DE"))).max() <= 1


def test_subject_core_keeps_its_colour():
    out, alpha, _ = _render(bg_hex="#00FF00", decontaminate=False)
    core = alpha > 0.999
    assert np.abs(out[core].mean(0) - np.array([200, 30, 40])).max() < 4


def test_shadow_off_means_flat_background():
    out, _, region = _render(bg_hex="#FFFFFF", keep_shadows=False, decontaminate=False)
    assert out[region].min() >= 254


def test_shadow_on_darkens_and_keeps_new_hue():
    out, _, region = _render(bg_hex="#FF8040", keep_shadows=True, shadow_strength=1.0, decontaminate=False)
    px = out[region].mean(0)
    assert px.sum() < 0.9 * (255 + 128 + 64), "shadow must darken the new colour"
    # multiply keeps channel ratios: R > G > B stays true
    assert px[0] > px[1] > px[2]


def test_strength_zero_equals_no_shadow():
    a, *_ = _render(bg_hex="#FF8040", keep_shadows=True, shadow_strength=0.0, decontaminate=False)
    b, *_ = _render(bg_hex="#FF8040", keep_shadows=False, decontaminate=False)
    assert np.array_equal(a, b)


def test_decontamination_runs_and_returns_same_size():
    out, alpha, _ = _render(bg_hex="#0000FF", decontaminate=True)
    assert out.shape == (*alpha.shape, 3)


def test_load_image_flattens_transparency_and_caps_size():
    rgba = Image.new("RGBA", (3000, 1000), (255, 0, 0, 0))
    img = load_image(rgba, max_side=1000)
    assert img.mode == "RGB" and max(img.size) == 1000
    assert img.getpixel((10, 10)) == (255, 255, 255)
