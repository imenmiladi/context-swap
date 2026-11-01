import numpy as np

from contextswap.shadow import extract_shadow_map
from tests.synthetic import make_scene


def test_shadow_detected_and_background_clean():
    rgb, alpha, region = make_scene()
    s, bg_lum = extract_shadow_map(rgb, alpha)
    assert abs(bg_lum - 225) < 5
    assert s[region].mean() > 0.2, "cast shadow should be recovered"
    far = np.zeros_like(region); far[:40, :40] = True  # empty top-left corner
    assert s[far].max() < 0.02, "clean background must give no shadow"


def test_no_shadow_in_flat_scene():
    rgb, alpha, _ = make_scene(with_shadow=False)
    s, _ = extract_shadow_map(rgb, alpha)
    assert s.max() < 0.05


def test_gradient_model_beats_uniform_on_gradient_backdrop():
    # strong left->right gradient: with a flat model the dark side looks like a shadow
    # AND touches the subject, so the connected-component filter cannot save us.
    rgb, alpha, region = make_scene(gradient=120, with_shadow=False)
    s_uni, _ = extract_shadow_map(rgb, alpha, "uniform")
    s_grad, _ = extract_shadow_map(rgb, alpha, "gradient")
    assert s_uni.max() > 0.05, "flat model should produce a false shadow here"
    assert s_grad.max() < 0.03, "plane fit should remove it"


def test_subject_pixels_do_not_leak_as_shadow_far_from_edge():
    rgb, alpha, _ = make_scene(with_shadow=False)
    s, _ = extract_shadow_map(rgb, alpha)
    core = alpha > 0.99
    # deep inside the subject (eroded) the map must be zero
    import cv2
    deep = cv2.erode(core.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    assert s[deep].max() < 1e-3
