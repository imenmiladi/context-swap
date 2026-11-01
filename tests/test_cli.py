from PIL import Image

from contextswap.cli import main
from tests.synthetic import make_scene


def test_cli_end_to_end_with_grabcut(tmp_path):
    rgb, _, _ = make_scene()
    src = tmp_path / "in.png"
    Image.fromarray(rgb).save(src)
    out = tmp_path / "out.png"
    code = main([str(src), "--bg", "#2E86DE", "--engine", "grabcut", "--keep-shadows",
                 "--save-debug", "-o", str(out)])
    assert code == 0 and out.exists()
    assert Image.open(out).size == (480, 360)
    assert (tmp_path / "out_alpha.png").exists() and (tmp_path / "out_shadow.png").exists()
