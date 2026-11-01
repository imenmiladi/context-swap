import pytest

from contextswap.utils import hex_to_rgb


@pytest.mark.parametrize("value,expected", [
    ("#FF5733", (255, 87, 51)), ("ff5733", (255, 87, 51)), ("#fff", (255, 255, 255)), (" #000 ", (0, 0, 0)),
])
def test_hex_ok(value, expected):
    assert hex_to_rgb(value) == expected


@pytest.mark.parametrize("bad", ["", "#12", "#GGGGGG", "#1234567", "red"])
def test_hex_bad(bad):
    with pytest.raises(ValueError):
        hex_to_rgb(bad)
