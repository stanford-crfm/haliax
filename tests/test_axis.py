import pytest

from haliax.axis import Axis, eliminate_axes


def test_eliminate_axes():
    H = Axis("H", 3)
    W = Axis("W", 4)
    C = Axis("C", 5)

    assert eliminate_axes((H, W), (H,)) == (W,)
    assert eliminate_axes((H, W), (W,)) == (H,)
    assert eliminate_axes((H, W), (H, W)) == ()

    with pytest.raises(ValueError):
        eliminate_axes((H, W), (C,))

    with pytest.raises(ValueError):
        eliminate_axes((H, W), (H, C))
