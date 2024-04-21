import jax
import pytest

import haliax as hax
from haliax._src.state_dict import flatten_linear_layers, tree_to_state_dict, unflatten_linear_layers
from haliax.nn import Linear


@pytest.mark.parametrize("out_dims_first", [True, False])
def test_flatten_linear_layers(out_dims_first: bool):
    H = hax.Axis("H", 10)
    W = hax.Axis("W", 20)
    D = hax.Axis("D", 30)
    B = hax.Axis("B", 40)
    linear = hax.nn.Linear.init((H, W), (D, B), key=jax.random.PRNGKey(0), use_bias=True, out_first=out_dims_first)

    if out_dims_first:
        assert linear.weight.axes == (D, B, H, W)
    else:
        assert linear.weight.axes == (H, W, D, B)

    flat_linear = flatten_linear_layers(linear)

    flat_state_dict = tree_to_state_dict(flat_linear)
    if out_dims_first:
        assert flat_state_dict["weight"].shape == (D.size * B.size, H.size * W.size)
    else:
        assert flat_state_dict["weight"].shape == (H.size * W.size, D.size * B.size)
    assert flat_state_dict["bias"].shape == (D.size * B.size,)
    assert flat_state_dict["weight"].dtype == flat_state_dict["bias"].dtype == linear.weight.dtype

    # now unflatten it
    linear2 = Linear.init((H, W), (D, B), key=jax.random.PRNGKey(1), use_bias=True, out_first=out_dims_first)
    new_linear = unflatten_linear_layers(linear2, flat_linear)

    if out_dims_first:
        assert new_linear.weight.axes == (D, B, H, W)
    else:
        assert new_linear.weight.axes == (H, W, D, B)
    assert new_linear.bias.axes == (D, B)  # type: ignore

    assert linear == new_linear
