from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import haliax as hax
from haliax.nn import Linear
from haliax.nn.scan import _stack_state_dict, _unstack_state_dict
from haliax.state_dict import flatten_linear_layers, from_state_dict, to_state_dict, unflatten_linear_layers


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

    flat_state_dict = to_state_dict(flat_linear)
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


# Test cases for stack_state_dict
@pytest.mark.parametrize(
    "input_dict, prefix, expected_output",
    [
        # Single block stacking
        (
            {
                "block.0.weight": jnp.array([1, 2]),
                "block.0.bias": jnp.array([3]),
                "block.1.weight": jnp.array([4, 5]),
                "block.1.bias": jnp.array([6]),
            },
            "block",
            {
                "block.weight": jnp.array([[1, 2], [4, 5]]),
                "block.bias": jnp.array([[3], [6]]),
            },
        ),
        # Mixed data types and unmatched items remain unchanged
        (
            {
                "block.0.weight": jnp.array([1, 2]),
                "block.0.bias": jnp.array([3]),
                "block.1.weight": jnp.array([4, 5]),
                "block.1.bias": jnp.array([6.0]),
                "unrelated.item": jnp.array([7]),
            },
            "block",
            {
                "block.weight": jnp.array([[1, 2], [4, 5]]),
                "block.bias": jnp.array([[3.0], [6.0]]),
                "unrelated.item": jnp.array([7]),
            },
        ),
        # No items match prefix, all items should remain unchanged
        (
            {
                "module.0.param": jnp.array([1]),
                "module.1.param": jnp.array([2]),
            },
            "block",
            {
                "module.0.param": jnp.array([1]),
                "module.1.param": jnp.array([2]),
            },
        ),
    ],
)
def test_stack_state_dict(input_dict, prefix, expected_output):
    result = _stack_state_dict(input_dict, prefix)
    for key in expected_output:
        assert jnp.all(jnp.array_equal(result[key], expected_output[key])), f"Failed on key: {key}"

    # now unstack it
    unstacked = _unstack_state_dict(result, prefix)
    for key in input_dict:
        assert jnp.all(jnp.array_equal(unstacked[key], input_dict[key])), f"Failed on key: {key}"


class M(eqx.Module):
    a: Any
    b: Any

    def __init__(self, a, b):
        self.a = a
        self.b = b


def test_to_from_state_dict():
    a = jnp.array([1, 2])
    b = jnp.array([3, 4])
    m = M(a, b)

    state_dict = to_state_dict(m)
    assert state_dict == {"a": a, "b": b}

    m2 = M(jnp.array([0, 0]), jnp.array([0, 0]))
    m2 = from_state_dict(m2, state_dict)
    assert jnp.all(m2.a == a)
    assert jnp.all(m2.b == b)
