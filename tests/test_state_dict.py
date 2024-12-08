from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from optax.tree_utils import tree_zeros_like

import haliax as hax
from haliax._src.state_dict import _restack_stacked, _unstack_stacked
from haliax.nn import BlockSeq, Linear, Stacked
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


class Module(eqx.Module):
    named: hax.NamedArray
    array: jax.Array
    static: int = eqx.static_field()

    def __call__(self, x, *, key):
        return x + self.array + self.static

    @staticmethod
    def init(named, array, static):
        return Module(named=named, array=array, static=static)


class Mod2(eqx.Module):
    a: Stacked[Module]

    @staticmethod
    def init(Block2, named, array, static):
        return Mod2(a=Stacked.init(Block2, Module)(named=named, array=array, static=static))


def test_tree_unstacking():
    Block = hax.Axis("block", 4)
    E = hax.Axis("E", 10)

    initial_named = hax.random.uniform(jax.random.PRNGKey(0), (Block, E))

    m = Stacked.init(Block, Module)(named=initial_named, array=jax.numpy.ones(Block.size), static=1)

    assert m.stacked.named.axes == (Block, E)
    assert m.stacked.array.shape == (Block.size,)
    assert m.stacked.static == 1

    unstacked = _unstack_stacked(m)

    assert isinstance(unstacked, BlockSeq)

    z = tree_zeros_like(m)

    restacked = _restack_stacked(z, unstacked)

    assert restacked == m


def test_double_stacking():

    Block1 = hax.Axis("Block1", 4)
    Block2 = hax.Axis("Block2", 2)

    E = hax.Axis("E", 10)

    initial_named = hax.random.uniform(jax.random.PRNGKey(0), (Block1, Block2, E))

    m_stacked = Stacked.init(Block1, Mod2)(
        Block2, named=initial_named, array=jax.numpy.ones((Block1.size, Block2.size)), static=1
    )

    m_unstacked = _unstack_stacked(m_stacked)

    # ensure there are no stacked left
    leaves = jax.tree.leaves(m_unstacked, is_leaf=lambda x: isinstance(x, Stacked))
    assert not any(isinstance(leaf, Stacked) for leaf in leaves)

    m_restacked = _restack_stacked(tree_zeros_like(m_stacked), m_unstacked)

    assert m_stacked == m_restacked


def test_torch_compatible_state_dict_stacked():
    Block1 = hax.Axis("Block1", 4)
    Block2 = hax.Axis("Block2", 2)

    E = hax.Axis("E", 10)

    initial_named = hax.random.uniform(jax.random.PRNGKey(0), (Block1, Block2, E))

    m_stacked = Stacked.init(Block1, Mod2)(
        Block2, named=initial_named, array=jax.numpy.ones((Block1.size, Block2.size)), static=1
    )

    state_dict = hax.state_dict.to_torch_compatible_state_dict(m_stacked)

    # check for some keys:
    assert "0.a.0.array" in state_dict
    assert "1.a.1.named" in state_dict

    z = tree_zeros_like(m_stacked)

    m_unstacked = hax.state_dict.from_torch_compatible_state_dict(z, state_dict)

    assert m_stacked == m_unstacked


def test_torch_compatible_state_dict_stacked_linear():
    Block1 = hax.Axis("Block1", 4)
    Block2 = hax.Axis("Block2", 2)

    E = hax.Axis("E", 10)
    E2 = hax.Axis("E2", 5)

    class ModLinear(eqx.Module):
        a: hax.nn.Stacked[hax.nn.Linear]

        @staticmethod
        def init(Block2, key):
            return ModLinear(a=hax.nn.Stacked.init(Block2, hax.nn.Linear)(E, E2, key=key))

    m_stacked = Stacked.init(Block1, ModLinear)(
        Block2, key=jax.random.split(jax.random.PRNGKey(1), (Block1.size, Block2.size))
    )

    state_dict = hax.state_dict.to_torch_compatible_state_dict(m_stacked)

    # check for some keys:
    assert "0.a.0.bias" in state_dict
    assert "1.a.1." in state_dict

    z = tree_zeros_like(m_stacked)

    m_unstacked = hax.state_dict.from_torch_compatible_state_dict(z, state_dict)

    assert m_stacked == m_unstacked
