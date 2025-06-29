import jax.numpy as jnp

import haliax as hax
import haliax.typing as ht


def foo(a: ht.f32[{"B", "embed"}], x: ht.i32[{"pos", "B"}]):
    axes = ht.check_axes(a, x)
    return axes


def test_generic_axis_resolution():
    Batch, Pos, Embed = hax.make_axes(batch=12, pos=4, embed=4)
    a = hax.zeros((Batch, Embed))
    x = hax.zeros((Pos, Batch), dtype=jnp.int32)
    axes = foo(a, x)
    assert axes["B"] == hax.Axis("batch", 12)


def foo_multi(a: ht.f32[{"B", "Q"}], x: ht.i32[{"Y", "B"}]):
    axes = ht.check_axes(a, x)
    return axes


def test_multiple_generics():
    Batch, Query, YAx = hax.make_axes(batch=12, query=7, y=4)
    a = hax.zeros((Batch, Query))
    x = hax.zeros((YAx, Batch), dtype=jnp.int32)
    axes = foo_multi(a, x)
    assert axes["B"] == hax.Axis("batch", 12)
    assert axes["Q"] == hax.Axis("query", 7)
    assert axes["Y"] == hax.Axis("y", 4)
