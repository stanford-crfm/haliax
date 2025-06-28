from __future__ import annotations

import typing

import jax.numpy as jnp

from haliax import Axis, Named, NamedArray


def test_namedarray_type_syntax():
    axes1 = typing.get_args(NamedArray["batch", "embed"])[1]
    axes2 = typing.get_args(Named["batch embed"])[1]  # type: ignore
    assert axes1 == axes2

    axes3 = typing.get_args(NamedArray["batch embed ..."])[1]
    assert axes3.before == ("batch", "embed") and axes3.subset and axes3.after == ()

    axes4 = typing.get_args(NamedArray[{"batch", "embed"}])[1]
    assert set(axes4.before) == {"batch", "embed"} and not axes4.ordered

    axes5 = typing.get_args(NamedArray[{"batch", "embed", ...}])[1]
    assert set(axes5.before) == {"batch", "embed"} and not axes5.ordered and axes5.subset

    axes6 = typing.get_args(NamedArray["... embed"])[1]
    assert axes6.before == () and axes6.after == ("embed",) and axes6.subset

    axes7 = typing.get_args(NamedArray["batch ... embed"])[1]
    assert axes7.before == ("batch",) and axes7.after == ("embed",) and axes7.subset


def test_named_param_annotation():
    def foo(x: Named["batch", "embed"]):  # type: ignore
        pass

    axes = typing.get_args(typing.get_type_hints(foo, include_extras=True)["x"])[1]
    assert axes.before == ("batch", "embed")


def test_namedarray_runtime_check():
    Batch = Axis("batch", 2)
    Embed = Axis("embed", 3)
    arr = NamedArray(jnp.zeros((Batch.size, Embed.size)), (Batch, Embed))
    assert arr.matches_axes(NamedArray["batch", "embed"])
    assert arr.matches_axes(Named["batch embed"])  # type: ignore
    assert arr.matches_axes(NamedArray["batch embed ..."])
    assert arr.matches_axes(NamedArray[{"batch", "embed"}])
    assert arr.matches_axes(NamedArray[{"batch", "embed", ...}])
    assert not arr.matches_axes(NamedArray["embed batch"])
    assert not arr.matches_axes(NamedArray[{"batch", "foo", ...}])
