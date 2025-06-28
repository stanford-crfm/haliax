import jax.numpy as jnp
import haliax as hax
from haliax import Axis, NamedArray, Named
import typing


def test_namedarray_type_syntax():
    t1 = NamedArray["batch", "embed"]
    t2 = Named["batch embed"]
    assert typing.get_args(t1)[1] == typing.get_args(t2)[1]

    t3 = NamedArray["batch embed ..."]
    axes3 = typing.get_args(t3)[1]
    assert axes3.before == ("batch", "embed") and axes3.subset and axes3.after == ()

    t4 = NamedArray[{"batch", "embed"}]
    axes4 = typing.get_args(t4)[1]
    assert set(axes4.before) == {"batch", "embed"} and not axes4.ordered

    t5 = NamedArray[{"batch", "embed", ...}]
    axes5 = typing.get_args(t5)[1]
    assert set(axes5.before) == {"batch", "embed"} and not axes5.ordered and axes5.subset

    t6 = NamedArray["... embed"]
    axes6 = typing.get_args(t6)[1]
    assert axes6.before == () and axes6.after == ("embed",) and axes6.subset

    t7 = NamedArray["batch ... embed"]
    axes7 = typing.get_args(t7)[1]
    assert axes7.before == ("batch",) and axes7.after == ("embed",) and axes7.subset


def test_named_param_annotation():
    def foo(x: Named["batch embed"]):
        pass

    axes = typing.get_args(foo.__annotations__["x"])[1]
    assert axes.before == ("batch", "embed")


def test_namedarray_runtime_check():
    Batch = Axis("batch", 2)
    Embed = Axis("embed", 3)
    arr = NamedArray(jnp.zeros((Batch.size, Embed.size)), (Batch, Embed))
    assert arr.matches_axes(NamedArray["batch", "embed"])
    assert arr.matches_axes(Named["batch embed"])
    assert arr.matches_axes(NamedArray["batch embed ..."])
    assert arr.matches_axes(NamedArray[{"batch", "embed"}])
    assert arr.matches_axes(NamedArray[{"batch", "embed", ...}])
    assert not arr.matches_axes(NamedArray["embed batch"])
    assert not arr.matches_axes(NamedArray[{"batch", "foo", ...}])
