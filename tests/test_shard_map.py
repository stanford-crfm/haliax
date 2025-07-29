import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

import haliax as hax
from haliax import Axis
from haliax.partitioning import ResourceAxis, axis_mapping
from test_utils import skip_if_not_enough_devices

Dim = Axis("dim", 8)

@skip_if_not_enough_devices(2)
def test_shard_map_basic():
    mesh = Mesh(np.array(jax.devices()), (ResourceAxis.DATA,))

    def fn(x):
        return x + 1

    sm = hax.shard_map(fn, in_specs=Dim, out_specs=Dim, mesh=mesh, check_rep=False)
    x = hax.ones(Dim)
    with axis_mapping({"dim": ResourceAxis.DATA}), mesh:
        out = sm(x.array)
    assert out.axes == (Dim,)
    assert jnp.allclose(out.array, x.array + 1)


@skip_if_not_enough_devices(2)
def test_shard_map_infer_out():
    mesh = Mesh(np.array(jax.devices()), (ResourceAxis.DATA,))

    def fn(x):
        return x + 2

    sm = hax.shard_map(fn, in_specs=Dim, out_specs=None, mesh=mesh, check_rep=False)
    x = hax.ones(Dim)
    with axis_mapping({"dim": ResourceAxis.DATA}), mesh:
        out = sm(x.array)
    assert out.axes == (Dim,)
    assert jnp.allclose(out.array, x.array + 2)


@skip_if_not_enough_devices(2)
def test_shard_map_infer_in_specs():
    mesh = Mesh(np.array(jax.devices()), (ResourceAxis.DATA,))

    def fn(x):
        return x * 3

    spec = hax.ones(Dim)
    sm = hax.shard_map(fn, in_specs=spec, out_specs=Dim, mesh=mesh, check_rep=False)
    x = hax.ones(Dim)
    with axis_mapping({"dim": ResourceAxis.DATA}), mesh:
        out = sm(x.array)
    assert out.axes == (Dim,)
    assert jnp.allclose(out.array, x.array * 3)


@skip_if_not_enough_devices(2)
def test_shard_map_pytree_multidim_output():
    mesh = Mesh(np.array(jax.devices()), (ResourceAxis.DATA,))

    B = Axis("b", 8)
    C = Axis("c", 4)
    D = Axis("d", 2)

    def fn(x):
        return {
            "expanded": hax.broadcast_axis(x, D),
            "twice": x + x,
        }

    x = hax.ones((B, C))
    sm = hax.shard_map(fn, in_specs=x, out_specs=None, mesh=mesh, check_rep=False)
    with axis_mapping({"b": ResourceAxis.DATA}), mesh:
        out = sm(x.array)

    assert isinstance(out, dict)
    assert out["expanded"].axes == (D, B, C)
    assert out["twice"].axes == (B, C)
    expected_expanded = jnp.broadcast_to(x.array, (D.size, B.size, C.size))
    assert jnp.allclose(out["expanded"].array, expected_expanded)
    assert jnp.allclose(out["twice"].array, x.array * 2)


@skip_if_not_enough_devices(2)
def test_shard_map_infer_in_specs_multiple_args():
    mesh = Mesh(np.array(jax.devices()), (ResourceAxis.DATA,))

    def fn(a, b):
        return a + b

    spec_a = hax.ones(Dim)
    spec_b = hax.ones(Dim)
    sm = hax.shard_map(fn, in_specs=(spec_a, spec_b), out_specs=Dim, mesh=mesh, check_rep=False)
    x = hax.ones(Dim)
    y = hax.arange(Dim)
    with axis_mapping({"dim": ResourceAxis.DATA}), mesh:
        out = sm(x.array, y.array)

    assert out.axes == (Dim,)
    assert jnp.allclose(out.array, x.array + y.array)
