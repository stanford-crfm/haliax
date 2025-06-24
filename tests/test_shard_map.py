import numpy as np
import jax
import jax.numpy as jnp

import haliax as hax
from haliax import Axis
from haliax.partitioning import ResourceAxis
from test_utils import skip_if_not_enough_devices


@skip_if_not_enough_devices(1)
def test_shard_map_basic():
    X = Axis("X", 8)
    mesh = jax.sharding.Mesh(np.array(jax.devices()), (ResourceAxis.DATA,))

    def plus_one(x: hax.NamedArray):
        return x + 1

    fn = hax.shard_map(plus_one, mesh=mesh, out_specs=X, check_rep=False)

    x = hax.arange(X)
    with mesh:
        result = fn(x)

    assert isinstance(result, hax.NamedArray)
    assert result.axes == (X,)
    assert jnp.allclose(result.array, x.array + 1)


@skip_if_not_enough_devices(1)
def test_shard_map_explicit_in_specs():
    X = Axis("X", 8)
    mesh = jax.sharding.Mesh(np.array(jax.devices()), (ResourceAxis.DATA,))

    def plus_one(x: hax.NamedArray):
        return x + 1

    fn = hax.shard_map(plus_one, mesh=mesh, in_specs=(X,), out_specs=X, check_rep=False)

    x = hax.arange(X)
    with mesh:
        result = fn(x)

    assert isinstance(result, hax.NamedArray)
    assert result.axes == (X,)
    assert jnp.allclose(result.array, x.array + 1)
