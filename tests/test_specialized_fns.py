import jax
import jax.numpy as jnp

import haliax.specialized_fns as hfns
from haliax import Axis, NamedArray


def test_top_k():
    H = Axis("H", 5)
    W = Axis("W", 6)
    D = Axis("D", 7)

    rand = jax.random.uniform(jax.random.PRNGKey(0), (H.size, W.size, D.size))
    n_rand = NamedArray(rand, (H, W, D))

    assert hfns.top_k(n_rand, D, 2).array.shape == (H.size, W.size, 2)
    assert jnp.all(
        jnp.equal(jax.lax.top_k(rand, 2)[0], hfns.top_k(n_rand, D, 2).array)
    )  # test that selecting last axis is same as default

    for idx, i in enumerate(n_rand.axes):  # then test selecting all axes
        t = jnp.transpose(rand, (*range(idx), *range(idx + 1, len(n_rand.axes)), idx))
        t = jax.lax.top_k(t, 2)[0]
        t = jnp.moveaxis(t, -1, idx)
        assert jnp.all(jnp.equal(t, hfns.top_k(n_rand, i, 2).array))
