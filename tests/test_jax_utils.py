import jax.lax
import jax.numpy as jnp

from haliax.jax_utils import checkpointed_scan


def test_checkpointed_scan():
    def body_fn(carry, x):
        return carry - x, carry + jnp.log1p(x)

    init = 0
    xs = jnp.arange(2 * 3 * 4, dtype=jnp.float32)

    lengths = [2, 3, 4]

    result, partials = checkpointed_scan(body_fn, init, xs, lengths)

    # compare to vanilla
    vanilla_result, vanilla_partials = jax.lax.scan(body_fn, init, xs)

    assert jnp.all(result == vanilla_result)
    assert jnp.all(partials == vanilla_partials)

    # check derivatives
    def f(x):
        x, y = checkpointed_scan(body_fn, init, x, lengths=lengths)
        return x + y.sum()

    def vanilla_f(x):
        x, y = jax.lax.scan(body_fn, init, x)
        return x + y.sum()

    z = jax.jit(jax.grad(f))(xs)

    vanilla_z = jax.jit(jax.grad(vanilla_f))(xs)

    assert jnp.allclose(z, vanilla_z)
