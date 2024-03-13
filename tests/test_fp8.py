import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util
import numpy as np

import haliax as hax
from haliax.fp8 import Fp8DotGeneralOp, apply_updates, compute_scale, partition_for_grad_overwrite
from haliax.nn import Linear


def test_fp8_is_reasonable():
    In = hax.Axis("In", 8)
    Out = hax.Axis("Out", 8)
    linear = Linear.init(In, Out, key=jrandom.PRNGKey(0))

    fp8_linear = Linear.init(In, Out, key=jrandom.PRNGKey(0), dot_general=hax.fp8.Fp8DotGeneralOp.init())

    input = hax.random.normal(jrandom.PRNGKey(0), In)
    output = linear(input)
    fp8_output = fp8_linear(input)

    assert output.shape == fp8_output.shape
    assert output.dtype == fp8_output.dtype

    assert jnp.allclose(output.array, fp8_output.array, atol=1e-1, rtol=1e-1)
    assert not jnp.all(output.array == fp8_output.array)


# https://github.com/google/flax/blob/6f2b08e024c2fd2f8cec42a6c82408cb35412319/tests/linen/linen_test.py#L1222
def test_fp_loop():
    key, init_key, random_key = jrandom.split(jrandom.PRNGKey(seed=123), 3)
    Batch = hax.Axis("Batch", 16)
    In = hax.Axis("In", 16)
    Out = hax.Axis("Out", 32)
    linear = Linear.init(In, Out, key=init_key, dot_general=Fp8DotGeneralOp.init())

    def _roll_and_update(amax_h, update):
        return jnp.roll(amax_h, shift=-1, axis=0).at[0].set(update)

    lr = 1e-3

    def apply_gradients(model, grads):
        overwrites, grads = partition_for_grad_overwrite(grads)
        updates = jax.tree_util.tree_map(lambda g: -lr * g, grads)
        model = apply_updates(model, updates, overwrites)
        return model

    def _train_step(model, x, dy):
        def loss_fn(lin):
            y = lin(x)
            loss = y * dy.astype(y.dtype)
            return hax.sum(loss).scalar()

        grad_fn = eqx.filter_grad(loss_fn)
        grads = grad_fn(model)
        return apply_gradients(model, grads)

    train_fn = eqx.filter_jit(_train_step)

    scale_x, amax_history_x = jnp.ones(()), jnp.zeros((1024,))
    scale_k, amax_history_k = jnp.ones(()), jnp.zeros((1024,))
    scale_g, amax_history_g = jnp.ones(()), jnp.zeros((1024,))
    e4m3_max = jnp.finfo(jnp.float8_e4m3fn).max.astype(jnp.float32)
    e5m2_max = jnp.finfo(jnp.float8_e5m2).max.astype(jnp.float32)

    for _ in range(5):
        key, random_key = jrandom.split(key, 2)
        # x = jrandom.normal(random_key, (16, 16), dtype=jnp.float32)
        # g = jrandom.normal(random_key, (16, 32), dtype=jnp.float32)
        x = hax.random.normal(
            random_key,
            (
                Batch,
                In,
            ),
        )
        g = hax.random.normal(
            random_key,
            (
                Batch,
                Out,
            ),
        )

        # Manually compute the expected amax history and scaling factors.
        amax_from_history_x = jnp.max(amax_history_x, axis=0)
        amax_from_history_k = jnp.max(amax_history_k, axis=0)
        amax_from_history_g = jnp.max(amax_history_g, axis=0)
        scale_x = compute_scale(amax_from_history_x, scale_x, e4m3_max)
        scale_k = compute_scale(amax_from_history_k, scale_k, e4m3_max)
        scale_g = compute_scale(amax_from_history_g, scale_g, e5m2_max)
        amax_history_x = _roll_and_update(amax_history_x, jnp.max(jnp.abs(x.array)))
        amax_history_k = _roll_and_update(amax_history_k, jnp.max(jnp.abs(linear.weight.array)))
        amax_history_g = _roll_and_update(amax_history_g, jnp.max(jnp.abs(g.array)))

        linear = train_fn(linear, x, g)

        rtol, atol = 0.001, 0.001
        np.testing.assert_allclose(
            linear.dot_general.input_amax_history,  # type: ignore
            amax_history_x,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            linear.dot_general.kernel_amax_history,  # type: ignore
            amax_history_k,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            linear.dot_general.output_grad_amax_history,  # type: ignore
            amax_history_g,
            rtol=rtol,
            atol=atol,
        )

        np.testing.assert_allclose(linear.dot_general.input_scale, scale_x, rtol=rtol, atol=atol)  # type: ignore
        np.testing.assert_allclose(linear.dot_general.kernel_scale, scale_k, rtol=rtol, atol=atol)  # type: ignore
        np.testing.assert_allclose(linear.dot_general.output_grad_scale, scale_g, rtol=rtol, atol=atol)  # type: ignore
