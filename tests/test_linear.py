import jax.numpy as jnp
import jax.random as jrandom

import haliax as hax


def test_linear_compute_precision():
    In = hax.Axis("In", 16)
    Out = hax.Axis("Out", 32)

    linear = hax.nn.Linear.init(In, Out, key=jrandom.PRNGKey(0))
    assert linear.weight.dtype == jnp.float32
    assert linear.bias.dtype == jnp.float32  # type: ignore
    input = hax.arange(In, dtype=jnp.bfloat16)
    out = linear(input)
    assert out.dtype == jnp.float32

    with hax.resource_env(mp="p=f32,c=bf16,o=f32"):
        out = linear(input)
        assert out.dtype == jnp.bfloat16
