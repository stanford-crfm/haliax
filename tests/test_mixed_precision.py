import jax.numpy as jnp
import jmp

import haliax as hax
import haliax.mixed_precision as hmp


def test_cast_floating():
    policy = jmp.get_policy("p=f32,c=bf16,o=f32")

    D = hax.Axis("D", 16)
    x = hax.arange(D, dtype=float)

    assert policy.cast_to_compute(x).dtype == jnp.bfloat16
    assert policy.cast_to_output(x).dtype == jnp.float32
    assert policy.cast_to_param(x).dtype == jnp.float32

    assert hmp.cast_floating(x, "compute", policy).dtype == jnp.bfloat16
    assert hmp.cast_floating(x, "param", policy).dtype == jnp.float32


def test_cast_floating_with_context_manager():
    D = hax.Axis("D", 16)
    x = hax.arange(D, dtype=float)

    with hax.resource_env(mp="p=f32,c=bf16,o=f32"):
        assert hmp.cast_floating(x, "compute").dtype == jnp.bfloat16
        assert hmp.cast_floating(x, "param").dtype == jnp.float32

    # The default env is fp32
    assert hmp.cast_floating(x, "compute").dtype == jnp.float32
    assert hmp.cast_floating(x, "param").dtype == jnp.float32
