import equinox as eqx
import jax
import jax.numpy as jnp

import haliax as hax
from haliax.nn.conv import Conv


def test_conv_basic_equiv_to_eqx():
    key = jax.random.PRNGKey(0)
    In = hax.Axis("In", 3)
    Out = hax.Axis("Out", 4)
    hax_conv = Conv.init(("Height", "Width"), In, Out, kernel_size=3, key=key)
    eqx_conv = eqx.nn.Conv(2, 3, 4, kernel_size=3, key=key)

    assert hax_conv.weight.array.shape == eqx_conv.weight.shape
    assert hax_conv.bias.array.shape == eqx_conv.bias.shape[0:1]
    assert jnp.all(hax_conv.weight.array == eqx_conv.weight)

    input = hax.random.normal(jax.random.PRNGKey(1), (In, hax.Axis("Height", 5), hax.Axis("Width", 6)))
    hax_output = hax_conv(input)
    eqx_output = eqx_conv(input.array)

    assert hax_output.array.shape == eqx_output.shape
    assert jnp.all(hax_output.array == eqx_output)

    # test batched
    input = hax.random.normal(
        jax.random.PRNGKey(1), (hax.Axis("Batch", 2), In, hax.Axis("Height", 5), hax.Axis("Width", 6))
    )
    hax_output = hax_conv(input)
    eqx_output = eqx.filter_vmap(eqx_conv)(input.array)

    assert hax_output.array.shape == eqx_output.shape
    assert jnp.all(hax_output.array == eqx_output)

    # test weird orders
    input = input.rearrange(("In", "Height", "Width", "Batch"))
    hax_output = hax_conv(input).rearrange(("Batch", "Out", "Height", "Width"))

    assert hax_output.array.shape == eqx_output.shape
    assert jnp.allclose(hax_output.array, eqx_output)
