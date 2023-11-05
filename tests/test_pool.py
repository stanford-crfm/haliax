import jax.numpy as jnp

import haliax
import haliax as hax
from haliax.nn.pool import max_pool


# Tests largely cribbed from equinox


def test_maxpool1d():
    D = hax.Axis("D", 14)
    x = hax.arange(D)
    output = max_pool((D.resize(2),), x, stride=(3,))
    answer = hax.named(jnp.array([1, 4, 7, 10, 13], dtype=jnp.int32), "D")

    assert jnp.all((output == answer).array)

    answer = jnp.array([2, 5, 8, 11])
    output = max_pool(D.resize(3), x, stride=(3,), padding=0)
    assert jnp.all(output.array == answer)

    # test batch axes
    B = hax.Axis("B", 2)
    x = x.rearrange("(B D) -> B D", B=B)
    output = max_pool(D.resize(2), x, stride=(3,))
    print(output)
    answer = jnp.array([[1, 4], [8, 11]])
    assert jnp.all(output.array == answer)

    output = max_pool(D.resize(3), x, stride=(3,), padding=0)
    answer = jnp.array([[2, 5], [9, 12]])
    assert jnp.all(output.array == answer)

    output = max_pool(D.resize(2), x, stride=(3,), padding="SAME")
    answer = jnp.array([[1, 4, 6], [8, 11, 13]])
    assert jnp.all(output.array == answer)


def test_maxpool2d():
    _x = jnp.arange(36).reshape(6, 6)
    x = hax.named(_x, ("H", "W"))

    # max_pool = eqx.nn.MaxPool2d(2, (3, 2))
    output = max_pool((hax.Axis("H", 2), hax.Axis("W", 2)), x, stride=(3, 2))
    answer = jnp.array([[7, 9, 11], [25, 27, 29]])

    assert jnp.all(output.array == answer)

    output = max_pool((hax.Axis("H", 3), hax.Axis("W", 3)), x, stride=2, padding=1)
    answer = jnp.array([[7, 9, 11], [19, 21, 23], [31, 33, 35]])

    assert jnp.all(output.array == answer)

    # test batch axes
    B = hax.Axis("B", 2)
    x = haliax.stack(B, [x, x])

    output = max_pool((hax.Axis("H", 2), hax.Axis("W", 2)), x, stride=(3, 2))
    answer = jnp.array([[[7, 9, 11], [25, 27, 29]], [[7, 9, 11], [25, 27, 29]]])

    assert jnp.all(output.array == answer)


def test_maxpool3d():
    _x = jnp.arange(64).reshape(4, 4, 4)
    x = hax.named(_x, ("H", "W", "D"))
    # max_pool = eqx.nn.MaxPool3d(2, (3, 2, 1))
    output = max_pool((hax.Axis("H", 2), hax.Axis("W", 2), hax.Axis("D", 2)), x, stride=(3, 2, 1))

    answer = jnp.array([[[21, 22, 23], [29, 30, 31]]])

    assert jnp.all(output.array == answer)

    # max_pool = eqx.nn.MaxPool3d(
    #     kernel_size=3, padding=(0, 1, 1), stride=2, use_ceil=True
    # )
    # answer = jnp.asarray(
    #     [
    #         [[37, 39, 39], [45, 47, 47], [45, 47, 47]],
    #         [[53, 55, 55], [61, 63, 63], [61, 63, 63]],
    #     ]
    # )
    # output = max_pool(
    #     (hax.Axis("H", 3), hax.Axis("W", 3), hax.Axis("D", 3)),
    #     x,
    #     stride=2,
    #     padding=( (0, 1), (2, 2), (2, 2) ),
    # )
    # assert jnp.all(output.array == answer)
