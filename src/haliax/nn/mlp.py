from typing import Callable, Sequence

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray

from ..axis import Axis, AxisSpec
from ..core import NamedArray
from ..jax_utils import maybe_rng_split
from .activations import relu
from .linear import Linear


DEFAULT_WIDTH_NAME = "mlp"


class MLP(eqx.Module):
    """
    A multilayer perceptron (MLP) / feed-forward neural network (FFNN).

    MLPs, with stacked linear layers often of the same size with non-semantic axes for hidden
    dims, are not a particular strength of Haliax's design philosophy. Nonetheless, they are a useful tool for
    prototyping and testing, and can be used as a simple baseline for more complex models.

    In Haliax, all axes must have names, and names must be unique within an array. We considered a few strategies
    for naming the axes of an MLP, and settled on the following: By default, we alternate hidden names between "mlp"
    and "mlp2". Input and output names must be specified, and are not repeated. This naming scheme is not perfect,
    but does mean that model parallelism works reasonably well.

    NB: unlike Equinox's MLP, this MLP uses a static field for activation. If you want a learnable activation, you
    likely want a unique activation per layer, which neither version provides. Instead, you should use a
    [haliax.nn.Stacked][] with a suitable block.
    """

    Input: Axis = eqx.field(static=True)
    Output: Axis = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)
    layers: Sequence[Linear]

    @staticmethod
    def init(
        Input: AxisSpec,
        Output: AxisSpec,
        width: int | Axis,
        depth: int,
        activation: Callable = relu,
        *,
        use_bias: bool = True,
        use_final_bias: bool = True,
        key: PRNGKeyArray,
    ):
        Width = _get_width(width)
        Width2 = Width.alias(Width.name + "2")

        keys = jax.random.split(key, depth + 1)

        layers = []

        if depth == 0:
            # special case: no hidden layers
            layers.append(Linear.init(Input, Output, use_bias=use_final_bias, key=keys[0]))
        else:
            # first hidden layer
            layers.append(Linear.init(Input, Width, use_bias=use_bias, key=keys[0]))
            # middle hidden layers
            cur = Width
            next = Width2
            for i in range(1, depth):
                layers.append(Linear.init(cur, next, use_bias=use_bias, key=keys[i]))
                cur, next = next, cur
            # final hidden layer
            layers.append(Linear.init(cur, Output, use_bias=use_final_bias, key=keys[-1]))

        return MLP(
            Input=Input,
            Output=Output,
            layers=tuple(layers),
            activation=activation,
        )

    def __call__(self, x: NamedArray, *, key) -> NamedArray:
        keys = maybe_rng_split(key, len(self.layers))
        for layer, k in zip(self.layers[:-1], keys):
            x = self.activation(layer(x, key=k))
        return self.layers[-1](x, key=keys[-1])


def _get_width(Width: int | Axis) -> Axis:
    if isinstance(Width, int):
        return Axis(DEFAULT_WIDTH_NAME, Width)
    else:
        return Width
