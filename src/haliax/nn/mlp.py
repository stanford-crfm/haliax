from typing import Callable, Optional, Sequence

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray

from ..axis import Axis, AxisSpec
from ..core import NamedArray
from ..jax_utils import maybe_rng_split
from ..mixed_precision import DTypeish
from .activations import relu
from .linear import Linear


DEFAULT_WIDTH_NAME = "mlp"


class MLP(eqx.Module):
    """
    A multilayer perceptron (MLP) / feed-forward neural network (FFNN).

    MLPs, with their stacked linear layers often with non-semantic axes for hidden
    dims, are not a particular strength of Haliax's design philosophy. Nonetheless, they are a useful tool for
    many tasks, and so we provide this module.

    In Haliax, all axes must have names, and names must be unique within an array. We considered a few strategies
    for naming the axes of an MLP, and settled on the following: By default, we alternate hidden names between "mlp"
    and "mlp2". Input and output names must be specified, and are not repeated. This naming scheme is not perfect,
    but does mean that model parallelism works reasonably well.

    NB: unlike Equinox's MLP, this MLP uses a static field for activation. If you want a learnable activation, you
    likely want a unique activation per layer, which neither version provides. Instead, you should use a
    [haliax.nn.Stacked][] with a suitable block.
    """

    activation: Callable = eqx.field(static=True)
    layers: Sequence[Linear]

    @staticmethod
    def init(
        Input: AxisSpec,
        Output: AxisSpec,
        width: int | Axis,
        depth: int,
        activation: Callable = relu,
        compute_dtype: Optional[DTypeish] = "compute",
        *,
        use_bias: bool = True,
        use_final_bias: bool = True,
        key: PRNGKeyArray,
    ) -> "MLP":
        """

        Args:
            Input: input axes
            Output: output axes
            width: width of hidden layers
            depth: number of hidden layers
            activation: activation function, defaults to relu
            compute_dtype: dtype to use for computation, or None to use jax default rules.
            use_bias: whether to include bias terms in hidden layers
            use_final_bias: whether to include bias term in final layer
            key: rng key for initialization.
        """
        Width = _get_width(width)
        Width2 = Width.alias(Width.name + "2")

        keys = jax.random.split(key, depth + 1)

        layers = []

        if depth == 0:
            # special case: no hidden layers
            layers.append(Linear.init(Input, Output, compute_dtype, use_bias=use_final_bias, key=keys[0]))
        else:
            # first hidden layer
            layers.append(Linear.init(Input, Width, compute_dtype, use_bias=use_bias, key=keys[0]))
            # middle hidden layers
            cur = Width
            next = Width2
            for i in range(1, depth):
                layers.append(Linear.init(cur, next, compute_dtype, use_bias=use_bias, key=keys[i]))
                cur, next = next, cur
            # final hidden layer
            layers.append(Linear.init(cur, Output, compute_dtype, use_bias=use_final_bias, key=keys[-1]))

        return MLP(
            layers=tuple(layers),
            activation=activation,
        )

    @property
    def In(self) -> AxisSpec:
        return self.layers[0].In

    @property
    def Out(self) -> AxisSpec:
        return self.layers[-1].Out

    def __call__(self, x: NamedArray, *, key=None) -> NamedArray:
        keys = maybe_rng_split(key, len(self.layers))
        for layer, k in zip(self.layers[:-1], keys):
            x = self.activation(layer(x, key=k))
        return self.layers[-1](x, key=keys[-1])


def _get_width(Width: int | Axis) -> Axis:
    if isinstance(Width, int):
        return Axis(DEFAULT_WIDTH_NAME, Width)
    else:
        return Width
