from typing import Optional

import equinox as eqx

import haliax as hax

from ..core import NamedArray
from ..jax_utils import named_call
from ..types import AxisSpec


class Linear(eqx.Module):
    """A named Linear layer. This module allows you to specify multiple named axes for both input
    and output, which is occasionally useful."""

    weight: NamedArray
    bias: Optional[NamedArray]

    In: AxisSpec = eqx.static_field()
    Out: AxisSpec = eqx.static_field()

    @staticmethod
    def init(In: AxisSpec, Out: AxisSpec, *, key, use_bias=True, out_first: bool = False) -> "Linear":
        """

        :param In: Input axes
        :param Out: Output axes
        :param key: rng key for initialization
        :param use_bias: whether to include bias term
        :param out_first: whether to put output axes first in the weight matrix. out_first is how PyTorch does it.
        :return:
        """
        joint_spec = hax.concat_axis_specs(Out, In) if out_first else hax.concat_axis_specs(In, Out)
        weight = hax.random.normal(key, joint_spec) * 0.02
        bias = hax.zeros(Out) if use_bias else None
        return Linear(weight, bias, In, Out)

    @named_call
    def __call__(self, inputs):
        q = inputs.dot(self.In, self.weight)
        q = hax.auto_sharded(q)

        if self.bias is not None:
            q = q + self.bias
            q = hax.auto_sharded(q)

        return q

    @property
    def out_first(self):
        # We do it this way because of scan layers
        if isinstance(self.Out, hax.Axis):
            return self.weight.axes[-1] != self.Out
        else:
            return self.weight.axes[-len(self.Out) :] != self.Out
