import math
from typing import Callable, Optional

import equinox as eqx
import jax.lax
from jaxtyping import PRNGKeyArray

import haliax as hax

from ..axis import AxisSpec
from ..core import NamedArray
from ..jax_utils import named_call
from ..quantization import DotGeneralOp


class Linear(eqx.Module):
    """A named Linear layer. This module allows you to specify multiple named axes for both input
    and output, which is occasionally useful."""

    weight: NamedArray
    bias: Optional[NamedArray]

    In: AxisSpec = eqx.static_field()
    Out: AxisSpec = eqx.static_field()
    dot_general: DotGeneralOp = eqx.field(default_factory=DotGeneralOp.default)

    @staticmethod
    def init(
        In: AxisSpec,
        Out: AxisSpec,
        *,
        key,
        use_bias=True,
        out_first: bool = False,
        dot_general=None,
        init_scale: float = 1.0,
    ) -> "Linear":
        """

        Args:
            In: AxisSpec: The input axis spec
            Out: AxisSpec: The output axis spec
            key: PRNGKeyArray: The PRNG key to use for initialization
            use_bias: bool: Whether to use a bias term
            out_first: bool: Whether to put output axes first in the weight matrix. out_first is how PyTorch does it.
            dot_general: Callable: The dot_general function to use. Defaults to jax.lax.dot_general. For fp8 or int8
            init_scale: float: The scale to use for initialization. We scale init by 1/sqrt(Input.size)*init_scale
        """
        joint_spec = hax.concat_axis_specs(Out, In) if out_first else hax.concat_axis_specs(In, Out)
        input_size = hax.axis_size(In)
        weight = hax.random.truncated_normal(key, joint_spec, -3, 3) * (init_scale / math.sqrt(input_size))
        bias = hax.zeros(Out) if use_bias else None

        if dot_general is None:
            dot_general = DotGeneralOp.default()

        return Linear(weight, bias, In, Out, dot_general=dot_general)

    @named_call
    def __call__(self, inputs, *, key: Optional[PRNGKeyArray] = None):
        """
        Args:
            inputs (NamedArray): Input array
            key: Not used, but there for compat with other modules
        """
        del key
        q = inputs.dot(self.weight, axis=self.In, dot_general=self.dot_general)
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
