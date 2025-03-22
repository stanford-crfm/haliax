import dataclasses
import math
from typing import Optional

import equinox as eqx
from jax.random import PRNGKey

import haliax as hax

from .._src.state_dict import Mod, ModuleWithStateDictSerialization
from ..axis import AxisSpec
from ..core import NamedArray
from ..jax_utils import named_call
from ..quantization import DotGeneralOp


class Linear(ModuleWithStateDictSerialization):
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
        key: PRNGKey,
        use_bias: bool = True,
        out_first: bool = True,
        dot_general: Optional[DotGeneralOp] = None,
        init_scale: float = 1.0,
    ) -> "Linear":
        """
        Args:
            In: AxisSpec: The input axis spec
            Out: AxisSpec: The output axis spec
            key: PRNGKeyArray: The PRNG key to use for initialization
            use_bias: bool: Whether to use a bias term
            out_first: bool: Whether to put output axes first in the weight matrix. out_first is how PyTorch does it.
            dot_general: Callable: The dot_general function to use. Defaults to jax.lax.dot_general.
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
    def __call__(self, inputs, *, key: Optional[PRNGKey] = None):
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

    def flatten_for_export(self: Mod) -> Mod:
        if isinstance(self.Out, hax.Axis) and isinstance(self.In, hax.Axis):
            return self

        weight = self.weight
        bias = self.bias

        new_Out = hax.flatten_axes(self.Out, "__OUT__")
        new_In = hax.flatten_axes(self.In, "__IN__")

        if weight.array is not None:
            out_first = self._out_first
            weight = weight.flatten_axes(self.Out, new_Out).flatten_axes(self.In, new_In)

            if out_first:
                weight = weight.rearrange((..., "__OUT__", "__IN__"))
            else:
                weight = weight.rearrange((..., "__IN__", "__OUT__"))

        if isinstance(bias, NamedArray):
            bias = bias.flatten_axes(self.Out, new_Out)

        return dataclasses.replace(self, weight=weight, bias=bias, In=new_In, Out=new_Out)

    def unflatten_from_export(self: Mod, template: Mod) -> Mod:
        weight = self.weight
        bias = self.bias

        if (template.In, template.Out) == (self.In, self.Out):
            return self

        if weight.array is not None:
            weight = weight.unflatten_axis("__OUT__", template.Out).unflatten_axis("__IN__", template.In)
            weight = weight.rearrange(template.weight.axes)

        if isinstance(bias, NamedArray):
            bias = bias.unflatten_axis("__OUT__", template.Out)
            bias = bias.rearrange(template.bias.axes)

        return dataclasses.replace(template, weight=weight, bias=bias)

    @property
    def _out_first(self):
        """
        Returns: bool: Whether the output axes are first in the weight matrix
        """
        # We do it this way because of scan layers
        if isinstance(self.Out, hax.Axis):
            return self.weight.axes[-1] != self.Out
        else:
            return self.weight.axes[-len(self.Out) :] != self.Out
