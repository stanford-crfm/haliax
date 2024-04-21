from typing import Optional

import equinox as eqx
from jaxtyping import PRNGKeyArray

import haliax as hax

from .._src.state_dict import ModuleWithStateDictSerialization, StateDict, apply_prefix
from ..axis import AxisSpec
from ..core import NamedArray
from ..jax_utils import named_call
from ..quantization import DotGeneralOp
from ..util import ensure_tuple


class Linear(ModuleWithStateDictSerialization):
    """A named Linear layer. This module allows you to specify multiple named axes for both input
    and output, which is occasionally useful."""

    weight: NamedArray
    bias: Optional[NamedArray]

    In: AxisSpec = eqx.static_field()
    Out: AxisSpec = eqx.static_field()
    dot_general: DotGeneralOp = eqx.field(default_factory=DotGeneralOp.default)

    flatten_for_state_dict: bool = eqx.static_field(default=True)
    """Whether to flatten the input axes in state_dict serialization. This is useful for serializing to be
    compatible with other libraries like PyTorch."""

    @staticmethod
    def init(
        In: AxisSpec,
        Out: AxisSpec,
        *,
        key,
        use_bias=True,
        out_first: bool = False,
        dot_general=None,
        flatten_for_state_dict: bool = True,
    ) -> "Linear":
        """

        Args:
            In: AxisSpec: The input axis spec
            Out: AxisSpec: The output axis spec
            key: PRNGKeyArray: The PRNG key to use for initialization
            use_bias: bool: Whether to use a bias term
            out_first: bool: Whether to put output axes first in the weight matrix. out_first is how PyTorch does it.
            dot_general: Callable: The dot_general function to use. Defaults to jax.lax.dot_general. For fp8 or int8
        """
        joint_spec = hax.concat_axis_specs(Out, In) if out_first else hax.concat_axis_specs(In, Out)
        weight = hax.random.normal(key, joint_spec) * 0.02
        bias = hax.zeros(Out) if use_bias else None

        if dot_general is None:
            dot_general = DotGeneralOp.default()

        return Linear(weight, bias, In, Out, dot_general=dot_general, flatten_for_state_dict=flatten_for_state_dict)

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

    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> "Linear":
        if self.flatten_for_state_dict:
            weight = state_dict[apply_prefix(prefix, "weight")]
            bias = state_dict.get(apply_prefix(prefix, "bias"), None)
            weight, bias = _unflatten_linear(self, weight, bias)
        else:
            raw_weight = state_dict[apply_prefix(prefix, "weight")]

            weight = hax.named(raw_weight, self.weight.axes)
            if self.bias is not None:
                raw_bias = state_dict.get(apply_prefix(prefix, "bias"), None)
                bias = hax.named(raw_bias, self.bias.axes) if raw_bias is not None else None

        if bias is None:
            if self.bias is not None:
                raise ValueError("Bias is not None in the module, but not found in the state_dict")
        else:
            if self.bias is None:
                raise ValueError("Bias is not None in the state_dict, but not found in the module")

        return Linear(
            weight,
            bias,
            self.In,
            self.Out,
            dot_general=self.dot_general,
            flatten_for_state_dict=self.flatten_for_state_dict,
        )

    def update_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        if self.flatten_for_state_dict:
            weight, bias = _flatten_linear(self)
        else:
            weight = self.weight.array
            if self.bias is not None:
                bias = self.bias.array

        state_dict[apply_prefix(prefix, "weight")] = weight

        if bias is not None:
            state_dict[apply_prefix(prefix, "bias")] = bias

        return state_dict


def _flatten_linear(layer, *, out_dims_first_in_dict=None):
    weight = layer.weight
    bias = layer.bias

    if weight.array is not None:
        weight = weight.flatten_axes(layer.Out, "__OUT__").flatten_axes(layer.In, "__IN__")
        if bias is not None:
            bias = bias.flatten_axes(layer.Out, "__OUT__")

        if out_dims_first_in_dict is True:
            weight = weight.rearrange((..., "__OUT__", "__IN__"))
        elif out_dims_first_in_dict is False:
            weight = weight.rearrange((..., "__IN__", "__OUT__"))
        else:
            pass

    return weight.array, bias.array


def _unflatten_linear(layer, weight, bias, *, out_dims_first_in_dict=None):
    Out = ensure_tuple(layer.Out)
    In = ensure_tuple(layer.In)
    InOut = In + Out
    extra_dims = tuple(ax for ax in layer.weight.axes if ax not in InOut)

    if out_dims_first_in_dict is None:
        out_dims_first_in_dict = layer.out_first

    if out_dims_first_in_dict:
        weight = hax.named(weight, hax.concat_axis_specs(extra_dims, ("__OUT__", "__IN__")))
    else:
        weight = hax.named(weight, hax.concat_axis_specs(extra_dims, ("__IN__", "__OUT__")))

    if layer.out_first:
        weight = weight.rearrange((..., "__OUT__", "__IN__"))
    else:
        weight = weight.rearrange((..., "__IN__", "__OUT__"))

    # now unflatten
    weight = weight.unflatten_axis("__OUT__", layer.Out).unflatten_axis("__IN__", layer.In)

    if bias is not None:
        bias = hax.named(bias, hax.concat_axis_specs(extra_dims, ("__OUT__",)))
        bias = bias.unflatten_axis("__OUT__", layer.Out)

    return weight, bias
