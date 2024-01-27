from typing import Optional

import equinox as eqx
from jax.typing import DTypeLike
from jaxtyping import PRNGKeyArray

import haliax as hax

from ..axis import AxisSpec
from ..core import NamedArray
from ..jax_utils import named_call


class Linear(eqx.Module):
    """A named Linear layer. This module allows you to specify multiple named axes for both input
    and output, which is occasionally useful."""

    weight: NamedArray
    bias: Optional[NamedArray]

    In: AxisSpec = eqx.static_field()
    Out: AxisSpec = eqx.static_field()
    compute_dtype: Optional[DTypeLike] = eqx.static_field()

    @staticmethod
    def init(
        In: AxisSpec,
        Out: AxisSpec,
        compute_dtype: Optional[DTypeLike] = None,
        *,
        key,
        use_bias=True,
        out_first: bool = False,
    ) -> "Linear":
        """

        :param In: Input axes
        :param Out: Output axes
        :param compute_dtype: dtype to use for computation, or None to use the context default
        :param key: rng key for initialization
        :param use_bias: whether to include bias term
        :param out_first: whether to put output axes first in the weight matrix. out_first is how PyTorch does it.
        :return:
        """
        joint_spec = hax.concat_axis_specs(Out, In) if out_first else hax.concat_axis_specs(In, Out)
        weight = hax.random.normal(key, joint_spec) * 0.02
        bias = hax.zeros(Out) if use_bias else None
        return Linear(weight, bias, In, Out, compute_dtype)

    @named_call
    def __call__(self, inputs, *, key: Optional[PRNGKeyArray] = None):
        """
        Args:
            inputs (NamedArray): Input array
            key: Not used, but there for compat with other modules
        """
        del key

        weight = self.weight

        compute_dtype = self.compute_dtype
        if compute_dtype is None:
            compute_dtype = hax.current_mp_policy().compute_dtype
        if compute_dtype is not None:
            weight = weight.astype(compute_dtype)

        q = inputs.dot(weight, axis=self.In)
        q = hax.auto_sharded(q)

        bias = self.bias
        if bias is not None:
            if compute_dtype is not None:
                bias = bias.astype(compute_dtype)
            q = q + bias
            q = hax.auto_sharded(q)

        return q

    @property
    def out_first(self):
        # We do it this way because of scan layers
        if isinstance(self.Out, hax.Axis):
            return self.weight.axes[-1] != self.Out
        else:
            return self.weight.axes[-len(self.Out) :] != self.Out
