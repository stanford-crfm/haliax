from typing import Optional, TypeVar

import equinox as eqx
from jax import nn as jnn
from jax import numpy as jnp

import haliax
import haliax as hax

from ..axis import AxisSelection, AxisSpec
from ..core import NamedArray
from ..types import Scalar
from ..wrap import unwrap_namedarrays, wrap_axiswise_call, wrap_reduction_call


A = TypeVar("A", Scalar, NamedArray, jnp.ndarray)


class LayerNorm(eqx.Module):
    r"""
    Normalises the input along the specified axis (or axes), using the mean and variance of the
    input along that axis.
    """
    axis: AxisSpec = eqx.static_field()
    weight: Optional[NamedArray]
    bias: Optional[NamedArray]

    eps: float = eqx.static_field(default=1e-5)

    @staticmethod
    def init(axis: AxisSpec, eps: float = 1e-5, use_weight: bool = True, use_bias: bool = True):
        if use_weight:
            weight = hax.ones(axis)
        else:
            weight = None
        if use_bias:
            bias = hax.zeros(axis)
        else:
            bias = None

        return LayerNorm(axis, weight, bias, eps)

    def __call__(self, x: NamedArray) -> NamedArray:
        mean = x.mean(self.axis)
        var = x.var(self.axis)
        inv = hax.rsqrt(var + self.eps)
        out = (x - mean) * inv

        if self.weight is not None:
            out = self.weight * out
        if self.bias is not None:
            out = out + self.bias
        return out


def logsumexp(a: A, axis: Optional[AxisSelection] = None) -> A:
    # TODO: logsumexp indirectly supports where via `b`. we should support it directly
    return wrap_reduction_call(jnn.logsumexp, a, axis=axis, single_axis_only=False, supports_where=False)


def softmax(a: A, axis: Optional[AxisSelection] = None) -> A:
    return wrap_axiswise_call(jnn.softmax, a, axis=axis, single_axis_only=False)


def log_softmax(a: A, axis: Optional[AxisSelection] = None) -> A:
    return wrap_axiswise_call(jnn.log_softmax, a, axis=axis, single_axis_only=False)


def standardize(
    x: NamedArray,
    axis: AxisSpec,
    *,
    mean: Optional[NamedArray] = None,
    variance: Optional[NamedArray] = None,
    epsilon: float = 1e-5,
    where: Optional[NamedArray] = None,
) -> NamedArray:
    """Analogous to [jax.nn.standardize][], but with support for NamedArrays."""
    x, mean, variance, where = haliax.broadcast_arrays(x, mean, variance, where)  # type: ignore
    raw_x, mean, variance, where = unwrap_namedarrays(x, mean, variance, where)
    axis_indices = x._lookup_indices(axis)

    plain = jnn.standardize(raw_x, axis_indices, mean=mean, variance=variance, epsilon=epsilon, where=where)
    return NamedArray(plain, x.axes)
