import functools
import typing
import warnings
from typing import Optional, Tuple, Union

import jax.nn as jnn
import jax.numpy as jnp

import haliax
import haliax as hax
import haliax.nn.attention as attention

from ..axis import Axis, AxisSelection, AxisSelector, AxisSpec
from ..core import NamedArray
from ..types import Scalar
from ..util import UNSPECIFIED, Unspecified
from ..wrap import ReductionFunction, unwrap_namedarrays, wrap_axiswise_call, wrap_elemwise_unary, wrap_reduction_call
from .conv import Conv, ConvTranspose
from .dropout import Dropout, dropout
from .embedding import Embedding
from .linear import Linear
from .normalization import LayerNorm
from .scan import Stacked


A = typing.TypeVar("A", Scalar, NamedArray, jnp.ndarray)


def relu(a: A) -> A:
    return wrap_elemwise_unary(jnn.relu, a)


def relu6(a: A) -> A:
    return wrap_elemwise_unary(jnn.relu6, a)


def sigmoid(a: A) -> A:
    return wrap_elemwise_unary(jnn.sigmoid, a)


def softplus(a: A) -> A:
    return wrap_elemwise_unary(jnn.softplus, a)


def soft_sign(a: A) -> A:
    return wrap_elemwise_unary(jnn.soft_sign, a)


def silu(a: A) -> A:
    return wrap_elemwise_unary(jnn.silu, a)


def swish(a: A) -> A:
    return wrap_elemwise_unary(jnn.swish, a)


def log_sigmoid(a: A) -> A:
    return wrap_elemwise_unary(jnn.log_sigmoid, a)


def leaky_relu(a: A) -> A:
    return wrap_elemwise_unary(jnn.leaky_relu, a)


def hard_sigmoid(a: A) -> A:
    return wrap_elemwise_unary(jnn.hard_sigmoid, a)


def hard_silu(a: A) -> A:
    return wrap_elemwise_unary(jnn.hard_silu, a)


def hard_swish(a: A) -> A:
    return wrap_elemwise_unary(jnn.hard_swish, a)


def hard_tanh(a: A) -> A:
    return wrap_elemwise_unary(jnn.hard_tanh, a)


def elu(a: A) -> A:
    return wrap_elemwise_unary(jnn.elu, a)


def celu(a: A) -> A:
    return wrap_elemwise_unary(jnn.celu, a)


def selu(a: A) -> A:
    return wrap_elemwise_unary(jnn.selu, a)


def gelu(a: A, approximate: bool = True) -> A:
    return wrap_elemwise_unary(jnn.gelu, a, approximate=approximate)


def glu(x: NamedArray, axis: Axis) -> NamedArray:
    axis_index = x.axes.index(axis)
    return jnn.glu(x.array, axis_index)


def logsumexp(a: A, axis: Optional[AxisSelection] = None) -> A:
    # TODO: logsumexp indirectly supports where via `b`. we should support it directly
    return wrap_reduction_call(jnn.logsumexp, a, axis=axis, single_axis_only=False, supports_where=False)


# TODO: support where in softmax, etc
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


@functools.wraps(jnn.one_hot)
def one_hot(x: Union[NamedArray, int], class_axis: Axis, *, dtype=jnp.float_) -> NamedArray:
    if isinstance(x, NamedArray):
        array = jnn.one_hot(x.array, num_classes=class_axis.size, dtype=dtype)
        return NamedArray(array, x.axes + (class_axis,))
    else:
        assert isinstance(x, int)
        assert class_axis.size > x >= -class_axis.size

        array = jnp.zeros(class_axis.size, dtype=dtype).at[x].set(1)
        return haliax.named(array, class_axis)


def cross_entropy_loss(
    pred_y: NamedArray,
    Label: AxisSelector,
    target_y: NamedArray,
    reduction: Optional[ReductionFunction] | Unspecified = UNSPECIFIED,
    where: Optional[NamedArray] = None,
    reduction_axis: Optional[AxisSelector] = None,
) -> NamedArray:
    loss, _ = cross_entropy_loss_and_log_normalizers(pred_y, Label, target_y)

    # if target_y isn't some kind of floating point, something is wrong, so warn
    if not jnp.issubdtype(target_y.dtype, jnp.floating):
        warnings.warn(
            f"target_y has dtype {target_y.dtype}, which is not a floating point type. This is probably a mistake."
        )

    if reduction is UNSPECIFIED:
        reduction = haliax.mean

    if reduction is not None:
        loss = reduction(loss, where=where, axis=reduction_axis)

    return loss


def cross_entropy_loss_and_log_normalizers(
    pred_y: NamedArray,
    Label: AxisSelector,
    target_y: NamedArray,
) -> Tuple[NamedArray, NamedArray]:
    """
    Compute the cross entropy loss and log normalizers for a batch of predictions and targets.

    :param pred_y: a NamedArray with the Label axis (and possibly others for e.g. batch and seq) containing the logits
    :param Label: the Label axis
    :param target_y: a NamedArray with the Label axis (and possibly others) containing the targets

    :return: tuple of two named arrays, with "per position" losses and log normalizers
    """
    log_normalizers = hax.nn.logsumexp(pred_y, Label)
    neg_log_normalized = log_normalizers - pred_y

    loss = hax.dot(Label, target_y, neg_log_normalized)

    return loss, log_normalizers


def quick_gelu(x):
    return x * sigmoid(1.702 * x)


__all__ = [
    "attention",
    "relu",
    "relu6",
    "sigmoid",
    "softplus",
    "soft_sign",
    "silu",
    "swish",
    "log_sigmoid",
    "leaky_relu",
    "hard_sigmoid",
    "hard_silu",
    "hard_swish",
    "hard_tanh",
    "elu",
    "celu",
    "selu",
    "gelu",
    "logsumexp",
    "softmax",
    "log_softmax",
    "one_hot",
    "cross_entropy_loss",
    "cross_entropy_loss_and_log_normalizers",
    "quick_gelu",
    "Conv",
    "ConvTranspose",
    "Dropout",
    "dropout",
    "LayerNorm",
    "Linear",
    "Embedding",
    "Stacked",
]
