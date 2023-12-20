import functools
import warnings
from typing import Optional, Tuple, Union

import jax.nn as jnn
import jax.numpy as jnp

import haliax
import haliax as hax
import haliax.nn.activations
import haliax.nn.attention as attention
import haliax.nn.normalization

from ..axis import Axis, AxisSelector
from ..core import NamedArray
from ..util import UNSPECIFIED, Unspecified
from ..wrap import ReductionFunction
from .activations import (
    celu,
    elu,
    gelu,
    glu,
    hard_sigmoid,
    hard_silu,
    hard_swish,
    hard_tanh,
    leaky_relu,
    log_sigmoid,
    quick_gelu,
    relu,
    relu6,
    selu,
    sigmoid,
    silu,
    soft_sign,
    softplus,
    swish,
)
from .conv import Conv, ConvTranspose
from .dropout import Dropout, dropout
from .embedding import Embedding
from .linear import Linear
from .mlp import MLP
from .normalization import LayerNorm, log_softmax, logsumexp, softmax, standardize
from .pool import max_pool, mean_pool, min_pool
from .scan import Stacked


# TODO: support where in softmax, etc


@functools.wraps(jnn.one_hot)
def one_hot(x: Union[NamedArray, int], class_axis: Axis, *, dtype=None) -> NamedArray:
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

    if reduction is not None:
        if reduction is UNSPECIFIED:
            reduction = haliax.mean
        loss = reduction(loss, where=where, axis=reduction_axis)
    elif where is not None:
        loss = hax.where(where, loss, 0)

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
    log_normalizers = haliax.nn.normalization.logsumexp(pred_y, Label)
    neg_log_normalized = log_normalizers - pred_y

    loss = hax.dot(Label, target_y, neg_log_normalized)

    return loss, log_normalizers


__all__ = [
    "attention",
    "one_hot",
    "cross_entropy_loss",
    "cross_entropy_loss_and_log_normalizers",
    "Conv",
    "ConvTranspose",
    "Dropout",
    "dropout",
    "LayerNorm",
    "Linear",
    "Embedding",
    "Stacked",
    "relu",
    "gelu",
    "quick_gelu",
    "glu",
    "relu6",
    "sigmoid",
    "soft_sign",
    "softplus",
    "swish",
    "silu",
    "log_sigmoid",
    "leaky_relu",
    "hard_sigmoid",
    "hard_silu",
    "hard_swish",
    "hard_tanh",
    "logsumexp",
    "softmax",
    "log_softmax",
    "standardize",
    "elu",
    "celu",
    "selu",
    "max_pool",
    "mean_pool",
    "min_pool",
]
