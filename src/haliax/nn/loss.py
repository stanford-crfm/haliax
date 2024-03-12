import typing
import warnings
from typing import Optional

from jax import numpy as jnp

import haliax as hax
from haliax.axis import AxisSelection, AxisSelector
from haliax.core import NamedArray
from haliax.util import UNSPECIFIED, Unspecified
from haliax.wrap import ReductionFunction


@typing.overload
def cross_entropy_loss(
    logits: NamedArray,
    Label: AxisSelector,
    targets: NamedArray,
    reduction: Optional[ReductionFunction] | Unspecified = UNSPECIFIED,
    where: Optional[NamedArray] = None,
    reduction_axis: None = None,
) -> jnp.ndarray | NamedArray:
    ...


@typing.overload
def cross_entropy_loss(
    logits: NamedArray,
    Label: AxisSelector,
    targets: NamedArray,
    reduction: Optional[ReductionFunction] | Unspecified = UNSPECIFIED,
    where: Optional[NamedArray] = None,
    reduction_axis: AxisSelection = ...,
) -> NamedArray:
    ...


def cross_entropy_loss(
    logits: NamedArray,
    Label: AxisSelector,
    targets: NamedArray,
    reduction: Optional[ReductionFunction] | Unspecified = UNSPECIFIED,
    where: Optional[NamedArray] = None,
    reduction_axis: Optional[AxisSelection] = None,
) -> jnp.ndarray | NamedArray:
    loss = cross_entropy(logits, Label, targets)

    # if target_y isn't some kind of floating point, something is wrong, so warn
    if not jnp.issubdtype(targets.dtype, jnp.floating):
        warnings.warn(
            f"target_y has dtype {targets.dtype}, which is not a floating point type. This is probably a mistake."
        )

    loss = maybe_reduce_loss(loss, reduction, reduction_axis, where)

    return loss


@typing.overload
def binary_cross_entropy_loss(
    logits: NamedArray,
    targets: NamedArray,
    reduction: Optional[ReductionFunction] | Unspecified = UNSPECIFIED,
    where: Optional[NamedArray] = None,
    reduction_axis: None = None,
) -> jnp.ndarray | NamedArray:
    ...


@typing.overload
def binary_cross_entropy_loss(
    logits: NamedArray,
    targets: NamedArray,
    reduction: Optional[ReductionFunction] | Unspecified = UNSPECIFIED,
    where: Optional[NamedArray] = None,
    reduction_axis: AxisSelection = ...,
) -> NamedArray:
    ...


def binary_cross_entropy_loss(
    logits: NamedArray,
    targets: NamedArray,
    reduction: Optional[ReductionFunction] | Unspecified = UNSPECIFIED,
    where: Optional[NamedArray] = None,
    reduction_axis: Optional[AxisSelection] = None,
) -> jnp.ndarray | NamedArray:
    loss = binary_cross_entropy(logits, targets)
    loss = maybe_reduce_loss(loss, reduction, reduction_axis, where)
    return loss


def maybe_reduce_loss(
    arr,
    reduction: Optional[ReductionFunction] | Unspecified,
    reduction_axis: Optional[AxisSelection],
    where: Optional[NamedArray],
):
    if reduction is not None and reduction_axis != ():
        if reduction is UNSPECIFIED:
            reduction = hax.mean
        arr = reduction(arr, where=where, axis=reduction_axis)
    elif where is not None:
        arr = hax.where(where, arr, 0)
    return arr


def cross_entropy_loss_and_log_normalizers(
    pred_y: NamedArray,
    Label: AxisSelector,
    target_y: NamedArray,
) -> tuple[NamedArray, NamedArray]:
    """
    Compute the cross entropy loss and log normalizers for a batch of predictions and targets.

    Args:
         pred_y: a NamedArray with the Label axis (and possibly others for e.g. batch and seq) containing the logits
         Label: the Label axis
         target_y: a NamedArray with the Label axis (and possibly others) containing the targets

    Returns:
        a tuple of two named arrays:
        - the "per position" losses
        - the log normalizers

    """
    log_normalizers = hax.nn.logsumexp(pred_y, Label)
    neg_log_normalized = log_normalizers - pred_y

    loss = hax.dot(target_y, neg_log_normalized, axis=Label)

    return loss, log_normalizers


def cross_entropy(
    logits: NamedArray,
    Label: AxisSelector,
    targets: NamedArray,
) -> NamedArray:
    """
    Compute the cross entropy loss for a batch of predictions and targets.

    Args:
       logits: a NamedArray with the Label axis (and possibly others for e.g. batch and seq) containing the logits
       Label: the Label axis
       targets: a NamedArray with the Label axis (and possibly others) containing the targets

    Returns:
       a named array with the "per position" losses
    """
    return cross_entropy_loss_and_log_normalizers(logits, Label, targets)[0]


def binary_cross_entropy(logits: NamedArray, targets: NamedArray):
    """
    Compute the binary cross entropy loss for a batch of predictions and targets.
    Returns the loss for each position in the batch.

    This function is agnostic to all dimensions.

    Args:
        logits: NamedArray with the same shape as targets.
        targets: NamedArray with the same shape as logits.

    Returns:
        NamedArray with the same shape as logits and targets

    """
    log_p = hax.nn.log_sigmoid(logits)
    log_not_p = hax.nn.log_sigmoid(-logits)  # == log(1-sigmoid(x))
    targets = targets.astype(logits.dtype)
    loss = -targets * log_p - (1.0 - targets) * log_not_p
    return loss
