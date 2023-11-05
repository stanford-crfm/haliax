# Pooling operations, inspired by Flax
from functools import reduce
from typing import Callable, Literal, Optional

import jax

import haliax

from .. import Scalar
from ..axis import AxisSpec, unsize_axes
from ..core import NamedArray
from ..util import ensure_tuple


Padding = Literal["SAME", "VALID"] | int | tuple[tuple[int, int], ...]


def pool(
    Window: AxisSpec,
    inputs: NamedArray,
    init: Scalar,
    reduce_fn: Callable[[Scalar, Scalar], Scalar],
    strides: Optional[tuple[int, ...]] = None,
    padding: Padding = "VALID",
) -> NamedArray:
    """
    General function for pooling. Broadly based on the Flax implementation.

    Pooling functions are implemented using the ReduceWindow XLA op.

    Notes:
        JAX only implements pooling for a few specific reductions (min, max, sum).

    Args:
        Window: the size of the window to pool over
        inputs: input data with dimensions (batch, window dims..., features).
        init: the initial value for the reduction
        reduce_fn: a reduce function of the form `(T, T) -> T`.
        strides: a sequence of `n` integers, representing the inter-window
          strides (default: `(1, ..., 1)`).
        padding: either the string `'SAME'`, the string `'VALID'`, or a sequence
          of `n` `(low, high)` integer pairs that give the padding to apply before
          and after each spatial dimension, or an integer to pad all dimensions
    Returns:
      The output of the reduction for each window slice.
    """
    Window = ensure_tuple(Window)

    reduce_fn = _patch_up_reduce_fn(reduce_fn)

    window_map = {w.name: w.size for w in Window}
    dims = []
    for ax in inputs.axes:
        if ax.name in window_map:
            dims.append(window_map[ax.name])
        else:
            dims.append(1)

    if strides is not None and len(Window) != len(strides):
        raise ValueError(f"len(Window) ({len(Window)}) != len(strides) ({len(strides)})")

    if strides is not None:
        strides = ensure_tuple(strides)
        stride_map = {w.name: s for w, s in zip(Window, strides)}
        strides_out = []
        for ax in inputs.axes:
            if ax.name in stride_map:
                strides_out.append(stride_map[ax.name])
            else:
                strides_out.append(1)

        strides = tuple(strides_out)
        del strides_out
    else:
        strides = (1,) * len(dims)

    if isinstance(padding, int):
        pout = []
        for ax in inputs.axes:
            if ax.name in window_map:
                pout.append((padding, padding))
            else:
                pout.append((0, 0))

        padding = tuple(pout)

    elif not isinstance(padding, str):
        padding = tuple(map(tuple, padding))  # type: ignore
        if len(padding) != len(Window):
            raise ValueError(f"len(padding) ({len(padding)}) != len(Window) ({len(Window)})")
        if not all([len(x) == 2 for x in padding]):
            raise ValueError(f"each entry in padding must be length 2, got {padding}")

        padding_map = {w.name: p for w, p in zip(Window, padding)}
        padding_out = []
        for ax in inputs.axes:
            if ax.name in padding_map:
                padding_out.append(padding_map[ax.name])
            else:
                padding_out.append((0, 0))

        padding = tuple(padding)

    # TODO: Flax suggests there must be a batch dim? Is this true?

    assert inputs.ndim == len(dims), f"len({inputs.shape}) != len({dims})"
    y = jax.lax.reduce_window(inputs.array, init, reduce_fn, dims, strides, padding)

    out_axes = unsize_axes(inputs.axes, Window)

    return haliax.named(y, out_axes)


def _patch_up_reduce_fn(reduce_fn):
    if reduce_fn is haliax.max:
        reduce_fn = jax.lax.max
    elif reduce_fn is haliax.min:
        reduce_fn = jax.lax.min
    elif reduce_fn is haliax.sum:
        reduce_fn = jax.lax.add
    elif reduce_fn is haliax.prod:
        reduce_fn = jax.lax.mul

    return reduce_fn


def max_pool(
    Window: AxisSpec, inputs: NamedArray, strides: Optional[tuple[int, ...]] = None, padding: Padding = "VALID"
) -> NamedArray:
    """
    Max pooling.

    Args:
        Window: the size of the window to pool over
        inputs: input data with dimensions (batch, window dims..., features).
        strides: a sequence of `n` integers, representing the inter-window
          strides (default: `(1, ..., 1)`).
        padding: either the string `'SAME'`, the string `'VALID'`, or a sequence
          of `n` `(low, high)` integer pairs that give the padding to apply before
          and after each spatial dimension.
    Returns:
      The maximum value in each window slice.
    """
    return pool(Window, inputs, -float("inf"), jax.lax.max, strides, padding)


def min_pool(
    Window: AxisSpec, inputs: NamedArray, strides: Optional[tuple[int, ...]] = None, padding: Padding = "VALID"
) -> NamedArray:
    """
    Min pooling.

    Args:
        Window: the size of the window to pool over
        inputs: input data with dimensions (batch, window dims..., features).
        strides: a sequence of `n` integers, representing the inter-window
        strides (default: `(1, ..., 1)`).
        padding: either the string `'SAME'`, the string `'VALID'`, or a sequence
        of `n` `(low, high)` integer pairs that give the padding to apply before
        and after each spatial dimension.
    Returns:
    The minimum value in each window slice.
    """
    return pool(Window, inputs, float("inf"), jax.lax.min, strides, padding)


def mean_pool(
    Window: AxisSpec, inputs: NamedArray, strides: Optional[tuple[int, ...]] = None, padding: Padding = "VALID"
):
    """
    Mean pooling.

    Args:
        Window: the size of the window to pool over
        inputs: input data with dimensions (batch, window dims..., features).
        strides: a sequence of `n` integers, representing the inter-window
          strides (default: `(1, ..., 1)`).
        padding: either the string `'SAME'`, the string `'VALID'`, or a sequence
          of `n` `(low, high)` integer pairs that give the padding to apply before
          and after each spatial dimension.
    Returns:
      The mean value in each window slice.
    """
    tots = pool(Window, inputs, 0, jax.lax.add, strides, padding)
    Window = ensure_tuple(Window)
    window_size = reduce(lambda x, y: x * y, [w.size for w in Window])
    return tots / window_size
