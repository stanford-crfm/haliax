import functools as ft
import typing
import warnings
from types import EllipsisType
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

import haliax
from haliax._src.util import index_where
from haliax.axis import (
    Axis,
    AxisSelection,
    AxisSelector,
    axis_name,
    axis_spec_to_shape_dict,
    eliminate_axes,
    union_axes,
)
from haliax.core import NamedArray
from haliax.types import PrecisionLike
from haliax.util import ensure_tuple


PartialAxisSpec = tuple[EllipsisType | AxisSelector, ...]


# deprecated overload
@typing.overload
def dot(
    axis: None,
    *arrays: NamedArray,
    precision: PrecisionLike = None,
    preferred_element_type: Optional[DTypeLike] = None,
    out_axes: Optional[PartialAxisSpec] = ...,
) -> jnp.ndarray:
    ...


# deprecated overload
@typing.overload
def dot(
    axis: AxisSelection,
    *arrays: NamedArray,
    precision: PrecisionLike = None,
    preferred_element_type: Optional[DTypeLike] = None,
    out_axes: Optional[PartialAxisSpec] = ...,
) -> NamedArray:
    ...


@typing.overload
def dot(
    *arrays: NamedArray,
    axis: AxisSelection,
    precision: PrecisionLike = None,
    preferred_element_type: Optional[DTypeLike] = None,
    out_axes: Optional[PartialAxisSpec] = ...,
) -> NamedArray:
    ...


@typing.overload
def dot(
    *arrays: NamedArray,
    axis: None,
    precision: PrecisionLike = None,
    preferred_element_type: Optional[DTypeLike] = None,
    out_axes: Optional[PartialAxisSpec] = ...,
) -> jnp.ndarray:
    ...


def dot(
    *arrays,
    precision: PrecisionLike = None,
    preferred_element_type: Optional[DTypeLike] = None,
    out_axes: Optional[PartialAxisSpec] = None,
    **kwargs,
) -> jnp.ndarray | NamedArray:
    """Returns the tensor product of two NamedArrays. The axes `axis` are contracted over,
    and any other axes that are shared between the arrays are batched over. Non-contracted Axes in one
    that are not in the other are preserved.

    Note that if `axis` is None, the result will be a scalar, not a NamedArray. The semantics of `axis=None` are
    similar to e.g. how `sum` and other reduction functions work in numpy. If `axis=()`, then the result will be
    an "outer product" of the arrays, i.e. a tensor with shape equal to the concatenation of the shapes of the arrays.

    By default, the order of output axes is determined by the order of the input axes, such that each output axis
    occurs in the same order as it first occurs in the concatenation of the input axes.

    If `out_axes` is provided, the output will be transposed to match the provided axes. `out_axes` may be a partial
    specification of the output axes (using ellipses), in which case the output will be rearranged to be consistent
    with the partial specification. For example, if `out_axes=(..., Height, Width)` and the output axes are
    `(Width, Height, Depth)`, the output will be transposed to `(Depth, Height, Width)`. Multiple ellipses
    are supported, in which case axes will be inserted according to a greedy heuristic that prefers to place
    unconstrained axes as soon as all prior axes in the "natural" order are covered.

    Args:
        *arrays (NamedArray): The arrays to contract.
        axis (AxisSelection): The axes to contract over.
        precision (PrecisionLike, optional): The precision to use. Defaults to None. This argument is passed to `jax.numpy.einsum`,
            which in turn passes it to jax.lax.dot_general.
        preferred_element_type (DTypeLike, optional): The preferred element type of the result. Defaults to None.
            This argument is passed to `jax.numpy.einsum`.
        out_axes (Optional[PartialAxisSpec], optional): a potentially partial specification of the output axes.
            If provided, the output will be transposed to match the provided axes. Defaults to None.


    Returns:
        NamedArray: The result of the contraction.
    """
    if len(arrays) == 0:
        raise ValueError("Must provide at least one array to dot")

    if "axis" in kwargs:
        axis = kwargs["axis"]
    else:
        axis = arrays[0]
        arrays = arrays[1:]
        if isinstance(axis, NamedArray):
            raise ValueError("Must provide an axis to dot")

        warnings.warn("Axis has been changed to a keyword argument. Please update your code.", DeprecationWarning)

    _ensure_no_mismatched_axes(*arrays)

    all_axes: Tuple[Axis, ...] = ft.reduce(union_axes, (a.axes for a in arrays), ())  # type: ignore
    output_axes: Tuple[Axis, ...]
    if axis is None:
        # we want to contract over all the axes
        output_axes = ()
    else:
        output_axes = eliminate_axes(all_axes, axis)

    if out_axes is not None:
        output_axes = rearrange_to_fit_order(out_axes, output_axes)

    array_specs = []

    next_index = 0
    axis_mappings: Dict[str, int] = {}

    for a in arrays:
        spec = ""
        for ax in a.axes:
            if ax.name in axis_mappings:
                spec += f"{axis_mappings[ax.name]} "
            else:
                axis_mappings[ax.name] = next_index
                spec += f"{next_index} "
                next_index += 1

        array_specs.append(spec)

    # now compute the output axes:
    output_spec = " ".join(str(axis_mappings[ax.name]) for ax in output_axes)

    # get a name for jax so it's easier to interpret logs
    if axis is None:
        jax_str = f"contract {', '.join(axis_name(ax) for ax in all_axes)} -> <scalar>"
    else:
        axis = ensure_tuple(axis)
        jax_str = f"contract {', '.join(axis_name(ax) for ax in axis)} -> {', '.join(a.name for a in output_axes)}"

    with jax.named_scope(jax_str):
        output = jnp.einsum(
            ", ".join(array_specs) + "-> " + output_spec,
            *[a.array for a in arrays],
            precision=precision,
            preferred_element_type=preferred_element_type,
        )

    if axis is None:
        assert output.ndim == 0
        return output
    else:
        out = NamedArray(output, output_axes)
        return haliax.auto_sharded(out)


def _ensure_no_mismatched_axes(*arrays: NamedArray):
    """Ensure that all the arrays have no axes with the same name but different sizes"""
    if len(arrays) <= 1:
        return

    known_sizes: dict[str, int] = {}
    for a in arrays:
        for ax in a.axes:
            if ax.name in known_sizes:
                if known_sizes[ax.name] != ax.size:
                    raise ValueError(f"Axis {ax.name} has multiple sizes: {known_sizes[ax.name]} and {ax.size}")
            else:
                known_sizes[ax.name] = ax.size


Ax = typing.TypeVar("Ax", AxisSelector, Axis)


def rearrange_to_fit_order(
    partial_order: tuple[AxisSelector | EllipsisType, ...], axes: tuple[Ax, ...]
) -> tuple[Ax, ...]:
    """Rearrange the axes to fit the provided partial order.
    Uses a greedy algorithm that tries to keep elements in roughly the same order they came in
     (subject to the partial order), but moves them to the earliest slot that is after all prior axes
     in the original order.
     The exact behavior of this function is not guaranteed to be stable, but it should be stable
     for most reasonable use cases. If you really need a specific order, you should provide a full
     order instead of a partial order.
    """

    if partial_order == (Ellipsis,):
        return axes

    spec = axis_spec_to_shape_dict(axes)

    def as_axis(ax_name: str) -> Ax:
        if spec[ax_name] is None:
            return ax_name  # type: ignore
        else:
            return Axis(ax_name, spec[ax_name])  # type: ignore

    if Ellipsis not in partial_order:
        pa: tuple[AxisSelector, ...] = partial_order  # type: ignore
        if set(axis_name(a) for a in pa) != set(spec.keys()) or len(pa) != len(spec.keys()):
            raise ValueError("Partial order must be a permutation of the axes if no ellipsis is provided")

        # reorder axes to match partial order
        return tuple(as_axis(axis_name(name)) for name in pa)

    partial_order_names = [axis_name(s) for s in partial_order if s is not ...]

    uncovered_ordered_elements = set(partial_order_names)

    if len(partial_order_names) != len(uncovered_ordered_elements):
        raise ValueError("Partial order must not contain duplicate elements")

    # replace ... with [], which is where we'll put the remaining axes

    out_order = [[axis_name(a)] if a is not ... else [] for a in partial_order]

    # now we'll fill in the ordered elements
    target_pos = index_where(lambda x: x == [], out_order)

    for ax in axes:
        ax_name = axis_name(ax)
        if ax_name in uncovered_ordered_elements:
            uncovered_ordered_elements.remove(ax_name)
            # already in the right place
            # update target_pos to come after this if possible
            try:
                this_pos = index_where(lambda x: ax_name in x, out_order)
                # find first empty slot after this_pos. prefer not to go backwards
                this_pos = max(this_pos + 1, target_pos)
                target_pos = index_where(lambda x: x == [], out_order, start=this_pos)
            except ValueError:
                # leave it where it is
                pass
        elif ax_name in partial_order_names:
            raise ValueError(f"Axis {ax_name} appears multiple times in the partial order")
        else:
            # this can appear in any ... slot. our heuristic is to put it in the first
            # slot that comes after the most recently seen ordered element
            out_order[target_pos].append(ax_name)

    if len(uncovered_ordered_elements) > 0:
        raise ValueError(f"The following axes are not present in output: {' '.join(uncovered_ordered_elements)}")

    # now we have a list of lists of axis names. we need to flatten it and convert to axes
    return tuple(as_axis(name) for name in sum(out_order, []))
