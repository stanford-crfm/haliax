import functools as ft
import typing
import warnings
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

import haliax
from haliax.axis import Axis, AxisSelection, axis_name, eliminate_axes, union_axes
from haliax.core import NamedArray
from haliax.types import PrecisionLike
from haliax.util import ensure_tuple


# deprecated overload
@typing.overload
def dot(
    axis: None,
    *arrays: NamedArray,
    precision: PrecisionLike = None,
    preferred_element_type: Optional[DTypeLike] = None,
) -> jnp.ndarray:
    ...


# deprecated overload
@typing.overload
def dot(
    axis: AxisSelection,
    *arrays: NamedArray,
    precision: PrecisionLike = None,
    preferred_element_type: Optional[DTypeLike] = None,
) -> NamedArray:
    ...


@typing.overload
def dot(
    *arrays: NamedArray,
    axis: AxisSelection,
    precision: PrecisionLike = None,
    preferred_element_type: Optional[DTypeLike] = None,
) -> NamedArray:
    ...


@typing.overload
def dot(
    *arrays: NamedArray,
    axis: None,
    precision: PrecisionLike = None,
    preferred_element_type: Optional[DTypeLike] = None,
) -> jnp.ndarray:
    ...


def dot(
    *arrays,
    precision: PrecisionLike = None,
    preferred_element_type: Optional[DTypeLike] = None,
    **kwargs,
) -> jnp.ndarray | NamedArray:
    """Returns the tensor product of two NamedArrays. The axes `axis` are contracted over,
    and any other axes that are shared between the arrays are batched over. Non-contracted Axes in one
    that are not in the other are preserved.

    Args:
        *arrays (NamedArray): The arrays to contract.
        axis (AxisSelection): The axes to contract over.
        precision (PrecisionLike, optional): The precision to use. Defaults to None. This argument is passed to `jax.numpy.einsum`,
            which in turn passes it to jax.lax.dot_general.
        preferred_element_type (DTypeLike, optional): The preferred element type of the result. Defaults to None.
            This argument is passed to `jax.numpy.einsum`,

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
