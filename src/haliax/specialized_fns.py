import jax
import jax.numpy as jnp

from .axis import Axis, AxisSelector
from .core import NamedArray


def top_k(arr: NamedArray, axis: AxisSelector, k: int) -> NamedArray:
    pos = arr._lookup_indices(axis)
    if pos is None:
        raise ValueError(f"Axis {axis} not found in {arr}")
    new_array = jnp.moveaxis(arr.array, pos, -1)  # move axis to the last position
    values, _ = jax.lax.top_k(new_array, k=k)
    values = jnp.moveaxis(values, -1, pos)  # move axis back to its original position

    axis = arr.resolve_axis(axis)
    new_axis = Axis(f"top_{k}_of_{axis.name}", size=k)
    updated_axes = arr.axes[:pos] + (new_axis,) + arr.axes[pos + 1 :]
    return NamedArray(values, updated_axes)
