from __future__ import annotations

import inspect
import typing as tp
from dataclasses import dataclass, replace

import jax.numpy as jnp

from .axis import Axis
from .core import NamedArray, NamedArrayAxes, _parse_namedarray_axes


@dataclass(frozen=True)
class DTypeCategory:
    """Represents a dtype category such as ``float`` or ``int``."""

    name: str
    category: tp.Any

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return self.name


def _wrap_namedarray_with_dtype(dtype):
    class DTypeType:
        def __class_getitem__(cls, axes_spec):
            axes = _parse_namedarray_axes(axes_spec)
            axes_with_dtype = replace(axes, dtype=dtype)
            return tp.Annotated[NamedArray, axes_with_dtype]

    return DTypeType


def _wrap_namedarray_with_category(category: DTypeCategory):
    class DTypeType:
        def __class_getitem__(cls, axes_spec):
            axes = _parse_namedarray_axes(axes_spec)
            axes_with_dtype = replace(axes, dtype=category)
            return tp.Annotated[NamedArray, axes_with_dtype]

    return DTypeType


f32 = _wrap_namedarray_with_dtype(jnp.float32)
f64 = _wrap_namedarray_with_dtype(jnp.float64)
f16 = _wrap_namedarray_with_dtype(jnp.float16)
bf16 = _wrap_namedarray_with_dtype(jnp.bfloat16)

i8 = _wrap_namedarray_with_dtype(jnp.int8)
i16 = _wrap_namedarray_with_dtype(jnp.int16)
i32 = _wrap_namedarray_with_dtype(jnp.int32)
i64 = _wrap_namedarray_with_dtype(jnp.int64)

u8 = _wrap_namedarray_with_dtype(jnp.uint8)
u16 = _wrap_namedarray_with_dtype(jnp.uint16)
u32 = _wrap_namedarray_with_dtype(jnp.uint32)
u64 = _wrap_namedarray_with_dtype(jnp.uint64)

bool_ = _wrap_namedarray_with_dtype(jnp.bool_)
complex64 = _wrap_namedarray_with_dtype(jnp.complex64)
complex128 = _wrap_namedarray_with_dtype(jnp.complex128)


Float = _wrap_namedarray_with_category(DTypeCategory("float", jnp.floating))
Complex = _wrap_namedarray_with_category(DTypeCategory("complex", jnp.complexfloating))
Int = _wrap_namedarray_with_category(DTypeCategory("int", jnp.signedinteger))
UInt = _wrap_namedarray_with_category(DTypeCategory("uint", jnp.unsignedinteger))


__all__ = [
    "f32",
    "f64",
    "f16",
    "bf16",
    "i8",
    "i16",
    "i32",
    "i64",
    "u8",
    "u16",
    "u32",
    "u64",
    "bool_",
    "complex64",
    "complex128",
    "Float",
    "Complex",
    "Int",
    "UInt",
    "check_axes",
]


def _candidate_axes(arr: NamedArray, spec: NamedArrayAxes) -> dict[str, set[str]]:
    """Return possible axis names for each generic appearing in ``spec``."""

    generics = set(spec.generics)
    result: dict[str, set[str]] = {}

    if spec.ordered:
        axes = arr.axes
        if not spec.subset:
            if len(axes) != len(spec.before):
                raise ValueError("Axis count mismatch")
            axis_iter = list(zip(spec.before, axes))
        else:
            if len(axes) < len(spec.before) + len(spec.after):
                raise ValueError("Axis count mismatch")
            axis_iter = list(zip(spec.before, axes[: len(spec.before)]))
            axis_iter += list(zip(spec.after, axes[-len(spec.after) :]))
        for name, axis in axis_iter:
            if name in generics:
                result.setdefault(name, set()).add(axis.name)
            elif name != axis.name:
                raise ValueError(f"Expected axis {name} but got {axis.name}")
    else:
        axes_list = list(arr.axes)
        explicit = [n for n in spec.before if n not in generics]
        for n in explicit:
            for i, ax in enumerate(axes_list):
                if ax.name == n:
                    axes_list.pop(i)
                    break
            else:
                raise ValueError(f"Axis {n} missing")

        leftovers = {ax.name for ax in axes_list}
        for g in spec.generics:
            result.setdefault(g, set()).update(leftovers)

    return result


def _axis_lookup(arrays: list[NamedArray]) -> dict[str, Axis]:
    lookup: dict[str, Axis] = {}
    for arr in arrays:
        for ax in arr.axes:
            lookup.setdefault(ax.name, ax)
    return lookup


def check_axes(*arrays: NamedArray) -> dict[str, Axis]:
    """Resolve generic axes across multiple arrays based on the caller's type annotations."""

    frame = inspect.currentframe()
    if frame is None or frame.f_back is None:
        raise RuntimeError("check_axes must be called from within a function")
    frame = frame.f_back
    func = frame.f_globals.get(frame.f_code.co_name)
    if func is None:
        raise RuntimeError("Could not determine calling function")

    hints = tp.get_type_hints(func, include_extras=True)
    param_names = list(inspect.signature(func).parameters.keys())

    specs: list[tuple[NamedArray, NamedArrayAxes]] = []
    for arr, name in zip(arrays, param_names):
        ann = hints.get(name)
        if ann is None:
            continue
        spec = _parse_namedarray_axes(ann)
        specs.append((arr, spec))

    candidate_map: dict[str, list[set[str]]] = {}
    for arr, spec in specs:
        cands = _candidate_axes(arr, spec)
        for g, names in cands.items():
            candidate_map.setdefault(g, []).append(names)

    lookup = _axis_lookup([arr for arr, _ in specs])
    resolved: dict[str, Axis] = {}

    for g, sets_list in candidate_map.items():
        intersection = set.intersection(*sets_list)
        if intersection:
            axis_name = next(iter(intersection))
        elif len(sets_list) == 1:
            options = sets_list[0]
            if g.lower() in options:
                axis_name = g.lower()
            elif len(options) == 1:
                axis_name = next(iter(options))
            else:
                raise ValueError(f"Ambiguous generic axis {g}")
        else:
            raise ValueError(f"Could not resolve generic axis {g}")

        axis = lookup.get(axis_name)
        if axis is None:
            raise ValueError(f"Axis {axis_name} not found")

        # verify sizes match for all arrays using this generic
        for arr, spec in specs:
            if g in spec.generics:
                for ax in arr.axes:
                    if ax.name == axis_name:
                        if ax.size != axis.size:
                            raise ValueError(f"Axis size mismatch for generic {g}: {axis.size} vs {ax.size}")
                        break
                else:
                    raise ValueError(f"Axis {axis_name} missing for generic {g}")

        resolved[g] = axis

    return resolved
