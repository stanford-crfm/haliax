from __future__ import annotations

import typing as tp
from dataclasses import dataclass, replace

import jax.numpy as jnp

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
]
