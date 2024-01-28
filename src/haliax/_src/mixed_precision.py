import enum
from typing import Literal, Optional, TypeAlias, TypeVar

import jmp
from jax import numpy as jnp
from jax.typing import DTypeLike
from strenum import LowercaseStrEnum

import haliax
from haliax._src.resource_env import current_resource_env
from haliax.util import is_jax_or_hax_array_like


T = TypeVar("T")


class SemanticDType(LowercaseStrEnum):
    """Semantic DTypes work with [jmp.Policy][] to specify the precision of computation.
    There are three semantic DTypes: "compute", "parameter", and "output", which correspond to the three kinds of
    dtypes that jmp supports."""

    COMPUTE = enum.auto()
    PARAM = enum.auto()
    OUTPUT = enum.auto()

    def to_dtype(self, mp: Optional[jmp.Policy] = None) -> jnp.dtype:
        if mp is None:
            mp = current_mp_policy()

        match self:
            case SemanticDType.COMPUTE:
                return mp.compute_dtype
            case SemanticDType.PARAM:
                return mp.param_dtype
            case SemanticDType.OUTPUT:
                return mp.output_dtype
            case _:
                raise ValueError(f"Unknown semantic dtype {self}")


DTypeish: TypeAlias = DTypeLike | SemanticDType | Literal["compute", "parameter", "output"]
"""
A type alias for a dtypeish. A dtypeish can be a dtype, a string version of a dtype,
a SemanticDType, or a string version of a SemanticDType.
"""


def cast_floating(x: T, dtype: Optional[DTypeish], mp: Optional[jmp.Policy] = None) -> T:
    """
    Cast x to dtype if dtype is not None. If dtype is a SemanticDType, use the current mixed-precision policy to
    determine the dtype to cast to.
    """
    if dtype is None:
        return x

    dtype = resolve_dtype(dtype, mp)

    return _cast_floating_to(x, dtype)


def resolve_dtype(dtype: DTypeish, mp: Optional[jmp.Policy] = None) -> DTypeLike:
    """
    Resolve a dtypeish to a dtype. If dtype is a SemanticDType, use the current mixed-precision policy to determine
    the dtype to cast to. Otherwise, returns dtype.
    """
    if isinstance(dtype, str):  # SemanticDType is a subclass of str
        # see if it's a semantic dtype
        try:
            if dtype == "parameter":
                dtype = "param"
            semantic_dtype = SemanticDType(dtype.lower())
            dtype = semantic_dtype.to_dtype(mp)
        except KeyError:
            pass
    return dtype


def _cast_floating_to(tree: T, dtype: jnp.dtype) -> T:
    def conditional_cast(x):
        if is_jax_or_hax_array_like(x) and jnp.issubdtype(x.dtype, jnp.floating):
            x = x.astype(dtype)
        return x

    return haliax.tree_util.tree_map(conditional_cast, tree)


def current_mp_policy() -> jmp.Policy:
    return current_resource_env().mp
