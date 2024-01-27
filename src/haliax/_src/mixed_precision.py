import enum
from typing import Literal, Optional, TypeAlias, TypeVar

import jmp
from jax import numpy as jnp
from jax.typing import DTypeLike
from strenum import LowercaseStrEnum

import haliax
from haliax.util import is_jax_or_hax_array_like


T = TypeVar("T")


class SemanticDType(LowercaseStrEnum):
    """Semantic DTypes work with [jmp.Policy][] to specify the precision of computation.
    There are three semantic DTypes: "compute", "parameter", and "output", which correspond to the three kinds of
    dtypes that jmp supports."""

    COMPUTE = enum.auto()
    PARAMETER = enum.auto()
    OUTPUT = enum.auto()

    def to_dtype(self, mp: Optional[jmp.Policy] = None) -> jnp.dtype:
        if mp is None:
            mp = haliax.current_mp_policy()

        match self:
            case SemanticDType.COMPUTE:
                return mp.compute_dtype
            case SemanticDType.PARAMETER:
                return mp.param_dtype
            case SemanticDType.OUTPUT:
                return mp.output_dtype
            case _:
                raise ValueError(f"Unknown semantic dtype {self}")


DTypeish: TypeAlias = DTypeLike | SemanticDType | Literal["compute", "parameter", "output"]


def maybe_cast(x: T, dtype: Optional[DTypeish]) -> T:
    """
    Cast x to dtype if dtype is not None. If dtype is a SemanticDType, use the current mixed-precision policy to
    determine the dtype to cast to.
    """
    if dtype is None:
        return x

    if isinstance(dtype, str):
        # see if it's a semantic dtype
        try:
            semantic_dtype = SemanticDType(dtype.lower())
            dtype = semantic_dtype.to_dtype()
        except KeyError:
            pass

    return _cast_floating_to(x, dtype)


def _cast_floating_to(tree: T, dtype: jnp.dtype) -> T:
    def conditional_cast(x):
        if is_jax_or_hax_array_like(x) and jnp.issubdtype(x.dtype, jnp.floating):
            x = x.astype(dtype)
        return x

    return haliax.tree_util.tree_map(conditional_cast, tree)
