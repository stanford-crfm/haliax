from typing import Mapping, Optional, Protocol, Sequence, TypeAlias, Union

import jax.numpy as jnp
import numpy as np
from jax.lax import Precision


DType: TypeAlias = np.dtype

try:
    from jax.typing import DTypeLike
except ImportError:
    # Cribbed from jax.typing, for older versions of JAX
    class SupportsDType(Protocol):
        @property
        def dtype(self) -> DType:
            ...

    DTypeLike = Union[
        str,  # like 'float32', 'int32'
        type,  # like np.float32, np.int32, float, int
        np.dtype,  # like np.dtype('float32'), np.dtype('int32')
        SupportsDType,  # like jnp.float32, jnp.int32
    ]


Scalar = Union[float, int, jnp.ndarray]  # ndarray b/c array(1) is a scalar
IntScalar = Union[int, jnp.ndarray]

PrecisionLike = Optional[str | Precision | tuple[str, str] | tuple[Precision, Precision]]


PhysicalAxisSpec = str | Sequence[str]
ResourceMapping = Mapping[str, PhysicalAxisSpec]
"""Mapping from logical axis names to physical axis names"""
