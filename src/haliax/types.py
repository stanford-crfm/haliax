from typing import Tuple, Union

import jax.numpy as jnp
from jax.lax import Precision


Scalar = Union[float, int, jnp.ndarray]  # ndarray b/c array(1) is a scalar
IntScalar = Union[int, jnp.ndarray]

PrecisionLike = Union[None, str, Precision, Tuple[str, str], Tuple[Precision, Precision]]
