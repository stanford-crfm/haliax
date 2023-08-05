from typing import Tuple, Union

from jax.lax import Precision


Scalar = Union[float, int]

PrecisionLike = Union[None, str, Precision, Tuple[str, str], Tuple[Precision, Precision]]
