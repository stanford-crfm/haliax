from typing import Optional, TypeVar

import equinox

from haliax.jax_utils import is_jax_array_like
from haliax.types import FilterSpec

from ._src.state_dict import (
    ModuleWithStateDictSerialization,
    StateDict,
    flatten_linear_layers,
    from_state_dict,
    from_torch_compatible_state_dict,
    load_state_dict,
    save_state_dict,
    to_numpy_state_dict,
    to_state_dict,
    unflatten_linear_layers,
    with_prefix,
)


T = TypeVar("T")


def to_torch_compatible_state_dict(
    t: T, *, flatten_linear: bool = True, prefix: Optional[str] = None, filter: FilterSpec = is_jax_array_like
) -> StateDict:
    """
    Convert a tree to a state dict that is compatible with torch-style state dicts.

    This applies [haliax.state_dict.flatten_linear_layers][] followed by [haliax.state_dict.to_state_dict][]

    Args:
        t: The tree to convert
        flatten_linear: Whether to flatten linear layers
        prefix: The prefix to use for the state dict keys
        filter: The filter to use for selecting which nodes to include in the state dict. By default, this includes only
            array-like objects (e.g. JAX and NumPy arrays).
    """
    t = equinox.filter(t, filter)
    if flatten_linear:
        t = flatten_linear_layers(t)
    return to_numpy_state_dict(t, prefix=prefix)


__all__ = [
    "ModuleWithStateDictSerialization",
    "from_torch_compatible_state_dict",
    "load_state_dict",
    "save_state_dict",
    "from_state_dict",
    "flatten_linear_layers",
    "unflatten_linear_layers",
    "with_prefix",
    "to_state_dict",
    "to_numpy_state_dict",
    "StateDict",
    "to_torch_compatible_state_dict",
]
