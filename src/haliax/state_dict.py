from typing import Optional, TypeVar

from ._src.state_dict import (
    ModuleWithStateDictSerialization,
    StateDict,
    apply_prefix,
    flatten_linear_layers,
    from_state_dict,
    from_torch_compatible_state_dict,
    load_state_dict,
    save_state_dict,
    to_numpy_state_dict,
    to_state_dict,
    unflatten_linear_layers,
    update_state_dict,
)


T = TypeVar("T")


def to_torch_compatible_state_dict(t: T, *, flatten_linear: bool = True, prefix: Optional[str] = None) -> StateDict:
    """
    Convert a tree to a state dict that is compatible with torch-style state dicts.

    This applies [haliax.state_dict.flatten_linear_layers][] followed by [haliax.state_dict.to_state_dict][]
    """
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
    "apply_prefix",
    "update_state_dict",
    "to_state_dict",
    "to_numpy_state_dict",
    "StateDict",
    "to_torch_compatible_state_dict",
]
