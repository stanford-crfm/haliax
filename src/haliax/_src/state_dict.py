# Module to support torch-style "state dict" serialization
# via safetensors
import dataclasses
from typing import Any, Optional, Sequence, TYPE_CHECKING, TypeVar, overload

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.tree_util import SequenceKey, DictKey, FlattenedIndexKey, GetAttrKey
from jaxtyping import PyTree

import haliax.partitioning as partitioning
from ..core import NamedArray, named
from ..jax_utils import is_jax_array_like

try:
    from . import safetensors
except ImportError:
    safetensors = None



StateDict = dict[str, Any]
Mod = TypeVar("Mod", bound=eqx.Module)


def apply_prefix(prefix: Optional[str], leaf: Optional[str]) -> Optional[str]:
    if prefix is None:
        return leaf
    elif leaf is None:
        return prefix
    else:
        return f"{prefix}.{leaf}"


class StateDictSerializationMixin:
    """An eqx.Module that can be serialized to a torch-style state dict."""

    def to_state_dict(self, prefix: Optional[str] = None) -> StateDict:
        return state_dict_to_jax_tree(self, prefix)

    def from_state_dict(self: Mod, state_dict: StateDict, prefix: Optional[str] = None) -> Mod:
        return default_eqx_module_from_state_dict(self, state_dict, prefix)

    def update_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        return default_update_state_dict_with_eqx_module(state_dict, self, prefix)

    def _state_dict_key_map(self) -> dict[str, Optional[str]]:
        """Returns a dict mapping eqx.Module keys to torch keys that need to be renamed for serialization"""
        return {}

def jax_tree_from_state_dict(tree: PyTree, state_dict: StateDict, prefix: Optional[str] = None) -> PyTree:
    # TODO: assert compatibility of old and new values (type, shape, etc.)
    if isinstance(tree, eqx.Module):
        if hasattr(tree, "from_state_dict"):
            return tree.from_state_dict(state_dict, prefix)
        else:
            return default_eqx_module_from_state_dict(tree, state_dict, prefix)
    elif isinstance(tree, list):
        return [
            jax_tree_from_state_dict(item, state_dict, apply_prefix(prefix, str(i))) for i, item in enumerate(tree)
        ]
    elif isinstance(tree, dict):
        return {k: jax_tree_from_state_dict(v, state_dict, prefix=apply_prefix(prefix, k)) for k, v in tree.items()}
    elif isinstance(tree, NamedArray):
        if prefix is None:
            raise ValueError("Cannot extract a leaf value from a torch dict without a prefix")

        array = state_dict[prefix]

        if isinstance(array, np.ndarray):
            mesh = partitioning._get_mesh()
            if mesh.devices.size > 1:  # this happens with the default mesh
                pspec = partitioning.pspec_for_axis(tree.axes)
                sharding = jax.sharding.NamedSharding(mesh, pspec)
                array = jax.make_array_from_callback(tree.array.shape, sharding, lambda indices: array[indices])
            else:
                array = jnp.array(array)
            array = named(array, tree.axes)
        else:
            array = named(array, tree.axes)
            array = partitioning.auto_sharded(array)

        return array
    elif is_jax_array_like(tree):
        if prefix is None:
            raise ValueError("Cannot extract a leaf value from a state dict without a prefix")
        # TODO: add "strict" flag so we can return None in cases where it's just missing
        return jnp.array(state_dict[prefix])
    else:
        if prefix is None:
            return tree
        return state_dict.get(prefix, tree)


def update_state_dict_with_jax_tree(tree: PyTree, state_dict: StateDict, prefix: Optional[str] = None) -> None:
    if isinstance(tree, eqx.Module):
        if hasattr(tree, "update_state_dict"):
            tree.update_state_dict(state_dict, prefix)
        else:
            default_update_state_dict_with_eqx_module(state_dict, tree, prefix)
    elif isinstance(tree, list):
        for i, item in enumerate(tree):
            update_state_dict_with_jax_tree(item, state_dict, prefix=apply_prefix(prefix, str(i)))
    elif isinstance(tree, dict):
        for k, v in tree.items():
            update_state_dict_with_jax_tree(v, state_dict, prefix=apply_prefix(prefix, k))
    elif isinstance(tree, NamedArray):
        assert prefix is not None
        state_dict[prefix] = tree.array
    elif is_jax_array_like(tree):
        if prefix is not None:
            if tree is not None:
                state_dict[prefix] = tree  # type: ignore
        else:
            raise ValueError("Cannot update torch dict with a leaf value.")
    else:
        pass


def state_dict_to_jax_tree(tree: PyTree, prefix: Optional[str] = None) -> StateDict:
    state_dict: StateDict = {}
    update_state_dict_with_jax_tree(tree, state_dict, prefix)
    return state_dict


def default_eqx_module_from_state_dict(mod: Mod, state_dict: StateDict, prefix: Optional[str] = None) -> Mod:
    # TODO: move into BlockSeq
    try:
        from haliax.nn.scan import BlockSeq

        if isinstance(mod, BlockSeq):
            return block_seq_from_state_dict(mod, state_dict, prefix)
    except ImportError:
        pass

    key_map: Dict[str, Optional[str]] = getattr(mod, "_state_dict_key_map", lambda: {})()  # type: ignore
    names = []
    values = []
    for field in dataclasses.fields(mod):
        if field.metadata.get("static", False):
            continue
        key = key_map.get(field.name, field.name)
        value = getattr(mod, field.name)
        # TODO: might want to add a flag that allows missing keys?
        new = jax_tree_from_state_dict(value, state_dict, apply_prefix(prefix, key))
        # Do not try to update parameters that are never defined
        if value is None and new is None:
            continue
        names.append(field.name)
        values.append(new)
    return eqx.tree_at(lambda m: [getattr(m, name) for name in names], mod, values)


def default_state_dict_from_eqx_module(mod: eqx.Module, prefix: Optional[str] = None) -> StateDict:
    state_dict: StateDict = {}
    default_update_state_dict_with_eqx_module(state_dict, mod, prefix)
    return state_dict


def default_update_state_dict_with_eqx_module(
    state_dict: StateDict, mod: eqx.Module, prefix: Optional[str] = None
) -> StateDict:
    try:
        from haliax.nn.scan import BlockSeq

        if isinstance(mod, BlockSeq):
            return update_block_seq_state_dict(state_dict, mod, prefix)
    except ImportError:
        pass

    key_map: Dict[str, Optional[str]] = getattr(mod, "_state_dict_key_map", lambda: {})()  # type: ignore
    for field in dataclasses.fields(mod):
        if field.metadata.get("static", False):
            continue
        key = key_map.get(field.name, field.name)
        value = getattr(mod, field.name)
        update_state_dict_with_jax_tree(value, state_dict, apply_prefix(prefix, key))
    return state_dict




def format_path_for_state_dict(prefix: Optional[str], path: Sequence) -> str:
    res =  "".join(_format_key_path_element(path_elem) for path_elem in path)
    # res will have a .
    if prefix is not None:
        res = f"{prefix}{res}"
    elif res.startswith("."):
        res = res[1:]

    return res


# Torch compatible KeyPath formatting. Torch just always uses .
def _format_key_path_element(path_elem) -> str:
    match path_elem:
        case SequenceKey(idx):
            return f".{idx}"
        case DictKey(key):
            return f".{key}"
        case GetAttrKey():
            return str(path_elem)
        case FlattenedIndexKey(idx):
            return f".{idx}"
        case _:
            # The convention in JAX is to append the separator in the element itself
            # so we expect it to have
            path_elem = str(path_elem)
            if path_elem.startswith("."):
                return path_elem
            else:
                return f".{path_elem}"