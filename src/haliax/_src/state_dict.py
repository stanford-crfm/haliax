# Module to support torch-style "state dict" serialization via safetensors
import dataclasses
import typing
from typing import Any, Optional, Sequence, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import ShapeDtypeStruct
from jax.experimental.multihost_utils import sync_global_devices
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.tree_util import DictKey, FlattenedIndexKey, GetAttrKey, SequenceKey
from jaxtyping import PyTree

import haliax.partitioning as partitioning
from haliax._src.util import index_where
from haliax.core import NamedArray, named
from haliax.jax_utils import is_jax_array_like, is_scalarish


try:
    import safetensors
except ImportError:
    safetensors = None


StateDict = dict[str, Any]
Mod = TypeVar("Mod", bound=eqx.Module)
T = TypeVar("T")


def from_torch_compatible_state_dict(
    t: T, state_dict: StateDict, *, unflatten_linear: bool = True, prefix: Optional[str] = None
) -> T:
    """
    Convert a state dict to a tree that is compatible with the structure of `t`.

    This applies [haliax.state_dict.from_state_dict][] followed by [haliax.state_dict.unflatten_linear_layers][].
    """
    if unflatten_linear:
        t = _flatten_to_unflatten(t, state_dict, prefix)
    else:
        t = from_state_dict(t, state_dict, prefix=prefix)

    return t


def _flatten_to_unflatten(t, state_dict, prefix):
    """
    Flatten the torch compatible state_dict before loading into t, and then recover the unflattened layers.
    """
    # typically, `t` is a bunch of ShapeDtypeStructs, which can't be transposed etc. so we instead have to zeros()
    # into real arrays (that aren't actually real b/c this is inside a jit)
    def _dt_struct_to_array(struct):
        if not isinstance(struct, ShapeDtypeStruct):
            return struct
        return jnp.zeros(struct.shape, struct.dtype)

    t = jax.tree.map(_dt_struct_to_array, t)
    flat_t = flatten_linear_layers(t)
    flat_t = from_state_dict(flat_t, state_dict, prefix=prefix)
    t = unflatten_linear_layers(t, flat_t)
    return t


@typing.overload
def with_prefix(prefix: str | None, leaf: str) -> str:
    ...


@typing.overload
def with_prefix(prefix: str, leaf: None) -> str:
    ...


@typing.overload
def with_prefix(prefix: Optional[str], leaf: Optional[str]) -> Optional[str]:
    ...


def with_prefix(prefix: Optional[str], leaf: Optional[str]) -> Optional[str]:
    """Joins two optional path strings in a way compatible with pytorch state dict serialization"""
    if prefix is None:
        return leaf
    elif leaf is None:
        return prefix
    else:
        return f"{prefix}.{leaf}"


class ModuleWithStateDictSerialization(eqx.Module):
    """An eqx.Module that can be serialized to a torch-style state dict."""

    def to_state_dict(self, prefix: Optional[str] = None) -> StateDict:
        return default_eqx_module_to_state_dict(self, prefix)

    def from_state_dict(self: Mod, state_dict: StateDict, prefix: Optional[str] = None) -> Mod:
        return default_eqx_module_from_state_dict(self, state_dict, prefix)

    def _state_dict_key_map(self) -> dict[str, Optional[str]]:
        """Returns a dict mapping eqx.Module keys to torch keys that need to be renamed for serialization"""
        return {}


def from_state_dict(tree: T, state_dict: StateDict, prefix: Optional[str] = None) -> T:
    """
    Given a (template) tree and a state dict, return a new tree with the same structure as the input tree, but with
    the values from the state dict.

    Args:
        tree: The template tree
        state_dict: The state dict
        prefix: The prefix to use when looking up keys in the state dict

    Returns:
        A new tree with the same structure as the input tree, but with the values from the state dict.

    """
    # TODO: assert compatibility of old and new values (type, shape, etc.)
    if isinstance(tree, eqx.Module):
        if hasattr(tree, "from_state_dict"):
            return tree.from_state_dict(state_dict, prefix)
        else:
            return default_eqx_module_from_state_dict(tree, state_dict, prefix)
    elif isinstance(tree, list):
        return [from_state_dict(item, state_dict, with_prefix(prefix, str(i))) for i, item in enumerate(tree)]  # type: ignore
    elif isinstance(tree, dict):
        return {k: from_state_dict(v, state_dict, prefix=with_prefix(prefix, k)) for k, v in tree.items()}  # type: ignore
    elif isinstance(tree, NamedArray):
        if prefix is None:
            raise ValueError("Cannot extract a leaf value from a torch dict without a prefix")

        array = state_dict[prefix]

        if isinstance(array, np.ndarray):
            mesh = partitioning._get_mesh()
            # TODO: modernize this
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


def to_state_dict(tree: PyTree, prefix: Optional[str] = None) -> StateDict:
    """
    Convert a PyTree to a state dict.

    Returns:
        The state dict representation of the input tree.
    """
    if tree is None:
        return {}
    elif isinstance(tree, eqx.Module):
        if hasattr(tree, "to_state_dict"):
            state_dict = tree.to_state_dict(prefix)
        else:
            state_dict = default_eqx_module_to_state_dict(tree, prefix)
    elif isinstance(tree, list):
        state_dict = {}
        for i, item in enumerate(tree):
            child = to_state_dict(item, prefix=with_prefix(prefix, str(i)))
            # TODO: check for conflicts?
            state_dict.update(child)
    elif isinstance(tree, dict):
        state_dict = {}
        for k, v in tree.items():
            child = to_state_dict(v, prefix=with_prefix(prefix, k))
            # TODO: check for conflicts?
            state_dict.update(child)
    elif isinstance(tree, NamedArray):
        if prefix is None:
            raise ValueError("Cannot convert a leaf value to a state dict without a prefix")
        if tree.array is not None:
            state_dict = {prefix: tree.array}
        else:
            state_dict = {}
    elif is_jax_array_like(tree):
        if prefix is not None:
            if tree is not None:
                state_dict = {prefix: tree}
            else:
                state_dict = {}
        else:
            raise ValueError("Cannot convert a leaf value to a state dict without a prefix")
    elif is_scalarish(tree):
        if prefix is not None:
            state_dict = {prefix: tree}
        else:
            raise ValueError("Cannot convert a leaf value to a state dict without a prefix")
    else:
        raise ValueError(f"Unsupported type {type(tree)}")

    return state_dict


def default_eqx_module_from_state_dict(mod: Mod, state_dict: StateDict, prefix: Optional[str] = None) -> Mod:
    key_map: Dict[str, Optional[str]] = getattr(mod, "_state_dict_key_map", lambda: {})()  # type: ignore
    names = []
    values = []
    for field in dataclasses.fields(mod):
        # only encode dynamic fields (i.e. fields that don't have the static flag)
        if field.metadata.get("static", False):
            continue
        key = key_map.get(field.name, field.name)
        value = getattr(mod, field.name)
        # TODO: might want to add a flag that allows missing keys?
        new = from_state_dict(value, state_dict, with_prefix(prefix, key))
        # Do not try to update parameters that are never defined
        if value is None and new is None:
            continue
        names.append(field.name)
        values.append(new)
    return eqx.tree_at(lambda m: [getattr(m, name) for name in names], mod, values)


def default_eqx_module_to_state_dict(mod: eqx.Module, prefix: Optional[str] = None) -> StateDict:
    """
    Convert an eqx.Module to a state dict. This is the default implementation of the to_state_dict method for
    eqx.Modules. It works by iterating over the fields of the module and calling to_state_dict on each field.
    Args:
        mod:
        prefix:

    Returns:

    """
    state_dict: StateDict = {}
    key_map: Dict[str, Optional[str]] = getattr(mod, "_state_dict_key_map", lambda: {})()  # type: ignore
    for field in dataclasses.fields(mod):
        if field.metadata.get("static", False):
            continue
        key = key_map.get(field.name, field.name)
        value = getattr(mod, field.name)
        child = to_state_dict(value, with_prefix(prefix, key))
        # TODO: should we check for conflicts?
        state_dict.update(child)
    return state_dict


def format_path_for_state_dict(prefix: Optional[str], path: Sequence) -> str:
    res = "".join(_format_key_path_element(path_elem) for path_elem in path)
    # res will have a .
    if prefix is not None:
        res = f"{prefix}{res}"
    elif res.startswith("."):
        res = res[1:]

    return res


# Torch compatible KeyPath formatting. Torch just always uses .
def _format_key_path_element(path_elem) -> str:
    match path_elem:
        case SequenceKey(idx):  # type: ignore
            return f".{idx}"
        case DictKey(key):  # type: ignore
            return f".{key}"
        case GetAttrKey():  # type: ignore
            return str(path_elem)
        case FlattenedIndexKey(idx):  # type: ignore
            return f".{idx}"
        case _:
            # The convention in JAX is to append the separator in the element itself
            # so we expect it to have
            path_elem = str(path_elem)
            if path_elem.startswith("."):
                return path_elem
            else:
                return f".{path_elem}"


def to_numpy_state_dict(model, prefix: Optional[str] = None) -> StateDict:
    """
    Convert a model to a state dict by first creating desharded copies of all parameters that reside in CPU
    memory.

    This method is especially useful for saving models distributed across multiple hosts.
    """

    with jax.default_device(jax.local_devices(backend="cpu")[0]):

        def get_to_cpu(arr):
            if is_scalarish(arr):
                return arr
            elif not is_jax_array_like(arr):
                return arr
            elif isinstance(arr, np.ndarray):
                return arr
            elif is_jax_array_like(arr):
                if arr.is_fully_addressable:
                    r = np.array(arr)
                    return r
                else:
                    # unfortunately, jax's allgather seems to replicate to every device rather than every host
                    # which doesn't work for ~7B parameter models on TPU (assuming we also have optimizer state)
                    # this approach limits us to <64B parameters, but that's good enough for now
                    # we're going to do something a bit fancy, where we shard the model into a (process, device) mesh,
                    # then look for some axis along which we can shard the array, and then we'll do an allgather
                    # via pjit. If we can't find one, we'll just fully replicate since it probably isn't that big.
                    # TODO: ensure that this mesh arranges devices correctly
                    # (jax seems to do this internally itself, so we should be fine?)
                    process_mesh = Mesh(
                        np.array(jax.devices()).reshape((jax.process_count(), -1)), ("process", "device")
                    )
                    # now we need to find an axis along which we can shard the array.
                    # for this, we need to find an axis s.t. size(axis) % local_devices == 0

                    try:
                        axis_to_shard = index_where(
                            lambda axis_size: axis_size % process_mesh.devices.size == 0, arr.shape
                        )
                    except ValueError:
                        return np.array(arr)

                    shardings = [None if i != axis_to_shard else "device" for i in range(len(arr.shape))]
                    sharding = NamedSharding(process_mesh, PartitionSpec(*shardings))
                    out = jax.jit(lambda x: x, out_shardings=sharding)(arr)
                    return np.array(out)
            elif is_scalarish(arr):
                return np.asarray(arr)
            else:
                raise ValueError(f"Unsupported type {type(arr)}")

        # need to make sure the model is on *this machine* and *this machine's CPU* before saving
        model = jax.tree.map(lambda arr: get_to_cpu(arr), model)
        # TODO: it would be nice if safetensors supported an iterator or something so we could do the allgather one at a time
        state_dict = to_state_dict(model, prefix=prefix)
        return state_dict


_GLOBAL_SAVE_COUNT = 0


def save_state_dict(state_dict: StateDict, path):
    """
    Save a model's state dict to a file, bringing all tensors to the CPU first and then converting to numpy.
    This will save using safetensors format
    """
    state_dict = {k: v for k, v in state_dict.items() if v is not None}
    if jax.process_index() == 0:
        # the "pt" is a lie but it doesn't seem to actually matter and HF demands it
        safetensors.numpy.save_file(state_dict, path, metadata={"format": "pt"})
    global _GLOBAL_SAVE_COUNT
    sync_global_devices(f"save_state_dict {_GLOBAL_SAVE_COUNT}")
    _GLOBAL_SAVE_COUNT += 1


def load_state_dict(path):
    """
    Load a model's state dict from a file, bringing all tensors to the CPU first and then converting to numpy.
    This will load using safetensors format
    """
    state_dict = safetensors.numpy.load_file(path)
    return state_dict


def flatten_linear_layers(tree: T) -> T:
    """
    In PyTorch, linear layers are stored as a 2d weight matrix and a 1d bias vector. In Haliax,
    linear layers can have arbitrary dimensions, grouped into input and output axes. This function
    flattens the linear layers in a tree to be compatible with PyTorch-style state dicts.

    :param tree:
    """
    from haliax.nn import Linear

    def _flatten_linear(layer):
        if not isinstance(layer, Linear):
            return layer

        weight = layer.weight
        bias = layer.bias

        if weight.array is not None:
            out_first = layer.out_first
            weight = weight.flatten_axes(layer.Out, "__OUT__").flatten_axes(layer.In, "__IN__")

            if out_first:
                weight = weight.rearrange((..., "__OUT__", "__IN__"))
            else:
                weight = weight.rearrange((..., "__IN__", "__OUT__"))

            if bias is not None:
                bias = bias.flatten_axes(layer.Out, "__OUT__")

            In = weight.resolve_axis("__IN__")
            Out = weight.resolve_axis("__OUT__")

            return dataclasses.replace(layer, weight=weight, bias=bias, In=In, Out=Out)  # type: ignore
        else:
            return layer

    return jax.tree.map(_flatten_linear, tree, is_leaf=lambda x: isinstance(x, Linear))


def unflatten_linear_layers(template: T, tree_with_flattened_linears: T) -> T:
    """
    Unflattens linear layers in a tree that was flattened with [haliax.state_dict.flatten_linear_layers][].
    Template has the same structure as the tree that was flattened, but with the original (unflattened)
    linear layers.

    Returns:
        The same tree as `tree_with_flattened_linears`, but with the linear layers unflattened to match
        the structure of `template`.
    """

    from haliax.nn import Linear

    def _unflatten_linear(template, flattened):
        assert isinstance(template, Linear) == isinstance(flattened, Linear)

        if not isinstance(template, Linear):
            return flattened

        weight = flattened.weight
        bias = flattened.bias

        if weight.array is not None:
            weight = weight.unflatten_axis("__OUT__", template.Out).unflatten_axis("__IN__", template.In)
            weight = weight.rearrange(template.weight.axes)

        if bias is not None:
            bias = bias.unflatten_axis("__OUT__", template.Out)
            assert template.bias is not None, "Flattened bias but template has no bias"
            bias = bias.rearrange(template.bias.axes)

        return dataclasses.replace(template, weight=weight, bias=bias)  # type: ignore

    return jax.tree.map(
        _unflatten_linear, template, tree_with_flattened_linears, is_leaf=lambda x: isinstance(x, Linear)
    )
