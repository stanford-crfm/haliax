import functools as ft
from typing import Any, Callable, List, Optional, Sequence, Union

import equinox as eqx
import jax
import numpy as np
from jax import numpy as jnp
from jax import random as jrandom
from jaxtyping import PRNGKeyArray


class Static(eqx.Module):
    value: Any = eqx.field(static=True)


def shaped_rng_split(key, split_shape: Union[int, Sequence[int]] = 2) -> jrandom.KeyArray:
    if isinstance(split_shape, int):
        num_splits = split_shape
        split_shape = (num_splits,) + key.shape
    else:
        num_splits = np.prod(split_shape)
        split_shape = tuple(split_shape) + key.shape

    if num_splits == 1:
        return jnp.reshape(key, split_shape)

    unshaped = maybe_rng_split(key, num_splits)
    return jnp.reshape(unshaped, split_shape)


def maybe_rng_split(key: Optional[PRNGKeyArray], num: int = 2):
    """Splits a random key into multiple random keys. If the key is None, then it replicates the None. Also handles
    num == 1 case"""
    if key is None:
        return [None] * num
    elif num == 1:
        return jnp.reshape(key, (1,) + key.shape)
    else:
        return jrandom.split(key, num)


@ft.wraps(eqx.filter_eval_shape)
def filter_eval_shape(*args, **kwargs):
    import warnings

    warnings.warn("filter_eval_shape is deprecated, use eqx.filter_eval_shape instead", DeprecationWarning)
    return eqx.filter_eval_shape(*args, **kwargs)


def filter_checkpoint(fun: Callable, *, prevent_cse: bool = True, policy: Optional[Callable[..., bool]] = None):
    """As `jax.checkpoint`, but allows any Python object as inputs and outputs"""

    @ft.wraps(fun)
    def _fn(_static, _dynamic):
        _args, _kwargs = eqx.combine(_static, _dynamic)
        _out = fun(*_args, **_kwargs)
        _dynamic_out, _static_out = eqx.partition(_out, is_jax_array_like)
        return _dynamic_out, Static(_static_out)

    checkpointed_fun = jax.checkpoint(_fn, prevent_cse=prevent_cse, policy=policy, static_argnums=(0,))

    @ft.wraps(fun)
    def wrapper(*args, **kwargs):
        dynamic, static = eqx.partition((args, kwargs), is_jax_array_like)
        dynamic_out, static_out = checkpointed_fun(static, dynamic)

        return eqx.combine(dynamic_out, static_out.value)

    return wrapper


def is_jax_array_like(x):
    return hasattr(x, "shape") and hasattr(x, "dtype")


# adapted from jax but exposed so i can use it
def broadcast_prefix(prefix_tree: Any, full_tree: Any, is_leaf: Optional[Callable[[Any], bool]] = None) -> List[Any]:
    """Broadcast a prefix tree to match the structure of a full tree."""
    result = []
    num_leaves = lambda t: jax.tree_util.tree_structure(t).num_leaves  # noqa: E731
    add_leaves = lambda x, subtree: result.extend([x] * num_leaves(subtree))  # noqa: E731
    jax.tree_util.tree_map(add_leaves, prefix_tree, full_tree, is_leaf=is_leaf)
    full_structure = jax.tree_util.tree_structure(full_tree)

    return jax.tree_util.tree_unflatten(full_structure, result)


@ft.wraps(eqx.combine)
def combine(*args, **kwargs):
    import warnings

    warnings.warn("combine is deprecated, use eqx.combine instead", DeprecationWarning)
    return eqx.combine(*args, **kwargs)


def _UNSPECIFIED():
    raise ValueError("unspecified")


def named_call(f=_UNSPECIFIED, name: Optional[str] = None):
    if f is _UNSPECIFIED:
        return lambda f: named_call(f, name)  # type: ignore
    else:
        if name is None:
            name = f.__name__
            if name == "__call__":
                if hasattr(f, "__self__"):
                    name = f.__self__.__class__.__name__  # type: ignore
                else:
                    name = f.__qualname__.rsplit(".", maxsplit=1)[0]  # type: ignore
            else:
                name = f.__qualname__

        return jax.named_scope(name)(f)
