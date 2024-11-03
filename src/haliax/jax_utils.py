import functools as ft
import typing
from typing import Any, Callable, Optional, Sequence, Union

import equinox as eqx
import jax
import numpy as np
from jax import Array
from jax import numpy as jnp
from jax import random as jrandom
from jax._src.numpy import lax_numpy
from jax._src.typing import DTypeLike
from jaxtyping import PRNGKeyArray

import haliax
from haliax.types import PrecisionLike


F = typing.TypeVar("F", bound=Callable[..., Any])


class Static(eqx.Module):
    value: Any = eqx.field(static=True)


def shaped_rng_split(key, split_shape: Union[int, Sequence[int]] = 2) -> PRNGKeyArray:
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
    return hasattr(x, "shape") and hasattr(x, "dtype")  # and not isinstance(x, haliax.NamedArray)


# adapted from jax but exposed so i can use it
def broadcast_prefix(prefix_tree: Any, full_tree: Any, is_leaf: Optional[Callable[[Any], bool]] = None):
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


@typing.overload
def named_call(f: F, name: Optional[str] = None) -> F:
    ...


@typing.overload
def named_call(*, name: Optional[str] = None) -> Callable[[F], F]:
    ...


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


def is_in_jit():
    return isinstance(jnp.zeros((), dtype=jnp.float32), jax.core.Tracer)


def is_pallas_dslice(x: object) -> bool:
    try:
        from jax.experimental.pallas import dslice as pdslice
    except ImportError:
        return False

    _PALLAS_DSLICE_TYPE = type(pdslice(0, 1))
    return isinstance(x, _PALLAS_DSLICE_TYPE)


def is_scalarish(x):
    if isinstance(x, haliax.NamedArray):
        return x.ndim == 0
    else:
        return jnp.isscalar(x) or x.shape == ()


def is_on_mac_metal():
    return jax.devices()[0].platform.lower() == "metal"


def _jittable_dg_einsum(
    subscripts,
    /,
    *operands,
    out: None = None,
    optimize: str = "optimal",
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    _dot_general: Callable[..., Array] = jax.lax.dot_general,
) -> Array:
    operands = (subscripts, *operands)
    if out is not None:
        raise NotImplementedError("The 'out' argument to jnp.einsum is not supported.")
    spec = operands[0] if isinstance(operands[0], str) else None
    optimize = "optimal" if optimize is True else optimize

    import opt_einsum

    # Allow handling of shape polymorphism
    non_constant_dim_types = {
        type(d) for op in operands if not isinstance(op, str) for d in np.shape(op) if not jax.core.is_constant_dim(d)
    }
    if not non_constant_dim_types:
        contract_path = opt_einsum.contract_path
    else:
        ty = next(iter(non_constant_dim_types))
        contract_path = lax_numpy._poly_einsum_handlers.get(ty, lax_numpy._default_poly_einsum_handler)
    # using einsum_call=True here is an internal api for opt_einsum... sorry
    operands, contractions = contract_path(*operands, einsum_call=True, use_blas=True, optimize=optimize)

    contractions = tuple((a, frozenset(b), c) for a, b, c, *_ in contractions)

    einsum = eqx.filter_jit(lax_numpy._einsum, inline=True)
    if spec is not None:
        einsum = jax.named_call(einsum, name=spec)
    return einsum(operands, contractions, precision, preferred_element_type, _dot_general)  # type: ignore[operator]
