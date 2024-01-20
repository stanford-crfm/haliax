import dataclasses
import inspect
from functools import wraps
from typing import Any, Callable, ParamSpec, Protocol, Tuple, TypeVar, Union, overload

import equinox as eqx
import jax
import jax.lax as lax
from jaxtyping import PyTree

import haliax
import haliax.tree_util as htu

from ._src.util import index_where
from .axis import Axis, AxisSelector, selects_axis
from .core import NamedArray
from .jax_utils import Static, broadcast_prefix, is_jax_array_like
from .partitioning import physical_axis_name
from .util import is_jax_or_hax_array_like, is_named_array


BoolAxisSpec = Union[bool, Callable[[Any], bool]]
Carry = TypeVar("Carry")
X = TypeVar("X", contravariant=True)
Y = TypeVar("Y", covariant=True)
Args = ParamSpec("Args")


def is_named_or_shaped_array_like(x):
    return (is_jax_array_like(x) and x.ndim >= 1) or is_named_array(x)


class ScanFn(Protocol[Carry, Args, Y]):
    """ """

    def __call__(self, carry: Carry, *args: Args.args, **kwargs: Args.kwargs) -> Tuple[Carry, Y]:
        ...


@overload
def scan(
    f: Callable[[Carry, X], Tuple[Carry, Y]],
    axis: AxisSelector,
    *,
    reverse: bool = False,
    unroll: int = 1,
    is_scanned: BoolAxisSpec = is_named_or_shaped_array_like,
) -> Callable[[Carry, PyTree[X]], Tuple[Carry, PyTree[Y]]]:
    ...


@overload
def scan(
    f: Callable,
    axis: AxisSelector,
    *,
    reverse: bool = False,
    unroll: int = 1,
    is_scanned: BoolAxisSpec = is_named_or_shaped_array_like,
) -> Callable:
    ...


def scan(
    f: Callable,  # : ScanFn[Carry, Args, Y],  This confuses mypy too much
    axis: AxisSelector,
    *,
    reverse=False,
    unroll=1,
    is_scanned: BoolAxisSpec = is_named_or_shaped_array_like,
):
    """
    Scan over a named axis. Non-scalar unnamed arrays will have their first axis scanned over.

    Unlike [jax.lax.scan][], this function is curried: it takes the function, axis, and configuration arguments first, and
    then the initial carry and then any arguments to scan over as a separate curried function call.

    That is, `scan(f, axis)(init, xs)` is equivalent to `jax.lax.scan(f, init, xs)`

    Args:
        f (Callable): function to scan over
        axis (AxisSelector): axis to scan over
        reverse (bool): if True, scan in reverse
        unroll (int): unroll the loop by this amount
        is_scanned (BoolAxisSpec): a function that takes a leaf of the tree and returns True if it should be scanned over,
                    False otherwise. Behaves similarly to the `default` argument in filter_jit
    """

    if isinstance(is_scanned, bool):
        q = is_scanned  # this is to make mypy happy
        is_scanned = lambda _: q

    def is_scanned_with_axis(leaf):
        if is_named_array(leaf):
            return selects_axis(leaf.axes, axis) and is_scanned(leaf)
        else:
            return is_scanned(leaf)

    def scanned_f(init, *args, **kwargs):
        # This implementation is a bit tricky.

        # first we want to partition the arguments into scanned and unscanned
        # unscanned arguments are just passed through, essentially captured as part of a lambda
        # scanned arguments are passed through the scan, which means we need to hoist the axis to the front
        xs = (args, kwargs)
        scanned_xs, unscanned_xs = eqx.partition(xs, is_scanned_with_axis, is_leaf=is_named_array)

        # Next we have to hoist the axis we're scanning over to the front of the array, because that's what scan
        # expects. Then we have to scan over the 0th dim of the arrays (as flattened non-pytrees)
        # We have to be careful that we don't try to create NamedArrays that have the shape of the scanned result
        # but don't yet have the scanned axis as ones of `axes`, so we use _ScannedArrayResult that doesn't check
        # invariants until we're ready to create the result.
        axis_first_xs = htu.tree_map(_ensure_first(axis), scanned_xs)

        # now get a template of an element of "X"
        x_elem = htu.tree_map(_select_0th(axis), axis_first_xs)
        # NB: we don't want to use htu.tree_structure here because we want to eliminate the leading axis
        x_elem_structure = jax.tree_util.tree_structure(x_elem)

        # now we can fold over the axis
        @wraps(f)
        def wrapped_fn(carry, scanned_x_leaves):
            scanned_x = jax.tree_util.tree_unflatten(x_elem_structure, scanned_x_leaves)
            # this part is the most delicate: combining the scanned x with the unscanned x
            scanned_x = eqx.combine(scanned_x, unscanned_xs, is_leaf=is_named_array)
            args, kwargs = scanned_x
            carry, y = f(carry, *args, **kwargs)
            y = htu.tree_map(_pacify_named_arrays, y)
            return carry, y

        # as above, we don't want to use htu.tree_leaves here because we want to eliminate the leading axis
        leaves = jax.tree_util.tree_leaves(axis_first_xs)
        with jax.named_scope(f"scan({haliax.axis_name(axis)})"):
            carry, ys = lax.scan(wrapped_fn, init, leaves, reverse=reverse, unroll=unroll)
        true_axis = _infer_axis_size_from_result(ys, axis)
        ys = jax.tree_util.tree_map(_prepend_named_batch_axis(true_axis), ys, is_leaf=_is_passive_array)

        return carry, ys

    return scanned_f


@overload
def fold(
    fn: Callable[[Carry, X], Carry],
    axis: AxisSelector,
    *,
    reverse: bool = False,
    unroll: int = 1,
    is_scanned: BoolAxisSpec = is_jax_or_hax_array_like,
) -> Callable[[Carry, PyTree[X]], Carry]:
    ...


@overload
def fold(
    fn: Callable,
    axis: AxisSelector,
    *,
    reverse: bool = False,
    unroll: int = 1,
    is_scanned: BoolAxisSpec = is_jax_or_hax_array_like,
) -> Callable:
    ...


def fold(
    fn: Callable,
    axis: AxisSelector,
    *,
    reverse: bool = False,
    unroll: int = 1,
    is_scanned: BoolAxisSpec = is_named_or_shaped_array_like,
) -> Callable:
    """
    Slightly simpler implementation of scan that folds over the named axis of the array, not returning intermediates.

    As with scan, this function is curried: it takes the function, axis, and configuration arguments first, and
    then the initial carry and then any arguments to scan over as a separate curried function call.

    Unnamed arrays will have their first axis scanned over, unless they are scalars, in which case they will be passed
    through unchanged.

    Args:
        fn: function to reduce over
        axis: axis to reduce over
        reverse: if True, reduce in reverse
        unroll: unroll the loop by this amount
        is_scanned: a function that takes a leaf of the tree and returns True if it should be scanned over,
                    False otherwise. Behaves similarly to the `default` argument in filter_jit

    Returns:
        A function that takes the initial carry and then the arguments to reduce over, and returns the final carry
    """

    def scan_compatible_fn(carry, *args, **kwargs):
        return fn(carry, *args, **kwargs), None

    scan_preconfig = scan(scan_compatible_fn, axis, reverse=reverse, unroll=unroll, is_scanned=is_scanned)

    def scanned_f(init, *args, **kwargs):
        return scan_preconfig(init, *args, **kwargs)[0]

    return scanned_f


def map(
    fn: Callable[[X], Y],
    axis: Axis,
    reverse: bool = False,
    unroll: int = 1,
    is_mapped: BoolAxisSpec = is_jax_or_hax_array_like,
) -> Callable[[PyTree[X]], PyTree[Y]]:
    """
    NamedArray aware version of jax.lax.map. Normal arrays are mapped according to the specs as in equinox.filter_map,
    except that the output axis is always 0 b/c it's annoying to make anything else work.

    You'll typically want to use map (instead of a vmap or just vectorized code) when you want to encourage XLA to
    loop over the axis to control memory.

    Args:
        fn (Callable):  function to map over
        axis (Axis): axis to map over
        reverse (bool): if True, map in reverse
        unroll (int): unroll the loop by this amount

    """

    def scan_compatible_fn(_, x):
        del _
        return None, fn(x)

    scan_preconfig = scan(scan_compatible_fn, axis, reverse=reverse, unroll=unroll, is_scanned=is_mapped)

    def scanned_f(*args, **kwargs):
        return scan_preconfig(None, *args, **kwargs)[1]

    return scanned_f


ResolvedUnnamedAxisSpec = Union[int, None]
UnnamedAxisSpec = Union[ResolvedUnnamedAxisSpec, Callable[[Any], ResolvedUnnamedAxisSpec]]


def _zero_if_array_else_none(x: Any) -> ResolvedUnnamedAxisSpec:
    return 0 if is_jax_array_like(x) else None


def vmap(
    fn,
    axis: AxisSelector,
    *,
    default: PyTree[UnnamedAxisSpec] = _zero_if_array_else_none,
    args: PyTree[UnnamedAxisSpec] = (),
    kwargs: PyTree[UnnamedAxisSpec] = None,
):
    """
    [haliax.NamedArray][]-aware version of [jax.vmap][]. Normal arrays are mapped according to the specs as in
     [equinox.filter_vmap][]

    Because of NamedArrays, vmap is typically less useful than in vanilla JAX, but it is sometimes
    useful for initializing modules that will be scanned over. See [haliax.nn.Stacked][] for an example.

    Args:
        fn (Callable): function to vmap over
        axis (Axis): axis to vmap over
        default: how to handle (unnamed) arrays by default. Should be either an integer or None, or a callable that takes a PyTree leaf
            and returns an integer or None, or a PyTree prefix of the same. If an integer, the array will be mapped over that axis. If None, the array will not be mapped over.
        args: optional per-argument overrides for how to handle arrays. Should be a PyTree prefix of the same type as default.
        kwargs: optional per-keyword-argument overrides for how to handle arrays. Should be a PyTree prefix of the same type as default.
    """

    if kwargs is None:
        kwargs = {}

    signature = inspect.signature(fn)

    # this mirrors equinox's filter_vmap, but it's not really documented there so:
    # we use inspect.signature to align args/kwargs specified in vmap to what actually gets passed in
    # axis_spec_bound_sig's job is to hold that mapping
    signature_default = signature.replace(
        parameters=[
            p
            if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            else p.replace(default=default)
            for p in signature.parameters.values()
        ]
    )
    axis_spec_bound_sig = signature_default.bind_partial(*args, **kwargs)
    axis_spec_bound_sig.apply_defaults()
    del args, kwargs

    def _index_of_batch_axis(array, default):
        if isinstance(array, NamedArray):
            return array._lookup_indices(axis)
        elif callable(default):
            return default(array)
        else:
            return default

    # TODO: tests to exercise this more
    @wraps(fn)
    def wrapped_vmap_fn(*args, **kwargs):
        # TODO: this probably results in a lot of compilation misses. Need to think about it.
        actual_bound = signature.bind(*args, **kwargs)
        actual_bound.apply_defaults()

        # now that we have args, we can figure out what the axis spec is for each arg
        padded_spec_args = axis_spec_bound_sig.args + (default,) * (
            len(actual_bound.args) - len(axis_spec_bound_sig.args)
        )

        padded_spec_kwargs = {
            **axis_spec_bound_sig.kwargs,
            **{k: default for k in actual_bound.kwargs.keys() - axis_spec_bound_sig.kwargs.keys()},
        }

        # want to support padded_spec_args being a tree prefix of the actual args, which this enables
        padded_spec_args = broadcast_prefix(padded_spec_args, actual_bound.args, is_leaf=is_named_array)
        padded_spec_kwargs = broadcast_prefix(padded_spec_kwargs, actual_bound.kwargs)

        arg_axis_specs = htu.tree_map(_index_of_batch_axis, actual_bound.args, padded_spec_args)

        kwarg_axis_specs = htu.tree_map(_index_of_batch_axis, actual_bound.kwargs, padded_spec_kwargs)

        # now we can actually vmap. We used "pacified" versions of NamedArrays that don't check
        # invariants, because intermediates creating during tracing won't have the axes right
        arg_axis_specs = htu.tree_map(_pacify_named_arrays, arg_axis_specs)
        kwarg_axis_specs = htu.tree_map(_pacify_named_arrays, kwarg_axis_specs)

        def wrapped_fn(args, kwargs):
            # the args that come in here are pacified. Their names will still have the batch axis even though the array
            # itself will already have that one removed. We need to turn them back into NamedArrays by removing the axis
            unchilled_args = jax.tree_util.tree_map(_to_unbatched_named_array(axis), args, is_leaf=_is_passive_array)
            unchilled_kwargs = jax.tree_util.tree_map(
                _to_unbatched_named_array(axis), kwargs, is_leaf=_is_passive_array
            )

            out = fn(*unchilled_args, **unchilled_kwargs)

            # now we need to pacify the output, which may include NamedArrays, and add the batch axis back at the end
            chilled = htu.tree_map(_pacify_named_arrays, out)
            arrays, nonarrays = eqx.partition(chilled, is_jax_array_like)
            return arrays, Static(nonarrays)

        spmd_axis_name = physical_axis_name(axis)

        args = htu.tree_map(_pacify_named_arrays, actual_bound.args)
        kwargs = htu.tree_map(_pacify_named_arrays, actual_bound.kwargs)

        result_dynamic, result_static = jax.vmap(
            wrapped_fn,
            in_axes=(arg_axis_specs, kwarg_axis_specs),
            out_axes=0,
            axis_size=axis.size if isinstance(axis, Axis) else None,
            spmd_axis_name=spmd_axis_name,
        )(args, kwargs)

        result = eqx.combine(result_dynamic, result_static.value)

        # if we were passed in a string arg, we need to get its axis size out from some result
        true_axis = _infer_axis_size_from_result(result, axis)
        if true_axis is None:
            raise ValueError("vmap failed to infer axis size from result")

        result = jax.tree_util.tree_map(_prepend_named_batch_axis(true_axis), result, is_leaf=_is_passive_array)
        return result

    return wrapped_vmap_fn


def _infer_axis_size_from_result(result, axis):
    if isinstance(axis, str):
        result_leaves = jax.tree_util.tree_leaves(result, is_leaf=_is_passive_array)
        if len(result_leaves) == 0:
            # this really shouldn't happen
            return None
        if isinstance(result_leaves[0], _PassiveNamedArray):
            true_axis_size = result_leaves[0].array.shape[0]  # batch axis is defined to be 0 above
            true_axis = Axis(axis, true_axis_size)
        else:
            true_axis_size = result_leaves[0].shape[0]  # batch axis is defined to be 0 above
            true_axis = Axis(axis, true_axis_size)
    else:
        true_axis = axis
    return true_axis


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class _PassiveNamedArray:
    """For some higher-order jax manipulations, jax will insert/remove an axis at the beginning of the array (or
    sometimes elsewhere). Jax doesn't know about our NamedArray wrapper, so we need a variant of NamedArray
    that doesn't care about the axis names. This is that variant.This class is a 'chill' version of NamedArray
    that doesn't check invariants until we're ready to create the result

    For example, with scan in NamedArray, we can't just have the scan tree prepend the scanned axis to the result,
    because we don't have a way to feed it the name of the scanned axis.
    """

    array: jax.numpy.ndarray
    main_axes: Tuple[Axis, ...]

    def as_scanned_result(self, scan_axis: Axis):
        return NamedArray(self.array, (scan_axis,) + self.main_axes)

    def strip_axis(self, axis: AxisSelector):
        if isinstance(axis, Axis):
            index = self.main_axes.index(axis)
        else:
            index = index_where(lambda a: a.name == axis, self.main_axes)
        return NamedArray(self.array, self.main_axes[:index] + self.main_axes[index + 1 :])

    def to_named_array(self):
        return NamedArray(self.array, self.main_axes)

    def tree_flatten(self) -> Any:
        return ((self.array,), self.main_axes)

    @classmethod
    def tree_unflatten(cls, aux, tree: Any) -> Any:
        assert len(tree) == 1
        return cls(tree[0], main_axes=aux)


def _is_passive_array(arr):
    return isinstance(arr, _PassiveNamedArray)


def _prepend_named_batch_axis(leading_axis: Axis):
    def to_active_named_array(leaf):
        if isinstance(leaf, _PassiveNamedArray):
            return leaf.as_scanned_result(leading_axis)
        else:
            return leaf

    return to_active_named_array


def _to_unbatched_named_array(axis_to_strip: AxisSelector):
    def to_unbatched_named_array(leaf):
        if isinstance(leaf, _PassiveNamedArray):
            if selects_axis(leaf.main_axes, axis_to_strip):
                return leaf.strip_axis(axis_to_strip)
            else:
                return leaf.to_named_array()
        else:
            return leaf

    return to_unbatched_named_array


def _pacify_named_arrays(leaf):
    if isinstance(leaf, NamedArray):
        return _PassiveNamedArray(leaf.array, leaf.axes)
    elif isinstance(leaf, _PassiveNamedArray):
        assert False, "PassiveNamedArray should not be present in the tree"
    else:
        return leaf


def _select_0th(axis):
    def select_0th(leaf):
        if isinstance(leaf, NamedArray):
            return leaf.take(axis, 0)
        elif isinstance(leaf, _PassiveNamedArray):
            assert False, "PassiveNamedArray should not be present in the tree"
        else:
            # other leaves don't matter
            return leaf

    return select_0th


def _ensure_first(axis):
    def ensure_first(leaf):
        if isinstance(leaf, NamedArray):
            return leaf.rearrange((axis, ...))
        elif isinstance(leaf, _PassiveNamedArray):
            assert False, "PassiveNamedArray should not be present in the tree"
        else:
            return leaf

    return ensure_first


__all__ = ["scan", "fold", "vmap", "map"]
