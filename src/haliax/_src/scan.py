import dataclasses
import functools as ft
from typing import Any, Callable, ParamSpec, Protocol, Tuple, TypeVar, Union, overload

import equinox as eqx
import jax
from jaxtyping import PyTree

import haliax
import haliax.tree_util as htu
from haliax._src.util import index_where
from haliax.axis import Axis, AxisSelector, selects_axis
from haliax.core import NamedArray
from haliax.jax_utils import checkpointing_scan, is_jax_array_like
from haliax.util import is_jax_or_hax_array_like, is_named_array


BoolAxisSpec = Union[bool, Callable[[Any], bool]]
Carry = TypeVar("Carry")
X = TypeVar("X", contravariant=True)
Y = TypeVar("Y", covariant=True)
Args = ParamSpec("Args")


def is_named_or_shaped_array_like(x):
    return (is_jax_array_like(x) and x.ndim >= 1) or is_named_array(x)


class ScanFn(Protocol[Carry, Args, Y]):
    """ """

    def __call__(self, carry: Carry, *args: Args.args, **kwargs: Args.kwargs) -> tuple[Carry, Y]:
        ...


@overload
def scan(
    f: Callable[[Carry, X], tuple[Carry, Y]],
    axis: AxisSelector,
    *,
    nested_scan: bool | int = False,
    reverse: bool = False,
    unroll: int = 1,
    is_scanned: BoolAxisSpec = is_named_or_shaped_array_like,
) -> Callable[[Carry, PyTree[X]], tuple[Carry, PyTree[Y]]]:
    ...


@overload
def scan(
    f: Callable,
    axis: AxisSelector,
    *,
    nested_scan: bool | int = False,
    reverse: bool = False,
    unroll: int = 1,
    is_scanned: BoolAxisSpec = is_named_or_shaped_array_like,
) -> Callable:
    ...


def scan(
    f: Callable,  # : ScanFn[Carry, Args, Y],  This confuses mypy too much
    axis: AxisSelector,
    *,
    nested_scan: bool | int = False,
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
        nested_scan: Use nested scans to reduce memory usage. If an integer, use that many nested scans.
                If true, use the closest int to sqrt(axis.size) that divides axis.size, which gives
                O(sqrt(N)) memory and O(N) time.
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
        @ft.wraps(f)
        def wrapped_fn(carry, scanned_x_leaves):
            scanned_x = jax.tree_util.tree_unflatten(x_elem_structure, scanned_x_leaves)
            # this part is the most delicate: combining the scanned x with the unscanned x
            scanned_x = eqx.combine(scanned_x, unscanned_xs, is_leaf=is_named_array)
            args, kwargs = scanned_x
            carry, y = f(carry, *args, **kwargs)
            y = htu.tree_map(_pacify_named_arrays, y)
            return carry, y

        axis_size = _infer_axis_size_from_tree(axis_first_xs, axis).size

        outer_block_size: int | None
        if nested_scan is True:
            outer_block_size = find_closest_divisible_int_to_sqrt(axis_size)
        elif nested_scan is False:
            outer_block_size = None
        else:
            outer_block_size = nested_scan

        # as above, we don't want to use htu.tree_leaves here because we want to eliminate the leading axis
        leaves = jax.tree_util.tree_leaves(axis_first_xs)
        with jax.named_scope(f"scan({haliax.axis_name(axis)})"):
            if outer_block_size is not None:
                carry, ys = checkpointing_scan(
                    wrapped_fn, init, leaves, outer_block_size, reverse=reverse, unroll=unroll, length=axis_size
                )
            else:
                carry, ys = jax.lax.scan(wrapped_fn, init, leaves, reverse=reverse, unroll=unroll, length=axis_size)
        true_axis = _infer_axis_size_from_tree(ys, axis)
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
    nested_scan: bool | int = False,
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
    nested_scan: bool | int = False,
    is_scanned: BoolAxisSpec = is_jax_or_hax_array_like,
) -> Callable:
    ...


def fold(
    fn: Callable,
    axis: AxisSelector,
    *,
    reverse: bool = False,
    unroll: int = 1,
    nested_scan: bool | int = False,
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
        nested_scan: Use nested scans to reduce memory usage. If an integer, use that many nested scans.
                If true, use the closest int to sqrt(axis.size) that divides axis.size, which gives
                O(sqrt(N)) memory and O(N) time.
        is_scanned: a function that takes a leaf of the tree and returns True if it should be scanned over,
                    False otherwise. Behaves similarly to the `default` argument in filter_jit

    Returns:
        A function that takes the initial carry and then the arguments to reduce over, and returns the final carry
    """

    def scan_compatible_fn(carry, *args, **kwargs):
        return fn(carry, *args, **kwargs), None

    scan_preconfig = scan(
        scan_compatible_fn, axis, reverse=reverse, unroll=unroll, is_scanned=is_scanned, nested_scan=nested_scan
    )

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


def _infer_axis_size_from_tree(result, axis):
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


def find_closest_divisible_int_to_sqrt(n: int) -> int:
    """
    Find the closest integer to the square root of n (less than or equal to sqrt(n)) that divides n.
    """
    assert n > 0, f"Expected n > 0, got {n}"
    for i in range(int(n**0.5), 0, -1):
        if n % i == 0:
            return i

    return 1


def _is_passive_array(arr):
    return isinstance(arr, _PassiveNamedArray)


def _prepend_named_batch_axis(leading_axis: Axis):
    def to_active_named_array(leaf):
        if isinstance(leaf, _PassiveNamedArray):
            return leaf.as_scanned_result(leading_axis)
        else:
            return leaf

    return to_active_named_array
