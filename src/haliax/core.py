import contextlib
import functools as ft
import typing
import warnings
from dataclasses import dataclass
from math import prod
from types import EllipsisType
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union, overload

import jax
import jax.numpy as jnp
import numpy as np

import haliax
import haliax.axis
from haliax.jax_utils import is_jax_array_like, is_pallas_dslice
from haliax.util import ensure_tuple

from ._src.util import index_where, py_slice, slice_t
from .axis import Axis, AxisSelection, AxisSelector, AxisSpec, axis_name, dslice, eliminate_axes, selects_axis
from .types import GatherScatterModeStr, IntScalar, PrecisionLike, Scalar


NamedOrNumeric = Union[Scalar, "NamedArray"]
NamedIndex = Union[int, slice_t, "NamedArray", dslice, list[int], jnp.ndarray]

SliceSpec = Union[
    tuple[AxisSelector, NamedIndex],
    tuple[AxisSelector, NamedIndex, AxisSelector, NamedIndex],
    tuple[AxisSelector, NamedIndex, AxisSelector, NamedIndex, AxisSelector, NamedIndex],
    tuple[AxisSelector | NamedOrNumeric, ...],
    Mapping[AxisSelector, NamedIndex],
]


_ENABLE_SHAPE_CHECKS = True


@contextlib.contextmanager
def enable_shape_checks(enabled):
    """
    Sometimes we end up in situations where an array that jax makes is passed into the NamedArray constructor that
    doesn't conform to the shape we expect. This shows up in particular when we are using jax.vmap or jax.scan,
    and we sometimes have weird situations with deserialization

    Yields the old value because we sometimes want to nest this
    """
    global _ENABLE_SHAPE_CHECKS
    old = _ENABLE_SHAPE_CHECKS
    _ENABLE_SHAPE_CHECKS = enabled
    try:
        yield old
    finally:
        _ENABLE_SHAPE_CHECKS = old


def are_shape_checks_enabled():
    return _ENABLE_SHAPE_CHECKS


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class NamedArray:
    array: jnp.ndarray
    axes: Tuple[Axis, ...]

    def __post_init__(self):
        if not isinstance(self.axes, tuple):
            object.__setattr__(self, "axes", tuple(self.axes))
        # ensure axes are all Axis objects
        # TODO: anonymous positional axes?
        for axis in self.axes:
            if not isinstance(axis, Axis):
                raise TypeError(f"Expected Axis, got {type(axis)}")

        # ensure unique axes for now
        if len(set(a.name for a in self.axes)) != len(self.axes):
            raise ValueError(f"Axes must be unique, but {self.axes} are not")

        if are_shape_checks_enabled():
            self._ensure_shape_matches_axes()

    def _ensure_shape_matches_axes(self):
        """This is typically called automatically, but sometimes we need to call it manually if
        are_shape_checks_enabled() is False"""
        if is_jax_array_like(self.array):
            s = jnp.shape(self.array)
            if s != tuple(a.size for a in self.axes):
                raise ValueError(f"Shape of underlying array {s} does not match shape of axes {self.axes}")

    def item(self):
        """Returns the value of this NamedArray as a python scalar."""
        return self.array.item()

    def scalar(self) -> jnp.ndarray:
        """
        Returns a scalar array corresponding to the value of this NamedArray.
        Raises an error if the NamedArray is not scalar.

        We sometimes use this to convert a NamedArray to a scalar for returning a loss or similar. Losses
        have to be jnp.ndarrays, not NamedArrays, so we need to convert them. item doesn't work inside jitted
        functions because it returns a python scalar.

        You could just call array, but that's not as clear and doesn't assert.
        """
        if self.array.ndim != 0:
            raise ValueError(f"Expected scalar, got {self.array.ndim}-dimensional array")
        return self.array

    def __jax_array__(self):
        if self.ndim == 0:
            return self.array
        else:
            raise ValueError(
                "Only scalar NamedArrays can be implicitly converted to jax arrays, but "
                f"got {self.shape} array. This error typically occurs when you pass a "
                "NamedArray to a plain jax.numpy function. Please use `x.array` instead."
            )

    @ft.cached_property
    def shape(self) -> Dict[str, int]:
        return {axis.name: axis.size for axis in self.axes}

    dtype = property(lambda self: self.array.dtype)
    """The dtype of the underlying array"""
    ndim = property(lambda self: self.array.ndim)
    """The number of dimensions of the underlying array"""
    size = property(lambda self: self.array.size)
    """The number of elements in the underlying array"""
    nbytes = property(lambda self: self.array.nbytes)
    """The number of bytes in the underlying array"""

    def tree_flatten(self) -> Any:
        return ((self.array,), self.axes)

    @classmethod
    def tree_unflatten(cls, aux, tree: Any) -> Any:
        assert len(tree) == 1
        # We don't want check shapes b/c there are intermediate states where the shape is wrong
        # e.g. in eqxi.while_loop
        with enable_shape_checks(False):
            return cls(tree[0], axes=aux)

    def has_axis(self, axis: AxisSelection) -> bool:
        """Returns true if the given axis is present in this NamedArray."""
        return self._lookup_indices(axis) is not None

    @overload
    def axis_size(self, axis: AxisSelector) -> int:  # type: ignore
        ...

    @overload
    def axis_size(self, axis: Sequence[AxisSelector]) -> Tuple[int, ...]:  # type: ignore
        ...

    def axis_size(self, axis: AxisSelection) -> Union[int, Tuple[int, ...]]:
        """
        Returns the size of the given axis, or a tuple of sizes if given multiple axes.
        """
        indices = self._lookup_indices(axis)
        if isinstance(indices, int):
            return self.axes[indices].size
        elif indices is None:
            raise ValueError(f"Axis {axis} not found")
        else:
            result = []
            for i in indices:
                if i is None:
                    raise ValueError(f"Axis {axis} not found")
                result.append(self.axes[i].size)
            return tuple(result)

    @overload
    def resolve_axis(self, axis: AxisSelector) -> Axis:  # type: ignore
        ...

    @overload
    def resolve_axis(self, axis: Sequence[AxisSelector]) -> Tuple[Axis, ...]:  # type: ignore
        ...

    def resolve_axis(self, axes: AxisSelection) -> AxisSpec:  # type: ignore
        """
        Returns the axes corresponding to the given axis selection.
        That is, it return the [haliax.Axis][] values themselves, not just their names.

        Raises a ValueError if any of the axes are not found.
        """
        indices = self._lookup_indices(axes)
        if isinstance(indices, int):
            return self.axes[indices]
        elif indices is None:
            raise ValueError(f"Axis {axes} not found")
        else:
            result = []
            assert isinstance(axes, Sequence)
            for i, ax in zip(indices, axes):
                if i is None:
                    raise ValueError(f"Axis {ax} not found in {self.shape}")
                result.append(self.axes[i])
            return tuple(result)

    def __str__(self):
        # we consider the following cases:
        # * array is a tracer, in which case we want just the named shape (and an indication that it's a tracer)
        # * array is a jnp.ndarray, in which case we want the named shape and the array

        if is_jax_array_like(self.array):
            if isinstance(self.array, jax.core.Tracer):
                return f"NamedArray(Tracer<{self.dtype}{self.shape}>)"
            elif self.ndim <= 1:
                return f"NamedArray({self.dtype}{self.shape}, {self.array})"
            else:
                return f"NamedArray({self.dtype}{self.shape},\n{self.array})"
        else:
            return f"NamedArray(???{self.shape}, {self.array})"

    def __tree_pp__(self, **kwargs):
        # For Equinox's tree pretty printer
        import jax._src.pretty_printer as pp

        if kwargs.get("short_arrays", True) and is_jax_array_like(self.array):
            return pp.text(f"Named({self.dtype}{self.shape})")
        else:
            return pp.text(str(self))

    @overload
    def _lookup_indices(self, axis: AxisSelector) -> Optional[int]:  # type: ignore
        ...

    @overload
    def _lookup_indices(self, axis: Sequence[AxisSelector]) -> Tuple[Optional[int], ...]:
        ...

    def _lookup_indices(self, axis: AxisSelection) -> Union[Optional[int], Tuple[Optional[int], ...]]:
        """
        For a single axis, returns an int corresponding to the index of the axis.
        For multiple axes, returns a tuple of ints corresponding to the indices of the axes.

        If the axis is not present, returns None for that position
        """
        if isinstance(axis, Axis):
            ax_name = axis.name
            try:
                return self.axes.index(axis)
            except ValueError:
                try:
                    axis_index = index_where(lambda a: a.name == ax_name, self.axes)
                    if axis_index >= 0:
                        warnings.warn("Found axis with same name but different size.", UserWarning)
                    return axis_index
                except ValueError:
                    return None
        elif isinstance(axis, str):
            try:
                return index_where(lambda a: a.name == axis, self.axes)
            except ValueError:
                return None
        elif isinstance(axis, str):
            try:
                return index_where(lambda a: a.name == axis, self.axes)
            except ValueError:
                return None
        else:
            return tuple(self._lookup_indices(a) for a in axis)

    # Axis rearrangement
    @typing.overload
    def rearrange(self, axes: Sequence[AxisSelector | EllipsisType]) -> "NamedArray":
        """See [haliax.rearrange][] for details."""
        pass

    @typing.overload
    def rearrange(self, expression: str, **bindings: AxisSelector | int) -> "NamedArray":
        """See [haliax.rearrange][] for details."""
        pass

    def rearrange(self, *args, **kwargs) -> "NamedArray":
        """See [haliax.rearrange][] for details."""
        return haliax.rearrange(self, *args, **kwargs)

    def broadcast_to(self, axes: AxisSpec) -> "NamedArray":
        axes = ensure_tuple(axes)
        return haliax.broadcast_to(self, axes=axes)

    def broadcast_axis(self, axis: AxisSpec) -> "NamedArray":
        return haliax.broadcast_axis(self, axis=axis)

    def split(self, axis: AxisSelector, new_axes: Sequence[Axis]) -> Sequence["NamedArray"]:
        return haliax.split(self, axis=axis, new_axes=new_axes)

    def flatten_axes(self, old_axes: AxisSelection, new_axis: AxisSelector) -> "NamedArray":
        return haliax.flatten_axes(self, old_axes=old_axes, new_axis=new_axis)

    def unflatten_axis(self, axis: AxisSelector, new_axes: AxisSpec) -> "NamedArray":
        return haliax.unflatten_axis(self, axis=axis, new_axes=new_axes)

    def ravel(self, new_axis_name: AxisSelector) -> "NamedArray":
        return haliax.ravel(self, new_axis_name=new_axis_name)

    def flatten(self, new_axis_name: AxisSelector) -> "NamedArray":
        return haliax.flatten(self, new_axis_name=new_axis_name)

    def unbind(self, axis: AxisSelector) -> Sequence["NamedArray"]:
        return haliax.unbind(self, axis=axis)

    def rename(self, renames: Mapping[AxisSelector, AxisSelector]) -> "NamedArray":
        return haliax.rename(self, renames=renames)

    # slicing
    @typing.overload
    def slice(
        self, axis: AxisSelector, new_axis: Optional[AxisSelector] = None, start: int = 0, length: Optional[int] = None
    ) -> "NamedArray":
        ...

    @typing.overload
    def slice(
        self, start: Mapping[AxisSelector, int], length: Mapping[AxisSelector, Union[int, Axis]]
    ) -> "NamedArray":
        ...

    def slice(self, *args, **kwargs) -> "NamedArray":
        return haliax.slice(self, *args, **kwargs)

    def updated_slice(self, start: Mapping[AxisSelector, int], update: "NamedArray") -> "NamedArray":
        return haliax.updated_slice(self, start=start, update=update)

    def take(self, axis: AxisSelector, index: Union[int, "NamedArray"]) -> "NamedArray":
        return haliax.take(self, axis=axis, index=index)

    @property
    def at(self) -> "_NamedIndexUpdateHelper":
        """
                Named analog of [jax's at method](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html).

                Docs from the JAX docs:

                The at property provides a functionally pure equivalent of in-place array modifications.

                In particular:

        | Alternate syntax | Equivalent In-place expression |
        |------------------|-------------------------------|
        | `x = x.at[idx].set(y)` | `x[idx] = y` |
        | `x[idx] = y` | `x = x.at[idx].set(y)` |
        | `x = x.at[idx].add(y)` | `x[idx] += y`|
        | `x = x.at[idx].multiply(y)` | `x[idx] *= y`|
        | `x = x.at[idx].divide(y)` | `x[idx] /= y`|
        | `x = x.at[idx].power(y)` | `x[idx] **= y`|
        | `x = x.at[idx].min(y)` | `x[idx] = minimum(x[idx], y)`|
        | `x = x.at[idx].max(y)` | `x[idx] = maximum(x[idx], y)`|
        | `x = x.at[idx].apply(ufunc)` | `ufunc.at(x, idx)`|

        x = x.at[idx].get()


        x = x[idx]

        None of the x.at expressions modify the original x; instead they return a modified copy of x. However, inside a jit() compiled function, expressions like x = x.at[idx].set(y) are guaranteed to be applied in-place.

        Unlike NumPy in-place operations such as x[idx] += y, if multiple indices refer to the same location, all updates will be applied (NumPy would only apply the last update, rather than applying all updates.) The order in which conflicting updates are applied is implementation-defined and may be nondeterministic (e.g., due to concurrency on some hardware platforms).

        By default, JAX assumes that all indices are in-bounds. Alternative out-of-bound index semantics can be specified via the mode parameter (see below).


                Returns:

        """
        return _NamedIndexUpdateHelper(self)

    def __getitem__(self, idx: SliceSpec) -> "NamedArray":
        """Syntactic sugar for [haliax.index][], which is the actual implementation.

        Supports indexing like:

        >>> X = Axis("x", 10)
        >>> Y = Axis("y", 20)
        >>> arr = haliax.random.randint(jax.random.PRNGKey(0), (X, Y), 0, X.size)
        # slice with ints or slices
        >>> arr[{"x": 1, "y": slice(0,10,new_axis=2)}]
        >>> Z = Axis("z", 3)
        # so-called "advanced indexing" with NamedArrays.
        >>> index_arr = NamedArray(np.array([1, 2, 3]), Z)
        >>> arr[{"x": 1, "y": index_arr}]

        A shorthand is provided that works with Python's slicing syntax:
        >>> arr["x", :] == arr[{"x": slice(None)}]
        >>> arr["y", 1, "x", 2] == arr[{"y": 1, "x": 2}]

        Advanced indexing is implemented by broadcasting all index arrays to the same shape (using Haliax's
        usual broadcasting rules).

        This returns a NamedArray if any axes remain, or a scalar (0-dimensional) jnp.ndarray if all axes are indexed out.
        """
        idx_dict = _convert_index_expr_to_dict(idx)
        return index(self, idx_dict)

    # np.ndarray methods:
    def all(self, axis: Optional[AxisSelection] = None, *, where: Optional["NamedArray"] = None) -> "NamedArray":
        return haliax.all(self, axis=axis, where=where)

    def any(self, axis: Optional[AxisSelection] = None, *, where: Optional["NamedArray"] = None) -> "NamedArray":
        return haliax.any(self, axis=axis, where=where)

    def argmax(self, axis: Optional[AxisSelector] = None) -> "NamedArray":
        return haliax.argmax(self, axis=axis)

    def argmin(self, axis: Optional[AxisSelector]) -> "NamedArray":
        return haliax.argmin(self, axis=axis)

    def argsort(self, axis: AxisSelector) -> "NamedArray":
        return haliax.argsort(self, axis=axis)

    def astype(self, dtype) -> "NamedArray":
        return NamedArray(self.array.astype(dtype), self.axes)

    def clip(self, a_min=None, a_max=None) -> Any:
        return haliax.clip(self, a_min=a_min, a_max=a_max)

    def conj(self) -> "NamedArray":
        return NamedArray(self.array.conj(), self.axes)

    def conjugate(self) -> "NamedArray":
        return NamedArray(self.array.conjugate(), self.axes)

    def copy(self) -> "NamedArray":
        return NamedArray(self.array.copy(), self.axes)

    def cumprod(self, axis: AxisSelector, *, dtype=None) -> "NamedArray":
        return haliax.cumprod(self, axis=axis, dtype=dtype)

    def cumsum(self, axis: AxisSelector, *, dtype=None) -> "NamedArray":
        return haliax.cumsum(self, axis=axis, dtype=dtype)

    # Deprecated overload
    @typing.overload
    def dot(
        self, axis: Optional[AxisSelection], *b, precision: PrecisionLike = None, dot_general=jax.lax.dot_general
    ) -> "NamedArray":
        ...

    @typing.overload
    def dot(
        self,
        *args: "NamedArray",
        axis: Optional[AxisSelection],
        precision: PrecisionLike = None,
        dot_general=jax.lax.dot_general,
    ) -> "NamedArray":
        ...

    def dot(self, *args, **kwargs) -> "NamedArray":
        if "axis" in kwargs or len(args) == 0:
            return haliax.dot(self, *args, **kwargs)
        else:
            axis = args[0]
            args = args[1:]
            # We want to get the deprecation warning for this style
            return haliax.dot(axis, self, *args, **kwargs)

    @property
    def imag(self) -> "NamedArray":
        return NamedArray(self.array.imag, self.axes)

    def max(self, axis: Optional[AxisSelection] = None, *, where=None) -> "NamedArray":
        return haliax.max(self, axis=axis, where=where)

    def mean(
        self, axis: Optional[AxisSelection] = None, *, dtype=None, where: Optional["NamedArray"] = None
    ) -> "NamedArray":
        return haliax.mean(self, axis=axis, dtype=dtype, where=where)

    def min(self, axis: Optional[AxisSelection] = None, *, where: Optional["NamedArray"] = None) -> "NamedArray":
        return haliax.min(self, axis=axis, where=where)

    def prod(
        self, axis: Optional[AxisSelection] = None, *, dtype=None, where: Optional["NamedArray"] = None
    ) -> "NamedArray":
        return haliax.prod(self, axis=axis, dtype=dtype, where=where)

    def product(
        self, axis: Optional[AxisSelection] = None, *, dtype=None, where: Optional["NamedArray"] = None
    ) -> "NamedArray":
        return haliax.product(self, axis=axis, dtype=dtype, where=where)

    def ptp(self, axis: Optional[AxisSelection] = None) -> "NamedArray":
        return haliax.ptp(self, axis=axis)

    @property
    def real(self) -> "NamedArray":
        return NamedArray(self.array.real, self.axes)

    def round(self, decimals=0) -> "NamedArray":
        return haliax.round(self, decimals=decimals)

    def sort(self, axis: AxisSelector) -> Any:
        return haliax.sort(self, axis=axis)

    def std(
        self, axis: Optional[AxisSelection] = None, *, dtype=None, ddof=0, where: Optional["NamedArray"] = None
    ) -> "NamedArray":
        return haliax.std(self, axis=axis, dtype=dtype, ddof=ddof, where=where)

    def sum(
        self, axis: Optional[AxisSelection] = None, *, dtype=None, where: Optional["NamedArray"] = None
    ) -> "NamedArray":
        return haliax.sum(
            self,
            axis=axis,
            dtype=dtype,
            where=where,
        )

    def tobytes(self, order="C") -> Any:
        return self.array.tobytes(order=order)

    def tolist(self) -> Any:
        return self.array.tolist()

    def trace(self, axis1: AxisSelector, axis2: AxisSelector, offset=0, dtype=None) -> "NamedArray":
        return haliax.trace(self, offset=offset, axis1=axis1, axis2=axis2, dtype=dtype)

    def var(
        self, axis: Optional[AxisSelection] = None, dtype=None, ddof=0, *, where: Optional["NamedArray"] = None
    ) -> "NamedArray":
        return haliax.var(self, axis=axis, dtype=dtype, ddof=ddof, where=where)

    # operators

    # Comparisons
    def __lt__(self, other) -> "NamedArray":
        return haliax.less(self, other)

    def __le__(self, other) -> "NamedArray":
        return haliax.less_equal(self, other)

    def __eq__(self, other):
        # special case because Jax sometimes call == on
        # types when they're in PyTrees
        if self.array is None:
            return other.array is None

        if hasattr(other, "array") and other.array is None:
            return False

        return haliax.equal(self, other)

    def __ne__(self, other):
        return haliax.not_equal(self, other)

    def __gt__(self, other) -> "NamedArray":
        return haliax.greater(self, other)

    def __ge__(self, other) -> "NamedArray":
        return haliax.greater_equal(self, other)

    # Unary arithmetic

    def __neg__(self) -> "NamedArray":
        return haliax.negative(self)

    def __pos__(self) -> "NamedArray":
        return haliax.positive(self)

    def __abs__(self) -> "NamedArray":
        return haliax.absolute(self)

    def __invert__(self) -> "NamedArray":
        return haliax.invert(self)

    # Binary arithmetic

    def __add__(self, other) -> "NamedArray":
        return haliax.add(self, other)

    def __sub__(self, other) -> "NamedArray":
        return haliax.subtract(self, other)

    def __mul__(self, other) -> "NamedArray":
        return haliax.multiply(self, other)

    def __matmul__(self, other) -> "NamedArray":
        raise ValueError("matmul is too ambiguous with NamedArrays. Use dot instead.")

    def __truediv__(self, other) -> "NamedArray":
        return haliax.true_divide(self, other)

    def __floordiv__(self, other) -> "NamedArray":
        return haliax.floor_divide(self, other)

    def __mod__(self, other) -> "NamedArray":
        return haliax.mod(self, other)

    def __divmod__(self, other) -> "NamedArray":
        return haliax.divmod(self, other)

    def __pow__(self, other) -> "NamedArray":
        return haliax.power(self, other)

    def __lshift__(self, other) -> "NamedArray":
        return haliax.left_shift(self, other)

    def __rshift__(self, other) -> "NamedArray":
        return haliax.right_shift(self, other)

    def __and__(self, other) -> "NamedArray":
        return haliax.bitwise_and(self, other)

    def __xor__(self, other) -> "NamedArray":
        return haliax.bitwise_xor(self, other)

    def __or__(self, other) -> "NamedArray":
        return haliax.bitwise_or(self, other)

    def __radd__(self, other) -> "NamedArray":
        return haliax.add(other, self)

    def __rsub__(self, other) -> "NamedArray":
        return haliax.subtract(other, self)

    def __rmul__(self, other) -> "NamedArray":
        return haliax.multiply(other, self)

    def __rmatmul__(self, other):
        raise ValueError("Matrix multiplication is too ambiguous with NamedArrays. Use dot instead.")

    def __rtruediv__(self, other) -> "NamedArray":
        return haliax.true_divide(other, self)

    def __rfloordiv__(self, other) -> "NamedArray":
        return haliax.floor_divide(other, self)

    def __rmod__(self, other) -> "NamedArray":
        return haliax.mod(other, self)

    def __rdivmod__(self, other) -> "NamedArray":
        return haliax.divmod(other, self)

    def __rpow__(self, other) -> "NamedArray":
        return haliax.power(other, self)

    def __rlshift__(self, other) -> "NamedArray":
        return haliax.left_shift(other, self)

    def __rrshift__(self, other) -> "NamedArray":
        return haliax.right_shift(other, self)

    def __rand__(self, other) -> "NamedArray":
        return haliax.bitwise_and(other, self)

    def __rxor__(self, other) -> "NamedArray":
        return haliax.bitwise_xor(other, self)

    def __ror__(self, other) -> "NamedArray":
        return haliax.bitwise_or(other, self)

    def __bool__(self) -> bool:
        return bool(self.array)

    def __complex__(self) -> complex:
        return complex(self.array)

    def __int__(self) -> int:
        return int(self.array)

    def __float__(self) -> float:
        return float(self.array)


def take(array: NamedArray, axis: AxisSelector, index: Union[int, NamedArray]) -> NamedArray:
    """
    Selects elements from an array along an axis, by an index or by another named array

    if index is a NamedArray, then those axes are added to the output array
    """
    axis_index = array._lookup_indices(axis)
    if axis_index is None:
        raise ValueError(f"axis {axis} not found in {array}")

    axis = array.axes[axis_index]
    if isinstance(index, int):
        # just drop the axis
        new_array = jnp.take(array.array, index, axis=axis_index)
        new_axes = array.axes[:axis_index] + array.axes[axis_index + 1 :]
        return NamedArray(new_array, new_axes)
    else:
        if not jnp.issubdtype(index.dtype, jnp.integer):
            raise ValueError(f"Index must be an integer array, got {index.dtype}")
        # #13: should broadcast/autobatch take
        remaining_axes = eliminate_axes(array.axes, axis)
        # axis order is generally [array.axes[:axis_index], index.axes, array.axes[axis_index + 1 :]]
        # except that index.axes may overlap with array.axes
        overlapping_axes: AxisSpec = haliax.axis.overlapping_axes(remaining_axes, index.axes)

        if overlapping_axes:
            # if the eliminated axis is also in the index, we rename it to a dummy axis that we can broadcast over it
            need_to_use_dummy_axis = index._lookup_indices(axis.name) is not None
            if need_to_use_dummy_axis:
                index = index.rename({axis.name: "__DUMMY_" + axis.name})
            array = haliax.broadcast_to(array, index.axes, ensure_order=False, enforce_no_extra_axes=False)
            new_axes = eliminate_axes(array.axes, axis)
            index = haliax.broadcast_to(index, new_axes, ensure_order=True, enforce_no_extra_axes=True)

            axis_index = array._lookup_indices(axis)  # if it moved
            index_array = jnp.expand_dims(index.array, axis=axis_index)
            new_array = jnp.take_along_axis(array.array, index_array, axis=axis_index)
            new_array = jnp.squeeze(new_array, axis=axis_index)

            out = NamedArray(new_array, new_axes)
            if need_to_use_dummy_axis:
                out = out.rename({"__DUMMY_" + axis.name: axis.name})
            return out
        else:
            new_axes = array.axes[:axis_index] + index.axes + array.axes[axis_index + 1 :]
            new_array = jnp.take(array.array, index.array, axis=axis_index)

            # new axes come from splicing the old axis with
            return NamedArray(new_array, new_axes)


@typing.overload
def slice(
    array: NamedArray,
    axis: AxisSelector,
    new_axis: Optional[AxisSelector] = None,
    start: int = 0,
    length: Optional[int] = None,
) -> NamedArray:
    pass


@typing.overload
def slice(
    array: NamedArray,
    start: Mapping[AxisSelector, IntScalar],
    length: Optional[Mapping[AxisSelector, int]] = None,
) -> NamedArray:
    """
    Slices the array along the specified axes, replacing them with new axes (or a shortened version of the old one)

    Args:
        start (Mapping[AxisSelector, Union[int, jnp.ndarray]]): the start index of each axis to slice. If an int, the axis will be sliced at that index. If a NamedArray, the axis will be sliced at the indices in the NamedArray
        length (Mapping[AxisSelector, int]): the length of the dimension for that slice.
    """
    pass


def slice(array: NamedArray, *args, **kwargs) -> NamedArray:
    """
    Slices the array along the specified axis or axes, replacing them with new axes (or a shortened version of the old one)

    This method has two signatures:

    * `slice(array, axis, new_axis=None, start=0, length=None)`
    * `slice(array, start: Mapping[AxisSelector, IntScalar], length: Mapping[AxisSelector, int])`

    They both do similar things. The former slices an array along a single axis, replacing it with a new axis.
    The latter slices an array along multiple axes, replacing them with new axes.
    """
    if len(args) >= 1:
        if isinstance(args[0], Mapping):
            return _slice_new(array, *args, **kwargs)
        else:
            return _slice_old(array, *args, **kwargs)
    elif "axis" in kwargs:
        return _slice_old(array, **kwargs)
    else:
        return _slice_new(array, **kwargs)


def _slice_old(
    array: NamedArray,
    axis: AxisSelector,
    new_axis: Optional[AxisSelector] = None,
    start: int = 0,
    length: Optional[int] = None,
) -> NamedArray:
    """
    Selects elements from an array along an axis, either by an index or by another named array.
    This method offers an advantage over 'take' when a contiguous slice of an array is wanted.

    Args:
        axis (AxisSelector): The axis to slice.
        new_axis (str, optional): The name of the new axis that replaces the old one.
                                  If none, the old name will be used.
        start (int): The index at which the slice will start.
        length (int, optional): The length of the slice. Either new_axis must be an `Axis` or
                      `length` must be specified.

    Note:
        This method is basically a wrapper around jax.lax.dynamic_slice_in_dim.
    """
    axis_index = array._lookup_indices(axis)
    if axis_index is None:
        raise ValueError(f"axis {axis} not found in {array}")

    if length is None:
        if not isinstance(new_axis, Axis):
            raise ValueError("either new_axis must be an Axis or length must be specified")
        length = new_axis.size

    if isinstance(new_axis, str):
        new_axis = Axis(new_axis, length)
    elif new_axis is None:
        new_axis = array.axes[axis_index].resize(length)

    assert isinstance(new_axis, Axis)

    sliced = jax.lax.dynamic_slice_in_dim(array.array, start, length, axis=axis_index)
    new_axes = array.axes[:axis_index] + (new_axis,) + array.axes[axis_index + 1 :]
    # new axes come from splicing the old axis with
    return NamedArray(sliced, new_axes)


def _slice_new(
    array: NamedArray,
    start: Mapping[AxisSelector, Union[int, jnp.ndarray]],
    length: Mapping[AxisSelector, Union[int, Axis]],
) -> NamedArray:
    array_slice_indices = [0] * len(array.axes)
    new_axes = list(array.axes)
    new_lengths = [axis.size for axis in array.axes]

    for axis, s in start.items():
        axis_index = array._lookup_indices(axis_name(axis))
        if axis_index is None:
            raise ValueError(f"axis {axis} not found in {array}")

        array_slice_indices[axis_index] = s
        try:
            length_or_axis = length[axis]
        except KeyError:
            raise ValueError(f"length of axis {axis} not specified")

        if isinstance(length_or_axis, Axis):
            new_axis = length_or_axis
            ax_len = length_or_axis.size
        else:
            ax_len = length_or_axis
            new_axis = array.axes[axis_index].resize(ax_len)

        new_axes[axis_index] = new_axis
        new_lengths[axis_index] = ax_len

        total_length = array.axes[axis_index].size
        if isinstance(s, int) and isinstance(ax_len, int):
            if s + ax_len > total_length:
                raise ValueError(f"slice {s}:{s} + {ax_len} is out of bounds for axis {axis} of length {total_length}")

    sliced_array = jax.lax.dynamic_slice(array.array, array_slice_indices, new_lengths)

    return NamedArray(sliced_array, tuple(new_axes))


def updated_slice(
    array: NamedArray, start: Mapping[AxisSelector, Union[int, jnp.ndarray]], update: NamedArray
) -> NamedArray:
    """
    Updates a slice of an array with another array.

    Args:
        array (NamedArray): The array to update.
        start (Mapping[AxisSelector, Union[int, jnp.ndarray]]): The starting index of each axis to update.
        update (NamedArray): The array to update with.

    Returns:
        NamedArray: The updated array.
    """

    array_slice_indices = [0] * len(array.axes)
    for axis, s in start.items():
        axis_index = array._lookup_indices(axis_name(axis))
        if axis_index is None:
            raise ValueError(f"axis {axis} not found in {array}")
        array_slice_indices[axis_index] = s
        total_length = array.axes[axis_index].size
        update_axis = update._lookup_indices(axis_name(axis))

        if update_axis is None:
            raise ValueError(f"axis {axis} not found in {update}")
        # if s is a tracer we can't check the size
        if isinstance(s, int) and update.axes[update_axis].size + s > total_length:
            raise ValueError(
                f"update axis {axis} is too large to start at {s}. Array size is {total_length}, update size is"
                f" {update.axes[update_axis].size}"
            )

    # broadcasting here is a bit delicate because the sizes aren't necessarily the same
    # we need to broadcast the update array to the same axis names as the array we're updating, adding them as necessary
    broadcasted_axes = []
    for axis in array.axes:
        update_axis = update._lookup_indices(axis.name)
        if update_axis is None:
            broadcasted_axes.append(axis)
        else:
            broadcasted_axes.append(update.axes[update_axis])

    update = haliax.broadcast_to(update, broadcasted_axes, enforce_no_extra_axes=True)

    updated = jax.lax.dynamic_update_slice(array.array, update.array, array_slice_indices)
    return NamedArray(updated, array.axes)


def index(array: NamedArray, slices: Mapping[AxisSelector, NamedIndex]) -> NamedArray:
    """
    Selects elements from an array along an axis via index or another named array.

    This function is typically invoked using `array[...]` syntax. For instance,
    you might use `array[{"batch": slice(0, 10)}]` or `array["batch", 0:10]` to select the first 10 elements
    of the 'batch' axis.

    See Also:
        * [haliax.NamedArray.at][] for a functional equivalent of in-place array modifications.

    Returns:
        NamedArray or jnp.ndarray: A NamedArray is returned if there are any axes remaining after selection,
        otherwise a scalar (0-dimensional) jnp.ndarray is returned if all axes are indexed out.
    """
    # indices where we have array args
    new_axes, ordered_slices = _compute_new_axes_and_slices_for_index(array, slices)
    sliced, ordered_slices = _handle_dynamic_slices(array.array, ordered_slices)
    sliced = sliced[tuple(ordered_slices)]

    return haliax.named(sliced, new_axes)


def _compute_new_axes_and_slices_for_index(
    array, slices
) -> tuple[AxisSpec, list[py_slice | dslice | jnp.ndarray | int | list[int]]]:
    ordered_slices: list = [py_slice(None, None, None)] * len(array.axes)  # type: ignore
    kept_axes = [True] * len(array.axes)
    array_slice_indices = []

    for axis, slice_ in slices.items():
        axis_index = array._lookup_indices(axis)
        if axis_index is None:
            raise ValueError(f"axis {axis} not found in {array}")
        if isinstance(slice_, py_slice) or isinstance(slice_, dslice) or is_pallas_dslice(slice_):
            ordered_slices[axis_index] = slice_
            kept_axes[axis_index] = True
        elif isinstance(slice_, int):
            ordered_slices[axis_index] = slice_
            kept_axes[axis_index] = False
        elif isinstance(slice_, NamedArray):
            ordered_slices[axis_index] = slice_
            array_slice_indices.append(axis_index)
            kept_axes[axis_index] = False
        elif isinstance(slice_, list):
            # we'll let JAX complain if this is wrong
            ordered_slices[axis_index] = slice_
        elif isinstance(slice_, jnp.ndarray):
            # we allow this if it's a 0-d or 1-d array
            if slice_.ndim == 0:
                ordered_slices[axis_index] = slice_
                kept_axes[axis_index] = False
            elif slice_.ndim == 1:
                # we allow this if it's a 1-d array, in which case we treat it as sugar for NamedArray(slice_, sliced-axis)
                ordered_slices[axis_index] = haliax.named(slice_, axis_name(axis))
                kept_axes[axis_index] = False
                array_slice_indices.append(axis_index)
            else:
                raise ValueError(
                    f"Only 0-d or 1-d unnamed arrays can be used for indexing. Got {slice_} for axis {axis}"
                )
        else:
            raise ValueError(f"Only NamedArrays can be used for advanced indexing. Got {slice_} for axis {axis}")

    # advanced indexing
    if len(array_slice_indices) > 0:
        # this requires broadcasting
        broadcasted_arrays, broadcasted_axes = broadcast_arrays_and_return_axes(
            *[ordered_slices[i] for i in array_slice_indices], require_subset=False, ensure_order=True
        )
        # this is tricky. NumPy distinguishes two cases when mixing advanced and basic indexing:
        # https://numpy.org/doc/stable/user/basics.indexing.html#combining-advanced-and-basic-indexing
        # The first is when the advanced indices are all contiguous, and the second is when they are not.
        # (NB that integers count as advanced indices, so this is a bit more complicated than it seems.)
        # When contiguous, the new axes go in the same place as the advanced indices, and the old axes surround them.
        # When not contiguous, the new axes go to the *front* of the array, and the (other) old axes go after them.
        # To tell what case we're in, we check if the advanced indices are contiguous. We can figure out by looking
        # at the "kept_axes": the Falses are the advanced indices.

        # check to make sure we're not accidentally duplicating axes
        for axis_index in range(len(array.axes)):
            if kept_axes[axis_index]:
                if selects_axis(broadcasted_axes, array.axes[axis_index].name):
                    raise ValueError(f"Array Axis {array.axes[axis_index]} is present in slice {slices}")

        for axis_index, selector_array in zip(array_slice_indices, broadcasted_arrays):
            ordered_slices[axis_index] = selector_array.array

        is_advanced_contiguous = True
        first_advanced_index = index_where(lambda x: not x, kept_axes)
        last_advanced_index = first_advanced_index
        true_found = False
        for i in range(first_advanced_index, len(kept_axes)):
            # now find the first True. If any False comes after it, we're not contiguous
            if true_found:
                if not kept_axes[i]:
                    is_advanced_contiguous = False
                    break
            elif kept_axes[i]:
                true_found = True
                last_advanced_index = i - 1

        if not true_found:
            last_advanced_index = len(kept_axes) - 1

        if is_advanced_contiguous:
            # the advanced indices are contiguous, so we can just insert the new axes in the same place
            # as the advanced indices
            new_axes = array.axes[:first_advanced_index] + broadcasted_axes + array.axes[last_advanced_index + 1 :]
        else:
            # the advanced indices are not contiguous, so we need to insert the new axes at the front
            new_axes = broadcasted_axes + tuple(ax for i, ax in enumerate(array.axes) if kept_axes[i])
    else:
        new_axes = tuple(axis for axis, keep in zip(array.axes, kept_axes) if keep)

    new_axes = tuple(axis_name(ax) for ax in new_axes)
    return new_axes, ordered_slices


def _handle_dynamic_slices(array: jnp.ndarray, slices):
    """
    Helper function to handle dynamic slices in the array. These have to be handled with jax.lax.dynamic_slice,
    which is for when the start index is not known at compile time. (Sizes must always be known at compile time.)

    Notes:
        **MUTATES `slices` IN PLACE**

    Returns:
        array.array: the sliced array

    """
    indices_for_dslice = [0] * array.ndim
    lengths_for_dslice = list(array.shape)
    dslice_indices = []
    need_to_slice = False
    for axis_index, slice_ in enumerate(slices):
        if isinstance(slice_, dslice) or is_pallas_dslice(slice_):
            dslice_indices.append(axis_index)
            indices_for_dslice[axis_index] = slice_.start
            lengths_for_dslice[axis_index] = slice_.size
            need_to_slice = True
    if need_to_slice:
        array = jax.lax.dynamic_slice(array, indices_for_dslice, lengths_for_dslice)
        for i in dslice_indices:
            slices[i] = py_slice(None, None, None)
    return array, slices


def split(a: NamedArray, axis: AxisSelector, new_axes: Sequence[Axis]) -> Sequence[NamedArray]:
    """
    Splits an array along an axis into multiple arrays, one for each element of new_axes.

    Args:
        a (NamedArray): the array to split
        axis (AxisSelector): the axis to split along
        new_axes (Sequence[Axis]): the axes to split into. Must have the same total length as the axis being split.
    """
    # check the lengths of the new axes
    index = a._lookup_indices(axis)
    if index is None:
        raise ValueError(f"Axis {axis} not found in {a.axes}")

    total_len = sum(x.size for x in new_axes)
    if isinstance(axis, Axis):
        if total_len != axis.size:
            raise ValueError(
                f"The total length of the new axes {total_len} does not match the length of the axis {axis}"
            )

    # now we can split the array
    offsets = np.cumsum([0] + [x.size for x in new_axes])[1:-1]

    new_arrays = np.split(a.array, indices_or_sections=offsets, axis=index)
    ret_axes = [tuple(ax2 if not selects_axis(axis, ax2) else new_axis for ax2 in a.axes) for new_axis in new_axes]

    return [NamedArray(x, ax) for x, ax in zip(new_arrays, ret_axes)]


def unbind(array: NamedArray, axis: AxisSelector) -> List[NamedArray]:
    """
    Unbind an array along an axis, returning a list of NamedArrays, one for each position on that axis.
    Analogous to torch.unbind or np.rollaxis
    """
    axis_index = array._lookup_indices(axis)
    if axis_index is None:
        raise ValueError(f"axis {axis} not found in {array}")
    new_axes = array.axes[:axis_index] + array.axes[axis_index + 1 :]
    # this implementation maybe triggers an all-gather in pjit so no good
    # arrays = jnp.rollaxis(array.array, axis=axis_index, start=0)
    # instead we just loop over the axes pulling one out at a time
    axis_size = array.axes[axis_index].size
    arrays = [jnp.take(array.array, i, axis=axis_index) for i in range(axis_size)]

    return [haliax.auto_sharded(NamedArray(a, new_axes)) for a in arrays]


def roll(array: NamedArray, shift: Union[int, Tuple[int, ...]], axis: AxisSelection) -> NamedArray:
    """
    Roll an array along an axis or axes. Analogous to np.roll
    """
    axis_indices = array._lookup_indices(axis)
    if axis_indices is None:
        raise ValueError(f"axis {axis} not found in {array}")
    return NamedArray(jnp.roll(array.array, shift, axis_indices), array.axes)


def rename(array: NamedArray, renames: Mapping[AxisSelector, AxisSelector]) -> NamedArray:
    """
    Rename the axes of an array.
    Args:
        array: the array to rename
        renames: a mapping from old axes to new axes. If the value is a string, the axis will be renamed to that string.
    """
    for old, new in renames.items():
        if isinstance(old, Axis) and isinstance(new, Axis) and old.size != new.size:
            raise ValueError(f"Cannot rename axis {old} to {new}: size mismatch")

    def _rename(ax: AxisSelector) -> Axis:
        new_axis = renames.get(ax, None)
        if new_axis is None and isinstance(ax, Axis):
            new_axis_name = renames.get(ax.name, None)
            if isinstance(new_axis_name, str):
                new_axis = Axis(new_axis_name, ax.size)
                return new_axis
            elif isinstance(new_axis_name, Axis):
                if new_axis_name.size != ax.size:
                    raise ValueError(f"Cannot rename axis {ax} to {new_axis_name}: size mismatch")
                return new_axis_name
            else:
                return ax
        elif isinstance(new_axis, Axis):
            return new_axis
        else:
            assert isinstance(new_axis, str)
            ax_size = array.axis_size(ax)
            return Axis(new_axis, ax_size)

    new_axes = tuple(_rename(ax) for ax in array.axes)
    return NamedArray(array.array, new_axes)


def flatten_axes(array: NamedArray, old_axes: AxisSelection, new_axis: AxisSelector) -> NamedArray:
    """
    Merge a sequence of axes into a single axis. The new axis must have the same size as the product of the old axes.

    The new axis is always inserted starting at the index of the first old axis in theunderlying array.
    """
    old_axes = ensure_tuple(old_axes)
    old_axes = array.resolve_axis(old_axes)
    total_axis_size = haliax.axis_size(old_axes)

    if isinstance(new_axis, Axis):
        if new_axis.size != total_axis_size:
            raise ValueError(f"Cannot merge {old_axes} into {new_axis}: size mismatch")
    else:
        assert isinstance(new_axis, str)
        new_axis = Axis(new_axis, total_axis_size)

    if len(old_axes) == 0:
        # just unsqueeze the array
        new_array = jnp.expand_dims(array.array, axis=0)
        return NamedArray(new_array, (new_axis,) + array.axes)

    # ensure that the old_axes are contiguous
    # we basically ensure that the old_axes occur after the index of the first old_axis
    intermediate_axes: List[Axis] = []
    new_axes: List[Axis] = []
    index_of_first_old_axis = None
    for i, ax in enumerate(array.axes):
        if ax in old_axes:
            if index_of_first_old_axis is None:
                index_of_first_old_axis = i
                intermediate_axes.extend(old_axes)
                new_axes.append(new_axis)
            else:
                continue
        else:
            intermediate_axes.append(ax)
            new_axes.append(ax)

    array = array.rearrange(intermediate_axes)
    raw_array = array.array.reshape([ax.size for ax in new_axes])
    return NamedArray(raw_array, tuple(new_axes))


def unflatten_axis(array: NamedArray, axis: AxisSelector, new_axes: AxisSpec) -> NamedArray:
    """
    Split an axis into a sequence of axes. The old axis must have the same size as the product of the new axes.
    """
    old_index = array._lookup_indices(axis)
    if old_index is None:
        raise ValueError(f"Axis {axis} not found in {array}")

    axis_size = array.axes[old_index].size

    new_axes = ensure_tuple(new_axes)

    if len(new_axes) == 0:
        if axis_size == 1:
            # just remove the old axis, akin to squeeze
            new_array = jnp.squeeze(array.array, axis=old_index)
            resolved_new_axes = array.axes[:old_index] + array.axes[old_index + 1 :]
            return NamedArray(new_array, resolved_new_axes)
        else:
            raise ValueError("Must specify at least one axis to split")

    if axis_size != prod(ax.size for ax in new_axes):
        raise ValueError(f"Cannot split {axis} into {new_axes}: size mismatch")

    resolved_new_axes = array.axes[:old_index] + tuple(new_axes) + array.axes[old_index + 1 :]
    new_array = jnp.reshape(array.array, [ax.size for ax in resolved_new_axes])
    return NamedArray(new_array, resolved_new_axes)


def ravel(array: NamedArray, new_axis_name: AxisSelector) -> NamedArray:
    """
    Returns a flattened view of the array, with all axes merged into one
    """
    flattened = flatten_axes(array, array.axes, new_axis_name)
    return flattened


def flatten(array: NamedArray, new_axis_name: AxisSelector) -> NamedArray:
    """
    Returns a flattened view of the array, with all axes merged into one. Aliax for [haliax.ravel][]
    """
    return ravel(array, new_axis_name)


def named(a, axis: AxisSelection) -> NamedArray:
    """Creates a NamedArray from a numpy array and a list of axes."""
    a = jnp.asarray(a)
    axes = check_shape(a.shape, axis)
    return NamedArray(a, axes)


# Broadcasting Support
def _broadcast_order(a: NamedArray, b: NamedArray, require_subset: bool = True) -> Tuple[Axis, ...]:
    """
    Returns an ordering of axes for broadcasting a and b.

    If require_subset is True, then one of the array's axes must be a subset of the other's. This requirement is
    a bit stricter than a straightforward generalization of numpy's broadcasting rules, but I've been bitten by
    numpy's rules too many times.
    """
    broadcasted = _broadcast_axes(a.axes, b.axes, require_subset)
    if broadcasted is None:
        # TODO: decide under which conditions we want to allow broadcasting both arrays
        # maybe just add a context manager to allow it?
        raise ValueError(
            f"Cannot broadcast {a} and {b}: no subset relationship. "
            "If you want to broadcast anyway, use the broadcast_axis function to explicitly add axes"
        )
    return broadcasted


def _broadcast_axes(
    a_axes: Tuple[Axis, ...], b_axes: Tuple[Axis, ...], require_subset: bool = True
) -> Optional[Tuple[Axis, ...]]:
    if a_axes == b_axes:
        return a_axes
    if len(a_axes) == 0:
        return b_axes
    if len(b_axes) == 0:
        return a_axes

    if require_subset:
        # check if one is a subset of the other
        if set(b_axes).issubset(set(a_axes)):
            return a_axes
        elif set(a_axes).issubset(set(b_axes)):
            return b_axes
        else:
            return None

    a_size = prod(ax.size for ax in a_axes)
    b_size = prod(ax.size for ax in b_axes)
    if a_size < b_size:
        a_axes, b_axes = b_axes, a_axes

    # we want to order the axes in such a way that we minimize movement, or at least allow
    # large blocks to be memcpy'd when possible.
    # In particular, we'd like to avoid the case of reordering [Y, Z] + [X, Y, Z] -> [Y, Z, X] or other major reshuffles

    # here's what we do: we try to preserve the order of axes in the bigger array, and then stick the axes from the
    # other array on the front (because everything is row major)
    # this ensures we only have to move one array around

    return tuple(x for x in b_axes if x not in a_axes) + a_axes


def broadcast_to(
    a: NamedOrNumeric, axes: AxisSpec, ensure_order: bool = True, enforce_no_extra_axes: bool = True
) -> NamedArray:
    """
    Broadcasts a so that it has the given axes.
     If ensure_order is True (default), then the returned array will have the same axes in the same order as the given
     axes. Otherwise, the axes may not be moved if they are already in the array. The axes may not be contiguous however

    If enforce_no_extra_axes is True and the array has axes that are not in axes, then a ValueError is raised.
    """
    axes = ensure_tuple(axes)

    if not isinstance(a, NamedArray):
        a = named(jnp.asarray(a), ())

    assert isinstance(a, NamedArray)  # mypy gets confused

    if a.axes == axes:
        return a

    to_add = tuple(ax for ax in axes if ax not in a.axes)

    all_axes = to_add + a.axes

    if enforce_no_extra_axes and len(all_axes) != len(axes):
        raise ValueError(f"Cannot broadcast {a} to {axes}: extra axes present")

    extra_axes = tuple(ax for ax in a.axes if ax not in axes)

    # broadcast whatever we need to the front and reorder
    a_array = jnp.broadcast_to(a.array, [ax.size for ax in all_axes])
    a = NamedArray(a_array, all_axes)

    # if the new axes are already in the right order, then we're done
    if ensure_order and not _is_subsequence(axes, all_axes):
        a = a.rearrange(axes + extra_axes)

    return typing.cast(NamedArray, a)


def _is_subsequence(needle, haystack):
    needle_i = 0
    haystack_j = 0
    while needle_i < len(needle) and haystack_j < len(haystack):
        if needle[needle_i] == haystack[haystack_j]:
            needle_i += 1
        haystack_j += 1

    if needle_i < len(needle):
        return False
    return True


@overload
def broadcast_arrays(
    *arrays: NamedArray, require_subset: bool = True, ensure_order: bool = True
) -> Tuple[NamedArray, ...]:
    ...


@overload
def broadcast_arrays(
    *arrays: Optional[NamedOrNumeric], require_subset: bool = True, ensure_order: bool = True
) -> Tuple[Optional[NamedOrNumeric], ...]:
    ...


def broadcast_arrays(
    *arrays: Optional[NamedOrNumeric],
    require_subset: bool = True,
    ensure_order: bool = True,
) -> Tuple[Optional[NamedOrNumeric], ...]:
    """
    Broadcasts a sequence of arrays to a common set of axes.
    Args:
        *arrays: Arrays, Scalars, or None. If None, then None is returned. Scalars and None are supported for convenience.
        require_subset: If true, then one of the arrays must be a subset of the others. This is a bit stricter than numpy's broadcasting
            rules, but I've been bitten by numpy's rules too many times. False is looser than numpy's rules, and allows
            broadcasting any pair of arrays (so long as the axes don't overtly conflict with different sizes for the same
            name.)
        ensure_order: If true, then the returned arrays will be reordered to all have the same axes in the same order.

    Returns:
        The arrays, broadcast to a common set of axes, reordered if necessary.

    """
    return broadcast_arrays_and_return_axes(*arrays, require_subset=require_subset, ensure_order=ensure_order)[0]


@overload
def broadcast_arrays_and_return_axes(
    *arrays: NamedArray, require_subset: bool = True, ensure_order: bool = True
) -> Tuple[Tuple[NamedArray, ...], Tuple[Axis, ...]]:
    ...


@overload
def broadcast_arrays_and_return_axes(
    *arrays: NamedOrNumeric, require_subset: bool = True, ensure_order: bool = True
) -> Tuple[Tuple[NamedOrNumeric, ...], Tuple[Axis, ...]]:
    ...


@overload
def broadcast_arrays_and_return_axes(
    *arrays: Optional[NamedOrNumeric], require_subset: bool = True, ensure_order: bool = True
) -> Tuple[Tuple[Optional[NamedOrNumeric], ...], Tuple[Axis, ...]]:
    ...


def broadcast_arrays_and_return_axes(
    *arrays: Optional[NamedOrNumeric],
    require_subset: bool = True,
    ensure_order: bool = True,
) -> Tuple[Tuple[Optional[NamedOrNumeric], ...], Tuple[Axis, ...]]:
    """
    Broadcasts a sequence of arrays to a common set of axes.

    Args:
        arrays: NamedArray
            The arrays to broadcast
        require_subset: bool
            If True, then one of the arrays must be a subset of the other. This is a bit stricter than numpy's broadcasting
            rules, but I've been bitten by numpy's rules too many times. False is looser than numpy's rules, and allows
            broadcasting any pair of arrays (so long as the axes don't overtly conflict with different sizes for the same
            name.)
        ensure_order: bool
            If True, then the returned arrays will have the same axes in the same order as the given axes. Otherwise, the
            axes may not be moved.
    """
    if len(arrays) == 0:
        return ((), ())

    # sort the arrays by size, so that we use the biggest ones to broadcast the others
    # need to hold on to the order so we can return the arrays in the same order
    actual_arrays = [x for x in arrays if isinstance(x, NamedArray)]
    size_order = sorted(range(len(actual_arrays)), key=lambda i: actual_arrays[i].size, reverse=True)
    all_axes = [actual_arrays[i].axes for i in size_order]
    full_axes = ft.reduce(lambda a, b: _broadcast_axes(a, b, require_subset) if a is not None else None, all_axes)  # type: ignore
    if full_axes is None:
        raise ValueError(f"Cannot broadcast arrays {arrays}: no subset relationship")

    arrays = tuple(
        broadcast_to(a, full_axes, ensure_order=ensure_order) if isinstance(a, NamedArray) else a for a in arrays
    )

    return arrays, full_axes


# TODO: convert to AxisSelection?
def broadcast_axis(a: NamedArray, axis: AxisSpec) -> NamedArray:
    """
    Broadcasts `a`, ensuring that it has all the axes in `axis`.
     `broadcast_axis` is an alias for `broadcast_to(a, axis, enforce_no_extra_axes=False, ensure_order=True)`

     You typically use this function when you want to broadcast an array to a common set of axes.
    """
    if isinstance(axis, Axis) and axis in a.axes:
        return a

    return broadcast_to(a, axis, enforce_no_extra_axes=False, ensure_order=True)


def check_shape(jnp_shape: Sequence[int], hax_axes: AxisSelection) -> Tuple[Axis, ...]:
    """Check that the shape of a jax array matches the axes of a NamedArray"""
    axes: Tuple[AxisSelector, ...] = ensure_tuple(hax_axes)
    if len(jnp_shape) != len(axes):
        raise ValueError(f"Shape mismatch: jnp_shape={jnp_shape} hax_axes={hax_axes}")
    result_axes: List[Axis] = []
    for i in range(len(axes)):
        ax = axes[i]
        if isinstance(ax, Axis):
            if ax.size != jnp_shape[i]:
                raise ValueError(f"Shape mismatch: jnp_shape={jnp_shape} hax_axes={hax_axes}")
            result_axes.append(ax)  # type: ignore
        elif isinstance(ax, str):
            result_axes.append(Axis(ax, jnp_shape[i]))
        else:
            raise ValueError(f"Invalid axis spec: {ax}")

    return tuple(result_axes)


def flatten_all_axes_but(
    a: NamedArray, axis_name: str, axis: AxisSelection, reorder_to_front: bool = False
) -> tuple[NamedArray, Callable[[NamedArray], NamedArray]]:
    """
    Flattens all axes of `a` except for `axes`.
    The flattened axes are merged into a single axis with the name `axis_name`.

    Also returns a function to restore the original axes. This function takes a NamedArray with at least the flattened axes
    and returns a NamedArray with an order that is broadly consistent with the original order of the axes.

    On TPU, this operation should be free as long as this doesn't impact the last two dims.

    Usually you can just use vmap, but sometimes you actually want there to be exactly 1 batch axis,
    or it's a pain to do a bunch of vmaps

    """

    result = a.flatten_axes(axis, axis_name)

    if reorder_to_front:
        result = result.rearrange((axis_name, ...))

    old_axes = a.axes
    axis = a.resolve_axis(axis)

    del a

    def unflatten(new_a: NamedArray) -> NamedArray:
        # we want to restore the array to an order that is as consistent as possible with the original order
        new_a = new_a.unflatten_axis(axis_name, axis)
        if new_a.axes == old_axes:
            return new_a

        axes_from_first_array_present_in_new = haliax.axis.replace_missing_with_ellipsis(
            haliax.axis_name(old_axes), new_a.axes
        )
        out_axes = haliax.axis.rearrange_for_partial_order(axes_from_first_array_present_in_new, new_a.axes)
        return new_a.rearrange(out_axes)

    return result, unflatten


class _NamedIndexUpdateHelper:
    def __init__(self, array: NamedArray):
        self.array = array

    def __getitem__(self, slices: SliceSpec) -> "_NamedIndexUpdateRef":
        return _NamedIndexUpdateRef(self.array, _convert_index_expr_to_dict(slices))


class _NamedIndexUpdateRef:
    def __init__(self, array: NamedArray, slices: SliceSpec):
        self._array = array
        self._slices = slices

    def get(
        self,
        *,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: Optional[GatherScatterModeStr] = None,
        fill_value: Optional[Scalar] = None,
    ) -> NamedArray:
        slices, sliced_axes = _raw_indices_for_at(self._array, self._slices)
        new_array = self._array.array.at[tuple(slices)].get(
            indices_are_sorted=indices_are_sorted, unique_indices=unique_indices, mode=mode, fill_value=fill_value
        )
        return NamedArray(new_array, sliced_axes)

    def set(
        self,
        update: NamedOrNumeric,
        *,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: Optional[GatherScatterModeStr] = None,
    ) -> NamedArray:
        slices, sliced_axes = _raw_indices_for_at(self._array, self._slices)
        update = haliax.broadcast_to(update, sliced_axes, enforce_no_extra_axes=True)
        new_array = self._array.array.at[tuple(slices)].set(
            update.array, indices_are_sorted=indices_are_sorted, unique_indices=unique_indices, mode=mode
        )
        return NamedArray(new_array, self._array.axes)

    def add(
        self,
        update: NamedOrNumeric,
        *,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: Optional[GatherScatterModeStr] = None,
    ) -> NamedArray:
        slices, sliced_axes = _raw_indices_for_at(self._array, self._slices)
        update = haliax.broadcast_to(update, sliced_axes, enforce_no_extra_axes=True)
        new_array = self._array.array.at[tuple(slices)].add(
            update.array, indices_are_sorted=indices_are_sorted, unique_indices=unique_indices, mode=mode
        )
        return NamedArray(new_array, self._array.axes)

    def multiply(
        self,
        update: NamedOrNumeric,
        *,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: Optional[GatherScatterModeStr] = None,
    ) -> NamedArray:
        slices, sliced_axes = _raw_indices_for_at(self._array, self._slices)
        update = haliax.broadcast_to(update, sliced_axes, enforce_no_extra_axes=True)
        new_array = self._array.array.at[tuple(slices)].multiply(
            update.array, indices_are_sorted=indices_are_sorted, unique_indices=unique_indices, mode=mode
        )
        return NamedArray(new_array, self._array.axes)

    def divide(
        self,
        update: NamedOrNumeric,
        *,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: Optional[GatherScatterModeStr] = None,
    ) -> NamedArray:
        slices, sliced_axes = _raw_indices_for_at(self._array, self._slices)
        update = haliax.broadcast_to(update, sliced_axes, enforce_no_extra_axes=True)
        new_array = self._array.array.at[tuple(slices)].divide(
            update.array, indices_are_sorted=indices_are_sorted, unique_indices=unique_indices, mode=mode
        )
        return NamedArray(new_array, self._array.axes)

    def max(
        self,
        update: NamedOrNumeric,
        *,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: Optional[GatherScatterModeStr] = None,
    ) -> NamedArray:
        slices, sliced_axes = _raw_indices_for_at(self._array, self._slices)
        update = haliax.broadcast_to(update, sliced_axes, enforce_no_extra_axes=True)
        new_array = self._array.array.at[tuple(slices)].max(
            update.array, indices_are_sorted=indices_are_sorted, unique_indices=unique_indices, mode=mode
        )
        return NamedArray(new_array, self._array.axes)

    def min(
        self,
        update: NamedOrNumeric,
        *,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: Optional[GatherScatterModeStr] = None,
    ) -> NamedArray:
        slices, sliced_axes = _raw_indices_for_at(self._array, self._slices)
        update = haliax.broadcast_to(update, sliced_axes, enforce_no_extra_axes=True)
        new_array = self._array.array.at[tuple(slices)].min(
            update.array, indices_are_sorted=indices_are_sorted, unique_indices=unique_indices, mode=mode
        )
        return NamedArray(new_array, self._array.axes)

    def power(
        self,
        update: NamedOrNumeric,
        *,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: Optional[GatherScatterModeStr] = None,
    ) -> NamedArray:
        slices, sliced_axes = _raw_indices_for_at(self._array, self._slices)
        update = haliax.broadcast_to(update, sliced_axes, enforce_no_extra_axes=True)
        new_array = self._array.array.at[tuple(slices)].power(
            update.array, indices_are_sorted=indices_are_sorted, unique_indices=unique_indices, mode=mode
        )
        return NamedArray(new_array, self._array.axes)

    def apply(
        self,
        func,
        *,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: Optional[GatherScatterModeStr] = None,
    ) -> NamedArray:
        # It's not really documented, but func can be any callable that takes a scalar array and returns a scalar array
        slices, sliced_axes = _raw_indices_for_at(self._array, self._slices)
        new_array = self._array.array.at[tuple(slices)].apply(
            func, indices_are_sorted=indices_are_sorted, unique_indices=unique_indices, mode=mode
        )
        return NamedArray(new_array, self._array.axes)


def _raw_indices_for_at(array, indexes):
    sliced_axes, ordered_slices = _compute_new_axes_and_slices_for_index(array, indexes)
    del sliced_axes
    # this isn't the fastest (it does the _compute_new_axes_and_slices_for_index twice)
    # but it's easy
    _sliced = index(array, indexes)
    # we have to handle dslices differently than for normal indexing, because we can't use
    # extra dynamic_slices...
    # we'd like to just replace these with iota, but we have account for broadcasting semantics
    # for the other arrays
    dslice_sizes = tuple(x.size for x in ordered_slices if isinstance(x, dslice) or is_pallas_dslice(x))  # type: ignore
    current_array_slice_shape = next((x.shape for x in ordered_slices if is_jax_array_like(x)), None)  # type: ignore
    dims_to_expand = list(range(len(dslice_sizes)))
    if current_array_slice_shape is not None:
        iota_shape = dslice_sizes + current_array_slice_shape
    else:
        iota_shape = dslice_sizes

    def iota_for_dslice(dslice, cur_dynamic_slice):
        return jax.lax.broadcasted_iota(int, iota_shape, cur_dynamic_slice) + dslice.start

    if len(dslice_sizes) > 0:
        cur_dynamic_slice = 0
        for i in range(len(ordered_slices)):
            if isinstance(ordered_slices[i], dslice) or is_pallas_dslice(ordered_slices[i]):
                ordered_slices[i] = iota_for_dslice(ordered_slices[i], cur_dynamic_slice)
                cur_dynamic_slice += 1
            elif is_jax_array_like(ordered_slices[i]):
                # prepend array slices with one 1 for each dynamic slice
                ordered_slices[i] = jnp.expand_dims(ordered_slices[i], axis=dims_to_expand)

        assert cur_dynamic_slice == len(dslice_sizes)

    # ok the ordered slices are now correct
    return ordered_slices, _sliced.axes


def _convert_index_expr_to_dict(idx) -> dict[AxisSelector, NamedIndex]:
    if isinstance(idx, tuple):
        if len(idx) == 1:
            idx = idx[0]
        else:
            if len(idx) % 2 != 0:
                raise ValueError(
                    "Must provide an even number of arguments to __getitem__ when using the shorthand syntax."
                )
            idx = {idx[i]: idx[i + 1] for i in range(0, len(idx), 2)}
    elif isinstance(idx, dict):
        pass
    else:
        raise ValueError(f"Invalid index type {type(idx)}")
    return idx


__all__ = [
    "NamedArray",
    "named",
    "slice",
    "updated_slice",
    "index",
    "take",
    "split",
    "flatten_axes",
    "unflatten_axis",
    "ravel",
    "flatten",
    "unbind",
    "roll",
    "_broadcast_order",
    "broadcast_to",
    "broadcast_axis",
    "broadcast_arrays",
    "enable_shape_checks",
    "are_shape_checks_enabled",
    "check_shape",
    "flatten_all_axes_but",
]
