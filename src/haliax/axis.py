import typing
from dataclasses import dataclass
from math import prod
from types import EllipsisType
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union, overload

import equinox as eqx

from haliax.util import ensure_tuple

from ._src.util import index_where


@dataclass(frozen=True)
class Axis:
    """Axis is a dataclass that represents an axis of an NamedArray. It has a name and a size."""

    name: str
    size: int

    def alias(self, new_name: str):
        return Axis(new_name, self.size)

    def resize(self, size) -> "Axis":
        return Axis(self.name, size)

    def __str__(self):
        return f"{self.name}({self.size})"


AxisSelector = Union[Axis, str]
"""AxisSelector is a type that can be used to select a single axis from an array. str or Axis"""
AxisSelection = Union[AxisSelector, Sequence[AxisSelector]]
"""AxisSelection is a type that can be used to select multiple axes from an array. str, Axis, or sequence of mixed
str and Axis"""
AxisSpec = Union[Axis, Sequence[Axis]]
"""AxisSpec is a type that can be used to specify the axes of an array, usually for creation or adding a new axis
 whose size can't be determined another way. Axis or sequence of Axis"""
ShapeDict = Mapping[str, int]
"""ShapeDict is a type that can be used to specify the axes of an array, usually for creation or adding a new axis"""
PartialShapeDict = Mapping[str, Optional[int]]
"""Similar to an AxisSelection, in dict form."""


PartialAxisSpec = Sequence[EllipsisType | AxisSelector]
"""Used for rearrange and dot. A tuple of AxisSelectors and Ellipsis. Ellipsis means "any number of axes."
Some functions may require that the Ellipsis is present at most once, while others may allow it to be present
multiple times.
"""


def selects_axis(selector: AxisSelection, selected: AxisSelection) -> bool:
    """Returns true if the selector has every axis in selected and, if dims are given, that they match"""
    if isinstance(selector, Axis) or isinstance(selector, str):
        selected = ensure_tuple(selected)
        try:
            index = index_where(lambda ax: is_axis_compatible(ax, selector), selected)  # type: ignore
            return index >= 0
        except ValueError:
            return False

    selector_dict = axis_spec_to_shape_dict(selector)

    selected_tuple = ensure_tuple(selected)  # type: ignore
    for ax in selected_tuple:
        if isinstance(ax, Axis):
            selector_size = selector_dict.get(ax.name, _Sentinel)
            if selector_size is not None and selector_size != ax.size:
                return False
        elif isinstance(ax, str):
            if ax not in selector_dict:
                return False
        else:
            raise ValueError(f"Invalid axis spec: {ax}")

    return True


class _Sentinel:
    ...


def is_axis_compatible(ax1: AxisSelector, ax2: AxisSelector):
    """
    Returns true if the two axes are compatible, meaning they have the same name and, if both are Axis, the same size
    """
    if isinstance(ax1, str):
        if isinstance(ax2, str):
            return ax1 == ax2
        return ax1 == ax2.name
    if isinstance(ax2, str):
        return ax1.name == ax2
    return ax1.name == ax2.name


@overload
def axis_spec_to_shape_dict(axis_spec: AxisSpec) -> Dict[str, int]:  # type: ignore
    ...


@overload
def axis_spec_to_shape_dict(axis_spec: AxisSelection) -> Dict[str, Optional[int]]:  # type: ignore
    ...


def axis_spec_to_shape_dict(axis_spec: AxisSelection) -> Dict[str, Optional[int]]:  # type: ignore
    spec = ensure_tuple(axis_spec)  # type: ignore
    shape_dict: Dict[str, Optional[int]] = {}
    for ax in spec:
        if isinstance(ax, Axis):
            shape_dict[ax.name] = ax.size
        elif isinstance(ax, str):
            shape_dict[ax] = None
        else:
            raise ValueError(f"Invalid axis spec: {ax}")

    return shape_dict


def _dict_to_spec(axis_spec: Mapping[str, Optional[int]]) -> AxisSelection:
    return tuple(Axis(name, size) if size is not None else name for name, size in axis_spec.items())


@overload
def concat_axes(a1: AxisSpec, a2: AxisSpec) -> AxisSpec:
    pass


@overload
def concat_axes(a1: AxisSelection, a2: AxisSelection) -> AxisSelection:
    pass


def concat_axes(a1: AxisSelection, a2: AxisSelection) -> AxisSelection:
    """Concatenates two AxisSpec. Raises ValueError if any axis is present in both specs"""

    if isinstance(a1, Axis) and isinstance(a2, Axis):
        if axis_name(a1) == axis_name(a2):
            raise ValueError(f"Axis {a1} specified twice")
        return (a1, a2)
    else:
        a1 = ensure_tuple(a1)
        a2 = ensure_tuple(a2)

        a1_names = [axis_name(ax) for ax in a1]
        a2_names = [axis_name(ax) for ax in a2]

        if len(set(a1_names) & set(a2_names)) > 0:
            overlap = [ax for ax in a1_names if ax in a2_names]
            raise ValueError(f"AxisSpecs overlap! {' '.join(str(x) for x in overlap)}")
        return a1 + a2


@typing.overload
def union_axes(a1: AxisSpec, a2: AxisSpec) -> AxisSpec:
    ...


@typing.overload
def union_axes(a1: AxisSelection, a2: AxisSelection) -> AxisSelection:
    ...


def union_axes(a1: AxisSelection, a2: AxisSelection) -> AxisSelection:
    """
    Similar to concat_axes, but allows axes to be specified multiple times. The resulting AxisSpec will have the
    order of just concatenating each axis spec, but with any duplicate axes removed.

    Raises if any axis is present in both specs with different sizes
    """
    a1 = ensure_tuple(a1)
    a2 = ensure_tuple(a2)

    a1_dict = axis_spec_to_shape_dict(a1)
    a2_dict = axis_spec_to_shape_dict(a2)

    for ax, sz in a2_dict.items():
        if ax in a1_dict:
            if sz is not None and a1_dict[ax] is not None and sz != a1_dict[ax]:
                raise ValueError(f"Axis {ax} present in both specs with different sizes")
        else:
            a1_dict[ax] = sz

    return _dict_to_spec(a1_dict)


@overload
def eliminate_axes(axis_spec: AxisSpec, axes: AxisSelection) -> Tuple[Axis, ...]:  # type: ignore
    ...


@overload
def eliminate_axes(axis_spec: AxisSelection, axes: AxisSelection) -> AxisSelection:  # type: ignore
    ...


def eliminate_axes(axis_spec: AxisSelection, to_remove: AxisSelection) -> AxisSelection:  # type: ignore
    """Returns a new axis spec that is the same as the original, but without any axes in axes. Raises if any axis in to_remove is
    not present in axis_spec"""
    to_remove = ensure_tuple(to_remove)
    axis_spec_dict = axis_spec_to_shape_dict(axis_spec)
    for ax in to_remove:
        name = axis_name(ax)
        if name not in axis_spec_dict:
            raise ValueError(f"Axis {name} not present in axis spec {axis_spec}")
        del axis_spec_dict[name]

    return _dict_to_spec(axis_spec_dict)


@typing.overload
def without_axes(axis_spec: AxisSpec, to_remove: AxisSelection) -> AxisSpec:  # type: ignore
    ...


@typing.overload
def without_axes(axis_spec: AxisSelection, to_remove: AxisSelection) -> AxisSelection:  # type: ignore
    """As eliminate_axes, but does not raise if any axis in to_remove is not present in axis_spec"""


def without_axes(axis_spec: AxisSelection, to_remove: AxisSelection) -> AxisSelection:  # type: ignore
    """As eliminate_axes, but does not raise if any axis in to_remove is not present in axis_spec"""

    to_remove = ensure_tuple(to_remove)
    axis_spec_dict = axis_spec_to_shape_dict(axis_spec)
    for ax in to_remove:
        name = axis_name(ax)
        if name in axis_spec_dict:
            del axis_spec_dict[name]

    return _dict_to_spec(axis_spec_dict)


@typing.overload
def unsize_axes(axis_spec: AxisSelection, to_unsize: AxisSelection) -> AxisSelection:
    ...


@typing.overload
def unsize_axes(axis_spec: AxisSelection) -> AxisSelection:
    ...


def unsize_axes(axis_spec: AxisSelection, to_unsize: Optional[AxisSelection] = None) -> AxisSelection:
    """
    This function is used to remove the sizes of axes in an axis spec.
    There are two overloads:
    - If to_unsize is None, then all axes in axis_spec will be unsized
    - If to_unsize is not None, then all axes in to_unsize will be unsized. Raises if any axis in to_unsize is not present in axis_spec
    """

    if to_unsize is None:
        return tuple(axis_name(ax) for ax in ensure_tuple(axis_spec))  # type: ignore

    to_unsize = ensure_tuple(to_unsize)
    axis_spec_dict: dict[str, Optional[int]] = axis_spec_to_shape_dict(axis_spec)  # type: ignore
    for ax in to_unsize:
        name = axis_name(ax)
        if name not in axis_spec_dict:
            raise ValueError(f"Axis {name} not present in axis spec {axis_spec}")
        axis_spec_dict[name] = None

    return _dict_to_spec(axis_spec_dict)


@overload
def replace_axis(axis_spec: AxisSpec, old: AxisSelector, new: AxisSpec) -> AxisSpec:
    ...


@overload
def replace_axis(axis_spec: AxisSelection, old: AxisSelector, new: AxisSelection) -> AxisSelection:
    ...


def replace_axis(axis_spec: AxisSelection, old: AxisSelector, new: AxisSelection) -> AxisSelection:
    """Returns a new axis spec that is the same as the original, but with any axes in old replaced with new. Raises if old is
    not present in axis_spec"""
    axis_spec = ensure_tuple(axis_spec)
    index_of_old = index_where(lambda ax: is_axis_compatible(ax, old), axis_spec)

    if index_of_old < 0:
        raise ValueError(f"Axis {old} not present in axis spec {axis_spec}")

    return axis_spec[:index_of_old] + ensure_tuple(new) + axis_spec[index_of_old + 1 :]  # type: ignore


@overload
def overlapping_axes(ax1: AxisSpec, ax2: AxisSelection) -> Tuple[Axis, ...]:
    ...


@overload
def overlapping_axes(ax1: AxisSelection, ax2: AxisSpec) -> Tuple[Axis, ...]:
    ...


@overload
def overlapping_axes(ax1: AxisSelection, ax2: AxisSelection) -> Tuple[AxisSelector, ...]:
    ...


def overlapping_axes(ax1: AxisSelection, ax2: AxisSelection) -> Tuple[AxisSelector, ...]:
    """Returns a tuple of axes that are present in both ax1 and ax2.
    The returned order is the same as ax1.
    """
    ax2_dict = axis_spec_to_shape_dict(ax2)
    out: List[AxisSelector] = []
    ax1 = ensure_tuple(ax1)

    for ax in ax1:
        if isinstance(ax, Axis):
            if ax.name in ax2_dict:
                sz = ax2_dict[ax.name]
                if sz is not None and sz != ax.size:
                    raise ValueError(f"Axis {ax.name} has different sizes in {ax1} and {ax2}")
                out.append(ax)
        elif isinstance(ax, str):
            if ax in ax2_dict:
                ax_sz = ax2_dict[ax]
                if ax_sz is not None:
                    out.append(Axis(ax, ax_sz))
                else:
                    out.append(ax)
        else:
            raise ValueError(f"Invalid axis spec: {ax}")

    return tuple(out)


@overload
def axis_name(ax: AxisSelector) -> str:  # type: ignore
    ...


@overload
def axis_name(ax: Sequence[AxisSelector]) -> Tuple[str, ...]:  # type: ignore
    ...


def axis_name(ax: AxisSelection) -> Union[str, Tuple[str, ...]]:
    """
    Returns the name of the axis. If ax is a string, returns ax. If ax is an Axis, returns ax.name
    """

    def _ax_name(ax: AxisSelector) -> str:
        if isinstance(ax, Axis):
            return ax.name
        else:
            return ax

    if isinstance(ax, (Axis, str)):
        return _ax_name(ax)
    else:
        return tuple(_ax_name(x) for x in ax)


def axis_size(ax: AxisSpec) -> int:
    """
    Returns the size of the axis or the product of the sizes of the axes in the axis spec
    """

    if isinstance(ax, Axis):
        return ax.size
    else:
        return prod(axis.size for axis in ensure_tuple(ax))  # type: ignore


class dslice(eqx.Module):
    """
    Dynamic slice, comprising a (start, length) pair. Also aliased as ds.

    Normal numpy-isms like a[i:i+16] don't work in Jax inside jit, because slice doesn't like tracers and JAX
    can't see that the slice is constant. This is a workaround that lets you do a[dslice(i, 16)] or even a[ds(i, 16)]
    instead.

    This class's name is taken from [jax.experimental.pallas.dslice][].
    """

    start: int
    size: int

    def to_slice(self) -> slice:
        return slice(self.start, self.start + self.size)

    def __init__(self, start: int, length: Union[int, Axis]):
        """
        As a convenience, if length is an Axis, it will be converted to `length.size`
        Args:
            start:
            length:
        """
        self.start = start
        if isinstance(length, Axis):
            self.size = length.size
        else:
            self.size = length

    @staticmethod
    def block(idx: int, size: int) -> "dslice":
        """
        Returns a dslice that selects a single block of size `size` starting at `idx`
        """
        return dslice(idx * size, size)


ds: typing.TypeAlias = dslice


def dblock(idx: int, size: int) -> dslice:
    """
    Returns a dslice that selects a single block of size `size` starting at `idx`
    """
    return dslice(idx * size, size)


Ax = typing.TypeVar("Ax", AxisSelector, Axis)


def rearrange_for_partial_order(partial_order: PartialAxisSpec, axes: tuple[Ax, ...]) -> tuple[Ax, ...]:
    """Rearrange the axes to fit the provided partial order.
    Uses a greedy algorithm that tries to keep elements in roughly the same order they came in
     (subject to the partial order), but moves them to the earliest slot that is after all prior axes
     in the original order.
     The exact behavior of this function is not guaranteed to be stable, but it should be stable
     for most reasonable use cases. If you really need a specific order, you should provide a full
     order instead of a partial order.
    """

    if partial_order == (Ellipsis,):
        return axes

    spec = axis_spec_to_shape_dict(axes)

    def as_axis(ax_name: str) -> Ax:
        if spec[ax_name] is None:
            return ax_name  # type: ignore
        else:
            return Axis(ax_name, spec[ax_name])  # type: ignore

    if Ellipsis not in partial_order:
        pa: tuple[AxisSelector, ...] = partial_order  # type: ignore
        if set(axis_name(a) for a in pa) != set(spec.keys()) or len(pa) != len(spec.keys()):
            raise ValueError(
                "Partial order must be a permutation of the axes if no ellipsis is provided."
                f" However {pa} is not a permutation of {axes}"
            )

        # reorder axes to match partial order
        return tuple(as_axis(axis_name(name)) for name in pa)

    partial_order_names = [axis_name(s) for s in partial_order if s is not ...]

    uncovered_ordered_elements = set(partial_order_names)

    if len(partial_order_names) != len(uncovered_ordered_elements):
        raise ValueError("Partial order must not contain duplicate elements")

    # replace ... with [], which is where we'll put the remaining axes

    out_order = [[axis_name(a)] if a is not ... else [] for a in partial_order]

    # now we'll fill in the ordered elements
    target_pos = index_where(lambda x: x == [], out_order)

    for ax in axes:
        ax_name = axis_name(ax)
        if ax_name in uncovered_ordered_elements:
            uncovered_ordered_elements.remove(ax_name)
            # already in the right place
            # update target_pos to come after this if possible
            try:
                this_pos = index_where(lambda x: ax_name in x, out_order)
                # find first empty slot after this_pos. prefer not to go backwards
                this_pos = max(this_pos + 1, target_pos)
                target_pos = index_where(lambda x: x == [], out_order, start=this_pos)
            except ValueError:
                # leave it where it is
                pass
        elif ax_name in partial_order_names:
            raise ValueError(f"Axis {ax_name} appears multiple times in the partial order")
        else:
            # this can appear in any ... slot. our heuristic is to put it in the first
            # slot that comes after the most recently seen ordered element
            out_order[target_pos].append(ax_name)

    if len(uncovered_ordered_elements) > 0:
        raise ValueError(f"The following axes are not present in output: {' '.join(uncovered_ordered_elements)}")

    # now we have a list of lists of axis names. we need to flatten it and convert to axes
    return tuple(as_axis(name) for name in sum(out_order, []))


def replace_missing_with_ellipsis(ax1: AxisSelection, ax2: AxisSelection) -> PartialAxisSpec:
    """Returns ax1, except that:

    * any axis not in ax2 is replaced with Ellipsis
    * if ax1 has a str axis where ax2 has an Axis, then it is replaced with the Axis

    Raises if ax1 and ax2 have any axes with the same name but different sizes
    """
    ax2_dict = axis_spec_to_shape_dict(ax2)
    out: List[AxisSelector | EllipsisType] = []
    ax1 = ensure_tuple(ax1)

    for ax in ax1:
        if isinstance(ax, Axis):
            if ax.name in ax2_dict:
                sz = ax2_dict[ax.name]
                if sz is not None and sz != ax.size:
                    raise ValueError(f"Axis {ax.name} has different sizes in {ax1} and {ax2}")
                out.append(ax)
            else:
                out.append(Ellipsis)
        elif isinstance(ax, str):
            if ax in ax2_dict:
                ax_sz = ax2_dict[ax]
                if ax_sz is not None:
                    out.append(Axis(ax, ax_sz))
                else:
                    out.append(ax)
            else:
                out.append(Ellipsis)
        else:
            raise ValueError(f"Invalid axis spec: {ax}")

    return tuple(out)


__all__ = [
    "Axis",
    "AxisSelector",
    "AxisSelection",
    "AxisSpec",
    "PartialAxisSpec",
    "PartialShapeDict",
    "ShapeDict",
    "axis_name",
    "axis_size",
    "concat_axes",
    "union_axes",
    "axis_spec_to_shape_dict",
    "ds",
    "dslice",
    "dblock",
    "eliminate_axes",
    "is_axis_compatible",
    "overlapping_axes",
    "replace_axis",
    "selects_axis",
    "union_axes",
    "without_axes",
    "unsize_axes",
    "rearrange_for_partial_order",
]
