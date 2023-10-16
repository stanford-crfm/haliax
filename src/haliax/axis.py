import typing
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union, overload

import equinox as eqx
from jax.experimental.pallas import dslice as pdslice

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


AxisSelector = Union[Axis, str]
"""AxisSelector is a type that can be used to select a single axis from an array. str or Axis"""
AxisSelection = Union[AxisSelector, Sequence[AxisSelector]]
"""AxisSelection is a type that can be used to select multiple axes from an array. str, Axis, or sequence of mixed
str and Axis"""
AxisSpec = Union[Axis, Sequence[Axis]]
"""AxisSpec is a type that can be used to specify the axes of an array, usually for creation or adding a new axis
 whose size can't be determined another way. Axis or sequence of Axis"""


def selects_axis(selector: AxisSelection, selected: AxisSelection) -> bool:
    """Returns true if the selector has every axis in selected and, if dims are given, that they match"""
    if isinstance(selector, Axis) or isinstance(selector, str):
        selected = ensure_tuple(selected)
        try:
            index = index_where(lambda ax: is_axis_compatible(ax, selector), selected)  # type: ignore
            return index >= 0
        except ValueError:
            return False

    selector_dict = _spec_to_dict(selector)

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
def _spec_to_dict(axis_spec: AxisSpec) -> Dict[str, int]:  # type: ignore
    ...


@overload
def _spec_to_dict(axis_spec: AxisSelection) -> Dict[str, Optional[int]]:  # type: ignore
    ...


def _spec_to_dict(axis_spec: AxisSelection) -> Dict[str, Optional[int]]:  # type: ignore
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

    a1_dict = _spec_to_dict(a1)
    a2_dict = _spec_to_dict(a2)

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
    axis_spec_dict = _spec_to_dict(axis_spec)
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
    axis_spec_dict = _spec_to_dict(axis_spec)
    for ax in to_remove:
        name = axis_name(ax)
        if name in axis_spec_dict:
            del axis_spec_dict[name]

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
    """Returns a tuple of axes that are present in both ax1 and ax2"""
    ax2_dict = _spec_to_dict(ax2)
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


def axis_name(ax: AxisSelector) -> str:
    """
    Returns the name of the axis. If ax is a string, returns ax. If ax is an Axis, returns ax.name
    """
    if isinstance(ax, Axis):
        return ax.name
    else:
        return ax


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


ds: typing.TypeAlias = dslice


_PALLAS_DSLICE_TYPE = type(pdslice(0, 1))


def is_pallas_dslice(x: object) -> bool:
    return isinstance(x, _PALLAS_DSLICE_TYPE)


__all__ = [
    "Axis",
    "AxisSelector",
    "AxisSelection",
    "AxisSpec",
    "axis_name",
    "concat_axes",
    "union_axes",
    "ds",
    "dslice",
    "eliminate_axes",
    "is_axis_compatible",
    "overlapping_axes",
    "replace_axis",
    "selects_axis",
    "union_axes",
    "without_axes",
    "is_pallas_dslice",
]
