from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Union, overload

from haliax.util import ensure_tuple, index_where


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


def _dict_to_spec(axis_spec: Dict[str, Optional[int]]) -> AxisSelection:
    return tuple(Axis(name, size) if size is not None else name for name, size in axis_spec.items())


@overload
def concat_axes(a1: AxisSpec, a2: AxisSpec) -> AxisSpec:
    pass


@overload
def concat_axes(a1: AxisSelection, a2: AxisSelection) -> AxisSelection:
    pass


def concat_axes(a1: AxisSelection, a2: AxisSelection) -> AxisSelection:
    """Concatenates two AxisSpec. Raises ValueError if any axis is present in both specs"""

    def _ax_name(ax: AxisSelector) -> str:
        if isinstance(ax, Axis):
            return ax.name
        else:
            return ax

    if isinstance(a1, Axis) and isinstance(a2, Axis):
        if _ax_name(a1) == _ax_name(a2):
            raise ValueError(f"Axis {a1} specified twice")
        return (a1, a2)
    else:
        a1 = ensure_tuple(a1)
        a2 = ensure_tuple(a2)

        a1_names = [_ax_name(ax) for ax in a1]
        a2_names = [_ax_name(ax) for ax in a2]

        if len(set(a1_names) & set(a2_names)) > 0:
            overlap = [ax for ax in a1_names if ax in a2_names]
            raise ValueError(f"AxisSpecs overlap! {' '.join(str(x) for x in overlap)}")
        return a1 + a2


@overload
def eliminate_axes(axis_spec: AxisSpec, axes: AxisSelection) -> AxisSpec:  # type: ignore
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
        if isinstance(ax, Axis):
            if ax.name not in axis_spec_dict:
                raise ValueError(f"Axis {ax.name} not present in axis spec {axis_spec}")
            del axis_spec_dict[ax.name]
        elif isinstance(ax, str):
            if ax not in axis_spec_dict:
                raise ValueError(f"Axis {ax} not present in axis spec {axis_spec}")
            del axis_spec_dict[ax]
        else:
            raise ValueError(f"Invalid axis spec: {ax}")

    return _dict_to_spec(axis_spec_dict)
