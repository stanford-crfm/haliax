import functools
from types import EllipsisType
from typing import Optional, Sequence, Tuple, Union

import jax.lax
import jax.numpy as jnp

import haliax

from ..axis import AxisSelector, axis_name, eliminate_axes, rearrange_for_partial_order, union_axes
from ..core import NamedArray
from ..jax_utils import _jittable_dg_einsum
from ..types import DTypeLike, PrecisionLike
from ..util import ensure_tuple
from .parsing import AliasTable, parse_einsum, raise_parse_error


def einsum(
    equation: str,
    *arrays: NamedArray,
    precision: PrecisionLike = None,
    preferred_element_type: Optional[DTypeLike] = None,
    _dot_general=jax.lax.dot_general,
) -> NamedArray:
    """Compute the tensor contraction of the input arrays according to Haliax's named variant of the Einstein summation
    convention.

    Examples:
       >>> # normal einsum
       >>> import haliax as hax
       >>> H = hax.Axis("H", 32)
       >>> W = hax.Axis("W", 32)
       >>> D = hax.Axis("D", 64)
       >>> a = hax.zeros((H, W, D))
       >>> b = hax.zeros((D, W, H))
       >>> hax.einsum("h w d, d w h -> h w", a, b)
       >>> # named einsum
       >>> hax.einsum("{H W D} -> H W", a, b)
       >>> hax.einsum("{D} -> ", a, b)  # same as the previous example
       >>> hax.einsum("-> H W", a, b)  # same as the first example

    Args:
       equation: The einsum equation.
       arrays: The input arrays.
       precision: The precision of the computation.
       preferred_element_type: The preferred element type of the computation.

    Returns:
       The result of the einsum.
    """
    lhses, rhs = parse_einsum(equation)

    # we have essentially 3 cases:
    # 1. normal positional einsum
    # 2. named einsum where a subset of dims are contracted and the rest are kept
    # 3. named einsum without an lhs, with a specific set of dims to keep on the rhs
    # in each case we need

    # NB: we're using JAX's einsum which only supports one letter names for dims
    if len(lhses) == 1 and len(lhses[0].captures) == 0 and lhses[0].is_ordered:
        # case 3: get the output axes, contract the others
        spec, out_axes = _output_only_named_einsum(equation, arrays, rhs)
    elif len(lhses) == 1 and not lhses[0].is_ordered:
        # case 2: some axes are named. Those named only on the lhs are contracted, the others are kept
        # subcase: if there's an ellipsis on the lhs, we contract all the axes that are not named on the rhs
        spec, out_axes = _unordered_einsum(arrays, equation, lhses, rhs)
    else:
        # general case: we have a normal einsum. we don't allow unordered axes here
        if any(not lhs.is_ordered for lhs in lhses):
            raise_parse_error("Cannot have multiple unordered axes in an einsum", equation, None)

        spec, out_axes = _positional_einsum_spec(equation, arrays, lhses, rhs)

    out_raw = _jittable_dg_einsum(
        spec,
        *[a.array for a in arrays],
        precision=precision,
        preferred_element_type=preferred_element_type,
        _dot_general=_dot_general,
    )

    out = haliax.named(out_raw, out_axes)
    return haliax.auto_sharded(out)


def _unordered_einsum(arrays, equation, lhses, rhs):
    used_letters: set[str] = set()
    name_mappings_for_einsum: dict[str, str] = {}
    lhs = lhses[0]
    candidate_axes, has_ellipsis_lhs = _captures_to_axis_names(equation, lhs)
    rhs_axes, has_ellipsis_rhs = _captures_to_axis_names(equation, rhs)
    all_input_axes = _all_input_axes(arrays)
    if has_ellipsis_rhs:
        out_axes = rearrange_for_partial_order(rhs_axes, all_input_axes)

    elif has_ellipsis_lhs:
        # if the lhs has an ellipsis but not the right, we contract all the axes that are not named on the rhs
        out_axes = tuple(rhs_axes)
    else:
        named_on_left_but_not_right = eliminate_axes(candidate_axes, rhs_axes)  # type: ignore
        # if neither has an ellipsis, we contract the axes that are named on the lhs but not on the rhs
        almost_out_axes = eliminate_axes(all_input_axes, named_on_left_but_not_right)  # type: ignore
        # we now need to rearrange to be consistent with the rhs order.
        # since there's no ellipsis on the rhs, we arbitrarily insert one at the beginning which is usually
        # what people expect
        rhs_axes = [Ellipsis] + rhs_axes  # type: ignore
        out_axes = rearrange_for_partial_order(rhs_axes, almost_out_axes)
    spec = _make_einsum_spec(name_mappings_for_einsum, used_letters, arrays, out_axes)
    return spec, out_axes


def _output_only_named_einsum(equation, arrays, rhs):
    used_letters: set[str] = set()
    name_mappings_for_einsum: dict[str, str] = {}

    out_axes = []
    for capture in rhs.captures:
        if capture is Ellipsis:
            raise_parse_error("Can't use ellipsis on the rhs of an einsum without an lhs", equation, None)
        elif capture.binding is None or len(capture.axes) > 1:
            raise_parse_error(
                "Parenthesized axes are not currently supported in the output of an einsum",
                equation,
                capture.char_range,
            )
        else:
            name = capture.binding

            if name in out_axes:
                raise_parse_error(f"Axis name {name} occurs multiple times on the rhs", equation, capture.char_range)

            out_axes.append(name)

    spec = _make_einsum_spec(name_mappings_for_einsum, used_letters, arrays, out_axes)
    return spec, out_axes


def _positional_einsum_spec(equation, arrays, lhses, rhs):
    used_letters: set[str] = set()
    name_mappings_for_einsum: dict[str, str] = {}

    if len(lhses) != len(arrays):
        raise ValueError(f"Number of lhses ({len(lhses)}) does not match number of arrays ({len(arrays)})")
    table = AliasTable()
    # ok, we're going to lead pretty heavily on einsum here. We just need to figure out the names of the axes
    # and do any error checking (that there are no mismatched names)
    # once we do that, we can pass a slightly modified spec to einsum (namely that we shorten the names of the axes)
    # and we're good to go
    spec = ""
    for lhs, a in zip(lhses, arrays):
        if len(spec):
            spec += ","

        # have to deal with ellipsis: we support at most one per lhs and it can appear anywhere
        has_ellipsis_lhs = False
        axis_off = 0
        for capture in lhs.captures:
            if capture is Ellipsis:
                has_ellipsis_lhs = True
                spec += "..."
                axis_off += 1
                break
            elif capture.binding is None or len(capture.axes) > 1:
                raise_parse_error("Parenthesized axes are not currently supported", equation, capture.char_range)
            else:
                name = capture.binding
                if axis_off >= len(a.axes):
                    raise ValueError("Mismatched number of axes in einsum")
                table.bind_alias(name, a.axes[axis_off], equation, capture.char_range)
                letter = _assign_letter_to_name(name, name_mappings_for_einsum, used_letters)
                spec += letter
                axis_off += 1

        if has_ellipsis_lhs:
            # check there aren't two ellipses
            if Ellipsis in lhs.captures[axis_off:]:
                raise_parse_error("Can't have two ellipses in an einsum", equation, None)

            final_lhs_axis_off = axis_off

            axis_off = len(a.axes) - 1
            for capture in reversed(lhs.captures):
                if capture is Ellipsis:
                    break
                else:
                    name = capture.binding
                    if axis_off < final_lhs_axis_off:
                        raise ValueError("Mismatched number of axes in einsum")
                    table.bind_alias(name, a.axes[axis_off], equation, capture.char_range)
                    letter = _assign_letter_to_name(name, name_mappings_for_einsum, used_letters)
                    spec += letter
                    axis_off -= 1
        else:
            if axis_off != len(a.axes):
                raise ValueError("Mismatched number of axes in einsum")

    named_on_left_but_not_right = set(table.bindings.keys())
    spec += "->"
    out_axes: list[AxisSelector | EllipsisType] = []
    has_ellipsis_rhs = False

    for capture in rhs.captures:
        if capture is Ellipsis:
            spec += "..."
            out_axes.append(Ellipsis)
            if has_ellipsis_rhs:
                raise_parse_error("Can't have two ellipses in an ordered einsum", equation, None)
            has_ellipsis_rhs = True
        elif capture.binding is None or len(capture.axes) > 1:
            raise_parse_error(
                "Parenthesized axes are not currently supported in the output of an einsum",
                equation,
                capture.char_range,
            )
        else:
            name = capture.binding
            axis = table.dealias_binding(name)
            if axis is None:
                raise_parse_error(f"Axis {name} not found in the input arrays", equation, capture.char_range)

            if name_mappings_for_einsum.get(name) is not None:
                letter = name_mappings_for_einsum[name]
            else:
                raise_parse_error(
                    f"Axis name {name} does not occur on the left hand side", equation, capture.char_range
                )

            named_on_left_but_not_right.discard(letter)

            spec += letter
            out_axes.append(axis)

    if has_ellipsis_rhs:
        all_input_axes = _all_input_axes(arrays)
        # eliminate the axes that are contracted
        unmentioned = tuple(table.dealias_binding(name) for name in named_on_left_but_not_right)
        out = eliminate_axes(all_input_axes, unmentioned)  # type: ignore
        return spec, out
    else:
        return spec, out_axes


def _all_input_axes(arrays):
    return ensure_tuple(functools.reduce(union_axes, (a.axes for a in arrays), ()))  # type: ignore


def _captures_to_axis_names(equation, lhs) -> Tuple[list[str | EllipsisType], bool]:
    candidate_axes: list[str | EllipsisType] = []
    has_ellipsis = False
    for capture in lhs.captures:
        if capture is Ellipsis:
            has_ellipsis = True
            candidate_axes.append(Ellipsis)
        elif capture.binding is None or len(capture.axes) > 1:
            raise_parse_error("Parenthesized axes are not currently supported", equation, capture.char_range)
        else:
            name = capture.binding
            candidate_axes.append(name)
    return candidate_axes, has_ellipsis


def _make_einsum_spec(name_mappings_for_einsum, used_letters, arrays, out_axes):
    spec = ""
    for operand in arrays:
        if len(spec):
            spec += ","
        for axis in operand.axes:
            letter = _assign_letter_to_name(axis.name, name_mappings_for_einsum, used_letters)
            spec += letter
    spec += "->"
    for out in out_axes:
        letter = name_mappings_for_einsum[axis_name(out)]
        spec += letter
    return spec


def _assign_letter_to_name(name, name_mappings_for_einsum, used_letters):
    if name_mappings_for_einsum.get(name) is not None:
        return name_mappings_for_einsum[name]

    letter = name[0]
    if letter in used_letters:
        # try to find another letter. This is obviously not efficient, but it's not a big deal
        for letter in "abcdefghijklmnopqrstuvwxyz":
            if letter not in used_letters:
                break
        else:
            raise ValueError("Too many axes to contract")
    name_mappings_for_einsum[name] = letter
    used_letters.add(letter)
    return letter
