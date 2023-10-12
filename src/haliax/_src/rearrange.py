# Support for einops-style rearrangement strings, but supporting named axes and unordered matching
import dataclasses
from types import EllipsisType
from typing import Mapping, NoReturn, Optional, Sequence

import jax.lax

from .. import auto_sharded
from ..axis import Axis, AxisSelector, axis_name
from ..core import NamedArray


@dataclasses.dataclass
class _AxisCapture:
    binding: Optional[str] = None
    axes: tuple[str, ...] = ()
    char_range: Optional[tuple[int, int]] = None


@dataclasses.dataclass
class Expression:
    captures: Sequence[_AxisCapture | EllipsisType]
    is_ordered: bool


def _raise_error(message: str, expression: str, pos: Optional[int | tuple[int, int]]) -> NoReturn:
    """Raise a ValueError with a message and the position in the expression."""
    fmt = f"Error while parsing:\n    {expression}"
    if pos is not None:
        if isinstance(pos, int):
            fmt += f'\n    {" " * pos}^'
        else:
            fmt += f"\n    {' ' * pos[0]}{'^' * max(1, pos[1] - pos[0])}"

    fmt += f"\n{message}"

    raise ValueError(fmt)


def _parse_quoted_string(expression: str, pos: int) -> tuple[str, int]:
    """Parse a quoted string from an einops-style haliax rearrangement string."""

    if expression[pos] not in "'\"":
        _raise_error(f"Expected \" or ' at position {pos}", expression, pos)
    quote = expression[pos]
    pos += 1
    ident = ""
    while pos < len(expression):
        if expression[pos] == quote:
            pos += 1
            break
        elif expression[pos] == "\\":
            pos += 1
            if pos >= len(expression):
                _raise_error(f"Unexpected end of string at position {pos}", expression, pos)
            ident += expression[pos]
            pos += 1
            continue
        else:
            ident += expression[pos]
            pos += 1
            continue
    if len(ident) == 0:
        _raise_error("Empty strings are not valid identifiers", expression, pos)

    return ident, pos


def _parse_ident(expression: str, pos: int) -> tuple[str, int]:
    """parses an identifier or string literal from an einops-style haliax rearrangement string."""
    if expression[pos] in "'\"":
        return _parse_quoted_string(expression, pos)
    else:
        ident = ""
        while pos < len(expression):
            if str.isalnum(expression[pos]) or expression[pos] == "_":
                if len(ident) == 0 and str.isdigit(expression[pos]):
                    _raise_error("Identifiers cannot start with a number", expression, pos)
                ident += expression[pos]
                pos += 1
                continue
            else:
                break
        if len(ident) == 0:
            _raise_error("Identifier expected", expression, pos)

        return ident, pos


def _parse_group(expression, pos):
    # parses a group of axes like (a b c) or (a: b c)
    pos_in = pos
    if expression[pos] != "(":
        raise ValueError("Expected (")
    pos += 1
    binding = None
    axes = []
    current_ident = ""
    while pos < len(expression):
        if expression[pos] == ")":
            pos += 1
            break
        elif expression[pos] == ":":
            if binding is not None:
                _raise_error("Only one binding allowed per group", expression, pos)
            if not current_ident:
                _raise_error("Binding cannot be empty", expression, pos)
            if len(axes) > 0:
                _raise_error("Binding must come before axes", expression, pos)
            binding = current_ident
            current_ident = ""
            pos += 1
            continue
        elif str.isspace(expression[pos]) or expression[pos] == ",":
            if current_ident:
                axes.append(current_ident)
                current_ident = ""
            pos += 1
            continue
        elif expression[pos] == "(":
            _raise_error("Only one level of nesting is allowed", expression, pos)
        elif expression[pos] == "}":
            raise ValueError(f"Unexpected }} at {pos}")
        elif str.isalnum(expression[pos]) or expression[pos] == "_":
            # don't allow numbers at the start of an identifier
            if len(current_ident) == 0 and str.isdigit(expression[pos]):
                _raise_error("Identifiers cannot start with a number", expression, pos)
            current_ident += expression[pos]
            pos += 1
            continue
        elif expression[pos] in "'\"":
            # parse quoted string as identifier
            if current_ident:
                axes.append(current_ident)

            ident, pos = _parse_quoted_string(expression, pos)
            current_ident = ident
            continue
        else:
            _raise_error(f"Unexpected character {expression[pos]}", expression, pos)

    if current_ident:
        axes.append(current_ident)

    if len(axes) == 0:
        _raise_error("No axes found", expression, pos_in)

    # todo: should we allow anonymous/literal
    char_range = (pos_in, pos)
    return _AxisCapture(binding, tuple(axes), char_range), pos


def _parse_expression(expression: str, pos) -> tuple[Expression, int]:
    """Parse one side of an einops-style haliax rearrangement string."""
    captures = []
    is_ordered = True
    seen_char = False
    finished = False

    while pos < len(expression):
        if expression[pos] == "{":
            if seen_char:
                _raise_error("Unexpected {", expression, pos)
            seen_char = True
            is_ordered = False
            pos += 1
            continue
        elif expression[pos] == "}":
            if is_ordered:
                _raise_error("Unexpected }", expression, pos)
            pos += 1
            finished = True
            continue
        elif expression[pos] == "(":
            if finished:
                _raise_error("Unexpected ( after }", expression, pos)
            seen_char = True
            capture, pos = _parse_group(expression, pos)
            captures.append(capture)
            continue
        elif str.isspace(expression[pos]) or expression[pos] == ",":
            pos += 1
            continue
        elif expression[pos : pos + 3] == "...":
            seen_char = True
            if finished:
                _raise_error("Unexpected ... after }", expression, pos)
            captures.append(Ellipsis)
            pos += 3
            continue
        elif expression[pos] == "-":
            if not seen_char:
                _raise_error("Unexpected -", expression, pos)
            if pos + 1 >= len(expression):
                _raise_error("Unexpected end of string", expression, pos)
            if expression[pos + 1] != ">":
                _raise_error("Expected >", expression, pos)
            break
        else:
            if finished:
                _raise_error("Unexpected character after }", expression, pos)
            ident, new_pos = _parse_ident(expression, pos)
            captures.append(_AxisCapture(binding=ident, axes=(ident,), char_range=(pos, new_pos)))
            seen_char = True
            pos = new_pos
            continue

    if not finished and not is_ordered:
        _raise_error("Expected }", expression, pos)

    return Expression(captures, is_ordered), pos


def parse_rearrangement(expression: str) -> tuple[Expression, Expression]:
    """Parse an einops-style haliax rearrangement string."""
    pos = 0
    lhs, pos = _parse_expression(expression, pos)

    # consume the ->
    if pos + 2 >= len(expression):
        _raise_error("Unexpected end of string", expression, pos)
    if expression[pos : pos + 2] != "->":
        _raise_error("Expected ->", expression, pos)

    pos += 2
    rhs, pos = _parse_expression(expression, pos)

    # make sure we consumed the whole string
    if pos != len(expression):
        _raise_error("Unexpected character", expression, pos)

    return lhs, rhs


@dataclasses.dataclass
class _Plan:
    intermediate_axes: tuple[Axis, ...]
    transpose: Optional[tuple[int, ...]]
    needs_final_reshape: bool

    final_axes: tuple[Axis, ...]


def rearrange(array: NamedArray, expression: str, **bindings: AxisSelector | int) -> NamedArray:
    """Rearrange a tensor according to an einops-style haliax rearrangement string."""
    lhs, rhs = parse_rearrangement(expression)

    # the fundamental xla op for rearranging is reshape, which combines both transpose and reshape
    # all rearranges are fundamentally a reshape, followed by a transpose, followed by a reshape
    # as an example of an op that needs both, consider:
    #  '(a b) c d -> (c a) (b d)'
    # where a = 2, b = 3, c = 4, d = 5
    # which is equivalent to:
    # x.reshape((2, 3, 4, 5)).transpose((2, 0, 1, 3)).reshape((4 * 2, 3 * 5))
    plan = _plan_rearrange(expression, lhs, rhs, array, bindings)

    raw_array = array.array

    if plan.intermediate_axes != array.axes:
        raw_array = raw_array.reshape([ax.size for ax in plan.intermediate_axes])
        array = NamedArray(raw_array, plan.intermediate_axes)
        array = auto_sharded(array)
        raw_array = array.array

    final_shape = tuple(ax.size for ax in plan.final_axes)

    if plan.needs_final_reshape:
        finished_array = jax.lax.reshape(raw_array, new_sizes=final_shape, dimensions=plan.transpose)
    elif plan.transpose is not None:
        finished_array = jax.lax.transpose(raw_array, permutation=plan.transpose)
    else:
        finished_array = raw_array

    return auto_sharded(NamedArray(finished_array, plan.final_axes))


class AliasTable:
    bindings: dict[str, Axis]
    aliases: dict[str, str]

    def __init__(self, bindings, aliases):
        self.bindings = bindings
        self.aliases = aliases

    def dealias_binding(self, binding: str) -> Optional[AxisSelector]:
        if binding in self.bindings:
            return self.bindings[binding]
        elif binding in self.aliases:
            return self.aliases[binding]
        else:
            return None

    def bind_alias(self, alias: str, axis: Axis, expr, char_range):
        if alias in self.aliases:
            if self.aliases[alias] != axis.name:
                _raise_error(f"Alias {alias} is bound to more than one axis", expr, char_range)
        else:
            self.aliases[alias] = axis.name

        if axis.name in self.bindings:
            if self.bindings[alias] != axis:
                _raise_error(f"Alias {alias} is bound to more than one axis", expr, char_range)
        else:
            self.bindings[alias] = axis


def _plan_rearrange(
    original_str, lhs: Expression, rhs: Expression, array: NamedArray, input_bindings: Mapping[str, AxisSelector | int]
) -> _Plan:
    aliases = _resolve_binding_sizes(array, input_bindings)
    grouped_new_shapes = _determine_initial_reshape(original_str, lhs, array, aliases)
    intermediate_axes = tuple(ax for split_axes in grouped_new_shapes for ax in split_axes)

    # Now figure out the transpose and final reshape
    transpose: Optional[tuple[int, ...]]
    transpose, final_axes = _determine_final_transpose_and_reshape(original_str, rhs, aliases, intermediate_axes)

    transposed_intermediate_axes = tuple(intermediate_axes[i] for i in transpose)
    if transposed_intermediate_axes == final_axes:
        needs_final_reshape = False
    else:
        needs_final_reshape = True

    if transpose == tuple(range(len(transpose))):
        transpose = None

    return _Plan(intermediate_axes, transpose, needs_final_reshape, final_axes)


def _determine_final_transpose_and_reshape(
    expression, rhs: Expression, aliases: AliasTable, intermediate_axes: tuple[Axis, ...]
) -> tuple[tuple[int, ...], tuple[Axis, ...]]:
    # The rhs tells us two things:
    # 1. how to reorder the intermediate axes
    # 2. how to merge the intermediate axes into the final axes

    transposition_order: list[int | EllipsisType] = []

    final_axes: list[Axis | EllipsisType] = []
    used_axes = set()  # axes must be used exactly once

    position_of_ellipsis = None  # If there's an ellipsis, we want to put any remaining axes there
    position_of_ellipsis_in_transposition = None  # if there's an ellipsis, we want to put any remaining axes there

    # each capture, except for ellipsis results in one final axis
    for cpos, capture in enumerate(rhs.captures):
        if capture == Ellipsis:
            if position_of_ellipsis is not None:
                _raise_error("Only one ellipsis allowed", expression, capture.char_range)
            position_of_ellipsis = cpos
            final_axes.append(Ellipsis)
            transposition_order.append(Ellipsis)
            position_of_ellipsis_in_transposition = len(transposition_order) - 1
            continue

        assert not isinstance(capture, EllipsisType)

        binding = capture.binding
        if binding is None:
            _raise_error("All rhs axes must have a name", expression, capture.char_range)

        # now look at the captured axes. these are the axes that we're going to merge into one final axis
        # we're also going to move them around to match the order of the intermediate axes
        sz = 1
        for ax_name in capture.axes:
            axis = aliases.dealias_binding(ax_name)
            if not isinstance(axis, Axis):
                _raise_error(f"Axis {ax_name} is not bound on the lhs", expression, capture.char_range)

            if axis in used_axes:
                _raise_error(f"Axis {axis} is used more than once", expression, capture.char_range)
            used_axes.add(axis)

            sz *= axis.size
            # now find the position of this axis in the intermediate axes
            try:
                index_in_intermediate = intermediate_axes.index(axis)
                transposition_order.append(index_in_intermediate)
            except ValueError:
                _raise_error(f"Axis {ax_name} is not in the lhs", expression, capture.char_range)

        # figure out the name of the final axis
        axis = aliases.dealias_binding(binding)
        if axis is None:
            new_axis = Axis(binding, sz)
        else:
            new_axis = Axis(axis_name(axis), sz)

        aliases.bind_alias(binding, new_axis, expression, capture.char_range)

        final_axes.append(new_axis)

    if position_of_ellipsis is not None:
        unused_intermediate_axes = [ax for ax in intermediate_axes if ax not in used_axes]
        if len(unused_intermediate_axes) > 0:
            # we need to put the unused axes in the ellipsis
            final_axes = (
                final_axes[:position_of_ellipsis] + unused_intermediate_axes + final_axes[position_of_ellipsis + 1 :]
            )

            unused_indices = [i for i in range(len(intermediate_axes)) if intermediate_axes[i] not in used_axes]

            assert position_of_ellipsis_in_transposition is not None

            transposition_order = (
                transposition_order[:position_of_ellipsis_in_transposition]
                + unused_indices
                + transposition_order[position_of_ellipsis_in_transposition + 1 :]
            )

    else:
        # make sure we used all the axes
        if len(used_axes) < len(intermediate_axes):
            # figure out their binding names as possible:
            inverse_bindings = {v: k for k, v in aliases.bindings.items()}
            unused_intermediate_axis_names = [
                inverse_bindings.get(ax, ax.name) for ax in intermediate_axes if ax not in used_axes
            ]
            _raise_error(
                f"Not all intermediate axes are used: {','.join(unused_intermediate_axis_names) }",
                expression,
                len(expression) - 1,
            )

    return tuple(transposition_order), tuple(final_axes)  # type: ignore


def _determine_initial_reshape(
    expression,
    lhs: Expression,
    array: NamedArray,
    aliases: AliasTable,
) -> list[list[Axis]]:
    # MUTATES `aliases`
    # the lhs all need to be bound to axes in the array, or synthesized as parts of axes.
    # In the lhs, bindings look like either a name, or a name and a list of (new) axes.
    # bindings can either be done by name, or by position, depending on if lhs.is_ordered
    new_shapes: list[Optional[list[Axis]]] = [None] * len(array.axes)
    used_new_names: set[str] = set()  # names can only be used once on a side

    if lhs.is_ordered:
        # this is the easy case, bind axes in order of appearance in the lhs
        # We do have to be careful about ellipses
        # if we start with an ellipsis, we bind from the right
        # if we end with an ellipsis, we bind from the left
        ellipsis_pos = None
        axis_index_for_capture: list[Optional[int]] = [None] * len(lhs.captures)
        covered_axes = set()
        # bind from the left
        axis_pos = 0
        for cpos, capture in enumerate(lhs.captures):
            if capture == Ellipsis:
                if ellipsis_pos is not None:
                    assert False, "should not be here"  # pragma: no cover
                ellipsis_pos = cpos
                break

            assert not isinstance(capture, EllipsisType)

            if axis_pos >= len(array.axes):
                _raise_error("Too many axes in lhs", expression, capture.char_range)

            if capture.binding is not None:
                _bind_capture_to_positional_axis(capture, array.axes[axis_pos], aliases, expression)

            axis_index_for_capture[cpos] = axis_pos
            covered_axes.add(axis_pos)
            axis_pos += 1

        # bind from the right, take care to check second ellipsis
        if ellipsis_pos is not None:
            axis_pos = len(array.axes) - 1
            for cpos in range(len(lhs.captures) - 1, ellipsis_pos, -1):
                capture = lhs.captures[cpos]
                capture = capture
                if capture == Ellipsis:
                    _raise_error("Only one ellipsis allowed", expression, None)

                assert not isinstance(capture, EllipsisType)

                if capture.binding is not None:
                    _bind_capture_to_positional_axis(capture, array.axes[axis_pos], aliases, expression)

                axis_index_for_capture[cpos] = axis_pos
                if axis_pos in covered_axes:
                    _raise_error(
                        f"Axis {array.axes[axis_pos]} is bound more than once",
                        expression,
                        capture.char_range,
                    )
                covered_axes.add(axis_pos)
                axis_pos -= 1
        else:
            # no ellipsis, so we need to check that we covered all axes
            if len(covered_axes) < len(array.axes):
                _raise_error(
                    "Not all axes are bound, use ... to skip missing axes", expression, len(expression) - 1
                )  # type: ignore
            elif len(covered_axes) > len(array.axes):
                _raise_error("Too many axes are bound", expression, lhs.captures[-1].char_range)  # type: ignore

        # now that we have the bindings, we can figure out the new shapes
        for cpos, capture in enumerate(lhs.captures):
            if capture == Ellipsis:
                continue
            assert not isinstance(capture, EllipsisType)

            axis_index = axis_index_for_capture[cpos]
            if axis_index is None:
                _raise_error("Internal error", expression, capture.char_range)

            axis = array.axes[axis_index]
            if new_shapes[axis_index] is not None:
                _raise_error(f"Axis {axis} is bound more than once", expression, capture.char_range)

            new_axes = _solve_split_axes(axis, capture, aliases, used_new_names, expression)
            new_shapes[axis_index] = new_axes  # type: ignore

    else:
        # we just need to bind the axes in the lhs to the axes in the array
        for capture in lhs.captures:
            # ellipses are ignored in unordered rearrangements
            if capture == Ellipsis:
                continue

            assert not isinstance(capture, EllipsisType)

            if capture.binding is None:
                _raise_error(
                    "Unordered axes must be bound by name, e.g. (a: b) or just a", expression, capture.char_range
                )

            # let's see if we're aliasing it in the bindings
            maybe_alias = aliases.dealias_binding(capture.binding)
            if maybe_alias is None:
                maybe_alias = capture.binding

            if not isinstance(maybe_alias, Axis):
                try:
                    axis = array.resolve_axis(maybe_alias)
                    aliases.bind_alias(capture.binding, axis, expression, capture.char_range)
                except ValueError:
                    _raise_error(f"Could not resolve axis {maybe_alias}", expression, capture.char_range)
            else:
                axis = maybe_alias

            if axis is None:
                _raise_error(f"Could not resolve axis {capture.binding}", expression, capture.char_range)

            index_of_binding = array.axes.index(axis)
            if new_shapes[index_of_binding] is not None:
                _raise_error(f"Axis {axis} is bound more than once", expression, capture.char_range)

            new_axes = _solve_split_axes(axis, capture, aliases, used_new_names, expression)
            new_shapes[index_of_binding] = new_axes  # type: ignore

    for i in range(len(array.axes)):
        if new_shapes[i] is None:
            new_shapes[i] = [array.axes[i]]

    return new_shapes  # type: ignore


def _solve_split_axes(axis, capture, aliases, used_new_names, expression):
    """
    Given an axis and a capture of the form (a: b c) or (b c), solve for the new axes.
    """
    new_axes: list[Optional[Axis]] = []
    unsolved_axis_index: Optional[int] = None

    # easy case: 1 axis in capture
    if len(capture.axes) == 1:
        new_axis_name = capture.axes[0]
        if new_axis_name in used_new_names:
            _raise_error(f"Axis {new_axis_name} is bound more than once", expression, capture.char_range)
        used_new_names.add(new_axis_name)

        new_axis = aliases.dealias_binding(new_axis_name)
        if new_axis is None:
            new_axis = new_axis_name
        if isinstance(new_axis, Axis):
            if new_axis.size != axis.size:
                _raise_error(
                    f"{new_axis} is mapped to {new_axis_name} but has a different size than {axis}",
                    expression,
                    capture.char_range,
                )
            new_axes.append(new_axis)
        else:
            new_axis_name = Axis(new_axis_name, axis.size)
            new_axes.append(new_axis_name)
            aliases.bind_alias(new_axis_name, new_axis_name, expression, capture.char_range)

        return new_axes

    remaining_size = axis.size

    for new_axis_name in capture.axes:
        if new_axis_name in used_new_names:
            _raise_error(f"Capture {new_axis_name} is assigned more than once in lhs", expression, capture.char_range)

        used_new_names.add(new_axis_name)

        new_axis = aliases.dealias_binding(new_axis_name)
        if new_axis is None:
            new_axis = new_axis_name

        if isinstance(new_axis, Axis):
            new_axes.append(new_axis)
            if remaining_size % new_axis.size:
                _raise_error(f"Axes do not divide evenly into axis {axis}", expression, capture.char_range)
            remaining_size //= new_axis.size
        else:
            if unsolved_axis_index is not None:
                _raise_error(
                    "Sizes for this split axis are ambiguous. You must provide a size as a kwarg.",
                    expression,
                    capture.char_range,
                )
            unsolved_axis_index = len(new_axes)
            new_axes.append(None)

    if unsolved_axis_index is not None:
        # we need to solve for this axis
        unsolved_alias = aliases.dealias_binding(capture.axes[unsolved_axis_index])
        assert not isinstance(unsolved_alias, Axis)
        if unsolved_alias is None:
            unsolved_alias = capture.axes[unsolved_axis_index]
        assert isinstance(unsolved_alias, str)

        new_axis = Axis(unsolved_alias, remaining_size)
        new_axes[unsolved_axis_index] = new_axis
        aliases.bind_alias(capture.axes[unsolved_axis_index], new_axis, expression, capture.char_range)
    return new_axes


def _bind_capture_to_positional_axis(capture, new_axis, bindings: AliasTable, expression):
    already_bound_axis: Optional[AxisSelector] = bindings.dealias_binding(capture.binding)

    if already_bound_axis is None:
        bindings.bind_alias(capture.binding, new_axis, expression, capture.char_range)
    elif isinstance(already_bound_axis, Axis):
        if already_bound_axis != new_axis:
            _raise_error(
                f"Pattern {capture.binding} is bound to {already_bound_axis} and {new_axis}",
                expression,
                capture.char_range,
            )
    else:
        bindings.bind_alias(capture.binding, new_axis, expression, capture.char_range)


def _resolve_binding_sizes(array, bindings) -> AliasTable:
    b: dict[str, Axis] = {}
    aliases: dict[str, str] = {}
    for name, selector in bindings.items():
        if isinstance(selector, str):
            try:
                selector = array.resolve_axis(selector)
            except ValueError:
                pass
        elif isinstance(selector, int):
            selector = Axis(name, selector)
        b[name] = selector
        aliases[name] = axis_name(selector)
    return AliasTable(b, aliases)
