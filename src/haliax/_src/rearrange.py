# Support for einops-style rearrangement strings, but supporting named axes and unordered matching
import dataclasses
import typing
from types import EllipsisType
from typing import Mapping, NoReturn, Optional, Sequence

import jax.lax
import jax.numpy as jnp

from ..axis import Axis, AxisSelector, axis_name
from ..core import NamedArray
from ..partitioning import auto_sharded


@dataclasses.dataclass
class _AxisCapture:
    binding: Optional[str] = None
    axes: tuple[str, ...] = ()
    char_range: Optional[tuple[int, int]] = None

    def __post_init__(self):
        if len(self.axes) == 0:
            raise ValueError("Empty axes not allowed")


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


@typing.overload
def rearrange(array: NamedArray, axes: Sequence[AxisSelector | EllipsisType]) -> NamedArray:
    pass


@typing.overload
def rearrange(array: NamedArray, expression: str, **bindings: AxisSelector | int) -> NamedArray:
    pass


def rearrange(array: NamedArray, *args, **kwargs) -> NamedArray:
    """
    Rearrange a tensor according to an einops-style haliax rearrangement string or a sequence of axes.
    See full documentation here: [Rearrange](https://haliax.readthedocs.io/en/latest/rearrange/)

    The sequence form of `rearrange` rearranges an array so that its underlying storage conforms to axes.
    axes may include up to 1 ellipsis, indicating that the remaining axes should be
    permuted in the same order as the array's axes.

    For example, if array has axes (a, b, c, d, e, f) and axes is (e, f, a, ..., c),
    then the output array will have axes (e, f, a, b, c, d).

    The string form of `rearrange` works similarly to einops.rearrange, but also supports named axes and unordered matching.
    The string form of `rearrange` comes in two forms:

    * **Ordered strings** are like einops strings, with the only significant difference that flattened axes are named with
    a colon, e.g. `B H W C -> B (E: H W C)`.
    * **Unordered strings** match axes by name and are marked with the addition of curly braces,
    e.g. `{H W C} -> ... C H W` or `{H W C} -> ... (E: H W C)`

    As with einops, you can provide axis sizes to unflatten axes. For instance, to turn an image patches,
    `hax.rearrange(x, '{ B (H: w1 H) (W: w1 W)} -> (B: B h1 w1) H W ...', H=32, W=32)
    will turn a batch of images into a batch of image patches. Bindings can also be [haliax.Axis][] objects,
    or strings that will be used as the actual name of the resulting axis.

    Examples:
        >>> import haliax as hax
        >>> import jax.random as jrandom
        >>> B, H, W, C = hax.Axis("B", 8), hax.Axis("H", 32), hax.Axis("W", 32), hax.Axis("C", 3)
        >>> x = hax.random.normal( (B, H, W, C))
        >>> # Sequence-based rearrange
        >>> hax.rearrange(x, (C, B, H, W))
        >>> hax.rearrange(x, (C, ...)) # ellipsis means "keep the rest of the axes in the same order"
        >>> # String-based rearrange
        >>> # permute the axes
        >>> hax.rearrange(x, "B H W C -> C B H W")
        >>> # flatten the image (note the assignment of a new name to the flattened axis)
        >>> hax.rearrange(x, "B H W C -> B (E: H W C)")
        >>> # turn the image into patches
        >>> hax.rearrange(x, "{ B (H: h1 H) (W: w1 W) C } -> (B: B h1 w1) (E: H W C) ...", H=2, W=2)
        >>> # names can be longer than one character
        >>> hax.rearrange(x, "{ B (H: h1 H) (W: w1 W) C } -> (B: B h1 w1) (embed: H W C) ...", H=2, W=2)
    """

    if len(args) == 1:
        axes = args[0]
        if isinstance(axes, str):
            return einops_rearrange(array, axes, **kwargs)
        else:
            return axis_spec_rearrange(array, axes)
    elif len(args) > 1:
        raise TypeError("Only one positional argument allowed")

    kwargs = dict(kwargs)
    expression = kwargs.pop("expression", None)
    if expression is not None:
        return einops_rearrange(array, expression, **kwargs)
    else:
        axes = kwargs.pop("axes", None)
        if axes is None:
            raise TypeError("Must specify either axes or expression")
        return axis_spec_rearrange(array, axes)


def axis_spec_rearrange(array: NamedArray, axes: Sequence[AxisSelector | EllipsisType]) -> NamedArray:
    if len(axes) == 0 and len(array.axes) != 0:
        raise ValueError("No axes specified")

    # various fast paths
    if len(axes) == 1 and axes[0] is Ellipsis:
        return array

    if axes == array.axes:
        return array

    if axes[-1] is Ellipsis and array.axes[0 : len(axes) - 1] == axes[0 : len(axes) - 1]:
        return array

    if axes[0] is Ellipsis and array.axes[len(axes) - 1 :] == axes[1:]:
        return array

    if axes.count(Ellipsis) > 1:
        raise ValueError("Only one ellipsis allowed")

    used_indices = [False] * len(array.axes)
    permute_spec: list[int | EllipsisType] = []
    ellipsis_pos = None
    for ax in axes:
        if ax is Ellipsis:
            permute_spec.append(Ellipsis)  # will revisit
            ellipsis_pos = len(permute_spec) - 1
        else:
            assert isinstance(ax, Axis) or isinstance(ax, str)  # please mypy
            index = array._lookup_indices(ax)
            if index is None:
                raise ValueError(f"Axis {ax} not found in {array}")
            if used_indices[index]:
                raise ValueError(f"Axis {ax} specified more than once")
            used_indices[index] = True
            permute_spec.append(index)

    if not all(used_indices):
        # find the ellipsis position, replace it with all the unused indices
        if ellipsis_pos is None:
            missing_axes = [ax for i, ax in enumerate(array.axes) if not used_indices[i]]
            raise ValueError(f"Axes {missing_axes} not found and no ... specified. Array axes: {array.axes}") from None

        permute_spec[ellipsis_pos : ellipsis_pos + 1] = tuple(i for i in range(len(array.axes)) if not used_indices[i])
    elif ellipsis_pos is not None:
        permute_spec.remove(Ellipsis)

    out_axes = tuple(array.axes[i] for i in typing.cast(list[int], permute_spec))
    return NamedArray(jnp.transpose(array.array, permute_spec), out_axes)


def einops_rearrange(array: NamedArray, expression: str, **bindings: AxisSelector | int) -> NamedArray:
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


def _plan_rearrange(
    original_str, lhs: Expression, rhs: Expression, array: NamedArray, input_bindings: Mapping[str, AxisSelector | int]
) -> _Plan:
    aliases = _resolve_bindings(array, input_bindings)
    grouped_new_shapes = _determine_initial_reshape(original_str, lhs, array, aliases)
    intermediate_axes = tuple(ax for split_axes in grouped_new_shapes for ax in split_axes)

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


class AliasTable:
    bindings: dict[str, AxisSelector]  # names in the string to either axes or

    def __init__(self, bindings):
        self.bindings = bindings

    def dealias_binding(self, binding: str) -> Optional[AxisSelector]:
        return self.bindings.get(binding, None)

    def bind_alias(self, alias: str, axis: Axis, expr, char_range):
        if axis.name in self.bindings:
            if self.bindings[alias] != axis:
                _raise_error(f"Alias {alias} is assigned to more than one axis", expr, char_range)
        else:
            self.bindings[alias] = axis


def _resolve_bindings(array, bindings: Mapping[str, Axis | str | int]) -> AliasTable:
    b: dict[str, AxisSelector] = {}
    for name, selector in bindings.items():
        if isinstance(selector, str):
            try:
                selector = array.resolve_axis(selector)
            except ValueError:
                pass
        elif isinstance(selector, int):
            selector = Axis(name, selector)
        assert not isinstance(selector, int)
        b[name] = selector
    return AliasTable(b)


def _determine_final_transpose_and_reshape(
    expression, rhs: Expression, aliases: "AliasTable", intermediate_axes: tuple[Axis, ...]
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
                previous_capture = rhs.captures[cpos - 1]
                _raise_error("Only one ellipsis allowed", expression, previous_capture.char_range)
            position_of_ellipsis = cpos
            final_axes.append(Ellipsis)
            transposition_order.append(Ellipsis)
            position_of_ellipsis_in_transposition = len(transposition_order) - 1
            continue

        assert not isinstance(capture, EllipsisType)

        binding = capture.binding
        if binding is None:
            _raise_error("All rhs axes must have a name. Use (name: bindings)", expression, capture.char_range)

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

        # Do not bind here because axis names can be reused

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

    # one subtle difference between the lhs and the rhs is the handling of binding in expressions like (a: b c)
    # in the lhs, this means that a is bound to an axis, and b and c are split out from a
    # in the rhs, this means that b and c must already be bound to axes, and a is a new axis name
    # this makes sense in the same way that pattern matching works

    if lhs.is_ordered:
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

            axis_index_for_capture[cpos] = axis_pos
            covered_axes.add(axis_pos)
            axis_pos += 1

        # bind from the right, take care to check second ellipsis
        if ellipsis_pos is not None:
            axis_pos = len(array.axes) - 1
            for cpos in range(len(lhs.captures) - 1, ellipsis_pos, -1):
                capture = lhs.captures[cpos]
                if capture == Ellipsis:
                    _raise_error("Only one ellipsis allowed", expression, None)

                assert not isinstance(capture, EllipsisType)

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
            else:
                maybe_alias = axis_name(maybe_alias)

            try:
                axis = array.resolve_axis(maybe_alias)
                # aliases.bind_alias(capture.binding, axis, expression, capture.char_range)
            except ValueError:
                _raise_error(f"Could not resolve axis {maybe_alias}", expression, capture.char_range)

            try:
                axis_index = array.axes.index(axis)
            except ValueError:
                _raise_error(f"Axis {axis} is not in the array", expression, capture.char_range)
            if new_shapes[axis_index] is not None:
                _raise_error(f"Axis {axis} is bound more than once", expression, capture.char_range)

            new_axes = _solve_split_axes(axis, capture, aliases, used_new_names, expression)
            new_shapes[axis_index] = new_axes  # type: ignore

    for i in range(len(array.axes)):
        if new_shapes[i] is None:
            new_shapes[i] = [array.axes[i]]

    return new_shapes  # type: ignore


def _solve_split_axes(axis, capture, aliases, used_new_names, expression):
    """
    Given an axis and a capture of the form (a: b c) or (b c) on the lhs, solve for the new axes.
    """
    new_axes: list[Optional[Axis]] = []
    unsolved_axis_index: Optional[int] = None

    # easy case: 1 axis in capture
    if len(capture.axes) == 1:
        new_axis_name = capture.axes[0]
        if new_axis_name in used_new_names:
            _raise_error(f"Capture {new_axis_name} is assigned more than once", expression, capture.char_range)
        used_new_names.add(new_axis_name)

        new_axes.append(axis)
        aliases.bind_alias(new_axis_name, axis, expression, capture.char_range)

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
