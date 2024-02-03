import pytest

from haliax._src.parsing import parse_rearrangement


def _simplify_captures(expr):
    def simplify_capture(capture):
        if capture == Ellipsis:
            return Ellipsis
        elif (capture.binding == capture.axes[0] or capture.binding is None) and len(capture.axes) == 1:
            return capture.axes[0]
        elif capture.binding is None:
            return capture.axes
        else:
            return {capture.binding: capture.axes}

    return [simplify_capture(capture) for capture in expr.captures]


def test_parse_rearrangement_simple():
    lhs, rhs = parse_rearrangement("a b c d -> b, c, a, d")
    assert lhs.is_ordered
    assert _simplify_captures(lhs) == ["a", "b", "c", "d"]
    assert rhs.is_ordered
    assert _simplify_captures(rhs) == ["b", "c", "a", "d"]

    lhs, rhs = parse_rearrangement("a ... c d -> b c a d")
    assert lhs.is_ordered
    assert _simplify_captures(lhs) == ["a", Ellipsis, "c", "d"]
    assert rhs.is_ordered
    assert _simplify_captures(rhs) == ["b", "c", "a", "d"]

    # longer identifiers
    lhs, rhs = parse_rearrangement("a_longer b123 c d -> b123 c a_longer d")
    assert lhs.is_ordered
    assert _simplify_captures(lhs) == ["a_longer", "b123", "c", "d"]
    assert rhs.is_ordered
    assert _simplify_captures(rhs) == ["b123", "c", "a_longer", "d"]


def test_parse_paren_groups():
    lhs, rhs = parse_rearrangement("a (b c) d -> b c a d")
    assert lhs.is_ordered
    assert _simplify_captures(lhs) == ["a", ("b", "c"), "d"]
    assert rhs.is_ordered
    assert _simplify_captures(rhs) == ["b", "c", "a", "d"]

    lhs, rhs = parse_rearrangement("a (b: c) (d: e f) -> b c a d")
    assert lhs.is_ordered
    assert _simplify_captures(lhs) == ["a", {"b": ("c",)}, {"d": ("e", "f")}]


def test_parse_unordered():
    lhs, rhs = parse_rearrangement("{a b c d} -> {b c a d}")
    assert not lhs.is_ordered
    assert _simplify_captures(lhs) == ["a", "b", "c", "d"]
    assert not rhs.is_ordered
    assert _simplify_captures(rhs) == ["b", "c", "a", "d"]

    lhs, rhs = parse_rearrangement("{(c: a b), d e,} -> (q: a d e) b")
    assert not lhs.is_ordered
    assert _simplify_captures(lhs) == [{"c": ("a", "b")}, "d", "e"]
    assert rhs.is_ordered
    assert _simplify_captures(rhs) == [{"q": ("a", "d", "e")}, "b"]


def test_parse_quoted_identifiers():
    lhs, rhs = parse_rearrangement("a \"b c\" d -> 'b c' a d")
    assert lhs.is_ordered
    assert _simplify_captures(lhs) == ["a", "b c", "d"]
    assert rhs.is_ordered
    assert _simplify_captures(rhs) == ["b c", "a", "d"]

    lhs, rhs = parse_rearrangement("{a \"b c\" (d: 'hello')} -> b c a d")
    assert not lhs.is_ordered
    assert _simplify_captures(lhs) == ["a", "b c", {"d": ("hello",)}]
    assert rhs.is_ordered
    assert _simplify_captures(rhs) == ["b", "c", "a", "d"]


def test_parse_errors():
    with pytest.raises(ValueError, match="Unexpected end of string"):
        parse_rearrangement("a b")

    with pytest.raises(ValueError, match="Expected }"):
        parse_rearrangement("{ a -> c")

    with pytest.raises(ValueError, match="Unexpected }"):
        parse_rearrangement("a } -> c")

    with pytest.raises(ValueError, match="Unexpected }"):
        parse_rearrangement("(a: b } -> c")

    with pytest.raises(ValueError, match="Unexpected character"):
        parse_rearrangement("(a b ! -> c d e")

    with pytest.raises(ValueError, match="Identifier expected"):
        parse_rearrangement("a b ! -> c d e")
