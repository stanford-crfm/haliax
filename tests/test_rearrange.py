from jax.random import PRNGKey

import haliax as hax
from haliax import Axis
from haliax._src.rearrange import parse_rearrangement, rearrange


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


# some axes
W = Axis("W", 4)
H = Axis("H", 6)
C = Axis("C", 3)
D = Axis("D", 2)
B = Axis("B", 5)
Q = hax.Axis("Q", B.size * H.size)

z = hax.random.randint(PRNGKey(0), (B, D, H, W, C), 0, 255)
zq = hax.random.randint(PRNGKey(0), (Q, D, W, C), 0, 255)


def test_basic_rearrange():
    assert rearrange(z, "b d h w c -> b h w d c").axes == (B, H, W, D, C)
    # make sure the values are right too
    z_t = z.array.transpose((0, 2, 3, 1, 4))
    assert (rearrange(z, "b d h w c -> b h w d c").array == z_t).all()


def test_rearrange_with_ellipsis():
    assert rearrange(z, "... w c -> ... c w").axes == (B, D, H, C, W)
    assert rearrange(z, "b d ... -> d ... b").axes == (D, H, W, C, B)

    assert rearrange(z, "b ... c -> b c ...").axes == (B, C, D, H, W)
    # make sure the values are right too
    z_t = z.array.transpose((0, 4, 1, 2, 3))
    assert (rearrange(z, "b ... c -> b c ...").array == z_t).all()


def test_rearrange_with_flattening():
    assert rearrange(z, "b d h w c -> d (Q: b h) w c").axes == (D, Q, W, C)
    # make sure the values are right too
    z_t = z.array.transpose((1, 0, 2, 3, 4)).reshape((D.size, Q.size, W.size, C.size))
    assert (rearrange(z, "b d h w c -> d (Q: b h) w c").array == z_t).all()

    # test with ellipsis
    assert rearrange(z, "b d h ... c -> d (Q: b h) ... c").axes == (D, Q, W, C)
    # make sure the values are right too
    z_t = z.array.transpose((1, 0, 2, 3, 4)).reshape((D.size, Q.size, W.size, C.size))
    assert (rearrange(z, "b d h ... c -> d (Q: b h) ... c").array == z_t).all()


def test_rearrange_with_unflatten():
    assert rearrange(zq, "(Q: B H) d, w c -> B d H w c", B=5).axes == (B, D, H, W, C)
    # make sure the values are right too
    z_t = zq.array.reshape((B.size, H.size, D.size, W.size, C.size)).transpose((0, 2, 1, 3, 4))
    assert (rearrange(zq, "(Q: B H) d, w c -> B d H w c", B=5).array == z_t).all()

    # test with explicit axis arg
    assert rearrange(zq, "(Q: b H) d, w c -> b d H w c", b=B).axes == (B, D, H, W, C)
    # make sure the values are right too
    z_t = zq.array.reshape((B.size, H.size, D.size, W.size, C.size)).transpose((0, 2, 1, 3, 4))
    assert (rearrange(zq, "(Q: b H) d, w c -> b d H w c", b=B).array == z_t).all()


def test_with_unflatten_flatten():
    Z = Axis("Z", B.size * C.size)
    assert rearrange(zq, "(Q: B H) d w c -> d (Z: B c) w H", H=H).axes == (D, Z, W, H)
    # make sure the values are right too
    z_t = (
        zq.array.reshape((B.size, H.size, D.size, W.size, C.size))
        .transpose((2, 0, 4, 3, 1))
        .reshape((D.size, Z.size, W.size, H.size))
    )
    assert (rearrange(zq, "(Q: B H) d w c -> d (Z: B c) w H", H=H).array == z_t).all()
