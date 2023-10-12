import pytest
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


# some axes
W = Axis("W", 4)
H = Axis("H", 6)
C = Axis("C", 3)
D = Axis("D", 2)
B = Axis("B", 5)
Q = hax.Axis("Q", B.size * H.size)

z = hax.random.randint(PRNGKey(0), (B, D, H, W, C), 0, 255)
zq = hax.random.randint(PRNGKey(0), (Q, D, W, C), 0, 255)


def test_basic_identity():
    assert rearrange(z, "b d h w c -> b d h w c").axes == (B, D, H, W, C)
    assert (rearrange(z, "b d h w c -> b d h w c").array == z.array).all()


def test_basic_rearrange_transpose():
    assert rearrange(z, "b d h w c -> b h w d c").axes == (B, H, W, D, C)
    # make sure the values are right too
    z_t = z.array.transpose((0, 2, 3, 1, 4))
    assert (rearrange(z, "b d h w c -> b h w d c").array == z_t).all()


def test_basic_rearrange_unordered():
    assert rearrange(z, "{B D H W C} -> B H W D C").axes == (B, H, W, D, C)
    z_t = z.array.transpose((0, 2, 3, 1, 4))
    assert (rearrange(z, "{B D H W C} -> B H W D C").array == z_t).all()

    assert rearrange(z, "{C W H D B} -> B H W D C").axes == (B, H, W, D, C)
    assert (rearrange(z, "{C W H D B} -> B H W D C").array == z_t).all()


def test_rearrange_with_ellipsis():
    assert rearrange(z, "... w c -> ... c w").axes == (B, D, H, C, W)
    assert rearrange(z, "b d ... -> d ... b").axes == (D, H, W, C, B)

    assert rearrange(z, "b ... c -> b c ...").axes == (B, C, D, H, W)
    # make sure the values are right too
    z_t = z.array.transpose((0, 4, 1, 2, 3))
    assert (rearrange(z, "b ... c -> b c ...").array == z_t).all()


def test_rearrange_with_ellipsis_unordered():
    assert rearrange(z, "{W C} -> ... C W").axes == (B, D, H, C, W)
    assert rearrange(z, "{B D} -> D ... B").axes == (D, H, W, C, B)

    assert rearrange(z, "{B C} -> B C ...").axes == (B, C, D, H, W)
    # make sure the values are right too
    z_t = z.array.transpose((0, 4, 1, 2, 3))
    assert (rearrange(z, "{B C} -> B C ...").array == z_t).all()
    assert (rearrange(z, "{qqq C} -> qqq C ...", qqq=B).array == z_t).all()


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


def test_rearrange_with_flattening_unordered():
    assert rearrange(z, "{B D H} -> D (Q: B H) ...").axes == (D, Q, W, C)
    # make sure the values are right too
    z_t = z.array.transpose((1, 0, 2, 3, 4)).reshape((D.size, Q.size, W.size, C.size))
    assert (rearrange(z, "{B D H} -> D (Q: B H) ...").array == z_t).all()

    # test with ellipsis
    assert rearrange(z, "{B D H ... C} -> D (Q: B H) ... C").axes == (D, Q, W, C)
    # make sure the values are right too
    z_t = z.array.transpose((1, 0, 2, 3, 4)).reshape((D.size, Q.size, W.size, C.size))
    assert (rearrange(z, "{B D H ... C} -> D (Q: B H) ... C").array == z_t).all()


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


def test_rearrange_with_unflatten_unordered():
    assert rearrange(zq, "{ (Q: B H) D } -> B D H ... ", B=5).axes == (B, D, H, W, C)
    # make sure the values are right too
    z_t = zq.array.reshape((B.size, H.size, D.size, W.size, C.size)).transpose((0, 2, 1, 3, 4))
    assert (rearrange(zq, "{ (Q: B H) D } -> B D H ... ", B=5).array == z_t).all()

    # test with explicit axis arg
    assert rearrange(zq, "{ (Q: b H) D } -> b D H ... ", b=B).axes == (B, D, H, W, C)
    # make sure the values are right too
    z_t = zq.array.reshape((B.size, H.size, D.size, W.size, C.size)).transpose((0, 2, 1, 3, 4))
    assert (rearrange(zq, "{ (Q: b H) D } -> b D H ... ", b=B).array == z_t).all()


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


def test_with_unflatten_flatten_unordered():
    Z = Axis("Z", B.size * C.size)
    assert rearrange(zq, "{ W D C (Q: B H)} -> D (Z: B C) W H", H=H).axes == (D, Z, W, H)
    # make sure the values are right too
    z_t = (
        zq.array.reshape((B.size, H.size, D.size, W.size, C.size))
        .transpose((2, 0, 4, 3, 1))
        .reshape((D.size, Z.size, W.size, H.size))
    )
    assert (rearrange(zq, "{ W D C (Q: B H)} -> D (Z: B C) W H", H=H).array == z_t).all()


def test_semantic_errors():
    with pytest.raises(ValueError, match="Too many axes in lhs"):
        rearrange(z, "b d h w c q -> b d h w c q")

    with pytest.raises(ValueError, match="Not all axes are bound"):
        rearrange(z, "b d h w -> b d h w c q")

    with pytest.raises(ValueError, match="Too many axes in lhs"):
        rearrange(z, "b d h w x y z -> b d h w c q")

    with pytest.raises(ValueError, match="Axis q is not bound on the lhs"):
        rearrange(z, "b d h w c -> b d h w c q")

    with pytest.raises(ValueError, match="Only one ellipsis allowed"):
        rearrange(z, "b d ... h w c ... -> b d h w c ...")

    with pytest.raises(ValueError, match="Pattern q is bound to"):
        rearrange(z, "b d c q q -> b d h w c q")

    with pytest.raises(ValueError, match="Capture q is assigned more than once"):
        rearrange(z, "b d c (q q) z -> b d h w c q", q=4)

    with pytest.raises(ValueError, match="Not all intermediate axes are used"):
        rearrange(z, "b d c q z -> b d c z")

    with pytest.raises(ValueError, match="is bound more than once"):
        rearrange(z, "b d c q z ... d e f -> b d c q d e f")

    with pytest.raises(ValueError, match="Axis q is not bound on the lhs"):
        rearrange(z, "b d h w c -> b d h w c q")

    with pytest.raises(ValueError, match="Only one ellipsis allowed"):
        rearrange(z, "b d ... h w c ... -> b d h w c ...")

    with pytest.raises(ValueError, match="Pattern q is bound to"):
        rearrange(z, "b d c q q -> b d h w c q")

    with pytest.raises(ValueError, match="Capture q is assigned more than once"):
        rearrange(z, "b d c (q q) z -> b d h w c q", q=4)

    with pytest.raises(ValueError, match="Not all intermediate axes are used"):
        rearrange(z, "b d c q z -> b d c z")

    with pytest.raises(ValueError, match="is bound more than once"):
        rearrange(z, "b d c q z ... d e f -> b d c q d e f")

    with pytest.raises(ValueError, match="not divide"):
        rearrange(z, "(x y) d c q z -> d c q x y z", x=13, y=17)

    with pytest.raises(ValueError, match="not divide"):
        rearrange(z, "(x y) d c q z -> d c q x y z", x=13)


def test_examples():
    # Cases to support:
    # identity
    # * 'a b c d -> a b c d' or 'a, b, c, d -> a, b, c, d'
    assert rearrange(z, "a b c d e -> a b c d e").axes == (B, D, H, W, C)
    # merge a and b into m
    # * 'a b c d -> (m: a b) c d'
    assert rearrange(z, "a b c d e -> (M: a b) c d e").axes == (Axis("M", B.size * D.size), H, W, C)
    #  > without rearrange or names: x.reshape((a * b, c, d))
    # split a into b and c
    # * '(a: b c) d -> b c d'
    assert rearrange(zq, "(q: b h) d, w, c -> b h d w c", b=B, h="H").axes == (B, H, D, W, C)
    #  > without rearrange or names: x.reshape((b, c, d))
    # reorder a and b
    # * 'a b ... -> b a ...'
    #  > without rearrange or names: x.transpose((1, 0, ...))
    # split into 2 groups, rename to x and y
    # * 'a b c d -> (x: a b) (y: c d)'  # split into two groups, rename to x and y
    #  > without rearrange or names: x.reshape((a * b, c * d))
    # unordered match of a, b, c by name, move to end
    # * `{c b a} -> ... a b c`
    #   > without rearrange or names: x.transpose((2, 1, 0, ...))
    # unordered match of a and d by name, split a into b and c, reorder
    # * `{(a: b c) d} -> ... b d c`
    #   > without rearrange or names: x.reshape(..., b, c, d).transpose((..., -3, -1, -2))
    # unordered match of a and d by name, split a into b and c, d into e and f, reorder
    # * `{(a: b c) (d: e f)} -> ... b e c f`
    # flatten each image into a vector
    # * `{h w c} -> (c: c h w)`
    # split each image into 4 smaller (top-left, top-right, bottom-left, bottom-right), 128 = 32 * 2 * 2:
    # * `{b (h: h1 h) (w: w1 w)} -> (b: b h1 w1) h w ...`
    # sequence of flattened patches:
    # * `{b (h: h1 h) (w: w1 w) c} -> (b: b h1 w1) (c: c h w) ...`
    # unet attention reordering:
    # * { (embed: qkv heads c) h w } -> qkv heads c (pos: h w)
    # space to depth
    # * {b (h: h1 h) (w: w1 w) c} -> ... b h w (c: c h1 w1)
