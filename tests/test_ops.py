from typing import Callable

import jax.numpy as jnp
import pytest
from jax.random import PRNGKey

import haliax as hax
from haliax import Axis, NamedArray


def test_trace():
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)
    named1 = hax.random.uniform(PRNGKey(0), (Width, Depth))
    trace1 = hax.trace(named1, Width, Depth)
    assert jnp.all(jnp.isclose(trace1.array, jnp.trace(named1.array)))
    assert len(trace1.axes) == 0

    trace1 = hax.trace(named1, "Width", "Depth")
    assert jnp.all(jnp.isclose(trace1.array, jnp.trace(named1.array)))
    assert len(trace1.axes) == 0

    Height = Axis("Height", 10)
    named2 = hax.random.uniform(PRNGKey(0), (Height, Width, Depth))
    trace2 = hax.trace(named2, Width, Depth)
    assert jnp.all(jnp.isclose(trace2.array, jnp.trace(named2.array, axis1=1, axis2=2)))
    assert trace2.axes == (Height,)

    trace2 = hax.trace(named2, "Width", "Depth")
    assert jnp.all(jnp.isclose(trace2.array, jnp.trace(named2.array, axis1=1, axis2=2)))
    assert trace2.axes == (Height,)


def test_add():
    Height = Axis("Height", 10)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    named1 = hax.random.uniform(PRNGKey(0), (Height, Width, Depth))
    named2 = hax.random.uniform(PRNGKey(1), (Height, Width, Depth))

    named3 = named1 + named2
    assert jnp.all(jnp.isclose(named3.array, named1.array + named2.array))

    named2_reorder = named2.rearrange((Width, Height, Depth))
    named4 = named1 + named2_reorder
    named4 = named4.rearrange((Height, Width, Depth))
    assert jnp.all(jnp.isclose(named4.array, named1.array + named2.array))


def test_add_broadcasting():
    Height = Axis("Height", 10)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    named1 = hax.random.uniform(PRNGKey(0), (Height, Width, Depth))
    named2 = hax.random.uniform(PRNGKey(1), (Width, Depth))

    named3 = named1 + named2
    assert jnp.all(jnp.isclose(named3.array, named1.array + named2.array))

    named2_reorder = named2.rearrange((Depth, Width))
    named4 = named1 + named2_reorder
    named4 = named4.rearrange((Height, Width, Depth))

    assert jnp.all(jnp.isclose(named4.array, named1.array + named2.array))

    # now for the broadcasting we don't like
    named5 = hax.random.uniform(PRNGKey(1), (Height, Depth))
    named6 = hax.random.uniform(PRNGKey(2), (Width, Depth))

    with pytest.raises(ValueError):
        _ = named5 + named6


def test_add_scalar():
    Height = Axis("Height", 10)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    named1 = hax.random.uniform(PRNGKey(0), (Height, Width, Depth))
    named2 = named1 + 1.0
    assert jnp.all(jnp.isclose(named2.array, named1.array + 1.0))

    named3 = 1.0 + named1
    assert jnp.all(jnp.isclose(named3.array, named1.array + 1.0))


def test_add_no_overlap():
    Height = Axis("Height", 10)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    named1: NamedArray = hax.random.uniform(PRNGKey(0), (Height))
    named2 = hax.random.uniform(PRNGKey(1), (Width, Depth))

    with pytest.raises(ValueError):
        _ = named1 + named2

    named3 = named1.broadcast_to((Height, Width, Depth)) + named2

    assert jnp.all(
        jnp.isclose(named3.array, named1.array.reshape((-1, 1, 1)) + named2.array.reshape((1,) + named2.array.shape))
    )


# TODO: tests for other ops:


@pytest.mark.parametrize("use_jit", [False, True])
def test_where(use_jit):
    Height = Axis("Height", 10)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    hax_where: Callable = hax.where
    if use_jit:
        hax_where = hax.named_jit(hax_where)

    named1 = hax.random.uniform(PRNGKey(0), (Height, Width, Depth))
    named2 = hax.random.uniform(PRNGKey(1), (Height, Width, Depth))

    hax_where(0.0, named1, 0.0)

    named3 = hax_where(named1 > named2, named1, named2)

    assert jnp.all(jnp.isclose(named3.array, jnp.where(named1.array > named2.array, named1.array, named2.array)))

    named2_reorder = named2.rearrange((Width, Height, Depth))
    named4 = hax_where(named1 > named2_reorder, named1, named2_reorder)
    named4 = named4.rearrange((Height, Width, Depth))
    assert jnp.all(jnp.isclose(named4.array, jnp.where(named1.array > named2.array, named1.array, named2.array)))

    # now some broadcasting
    named5 = hax.random.uniform(PRNGKey(1), (Height, Width))
    named6 = hax.random.uniform(PRNGKey(2), Width)

    named7 = hax_where(named5 > named6, named5, named6)
    named7 = named7.rearrange((Height, Width))
    assert jnp.all(jnp.isclose(named7.array, jnp.where(named5.array > named6.array, named5.array, named6.array)))

    # now for the broadcasting we don't like
    named5 = hax.random.uniform(PRNGKey(1), (Height, Depth))
    named6 = hax.random.uniform(PRNGKey(2), (Width, Depth))

    with pytest.raises(ValueError):
        _ = hax_where(named5 > named6, named5, named6)

    # now single argument mode
    Volume = hax.Axis("Volume", Height.size * Width.size * Depth.size)
    named7 = hax.random.uniform(PRNGKey(0), (Height, Width, Depth))
    named8, named9, named10 = hax_where(named7 > 0.5, fill_value=-1, new_axis=Volume)
    unnamed_7 = named7.array
    unnamed_8, unnamed_9, unnamed_10 = jnp.where(unnamed_7 > 0.5, size=Volume.size, fill_value=-1)
    assert jnp.all(unnamed_8 == named8.array)


def test_clip():
    Height = Axis("Height", 10)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    named1 = hax.random.uniform(PRNGKey(0), (Height, Width, Depth))
    named2 = hax.clip(named1, 0.3, 0.7)
    assert jnp.all(jnp.isclose(named2.array, jnp.clip(named1.array, 0.3, 0.7)))

    named2_reorder = named2.rearrange((Width, Height, Depth))
    named3 = hax.clip(named2_reorder, 0.3, 0.7)
    named3 = named3.rearrange((Height, Width, Depth))
    assert jnp.all(jnp.isclose(named3.array, jnp.clip(named2.array, 0.3, 0.7)))

    # now some interesting broadcasting
    lower = hax.full((Height, Width), 0.3)
    upper = hax.full((Width, Depth), 0.7)
    named4 = hax.clip(named1, lower, upper)
    named4 = named4.rearrange((Height, Width, Depth))

    assert jnp.all(
        jnp.isclose(
            named4.array,
            jnp.clip(
                named1.array,
                lower.array.reshape((Height.size, Width.size, 1)),
                upper.array.reshape((1, Width.size, Depth.size)),
            ),
        )
    )


def test_tril_triu():
    Height = Axis("Height", 10)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    for hax_fn, jnp_fn in [(hax.tril, jnp.tril), (hax.triu, jnp.triu)]:
        named1 = hax.random.uniform(PRNGKey(0), (Height, Width, Depth))
        named2 = hax_fn(named1, Width, Depth)
        assert jnp.all(jnp.isclose(named2.array, jnp_fn(named1.array)))

        named3 = hax_fn(named1, Width, Depth, k=1)
        assert jnp.all(jnp.isclose(named3.array, jnp_fn(named1.array, k=1)))

        named4 = hax_fn(named1, Width, Depth, k=-1)
        assert jnp.all(jnp.isclose(named4.array, jnp_fn(named1.array, k=-1)))

        named5 = hax_fn(named1, Height, Depth)
        expected5 = jnp_fn(named1.array.transpose([1, 0, 2]))
        assert jnp.all(jnp.isclose(named5.array, expected5))


def test_mean_respects_where():
    Height = Axis("Height", 10)
    Width = Axis("Width", 3)

    named1 = hax.random.uniform(PRNGKey(0), (Height, Width))
    where = hax.random.uniform(PRNGKey(1), (Height, Width)) > 0.5

    assert not jnp.all(jnp.isclose(hax.mean(named1), hax.mean(named1, where=where)))
    assert jnp.all(jnp.isclose(hax.mean(named1, where=where), jnp.mean(named1.array, where=where.array)))

    # check broadcasting
    where = hax.random.uniform(PRNGKey(2), (Height,)) > 0.5
    assert not jnp.all(jnp.isclose(hax.mean(named1), hax.mean(named1, where=where)))
    assert jnp.all(
        jnp.isclose(hax.mean(named1, where=where), jnp.mean(named1.array, where=where.array.reshape((-1, 1))))
    )


def test_reductions_produce_scalar_named_arrays_when_None_axis():
    Height = Axis("Height", 10)
    Width = Axis("Width", 3)

    named1 = hax.random.uniform(PRNGKey(0), (Height, Width))

    assert isinstance(hax.mean(named1, axis=None), NamedArray)

    # But if we specify axes, we always get a NamedArray, even if it's a scalar
    assert isinstance(hax.mean(named1, axis=("Height", "Width")), NamedArray)
    assert hax.mean(named1, axis=("Height", "Width")).axes == ()


def test_norm():
    H = Axis("H", 3)
    W = Axis("W", 4)
    D = Axis("D", 5)

    x_array = jnp.arange(H.size * W.size * D.size, dtype=jnp.float32).reshape((H.size, W.size, D.size))
    x_named = NamedArray(x_array, (H, W, D))

    # Test case 1: Default ord, axis=None (vector 2-norm of flattened array)
    norm_none_axis = hax.norm(x_named)
    expected_none_axis = jnp.linalg.norm(x_array)
    assert norm_none_axis.axes == ()
    assert jnp.allclose(norm_none_axis.array, expected_none_axis)

    # Test case 2: Vector norm (ord=1, axis=H)
    norm_vec_h = hax.norm(x_named, ord=1, axis=H)
    expected_vec_h = jnp.linalg.norm(x_array, ord=1, axis=0)
    assert norm_vec_h.axes == (W, D)
    assert jnp.allclose(norm_vec_h.array, expected_vec_h)

    # Test case 3: Matrix norm (ord='fro', axis=(H, W))
    norm_mat_hw = hax.norm(x_named, ord="fro", axis=(H, W))
    expected_mat_hw = jnp.linalg.norm(x_array, ord="fro", axis=(0, 1))
    assert norm_mat_hw.axes == (D,)
    assert jnp.allclose(norm_mat_hw.array, expected_mat_hw)

    # Test case 4: ord=inf, single axis W
    norm_inf_w = hax.norm(x_named, ord=jnp.inf, axis=W)
    expected_inf_w = jnp.linalg.norm(x_array, ord=jnp.inf, axis=1)
    assert norm_inf_w.axes == (H, D)
    assert jnp.allclose(norm_inf_w.array, expected_inf_w)

    # Test case 5: Matrix norm (ord=2, axis=(W, D))
    # For matrix ord=2 (largest singular value)
    M = Axis("M", 4)
    N = Axis("N", 5) # Non-square is fine for singular values
    K = Axis("K", 3)
    # Ensure non-zero values for more stable singular value computation if array contains zeros
    y_array = jnp.arange(M.size * N.size * K.size, dtype=jnp.float32).reshape((M.size, N.size, K.size)) + 1.0
    y_named = NamedArray(y_array, (M, N, K))

    norm_mat_mn_ord2 = hax.norm(y_named, ord=2, axis=(M, N))
    expected_mat_mn_ord2 = jnp.linalg.norm(y_array, ord=2, axis=(0, 1))
    assert norm_mat_mn_ord2.axes == (K,)
    # Relax tolerance for singular value computations as they can be sensitive
    assert jnp.allclose(norm_mat_mn_ord2.array, expected_mat_mn_ord2, atol=1e-5, rtol=1e-5)


    # Test case 6: Scalar input
    scalar_array = jnp.array(5.0, dtype=jnp.float32)
    scalar_named = NamedArray(scalar_array, ())
    norm_scalar = hax.norm(scalar_named)
    expected_scalar = jnp.linalg.norm(scalar_array)
    assert norm_scalar.axes == ()
    assert jnp.allclose(norm_scalar.array, expected_scalar)


    # Test with string axis selectors
    # Re-use expected_vec_h from Test Case 2
    norm_vec_h_str = hax.norm(x_named, ord=1, axis="H")
    assert norm_vec_h_str.axes == (W, D)
    assert jnp.allclose(norm_vec_h_str.array, expected_vec_h)

    # Re-use expected_mat_hw from Test Case 3
    norm_mat_hw_str = hax.norm(x_named, ord="fro", axis=("H", "W"))
    assert norm_mat_hw_str.axes == (D,)
    assert jnp.allclose(norm_mat_hw_str.array, expected_mat_hw)

    # Error cases for axis
    with pytest.raises(ValueError, match="Axis 'MissingAxis' not found"):
        hax.norm(x_named, axis="MissingAxis")

    with pytest.raises(ValueError, match="Axis 'MissingAxis' not found"):
        hax.norm(x_named, axis=(H, "MissingAxis"))

    # New error cases for axis tuple length
    with pytest.raises(ValueError, match="If `axis` is a tuple, it must contain 1 or 2 axes, but got 3."):
        hax.norm(x_named, axis=(H, W, D))

    with pytest.raises(ValueError, match="If `axis` is a tuple, it must contain 1 or 2 axes, but got 0."):
        hax.norm(x_named, axis=())


    # Test with a 1D array (vector)
    V = Axis("V", 10)
    v_array = jnp.arange(V.size, dtype=jnp.float32)
    v_named = NamedArray(v_array, (V,))

    # Default ord=2
    norm_v_axis_none = hax.norm(v_named, axis=None)
    expected_v_axis_none = jnp.linalg.norm(v_array)
    assert norm_v_axis_none.axes == ()
    assert jnp.allclose(norm_v_axis_none.array, expected_v_axis_none)

    # Default ord=2, specific axis
    norm_v_axis_v = hax.norm(v_named, axis=V)
    expected_v_axis_v = jnp.linalg.norm(v_array, axis=0)
    assert norm_v_axis_v.axes == ()
    assert jnp.allclose(norm_v_axis_v.array, expected_v_axis_v)

    # Test with tuple of one axis
    norm_v_axis_tuple_v = hax.norm(v_named, axis=(V,))
    assert norm_v_axis_tuple_v.axes == ()
    assert jnp.allclose(norm_v_axis_tuple_v.array, expected_v_axis_v)


    # Test case: 0-dim array (scalar already tested, but ensure consistency)
    s_array = jnp.array(3.0)
    s_named = NamedArray(s_array, ())
    norm_s_ax_none = hax.norm(s_named, axis=None)
    assert norm_s_ax_none.axes == ()
    assert jnp.allclose(norm_s_ax_none.array, jnp.linalg.norm(s_array, axis=None))
