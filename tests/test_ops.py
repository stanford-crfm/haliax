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

    # Test case 2: Default ord, axis=None, keepdims=True
    norm_none_axis_keepdims = hax.norm(x_named, keepdims=True)
    expected_none_axis_keepdims_arr = jnp.linalg.norm(x_array, keepdims=True)
    # For axis=None, keepdims=True, haliax.norm creates new 1-sized axes with original names
    expected_axes_kept = tuple(Axis(name=ax.name, size=1) for ax in x_named.axes)
    assert norm_none_axis_keepdims.axes == expected_axes_kept
    assert norm_none_axis_keepdims.shape == {ax.name: 1 for ax in x_named.axes}
    assert jnp.allclose(norm_none_axis_keepdims.array, expected_none_axis_keepdims_arr)


    # Test case 3: Vector norm (ord=1, axis=H)
    norm_vec_h = hax.norm(x_named, ord=1, axis=H)
    expected_vec_h = jnp.linalg.norm(x_array, ord=1, axis=0)
    assert norm_vec_h.axes == (W, D)
    assert jnp.allclose(norm_vec_h.array, expected_vec_h)

    # Test case 4: Vector norm (ord=1, axis=H, keepdims=True)
    norm_vec_h_keepdims = hax.norm(x_named, ord=1, axis=H, keepdims=True)
    expected_vec_h_keepdims = jnp.linalg.norm(x_array, ord=1, axis=0, keepdims=True)
    assert norm_vec_h_keepdims.axes == (H, W, D) # Axis H is kept with size 1 implicitly by jax
    assert norm_vec_h_keepdims.shape == {H.name: H.size, W.name: W.size, D.name: D.size} # jax op keeps dim H.size
    # The shape of the underlying array for norm_vec_h_keepdims should be (1, W.size, D.size) if H was reduced and kept.
    # However, jax.linalg.norm with keepdims=True for axis=0 on (H,W,D) results in (1,W,D)
    # Haliax norm maps this to (H,W,D) where H axis in NamedArray has original size, but data has 1 along H.
    # This is a slight mismatch if H.size > 1.
    # Let's re-evaluate the keepdims logic for NamedArray.
    # If an axis is reduced and keepdims=True, its NamedArray Axis should become size 1.
    # The current implementation of hax.norm for keepdims=True when axis is specified:
    #   return NamedArray(result_array, x.axes)
    # This is only correct if result_array has the original x.axes dimensions.
    # jnp.linalg.norm(x.array, axis=0, keepdims=True) on (H,W,D) gives (1,W,D).
    # So NamedArray(jnp.ones((1,4,5)), (Axis("H",3), Axis("W",4), Axis("D",5))) is problematic.
    # The NamedArray's axes should reflect the actual data shape.
    # Let's adjust the expected axes for keepdims=True when specific axes are reduced.
    expected_axes_h_kept = (Axis(H.name, 1), W, D)
    # This requires hax.norm to modify the axis if it's reduced and kept.
    # The current hax.norm implementation:
    # if keepdims:
    #   if axis is None: kept_axes = tuple(Axis(name=ax.name, size=1) for ax in x.axes)
    #   else: # (this part was refactored)
    #       output_axes_list = list(x.axes)
    #       ...
    #       for index in reduced_int_indices: output_axes_list[index] = Axis(name=original_axis.name, size=1)
    #       return NamedArray(result_array, tuple(output_axes_list))
    expected_axes_h_kept = (Axis(H.name, 1), W, D)
    assert norm_vec_h_keepdims.axes == expected_axes_h_kept
    assert norm_vec_h_keepdims.shape == {H.name: 1, W.name: W.size, D.name: D.size}
    assert jnp.allclose(norm_vec_h_keepdims.array, expected_vec_h_keepdims)


    # Test case 5: Matrix norm (ord='fro', axis=(H, W))
    norm_mat_hw = hax.norm(x_named, ord="fro", axis=(H, W))
    expected_mat_hw = jnp.linalg.norm(x_array, ord="fro", axis=(0, 1))
    assert norm_mat_hw.axes == (D,)
    assert jnp.allclose(norm_mat_hw.array, expected_mat_hw)

    # Test case 6: Matrix norm (ord='fro', axis=(H, W), keepdims=True)
    norm_mat_hw_keepdims = hax.norm(x_named, ord="fro", axis=(H, W), keepdims=True)
    expected_mat_hw_keepdims = jnp.linalg.norm(x_array, ord="fro", axis=(0, 1), keepdims=True)
    expected_axes_hw_kept = (Axis(H.name, 1), Axis(W.name, 1), D)
    assert norm_mat_hw_keepdims.axes == expected_axes_hw_kept
    assert norm_mat_hw_keepdims.shape == {H.name: 1, W.name: 1, D.name: D.size}
    assert jnp.allclose(norm_mat_hw_keepdims.array, expected_mat_hw_keepdims)

    # Test case 7: ord=inf, single axis W
    norm_inf_w = hax.norm(x_named, ord=jnp.inf, axis=W)
    expected_inf_w = jnp.linalg.norm(x_array, ord=jnp.inf, axis=1)
    assert norm_inf_w.axes == (H, D)
    assert jnp.allclose(norm_inf_w.array, expected_inf_w)

    # Test case 8: ord=-inf, single axis D, keepdims=True
    norm_ninf_d_keepdims = hax.norm(x_named, ord=-jnp.inf, axis=D, keepdims=True)
    expected_ninf_d_keepdims = jnp.linalg.norm(x_array, ord=-jnp.inf, axis=2, keepdims=True)
    expected_axes_d_kept = (H, W, Axis(D.name, 1))
    assert norm_ninf_d_keepdims.axes == expected_axes_d_kept
    assert norm_ninf_d_keepdims.shape == {H.name: H.size, W.name: W.size, D.name: 1}
    assert jnp.allclose(norm_ninf_d_keepdims.array, expected_ninf_d_keepdims)

    # Test case 9: Matrix norm (ord=2, axis=(W, D))
    # For matrix ord=2, JAX requires square matrices if axis is a tuple of 2 ints.
    # jnp.linalg.norm(x, ord=2, axis=(ax1, ax2)) computes operator 2-norm (largest singular value)
    # Let's use a compatible shape for this.
    M = Axis("M", 4)
    N = Axis("N", 4) # For ord=2, if axis is tuple, it's matrix norm. If M!=N, it's fine.
    K = Axis("K", 3)
    y_array = jnp.arange(M.size * N.size * K.size, dtype=jnp.float32).reshape((M.size, N.size, K.size))
    y_named = NamedArray(y_array, (M, N, K))

    norm_mat_mn_ord2 = hax.norm(y_named, ord=2, axis=(M, N))
    expected_mat_mn_ord2 = jnp.linalg.norm(y_array, ord=2, axis=(0, 1))
    assert norm_mat_mn_ord2.axes == (K,)
    assert jnp.allclose(norm_mat_mn_ord2.array, expected_mat_mn_ord2)

    # Test case 10: Scalar input
    scalar_array = jnp.array(5.0, dtype=jnp.float32)
    scalar_named = NamedArray(scalar_array, ())
    norm_scalar = hax.norm(scalar_named)
    expected_scalar = jnp.linalg.norm(scalar_array)
    assert norm_scalar.axes == ()
    assert jnp.allclose(norm_scalar.array, expected_scalar)

    norm_scalar_keepdims = hax.norm(scalar_named, keepdims=True)
    expected_scalar_keepdims = jnp.linalg.norm(scalar_array, keepdims=True) # jax returns array(5.) not array([5.])
    assert norm_scalar_keepdims.axes == () # For scalar, keepdims doesn't change axes
    assert jnp.allclose(norm_scalar_keepdims.array, expected_scalar_keepdims)


    # Test with string axis selectors
    norm_vec_h_str = hax.norm(x_named, ord=1, axis="H")
    assert norm_vec_h_str.axes == (W, D)
    assert jnp.allclose(norm_vec_h_str.array, expected_vec_h)

    norm_mat_hw_str = hax.norm(x_named, ord="fro", axis=("H", "W"))
    assert norm_mat_hw_str.axes == (D,)
    assert jnp.allclose(norm_mat_hw_str.array, expected_mat_hw)

    # Error cases
    with pytest.raises(ValueError):
        hax.norm(x_named, axis="MissingAxis")

    with pytest.raises(ValueError):
        hax.norm(x_named, axis=(H, "MissingAxis"))

    # Test with a 1D array (vector)
    V = Axis("V", 10)
    v_array = jnp.arange(V.size, dtype=jnp.float32)
    v_named = NamedArray(v_array, (V,))

    norm_v_axis_none = hax.norm(v_named, axis=None) # Default ord=2
    expected_v_axis_none = jnp.linalg.norm(v_array)
    assert norm_v_axis_none.axes == ()
    assert jnp.allclose(norm_v_axis_none.array, expected_v_axis_none)

    norm_v_axis_v = hax.norm(v_named, axis=V) # Default ord=2
    expected_v_axis_v = jnp.linalg.norm(v_array, axis=0)
    assert norm_v_axis_v.axes == ()
    assert jnp.allclose(norm_v_axis_v.array, expected_v_axis_v)

    norm_v_axis_v_keepdims = hax.norm(v_named, axis=V, keepdims=True)
    expected_v_axis_v_keepdims = jnp.linalg.norm(v_array, axis=0, keepdims=True) # shape (1,)
    expected_v_kept_axis_v = (Axis(V.name, 1),)
    assert norm_v_axis_v_keepdims.axes == expected_v_kept_axis_v
    assert norm_v_axis_v_keepdims.shape == {V.name: 1}
    assert jnp.allclose(norm_v_axis_v_keepdims.array, expected_v_axis_v_keepdims)

    # Test case: axis=None, keepdims=True for 1D array
    norm_v_none_keepdims = hax.norm(v_named, keepdims=True)
    expected_v_none_keepdims_arr = jnp.linalg.norm(v_array, keepdims=True) # shape (1,)
    expected_v_axes_kept = (Axis(name=V.name, size=1),)
    assert norm_v_none_keepdims.axes == expected_v_axes_kept
    assert norm_v_none_keepdims.shape == {V.name: 1}
    assert jnp.allclose(norm_v_none_keepdims.array, expected_v_none_keepdims_arr)

    # Test case: 0-dim array (scalar already tested, but ensure consistency)
    s_array = jnp.array(3.0)
    s_named = NamedArray(s_array, ())
    norm_s_ax_none = hax.norm(s_named, axis=None)
    assert norm_s_ax_none.axes == ()
    assert jnp.allclose(norm_s_ax_none.array, jnp.linalg.norm(s_array, axis=None))

    norm_s_ax_none_keepdims = hax.norm(s_named, axis=None, keepdims=True)
    # jnp.linalg.norm(jnp.array(3.0), axis=None, keepdims=True) is array(3.)
    # s_named.axes is (), so expected axes is ()
    assert norm_s_ax_none_keepdims.axes == ()
    assert jnp.allclose(norm_s_ax_none_keepdims.array, jnp.linalg.norm(s_array, axis=None, keepdims=True))
