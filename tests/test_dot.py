# these test if the rearrange logic works for partial orders
import pytest
from jax import numpy as jnp

import haliax as hax
from haliax import Axis, NamedArray
from haliax._src.dot import rearrange_to_fit_order


def test_dot():
    Height = Axis("Height", 2)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    m1 = NamedArray(jnp.ones((Height.size, Width.size, Depth.size)), (Height, Width, Depth))
    m2 = NamedArray(jnp.ones((Depth.size, Width.size, Height.size)), (Depth, Width, Height))

    assert jnp.all(jnp.equal(hax.dot(m1, m2, axis=Height).array, jnp.einsum("ijk,kji->jk", m1.array, m2.array)))
    assert jnp.all(
        jnp.equal(
            hax.dot(m1, m2, axis=(Height, Width)).array,
            jnp.einsum("ijk,kji->k", m1.array, m2.array),
        )
    )
    assert jnp.all(
        jnp.equal(
            hax.dot(m1, m2, axis=(Height, Width, Depth)).array,
            jnp.einsum("ijk,kji->", m1.array, m2.array),
        )
    )

    # reduce to scalar
    assert jnp.all(
        jnp.equal(
            hax.dot(m1, m2, axis=None),
            jnp.einsum("ijk,kji->", m1.array, m2.array),
        )
    )


def test_dot_string_selection():
    Height = Axis("Height", 2)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    m1 = NamedArray(jnp.ones((Height.size, Width.size, Depth.size)), (Height, Width, Depth))
    m2 = NamedArray(jnp.ones((Depth.size, Width.size, Height.size)), (Depth, Width, Height))

    assert jnp.all(jnp.equal(hax.dot(m1, m2, axis="Height").array, jnp.einsum("ijk,kji->jk", m1.array, m2.array)))
    assert jnp.all(
        jnp.equal(
            hax.dot(m1, m2, axis=("Height", "Width")).array,
            jnp.einsum("ijk,kji->k", m1.array, m2.array),
        )
    )
    assert jnp.all(
        jnp.equal(
            hax.dot(m1, m2, axis=("Height", "Width", "Depth")).array,
            jnp.einsum("ijk,kji->", m1.array, m2.array),
        )
    )


def test_dot_errors_if_different_sized_axes():
    Height = Axis("Height", 2)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    H2 = Axis("Height", 4)

    m1 = NamedArray(jnp.ones((Height.size, Width.size, Depth.size)), (Height, Width, Depth))
    m2 = NamedArray(jnp.ones((Depth.size, Width.size, H2.size)), (Depth, Width, H2))

    with pytest.raises(ValueError):
        hax.dot(m1, m2, axis="Height")


def assert_partial_order_respected(partial_order, output):
    positions = {el: i for i, el in enumerate(output)}

    last_pos = -1
    for el in partial_order:
        if el is ...:
            # Reset last_pos for flexible positions
            last_pos = -1
        else:
            # Check if the element is in the correct order
            assert el in positions, f"{el} is missing in the output"
            assert positions[el] > last_pos, f"Partial order not respected for {el}"
            last_pos = positions[el]


def test_basic_order():
    partial_order = ("apple", ..., "banana")
    candidates = ("banana", "apple", "cherry")
    expected_output = ("apple", "cherry", "banana")
    actual_output = rearrange_to_fit_order(partial_order, candidates)
    assert actual_output == expected_output
    assert_partial_order_respected(partial_order, actual_output)


def test_start_with_ellipsis():
    partial_order = (..., "apple", "banana")
    candidates = ("banana", "apple", "cherry")
    actual_output = rearrange_to_fit_order(partial_order, candidates)
    assert_partial_order_respected(partial_order, actual_output)
    assert actual_output == ("cherry", "apple", "banana")


def test_end_with_ellipsis():
    partial_order = ("apple", ..., "banana", ...)
    candidates = ("banana", "apple", "cherry")
    actual_output = rearrange_to_fit_order(partial_order, candidates)
    assert_partial_order_respected(partial_order, actual_output)

    # this one could be either but we'll assert the order so we catch changes
    assert actual_output == ("apple", "banana", "cherry")


def test_full_flexibility():
    partial_order = (...,)
    candidates = ("banana", "apple", "cherry")
    actual_output = rearrange_to_fit_order(partial_order, candidates)
    assert_partial_order_respected(partial_order, actual_output)


def test_no_flexibility():
    partial_order = ("apple", "banana")
    candidates = ("banana", "apple", "cherry")
    with pytest.raises(ValueError):
        rearrange_to_fit_order(partial_order, candidates)


def test_final_ellipsis():
    partial_order = ("apple", "banana", ...)
    candidates = ("banana", "apple", "cherry")
    actual_output = rearrange_to_fit_order(partial_order, candidates)
    assert_partial_order_respected(partial_order, actual_output)
    assert actual_output == ("apple", "banana", "cherry")


def test_lots_of_ellipsis():
    partial_order = ("apple", ..., "banana", ..., "cherry", ...)
    candidates = ("banana", "orange", "cherry", "apple", "grape")
    actual_output = rearrange_to_fit_order(partial_order, candidates)
    assert_partial_order_respected(partial_order, actual_output)
    assert actual_output == ("apple", "banana", "orange", "cherry", "grape")


def test_no_ellipsis():
    partial_order = ("apple", "banana", "cherry")
    candidates = ("banana", "apple", "cherry")
    actual_output = rearrange_to_fit_order(partial_order, candidates)
    assert_partial_order_respected(partial_order, actual_output)
    assert actual_output == ("apple", "banana", "cherry")


def test_no_elements():
    partial_order = (...,)
    candidates = ()
    actual_output = rearrange_to_fit_order(partial_order, candidates)
    assert_partial_order_respected(partial_order, actual_output)
    assert actual_output == ()


def test_missing_elements_errors():
    partial_order = ("qux", ...)
    candidates = ("banana", "apple", "cherry")
    with pytest.raises(ValueError):
        rearrange_to_fit_order(partial_order, candidates)


def test_duplicate_elements_errors():
    partial_order: tuple = ("apple", "apple", ...)
    candidates = ("banana", "apple", "cherry")
    with pytest.raises(ValueError):
        rearrange_to_fit_order(partial_order, candidates)

    candidates = ("banana", "apple", "apple")

    with pytest.raises(ValueError):
        rearrange_to_fit_order(partial_order, candidates)

    partial_order = ("apple", "banana", "grape", ...)

    with pytest.raises(ValueError):
        rearrange_to_fit_order(partial_order, candidates)


def test_dot_with_output_axes():
    Height = Axis("Height", 2)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    m1 = NamedArray(jnp.ones((Height.size, Width.size, Depth.size)), (Height, Width, Depth))
    m2 = NamedArray(jnp.ones((Depth.size, Width.size, Height.size)), (Depth, Width, Height))

    assert jnp.all(
        jnp.equal(
            hax.dot(m1, m2, axis=Height, out_axes=(Width, ...)).array,
            jnp.einsum("ijk,kji->jk", m1.array, m2.array),
        )
    )

    assert jnp.all(
        jnp.equal(
            hax.dot(m1, m2, axis=Height, out_axes=(Depth, ...)).array,
            jnp.einsum("ijk,kji->kj", m1.array, m2.array),
        )
    )

    assert jnp.all(
        jnp.equal(
            hax.dot(m1, m2, axis=Height, out_axes=(Depth, Width)).array,
            jnp.einsum("ijk,kji->kj", m1.array, m2.array),
        )
    )

    assert jnp.all(
        jnp.equal(
            hax.dot(m1, m2, axis=(), out_axes=(Depth, Height, ...)).array,
            jnp.einsum("ijk,kji->kij", m1.array, m2.array),
        )
    )

    assert jnp.all(
        jnp.equal(
            hax.dot(m1, m2, axis=(), out_axes=(..., Depth, Height)).array,
            jnp.einsum("ijk,kji->jki", m1.array, m2.array),
        )
    )

    assert jnp.all(
        jnp.equal(
            hax.dot(m1, m2, axis=(), out_axes=(..., Depth, ...)).array,
            jnp.einsum("ijk,kji->ijk", m1.array, m2.array),
        )
    )

    assert jnp.all(
        jnp.equal(
            hax.dot(m1, m2, axis=(), out_axes=(..., Depth, Height, ...)).array,
            jnp.einsum("ijk,kji->kij", m1.array, m2.array),
        )
    )
