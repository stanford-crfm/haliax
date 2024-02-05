import jax.numpy as jnp

import haliax as hax
from haliax import Axis, NamedArray
from haliax._src.einsum import einsum


def test_einsum_basic_positional():
    Height = Axis("Height", 2)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    m1 = NamedArray(jnp.ones((Height.size, Width.size, Depth.size)), (Height, Width, Depth))
    m2 = NamedArray(jnp.ones((Depth.size, Width.size, Height.size)), (Depth, Width, Height))

    assert jnp.all(jnp.equal(einsum("i j k,k j i-> j k", m1, m2).array, jnp.einsum("ijk,kji->jk", m1.array, m2.array)))
    assert jnp.all(
        jnp.equal(
            einsum("i j k,k j i->k", m1, m2).array,
            jnp.einsum("ijk,kji->k", m1.array, m2.array),
        )
    )
    assert jnp.all(
        jnp.equal(
            einsum("i j k,k j i->", m1, m2).array,
            jnp.einsum("ijk,kji->", m1.array, m2.array),
        )
    )

    # reduce to scalar
    assert jnp.all(
        jnp.equal(
            einsum("i j k,k j i->", m1, m2).array,
            jnp.einsum("ijk,kji->", m1.array, m2.array),
        )
    )


def test_einsum_basic_named():
    Height = Axis("Height", 2)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    m1 = hax.ones((Height, Width, Depth))
    m2 = hax.ones((Depth, Width, Height))

    assert jnp.all(
        jnp.equal(
            einsum("{Height Width Depth} -> Height Width", m1, m2).array, jnp.einsum("ijk,kji->ij", m1.array, m2.array)
        )
    )

    assert jnp.all(jnp.equal(einsum(" -> Height Width", m1, m2).array, jnp.einsum("ijk,kji->ij", m1.array, m2.array)))
    assert jnp.all(
        jnp.equal(
            einsum("{Height Width Depth} -> Depth", m1, m2).array,
            jnp.einsum("ijk,kji->k", m1.array, m2.array),
        )
    )
    assert jnp.all(
        jnp.equal(
            einsum("-> Depth", m1, m2).array,
            jnp.einsum("ijk,kji->k", m1.array, m2.array),
        )
    )

    assert jnp.all(
        jnp.equal(
            einsum("{Height Width Depth} -> ", m1, m2).array,
            jnp.einsum("ijk,kji->", m1.array, m2.array),
        )
    )

    # outer product
    assert jnp.all(
        jnp.equal(
            einsum("{} ->", m1, m2).array,
            jnp.einsum("ijk,kji->ijk", m1.array, m2.array),
        )
    )


def test_einsum_unordered_ellipses():
    Height = Axis("Height", 2)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    m1 = hax.ones((Height, Width, Depth))
    m2 = hax.ones((Depth, Width, Height))

    # collapse all
    assert jnp.all(
        jnp.equal(
            einsum("{...} ->", m1, m2).array,
            jnp.einsum("ijk,kji->", m1.array, m2.array),
        )
    )

    # keep all
    assert jnp.all(
        jnp.equal(
            einsum("{...} -> ...", m1, m2).array,
            jnp.einsum("ijk,kji->ijk", m1.array, m2.array),
        )
    )

    assert jnp.all(
        jnp.equal(
            einsum("{} -> ...", m1, m2).array,
            jnp.einsum("ijk,kji->ijk", m1.array, m2.array),
        )
    )

    assert jnp.all(
        jnp.equal(
            einsum("{Depth} -> Depth ...", m1, m2).array,
            jnp.einsum("ijk,kji->kij", m1.array, m2.array),
        )
    )

    assert jnp.all(
        jnp.equal(
            einsum("{Depth Height} -> Depth ... Height", m1, m2).array,
            jnp.einsum("ijk,kji->kji", m1.array, m2.array),
        )
    )

    assert jnp.all(
        jnp.equal(
            einsum("{Depth} -> ... Depth ...", m1, m2).array,
            jnp.einsum("ijk,kji->ijk", m1.array, m2.array),
        )
    )


def test_einsum_ordered_ellipsis():
    Height = Axis("Height", 2)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    m1 = hax.ones((Height, Width, Depth))
    m2 = hax.ones((Depth, Width, Height))

    assert jnp.all(
        jnp.equal(
            einsum("h ... d,d ... h-> ...", m1, m2).array,
            jnp.einsum("ijk,kji->j", m1.array, m2.array),
        )
    )

    assert jnp.all(
        jnp.equal(
            einsum("h ... d,d ... h-> ... d", m1, m2).array,
            jnp.einsum("ijk,kji->jk", m1.array, m2.array),
        )
    )
