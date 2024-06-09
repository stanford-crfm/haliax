import equinox as eqx
import jax.numpy as jnp
from jax.random import PRNGKey

import haliax as hax
from haliax import Axis, NamedArray
from haliax.util import is_named_array


def test_scan():
    Height = Axis("Height", 10)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)
    named1 = hax.random.uniform(PRNGKey(0), (Height, Width, Depth))

    def scan_fun(acc, x):
        return acc + jnp.sum(x.array), x.take(Width, 2)

    total, selected = hax.scan(scan_fun, Height)(0.0, named1)

    assert jnp.all(jnp.isclose(total, jnp.sum(named1.array, axis=(0, 1, 2))))
    assert jnp.all(jnp.equal(selected.array, named1.take(Width, 2).array))

    total, selected = hax.scan(scan_fun, "Height")(0.0, named1)

    assert jnp.all(jnp.isclose(total, jnp.sum(named1.array, axis=(0, 1, 2))))
    assert jnp.all(jnp.equal(selected.array, named1.take("Width", 2).array))


def test_scan_not_0th_axis():
    Height = Axis("Height", 10)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)
    named1 = hax.random.uniform(PRNGKey(0), (Height, Width, Depth))

    def scan_fun(acc, x):
        return acc + jnp.sum(x.array), x.take(Width, 2)

    total, selected = hax.scan(scan_fun, Depth)(0.0, named1)

    assert jnp.all(jnp.isclose(total, jnp.sum(named1.array, axis=(0, 1, 2))))
    assert jnp.all(jnp.equal(selected.array, named1.take(Width, 2).rearrange(selected.axes).array))


def test_scan_str_arg():
    Height = Axis("Height", 10)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)
    named1 = hax.random.uniform(PRNGKey(0), (Height, Width, Depth))

    def scan_fun(acc, x):
        return acc + jnp.sum(x.array), x.take("Width", 2)

    total, selected = hax.scan(scan_fun, "Height")(0.0, named1)

    assert jnp.all(jnp.isclose(total, jnp.sum(named1.array, axis=(0, 1, 2))))
    assert jnp.all(jnp.equal(selected.array, named1.take(Width, 2).array))


def test_scan_static_args():
    Height = Axis("Height", 10)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)
    named1 = hax.random.uniform(PRNGKey(0), (Height, Width, Depth))

    def scan_fun(acc, x, static1, *, static2):
        assert static1 is True
        assert static2 is False
        return acc + jnp.sum(x.array), x.take(Width, 2)

    total, selected = hax.scan(scan_fun, Depth, is_scanned=is_named_array)(0.0, named1, True, static2=False)

    assert jnp.all(jnp.isclose(total, jnp.sum(named1.array, axis=(0, 1, 2))))
    assert jnp.all(jnp.equal(selected.array, named1.take(Width, 2).rearrange(selected.axes).array))


def test_scan_doesnt_scan_scalars():
    Height = Axis("Height", 10)
    named1 = hax.random.uniform(PRNGKey(0), (Height,))

    def scan_fun(acc, z, x):
        return acc + z * x, x * z

    total, selected = hax.scan(scan_fun, Height)(0.0, 4.0, named1)

    assert jnp.all(jnp.isclose(total, jnp.sum(named1.array * 4.0)))
    assert jnp.all(jnp.equal(selected.array, named1.array * 4.0))


def test_scan_doesnt_scan_init():
    Height = Axis("Height", 10)
    named1 = hax.random.uniform(PRNGKey(0), (Height,))

    init = jnp.arange(Height.size, dtype=jnp.float32)

    def scan_fun(acc, z, x):
        out = acc + z * x, x * z
        return out

    total, selected = hax.scan(scan_fun, Height)(init, 4.0, named1)

    assert jnp.all(jnp.isclose(total, init + jnp.sum(named1.array * 4.0)))

    # double check with named array init
    total, selected = hax.scan(scan_fun, Height)(hax.named(init, "Height"), 4.0, named1)

    assert jnp.all(jnp.isclose(total.array, init + jnp.sum(named1.array * 4.0)))

    # now do fold
    def fold_fun(acc, z, x):
        return acc + z * x

    total = hax.fold(fold_fun, Height)(init, 4.0, named1)

    assert jnp.all(jnp.isclose(total, init + jnp.sum(named1.array * 4.0)))

    total = hax.fold(fold_fun, Height)(hax.named(init, "Height"), 4.0, named1)

    assert jnp.all(jnp.isclose(total.array, init + jnp.sum(named1.array * 4.0)))


def test_reduce():
    Height = Axis("Height", 10)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)
    named1 = hax.random.uniform(PRNGKey(0), (Height, Width, Depth))

    acc = hax.zeros((Height, Width))

    total = hax.fold(lambda x, y: x + y, Depth)(acc, named1)

    assert jnp.all(jnp.isclose(total.rearrange(acc.axes).array, jnp.sum(named1.array, axis=2)))


def test_reduce_str_args():
    Height = Axis("Height", 10)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)
    named1 = hax.random.uniform(PRNGKey(0), (Height, Width, Depth))

    acc = hax.zeros((Height, Width))

    total = hax.fold(lambda x, y: x + y, "Depth")(acc, named1)

    assert jnp.all(jnp.isclose(total.rearrange(acc.axes).array, jnp.sum(named1.array, axis=2)))


def test_reduce_static_args():
    Height = Axis("Height", 10)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)
    named1 = hax.random.uniform(PRNGKey(0), (Height, Width, Depth))

    def fold_fun(acc, x, static1, *, static2):
        assert static1 is True
        assert static2 is False
        return NamedArray(acc.array + x.rearrange(acc.axes).array, acc.axes)

    acc = hax.zeros((Height, Width))

    total = hax.fold(fold_fun, Depth)(acc, named1, True, static2=False)

    assert jnp.all(jnp.isclose(total.rearrange(acc.axes).array, jnp.sum(named1.array, axis=2)))


def test_reduce_doesnt_reduce_scalars():
    Height = Axis("Height", 10)
    named1 = hax.random.uniform(PRNGKey(0), (Height,))

    acc = hax.zeros((Height,))

    total = hax.fold(lambda x, z, y: x + z * y, Height)(acc, 4.0, named1)

    assert jnp.all(jnp.isclose(total.rearrange(acc.axes).array, jnp.sum(named1.array * 4.0)))


def test_vmap_unmapped_args():
    Batch = Axis("Batch", 10)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)
    named1 = hax.random.uniform(PRNGKey(0), (Width, Depth))

    def vmap_fun(x):
        return x.take(Width, 2)

    selected = hax.vmap(vmap_fun, Batch)(named1)

    expected_jax = jnp.array([named1.take(Width, 2).array for _ in range(Batch.size)])
    expected_names = (Batch, Depth)

    assert jnp.all(jnp.equal(selected.array, expected_jax))
    assert selected.axes == expected_names


def test_vmap_non_static_bool_fields():
    Batch = Axis("Batch", 10)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)
    named1 = hax.random.uniform(PRNGKey(0), (Width, Depth))

    class Foo(eqx.Module):
        field: NamedArray
        flag: bool = False

        def __init__(self):
            super().__init__()
            self.field = named1
            self.flag = True

    vmap_foo = hax.vmap(Foo, Batch)()

    assert vmap_foo.field.axes == (Batch, Width, Depth)
    assert vmap_foo.flag is True


def test_vmap_mapped_args():
    Batch = Axis("Batch", 10)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)
    named1 = hax.random.uniform(PRNGKey(0), (Batch, Width, Depth))

    def vmap_fun(x):
        return x.sum(Width)

    selected = hax.vmap(vmap_fun, Batch)(named1)

    expected_jax = jnp.array([named1.sum(Width).array for _ in range(Batch.size)])
    expected_names = (Batch, Depth)

    assert jnp.all(jnp.equal(selected.array, expected_jax))
    assert selected.axes == expected_names


def test_vmap_str_args():
    Batch = Axis("Batch", 10)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)
    named1 = hax.random.uniform(PRNGKey(0), (Batch, Width, Depth))

    def vmap_fun(x):
        return x.sum(Width)

    selected = hax.vmap(vmap_fun, "Batch")(named1)

    expected_jax = jnp.array([named1.sum(Width).array for _ in range(Batch.size)])
    expected_names = (Batch, Depth)

    assert jnp.all(jnp.equal(selected.array, expected_jax))
    assert selected.axes == expected_names

    # also ensure that this works when we return a non-haliax array
    def vmap_fun2(x):
        return x.sum(Width).array

    selected = hax.vmap(vmap_fun2, "Batch")(named1)

    assert jnp.all(jnp.equal(selected, expected_jax))


def test_vmap_mapped_kwarg():
    Batch = Axis("Batch", 10)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)
    named1 = hax.random.uniform(PRNGKey(0), (Batch, Width, Depth))

    def vmap_fun(x):
        return x.sum(Width)

    selected = hax.vmap(vmap_fun, Batch)(x=named1)

    expected_jax = jnp.array([named1.sum(Width).array for _ in range(Batch.size)])
    expected_names = (Batch, Depth)

    assert jnp.all(jnp.equal(selected.array, expected_jax))
    assert selected.axes == expected_names


def test_vmap_static_args():
    Batch = Axis("Batch", 10)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)
    named1 = hax.random.uniform(PRNGKey(0), (Batch, Width, Depth))

    def vmap_fun(x, y):
        return x.sum(Width) if y else x

    selected = hax.vmap(vmap_fun, Batch)(named1, True)

    expected = hax.sum(named1, Width)

    assert jnp.all(jnp.equal(selected.array, expected.array))
    assert selected.axes == expected.axes


def test_vmap_error_for_incorrectly_specified_args():
    class Module(eqx.Module):
        # this should usually be declared static, but we're simulating a user error
        field: Axis

        def __call__(self, x):
            return x.sum(self.field)

    Batch = Axis("Batch", 10)
    Width = Axis("Width", 3)

    hax.vmap(lambda a: Module(a), Batch)(Width)
