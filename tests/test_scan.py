import equinox as eqx
import jax

import haliax as hax
from haliax.nn.scan import BlockSeq, Stacked


def test_unstacked():
    class Module(eqx.Module):
        named: hax.NamedArray
        array: jax.Array
        static: int = eqx.static_field()

        def __call__(self, x, *, key):
            return x + self.array + self.static

        @staticmethod
        def init(named, array, static):
            return Module(named=named, array=array, static=static)

    Block = hax.Axis("block", 4)
    E = hax.Axis("E", 10)

    initial_named = hax.random.uniform(jax.random.PRNGKey(0), (Block, E))

    m = Stacked.init(Block, Module)(named=initial_named, array=jax.numpy.ones(Block.size), static=1)

    assert m.stacked.named.axes == (Block, E)
    assert m.stacked.array.shape == (Block.size,)
    assert m.stacked.static == 1

    unstacked = m.unstacked()

    assert len(unstacked) == Block.size

    for i, module in enumerate(unstacked):
        assert module.named.axes == (E,)
        assert module.array.shape == ()
        assert module.static == 1

        assert hax.all(module.named == m.stacked.named["block", i])
        assert hax.all(module.array == m.stacked.array[i])


def test_seq_and_stacked_give_same_results():
    class Module(eqx.Module):
        named: hax.NamedArray
        array: jax.Array
        static: int = eqx.static_field()

        def __call__(self, x, *, key):
            return x + self.array + self.static + hax.random.normal(key, x.axes)

        @staticmethod
        def init(named, array, static):
            return Module(named=named, array=array, static=static)

    Block = hax.Axis("block", 4)
    E = hax.Axis("E", 10)

    initial_named = hax.random.uniform(jax.random.PRNGKey(0), (Block, E))

    m = Stacked.init(Block, Module)(named=initial_named, array=jax.numpy.ones(Block.size), static=1)
    m_seq = BlockSeq.init(Block, Module)(named=initial_named, array=jax.numpy.ones(Block.size), static=1)

    x = hax.random.uniform(jax.random.PRNGKey(1), (E,))
    y = m.fold(x, key=jax.random.split(jax.random.PRNGKey(2), Block.size))
    y_seq = m_seq.fold(x, key=jax.random.split(jax.random.PRNGKey(2), Block.size))

    assert hax.all(hax.isclose(y, y_seq, atol=1e-5))
