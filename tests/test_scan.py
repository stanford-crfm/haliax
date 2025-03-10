import equinox as eqx
import jax
import pytest
from equinox import filter_value_and_grad

import haliax as hax
from haliax.jax_utils import tree_checkpoint_name
from haliax.nn.scan import BlockSeq, Stacked, StackedCheckpointPolicy


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

    with pytest.raises(ValueError):
        m.scan(x, key=jax.random.split(jax.random.PRNGKey(2), Block.size))

    with pytest.raises(ValueError):
        m_seq.scan(x, key=jax.random.split(jax.random.PRNGKey(2), Block.size))


def test_using_scan():
    class Module(eqx.Module):
        named: hax.NamedArray
        array: jax.Array
        static: int = eqx.static_field()

        def __call__(self, x, *, key):
            return x + self.array + self.static + hax.random.normal(key, x.axes), x * 2

        @staticmethod
        def init(named, array, static):
            return Module(named=named, array=array, static=static)

    Block = hax.Axis("block", 4)
    E = hax.Axis("E", 10)

    initial_named = hax.random.uniform(jax.random.PRNGKey(0), (Block, E))

    m = Stacked.init(Block, Module)(named=initial_named, array=jax.numpy.ones(Block.size), static=1)

    x = hax.random.uniform(jax.random.PRNGKey(1), (E,))
    y, intermediates = m.scan(x, key=jax.random.split(jax.random.PRNGKey(2), Block.size))

    assert y.axes == (E,)
    assert intermediates.axes == (Block, E)


def test_scan_with_aux_named_args():
    class Module(eqx.Module):
        named: hax.NamedArray
        array: jax.Array
        static: int = eqx.static_field()

        def __call__(self, x, y, *, key):
            return x + self.array + self.static + hax.random.normal(key, x.axes), x * 2 + y

        @staticmethod
        def init(named, array, static):
            return Module(named=named, array=array, static=static)

    Block = hax.Axis("block", 4)
    E = hax.Axis("E", 10)

    initial_named = hax.random.uniform(jax.random.PRNGKey(0), (Block, E))
    initial_y = hax.random.uniform(jax.random.PRNGKey(1), (E,))

    m = Stacked.init(Block, Module)(named=initial_named, array=jax.numpy.ones(Block.size), static=1)
    m_seq = BlockSeq.init(Block, Module)(named=initial_named, array=jax.numpy.ones(Block.size), static=1)

    x = hax.random.uniform(jax.random.PRNGKey(1), (E,))
    z, z_scan = m.scan(x, initial_y, key=jax.random.split(jax.random.PRNGKey(2), Block.size))
    z_seq, z_seq_scan = m_seq.scan(x, initial_y, key=jax.random.split(jax.random.PRNGKey(2), Block.size))
    assert hax.all(hax.isclose(z, z_seq, atol=1e-5))

    assert hax.all(hax.isclose(z_scan, z_seq_scan, atol=1e-5))


def test_stacked_to_state_dict():
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

    state_dict = m.to_state_dict()
    m2 = m.from_state_dict(state_dict)
    input = hax.random.uniform(jax.random.PRNGKey(1), (E,))
    key = jax.random.split(jax.random.PRNGKey(2), Block.size)

    y = m.fold(input, key=key)
    y2 = m2.fold(input, key=key)

    assert hax.all(hax.equal(y, y2))


def test_checkpoint_carries():
    class Module(eqx.Module):
        named: hax.NamedArray

        def __call__(self, x):
            y = tree_checkpoint_name(hax.sin(x + self.named), "sin")
            y = tree_checkpoint_name(hax.cos(y + x), "cos")
            return y + x

        @staticmethod
        def init(named):
            return Module(named=named)

    Block = hax.Axis("block", 4)
    E = hax.Axis("E", 10)

    initial_named = hax.random.uniform(jax.random.PRNGKey(0), (Block, E))

    carry_policy = StackedCheckpointPolicy(save_carries=True, save_outputs=False, save_block_internals=False)
    save_nothing = StackedCheckpointPolicy(save_carries=False, save_outputs=False, save_block_internals=False)
    save_everything = StackedCheckpointPolicy(save_carries=True, save_outputs=True, save_block_internals=True)
    save_internals = StackedCheckpointPolicy(save_carries=False, save_outputs=False, save_block_internals=True)
    save_cos = StackedCheckpointPolicy(save_carries=False, save_outputs=False, save_block_internals=["cos"])
    save_sin_carry = StackedCheckpointPolicy(save_carries=True, save_outputs=False, save_block_internals=["sin"])

    for name, (policy, expected_scan_shapes) in {
        "carry": (carry_policy, [(E.size,), (Block.size, E.size)]),
        "nothing": (save_nothing, [(E.size,)]),
        "everything": (save_everything, [(E.size,), (Block.size, E.size), (Block.size, E.size)]),
        "internals": (save_internals, [(E.size,), (Block.size, E.size), (Block.size, E.size)]),
        "cos": (save_cos, [(E.size,), (Block.size, E.size)]),
        "sin": (save_sin_carry, [(E.size,), (Block.size, E.size), (Block.size, E.size)]),
    }.items():
        m = Stacked.init(
            Block,
            Module,
            gradient_checkpointing=policy,
        )(named=initial_named)

        def loss_fn(m, x):
            y = m.fold(x)
            return hax.sum(y).scalar()

        grad_fn = filter_value_and_grad(loss_fn)

        jaxpr = jax.make_jaxpr(grad_fn)(m, hax.random.uniform(jax.random.PRNGKey(1), (E,)))
        closed_call = next(eqn for eqn in jaxpr.jaxpr.eqns if eqn.primitive == jax.core.closed_call_p)
        out_shapes = [out.aval.shape for out in closed_call.outvars]

        # saved_residuals doesn't give me sensible results, so I'm doing this by hand
        assert out_shapes == expected_scan_shapes, f"{name}: Expected {expected_scan_shapes}, got {out_shapes}"
