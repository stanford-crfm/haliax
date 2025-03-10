# Module Stacks

A core pattern for larger models in JAX is the "scan-over-layers" pattern, where you have a sequence of layers
that get stacked together, and you use [jax.lax.scan][] or [haliax.fold][] or [haliax.scan][] to apply them to a
sequence of inputs. In Haliax, layers are represented as [equinox.nn.Module][]s, and the [haliax.nn.Stacked][] module
provides a way to create a sequence of layers that can be applied to a sequence of inputs that implements the
scan-over-layers pattern.

## Stacked

[haliax.nn.Stacked][] lets you apply a layer sequentially to an input, scanning over a "Layers" axis. For instance,
a Transformer might use a Stacked for its Transformer blocks:


```python
class TransformerBlock(eqx.Module):

    def __init__(self, config: TransformerConfig, layer_index, *, key):
        attn_key, mlp_key = jax.random.split(key)
        self.attention = Attention.init(config, key=attn_key)
        self.mlp = MLP.init(config, key=mlp_key)
        self.ln1 = LayerNorm.init(config.Hidden)
        self.ln2 = LayerNorm.init(config.Hidden)
        self.layer_index = layer_index

    def __call__(self, x):
        y = self.attention(self.ln1(x))
        x = x + y
        y = self.mlp(self.ln2(x))
        return x + y

class Transformer(eqx.Module):
    def __init__(self, config: TransformerConfig):
        self.blocks = Stacked.init(Layers, TransformerBlock)(
            config,  # static configuration
            scale=hax.arange(Layers),  # dynamic configuration. Each layer gets a scalar scale value [0, 1, 2, ...]
            key=jax.random.split(key, Layers),  # dynamic configuration. Each layer gets a different key
        )

    def __call__(self, x: NamedArray) -> NamedArray:
        # morally the equivalent of:
        # for block in self.blocks:
        #     x = block(x)
        # Except that it works better with JAX compile times.

        return self.blocks.fold(x)
```

You can think of [haliax.nn.Stacked][] as an analog to PyTorch's
[torch.nn.Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html), except that
every layer in the sequence must have exactly the same shape and configuration.

Internally, a Stacked is a single copy of the module, except that every NamedArray inside that module
has a Block axis prepended (as though they were stacked with [haliax.stack][]). Similarly, every JAX array
inside the module has its first axis prepended with an axis of the same size as the Block axis, as though
they were stacked with [jax.numpy.stack][].

When you call the Stacked, it scans over the Block axis, applying the module to each element of the Block.

### Creating a Stacked

To create a Stacked, we provide `Stacked.init`, which takes a "Layers" [haliax.Axis][] and another Module as
well as args and kwargs for that module. The Layer is the axis that the Stacked will scan over, and the `args`
and `kwargs` are implicitly vmapped over the Layers.

For instance, to create a stack of GPT2 blocks, you might do:

```python
import jax.random

blocks = Stacked.init(Layers, Gpt2Block)(
    config,  # static configuration
    scale=hax.arange(Layers),  # dynamic configuration. Each layer gets a scalar scale value [0, 1, 2, ...]
    key=jax.random.split(key, Layers.size),  # dynamic configuration. Each layer gets a different key
)
```

Any NamedArray passed to the Stacked init will have its Layers axis (if present) vmapped over. Any
JAX array will have its first axis vmapped over.


### Fold Blocks vs Scan Blocks

The Stacked module provides two ways to apply the layers: `fold` and `scan`.  A fold is the moral equivalent of this for loop:

```python
for block in self.blocks:
    x = block(x)
```

while a scan is the moral equivalent of this for loop:

```python
out = []
for block in self.blocks:
    x, y = block(x)
    out.append(y)

return x, stack(out)
```

Blocks can be coded to either support fold or scan, but not both.
A "fold Block" should have the signature `def __call__(self, x: Carry) -> Carry`,
while a "scan Block" should have the signature `def __call__(self, x: Carry) -> Tuple[Carry, Output]`.

(See also [jax.lax.scan][], [haliax.fold][], and [haliax.scan][].)

### Requirements for Stacked Blocks

As we said above, the Stacked module requires that all the layers have the same shape and configuration.

A further constraint is that the elements of the stack must have the same Python control flow. This is the usual
constraint imposed on jit-compiled functions in JAX. All control flow must use `jax.lax` primitives like
[jax.lax.cond][], [jax.lax.while_loop][], and [jax.lax.scan][]. You can't use Python control flow like `if` or `for`
except for static control flow that is the same for all elements of the stack.

## BlockSeq and BlockFoldable

We also provide a way to create a sequence of layers that can be applied to a sequence of inputs that implements the
same interface as [haliax.nn.Stacked][], but with a different implementation. This is the [haliax.nn.BlockSeq][] module.
BlockSeq implements those for loops directly, rather than using [haliax.fold][] or [haliax.scan][].

[haliax.nn.scan.BlockFoldable][] is an interface that both [haliax.nn.Stacked][] and [haliax.nn.BlockSeq][] implement.

## Gradient Checkpointing

The [haliax.nn.Stacked][] module also provides a way to do gradient checkpointing, which can be useful for deep models.

Gradient checkpointing, aka rematerialization, is a technique for trading off memory usage for compute time.
Instead of storing all the intermediate activations of a model, you store only a subset and recompute the rest
as needed. (XLA automatically recomputes the rest for you as needed.)

[JAX's checkpointing mechanism]((https://docs.jax.dev/en/latest/gradient-checkpointing.html) is highly flexible,
and we provide a relatively simple interface to it for use with `Stacked`.

### Simple Checkpointing
In the simplest case, you can enable a usually-good-enough checkpointing policy by passing `gradient_checkpointing=True`
to the `Stacked.init` call:

```python
blocks = Stacked.init(Layers, TransformerBlock, gradient_checkpointing=True)(
    config,
    scale=hax.arange(Layers),
    key=jax.random.split(key, Layers.size),
)
```

This will preserve the intermediate "carries" and the "outputs" of the scans, while rematerializing (i.e. recomputing)
the rest of the computation as needed during backpropagation.

### Custom Checkpointing Policies

If you need more control over the checkpointing policy, you can pass a [haliax.nn.StackedCheckpointPolicy][] object to
the Stacked init:

```python
policy = StackedCheckpointPolicy(
   save_carries=True,  # default
   save_outputs=True,  # default
   save_intermediates=False,  # default
)
```

### Saving Block-Internal Values

"`intermediates`" refers to the internal computation of the block. If you set `save_intermediates=True`, then
all internals of every block will be saved. This can be expensive.

You can also pass a list of strings to `save_intermediates` to specify which intermediates to save.

You could, for instance, save the output of the attention layer using [haliax.tree_checkpoint_name][]:

```python
class TransformerBlock(eqx.Module):
    def __call__(self, x):
        y = self.attention(self.ln1(x))
        y = haliax.tree_checkpoint_name(y, "attn_out")
        x = x + y
        y = self.mlp(self.ln2(x))
        return x + y

policy = StackedCheckpointPolicy(save_carries=True, save_block_internals=["attn_out"])
```

With this policy, the output of the attention layer will be saved during the forward pass.

This will save an extra attention computation in the backward pass, adding $`O(N * Pos * Hidden)`$ memory usage,
which is double that required by the default policy.


### Offloading Checkpointed Values

Both `save_carries` and `save_outputs` can either be a boolean or the string "offload". If "offload", then the
checkpointed values will be offloaded to the host during the forward pass, and reloaded during the backward pass.


## API

::: haliax.nn.Stacked
::: haliax.nn.BlockSeq
::: haliax.nn.scan.BlockFoldable

::: haliax.nn.StackedCheckpointPolicy
