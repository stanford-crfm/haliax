# State Dicts and Serialization

Haliax follows the Equinox convention of mostly working in terms of module trees, similar to PyTorch. However,
sometimes we would rather the weights all be in one easy to control dictionary.
For example, when saving a model to disk, you typically want to save the weights and biases in a single file (especially
for compatibility with PyTorch and other ecosystems).
Similarly, someone coming from Flax might be more comfortable working with a parameter dictionary rather
than a bunch of PyTree-wrangling.

This is where state dicts come in.

A state dict is a Python dictionary that maps string keys to tensors. It is used to store the parameters
of a model (though typically not the model's structure or hyperparameters). The keys are typically the names of the
model's parameters, arranged as `.`-separated paths. For example, a model with a `conv1` layer might have a
state dict with keys like `conv1.weight` and `conv1.bias`. Sequences of modules (e.g., for lists of layers) are
serialized with keys like `layer.0.weight`, `layer.1.weight`, etc.

The values in a Haliax state dict are typically JAX or numpy arrays, and not NamedArrays.

## Basic State Dicts

To create a state dict from a module, use the [haliax.state_dict.to_state_dict][] function. This function takes a module
and returns a state dict:

```python
import haliax as hax
import jax.random as jrandom

# Create a module
Heads = hax.Axis("Heads", 8)
Dim = hax.Axis("Dim", 16)
Out = hax.Axis("Out", 5)

module = hax.nn.Linear.init(In=(Heads, Dim), Out=Out, key=jrandom.PRNGKey(0))

# Serialize the module to a state dict
state_dict = hax.state_dict.to_state_dict(module)
```

You can manipulate the state dict as you would any other Python dictionary. Note that the arrays are JAX arrays, not
NamedArrays or Numpy arrays. In particular, with `to_state_dict`, the arrays still preserve any sharding or vmapping.
This makes it a great choice for using inside JIT.

If you want a CPU-only state dict, you can use the [haliax.state_dict.to_numpy_state_dict][] function:

```python
# Serialize the module to a state dict

state_dict = hax.state_dict.to_numpy_state_dict(module)
```

To load a state dict back into a module, use the [haliax.state_dict.from_state_dict][] function. This function
requires a "template" module that has the same structure as the module that was serialized to the state dict:

```python
# Load the state dict into a module
module = hax.state_dict.from_state_dict(module, state_dict)
```

One trick is that you can use the `init` method of a module inside of [equinox.filter_eval_shape][] to create
an abstract version of the module that can be used as a template for loading the state dict. This is useful if you
want to avoid allocating a bunch of big arrays just to load the state dict.

```python
import equinox as eqx

module_template = eqx.filter_eval_shape(Linear.init, In=(Heads, Dim), Out=Out)
module = hax.state_dict.from_state_dict(module_template, state_dict)
```

### Saving a State Dict

!!! warning
    The default Haliax state dicts are not in general compatible with PyTorch. If you want to load a Haliax state dict
    you will need to convert it to a PyTorch-compatible state dict first and use the
    [safetensors](https://github.com/huggingface/safetensors) library to load it into PyTorch.
    See the [PyTorch-Compatible State Dicts](#pytorch-compatible-state-dicts) section for how to do this.

To save the state dict to a file, use the [haliax.state_dict.save_state_dict][] function together with the
[haliax.state_dict.to_numpy_state_dict][] function:

```python
# Save the state dict to a file
hax.state_dict.save_state_dict(state_dict, 'state_dict.safetensors')
```

Likewise, you can load a state dict from a file using the [haliax.state_dict.load_state_dict][] function:

```python
# Load the state dict from a file
state_dict = hax.state_dict.load_state_dict('state_dict.safetensors')
```

#### Things to know

Haliax supports serialization of modules (including any [equinox.Module][]) to and from PyTorch-compatible
state dicts using the [safetensors](https://github.com/huggingface/safetensors) library. For details on
how state dicts work in PyTorch, see the [PyTorch documentation](https://pytorch.org/docs/stable/notes/serialization.html#saving-and-loading-torch-nn-modules). (Levanter has JAX-native TensorStore based
serialization that we should upstream here.)

Haliax uses the [safetensors](https://github.com/huggingface/safetensors) library to serialize state dicts. This
library is a safer, more portable format developed by Hugging Face. (Serializing a native PyTorch state dict requires
PyTorch itself, and we want to avoid that dependency. Also, PyTorch uses pickles, which are in general not
safe to deserialize from untrusted sources.)

This does mean that you can't directly load a Haliax state dict into PyTorch, but safetensors is lightweight and
easy to use. Hugging Face natively supports it in their libraries.
(See the [PyTorch-Compatible State Dicts](#pytorch-compatible-state-dicts) section for more details
on how to convert a Haliax state dict to a PyTorch-compatible state dict.)


## Pytorch-Compatible State Dicts

Haliax provides a way to serialize a module to a PyTorch-compatible state dict. This is useful if you want to
load the weights of a Haliax module into a PyTorch model or vice versa. The [haliax.state_dict.to_torch_compatible_state_dict][]
and [haliax.state_dict.from_torch_compatible_state_dict][] functions allow you to convert a Haliax state dict to and from
a PyTorch-compatible state dict.

Note that these methods behave a bit differently from the basic state dict methods. See
the [Explanation](#explanation) section for more details.

### Saving a State Dict

To serialize a module to a Pytorch-compatible state dict, use the [haliax.state_dict.to_torch_compatible_state_dict][]
function. This function takes a module and returns a state dict. To save the state dict to a file, use the
[haliax.state_dict.save_state_dict][] function, which writes the state dict to a file in safetensor format.
`to_torch_compatible_state_dict` flattens [haliax.nn.Linear] module input and output axis specs to a format that
is compatible with PyTorch Linear modules (though `out_first=True` is necessary to match PyTorch's Linear module).

```python
import haliax
import jax.random as jrandom

# Create a module
Heads = haliax.Axis("Heads", 8)
Dim = haliax.Axis("Dim", 16)
Out = haliax.Axis("Out", 5)
module = haliax.nn.Linear.init(In=(Heads, Dim), Out=Out, key=jrandom.PRNGKey(0))

# Serialize the module to a state dict
state_dict = haliax.state_dict.to_torch_compatible_state_dict(module)

# Save the state dict to a file
haliax.state_dict.save_state_dict(state_dict, 'state_dict.safetensors')
```

Note that the state dict is saved in the [safetensors](https://github.com/huggingface/safetensors) format, which
is a safer, more portable format developed by Hugging Face. To load a model from a state dict in PyTorch, you
can use safetensors directly.

```python
import torch
from safetensors.torch import load_model

model = torch.nn.Linear(10, 5)

# Load the state dict from a file
state_dict = load_model(model, 'state_dict.safetensors')
```

### Loading a State Dict

Similarly, you can load a state dict from a file using the [haliax.state_dict.load_state_dict][] function. This
function reads a state dict from a file in safetensors format and returns a dictionary. To load the state dict
into a module, use the [haliax.state_dict.from_torch_compatible_state_dict][] function.

```python
import haliax.state_dict
import haliax as hax
import jax.random as jrandom

# Create a module
Heads = hax.Axis("Heads", 8)
Dim = hax.Axis("Dim", 16)
Out = hax.Axis("Out", 5)
module = hax.nn.Linear.init(In=(Heads, Dim), Out=Out, key=jrandom.PRNGKey(0))

# Load the state dict from a file
state_dict = hax.state_dict.load_state_dict('state_dict.safetensors')

# this will unflatten the state dict and load it into the module
module = haliax.state_dict.from_torch_compatible_state_dict(module, state_dict)
```

The `from_torch_compatible_state_dict` function will prepare the state dict and load it into the module. Note
that the module must have the same structure as the module that was serialized to the state dict. If the module
structure has changed, you may need to manually update the state dict keys to match the new structure.

### Explanation

By default, Haliax creates a state dict using key paths and arrays that mirror the module's structure. For example, a
Linear module with an `In` axis spec of `(Head, HeadDim)` and an `Out` axis spec of `Embed` will have a state dict
with a key `weight` that is a three-dimensional tensor with shape `(Embed, Head, HeadDim)`. Moreover,
instances of [haliax.nn.Stacked][] (i.e. our "scan-layers" module) will have a state dict with the vmapped
layers as a single module:

```python
import haliax as hax
import haliax.nn as hnn
import jax.random as jrandom

Heads = hax.Axis("Heads", 8)
Dim = hax.Axis("Dim", 16)
Out = hax.Axis("Out", 5)
Block = hax.Axis("Block", 3)

keys = jrandom.split(jrandom.PRNGKey(0), Block.size)

stacked_module = hnn.Stacked.init(Block, hnn.Linear)(In=(Heads, Dim), Out=Out, key=keys)

state_dict = hax.state_dict.to_state_dict(stacked_module)

for k, v in state_dict.items():
    print(k, v.shape)

# Output:
# weight (3, 5, 8, 16)
# bias (3, 5)
```

PyTorch expects the weights of a linear layer to be a 2D tensor with shape `(out_features, in_features)` and the bias
to be a 1D tensor with shape `(out_features,)`. Moreover, it expects the layers of a stacked module to be unstacked
into a `torch.nn.Sequential`. To do this, we use the [haliax.state_dict.to_torch_compatible_state_dict][] function:

```python
torch_state_dict = hax.state_dict.to_torch_compatible_state_dict(stacked_module)

for k, v in torch_state_dict.items():
    print(k, v.shape)

# Output:
# bias (3, 5)
# 0.weight (5, 128)
# 0.bias (5,)
# 1.weight (5, 128)
# 1.bias (5,)
# 2.weight (5, 128)
# 2.bias (5,)

# save it
hax.state_dict.save_state_dict(torch_state_dict, 'torch_state_dict.safetensors')

# load it
import torch
import safetensors.torch as st

model = torch.nn.Sequential(
    torch.nn.Linear(128, 5),
    torch.nn.Linear(128, 5),
    torch.nn.Linear(128, 5)
)

state_dict = st.load_model(model, 'torch_state_dict.safetensors')
```

## Customizing Serialization

### Changing the State Dict Key Names

If for some reason you want to use different names in the serialized state dict (e.g. because you
chose to use different names from a Hugging Face implementation), you can extend your class from  [haliax.state_dict.ModuleWithStateDictSerialization][]
and use `_state_dict_key_map` to rename keys. For instance, the `Gpt2Transformer` class in Levanter has this method:

```python
from typing import Optional
from haliax.state_dict import ModuleWithStateDictSerialization

class Gpt2Transformer(ModuleWithStateDictSerialization):
    ...

    def _state_dict_key_map(self) -> dict[str, Optional[str]]:
        return {"blocks": "h"}
```

This says that the field called `blocks` in this class should be (de)serialized as `h`,
because the Hugging Face GPT-2 implementation uses `h`, which is not very clear.
You can also "flatten" the submodules of a field by using `None`.

### Custom Serialization Logic

If your modules need fancier special logic, you'll need to extend your class from `ModuleWithStateDictSerialization` and
override the default functions `to_state_dict()` and `from_state_dict()`. It takes in and returns a modified
[haliax.state_dict.StateDict][]. As of June 2024, we almost never this in Levanter.

For implementation, there are a few helper methods from `haliax.state_dict` that you can use:
- To join specific prefix to the keys of Hugging Face state_dict, you can use the helper function `with_prefix()`.
  The prefix comes from the name of attributes defined at the beginning of your model class.

For example, below is the implementation of `to_state_dict()` in [levanter.models.backpack.BackpackLMHeadModel][].
In this class, we want to preserve HF compatibility by saving untied output embeddings. (We chose not to implement
non-weight-tied embeddings.)

```python
from typing import Optional

from haliax.state_dict import with_prefix, StateDict


class BackpackLMHeadModel(ModuleWithStateDictSerialization):
    ...

    def to_state_dict(self, prefix: Optional[str] = None) -> StateDict:
        state_dict = super().to_state_dict(prefix=prefix)
        # In levanter's implementation, we have a shared embedding matrix for both the word
        # embeddings and the sense embeddings
        state_dict[with_prefix(prefix, "backpack.word_embeddings.weight")] = state_dict[
            with_prefix(prefix, "backpack.gpt2_model.wte.weight")
        ]
        state_dict[with_prefix(prefix, "backpack.position_embeddings.weight")] = state_dict[
            with_prefix(prefix, "backpack.gpt2_model.wpe.weight")
        ]
        return state_dict
```

Similarly, to load weights from the state dict, you might need to implement `from_state_dict`. This function
takes in a state dict and the module with the updated weights. You can use the `with_prefix()` helper function
to join the prefix to the keys of the state dict.

```python
    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> T:
      ...
```

## API Reference

::: haliax.state_dict
