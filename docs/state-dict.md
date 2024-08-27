# Serialization

Haliax supports serialization of modules (including any [equinox.Module][]) to and from PyTorch-compatible
state dicts using the [safetensors](https://github.com/huggingface/safetensors) library. For details on
how state dicts work in PyTorch, see the [PyTorch documentation](https://pytorch.org/docs/stable/notes/serialization.html#saving-and-loading-torch-nn-modules).

A state dict is a Python dictionary that maps string keys to tensors. It is used to store the parameters
of a model (though typically not the model's structure or hyperparameters). The keys are typically the names of the
model's parameters, arranged as `.`-separated paths. For example, a model with a `conv1` layer might have a
state dict with keys like `conv1.weight` and `conv1.bias`. Sequences of modules (e.g., for lists of layers) are
serialize with keys like `layer.0.weight`, `layer.1.weight`, etc.

Haliax uses the [safetensors](https://github.com/huggingface/safetensors) library to serialize state dicts. This
library is a safer, more portable format developed by Hugging Face. Serializing a native PyTorch state dict requires
PyTorch itself, and we want to avoid that dependency. Also, PyTorch uses pickles, which are in general not
safe to deserialize from untrusted sources.

This does mean that you can't directly load a Haliax state dict into PyTorch, but safetensors is lightweight and
easy to use. Hugging Face natively supports it in their libraries.

## Saving a State Dict

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

## Loading a State Dict

Similarly, you can load a state dict from a file using the [haliax.state_dict.load_state_dict][] function. This
function reads a state dict from a file in safetensors format and returns a dictionary. To load the state dict
into a module, use the [haliax.state_dict.from_torch_compatible_state_dict][] function.

```python
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
module = hax.state_dict.from_torch_compatible_state_dict(module, state_dict)
```

The `from_torch_compatible_state_dict` function will unflatten the state dict and load it into the module. Note
that the module must have the same structure as the module that was serialized to the state dict. If the module
structure has changed, you may need to manually update the state dict keys to match the new structure.


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
