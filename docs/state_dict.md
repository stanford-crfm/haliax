# Serialization

Haliax supports serialization of modules (including any [equinox.Module][]) to and from PyTorch-compatible
state dicts using the [safetensors](https://github.com/huggingface/safetensors) library. For details on
how state dicts work in PyTorch, see the [PyTorch documentation](https://pytorch.org/docs/stable/notes/serialization.html).

A state dict is a Python dictionary that maps string keys to tensors. It is used to store the parameters
of a model (though typically not the model's structure or hyperparameters). The keys are typically the names of the
model's parameters, arranged as `.`-separated paths. For example, a model with a `conv1` layer might have a
state dict with keys like `conv1.weight` and `conv1.bias`. Sequences of modules (e.g., for lists of layers) are
serialize with keys like `layer.0.weight`, `layer.1.weight`, etc.


## Saving a state dict

To serialize a module to a Pytorch-compatible state dict, use the [haliax.state_dict.to_torch_compatible_state_dict][]
function. This function takes a module and returns a state dict. To save the state dict to a file, use the
[haliax.state_dict.save_state_dict][] function, which writes the state dict to a file in safetensor format.
`to_torch_compatible_state_dict` flattens [haliax.nn.Linear] module Input and Output axis specs to a format that
is compatible with PyTorch Linear modules (though `out_first=True` is necessary to match PyTorch's default).
