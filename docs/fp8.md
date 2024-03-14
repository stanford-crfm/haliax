# FP8 Training

It's actually very easy to use FP8 in Haliax. The FP8 in Haliax is currently designed with optimizing throughput
on FP8-enabled devices (currently H100). It's not designed to, e.g., quantize a model to FP8 for deployment,
though this shouldn't be that hard to add. (We would be happy to accept contributions to add this functionality,
and are happy to work with you to do so.)

## TL;DR

To enable FP8, do this:

```python
import haliax as hax
import haliax.quantization as haxq
# setup
module = haxq.fp8_linear_layers(module)

# if using optax. This saves a tiny amount of memory so you can skip it if you want
_, trainable_module = haxq.partition_for_grad_overwrite(module)
opt_state = opt.initial_state(trainable_module)

# train step
grads = eqx.filter_grad(loss_fn)(module, data)
overwrite, grads = haxq.partition_for_grad_overwrite(grads)
updates, opt_state = opt.update(grads, opt_state, params=module)  # or however you update your optimizer
module = haxq.apply_updates(module, updates, overwrite)
```

And train your model like normal.

## How to use FP8

FP8 is enabled by a transform on your model. Here's a simple example:


```python
import haliax as hax
import equinox as eqx
import jax

In = hax.Axis("In", 32)
Mid = hax.Axis("Mid", 128)
Out = hax.Axis("Out", 16)
Hidden = hax.Axis("Hidden", 64)


class MyModule(eqx.Module):
    up_proj: hax.nn.Linear
    down_proj: hax.nn.Linear

    @staticmethod
    def init(*, key):
        super().__init__()
        k_up, k_down = jax.random.split(key)
        return MyModule(
            up_proj=hax.nn.Linear.init(In, Mid, key=k_up),
            down_proj=hax.nn.Linear.init(Mid, Out, key=k_down),
        )

    def __call__(self, x):
        x = self.up_proj(x)
        x = hax.nn.relu(x)
        x = self.down_proj(x)
        return x

module = MyModule.init(key=jax.random.PRNGKey(0))

# Enable FP8
module = hax.quantization.fp8_linear_layers(module)

# Enable FP8 for a specific layer
from haliax.quantization import Fp8Config

config = Fp8Config(targets=["up_proj"])
module = hax.quantization.fp8_linear_layers(module, config)

# Train step
grads = eqx.filter_grad(loss_fn)(module, data)
overwrite, grads = haxq.partition_for_grad_overwrite(grads)
updates, opt_state = opt.update(grads, opt_state, params=module)  # or however you update your optimizer
module = hax.quantization.apply_updates(module, updates, grads)
```

That's it! Just a few lines of code to enabl e FP8. The `fp8_linear_layers` function will transform your module to use FP8
for linear layers (or a subset if you want), and the combo of `partition_for_grad_overwrite` and `apply_updates` function will apply the updates to the module
in a way that is compatible with FP8.

## How FP8 works

For an overview of the FP8, see the [FP8 Primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html).
You don't need to understand it though. Haliax's FP8 integration is more or less plug and play, as shown above.
The implementation of FP8 in Haliax is more or less a straightforward port (including some copy and paste) of the
[FP8 implementation in Flax](https://github.com/google/flax/blob/main/flax/linen/fp8_ops.py).

FP8 in JAX (as well as INT8) is typically implemented using "`dot_general` injection", where you pass
a custom implementation of `dot_general` to functions and modules like [haliax.dot][] and [haliax.nn.Linear][].
The `dot_general` for FP8 is implemented by scaling
the inputs, projecting the inputs to FP8, performing the computation in FP8, and then
scaling the result back to the original precision.
The subtle part of FP8 is that the scaling is a parameter that is trained based on a history of the inputs to the layer
(as well as gradients coming in from backward). This means that the FP8 `dot_general` needs to maintain state.
In Equinox, this means that the `dot_general` is actually a `Module` that packages together the state and the
computation. (Unlike [equinox.nn.StatefulLayer][] which returns a state object you pass back into the module, the FP8 `dot_general`
module hijacks the gradient computation to update its state. This is necessary because the FP8 scaling factors
depend on the gradients.)

The way this happens is by "hijacking" the gradient computation. When you call `eqx.filter_grad(loss_fn)(module, data)`,
you will get the gradient computation as normal, but you'll also get the updated state of the FP8 `dot_general` module.
This updated state needs to directly replace the state in the module (rather than be used for a gradient step), which is
why you need to use the `partition_for_grad_overwrite`

The FP8 `dot_general` module is implemented in [haliax.quantization.Fp8DotGeneral][]. It's actually not that complicated:

1) It holds a scaling factor and history of maximum values for each of (lhs, rhs, output) and updates them based on the
gradients.
2) When invoked, it scales the inputs, projects them to FP8, performs the computation, and scales the result back to the
original precision.  It remembers the maximum absolute value for each of the inputs.
3) For the gradients, it scales the gradients, projects them to FP8, does the backward computation,
and scales the gradients back to the original precision. It remembers the maximum absolute value for the incoming
gradient and stores it in the gradient.
