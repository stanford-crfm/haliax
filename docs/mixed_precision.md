# Mixed Precision in Haliax

## JMP

Haliax's mixed precision is currently built on [JMP], which is a simple library for mixed precision training.
JMP's principal class is [jmp.Policy][], which holds three dtypes:

* `param_dtype`: The dtype of the model parameters.
* `compute_dtype`: The dtype that computations are performed in.
* `output_dtype`: The dtype of the model outputs, typically used for loss functions or other more "numerically-unstable" operations.

Policies are typically represented as a string like `"p=f32,c=bf16,o=f32"`, which means that the parameters are stored in `f32`, computations are performed in `bf16`, and outputs are in `f32`.""

Once you have a policy, you can convert an arbitrary PyTree between dtypes using [jmp.Policy.cast_to_param] etc.

```python
import jmp
import haliax as hax
import jax.numpy as jnp

policy = jmp.get_policy("p=f32,c=bf16,o=f32")

D = hax.Axis("D", 16)
x = hax.arange(D, dtype=float)

assert policy.cast_to_compute(x).dtype == jnp.bfloat16
assert policy.cast_to_output(x).dtype == jnp.float32
assert policy.cast_to_param(x).dtype == jnp.float32
```

### Scaling

JMP also has support for scaling the loss when using FP16, but we don't typically use that ourselves, preferring
to stick with `bfloat16`.

## `SemanticDType` and `DTypeish`

Haliax extends this core idea from jmp.Policy to add an explicit [haliax.SemanticDType][] enum,
with three entries: `"compute"`, `"param"`, and `"output"`. Instances of this enum can be
resolved to a specific `dtype` using the `to_dtype` method and a [jmp.Policy][].
In addition, you can convert all floating point arrays in a PyTree using [haliax.mixed_precision.cast_floating][]:

```python
import haliax.mixed_precision as hmp

assert hmp.cast_floating(x, "compute", policy).dtype == jnp.bfloat16
assert hmp.cast_floating(x, "param", policy).dtype == jnp.float32
```

`cast_floating` actually accepts a [haliax.DTypeish][] for the second parameter,
which is a union of "real" dtypes (like `jnp.bfloat16`), SemanticDtypes, and strings like `"compute"`.
This is useful for writing generic code that can accept either a dtype or a SemanticDType.

The `policy` argument is optional, and if not provided, Haliax will use the global [haliax.ResourceEnv][] to
determine the current policy. See the next section for more details.


## haliax.ResourceEnv

Haliax uses a global (technically, thread-local) context manager called [haliax.ResourceEnv][] to manage
both [partitioning](partitioning.md) and mixed precision. For mixed precision, the `ResourceEnv` holds a
`jmp.Policy` that can be accessed via the `policy` attribute or via the [haliax.current_mp_policy][] function:

```python
import haliax as hax
import jax.numpy as jnp

with hax.resource_env(mp="p=f32,c=bf16,o=f32"):
    assert hmp.cast_floating(x, "compute").dtype == jnp.bfloat16
    assert hmp.cast_floating(x, "param").dtype == jnp.float32

# The default env is fp32
assert hmp.cast_floating(x, "compute").dtype == jnp.float32
assert hmp.cast_floating(x, "param").dtype == jnp.float32
```



## NN Modules

Many Haliax modules, including [haliax.nn.Linear][], [haliax.nn.LayerNorm][], and [haliax.nn.Conv][] accept
an optional `compute_dtype` argument. This argument defaults to `"compute"`, but can be set to `"param"` or
`"output"` or a specific dtype to override the global policy.

```python
import haliax as hax
import jax.numpy as jnp
import jax.random as jrandom

In = hax.Axis("In", 16)
Out = hax.Axis("Out", 32)

linear = hax.nn.Linear.init(In, Out, key=jrandom.PRNGKey(0))
assert linear.weight.dtype == jnp.float32
assert linear.bias.dtype == jnp.float32
input = hax.arange(In, dtype=jnp.bfloat16)
out = linear(input)
assert out.dtype == jnp.float32

with hax.resource_env(mp="p=f32,c=bf16,o=f32"):
    out = linear(input)
    assert out.dtype == jnp.bfloat16



```

## Loss Functions

XXX TODO

## Future: Quantization and FP8

This is not at all implemented yet, but the plan is to add support for quantization and FP8 in the future.

This section is not going to talk about how specific quantization schemes work, but rather how
structurally they are implemented in the JAX ecosystem.

For purposes of this discussion, I'm going to treat quantization and FP8 as the same thing, since they
end up requiring basically the same infrastructure.

### Quantized Training Overview

Most of this section is put together by my digging through [this blog post on AQT from Google](https://cloud.google.com/blog/products/compute/accurate-quantized-training-aqt-for-tpu-v5e/)
as well as the library itself. There's no paper yet, but I'm sure there will be one soon.
I don't pretend to understand all of it, particularly not the math parts, which aren't really
described in the blog post.

Let's talk just a bit about how quantization works. The basic idea is that you have
an array of floating point values, and you want to convert them to some low-precision
integer representation. The most common example is converting from FP32 to INT8, but there's also
FP8, INT4, etc.

Typically, you don't just project to the nearest representable value, at least not when doing training. Instead,
you want to scale the entire array so that you get as much high-resolution coverage as possible. For example,


### Quantization in JAX

There are two relevant libraries I'm basing my understanding of quantization on:

* [TransformerEngine](https://github.com/NVIDIA/TransformerEngine), which is NVIDIA's library for
  accelerated training of Transformers, including FP8 support.
* [AQT](https://github.com/google/aqt/), which is Google's library for quantization-aware training that
focuses on integer-based quantization (like int8 and int4).


The way quantization is shaping up to work in JAX is a combination of two mechanisms: "dot injection" and
what I'm going to call "grad hijacking."

#### Dot Injection

The first piece is dot injection, which is a mechanism for injecting alternative versions
of the  [jax.lax.dot_general][] primitive into higher level calls like [jax.numpy.einsum][].
(`einsum` is actually the only function in JAX that takes this argument, but it's seen in
[libraries like FLAX](https://github.com/google/flax/blob/61ece402d1b805e5ce797caf74b69ed8a7ae21ce/flax/linen/linear.py#L116-L117).)

This part is fairly intuitive: you are likely going to want custom logic for how to do
matrix multiplication in a quantized setting, and dot injection lets you do that.

#### Grad Hijacking

TODO: I am shocked but it seems like AQT doesn't need to do this? they just do the dumb thing, maybe
with stochastic rounding?

By itself,












### API Reference

::: haliax.DTypeish
::: haliax.SemanticDType
::: haliax.current_mp_policy

::: haliax.mixed_precision.cast_floating


::: haliax.ResourceEnv
::: haliax.resource_env
::: haliax.current_resource_env
