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










### Reference

::: haliax.DTypeish
::: haliax.SemanticDType
::: haliax.current_mp_policy

::: haliax.mixed_precision.cast_floating


::: haliax.ResourceEnv
::: haliax.resource_env
::: haliax.current_resource_env
