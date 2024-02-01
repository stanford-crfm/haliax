# ResourceEnvs

Haliax currently uses three bits of (optional!) global state:

1. the [jax.sharding.Mesh][] of devices;
2. a [haliax.partitioning.ResourceMapping][] that maps [haliax.Axis][]s to [jax.sharding.Axis][]s (See [Partitioning](partitioning.md));
3. and a `jmp.Policy` that controls [mixed precision](mixed-precision.md) behavior.

(1) and (2) are discussed in [Partitioning](partitioning.md) and (3) is discussed in [Mixed Precision](mixed-precision.md).

Haliax stores these three pieces of state in a [haliax.ResourceEnv][] object, which can be used either as a context
manager (like `Mesh`) or can be passed explicitly to functions that need it.

## Using `ResourceEnv`s

### As a context manager

You can use a `ResourceEnv` as a context manager to temporarily set the global state.

```python
import haliax as hax
import jax
import jmp

from jax.sharding import Mesh

mesh = Mesh(jax.devices(), ("dp"))
resource_mapping = {"embed": "dp"}
mp = jmp.get_policy("p=f32,c=bf16")

with hax.resource_env(resource_mapping, mp, mesh):
    # code that uses the resource env
```

### Explicitly passing a `ResourceEnv`

You can also pass a `ResourceEnv` explicitly to many functions that use one.
This is useful if you want to avoid using global state.

```python

import haliax as hax
import jax

from jax.sharding import Mesh

mesh = Mesh(jax.devices(), ("dp"))
resource_mapping = {"embed": "dp"}
mp = jmp.get_policy("p=f32,c=bf16")

env = hax.ResourceEnv(resource_mapping, mp, mesh)

# code that uses the resource env

H = hax.Axis("H", 128)
Embed = hax.Axis("embed", 128)


x = hax.shard(hax.zeros((H, Embed)), env)
```

#### Functions that can take a `ResourceEnv`

This is not an exhaustive list, but here are some functions that can take a `ResourceEnv`
as an explicit argument. Most of these will use the context `ResourceEnv` if one is not provided.
These are all sharding functions.

- [haliax.shard][]
- [haliax.named_jit][]
- [haliax.partitioning.physical_axis_name][]
- [haliax.partitioning.physical_axis_size][]
- [haliax.partitioning.sharding_for_axis][]




## Reference

::: haliax.ResourceEnv

::: haliax.resource_env

::: haliax.current_resource_env
