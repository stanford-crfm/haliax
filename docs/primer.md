# Haliax Primer

Haliax provides named tensors built on top of JAX.  This primer is written for LLM agents and other downstream libraries and collects the core ideas for quick reference.

## Axes and Named Arrays

Arrays are indexed by `Axis` objects. You can define them explicitly or generate several with `make_axes`.
You may also specify shapes with a **shape dict**, mapping axis names to sizes.

```python
import haliax as hax
from haliax import Axis

Batch = Axis("batch", 4)
Feature = Axis("feature", 8)
# or: Batch, Feature = hax.make_axes(batch=4, feature=8)
# using Axis objects
x = hax.zeros((Batch, Feature))
# or using a shape dict
shape = {"batch": 4, "feature": 8}
x = hax.zeros(shape)
```

Most functions accept either axes or shape dicts interchangeably.

A tensor with named axes is a [`NamedArray`][haliax.NamedArray]. Elementwise operations mirror `jax.numpy` but accept named axes.

## Indexing and Broadcasting

Use axis names when slicing. Dictionaries are convenient for several axes:

```python
first = x["batch", 0]
sub = x["batch", 1:3]
# or with a dict
first = x[{"batch": 0}]
sub = x[{"batch": slice(1, 3)}]
```

Axes broadcast by matching names. `broadcast_axis` adds a new axis to an array:

```python
row = hax.arange(Feature)
outer = row.broadcast_axis(Batch) * hax.arange(Batch)
```

See [Indexing and Slicing](indexing.md) and [Broadcasting](broadcasting.md) for details.

## Rearranging Axes

`rearrange` changes axis order and can merge or split axes using einops‑style syntax.  It is useful when interfacing with positional APIs.

```python
# transpose features and batch
x_t = hax.rearrange(x, "batch feature -> feature batch")
```

More examples appear in [Rearrange](rearrange.md).

## Matrix Multiplication

`dot` contracts over named axes while preserving order independence.

```python
Weight = Axis("weight", 8)
w = hax.ones((Feature, Weight))
prod = hax.dot(x, w, axis=Feature)
```

For more complex contractions use [`einsum`][haliax.einsum].  See [Matrix Multiplication](matmul.md).

## Scans and Folds

Use [`scan`][haliax.scan] or [`fold`][haliax.fold] to apply a function along an axis with optional gradient checkpointing.

```python
Time = Axis("time", 10)
sequence = hax.ones((Time, Feature))

def add(prev, cur):
    return prev + cur

result = hax.fold(add, Time)(hax.zeros((Feature,)), sequence)
```

See [Scan and Fold](scan.md) for checkpointing policies and stacked modules.

## Partitioning

Arrays and modules can be distributed across devices by mapping named axes to mesh axes:

```python
with hax.axis_mapping({"batch": "data"}):
    sharded = hax.shard(x)
```

The [Partitioning](partitioning.md) guide explains how to set up device meshes and shard arrays.

## Typing Support

Type annotations use `haliax.haxtyping` which extends `jaxtyping`:

```python
import haliax.haxtyping as ht

def f(t: ht.Float[hax.NamedArray, "batch feature"]):
    ...
```

See [Typing](typing.md) for matching runtime checks and dtype-aware annotations.

---

This primer highlights common patterns.  The [cheatsheet](cheatsheet.md) lists many additional conversions from JAX to Haliax.
