# Main API Reference
In general, the API is designed to be similar to JAX's version of NumPy's API, with the main difference being
that we use names (either strings or [haliax.Axis][] objects) to specify axes instead of integers. This shows up for creating
arrays (see [haliax.zeros][] and [haliax.ones][]) as well as things like reductions (see [haliax.sum][] and
[haliax.mean][]).

## Axis Types

There are four types related to [haliax.Axis][] you will see in the API reference:

* [haliax.Axis][]: This is the main type for representing axes. It is a dataclass with a name and a size.
* [haliax.AxisSelector][]: This is a type alias for either [haliax.Axis][] or a `str`. This type is used when we want
  one axis and the size can be inferred from the inputs.
* [haliax.AxisSpec][]: This is a type alias for either [haliax.Axis][] or a tuple/list of [haliax.Axis][]. This type is
  used when we want one or more axes and the sizes cannot be inferred from the inputs, for instance when creating arrays.
* [haliax.AxisSelection][]: This is a type alias for either [haliax.AxisSelector][] or a tuple/list of [haliax.AxisSelector][].
    This type is used when we want one or more axes and the sizes can be inferred from the inputs, for instance when
    reducing an array.

Occasionally, an axis size can be inferred in some circumstances but not others. When this happens, we still use
`AxisSelector` but document the behavior in the docstring. A RuntimeError will be raised if the size cannot be inferred.

::: haliax.Axis
::: haliax.AxisSelector
::: haliax.AxisSpec
::: haliax.AxisSelection

### Axis Manipulation

::: haliax.make_axes
::: haliax.axis.axis_name
::: haliax.axis.concat_axes
::: haliax.axis.union_axes
::: haliax.axis.intersect_axes
::: haliax.axis.eliminate_axes
::: haliax.axis.without_axes
::: haliax.axis.selects_axis
::: haliax.axis.is_axis_compatible


## Array Creation
::: haliax.named
::: haliax.zeros
::: haliax.ones
::: haliax.full
::: haliax.zeros_like
::: haliax.ones_like
::: haliax.full_like
::: haliax.arange
::: haliax.linspace
::: haliax.logspace
::: haliax.geomspace


### Combining Arrays


::: haliax.concatenate
::: haliax.stack

(We don't include `hstack` or `vstack` because they are subsumed by `stack`.)

## Array Manipulation

### Broadcasting

See also the section on [Broadcasting](broadcasting.md).

::: haliax.broadcast_axis
::: haliax.broadcast_to

### Slicing

See also the section on [Indexing and Slicing](indexing.md).

::: haliax.index
::: haliax.slice
::: haliax.take
::: haliax.updated_slice

#### Dynamic Slicing

::: haliax.dslice
::: haliax.dblock
::: haliax.ds

### Shape Manipulation

::: haliax.flatten
::: haliax.flatten_axes
::: haliax.rearrange
::: haliax.unbind
::: haliax.unflatten_axis
::: haliax.split
::: haliax.ravel


## Operations

[Binary](#binary-operations) and [unary](#unary-operations) operations are all more or less directly from JAX's NumPy API.
The only difference is they operate on named arrays instead.

### Matrix Multiplication

See also the page on [Matrix Multiplication](matmul.md) as well as the [cheat sheet section](cheatsheet.md#matrix-multiplication).

::: haliax.dot
::: haliax.einsum

## Reductions

Reduction operations are things like [haliax.sum][] and [haliax.mean][] that reduce an array along one or more axes.
Except for [haliax.argmin][] and [haliax.argmax][], they all have the form:

```python
def sum(x, axis: Optional[AxisSelection] = None, where: Optional[NamedArray] = None) -> haliax.NamedArray:
    ...
```

with the behavior closely following that of JAX's NumPy API. The `axis` argument can
be a single axis (or axis name), a tuple of axes, or `None` to reduce all axes. The `where` argument is a boolean array
that specifies which elements to include in the reduction. It must be broadcastable to the input array, using
Haliax's [broadcasting rules](broadcasting.md).

The result of a reduction operation is always [haliax.NamedArray][] with the reduced axes removed.
If you reduce all axes, the result is a NamedArray with 0 axes, i.e. a scalar.
You can convert it to a [jax.numpy.ndarray][] with [haliax.NamedArray.scalar][], or just [haliax.NamedArray.array][].

::: haliax.all
::: haliax.amax
::: haliax.amin
::: haliax.any
::: haliax.argmax
::: haliax.argmin
::: haliax.max
::: haliax.mean
::: haliax.min
::: haliax.nanargmax
::: haliax.nanargmin
::: haliax.nanmax
::: haliax.nanmean
::: haliax.nanmin
::: haliax.nanprod
::: haliax.nanstd
::: haliax.nansum
::: haliax.nanvar
::: haliax.prod
::: haliax.ptp
::: haliax.std
::: haliax.sum
::: haliax.var

### Axis-wise Operations
Axis-wise operations are things like [haliax.cumsum][] and [haliax.sort][] that operate on a single axis of an array but
don't reduce it.

::: haliax.cumsum
::: haliax.cumprod
::: haliax.nancumprod
::: haliax.nancumsum
::: haliax.sort
::: haliax.argsort

### Unary Operations

The `A` in these operations means [haliax.NamedArray][], a `Scalar`, or [jax.numpy.ndarray][].
These are all more or less directly from JAX's NumPy API.

::: haliax.abs
::: haliax.absolute
::: haliax.angle
::: haliax.arccos
::: haliax.arccosh
::: haliax.arcsin
::: haliax.arcsinh
::: haliax.arctan
::: haliax.arctanh
::: haliax.around
::: haliax.bitwise_count
::: haliax.bitwise_invert
::: haliax.bitwise_not
::: haliax.cbrt
::: haliax.ceil
::: haliax.conj
::: haliax.conjugate
::: haliax.copy
::: haliax.cos
::: haliax.cosh
::: haliax.deg2rad
::: haliax.degrees
::: haliax.exp
::: haliax.exp2
::: haliax.expm1
::: haliax.fabs
::: haliax.fix
::: haliax.floor
::: haliax.frexp
::: haliax.i0
::: haliax.imag
::: haliax.invert
::: haliax.iscomplex
::: haliax.isfinite
::: haliax.isinf
::: haliax.isnan
::: haliax.isneginf
::: haliax.isposinf
::: haliax.isreal
::: haliax.log
::: haliax.log10
::: haliax.log1p
::: haliax.log2
::: haliax.logical_not
::: haliax.ndim
::: haliax.negative
::: haliax.positive
::: haliax.rad2deg
::: haliax.radians
::: haliax.real
::: haliax.reciprocal
::: haliax.rint
::: haliax.round
::: haliax.rsqrt
::: haliax.sign
::: haliax.signbit
::: haliax.sin
::: haliax.sinc
::: haliax.sinh
::: haliax.square
::: haliax.sqrt
::: haliax.tan
::: haliax.tanh
::: haliax.trunc

### Binary Operations
::: haliax.add
::: haliax.arctan2
::: haliax.bitwise_and
::: haliax.bitwise_left_shift
::: haliax.bitwise_or
::: haliax.bitwise_right_shift
::: haliax.bitwise_xor
::: haliax.divide
::: haliax.divmod
::: haliax.equal
::: haliax.float_power
::: haliax.floor_divide
::: haliax.fmax
::: haliax.fmin
::: haliax.fmod
::: haliax.greater
::: haliax.greater_equal
::: haliax.hypot
::: haliax.left_shift
::: haliax.less
::: haliax.less_equal
::: haliax.logaddexp
::: haliax.logaddexp2
::: haliax.logical_and
::: haliax.logical_or
::: haliax.logical_xor
::: haliax.maximum
::: haliax.minimum
::: haliax.mod
::: haliax.multiply
::: haliax.nextafter
::: haliax.not_equal
::: haliax.power
::: haliax.remainder
::: haliax.right_shift
::: haliax.subtract
::: haliax.true_divide

### Other Operations

::: haliax.bincount
::: haliax.clip
::: haliax.packbits
::: haliax.unpackbits
::: haliax.isclose
::: haliax.allclose
::: haliax.array_equal
::: haliax.array_equiv
::: haliax.pad
::: haliax.searchsorted
::: haliax.top_k
::: haliax.trace
::: haliax.tril
::: haliax.triu
::: haliax.where



## Named Array Reference

Most methods on [haliax.NamedArray][] just call the corresponding `haliax` function with the array as the first argument,
just as with Numpy.
The exceptions are documented here:

::: haliax.NamedArray


## Partitioning API

See also the section on [Partitioning](partitioning.md).

::: haliax.partitioning.axis_mapping
::: haliax.shard
::: haliax.named_jit
::: haliax.fsdp
::: haliax.partitioning.round_axis_for_partitioning
::: haliax.partitioning.physical_axis_name
::: haliax.partitioning.physical_axis_size
::: haliax.partitioning.sharding_for_axis


## Gradient Checkpointing

Haliax mainly just defers to JAX and [equinox.filter_checkpoint][] for gradient checkpointing. However,
we provide a few utilities to make it easier to use.

See also [haliax.nn.ScanCheckpointPolicy][].

::: haliax.tree_checkpoint_name

### Old API

These functions are being deprecated and will be removed in a future release.

::: haliax.shard_with_axis_mapping
::: haliax.auto_sharded
