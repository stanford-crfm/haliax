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

::: haliax.axis.axis_name
::: haliax.axis.concat_axes
::: haliax.axis.union_axes
::: haliax.axis.eliminate_axes
::: haliax.axis.without_axes
::: haliax.axis.overlapping_axes
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

We don't include `hstack` or `vstack` because we prefer semantic axes.

::: haliax.concatenate
::: haliax.stack


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

## Matrix Multiplication

Haliax has two ways to do matrix multiplication (and tensor contractions more generally):
[haliax.dot][] and [haliax.einsum][]. [haliax.dot][] and [haliax.einsum][]
can both express any tensor contraction, though in different situations one or the other may be
more suitable for expressing a particular contraction.


### `haliax.dot`

With [haliax.dot][], you specify the axes to contract over, without needing to write out the
axes you want to keep (though you can if you want):

```python
import haliax as hax

H = hax.Axis("H", 3)
W = hax.Axis("W", 4)
D = hax.Axis("D", 5)

x = hax.ones((H, W, D))
w = hax.ones((D,))
y = hax.dot(x, w, axis=D)  # shape is (H, W), equivalent to np.einsum("hwd,d->hw", x, w)
```

[haliax.dot][] is at its best when you want to express a simple matrix multiplication over one or a few axes.
Syntactically, [haliax.dot][] is similar to reduction operations like [haliax.sum][] and [haliax.mean][].

The [cheat sheet](cheatsheet.md) has a section on [matrix multiplication](cheatsheet.md#matrix-multiplication)
that gives a few examples. Here are several more:

```python
import haliax as hax

H = hax.Axis("H", 3)
W = hax.Axis("W", 4)
D = hax.Axis("D", 5)
C = hax.Axis("C", 6)

x = hax.arange((H, W, D, C))
w = hax.arange((D, C))
c = hax.arange((C,))

y = hax.dot(x, c, axis=C) # shape is (H, W, D), equivalent to jnp.dot(x, c)

y = hax.dot(x, w, axis=(D, C))  # shape is (H, W), equivalent to np.einsum("...dc,dc->...", x, w)
y = hax.dot(x, w, axis=(D, C), out_axes=(W, H)) # shape is (W, H) instead of (H, W)
y = hax.dot(x, w, c, axis=(D, C)) # shape is (H, W), equivalent to np.einsum("...dc,dc,c->...", x, w, c)
y = hax.dot(x, c, axis=(H, D, C)) # shape is (W,), equivalent to np.einsum("hwdc,c->w", x, c)
s = hax.dot(x, w, axis=None)  # scalar output, equivalent to np.einsum("hwdc,dc->", x, w)
y = hax.dot(x, w, c, axis=())  # shape is (H, W, D, C), equivalent to np.einsum("hwdc,dc,c->hwdc", x, w, c)
y = hax.dot(x, w, c, axis=(), out_axes=(D, ..., H))  # shape is (D, W, C, H), equivalent to np.einsum("hwdc,dc,c->dwch", x, w, c)
```

### `haliax.einsum`

[haliax.einsum][] is at its best when you want to express a more complex tensor contraction.
It is similar to [numpy.einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html)
or [einops.einsum](https://einops.rocks/api/einsum/) in terms of syntax and behavior,
but extended to work with named axes, including added flexibility that named axes provide.
Our "flavor" of `einsum` is most similar to `einops.einsum`'s flavor, in that
it supports long names for axes (like `"batch h w, h w channel -> batch channel"`)
rather than the compact notation of `numpy.einsum` (like `"bhwc,hwc->bc"`).

Haliax's version of `einsum` comes in three modes: "ordered", "unordered", and "output axes".
These modes are all accessible through the same function without any flags: the syntax
of the `einsum` string determines which mode is used.

#### Ordered Mode

Haliax's `einsum` has an "ordered" mode that is similar to `einops.einsum`'s behavior.
In this mode, the axes in the input arrays are matched to the axes in the `einsum` string in order.
It supports ellipses in the same way as `einops.einsum`. The names in the einsum string need not
match the names of the axes in the input arrays, but the order of the axes must match.

```python
import haliax as hax

H = hax.Axis("H", 3)
W = hax.Axis("W", 4)
D = hax.Axis("D", 5)

x = hax.ones((H, W, D))
w = hax.ones((D,))
y = hax.einsum("h w d, d -> h w", x, w)  # shape is (H, W), equivalent to jnp.einsum("hwd,d->hw", x, w)
y = hax.einsum("... d, d -> ...", x, w)  # same as above
```

The `...` syntax is used to indicate that the axes in the input arrays that are not mentioned in the `einsum` string
should be preserved in the output. This is similar to `einops.einsum`'s behavior.

#### Unordered Mode

In "unordered" mode, the axes in the input arrays are matched to the axes in the `einsum` string by name,
using similar rules to [haliax.rearrange][]. Names involved in the operation are specified inside `{}`
on the left hand side of the `->` in the `einsum` string. Axes not specified are implicitly preserved.

```python
import haliax as hax

H = hax.Axis("H", 3)
W = hax.Axis("W", 4)
D = hax.Axis("D", 5)

x = hax.ones((H, W, D))
w = hax.ones((D,))

y = hax.einsum("{H W D} -> H W", x)  # shape is (H, W)
y = hax.einsum("{D} -> ", w)  # shape is (H, W)
y = hax.einsum("{...} -> ", x)  # shape is ()
y = hax.einsum("{H ...} -> H", x)  # shape is (H,)
y = hax.einsum("{H ...} -> ...", x)  # shape is (W, D)
```

This mode is most similar to [haliax.dot][]'s behavior, though it's a bit more expressive.

#### Output Axes Mode

In "output axes" mode, you only specify the axes that should be in the output. All other
axes are implicitly contracted over. This mode is a bit "dangerous" in that it's easy to
accidentally contract over axes you didn't mean to, but it can be useful for expressing
certain contractions concisely.

```python
import haliax as hax

H = hax.Axis("H", 3)
W = hax.Axis("W", 4)
D = hax.Axis("D", 5)

x = hax.ones((H, W, D))
w = hax.ones((D,))

y = hax.einsum("-> H W", x)  # shape is (H, W)
y = hax.einsum("-> D", w)  # shape is (D,)
```

We don't recommend using this mode except in cases when you're sure of the full shape of the input arrays
or you are sure you don't want to let users implicitly batch over any axes.

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
::: haliax.any
::: haliax.argmax
::: haliax.argmin
::: haliax.max
::: haliax.mean
::: haliax.min
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
::: haliax.bitwise_or
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

::: haliax.clip
::: haliax.isclose
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

### Old API

These functions are being deprecated and will be removed in a future release.

::: haliax.shard_with_axis_mapping
::: haliax.auto_sharded
