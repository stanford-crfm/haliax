# API Reference
In general, the API is designed to be similar to JAX's version of NumPy's API, with the main difference being
that we use names (either strings or [haliax.Axis][] objects) to specify axes instead of integers. This shows up for creating
arrays (see [haliax.zeros][] and [haliax.ones][]) as well as things like reductions (see [haliax.sum][] and
[haliax.mean][]).

## Axis Types
::: haliax.Axis
::: haliax.AxisSpec
::: haliax.AxisSelection
## Array Creation
::: haliax.zeros
::: haliax.ones
::: haliax.full
::: haliax.zeros_like
::: haliax.ones_like
::: haliax.full_like
::: haliax.arange

### Combining Arrays

We don't include `hstack` or `vstack` because we prefer semantic axes.

::: haliax.concatenate
::: haliax.stack

## Operations

[Binary](#binary-operations) and [unary](#unary-operations) operations are all more or less directly from JAX's NumPy API.
The only difference is they operate on named arrays instead.

### Reductions

Reduction operations are things like [haliax.sum][] and [haliax.mean][] that reduce an array along one or more axes.
With the exception of argmin and argmax, They all have the form:

```python
def sum(x, axis: Optional[AxisSelection] = None, where: Optional[NamedArray] = None) -> haliax.NamedArray:
    ...
```

with the behavior closely mirroring that of JAX's NumPy API. The `axis` argument can
be a single axis (or axis name), a tuple of axes, or `None` to reduce all axes. The `where` argument is a boolean array
that specifies which elements to include in the reduction. It must be broadcastable to the input array, using
Haliax's [broadcasting rules](broadcasting.md).

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

### Axis-Wise Operations
Axis-wise operations are things like [haliax.cumsum][] and [haliax.sort][] that operate on a single axis of an array but
don't reduce it.

::: haliax.cumsum
::: haliax.cumprod
::: haliax.sort
::: haliax.argsort

### Unary Operations

The `A` here means [haliax.NamedArray](), `Scalar`, or [jax.numpy.ndarray]().

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
