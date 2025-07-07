# NamedArray Type Annotations

Haliax supports a lightweight syntax for specifying the axes of a `NamedArray`
in type annotations.  Internally, `Named[...]` expands to
`Annotated[NamedArray, axes]`, so it works well with static type checkers like
``mypy``.  The syntax mirrors normal indexing with axis names.  Some examples:

```python
from haliax import Named

arr: Named["batch", "embed"]
arr: Named["batch embed ..."]      # starts with these axes
arr: Named["... embed"]             # ends with this axis
arr: Named["batch ... embed"]       # contains these axes in order
arr: Named[{"batch", "embed"}]      # has exactly these axes, order ignored
arr: Named[{"batch", "embed", ...}] # has at least these axes
```

At runtime you can verify that a `NamedArray` conforms to a particular
annotation using `matches_axes`:

```python
if not arr.matches_axes(Named["batch embed ..."]):
    raise ValueError("unexpected axes")
```

## DType-aware annotations

Sometimes it is useful to express both the axes **and** the dtype in the type
annotation.  The :mod:`haliax.typing` module defines symbolic types for all of
JAX's common dtypes that can be indexed just like ``Named``.  In documentation
examples we'll use ``import haliax.typing as ht``:

```python
import haliax.typing as ht

def foo(x: ht.f32["batch"]):
    ...

def bar(x: ht.i32["batch"]):
    ...
```

For convenience the module also provides aggregate categories ``Float``,
``Complex``, ``Int`` and ``UInt`` that match any floating point, complex,
signed integer or unsigned integer dtype respectively:

```python
def baz(x: ht.Float["batch"]):
    ...
```

At runtime ``matches_axes`` also checks the dtype when one is present:

```python
from haliax import Axis, zeros
import haliax.typing as ht

arr = zeros({"batch": 4})
assert arr.matches_axes(ht.f32["batch"])  # dtype and axes both match
```


## Inspiration

This type annotation syntax is inspired by the excellent [`jaxtyping`](https://docs.kidger.site/jaxtyping/)
library, which provides a similar approach for plain JAX arrays. Because Haliax has actual names,
we can provide a more powerful and flexible syntax that works with known names and generic names.


## FAQ

### Why not use `NamedArray` directly in type annotations?

Using `NamedArray` directly in type annotations is not recommended because it does not play nice with static type checkers like `mypy`.
The `Named` syntax is designed to be compatible with type checkers, allowing them to understand the structure of the data
without needing to resolve the actual `NamedArray` class at runtime. This makes it easier to catch type errors during development.
