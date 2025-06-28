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
