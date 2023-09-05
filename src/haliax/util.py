from typing import Callable, MutableMapping, Sequence, Tuple, Type, TypeAlias, TypeVar, Union

from haliax.jax_utils import is_jax_array_like


T = TypeVar("T")


py_slice = slice

slice_t = Type[slice]

Unspecified: TypeAlias = type("NotSpecified", (), {})  # type: ignore
UNSPECIFIED = Unspecified()


def is_named_array(leaf):
    from .core import NamedArray

    "Typically used as the is_leaf predicate in tree_map"
    return isinstance(leaf, NamedArray)


def ensure_tuple(x: Union[Sequence[T], T]) -> Tuple[T, ...]:
    if isinstance(x, str):
        return (x,)  # type: ignore
    elif isinstance(x, Sequence):
        return tuple(x)
    return (x,)


class StringHolderEnum(type):
    """Like a python enum but just holds string constants, as opposed to wrapped string constants"""

    # https://stackoverflow.com/questions/62881486/a-group-of-constants-in-python

    def __new__(cls, name, bases, members):
        # this just iterates through the class dict and removes
        # all the dunder methods
        cls.members = [v for k, v in members.items() if not k.startswith("__") and not callable(v)]
        return super().__new__(cls, name, bases, members)

    # giving your class an __iter__ method gives you membership checking
    # and the ability to easily convert to another iterable
    def __iter__(cls):
        yield from cls.members


def is_jax_or_hax_array_like(x):
    return is_jax_array_like(x) or is_named_array(x)


def index_where(pred: Callable[[T], bool], xs: Sequence[T]) -> int:
    for i, x in enumerate(xs):
        if pred(x):
            return i
    raise ValueError("No element satisfies predicate")


class IdentityMap(MutableMapping):
    """Map that compares keys by identity.

    This is a map that compares keys by identity instead of equality. It is
    useful for storing objects that are not hashable or that should be compared
    by identity.

    This is a mutable mapping, but it does not support the ``__hash__`` method
    and therefore cannot be used as a dictionary key or as an element of another
    set.
    """

    def __init__(self, iterable=None):
        self._data = {}
        if iterable is not None:
            self.update(iterable)

    def __contains__(self, key):
        return id(key) in self._data

    def __getitem__(self, key):
        return self._data[id(key)][1]

    def __setitem__(self, key, value):
        self._data[id(key)] = [key, value]

    def __delitem__(self, key):
        del self._data[id(key)]

    def __iter__(self):
        return (x[0] for x in self._data.values())

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"IdentityMap({list(repr(x) for x in self._data.values())})"

    def __str__(self):
        return f"IdentityMap({list(str(x) for x in self._data.values())})"


__all__ = [
    "is_named_array",
    "ensure_tuple",
    "StringHolderEnum",
    "is_jax_or_hax_array_like",
    "index_where",
    "slice_t",
    "py_slice",
    "IdentityMap",
]
