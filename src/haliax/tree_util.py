import equinox as eqx
import jax
import jax.tree_util as jtu
from jaxtyping import PRNGKeyArray, PyTree

from .core import Axis, NamedArray
from .util import is_named_array


def resize_axis(tree: PyTree[NamedArray], axis: Axis, key: PRNGKeyArray):
    """Resizes the NamedArrays of a PyTree along a given axis. If the array needs to grow, then the new elements are
    sampled from a truncated normal distribution with the same mean and standard deviation as the existing elements.
    If the array needs to shrink, then it's truncated."""
    import haliax.random

    def _resize_one(x, key):
        if not is_named_array(x):
            return x

        assert isinstance(x, NamedArray)

        try:
            current_axis = x.resolve_axis(axis.name)
        except ValueError:
            return x

        if axis.size == current_axis.size:
            return x
        elif current_axis.size > axis.size:
            return x.slice(current_axis, start=0, new_axis=axis)
        else:
            num_padding = axis.size - current_axis.size

            mean = x.mean(current_axis)
            std = x.std(current_axis)

            # the shape of the padding is the same as the original array, except with the axis size changed
            padding_axes = list(x.axes)
            padding_axes[padding_axes.index(current_axis)] = axis.resize(num_padding)

            padding = haliax.random.truncated_normal(key, padding_axes, lower=-2, upper=2) * std + mean

            return haliax.concatenate(axis, [x, padding])

    leaves, structure = jax.tree_util.tree_flatten(tree, is_leaf=is_named_array)
    keys = jax.random.split(key, len(leaves))

    new_leaves = [_resize_one(x, key) for x, key in zip(leaves, keys)]

    return jax.tree_util.tree_unflatten(structure, new_leaves)


# old version of eqx's partition functions
def hashable_partition(pytree, filter_spec):
    dynamic, static = eqx.partition(pytree, filter_spec)
    static_leaves, static_treedef = jtu.tree_flatten(static)
    static_leaves = tuple(static_leaves)
    return dynamic, (static_leaves, static_treedef)


def hashable_combine(dynamic, static):
    static_leaves, static_treedef = static
    static = jtu.tree_unflatten(static_treedef, static_leaves)
    return eqx.combine(dynamic, static)
