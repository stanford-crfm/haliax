import dataclasses
from typing import List, Tuple, Union

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu

from haliax.core import NamedArray
from haliax.util import is_jax_or_hax_array_like

from ._src.util import IdentityMap


ArrayLike = Union[jnp.ndarray, NamedArray]


def describe_array(arr):
    if isinstance(arr, NamedArray):
        return f"NamedArray(axes={arr.axes}, dtype={arr.dtype})"
    else:
        return f"ndarray(shape={arr.shape}, dtype={arr.dtype})"


class ModuleProblems(Exception):
    def __init__(self):
        self.reused_arrays: List[Tuple[ArrayLike, List]] = []
        self.static_arrays: List[str] = []

    def __bool__(self):
        return bool(self.reused_arrays or self.static_arrays)

    def __str__(self):
        if not self:
            return "No problems found"
        else:
            return "\n".join(
                [
                    "Found some problems with your module:",
                    *self._format_reused_arrays(),
                    *self._format_static_arrays(),
                ]
            )

    def _format_reused_arrays(self):
        return [f"  Reused array {describe_array(arr)} at paths {paths}" for arr, paths in self.reused_arrays]

    def _format_static_arrays(self):
        return [f"  Static array at field {field}" for field in self.static_arrays]


def diagnose_common_issues(module: eqx.Module):
    """
    Checks for common issues in a module, such as reused arrays and static arrays.
    Equinox modules (and therefore Haliax modules) should not have arrays that are stored
    in multiple places, and should not have arrays stored as static fields.

    We'll add more checks here as we find them.

    Args:
        module:  The module to check for problems

    Returns:
        None

    Raises:
        ModuleProblems: if any problems are found

    """

    problems = ModuleProblems()
    _check_for_reused_arrays(problems, module)
    _check_for_static_arrays(problems, module)

    if problems:
        raise problems

    # just in case we missed anything, raise equinox's errors:
    eqx.tree_check(module)


def _check_for_reused_arrays(problems, module):
    used_arrays = IdentityMap[ArrayLike, List[str]]()

    path_leaves, _ = jtu.tree_flatten_with_path(module, is_leaf=is_jax_or_hax_array_like)

    for path, leaf in path_leaves:
        if is_jax_or_hax_array_like(leaf):
            if leaf in used_arrays:
                used_arrays[leaf].append(jtu.keystr(path))
            else:
                used_arrays[leaf] = [jtu.keystr(path)]

    for arr, paths in used_arrays.items():
        if len(paths) > 1:
            problems.reused_arrays.append((arr, paths))


def _check_for_static_arrays(problems, module):
    static_arrays = []

    def recurse(module, path):
        if isinstance(module, eqx.Module):
            for field in dataclasses.fields(module):
                value = getattr(module, field.name)
                if field.metadata.get("static", False) and is_jax_or_hax_array_like(value):
                    static_arrays.append(f"{path}.{field.name}")
                else:
                    recurse(value, f"{path}.{field.name}")
        else:
            leaves, _ = eqx.tree_flatten_one_level(module)
            if leaves != [module]:
                leaves_with_names = jtu.tree_leaves_with_path(module, is_leaf=lambda x: x in leaves)
                for name, leaf in leaves_with_names:
                    recurse(leaf, f"{path}{name}")

    recurse(module, "")

    if static_arrays:
        problems.static_arrays.extend(static_arrays)
