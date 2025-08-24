import math
import jax
import jax.lax
import jax.numpy as jnp
import numpy as np
import pytest
import haliax as hax
import haliax.partitioning
from jaxtyping import PyTree
from typing import Tuple, Optional, Dict, List, Set
import jax.tree_util as jtu


def _get_mapped_axes(x: PyTree, kept_axes: Set[str]) -> List[hax.Axis]:
    """
    Identifies all axes in a PyTree that are not specified in the kept_axes set.

    Args:
        x (PyTree): The input PyTree containing NamedArrays.
        kept_axes (Set[str]): A set of axis names to exclude.

    Returns:
        List[hax.Axis]: A list of unique hax.Axis objects present in the PyTree 
                        whose names are not in kept_axes.
    """
    all_axes = set()

    # If the input itself is a NamedArray, collect its axes
    if isinstance(x, hax.NamedArray):
        all_axes.update(x.axes)
    else:
        # Otherwise, traverse the PyTree looking for NamedArrays
        def _collect_axes(leaf):
            if isinstance(leaf, hax.NamedArray):
                for ax in leaf.axes:
                    all_axes.add(ax)
        
        jtu.tree_map(_collect_axes, x)

    mapped_axes = [ax for ax in all_axes if ax.name not in kept_axes]
    # Sort for deterministic order, although not strictly necessary
    mapped_axes.sort(key=lambda ax: ax.name)
    return mapped_axes


def _reshape_in(x: PyTree, kept_axes: set, per_device_batch_size: int) -> Tuple[PyTree, Optional[Dict[str, int]], Optional[Dict[str, int]], Optional[str]]:
    """
    Reshape the input PyTree by splitting mapped axes into physical and local parts.

    Args:
        x (PyTree): The input PyTree to be reshaped.
        kept_axes (set): A set of axis names to keep (not map over).
        per_device_batch_size (int): The batch size per device for vectorization.

    Returns:
        Tuple[PyTree, Optional[Dict[str, int]], Optional[Dict[str, int]], Optional[str]]: 
            - The reshaped PyTree.
            - A dictionary of mesh sizes for each mapped axis.
            - A dictionary of local sizes for each mapped axis.
            - A string representing the local part of the rearrange operation.
    """
    # Determine the mapped axes (all axes not in kept_axes).
    mapped_axes = _get_mapped_axes(x, kept_axes)
    if not mapped_axes:
        return x, None, None, None  # No reshaping needed

    # Compute the physical and logical sizes for each mapped axis.
    mesh_sizes = {}
    local_sizes = {}
    for ax in mapped_axes:
        partitioned_size = haliax.partitioning.physical_axis_size(ax)
        if partitioned_size is None:
            partitioned_size = 1
        mesh_sizes[ax.name] = partitioned_size
        local_size = ax.size // partitioned_size
        local_sizes[ax.name] = local_size

    # Build a rearrange string to split each mapped axis into its physical and local parts.
    local_names = []
    rearrange_str = ""
    for ax in x.axes:
        assert ")" not in ax.name, f"Axis name {ax.name} contains a parenthesis"
        if ax.name in mesh_sizes:
            rearrange_str += f"({ax.name}: {ax.name} {ax.name}__local) "
            local_names.append(f"{ax.name}__local")
        else:
            rearrange_str += ax.name + " "

    rearrange_str = rearrange_str[:-1]
    local_part = f"(__LOCAL__: {' '.join(local_names)}) "
    rearrange_str += f"-> {local_part} "
    for ax in x.axes:
        rearrange_str += f"{ax.name} "
    rearrange_str = rearrange_str[:-1]

    # Rearrange the array to split mapped axes into their physical and local parts.
    x = hax.rearrange(x, rearrange_str, **mesh_sizes)
    x = hax.auto_sharded(x)

    return x, mesh_sizes, local_sizes, local_part


def _reshape_out(scanned: PyTree, mesh_sizes: Dict[str, int], local_sizes: Dict[str, int], local_part: str) -> PyTree:
    """
    Reshape the output PyTree by merging the physical and local parts back into the original mapped axes.

    Args:
        scanned (PyTree): The scanned PyTree to be reshaped.
        mesh_sizes (Dict[str, int]): A dictionary of mesh sizes for each mapped axis.
        local_sizes (Dict[str, int]): A dictionary of local sizes for each mapped axis.
        local_part (str): A string representing the local part of the rearrange operation.

    Returns:
        PyTree: The reshaped PyTree with original mapped axes restored.
    """
    # Merge the physical and local parts back into the original mapped axes.
    rearrange_str = "{" + local_part + " "
    for ax in scanned.axes:
        if ax.name == "__LOCAL__":
            continue
        rearrange_str += f"{ax.name} "

    rearrange_str = rearrange_str[:-1]
    rearrange_str += "}-> "

    for ax in scanned.axes:
        if ax.name == "__LOCAL__":
            continue
        if ax.name in mesh_sizes:
            rearrange_str += f"({ax.name}: {ax.name} {ax.name}__local) "
        else:
            rearrange_str += ax.name + " "

    rearrange_str = rearrange_str[:-1]

    scanned = hax.rearrange(scanned, rearrange_str,
                            **{f"{ax}__local": local_sizes[ax]
                               for ax in mesh_sizes})
    return scanned


def map_axes_other_than(f, kept_axes, per_device_batch_size=1):
    """
    Returns a new function that applies `f` over all axes not in `kept_axes`
    but only vectorizes up to `per_device_batch_size` elements per device.

    Args:
        f: The function to apply.
        kept_axes: A set or collection of axis names to keep (not map over).
        per_device_batch_size: The batch size per device for vectorization.
    """
    # Ensure kept_axes is a set
    kept_axes = set(kept_axes)

    def wrapped(x: PyTree, *args, **kwargs):
        x, mesh_sizes, local_sizes, local_part = _reshape_in(x, kept_axes, per_device_batch_size)
        if mesh_sizes is None:
            return f(x, *args, **kwargs)

        needs_vmapping = False
        local_size = x.axis_size("__LOCAL__")
        if per_device_batch_size != 1 and local_size > per_device_batch_size:
            if local_size % per_device_batch_size != 0:
                raise ValueError(f"The number of examples on a device {local_size} is not divisible by "
                                 f"per_device_batch_size {per_device_batch_size}.")
            needs_vmapping = True
            num_chunks = local_size // per_device_batch_size
            x = hax.unflatten_axis(x, "__LOCAL__",
                                   (hax.Axis("__LOCAL__", num_chunks), hax.Axis("__LOCAL__CHUNK", per_device_batch_size)))
            mod_f = hax.vmap(f, "__LOCAL__CHUNK")
        else:
            mod_f = f

        def scan_fn(carry, slice_x):
            out = mod_f(slice_x, *args, **kwargs)
            return carry, out

        _, scanned = haliax.scan(scan_fn, "__LOCAL__")(None, x, *args, **kwargs)

        if needs_vmapping:
            scanned = hax.flatten_axes(scanned, ("__LOCAL__", "__LOCAL__CHUNK"), "__LOCAL__")

        scanned = _reshape_out(scanned, mesh_sizes, local_sizes, local_part)
        return scanned

    return wrapped


# --- Tests ---

def test_no_mapped_axes():
    # When all axes are kept, map_axes_other_than should apply f directly.
    Batch = hax.Axis("batch", 4)
    x = hax.arange(Batch)

    # f simply adds one.
    def add_one(x):
        return x + 1

    pv = map_axes_other_than(add_one, kept_axes={"batch"}, per_device_batch_size=2)
    out = pv(x)
    np.testing.assert_allclose(out.array, (x.array + 1))
    assert [ax.name for ax in out.axes] == [ax.name for ax in x.axes]
    assert [ax.size for ax in out.axes] == [ax.size for ax in x.axes]


def test_single_mapped_axis():
    # One axis ("time") is mapped, "batch" is kept.
    Batch = hax.Axis("batch", 3)
    Time = hax.Axis("time", 8)
    x = hax.arange((Batch, Time))

    def mul_two(x):
        return x * 2

    pv = map_axes_other_than(mul_two, kept_axes={"time"}, per_device_batch_size=3)
    out = pv(x)
    np.testing.assert_allclose(out.array, x.array * 2)
    # Check that the axes remain unchanged.
    assert [ax.name for ax in out.axes] == [ax.name for ax in x.axes]
    assert [ax.size for ax in out.axes] == [ax.size for ax in x.axes]


def test_multiple_mapped_axes():
    # Two mapped axes: "time" and "step"; kept axis "batch".
    Batch = hax.Axis("batch", 2)
    Time = hax.Axis("time", 6)   # For per_device_batch_size=2: physical= (6//?); ensure divisibility.
    Step = hax.Axis("step", 4)   # per_device_batch_size=2 for "step": physical=2, logical=2.
    x = hax.arange((Batch, Time, Step))

    def plus_five(x):
        return x + 5

    pv = map_axes_other_than(plus_five, kept_axes={"batch"}, per_device_batch_size=2)
    out = pv(x)
    np.testing.assert_allclose(out.array, x.array + 5)
    # Verify that the axes are unchanged.
    assert [ax.name for ax in out.axes] == [ax.name for ax in x.axes]
    assert [ax.size for ax in out.axes] == [ax.size for ax in x.axes]


def test_per_device_batch_size_greater_than_axis():
    # If per_device_batch_size exceeds the axis size, the physical size should be the axis size.
    Batch = hax.Axis("batch", 3)
    Time = hax.Axis("time", 2)  # per_device_batch_size=3 > 2, so physical becomes 2 and logical is 1.
    x = hax.arange((Batch, Time))

    def identity(x):
        return x

    pv = map_axes_other_than(identity, kept_axes={"batch"}, per_device_batch_size=3)
    out = pv(x)
    np.testing.assert_allclose(out.array, x.array)
    assert [ax.name for ax in out.axes] == [ax.name for ax in x.axes]
    assert [ax.size for ax in out.axes] == [ax.size for ax in x.axes]


def test_error_on_non_divisible():
    # When an axis size is not divisible by the chosen physical size, the function should raise an error.
    Batch = hax.Axis("batch", 3)
    Time = hax.Axis("time", 7)  # 7 is not divisible by per_device_batch_size=3.
    x = hax.arange((Batch, Time))

    def dummy(x):
        return x

    print("\nTest setup:")
    print(f"Time axis size: {Time.size}")
    print(f"per_device_batch_size: 3")
    
    pv = map_axes_other_than(dummy, kept_axes={"batch"}, per_device_batch_size=3)
    with pytest.raises(ValueError, match="not divisible"):
        pv(x)
