import math
import jax
import jax.lax
import jax.numpy as jnp
import numpy as np
import pytest
import haliax as hax
import haliax.partitioning


def map_axes_other_than(f, kept_axes, per_device_batch_size=1):
    """
    Returns a new function that applies `f` over all axes not in `kept_axes`
    but only vectorizes up to `per_device_batch_size` elements per device.

    For example, if you have a function `f` that operates on the axis `Vocab`,
    but is implicitly batched over all other axes (e.g. `Batch`, `Time`, etc.),
    you can use this to vectorize over all axes except `kept_axes`.

    This has a similar effect to vmapping (in that it hides axes from the function f), but:
    (1) you specify which axes to keep (i.e., not map over);
    (2) it is "sharding-aware" meaning that it still parallelizes across devices,
        processing up to `per_device_batch_size` elements per device in parallel;
      (3) it limits the amount of vectorization that happens to help XLA not overallocate memory.
    """

    def wrapped(x, *args, **kwargs):
        # 1. Determine the mapped axes (all axes not in kept_axes).
        mapped_axes = [ax for ax in x.axes if ax.name not in kept_axes]
        if not mapped_axes:
            # Nothing to map over; just apply f directly.
            return f(x, *args, **kwargs)

        # 2. For each mapped axis, compute the physical and logical sizes.
        mesh_sizes = {}
        local_sizes = {}
        for ax in mapped_axes:
            partitioned_size = haliax.partitioning.physical_axis_size(ax)
            if partitioned_size is None:
                partitioned_size = 1
            mesh_sizes[ax.name] = partitioned_size
            local_size = ax.size // partitioned_size
            local_sizes[ax.name] = local_size

        # 3. Build a rearrange string that splits each mapped axis into two: a physical part and a local part.
        local_names = []
        rearrange_str = ""
        for i, ax in enumerate(x.axes):
            # Ensure that axis names do not contain a parenthesis (used for destructuring).
            assert ")" not in ax.name, f"Axis name {ax.name} contains a parenthesis"
            if ax.name in mesh_sizes:
                # This axis will be destructured into a physical and a local component.
                rearrange_str += f"({ax.name}: {ax.name} {ax.name}__local) "
                local_names.append(f"{ax.name}__local")
            else:
                rearrange_str += ax.name + " "

        rearrange_str = rearrange_str[:-1]

        # We want the local parts to be grouped together; we assign them to a new axis "__LOCAL__".
        local_part = f"(__LOCAL__: {' '.join(local_names)}) "

        rearrange_str += f"-> {local_part} "

        for i, ax in enumerate(x.axes):
            rearrange_str += f"{ax.name} "

        rearrange_str = rearrange_str[:-1]

        # Rearrange the array to split mapped axes into their physical and local parts.
        x = hax.rearrange(x, rearrange_str, **mesh_sizes)
        # Ensure the array is auto-sharded based on the current axis mapping.
        x = hax.auto_sharded(x)

        # 4. If the local (__LOCAL__) axis is larger than per_device_batch_size, further split it.
        needs_vmapping = False
        local_size = x.axis_size("__LOCAL__")
        if per_device_batch_size != 1 and local_size > per_device_batch_size:
            if local_size % per_device_batch_size != 0:
                # Raise an error to avoid unexpected behavior if the size is not divisible.
                raise ValueError(f"The number of examples on a device {local_size} is not divisible by "
                                 f"per_device_batch_size {per_device_batch_size}.")
            needs_vmapping = True
            # Split the "__LOCAL__" axis into two: one for the number of chunks and one for the per-device batch.
            num_chunks = local_size // per_device_batch_size
            x = hax.unflatten_axis(x, "__LOCAL__",
                                   (hax.Axis("__LOCAL__", num_chunks), hax.Axis("__LOCAL__CHUNK", per_device_batch_size)))
            # Use vmap over the "__LOCAL__CHUNK" axis.
            mod_f = hax.vmap(f, "__LOCAL__CHUNK")
        else:
            mod_f = f

        # 5. Run a scan over the "__LOCAL__" axis.
        def scan_fn(carry, slice_x):
            out = mod_f(slice_x, *args, **kwargs)
            return carry, out

        _, scanned = haliax.scan(scan_fn, "__LOCAL__")(None, x, *args, **kwargs)

        # 6. If vmap was used, flatten the "__LOCAL__" and "__LOCAL__CHUNK" axes back into "__LOCAL__".
        if needs_vmapping:
            scanned = hax.flatten_axes(scanned, ("__LOCAL__", "__LOCAL__CHUNK"), "__LOCAL__")

        # 7. Rearrange to merge the physical and local parts back into the original mapped axes.
        rearrange_str = "{" + local_part + " "
        for i, ax in enumerate(scanned.axes):
            if ax.name == "__LOCAL__":
                continue
            rearrange_str += f"{ax.name} "

        rearrange_str = rearrange_str[:-1]
        rearrange_str += "}-> "

        for i, ax in enumerate(scanned.axes):
            if ax.name == "__LOCAL__":
                continue
            if ax.name in mesh_sizes:
                # For mapped axes, we need to merge the physical and logical parts.
                rearrange_str += f"({ax.name}: {ax.name} {ax.name}__local) "
            else:
                rearrange_str += ax.name + " "

        rearrange_str = rearrange_str[:-1]

        scanned = hax.rearrange(scanned, rearrange_str,
                                **{f"{ax.name}__local": local_sizes[ax.name]
                                   for ax in mapped_axes})
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

    pv = map_axes_other_than(dummy, kept_axes={"batch"}, per_device_batch_size=3)
    with pytest.raises(ValueError, match="not divisible"):
        pv(x)
