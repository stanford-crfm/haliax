# Support for FP8
# Much of this is lifted from FLAX
# https://github.com/google/flax/blob/main/flax/linen/fp8_ops.py

# All of this is copy paste from flax/linen/fp8_ops.py
# (Until we get to the module)

# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional

import equinox as eqx
import jax
from jax import numpy as jnp
from jax.typing import DTypeLike

from haliax._src.fp8 import (
    dot_general_with_precision,
    in_qdq,
    in_qdq_bwd,
    in_qdq_fwd,
    out_qdq,
    out_qdq_bwd,
    out_qdq_fwd,
)
from haliax.types import PrecisionLike


class OverwriteWithGradient(eqx.Module):
    """
    Sometimes there is state that must be computed in the backward pass which we want to
    persist for subsequent passes. Typically, we see this with quantization, particularly
    FP8. This module is a marker that indicates to [haliax.quant.apply_updates][] that the
    gradient should be used to overwrite the state rather than added to it.
    """

    pass


def partition_for_grad_overwrite(tree):
    """
    This function is used to partition the state of a module into two parts: one that will be
    overwritten by the gradient and one that will be updated by the gradient. This is used by
    [eqx.apply_updates][] to determine which state should be updated and which should
    be overwritten.
    The usual pattern is something like:

        ```python
        grads = jax.grad(loss_fn)(model)
        overwrites, grads = partition_for_grad_overwrite(grads)
        updates = optimizer.update(grads, params=model)
        model = eqx.apply_updates(model, updates)
        model = eqx.combine(overwrites, model)
        ```

    """

    def is_overwrite_with_gradient(v):
        return isinstance(v, OverwriteWithGradient)

    x, y = eqx.partition(tree, is_overwrite_with_gradient, is_leaf=is_overwrite_with_gradient)
    return x, y


def apply_updates(tree, updates, overwrites):
    """
    A `jax.tree_util.tree_map`-broadcasted version of
    ```python
    if overwrite is not None:
        return overwrite
    if update is None:
        return model
    else:
        return model + update
    """

    def _apply_update(tree, update, overwrite):
        if overwrite is not None:
            return overwrite

        return eqx.apply_updates(tree, update)

    def is_leaf(x):
        return x is None or isinstance(x, OverwriteWithGradient)

    return jax.tree_util.tree_map(_apply_update, tree, updates, overwrites, is_leaf=is_leaf)


class Fp8DotGeneralOp(OverwriteWithGradient):
    input_scale: jnp.ndarray
    output_grad_scale: jnp.ndarray
    kernel_scale: jnp.ndarray
    input_amax_history: jnp.ndarray
    output_grad_amax_history: jnp.ndarray
    kernel_amax_history: jnp.ndarray
    compute_dtype: Optional[DTypeLike] = eqx.field(static=True)

    @classmethod
    def init(cls, amax_history_length: int = 1024, compute_dtype: DTypeLike = None):
        return cls(
            input_scale=jnp.ones(1, dtype=jnp.float32),
            output_grad_scale=jnp.ones(1, dtype=jnp.float32),
            kernel_scale=jnp.ones(1, dtype=jnp.float32),
            input_amax_history=jnp.zeros(amax_history_length, dtype=jnp.float32),
            output_grad_amax_history=jnp.zeros(amax_history_length, dtype=jnp.float32),
            kernel_amax_history=jnp.zeros(amax_history_length, dtype=jnp.float32),
            compute_dtype=compute_dtype,
        )

    # copied from flax
    def __call__(
        self,
        lhs,
        rhs,
        dimension_numbers,
        precision: PrecisionLike = None,
        preferred_element_type: DTypeLike | None = None,
    ):
        # Use the `k.dtype` since it aligns with the `dtype` of its layers,
        # namely, the computation data type.
        if self.compute_dtype is None:
            comp_dtype = rhs.dtype
        else:
            comp_dtype = self.compute_dtype
        lhs = jnp.asarray(lhs, comp_dtype)

        x_qdq = in_qdq(comp_dtype, lhs, self.input_scale, self.input_amax_history)
        k_qdq = in_qdq(comp_dtype, rhs, self.kernel_scale, self.kernel_amax_history)
        y_qdq = dot_general_with_precision(x_qdq, k_qdq, dimension_numbers, precision, preferred_element_type)
        y = out_qdq(
            comp_dtype,
            y_qdq,
            self.output_grad_scale,
            self.output_grad_amax_history,
        )

        return y
