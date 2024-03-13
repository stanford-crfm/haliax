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
import warnings
from functools import partial
from typing import Optional

import equinox as eqx
import jax
from jax import custom_jvp, custom_vjp, lax
from jax import numpy as jnp
from jax.typing import DTypeLike

from haliax.types import PrecisionLike


def get_fp8_max(fp8_dtype, out_dtype):
    assert fp8_dtype in (jnp.float8_e4m3fn, jnp.float8_e5m2)
    return jnp.finfo(fp8_dtype).max.astype(out_dtype)


def quantize(x, q_dtype, scale, compute_dtype):
    # Explicitly cast the max values to the compute dtype to avoid unnecessary
    # casting to FP32 during the subsequent math operations."
    dtype_max = get_fp8_max(q_dtype, compute_dtype)
    scaled_x = x / jnp.broadcast_to(scale.astype(compute_dtype), x.shape)
    clipped_x = jnp.clip(scaled_x, -dtype_max, dtype_max)
    return clipped_x.astype(q_dtype)


def dequantize(x, dq_dtype, scale):
    return x.astype(dq_dtype) * jnp.broadcast_to(scale.astype(dq_dtype), x.shape)


def quantize_dequantize(x, q_dtype, scale, compute_dtype):
    qx = quantize(x, q_dtype, scale, compute_dtype)
    return dequantize(qx, x.dtype, scale)


def compute_scale(amax, scale, fp8_max, margin=0):
    # The algorithm for computing the new scale is sourced from
    #   https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/jax.html#transformer_engine.jax.update_fp8_metas
    # wherein the `original_scale` corresponds to the reciprocal of the `scale`
    # passed in this function.
    scale = 1.0 / scale

    sf = (fp8_max / amax) / (2**margin)
    sf = jnp.where(amax > 0.0, sf, scale)
    sf = jnp.where(jnp.isfinite(amax), sf, scale)

    return 1.0 / sf


def compute_amax_history(x, amax_history):
    amax_update = jnp.max(jnp.abs(x)).astype(amax_history.dtype)
    new_history = jnp.roll(amax_history, shift=-1, axis=0).at[0].set(amax_update)
    return new_history


def qdq_and_return(x, q_dtype, scale, amax_history, compute_dtype):
    dtype_max = get_fp8_max(q_dtype, jnp.float32)
    amax_from_history = jnp.max(amax_history, axis=0)
    new_scale = compute_scale(amax_from_history, scale, dtype_max)

    qx = quantize_dequantize(x, q_dtype, new_scale, compute_dtype)

    new_history = compute_amax_history(x, amax_history)

    return qx, new_scale, new_history


@partial(custom_vjp, nondiff_argnums=(0,))
def in_qdq(compute_dtype, inp, scale, amax_history):
    qin, _, _ = qdq_and_return(inp, jnp.float8_e4m3fn, scale, amax_history, compute_dtype)
    return qin


def in_qdq_fwd(compute_dtype, inp, scale, amax_history):
    qin, new_scale, new_history = qdq_and_return(inp, jnp.float8_e4m3fn, scale, amax_history, compute_dtype)
    return qin, (new_scale, new_history)


def in_qdq_bwd(compute_dtype, res, g):
    new_scale, new_history = res
    q_g = g
    return q_g, new_scale, new_history


in_qdq.defvjp(in_qdq_fwd, in_qdq_bwd)


@partial(custom_vjp, nondiff_argnums=(0,))
def out_qdq(compute_dtype, out, scale, amax_history):
    return out


def out_qdq_fwd(compute_dtype, out, scale, amax_history):
    return out, (scale, amax_history)


def out_qdq_bwd(compute_dtype, res, g):
    scale, amax_history = res
    q_g, new_scale, new_history = qdq_and_return(g, jnp.float8_e5m2, scale, amax_history, compute_dtype)
    return q_g, new_scale, new_history


out_qdq.defvjp(out_qdq_fwd, out_qdq_bwd)


@partial(custom_jvp, nondiff_argnums=(2, 3, 4))
def dot_general_with_precision(lhs, rhs, dimension_numbers, precision=None, preferred_element_type=None):
    if precision is not None or preferred_element_type is not None:
        warnings.warn(
            "The function dot_general_with_precision will set the "
            "precision/preferred_element_type and disregard any provided "
            "values."
        )
    return lax.dot_general(lhs, rhs, dimension_numbers, precision=lax.Precision.DEFAULT)


@dot_general_with_precision.defjvp
def dot_general_with_precision_jvp(dimension_numbers, precision, preferred_element_type, primals, tangents):
    lhs, rhs = primals
    lhs_dot, rhs_dot = tangents

    out = lax.dot_general(lhs, rhs, dimension_numbers, precision=lax.Precision.DEFAULT)
    grad_out = lax.dot_general(lhs_dot, rhs, dimension_numbers, precision=lax.Precision.HIGHEST) + lax.dot_general(
        lhs, rhs_dot, dimension_numbers, precision=lax.Precision.HIGHEST
    )
    return out, grad_out


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
