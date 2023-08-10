import abc
import dataclasses
import functools as ft
import math
import typing
from typing import List, Optional, Tuple, TypeAlias, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PRNGKeyArray

import haliax
import haliax.random as hrandom
from haliax.axis import Axis, AxisSelection, AxisSelector, AxisSpec
from haliax.core import NamedArray
from haliax.types import PrecisionLike


# With attention we usually distinguish between the mask and the bias, though the former is just a special case of the
# latter. In practice, the mask is a boolean array that is applied using `where` to the logits, while the bias is a
# float array that is added to the logits. The mask is usually used to prevent attention to certain positions, while
# the bias is usually used to encourage or discourage attention to certain positions.
# The mask usually is head-independent, while the bias is frequently head-dependent

# because we use named axis we can be fairly loose about the shape of masks and biases: want to have a different
# mask for each head? fine. want to broadcast across the key sequence length? fine. etc etc


def dot_product_attention_weights(
    Head: Axis,
    KPos: AxisSelection,
    query: NamedArray,
    key: NamedArray,
    mask: Optional[Union[NamedArray, "AttentionMask"]] = None,
    bias: Optional[NamedArray] = None,
    attention_dtype: Optional[jnp.dtype] = None,
    precision: PrecisionLike = None,
) -> NamedArray:
    """
    NamedArray version of dot product attention. Computes the logits for the attention weights. Note that the
    "Pos" axis in query must be distinct from the "Pos" axis in key.

    :param Head: Axis of head dimension
    :param KPos: Axis of key sequence length. Can be an AxisSpec to attend along more than one axis.
    :param query: NamedArray of shape (QPos, KeySize)
    :param key: NamedArray of shape (KPos, KeySize)
    :param mask: Optional[NamedArray] broadcast compatible with (KeySize, QPos, KPos). Should be boolean
    :param bias: Optional[NamedArray] broadcast compatible with (KeySize, QPos, KPos). Should be float
    :param attention_dtype: Optional dtype to use for attention
    :param precision: PrecisionLike for dot product. See precision argument to jax.lax.dot_general
    :return: NamedArray of shape (QPos, KPos)
    """
    # cf https://github.com/google/flax/blob/509bf97ea272e130d932920f45307ac98947d994/flax/linen/attention.py#L40
    import haliax.nn as hnn

    orig_dtype = query.dtype
    query = query / jnp.sqrt(query.axis_size(Head))

    if attention_dtype is not None:
        query = query.astype(attention_dtype)
        key = key.astype(attention_dtype)

    weights = haliax.dot(Head, query, key, precision=precision)

    if bias is not None:
        weights = weights + bias
    if mask is not None:
        weights = haliax.where(materialize_mask(mask), weights, -1e9)

    weights = hnn.softmax(weights, axis=KPos)

    return weights.astype(orig_dtype)


def dot_product_attention(
    QPos: Axis,
    KPos: Axis,
    KeySize: Axis,
    query: NamedArray,
    key: NamedArray,
    value: NamedArray,
    mask: Optional[Union[NamedArray, "AttentionMask"]] = None,
    bias: Optional[NamedArray] = None,
    attention_dtype: Optional[jnp.dtype] = None,
    precision: PrecisionLike = None,
) -> NamedArray:
    """
    NamedArray version of dot product attention. This can be multi-headed or not.

    :param QPos: Axis of sequence length
    :param KPos: Axis of key sequence length
    :param KeySize: Axis of head dimension
    :param query: NamedArray of shape (QPos, KeySize)
    :param key: NamedArray of shape (KPos, KeySize)
    :param value: NamedArray of shape (KPos, KeySize)
    :param mask: Optional[NamedArray] broadcast compatible with (KeySize, QPos, KPos). Should be boolean
    :param bias: Optional[NamedArray] broadcast compatible with (KeySize, QPos, KPos). Should be float
    :param attention_dtype: Optional dtype to use for attention
    :param precision: PrecisionLike for dot product. See precision argument to jax.lax.dot_general
    :return: NamedArray of shape (QPos, KeySize)

    Mask and bias are given as separate arguments because they are often computed separately and have different shapes.
    For example, mask is frequently just a boolean array of shape (QPos, KPos), while bias is frequently a float
    array of shape (KeySize, QPos, KPos) or (KeySize, KPos)
    """
    # cf https://github.com/google/flax/blob/509bf97ea272e130d932920f45307ac98947d994/flax/linen/attention.py#L125

    # rename key/value length axis if it's the same as the query length axis
    if KPos == QPos:
        KPos = QPos.alias(KPos.name + "_key")
        key = key.rename({KPos: QPos})
        value = value.rename({KPos: QPos})

    weights = dot_product_attention_weights(KeySize, KPos, query, key, mask, bias, attention_dtype, precision)

    return haliax.dot(KPos, weights, value)


AttnMask: TypeAlias = Union[NamedArray, "AttentionMask"]


@typing.overload
def materialize_mask(mask: AttnMask) -> NamedArray:
    ...


@typing.overload
def materialize_mask(mask: Optional[AttnMask]) -> Optional[NamedArray]:
    ...


def materialize_mask(mask: Optional[AttnMask]) -> Optional[NamedArray]:
    """
    Materialize an attention mask if it is an AttentionMask. Otherwise, just return it.
    """
    if isinstance(mask, AttentionMask):
        mask = mask.materialize()
    return mask


class AttentionMask(eqx.Module, abc.ABC):
    """
    Represents an attention mask in a structured way to make it easier to optimize attention for particular use cases
    (causal, prefix, etc.). It is anticipated that this will be extended with new types of masks as needed.

    In general, it should be safe to batch Attention Masks, but it is important that *all members of a batch have the
    same sequence of combined masks*. Otherwise, the batching will not work and you'll get weird errors

    The interface exposed by this class is designed to work well with the attention functions in this module as
    well as something like flash attention.

    A mask can be materialized, in which case it returns the mask as a NamedArray.
    We can also ask for slices of a mask along a particular axis, or for a blocked version of the mask.

    The blocked version of the mask is basically a projection of the mask onto a smaller mask, where each position
    in the smaller mask is the max of the corresponding positions in the larger mask. This is useful for
    blockwise attention mechanisms, like flash or longformer.
    """

    @abc.abstractmethod
    def materialize(self) -> Optional[NamedArray]:
        """
        Materialize the mask as a NamedArray. This is useful for attention functions that don't support masks,
        or for the inner loop
        """
        raise NotImplementedError

    @abc.abstractmethod
    def slice(self, axis: AxisSelector, start: int, length: int) -> "AttentionMask":
        """
        Slice the mask along a particular axis. This is useful for extracting a particular slice of a mask
        for use in blocked attention.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def blocked(self, axis: AxisSelector, block_size: int) -> "AttentionMask":
        """
        Return a blocked version of the mask. This is useful for blockwise attention mechanisms, like flash or longformer.
        :param axis:
        :param block_size:
        :return:
        """
        raise NotImplementedError

    def __and__(self, other) -> "AttentionMask":
        if isinstance(self, AndAttentionMask):
            conjuncts = list(self.conjuncts)
        else:
            conjuncts = [self]

        if isinstance(other, AndAttentionMask):
            conjuncts.extend(other.conjuncts)
        else:
            conjuncts.append(other)

        return AndAttentionMask(conjuncts)

    def __or__(self, other) -> "AttentionMask":
        if isinstance(self, OrAttentionMask):
            disjuncts = list(self.disjuncts)
        else:
            disjuncts = [self]

        if isinstance(other, OrAttentionMask):
            disjuncts.extend(other.disjuncts)
        else:
            disjuncts.append(other)

        return OrAttentionMask(disjuncts)


class CausalAttentionMask(AttentionMask):
    Pos: Axis = eqx.field(static=True)
    KeyPos: Axis = eqx.field(static=True)
    pos_start: int = eqx.field(static=True, default=0)
    kpos_start: int = eqx.field(static=True, default=0)

    def materialize(self) -> Optional[NamedArray]:
        return causal_mask(self.Pos, self.KeyPos, self.pos_start, self.kpos_start)

    def slice(self, axis: AxisSelector, start: int, length: int) -> "AttentionMask":
        if haliax.selects_axis(axis, self.Pos):
            return dataclasses.replace(self, Pos=self.Pos.resize(length), pos_start=self.pos_start + start)
        elif haliax.selects_axis(axis, self.KeyPos):
            return dataclasses.replace(self, KeyPos=self.KeyPos.resize(length), kpos_start=self.kpos_start + start)
        else:
            raise ValueError(f"Invalid axis {axis}. Valid axes are {self.Pos} and {self.KeyPos}")

    def blocked(self, axis: AxisSelector, block_size: int) -> "CausalAttentionMask":
        # a blocked causal mask is just a smaller causal mask
        if haliax.selects_axis(axis, self.Pos):
            if self.Pos.size % block_size != 0:
                raise ValueError(f"Cannot block mask of size {self.Pos.size} with block size {block_size}")
            new_size = self.Pos.size // block_size
            return dataclasses.replace(self, Pos=self.Pos.resize(new_size), pos_start=self.pos_start // block_size)
        elif haliax.selects_axis(axis, self.KeyPos):
            if self.KeyPos.size % block_size != 0:
                raise ValueError(f"Cannot block mask of size {self.KeyPos.size} with block size {block_size}")
            new_size = self.KeyPos.size // block_size
            return dataclasses.replace(
                self, KeyPos=self.KeyPos.resize(new_size), kpos_start=self.kpos_start // block_size
            )
        else:
            raise ValueError(f"Invalid axis {axis}. Valid axes are {self.Pos} and {self.KeyPos}")


class PrefixAttentionMask(AttentionMask):
    Pos: Axis = eqx.field(static=True)
    KeyPos: Axis = eqx.field(static=True)
    # TODO: prefix size needs to be dynamic
    prefix_size: int = eqx.field(static=True)
    pos_start: int = eqx.field(static=True, default=0)
    kpos_start: int = eqx.field(static=True, default=0)

    def materialize(self) -> Optional[NamedArray]:
        return prefix_lm_mask(self.Pos, self.KeyPos, self.prefix_size, self.pos_start, self.kpos_start)

    def slice(self, axis: AxisSelector, start: int, length: int) -> "AttentionMask":
        if haliax.selects_axis(axis, self.Pos):
            return dataclasses.replace(self, Pos=self.Pos.resize(length), pos_start=self.pos_start + start)
        elif haliax.selects_axis(axis, self.KeyPos):
            return dataclasses.replace(self, KeyPos=self.KeyPos.resize(length), kpos_start=self.kpos_start + start)
        else:
            raise ValueError(f"Invalid axis {axis}. Valid axes are {self.Pos} and {self.KeyPos}")

    def blocked(self, axis: AxisSelector, block_size: int) -> "AttentionMask":
        if haliax.selects_axis(axis, self.Pos):
            if self.Pos.size % block_size != 0:
                raise ValueError(f"Cannot block mask of size {self.Pos.size} with block size {block_size}")
            new_size = self.Pos.size // block_size
            return dataclasses.replace(self, Pos=self.Pos.resize(new_size), pos_start=self.pos_start // block_size)
        elif haliax.selects_axis(axis, self.KeyPos):
            if self.KeyPos.size % block_size != 0:
                raise ValueError(f"Cannot block mask of size {self.KeyPos.size} with block size {block_size}")
            new_size = self.KeyPos.size // block_size
            return dataclasses.replace(
                self, KeyPos=self.KeyPos.resize(new_size), kpos_start=self.kpos_start // block_size
            )
        else:
            raise ValueError(f"Invalid axis {axis}. Valid axes are {self.Pos} and {self.KeyPos}")


class ExplicitAttentionMask(AttentionMask):
    mask: NamedArray

    def materialize(self) -> Optional[NamedArray]:
        return self.mask

    def slice(self, axis: AxisSelector, start: int, length: int) -> "AttentionMask":
        if haliax.selects_axis(self.mask.axes, axis):
            return dataclasses.replace(self, mask=self.mask.slice(axis, start=start, length=length))
        else:
            raise ValueError(f"Invalid axis {axis}. Valid axes are {self.mask}")

    def blocked(self, axis: AxisSelector, block_size: int) -> "AttentionMask":
        # we have to do blocked ourselves, and it's a bit messy
        axis = self.mask.resolve_axis(axis)

        if axis.size % block_size != 0:
            raise ValueError(f"Cannot block mask axis of size {axis.size} with block size {block_size}")

        new_size = self.mask.size // block_size

        block_axis = axis.alias(axis.name + "__block").resize(block_size)
        unflattened = self.mask.unflatten_axis(axis, (axis.resize(new_size), block_axis))
        blocked = haliax.any(unflattened, axis=block_axis)

        return dataclasses.replace(self, mask=blocked)


class AndAttentionMask(AttentionMask):
    conjuncts: Tuple[AttentionMask, ...]

    def materialize(self) -> Optional[NamedArray]:
        return ft.reduce(combine_masks_and, (conj.materialize() for conj in self.conjuncts))

    def slice(self, axis: AxisSelector, start: int, length: int) -> "AttentionMask":
        return dataclasses.replace(self, conjuncts=tuple(conj.slice(axis, start, length) for conj in self.conjuncts))

    def blocked(self, axis: AxisSelector, block_size: int) -> "AttentionMask":
        return dataclasses.replace(self, conjuncts=tuple(conj.blocked(axis, block_size) for conj in self.conjuncts))


class OrAttentionMask(AttentionMask):
    disjuncts: Tuple[AttentionMask, ...]

    def materialize(self) -> Optional[NamedArray]:
        return ft.reduce(combine_masks_or, (disj.materialize() for disj in self.disjuncts))

    def slice(self, axis: AxisSelector, start: int, length: int) -> "AttentionMask":
        return dataclasses.replace(self, disjuncts=tuple(disj.slice(axis, start, length) for disj in self.disjuncts))

    def blocked(self, axis: AxisSelector, block_size: int) -> "AttentionMask":
        return dataclasses.replace(self, disjuncts=tuple(disj.blocked(axis, block_size) for disj in self.disjuncts))


# TODO: padding mask
# TODO: FCM mask?
# TODO: sequence packing mask


def mask_to_bias(mask: NamedArray, mask_value: float = -1e9) -> NamedArray:
    return mask * mask_value


def combine_masks_and(mask1: Optional[NamedArray], mask2: Optional[NamedArray]) -> Optional[NamedArray]:
    if mask1 is None:
        return mask2
    if mask2 is None:
        return mask1
    return mask1 & mask2


def combine_masks_or(mask1: Optional[NamedArray], mask2: Optional[NamedArray]) -> Optional[NamedArray]:
    if mask1 is None:
        return mask2
    if mask2 is None:
        return mask1
    return mask1 | mask2


def causal_mask(QPos: Axis, KPos: Axis, q_start: int = 0, k_start: int = 0) -> NamedArray:
    """
    Creates a materialized causal mask for attention.

    :param QPos: Axis of query sequence length
    :param KPos: Axis of key sequence length
    :return: NamedArray of shape (QPos, KPos)
    """
    return haliax.arange(QPos, start=q_start).broadcast_axis(KPos) >= haliax.arange(KPos, start=k_start)


def prefix_lm_mask(QSeqLen: Axis, KSeqLen: Axis, prefix_len: int, q_start: int = 0, k_start: int = 0) -> NamedArray:
    """Mask for the PrefixLM objective: fully connected before prefix_len, then causal after."""
    # sometimes prefix_len is a tracer so we can't assert
    if isinstance(prefix_len, int):
        assert prefix_len >= 0
        # assert prefix_len <= KSeqLen.size

    causal = causal_mask(QSeqLen, KSeqLen, q_start=q_start, k_start=k_start)
    prefix = haliax.arange(KSeqLen, start=k_start) < (prefix_len + k_start)

    return prefix | causal


def dropout_mask(axes: AxisSpec, dropout_rate: float, *, key: PRNGKeyArray) -> NamedArray:
    """
    Really just an alias for haliax.random.bernoulli. You can pass in e.g. Head, QPos and KPos
    """
    return hrandom.bernoulli(key, shape=axes, p=1 - dropout_rate)


def forgetful_causal_mask(KPos: Axis, mask_prob: float, sample_prob: bool = True, *, key: PRNGKeyArray) -> NamedArray:
    """
    Forgetful Context Masking a la https://arxiv.org/abs/2210.13432. Randomly drops out positions from the key sequence.
    Reportedly better than normal attention dropout. Almost certainly faster.

    You're always allowed to attend to the 0th position. (They say BOS token, but we don't always start with bos)

    :param KPos: Axis of key sequence length
    :param mask_prob: Probability a position to mask
    :param sample_prob: If True, sample the prob between 0 and the provided prob (this is what the paper does)
    """
    zeroth_on = haliax.nn.one_hot(0, KPos, dtype=jnp.bool_)  # always allow 0th position
    if mask_prob == 0:
        return jnp.ones((KPos.size,), dtype=jnp.bool_)
    elif mask_prob == 1:
        return zeroth_on
    else:
        if sample_prob:
            key, subkey = jax.random.split(key)
            mask_prob = jax.random.uniform(subkey, shape=(), minval=0, maxval=mask_prob)
        base: NamedArray = hrandom.bernoulli(key, shape=(KPos,), p=1 - mask_prob)
        return base | zeroth_on


def _get_alibi_slopes(heads: int, bias_max: float) -> List[float]:
    # Mosaic supports "bias_max"
    log_bias_max = math.log2(bias_max)
    # from https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742

    def get_slopes_power_of_2(n: int):
        start = 2 ** (-(2 ** -(math.log2(n) - log_bias_max)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(heads).is_integer():
        return get_slopes_power_of_2(heads)
    closest_power_of_2 = 2 ** math.floor(math.log2(heads))
    return (
        get_slopes_power_of_2(closest_power_of_2)
        + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][: heads - closest_power_of_2]
    )


def alibi_attention_bias(Heads: Axis, KPos: Axis, bias_max: float = 8, dtype=jnp.float32) -> NamedArray:
    """
    Creates an attention bias for alibi attention.

    :param KPos: Axis of (key) sequence length
    :param Heads: Axis of heads
    :return: NamedArray of shape (Heads, KPos)
    """
    slopes = haliax.named(np.array(_get_alibi_slopes(Heads.size, bias_max)), Heads)
    positions = haliax.arange(KPos).broadcast_axis(Heads)

    biases = slopes * positions
    return biases.astype(dtype)
