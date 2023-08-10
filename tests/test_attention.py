import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKey

import haliax as hax
from haliax.nn.attention import (
    CausalAttentionMask,
    alibi_attention_bias,
    dot_product_attention_weights,
    forgetful_causal_mask,
)

from .test_utils import skip_if_no_torch


def test_alibi_attention_bias():
    KeyPos = hax.Axis("KeyPos", 20)
    NumHeads = hax.Axis("NumHeads", 1)
    Hid = hax.Axis("Hid", 8)

    bias = alibi_attention_bias(NumHeads, KeyPos)

    query = hax.ones((NumHeads, Hid))
    key = hax.ones((KeyPos, NumHeads, Hid))

    weights_bias = dot_product_attention_weights(Hid, KeyPos, query, key, bias=bias)
    weights_no_bias = dot_product_attention_weights(Hid, KeyPos, query, key)

    assert weights_bias.take(KeyPos, -1).item() > weights_bias.take(KeyPos, -2).item()
    assert weights_bias.take(KeyPos, -1).item() > weights_no_bias.take(KeyPos, -1).item()

    assert weights_no_bias.take(KeyPos, -1).item() == weights_no_bias.take(KeyPos, -2).item()


@skip_if_no_torch
def test_alibi_attention_compared_to_hf():
    import torch
    from transformers.models.bloom.modeling_bloom import build_alibi_tensor

    L = hax.Axis("L", 128)
    H = hax.Axis("NumHeads", 16)

    # Returns tensor shaped (batch_size * num_heads, 1, max_seq_len)
    torch_tensor = (
        build_alibi_tensor(torch.ones(1, L.size), H.size, dtype=torch.float32).numpy().reshape(H.size, L.size)
    )

    hax_tensor = np.array(alibi_attention_bias(H, L).array)

    assert np.allclose(torch_tensor, hax_tensor)


def test_fcm_attention_mask():
    KeyPos = hax.Axis("KeyPos", 20)

    mask = forgetful_causal_mask(KeyPos, mask_prob=0.6, sample_prob=False, key=PRNGKey(0))

    assert mask.axes == (KeyPos,)
    assert mask.array[0].item() == 1

    assert mask.astype(float).sum().item() <= KeyPos.size

    QueryPos = hax.Axis("QueryPos", 10)
    Head = hax.Axis("Head", 8)

    query = hax.arange(QueryPos).broadcast_axis(Head)
    key = hax.arange(KeyPos).broadcast_axis(Head)

    weights = dot_product_attention_weights(Head, KeyPos, query, key, mask=mask)

    # check that all masked out values are zero
    # TODO: think about how to make this work with named arrays
    weights = weights.rearrange((KeyPos, QueryPos)).array
    mask = mask.array

    assert weights[mask == 0].sum() == 0
    assert weights[mask == 1].sum() > 0


def test_causal_mask_blocking():
    pos = hax.Axis("pos", 128)
    key_pos = pos.alias("key_pos")

    mask = CausalAttentionMask(pos, key_pos)

    blocked_mask = mask.blocked(pos, 16).blocked(key_pos, 16)
    assert blocked_mask.Pos.size == 128 // 16
    assert blocked_mask.KeyPos.size == 128 // 16

    mat_blocked = blocked_mask.materialize()

    assert hax.all(mat_blocked == hax.nn.attention.causal_mask(pos.resize(8), key_pos.resize(8)))

    mat_mask = mask.materialize()

    for i in range(8):
        for j in range(8):
            assert mat_blocked.array[i, j] == jnp.any(mat_mask.array[i * 16 : (i + 1) * 16, j * 16 : (j + 1) * 16])


def test_causal_mask_slicing():
    pos = hax.Axis("pos", 128)
    key_pos = pos.alias("key_pos")

    mask = CausalAttentionMask(pos, key_pos)

    sliced = mask.slice(pos, 7, length=16).slice(key_pos, 24, length=16)

    mat_mask = mask.materialize()
    mat_sliced = sliced.materialize()

    for i in range(16):
        for j in range(16):
            assert mat_sliced.array[i, j] == mat_mask.array[7 + i, 24 + j]
