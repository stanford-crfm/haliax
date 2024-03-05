import dataclasses
from typing import Optional

import equinox as eqx
from jaxtyping import PRNGKeyArray

import haliax as hax

from ..axis import Axis, AxisSpec
from ..core import NamedArray
from ..jax_utils import named_call
from ..tree_util import resize_axis
from ..util import ensure_tuple


class Embedding(eqx.Module):
    weight: NamedArray

    # axes
    Vocab: Axis = eqx.static_field()
    Embed: AxisSpec = eqx.static_field()

    @staticmethod
    def init(Vocab: Axis, Embed: AxisSpec, initializer_range: float = 0.02, *, key):
        all_axes = (Vocab,) + ensure_tuple(Embed)
        weight = hax.random.normal(key, all_axes) * initializer_range
        return Embedding(weight=weight, Vocab=Vocab, Embed=Embed)

    def __call__(self, input_ids, *, key: Optional[PRNGKeyArray] = None):
        return self.embed(input_ids)

    @named_call
    def embed(self, input_ids):
        input_embeds = self.weight.take(self.Vocab, input_ids)
        return input_embeds

    def unembed(self, input_embeds):
        return input_embeds.dot(self.weight, axis=self.Embed)

    def resize_embeddings(self, new_size: int, key: Optional[PRNGKeyArray] = None):
        new_weights = resize_axis(self.weight, self.Vocab, new_size, key=key)
        return dataclasses.replace(self, Vocab=self.Vocab.resize(new_size), weight=new_weights)  # type: ignore
