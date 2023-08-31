import dataclasses
from typing import Optional

import equinox as eqx
from jaxtyping import PRNGKeyArray

import haliax as hax

from ..axis import Axis
from ..core import NamedArray
from ..jax_utils import named_call
from ..tree_util import resize_axis


class Embedding(eqx.Module):
    weight: NamedArray

    # axes
    Vocab: Axis = eqx.static_field()
    Embed: Axis = eqx.static_field()

    # TODO: should allow axisspec for Embed
    @staticmethod
    def init(Vocab: Axis, Embed: Axis, initializer_range: float = 0.02, *, key):
        weight = hax.random.normal(key, (Vocab, Embed)) * initializer_range
        return Embedding(weight=weight, Vocab=Vocab, Embed=Embed)

    @named_call
    def __call__(self, input_ids, *, key: Optional[PRNGKeyArray] = None):
        return self.embed(input_ids)

    def embed(self, input_ids):
        input_embeds = self.weight.take(self.Vocab, input_ids)
        return input_embeds

    def unembed(self, input_embeds):
        return input_embeds.dot(self.Embed, self.weight)

    def resize_embeddings(self, new_size: int, key: Optional[PRNGKeyArray] = None):
        new_weights = resize_axis(self.weight, self.Vocab, new_size, key=key)
        return dataclasses.replace(self, Vocab=self.Vocab.resize(new_size), weight=new_weights)
