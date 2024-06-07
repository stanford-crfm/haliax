import dataclasses
import math
import warnings
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
    def init(Vocab: Axis, Embed: AxisSpec, *, init_scale: float = 1, key, initializer_range: Optional[float] = None):
        if initializer_range is not None:
            warnings.warn("initializer_range is deprecated. Use init_std instead.", DeprecationWarning)
            init_scale = initializer_range

        all_axes = (Vocab,) + ensure_tuple(Embed)
        output_size = hax.axis_size(Embed)
        weight = hax.random.truncated_normal(key, all_axes, -3, 3) * (init_scale / math.sqrt(output_size))
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
