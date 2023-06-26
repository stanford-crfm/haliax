<!--haliax-intro-start-->
# Haliax

> *Though you don’t seem to be much for listening, it’s best to be careful. If you managed to catch hold of even just a piece of my name, you’d have all manner of power over me.*<br/>
> — Patrick Rothfuss, *The Name of the Wind*

Haliax is a [JAX](https:://github.com/google/jax) library for building neural networks with named tensors, in the tradition of Alexander Rush's [Tensor Considered Harmful](https://nlp.seas.harvard.edu/NamedTensor).
Named tensors improve the legibility and compositionality of tensor programs by using named axes instead of positional indices
as in NumPy, PyTorch, etc. Here's a minimal attention module implementation in Haliax. For a more detailed introduction,
please see the [Haliax tutorial](https://colab.research.google.com/drive/1TiTcQQ4V5mopbgCu1SVl-oqJtXn7rFnC).

```python

import equinox as eqx
import jax
import jax.numpy as jnp
import haliax as hax
import haliax.nn as hnn

Pos = hax.Axis("position", 1024)  # sequence length
KPos = Pos.alias("key_position")
Head = hax.Axis("head", 8)  # number of attention heads
Key = hax.Axis("key", 64)  # key size
Embed = hax.Axis("embed", 512)  # embedding size

def attention_scores(Key, KPos, query, key, mask):
  # how similar is each query to each key
  scores = hax.dot(Key, query, key) / jnp.sqrt(Key.size)

  if mask is not None:
    scores -= 1E9 * (1.0 - mask)

  # convert to probabilities
  scores = hax.nn.softmax(scores, KPos)
  return scores


def attention(Key, KPos, query, key, value, mask):
  scores = attention_scores(Key, KPos, query, key, mask)
  answers = hax.dot(KPos, scores, value)

  return answers

# Causal Mask means that if pos >= key_pos, then pos can attend to key_pos
causal_mask = hax.arange(Pos).broadcast_axis(KPos) >= hax.arange(KPos)

class Attention(eqx.Module):
  proj_qkv: hnn.Linear  # input projection from [Embed] -> [(q, k, v), Head, Key]
  proj_answer: hnn.Linear  # output projection from [Head, Key] -> [Embed]

  @staticmethod
  def init(Embed, Head, Key, *, key):
    Qkv = hax.Axis("qkv", 3)  # create all three at once

    k_qkv, k_ans = jax.random.split(key, 2)
    proj_qkv = hnn.Linear.init(In=Embed, Out=(Qkv, Head, Key), key=k_qkv)
    proj_answer = hnn.Linear.init(In=(Head, Key), Out=Embed, key=k_ans)
    return Attention(proj_qkv, proj_answer)

  def __call__(self, x, mask=None):
    qkv_out = self.proj_qkv(x)
    q, k, v = qkv_out.unbind("qkv")

    # Rename k and v's Pos as haliax doesn't support unnamed axes or duplicate axes
    k = k.rename({"position": "key_position"})
    v = v.rename({"position": "key_position"})

    answers = attention(Key, KPos, q, k, v, causal_mask)

    x = self.proj_answer(answers)
    return x
```

(We use the excellent [Equinox](https://github.com/patrick-kidger/equinox) library for its module system and tree transformations.)

Haliax is built to be fast: the generated code (using JAX/XLA) should be as fast as handwritten JAX code. Haliax is also built to be scalable: it
can support FSDP and Tensor Parallelism with just a few lines of code. Haliax's powers [Levanter](https://gihub.com/stanford-crfm/levanter),
our companion library for training large language models and other foundation models, with scale proven up to 20B parameters
and up to a TPU v3-256 pod slice.


Haliax was created by [Stanford's Center for Research on Foundation Models (CRFM)](https://crfm.stanford.edu/)'s research engineering team. ([We're hiring!](https://crfm.stanford.edu/apply.html))
You can find us in the #levanter channel on the unofficial [Jax LLM Discord](https://discord.gg/FkRGNX3ND).

<!--haliax-intro-end-->

## Documentation

Currently, we have two tutorials for Haliax:

* [Introduction to Haliax with Transformers](https://colab.research.google.com/drive/1TiTcQQ4V5mopbgCu1SVl-oqJtXn7rFnC)
* [Distributed Training in Haliax](https://colab.research.google.com/drive/1QX4yH3zRFF3Xiibf1aahETcSQ5nbcUMz) (including FSDP)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for more information.

## License

Haliax is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full license text.
