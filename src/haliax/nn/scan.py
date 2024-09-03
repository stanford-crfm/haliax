import functools
from typing import Dict, Generic, Optional, Protocol, Sequence, Type, TypeVar

import equinox as eqx
import jax

import haliax
import haliax.util
from haliax.jax_utils import filter_checkpoint

from ..axis import Axis


M = TypeVar("M", bound=eqx.Module)
M_co = TypeVar("M_co", bound=eqx.Module, covariant=True)
S = TypeVar("S", bound=eqx.Module)
T = TypeVar("T")


class ModuleInit(Protocol[M_co]):
    def __call__(self, *args, **kwargs) -> M_co:
        ...


class BlockFoldable(Protocol[M]):
    """
    A superclass for [haliax.nn.Stacked][] and [haliax.nn.BlockSeq][] that exposes the fold and scan methods, as
    well as a few other methods that are useful for these modules.

    This is a protocol, so you can use it as a type hint for a function that takes a Stacked or BlockSeq.
    Equinox modules can't directly inherit from Protocols, but you can use it as a type hint.
    """

    Block: Axis

    @classmethod
    def init(
        cls: Type[S], Block: Axis, module: Type[M], *, gradient_checkpointing: bool = False, prevent_cse: bool = False
    ) -> ModuleInit[S]:
        ...

    def scan(self, init: T, *extra_args, **extra_kwargs):
        ...

    def fold(self, init: T, *args, **kwargs) -> T:
        ...

    def unstacked(self) -> Sequence[M]:
        ...


class BlockSeq(eqx.Module, Generic[M]):
    """
    A "BlockSeq" wraps another module and produces a "sequential" version of it, where an input is applied
    to each instance of the sequential module in sequence. This is useful for e.g. transformers
    where you have multiple instances of the same transformer block and the input is applied in a fold/for loop
    in sequence.

    It's similar in spirit to an [equinox.nn.Sequential][]. Unlike [equinox.nn.Sequential][], BlockSeq does not need to be
    homogeneous (though the init method assumes that it is).
    """

    blocks: Sequence[M]
    Block: Axis = eqx.static_field()
    gradient_checkpointing: bool = eqx.static_field()

    @classmethod
    def init(
        cls: Type[S], Block: Axis, module: Type[M], *, gradient_checkpointing: bool = False, prevent_cse: bool = False
    ) -> ModuleInit[S]:
        """
        This is a curried init method that takes the Block and module and returns a function that takes
        the arguments to the module's init method. Any NamedArrays in the arguments will be sliced along the
        Block axis (if it exists). JAX arrays will be sliced along the first axis.
        """
        del prevent_cse  # not needed, but kept for compat with Stacked

        @functools.wraps(module)
        def fn(*args, **kwargs):
            # The only complexity here is that the args and kwargs might have a Block axis in them,
            # in which case we need to loop over them them to slice them out.

            def init_block(i):
                (block_args, block_kwargs) = haliax.tree_util.tree_map(
                    functools.partial(BlockSeq._slice_out, Block, i), (args, kwargs)
                )
                return module.init(*block_args, **block_kwargs)

            seq = [init_block(i) for i in range(Block.size)]

            return BlockSeq(seq, Block, gradient_checkpointing)

        return fn

    def scan(self, init: T, *extra_args, **extra_kwargs):
        out = []
        carry = init

        for i, block in enumerate(self.blocks):
            if self.gradient_checkpointing:
                block = filter_checkpoint(block)
            (block_args, block_kwargs) = haliax.tree_util.tree_map(
                functools.partial(BlockSeq._slice_out, self.Block, i), (extra_args, extra_kwargs)
            )
            block_result = block(carry, *block_args, **block_kwargs)
            if not isinstance(block_result, (tuple, list)) or len(block_result) != 2:
                raise ValueError(
                    f"BlockSeq.scan expects the block to return a pair of (carry, extra), got {block_result}"
                )

            carry, extra = block_result

            out.append(extra)

        # TODO: do we want to stack the outputs?
        return carry, out

    def fold(self, init: T, *args, **kwargs) -> T:
        carry = init
        for i, block in enumerate(self.blocks):
            if self.gradient_checkpointing:
                block = filter_checkpoint(block)
            (block_args, block_kwargs) = haliax.tree_util.tree_map(
                functools.partial(BlockSeq._slice_out, self.Block, i), (args, kwargs)
            )
            carry = block(carry, *block_args, **block_kwargs)
        return carry

    def unstacked(self) -> Sequence[M]:
        return self.blocks

    @staticmethod
    def _slice_out(Block, i, x):
        if haliax.is_named_array(x):
            if haliax.selects_axis(x.axes, Block):
                return x[Block, i]
            else:
                return x
        elif haliax.jax_utils.is_jax_array_like(x):
            return x[i]
        else:
            return x

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"blocks": None}


class Stacked(eqx.Module, Generic[M]):
    """
    A "Stacked" wraps another module and produces a "stacked" version of it, where an input is applied
    to each instance of the stacked module in sequence. This is useful for e.g. transformers
    where you have multiple instances of the same transformer block and the input is applied in a fold/for loop
    in sequence.

    It's similar in spirit to an [equinox.nn.Sequential], but it must be homogeneous. In Jax, this is much cheaper to
    compile than a sequential (or moral equivalent), because Jax compiles the module's method once, instead of unrolling
    the sequential and compiling everything as a giant graph. In Jax, this pattern is often called "scan layers" or
    "scan over layers".

    A further constraint is that the elements of the stack must have the same Python control flow. This is because
    Jax's scan primitive requires that the function you pass to it is pure, and the only way to do that is to ensure
    that the function has the same control flow for every element of the stack.

    Stacked supports both "fold" and "scan" semantics. "fold" is the same as a for loop that accumulates a single
    output, while "scan" is the same as a for loop that accumulates a list of intermediates as well as the final output.

    Stacked also supports gradient checkpointing, which is useful for very large models that don't fit in memory.

    Typically only one of "fold" or "scan" can be used with a given Stacked module, depending on the what the module
    returns: if the module returns a single output, use "fold"; if the module returns a sequence of intermediates and
    an output to be passed to the next layer, use "scan". More concretely, for a transformer, you would use "scan" if
    you wanted to return a kv cache (or the attention matrix) as well as the output of the transformer. If you just
    wanted the output of the transformer, you would use "fold".


    Example:
        ```python
        >>> import equinox as eqx
        >>> import haliax as hax
        >>> import haliax.nn as hnn
        >>> class MyModule(eqx.Module):
        ...     def __init__(self, num_layers: int, hidden: hax.Axis, *, key):
        ...         self.axis = hax.Axis("layer", num_layers)
        ...         split_key = jax.random.split(key, num_layers)
        ...         self.layers = Stacked.init(self.axis, hnn.Linear)(In=hidden, Out=hidden, key=split_key)
        ...
        ...     def __call__(self, x):
        ...         return self.layers.fold(x)  # applies each layer in sequence
        ...
        >>> Hidden = hax.Axis("hidden", 10)
        >>> mod = MyModule(5, Hidden)
        >>> mod(hax.ones(Hidden))
        ```
    """

    # TODO: we can probably make this module support pipeline parallelism, but that's a whole project in itself

    stacked: M
    Block: Axis = eqx.static_field()
    # TODO: support fancier gradient checkpointing
    gradient_checkpointing: bool = eqx.static_field()
    prevent_cse: bool = eqx.static_field()

    @classmethod
    def init(
        cls, Block: Axis, module: Type[M], *, gradient_checkpointing: bool = False, prevent_cse: bool = False
    ) -> ModuleInit["Stacked[M]"]:
        """
        Initialize a Stacked module. This method is curried: you can pass in the Block and module, and it will return
        a function that takes (batched) arguments to the vmapped module's init method.
        :param Block:
        :param module:
        :param gradient_checkpointing:
        :param prevent_cse:
        :return:
        """

        @functools.wraps(module)
        def fn(*args, **kwargs):
            stacked = haliax.vmap(module.init, Block)(*args, **kwargs)
            return Stacked(stacked, Block, gradient_checkpointing, prevent_cse)

        return fn

    def scan(self, init, *extra_args, **extra_kwargs):
        """
        Scan over the stacked module. This is the same as a for loop that applies each instance of the module in sequence
        to the input, passing the output of one instance to the next instance. It returns a stack of intermediates as
        well as the final output.

        That is, it behaves similarly to the following Python code:

        ```python
        carry = init
        intermediates = []

        for block in self.stacked:
            carry, extra = block(carry)
            intermediates.append(extra)

        return carry, hax.stack(Block, intermediates)
        ```

        Args:
            init:
            *extra_args:
            **extra_kwargs:

        Returns:

        """
        if self.gradient_checkpointing:
            do_block = filter_checkpoint(self._do_block, prevent_cse=self.prevent_cse)
        else:
            do_block = self._do_block
        return haliax.scan(do_block, self.Block)(init, self.stacked, *extra_args, **extra_kwargs)

    def fold(self, init, *args, **kwargs):
        """
        Fold over the stacked module. This is the same as a for loop that applies each instance of the module in sequence
        to the input, passing the output of one instance to the next instance.
        That is, it behaves similarly to the following Python code:

        ```python
        carry = init
        for block in self.stacked:
            carry = block(carry)

        return carry
        ```

        Args:
            init:
            *args:
            **kwargs:

        Returns:

        """
        if self.gradient_checkpointing:
            do_block = filter_checkpoint(self._do_block, prevent_cse=self.prevent_cse)
        else:
            do_block = self._do_block

        return haliax.fold(do_block, self.Block)(init, self.stacked, *args, **kwargs)

    @staticmethod
    def _do_block(carry, block, *extra_args, **extra_kwargs):
        return block(carry, *extra_args, **extra_kwargs)

    # TODO: this is for logic that's in levanter. We should move that logic to haliax I guess?
    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"stacked": None}

    def unstacked(self) -> Sequence[M]:
        """
        Returns the unstacked version of this module. This is useful for logging or saving checkpoints.
        Returns:
            A sequence of modules, one for each element of the stack
        """

        def unbatch_leaf(x):
            if haliax.is_named_array(x):
                if haliax.selects_axis(x.axes, self.Block):
                    return haliax.unbind(x, self.Block)
                else:
                    return tuple(x for _ in range(self.Block.size))
            elif haliax.jax_utils.is_jax_array_like(x):
                assert (
                    x.shape[0] == self.Block.size
                ), f"Expected first dimension to be {self.Block.size}, got {x.shape[0]}"
                return tuple(x[i] for i in range(self.Block.size))
            else:
                return tuple(x for _ in range(self.Block.size))

        leaves, structure = jax.tree_util.tree_flatten(self.stacked, is_leaf=haliax.is_named_array)
        unstacked_leaves = tuple(map(unbatch_leaf, leaves))
        # now we need to transpose the leaves
        unstacked_leaves = tuple(zip(*unstacked_leaves))
        return tuple(map(lambda x: jax.tree_util.tree_unflatten(structure, x), unstacked_leaves))
