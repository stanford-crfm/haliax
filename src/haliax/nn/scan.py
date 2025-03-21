import dataclasses
import functools
import re
import warnings
from typing import Any, Dict, Generic, Literal, Optional, Protocol, Sequence, Type, TypeVar, Union, cast

import equinox as eqx
import jax
from jax import numpy as jnp

import haliax
import haliax.util
from haliax.jax_utils import tree_checkpoint_name
from haliax.util import is_jax_or_hax_array_like

from .._src.state_dict import ModuleWithStateDictSerialization, StateDict, with_prefix
from ..axis import Axis


M = TypeVar("M", bound=eqx.Module)
M_co = TypeVar("M_co", bound=eqx.Module, covariant=True)
S = TypeVar("S", bound=eqx.Module)
T = TypeVar("T")


class ModuleInit(Protocol[M_co]):
    def __call__(self, *args, **kwargs) -> M_co:
        ...


@dataclasses.dataclass
class ScanCheckpointPolicy:
    """
    A class that represents a gradient checkpoint policy for blocks in a Stacked module. This is used to control
    gradient checkpointing in [haliax.nn.Stacked][] and [haliax.nn.BlockSeq][].

    Gradient checkpointing is a technique for reducing memory usage in training large models. It works by saving only a
    subset of the forward pass and recomputing the rest in the backward pass. (By doing parts of the forward pass again)
    JAX suggests that this usually isn't necessary when not using scan-over-layers (i.e. Stacked), so this is mostly
    useful for Stacked modules.

    A scan block takes a "carry" and some extra arguments, and returns a "carry" and an "output". The "carry" is passed
    to the next block, and the "output" is concatenated into a final result (sort of like an RNN).

    Schematically it might look like this:

    ```
          I       I       I       I
          |       |       |       |
    C ->  B -C->  B -C->  B -C->  B --> C
          |       |       |       |
          O       O       O       O
    ```

    where "C" is the carry and "O" is the output. A block will typically do some computation (e.g. a Transformer block)
    as well, which might require saving or recomputing in the backward pass.

    Imagine we save just the carries, then during the backward pass, we can recompute the outputs using the carries
    and the inputs (and the blocks), and then compute the gradient as usual. This requires O(N) memory and O(N) time,
    where N is the number of blocks. This is the default behavior in Haliax and works well for most models.

    Alternatively, we could only save the initial and final carry. (This corresponds to
    `StackedCheckpointPolicy(save_carries=False, save_outputs=False)` or `"recompute"`)
    Then, during the backward pass, for each block we
    can compute all blocks up to that point (to get its input carry) and then compute the block itself.
    This requires O(1) memory and O(N^2) time.

    Intermediate approaches exist (including O(sqrt(N)) memory and O(N) time), but we don't support them yet.

    Another choice is to "offload" carries and outputs to the host, which can reduce memory usage on the device.
    We support offloading carries and outputs to the host, but not internals.

    See Also:
        * [JAX docs on gradient checkpointing](https://docs.jax.dev/en/latest/gradient-checkpointing.html)
    """

    save_carries: bool | Literal["offload"] = True
    """
    Whether to save all carries in the forward pass. If True, carries are saved in the forward pass and used in the
    backward pass. If "offload", carries are saved in the forward pass and offloaded to the host
    """
    save_outputs: bool | Literal["offload"] = True
    """
    Whether to save scan outputs in the forward pass. If True, outputs are saved in the forward pass and
    used in the backward pass. If "offload", outputs are saved in the forward pass and offloaded to the host
    """
    save_block_internals: bool | list[str] = True
    """
    Whether to save internal state of blocks. If a list, only the listed names are saved, as
    with [jax.checkpoint_policies.save_only_these_names][].

    See Also: https://docs.jax.dev/en/latest/gradient-checkpointing.html#custom-policies-for-offload
    """
    prevent_cse: bool = False
    """
    Whether to prevent common subexpression elimination in the checkpointed function.
    """

    disable: bool = False
    """
    Whether to disable gradient checkpointing entirely. This is useful for debugging.
    """

    simple: bool = False
    """
    Whether to use the simple gradient checkpointing policy. This is useful for debugging.
    """

    nested_remat: bool | int = False
    """
    Allows for nested remat with a double scan. We reshape the stack into [nested_remat, -1] and then scan over both
    in sequence. If True, we find the closest int to sqrt(len(stack)) such that len(stack) % int == 0.
    If False, we don't do anything.
    """

    @property
    def is_save_nothing(self):
        return self.save_carries is False and self.save_outputs is False and self.save_block_internals is False

    @staticmethod
    def from_bool_or_str(remat_policy: bool | str):
        """
        Convert a boolean or string into a BlockCheckpointPolicy. This is useful for converting user input
        into a BlockCheckpointPolicy.

        Choices:
            * True: save outputs, don't save block internals. This is the classic Haliax behavior.
            * False: save everything.
            * "offload": offload outputs to the host, don't save block internals.
            * "recompute" or "full": don't save outputs or block internals.
            * "save_all": save outputs and block internals. Equivalent to False
        """
        if remat_policy == "offload":
            return ScanCheckpointPolicy(save_carries="offload", save_outputs="offload", save_block_internals=False)
        elif remat_policy == "recompute" or remat_policy == "full":
            return ScanCheckpointPolicy(save_carries=False, save_outputs=False, save_block_internals=False)
        elif remat_policy == "save_all":
            return ScanCheckpointPolicy(save_carries=True, save_outputs=True, save_block_internals=True)
        elif remat_policy is True:
            # return StackedCheckpointPolicy(save_carries=True, save_outputs=True, save_block_internals=False)
            return ScanCheckpointPolicy(simple=True)
        elif remat_policy is False:
            return ScanCheckpointPolicy(save_carries=True, save_outputs=True, save_block_internals=True)
        else:
            raise ValueError(f"Invalid checkpoint policy {remat_policy}")

    @staticmethod
    def _mk(remat_policy: Union[bool, str, "ScanCheckpointPolicy"]) -> "ScanCheckpointPolicy":
        if isinstance(remat_policy, ScanCheckpointPolicy):
            return remat_policy
        else:
            return ScanCheckpointPolicy.from_bool_or_str(remat_policy)

    def checkpoint(self, carry_name: str, output_name: str, callable):
        if self.disable:
            return callable
        policy = self._to_jax_policy(carry_name, output_name)
        if policy is None:
            return callable
        else:
            return eqx.filter_checkpoint(callable, policy=policy, prevent_cse=self.prevent_cse)

    def _to_jax_policy(self, carry_name: str, output_name: str):
        assert isinstance(carry_name, str)
        assert isinstance(output_name, str)
        our_names_to_save = []
        our_names_to_offload = []
        our_names_to_remat = []

        # return jax.checkpoint_policies.save_only_these_names(carry_name, output_name)

        if self.save_outputs is True:
            our_names_to_save.append(output_name)
        elif self.save_outputs == "offload":
            our_names_to_offload.append(output_name)
        else:
            assert self.save_outputs is False, f"Invalid save_outputs {self.save_outputs}"
            our_names_to_remat.append(output_name)

        if self.save_carries is True:
            our_names_to_save.append(carry_name)
        elif self.save_carries == "offload":
            our_names_to_offload.append(carry_name)
        else:
            assert self.save_carries is False, f"Invalid save_carries {self.save_carries}"
            our_names_to_remat.append(carry_name)

        if isinstance(self.save_block_internals, Sequence):
            our_names_to_save.extend(self.save_block_internals)

        if len(our_names_to_offload) > 0:
            if self.save_block_internals is True:
                raise ValueError("Can't save all block internals and offload outputs. Use a list of names instead.")

            return jax.checkpoint_policies.save_and_offload_only_these_names(
                names_which_can_be_saved=our_names_to_save,
                names_which_can_be_offloaded=our_names_to_offload,
                offload_src="device",
                offload_dst="pinned_host",
            )
        else:
            if len(our_names_to_remat) > 0:
                if self.save_block_internals is True:
                    p1 = jax.checkpoint_policies.save_anything_except_these_names(*our_names_to_remat)
                    if len(our_names_to_save) > 0:
                        p2 = jax.checkpoint_policies.save_only_these_names(*our_names_to_save)
                        return jax.checkpoint_policies.save_from_both_policies(p1, p2)
                    else:
                        return p1
                else:
                    return jax.checkpoint_policies.save_only_these_names(*our_names_to_save)
            elif len(our_names_to_save) > 0:
                p1 = jax.checkpoint_policies.save_only_these_names(*our_names_to_save)
                if self.save_block_internals is True:
                    p2 = jax.checkpoint_policies.save_anything_except_these_names(*our_names_to_remat)
                    return jax.checkpoint_policies.save_from_both_policies(p1, p2)
                else:
                    return p1
            elif self.save_block_internals is True:
                return jax.checkpoint_policies.save_anything_except_these_names(*our_names_to_remat)
            else:
                return None


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
        cls: Type[S],
        Block: Axis,
        module: Type[M],
        *,
        gradient_checkpointing: bool | ScanCheckpointPolicy = False,
        prevent_cse: bool = False,
    ) -> ModuleInit[S]:
        ...

    def scan(self, init: T, *extra_args, **extra_kwargs):
        ...

    def fold(self, init: T, *args, **kwargs) -> T:
        ...

    def unstacked(self) -> Sequence[M]:
        """
        Returns the unstacked version of this module. This is useful for logging or saving checkpoints.

        """
        ...


class BlockSeq(ModuleWithStateDictSerialization, Generic[M]):
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
    gradient_checkpointing: ScanCheckpointPolicy = eqx.static_field()

    @classmethod
    def init(
        cls: Type[S],
        Block: Axis,
        module: Type[M],
        *,
        gradient_checkpointing: bool | ScanCheckpointPolicy = False,
        prevent_cse: bool | None = None,
    ) -> ModuleInit[S]:
        """
        This is a curried init method that takes the Block and module and returns a function that takes
        the arguments to the module's init method. Any NamedArrays in the arguments will be sliced along the
        Block axis (if it exists). JAX arrays will be sliced along the first axis.
        """

        gradient_checkpointing = ScanCheckpointPolicy._mk(gradient_checkpointing)

        if prevent_cse is not None:
            warnings.warn(
                "The prevent_cse argument is deprecated and will be removed in a future version of Haliax. Use the"
                " StackedCheckpointPolicy instead.",
                DeprecationWarning,
            )
            gradient_checkpointing = dataclasses.replace(gradient_checkpointing, prevent_cse=prevent_cse)

        @functools.wraps(module)
        def fn(*args, **kwargs):
            # The only complexity here is that the args and kwargs might have a Block axis in them,
            # in which case we need to loop over them to slice them out.

            def init_block(i):
                (block_args, block_kwargs) = haliax.tree_util.tree_map(
                    functools.partial(BlockSeq._slice_out, Block, i), (args, kwargs)
                )
                return module.init(*block_args, **block_kwargs)

            seq = [init_block(i) for i in range(Block.size)]

            return BlockSeq(seq, Block, gradient_checkpointing)

        return fn

    def scan(self, init: T, *extra_args, **extra_kwargs):
        def do_scan(init, *extra_args, **extra_kwargs):
            out = []
            carry = init

            for i, block in enumerate(self.blocks):

                (block_args, block_kwargs) = haliax.tree_util.tree_map(
                    functools.partial(BlockSeq._slice_out, self.Block, i), (extra_args, extra_kwargs)
                )

                block_result = block(carry, *block_args, **block_kwargs)

                if not isinstance(block_result, (tuple, list)) or len(block_result) != 2:
                    raise ValueError(
                        f"BlockSeq.scan expects the block to return a pair of (carry, extra), got {block_result}"
                    )

                carry, extra = block_result

                carry = tree_checkpoint_name(carry, self._carry_ckpt_name)
                extra = tree_checkpoint_name(extra, self._output_ckpt_name)

                out.append(extra)

            return carry, haliax.tree_util.tree_map(lambda *x: haliax.stack(self.Block, x), *out)

        do_scan = self.gradient_checkpointing.checkpoint(self._carry_ckpt_name, self._output_ckpt_name, do_scan)

        return do_scan(init, *extra_args, **extra_kwargs)

    def fold(self, init: T, *args, **kwargs) -> T:
        def do_fold(init, *args, **kwargs):
            carry = init
            for i, block in enumerate(self.blocks):
                (block_args, block_kwargs) = haliax.tree_util.tree_map(
                    functools.partial(BlockSeq._slice_out, self.Block, i), (args, kwargs)
                )
                carry = block(carry, *block_args, **block_kwargs)
                carry = tree_checkpoint_name(carry, self._carry_ckpt_name)
            return carry

        do_fold = self.gradient_checkpointing.checkpoint(self._carry_ckpt_name, self._output_ckpt_name, do_fold)

        return do_fold(init, *args, **kwargs)

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

    def from_state_dict(self: M, state_dict: StateDict, prefix: Optional[str] = None) -> M:
        out_blocks = []
        for i, block in enumerate(self.blocks):
            my_prefix = with_prefix(prefix, str(i))
            block = block.from_state_dict(state_dict, my_prefix)
            out_blocks.append(block)

        return eqx.tree_at(lambda m: m.blocks, self, out_blocks)

    def to_state_dict(self, prefix: Optional[str] = None) -> StateDict:
        """
        Returns the unstacked format of the module, which is compatible with torch.nn.Sequential, with keys of the form (...). The stacked/vectorized format is required for haliax.nn.Stacked and vectorizes all such tensors into a single shared key.".
        """
        state_dict: StateDict = {}
        for i, block in enumerate(self.blocks):
            my_prefix = with_prefix(prefix, str(i))
            # we can't assume to_state_dict is implemented, so we have to do it manually
            block_dict = haliax.state_dict.to_state_dict(block, my_prefix)
            state_dict.update(block_dict)

        return state_dict

    @property
    def _output_ckpt_name(self):
        return f"BlockSeq[{self.Block}, {self.blocks[0].__class__.__name__}].outputs"

    @property
    def _carry_ckpt_name(self):
        return f"BlockSeq[{self.Block}, {self.blocks[0].__class__.__name__}].carry"


class Stacked(ModuleWithStateDictSerialization, Generic[M]):
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
    output, while "scan" is the same as a for loop that accumulates a list of outputs as well as the final output.

    Stacked also supports gradient checkpointing, which is useful for very large models that don't fit in memory.

    Typically only one of "fold" or "scan" can be used with a given Stacked module, depending on the what the module
    returns: if the module returns a single output, use "fold"; if the module returns a sequence of outputs and
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
    gradient_checkpointing: ScanCheckpointPolicy = eqx.static_field()

    @classmethod
    def init(
        cls,
        Block: Axis,
        module: Type[M],
        *,
        gradient_checkpointing: bool | ScanCheckpointPolicy | str = False,
        prevent_cse: bool | None = None,
    ) -> ModuleInit["Stacked[M]"]:
        """
        Initialize a Stacked module. This method is curried: you can pass in the Block and module, and it will return
        a function that takes (batched) arguments to the vmapped module's init method.

        Args:
            Block: The axis that will be stacked over. This is typically a "layer" axis, but could be any axis.
            module: The module that will be stacked. This module must take a batched input and return a batched output.
            gradient_checkpointing: Whether to use gradient checkpointing. If True, uses the default policy. If a string,
                uses the policy specified by the string. If a StackedCheckpointPolicy, uses that policy.
            prevent_cse: Whether to prevent common subexpression elimination in the checkpointed function. This is useful
                for debugging, but may slow down the function.
        """

        gradient_checkpointing = ScanCheckpointPolicy._mk(gradient_checkpointing)

        if prevent_cse is not None:
            warnings.warn(
                "The prevent_cse argument is deprecated and will be removed in a future version of Haliax. Use the"
                " StackedCheckpointPolicy instead.",
                DeprecationWarning,
            )

            gradient_checkpointing = dataclasses.replace(gradient_checkpointing, prevent_cse=prevent_cse)

        @functools.wraps(module)
        def fn(*args, **kwargs):
            stacked = haliax.vmap(module.init, Block)(*args, **kwargs)
            return Stacked(stacked, Block, gradient_checkpointing)

        return fn

    def scan(self, init, *extra_args, **extra_kwargs):
        """
        Scan over the stacked module. This is the same as a for loop that applies each instance of the module in sequence
        to the input, passing the output of one instance to the next instance. It returns a stack of outputs as
        well as the final output.

        That is, it behaves similarly to the following Python code:

        ```python
        carry = init
        outputs = []

        for block in self.stacked:
            carry, extra = block(carry)
            outputs.append(extra)

        return carry, hax.stack(Block, outputs)
        ```

        Args:
            init:
            *extra_args:
            **extra_kwargs:

        Returns:

        """
        carry_name = self._carry_ckpt_name
        output_name = self._output_ckpt_name

        def do_block(carry, block, *args, **kwargs):
            carry = tree_checkpoint_name(carry, carry_name)
            carry, out = block(carry, *args, **kwargs)
            out = tree_checkpoint_name(out, output_name)
            return carry, out

        def do_scan(init, *extra_args, **extra_kwargs):
            carry, out = haliax.scan(do_block, self.Block)(init, self.stacked, *extra_args, **extra_kwargs)
            return carry, out

        do_scan = self.gradient_checkpointing.checkpoint(carry_name, output_name, do_scan)

        return do_scan(init, *extra_args, **extra_kwargs)

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
            init: The initial value of carry to pass to the first block
            *args: Extra arguments to pass to the blocks. These are passed directly to the blocks
            **kwargs: Extra keyword arguments to pass to the blocks. These are passed directly to the blocks

        Returns:

        """
        carry_name = self._carry_ckpt_name

        def do_block(carry, block, *args, **kwargs):
            carry = tree_checkpoint_name(carry, carry_name)
            carry = block(carry, *args, **kwargs)
            return carry

        if self.gradient_checkpointing.is_save_nothing:
            pass
        elif self.gradient_checkpointing.simple:
            do_block = jax.checkpoint(do_block, prevent_cse=self.gradient_checkpointing.prevent_cse)
        else:
            do_block = self.gradient_checkpointing.checkpoint(carry_name, self._output_ckpt_name, do_block)

        def do_fold(init, *extra_args, **extra_kwargs):
            # if self.gradient_checkpointing.simple:
            carry = haliax.fold(do_block, self.Block)(init, self.stacked, *extra_args, **extra_kwargs)
            # else:
            #     carry = haliax.fold(do_block, self.Block)(init, self.stacked, *extra_args, **extra_kwargs)
            return carry

        if self.gradient_checkpointing.is_save_nothing:
            do_fold = jax.checkpoint(do_fold, prevent_cse=self.gradient_checkpointing.prevent_cse)

        # if not self.gradient_checkpointing.simple:
        #     do_fold = self.gradient_checkpointing.checkpoint(carry_name, self._output_ckpt_name, do_fold)

        return do_fold(init, *args, **kwargs)

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
            if isinstance(x, haliax.core.NamedArray):
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

    def to_state_dict(self, prefix: Optional[str] = None) -> StateDict:
        # this method needs to "devectorize" the blocks, so that we have a list of blocks h.0.FOO, h.1.FOO, etc.
        # first just do the normal thing with our own dict, which we'll post-process
        state_dict: StateDict = super().to_state_dict(prefix)

        return _unstack_state_dict(state_dict, prefix)

    def from_state_dict(self: M, state_dict: StateDict, prefix: Optional[str] = None) -> M:
        # this method needs to "vectorize" the blocks, so that we have a single block h.FOO
        # first just do the normal thing with our own dict, which we'll post-process
        stacked = _stack_state_dict(state_dict, prefix=prefix)
        out = super().from_state_dict(stacked, prefix=prefix)  # type: ignore
        return out

    @property
    def _carry_ckpt_name(self):
        # return f"Stacked[{self.Block}, {self.stacked.__class__.__name__}].carry"
        return "carry"

    @property
    def _output_ckpt_name(self):
        # return f"Stacked[{self.Block}, {self.stacked.__class__.__name__}].outputs"
        return "outputs"


def _stack_state_dict(state_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
    """
    Stack all keys matching prefix in a new state dict, returning a state dict that has all keys matching
    prefix stacked, but otherwise the same.

    Stacked in this case means roughly "compatible with a torch.nn.Sequential", which means that the
    keys are of the form "<prefix>.0.<key>", "<prefix>.1.<key>", etc.

    Mostly for use with [haliax.nn.Stacked][].
    """
    vectorized_dict: StateDict = {}

    tensors_to_vectorize: dict[str, list[Optional[Any]]] = {}
    if prefix is not None:
        prefix_for_pat = re.escape(prefix + ".")
    else:
        prefix_for_pat = ""
    pattern = re.compile(rf"{prefix_for_pat}(\d+)\.(.*)")

    for k, v in state_dict.items():
        match = pattern.match(k)
        if match:
            block_idx = int(match.group(1))
            block_key = match.group(2)
            tensors = tensors_to_vectorize.setdefault(block_key, [])
            if len(tensors) <= block_idx:
                tensors.extend([None] * (block_idx - len(tensors) + 1))
            assert tensors[block_idx] is None, f"Duplicate key {k}"
            tensors[block_idx] = v
        else:
            vectorized_dict[k] = v

    # now we have to vectorize the tensors
    for k, tensors in tensors_to_vectorize.items():
        vectorized_dict[cast(str, with_prefix(prefix, k))] = jnp.stack(tensors, axis=0)

    return vectorized_dict


def _unstack_state_dict(state_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
    """
    Unstack all keys matching prefix in a new state dict, returning a state dict that has all keys matching
    prefix unstacked, but otherwise the same. Mostly for use with [haliax.nn.Stacked][].

    Unstacked in this case means roughly "compatible with a torch.nn.Sequential", which means that the
    keys are of the form "<prefix>.0.<key>", "<prefix>.1.<key>", etc.
    """
    new_dict: StateDict = {}
    prefix = with_prefix(prefix, "")
    assert prefix is not None

    for k, v in state_dict.items():
        if k.startswith(prefix) and is_jax_or_hax_array_like(v):
            for i, v_i in enumerate(v):
                new_dict[f"{prefix}{i}.{k[len(prefix):]}"] = v_i
        else:
            new_dict[k] = v

    return new_dict
