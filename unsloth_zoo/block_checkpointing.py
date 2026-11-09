# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Block gradient checkpointing for diffusers, transformers or plain nn.Module models; every_n_blocks follows
transformers' every_n_layers (blocks 0, N, 2N); imports only torch."""

import contextlib
import functools
import inspect
import weakref

import torch
import torch.utils.checkpoint

__all__ = [
    "every_n_checkpoint_function",
    "find_repeated_blocks",
    "enable_block_gradient_checkpointing",
]

_MARK = "_unsloth_block_checkpoint_reentrant"
_ORDER = "_unsloth_block_checkpoint_order"
_INDEX = "_unsloth_block_checkpoint_index"
_FORWARD = "_unsloth_block_checkpoint_forward"
# A custom checkpoint function may not pickle and should not follow a copy: a copy falls back to torch's checkpoint.
_CUSTOM = weakref.WeakKeyDictionary()
# A pre-patched forward stays on the block: it often references the block (accelerate), so a global registry leaks it.
_PREVIOUS = "_unsloth_block_checkpoint_previous_forward"


class _Order:
    # Per model, on every submodule: a copy gets its own counter and keeps its blocks' indices.
    __slots__ = ("next",)

    def __init__(self):
        self.next = 0


def _stamp_order(model):
    order = _Order()
    for module in model.modules():
        module.__dict__.pop(_INDEX, None)
        module.__dict__[_ORDER] = order


def _owner(function):
    while isinstance(function, functools.partial):
        function = function.func
    return getattr(function, "__self__", function)


class _TorchCheckpoint:
    # Looked up per call: Unsloth's LLM shims force reentrant/offloaded recompute, so their stashed pristine checkpoint
    # wins; a class, since diffusers/transformers store it on the model and it must pickle.
    __slots__ = ("use_reentrant",)

    def __init__(self, use_reentrant):
        self.use_reentrant = bool(use_reentrant)

    def __call__(self, function, *args, **kwargs):
        ck = torch.utils.checkpoint
        fn = getattr(ck, "_unsloth_pristine_checkpoint", None) or ck.checkpoint
        if kwargs:
            # Bind block kwargs: reentrant rejects them, and non-reentrant would take debug / context_fn as its own.
            function = functools.partial(function, **kwargs)
        return fn(function, *args, use_reentrant = self.use_reentrant)



class _EveryN:
    __slots__ = ("every_n_blocks", "base", "fallback")

    def __init__(self, every_n_blocks, base):
        self.every_n_blocks, self.base, self.fallback = every_n_blocks, base, _Order()

    def __call__(self, function, *args, **kwargs):
        owner = _owner(function)
        state = getattr(owner, "__dict__", None)
        index = None if state is None else state.get(_INDEX)
        if index is None:
            order = (state or {}).get(_ORDER) or self.fallback
            index, order.next = order.next, order.next + 1
            if state is not None:
                state[_INDEX] = index
        if index % self.every_n_blocks:
            return function(*args, **kwargs)
        return self.base(function, *args, **kwargs)


def every_n_checkpoint_function(every_n_blocks = 1, use_reentrant = False, checkpoint_function = None):
    """Checkpoints every Nth distinct caller in first-call order; counting callers rather than a class list also covers
    blocks like SDXL ResnetBlock2D missing from _repeated_blocks."""
    base = checkpoint_function or _TorchCheckpoint(use_reentrant)
    every_n_blocks = max(1, int(every_n_blocks))
    return base if every_n_blocks == 1 else _EveryN(every_n_blocks, base)


def find_repeated_blocks(model):
    """The repeated blocks: modules whose class is named in _repeated_blocks or _no_split_modules, else the ModuleList holding the most parameters."""
    for attr in ("_repeated_blocks", "_no_split_modules"):
        names = set(getattr(model, attr, None) or ())
        if names:
            blocks = [m for m in model.modules() if type(m).__name__ in names]
            if blocks:
                return blocks
    best, best_count = [], 0
    for module in model.modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) >= 2:
            count = sum(p.numel() for p in module.parameters())
            if count > best_count or not best:
                best, best_count = list(module), count
    return best


def _accepts(method, name):
    try:
        return name in inspect.signature(method).parameters
    except (TypeError, ValueError):
        return False


def _inner_forward(block):
    previous = block.__dict__.get(_PREVIOUS)
    return previous if previous is not None else type(block).forward.__get__(block, type(block))


def _hooks(block, name):
    from torch.nn.modules import module

    return (*getattr(module, f"_global_{name}").items(), *getattr(block, f"_{name}").items())


def _recompute_body(block, inner, *args, **kwargs):
    # Recompute reruns hooks for side effects only; return value ignored (saved inputs carry the rewrite) and only flags
    # are read, since compiled checkpoints must not write.
    if block.__dict__.get(_FORWARD):
        return inner(*args, **kwargs)
    with_kwargs = block._forward_pre_hooks_with_kwargs
    for hook_id, hook in _hooks(block, "forward_pre_hooks"):
        hook(block, args, kwargs) if hook_id in with_kwargs else hook(block, args)
    output = inner(*args, **kwargs)
    from torch.nn.modules import module

    # torch < 2.6 has no global kwargs-hook registry.
    with_kwargs = {**block._forward_hooks_with_kwargs, **getattr(module, "_global_forward_hooks_with_kwargs", {})}
    for hook_id, hook in _hooks(block, "forward_hooks"):
        hook(block, args, kwargs, output) if hook_id in with_kwargs else hook(block, args, output)
    return output


class _CheckpointedForward:
    # Instance ``forward``: an object holding its block, so deepcopy / pickle keep it (a bound method reloads unwrapped).
    __slots__ = ("block",)

    def __init__(self, block):
        self.block = block

    def __call__(self, *args, **kwargs):
        block = self.block
        inner = _inner_forward(block)
        if not (block.training and torch.is_grad_enabled()):
            return inner(*args, **kwargs)
        run = _CUSTOM.get(block) or _TorchCheckpoint(getattr(block, _MARK, False))
        # Eager non-reentrant recompute stops early, before a releasing post-hook; compiled recompute runs no hooks.
        hooked = not torch.compiler.is_compiling() and (
            any(_hooks(block, "forward_pre_hooks")) or any(_hooks(block, "forward_hooks"))
        )
        early_stop = torch.utils.checkpoint.set_checkpoint_early_stop(False) if hooked else contextlib.nullcontext()
        block.__dict__[_FORWARD] = True
        try:
            with early_stop:
                return run(functools.partial(_recompute_body, block, inner), *args, **kwargs)
        finally:
            block.__dict__.pop(_FORWARD, None)


def _is_wrapped(block):
    return isinstance(block.__dict__.get("forward"), _CheckpointedForward)


def _unwrap(block):
    if _is_wrapped(block):
        previous = block.__dict__.pop(_PREVIOUS, None)
        if previous is not None:
            block.forward = previous
        else:
            del block.forward
    _CUSTOM.pop(block, None)
    block.__dict__.pop(_MARK, None)


def _wrap_blocks(model, every_n_blocks, use_reentrant, checkpoint_function):
    # A weight-shared block listed twice is one caller: a later alias must not undo the first one's wrapper.
    blocks = list({id(block): block for block in find_repeated_blocks(model)}.values())
    if not blocks:
        raise ValueError(f"{type(model).__name__} has no repeated blocks to checkpoint.")
    for index, block in enumerate(blocks):
        _unwrap(block)
        if index % every_n_blocks:
            continue
        if "forward" in block.__dict__:
            block.__dict__[_PREVIOUS] = block.__dict__["forward"]
        if checkpoint_function is not None:
            _CUSTOM[block] = checkpoint_function
        setattr(block, _MARK, bool(use_reentrant))
        block.forward = _CheckpointedForward(block)
    return "modules"


def enable_block_gradient_checkpointing(model, every_n_blocks = 1, use_reentrant = False, checkpoint_function = None):
    """Checkpoints repeated blocks and returns the path used (diffusers, transformers or modules); idempotent.
    Non-reentrant by default: supports kwargs and no-grad inputs; reentrant recompute of bnb 4-bit LoRA linears can
    hit an illegal memory access on FLUX."""
    every_n_blocks = max(1, int(every_n_blocks))
    function = every_n_checkpoint_function(every_n_blocks, use_reentrant, checkpoint_function)
    _stamp_order(model)
    enable = getattr(model, "enable_gradient_checkpointing", None)
    if callable(enable) and _accepts(enable, "gradient_checkpointing_func"):
        enable(gradient_checkpointing_func = function)
        return "diffusers"
    setter = getattr(model, "_set_gradient_checkpointing", None)
    if callable(getattr(model, "gradient_checkpointing_enable", None)) and callable(setter) and _accepts(
        setter, "gradient_checkpointing_func"
    ):
        # Enable once (it wires PEFT input grads; repeats stack embedding hooks), then the setter installs our function.
        if getattr(model, "is_gradient_checkpointing", False) is not True:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": use_reentrant})
        setter(enable = True, gradient_checkpointing_func = function)
        return "transformers"
    return _wrap_blocks(model, every_n_blocks, use_reentrant, checkpoint_function)
