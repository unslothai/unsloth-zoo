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

"""Block gradient checkpointing for any model: diffusers ModelMixin, transformers PreTrainedModel, or a plain nn.Module.

``every_n_blocks`` follows transformers' ``every_n_layers``: 1 checkpoints every block, N checkpoints blocks 0, N, 2N...
(fewer recomputes, more activation memory). Only torch is imported, so the file also loads on its own (no package init).
"""

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
# Per-block extras that cannot be pickled or should not follow a copy: a custom checkpoint function and a forward that
# was already patched on the instance. A copy falls back to torch's checkpoint and its class forward.
_CUSTOM = weakref.WeakKeyDictionary()
_PREVIOUS = weakref.WeakKeyDictionary()


def _owner(function):
    # diffusers passes the module, transformers' GradientCheckpointingLayer a partial of its bound __call__.
    while isinstance(function, functools.partial):
        function = function.func
    return getattr(function, "__self__", function)


def _torch_checkpoint(use_reentrant):
    # torch's own checkpoint, looked up per call. Unsloth's LLM shims (patch_gradient_checkpointing and friends) force
    # reentrant / offloaded recompute, so the pristine function they stash wins over whatever is patched in now.
    def run(function, *args, **kwargs):
        ck = torch.utils.checkpoint
        fn = getattr(ck, "_unsloth_pristine_checkpoint", None) or ck.checkpoint
        if kwargs:
            # Bind block kwargs: reentrant rejects them, and non-reentrant would take debug / context_fn as its own.
            function = functools.partial(function, **kwargs)
        return fn(function, *args, use_reentrant = use_reentrant)

    return run


def every_n_checkpoint_function(every_n_blocks = 1, use_reentrant = False, checkpoint_function = None):
    """A ``gradient_checkpointing_func`` that checkpoints every Nth distinct caller in first-call (forward) order.

    Counting callers rather than a class list makes it exact wherever a model routes the function: the SDXL UNet
    also checkpoints ResnetBlock2D, which its ``_repeated_blocks`` (BasicTransformerBlock only) does not list."""
    base = checkpoint_function or _torch_checkpoint(use_reentrant)
    every_n_blocks = max(1, int(every_n_blocks))
    if every_n_blocks == 1:
        return base
    order = {}

    def every_n(function, *args, **kwargs):
        index = order.setdefault(id(_owner(function)), len(order))
        if index % every_n_blocks:
            return function(*args, **kwargs)
        return base(function, *args, **kwargs)

    return every_n


def find_repeated_blocks(model):
    """The model's repeated blocks: its declared ``_repeated_blocks`` / ``_no_split_modules`` classes, else the
    ModuleList holding the most parameters."""
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
            if count > best_count:
                best, best_count = list(module), count
    return best


def _accepts(method, name):
    try:
        return name in inspect.signature(method).parameters
    except (TypeError, ValueError):
        return False


def _inner_forward(block):
    previous = _PREVIOUS.get(block)
    return previous if previous is not None else type(block).forward.__get__(block, type(block))


class _CheckpointedForward:
    # A block's instance ``forward``. A plain object holding its block: deepcopy and pickle carry it along with the
    # block (a bound method pickles as getattr(block, "forward"), which reloads as the unwrapped class forward).
    __slots__ = ("block",)

    def __init__(self, block):
        self.block = block

    def __call__(self, *args, **kwargs):
        block = self.block
        inner = _inner_forward(block)
        if block.training and torch.is_grad_enabled():
            run = _CUSTOM.get(block) or _torch_checkpoint(getattr(block, _MARK, False))
            return run(inner, *args, **kwargs)
        return inner(*args, **kwargs)


def _is_wrapped(block):
    return isinstance(block.__dict__.get("forward"), _CheckpointedForward)


def _unwrap(block):
    if _is_wrapped(block):
        previous = _PREVIOUS.pop(block, None)
        if previous is not None:
            block.forward = previous
        else:
            del block.forward
    _CUSTOM.pop(block, None)
    block.__dict__.pop(_MARK, None)


def _wrap_blocks(model, every_n_blocks, use_reentrant, checkpoint_function):
    blocks = find_repeated_blocks(model)
    if not blocks:
        raise ValueError(f"{type(model).__name__} has no repeated blocks to checkpoint.")
    for index, block in enumerate(blocks):
        _unwrap(block)
        if index % every_n_blocks:
            continue
        if "forward" in block.__dict__:
            _PREVIOUS[block] = block.__dict__["forward"]
        if checkpoint_function is not None:
            _CUSTOM[block] = checkpoint_function
        setattr(block, _MARK, bool(use_reentrant))
        block.forward = _CheckpointedForward(block)
    return "modules"


def enable_block_gradient_checkpointing(model, every_n_blocks = 1, use_reentrant = False, checkpoint_function = None):
    """Turn on gradient checkpointing for ``model``'s repeated blocks; returns the path used ("diffusers",
    "transformers" or "modules"). Idempotent: a second call replaces the first one's setting.

    Non-reentrant by default: it supports keyword arguments and inputs that do not require grad (frozen bases under
    LoRA), and reentrant recompute of a bnb 4-bit LoRA linear can hit an illegal memory access on FLUX."""
    every_n_blocks = max(1, int(every_n_blocks))
    function = every_n_checkpoint_function(every_n_blocks, use_reentrant, checkpoint_function)
    enable = getattr(model, "enable_gradient_checkpointing", None)
    if callable(enable) and _accepts(enable, "gradient_checkpointing_func"):
        enable(gradient_checkpointing_func = function)
        return "diffusers"
    setter = getattr(model, "_set_gradient_checkpointing", None)
    if callable(getattr(model, "gradient_checkpointing_enable", None)) and callable(setter) and _accepts(
        setter, "gradient_checkpointing_func"
    ):
        # gradient_checkpointing_enable also wires input grads for PEFT; the setter then installs our function.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": use_reentrant})
        setter(enable = True, gradient_checkpointing_func = function)
        return "transformers"
    return _wrap_blocks(model, every_n_blocks, use_reentrant, checkpoint_function)
