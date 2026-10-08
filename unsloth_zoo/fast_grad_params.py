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
"""Clip and zero only the parameters holding a grad, without walking the module tree per step.

HF Trainer calls `accelerator.clip_grad_norm_(model.parameters(), ...)` and `model.zero_grad()`
every optimizer step. Both walk every module: a PEFT Qwen3-30B-A3B (217,593 modules, 56,115
params) spends ~0.25 s per walk. torch's clip_grad_norm_ and Module.zero_grad only touch params
whose `.grad is not None`, in parameters() order, so filtering a cached `tuple(parameters())` by
`.grad is not None` at call time is bitwise identical, including frozen params that still hold a
grad and params unfrozen mid-run.

The tuple is cached per module against a process-wide generation, bumped by torch's global
parameter / module registration hooks when the target module was in a cached tree (adapter add,
register_parameter, setattr of a Parameter or Module; not the throwaway ModuleList that
`self.layers[:n]` builds every forward), by every Trainer.train() call, every
Accelerator.prepare_model and by bump() (direct `_parameters` edits, e.g. stacked expert LoRA).
Only flagged models (enable_fast_grad_params,
called by the Unsloth loaders) take the fast path; clipping only for plain / DDP Accelerators
(FSDP, DeepSpeed, XLA, Megatron and TP / CP / SP keep Accelerate's own path).

UNSLOTH_FAST_GRAD_PARAMS=0 turns it off (read at every call).
"""
import functools
import inspect
import os
import warnings

import torch

__all__ = ["enable_fast_grad_params", "bump", "FAST_GRAD_CALLS"]

_FLAG = "_unsloth_fast_grad_params"
_CACHE = "_unsloth_flat_params"
_TRACKED = "_unsloth_fast_grad_tracked"   # set on every module of a cached tree
_GEN = [0]
# Engagement counters: fast zero_grad / clip calls and tuple rebuilds.
FAST_GRAD_CALLS = {"zero_grad": 0, "clip": 0, "rebuild": 0}
_INSTALLED = [False]
_PARAMETERS_CODE = torch.nn.Module.parameters.__code__


def _enabled():
    return os.environ.get("UNSLOTH_FAST_GRAD_PARAMS", "1") != "0"


def bump(*args, **kwargs):
    """Invalidate every cached parameter tuple."""
    _GEN[0] += 1


def _registration_hook(module, name, value):
    # Only a module inside some cached tree can change that tree's parameters.
    if module.__dict__.get(_TRACKED, False):
        _GEN[0] += 1


def _flat_params(module):
    """tuple(module.parameters()) (same walk, dedup and order: torch's _named_members), cached
    until the next bump. Marks every module of the tree for _registration_hook."""
    d = module.__dict__
    hit = d.get(_CACHE)
    if hit is not None and hit[0] == _GEN[0]:
        return hit[1]
    memo = set()
    flat = []
    for m in module.modules():
        md = m.__dict__
        md[_TRACKED] = True
        for p in md["_parameters"].values():
            if p is not None and p not in memo:
                memo.add(p)
                flat.append(p)
    flat = tuple(flat)
    d[_CACHE] = (_GEN[0], flat)
    FAST_GRAD_CALLS["rebuild"] += 1
    return flat


def _with_grad(module):
    return [p for p in _flat_params(module) if p.grad is not None]


def _zero_grad(self, set_to_none: bool = True) -> None:
    # torch's Module.zero_grad over the cached selection
    if not _enabled():
        return type(self).zero_grad(self, set_to_none)
    if getattr(self, "_is_replica", False):
        warnings.warn(
            "Calling .zero_grad() from a module created with nn.DataParallel() has no effect. "
            "The parameters are copied (in a differentiable manner) from the original module. "
            "This means they are not leaf nodes in autograd and so don't accumulate gradients. "
            "If you need gradients in your forward method, consider using autograd.grad instead.",
            stacklevel = 2,
        )
    FAST_GRAD_CALLS["zero_grad"] += 1
    for p in _with_grad(self):
        if set_to_none:
            p.grad = None
        else:
            if p.grad.grad_fn is not None:
                p.grad.detach_()
            else:
                p.grad.requires_grad_(False)
            p.grad.zero_()


# A bound method pickles as getattr(obj, __name__): unpickling then binds torch's zero_grad.
_zero_grad.__name__ = _zero_grad.__qualname__ = "zero_grad"


def _flag(module):
    """Mark `module` (the exact object Trainer calls) for the fast path. Declines a class that
    overrides zero_grad or the parameter walk."""
    cls, M = type(module), torch.nn.Module
    for name in ("zero_grad", "parameters", "named_parameters", "_named_members", "named_modules", "modules"):
        if getattr(cls, name, None) is not getattr(M, name):
            return False
    d = module.__dict__
    if not d.get(_FLAG, False):
        d[_FLAG] = True
        d["zero_grad"] = _zero_grad.__get__(module)
    return True


def _is_flagged(module):
    return isinstance(module, torch.nn.Module) and module.__dict__.get(_FLAG, False)


def _offloaded(model):
    for m in (model, getattr(model, "base_model", None), getattr(getattr(model, "base_model", None), "model", None)):
        dm = getattr(m, "hf_device_map", None) if m is not None else None
        if isinstance(dm, dict) and any(str(v) in ("cpu", "disk") for v in dm.values()):
            return True   # accelerate's offload hooks swap _parameters entries every forward
    return False


def _accelerator_ok(acc):
    try:
        from accelerate.utils import DistributedType
        if acc.distributed_type not in (DistributedType.NO, DistributedType.MULTI_GPU):
            return False
        state = getattr(acc, "state", None)
        if getattr(acc, "is_fsdp2", False) or getattr(state, "torch_tp_plugin", None) is not None:
            return False
        pc = getattr(acc, "parallelism_config", None)
        if pc is not None and any(getattr(pc, k, False) for k in ("tp_enabled", "cp_enabled", "sp_enabled", "dp_shard_enabled")):
            return False
        if os.environ.get("ACCELERATE_USE_FSDP", "false").lower() == "true":
            return False
        return True
    except Exception:
        return False


def _module_of(parameters):
    """The module of an unstarted `module.parameters()` (recurse=True) generator, else None."""
    if not inspect.isgenerator(parameters) or parameters.gi_code is not _PARAMETERS_CODE:
        return None
    if inspect.getgeneratorstate(parameters) != inspect.GEN_CREATED:
        return None
    loc = inspect.getgeneratorlocals(parameters)
    m = loc.get("self")
    if loc.get("recurse", None) is not True:
        return None
    if _is_flagged(m) or (isinstance(m, torch.nn.parallel.DistributedDataParallel) and _is_flagged(m.module)):
        return m   # the cache is keyed on `m` itself, so its own parameters() order
    return None


def _wrap_accelerator():
    try:
        from accelerate import Accelerator
    except Exception:
        return
    clip = Accelerator.clip_grad_norm_
    if not getattr(clip, "_unsloth_fast_grad", False):
        @functools.wraps(clip)
        def clip_grad_norm_(self, parameters, max_norm, norm_type = 2):
            if _enabled():
                m = _module_of(parameters)
                if m is not None and _accelerator_ok(self):
                    FAST_GRAD_CALLS["clip"] += 1
                    parameters = _with_grad(m)
            return clip(self, parameters, max_norm, norm_type)
        clip_grad_norm_._unsloth_fast_grad = True
        Accelerator.clip_grad_norm_ = clip_grad_norm_

    prep = getattr(Accelerator, "prepare_model", None)
    if prep is not None and not getattr(prep, "_unsloth_fast_grad", False):
        @functools.wraps(prep)
        def prepare_model(self, model, *args, **kwargs):
            out = prep(self, model, *args, **kwargs)
            bump()
            try:   # DDP: Trainer then calls the wrapper's zero_grad / parameters()
                inner = getattr(out, "module", None)
                if out is not model and isinstance(out, torch.nn.parallel.DistributedDataParallel) and _is_flagged(inner):
                    _flag(out)
            except Exception:
                pass
            return out
        prepare_model._unsloth_fast_grad = True
        Accelerator.prepare_model = prepare_model


def _wrap_trainer():
    try:
        from transformers import Trainer
    except Exception:
        return
    train = Trainer.train
    if getattr(train, "_unsloth_fast_grad", False):
        return

    @functools.wraps(train)
    def wrapped(self, *args, **kwargs):
        bump()   # every train() rebuilds the tuples once
        return train(self, *args, **kwargs)

    wrapped._unsloth_fast_grad = True
    Trainer.train = wrapped


def _install():
    if _INSTALLED[0]:
        return True
    mod = torch.nn.modules.module
    p_hook = getattr(mod, "register_module_parameter_registration_hook", None)
    m_hook = getattr(mod, "register_module_module_registration_hook", None)
    if p_hook is None or m_hook is None:
        return False
    p_hook(_registration_hook)
    m_hook(_registration_hook)
    _wrap_accelerator()
    _wrap_trainer()
    _INSTALLED[0] = True
    return True


def enable_fast_grad_params(model):
    """Flag `model` so Trainer's per-step clip / zero_grad skip the module-tree walk. Idempotent;
    returns True if flagged. Never raises."""
    try:
        if not _enabled() or not isinstance(model, torch.nn.Module) or _offloaded(model):
            return False
        if not _install():
            return False
        bump()
        return _flag(model)
    except Exception:
        return False
