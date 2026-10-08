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
"""Process-wide expert-state stamp, so the grouped MoE readiness checks and NF4 pointer tables
skip their O(experts) keys on every block call (forward, checkpoint replay, decode token).

The stamp is (struct, step). `struct` moves on whatever can change a checked expert subtree:
torch's global module / parameter / buffer registration hooks on a tracked module (every module of
a checked subtree; not the throwaway ModuleList `self.layers[:n]` builds per forward), `_apply`
(.to / .cuda / .half) on a tracked module, Module.requires_grad_, PEFT tuner-layer state methods
(set_adapter, merge, unmerge, enable_adapters, delete_adapter, update_layer, scaling) and bump().
`step` moves on every optimizer step; a cache then re-reads only a lean storage key (weight /
absmax addresses, requires_grad), which bounds a hook-less `.data =` swap to one step.

A cached verdict is reused only while the stamp (or the lean key), the call context and a spot
signature of the first and last expert all match (see Record / valid). LoRA dropout is the one
input that depends on train / eval, so non-Identity dropouts are compared on every call instead
of wrapping Module.train (Trainer calls model.train() over every module each step). Subtrees with
accelerate hooks (`_hf_hook`, which swap `_parameters` every forward) never get a record.

UNSLOTH_MOE_FAST_READY=0 (read per call) runs the full checks on every call.
"""
import functools
import os
from operator import attrgetter, is_, itemgetter, methodcaller

import torch

__all__ = ["bump", "stamp", "enabled", "COUNTS"]

_EPOCH = [0, 0]            # [struct, step]
_INSTALLED = [None]        # None: not tried, True / False
_TRACKED = "_unsloth_moe_ready_tracked"
VALID = "_unsloth_moe_ready_valid"   # experts.__dict__: (stamp, gen) storage verified at stamp
# Engagement counters: full checks, step re-validations (lean key), cheap hits, table hits / rebuilds.
COUNTS = {"full": 0, "lean": 0, "cheap": 0, "table_full": 0, "table_cheap": 0}
_PEFT_METHODS = (
    "set_adapter", "merge", "unmerge", "enable_adapters", "delete_adapter", "update_layer",
    "set_scale", "scale_layer", "unscale_layer", "set_requires_grad",
    "_move_adapter_to_device_of_base_layer",
)
_WRAPPED = "_unsloth_moe_ready_wrapped"
# Pointer-table caches whose held storages a move must release (rebuilt on the next call).
_TABLES = ("_unsloth_nf4_stack_tables", "_unsloth_routed_nf4")


def bump(*args, **kwargs):
    """Invalidate every cached readiness verdict and table (full re-check on the next call)."""
    _EPOCH[0] += 1


def _step_hook(*args, **kwargs):
    _EPOCH[1] += 1


def stamp():
    return (_EPOCH[0], _EPOCH[1])


def _registration_hook(module, name, value):
    if module.__dict__.get(_TRACKED, False):
        _EPOCH[0] += 1


def _wrap_bumping(owner, name):
    f = owner.__dict__.get(name)
    if f is None or not callable(f) or getattr(f, _WRAPPED, False) or isinstance(f, (staticmethod, classmethod)):
        return

    @functools.wraps(f)
    def wrapped(self, *args, **kwargs):
        _EPOCH[0] += 1
        try:
            return f(self, *args, **kwargs)
        finally:
            _EPOCH[0] += 1

    setattr(wrapped, _WRAPPED, True)
    setattr(owner, name, wrapped)


def wrap_peft():
    """Wrap the state methods of PEFT's BaseTunerLayer and every subclass defined so far (lora.bnb
    and others import lazily, so the full checks call this again). Returns False without PEFT."""
    try:
        from peft.tuners.tuners_utils import BaseTunerLayer
    except Exception:
        return False
    seen, todo = set(), [BaseTunerLayer]
    while todo:
        cls = todo.pop()
        if cls in seen:
            continue
        seen.add(cls)
        todo += cls.__subclasses__()
        for name in _PEFT_METHODS:
            if name in cls.__dict__:
                _wrap_bumping(cls, name)
    return True


def install():
    """Register the hooks once; False when this torch lacks the global registration hooks."""
    if _INSTALLED[0] is not None:
        return _INSTALLED[0]
    _INSTALLED[0] = False
    try:
        mod = torch.nn.modules.module
        hooks = [getattr(mod, n, None) for n in (
            "register_module_module_registration_hook",
            "register_module_parameter_registration_hook",
            "register_module_buffer_registration_hook",
        )]
        from torch.optim.optimizer import register_optimizer_step_post_hook
        if any(h is None for h in hooks):
            return False
        for h in hooks:
            h(_registration_hook)
        register_optimizer_step_post_hook(_step_hook)

        M = torch.nn.Module
        apply = M._apply
        if not getattr(apply, _WRAPPED, False):
            @functools.wraps(apply)
            def _apply(self, *args, **kwargs):
                d = self.__dict__
                if d.get(_TRACKED, False):
                    _EPOCH[0] += 1
                    for name in _TABLES:
                        d.pop(name, None)
                return apply(self, *args, **kwargs)
            setattr(_apply, _WRAPPED, True)
            M._apply = _apply
        _wrap_bumping(M, "requires_grad_")
        wrap_peft()
    except Exception:
        return False
    _INSTALLED[0] = True
    return True


def enabled():
    return os.environ.get("UNSLOTH_MOE_FAST_READY", "1") != "0" and (_INSTALLED[0] or install())


def track(modules):
    for m in modules:
        m.__dict__[_TRACKED] = True


def scan(root):
    """Track every module under `root` (itself included); returns (an accelerate hook was seen,
    the non-Identity dropout modules)."""
    hooked = False
    drops = []
    dropout = torch.nn.modules.dropout._DropoutNd
    stack = [root]
    while stack:
        m = stack.pop()
        d = m.__dict__
        d[_TRACKED] = True
        if "_hf_hook" in d:
            hooked = True
        if isinstance(m, dropout):
            drops.append(m)
        stack += [c for c in d["_modules"].values() if c is not None]
    return hooked, drops


_get_weight = itemgetter("weight")
_get_bias = methodcaller("get", "bias")
_get_quant_state = attrgetter("quant_state")
_get_absmax = attrgetter("absmax")
_get_requires_grad = attrgetter("requires_grad")
_ptr = torch._C.TensorBase.data_ptr


def _qs_ptrs(qs):
    """Addresses the NF4 pointer tables embed: absmax, code and, when nested, state2 (absmax /
    code) and the offset (a hook-less swap of any of them must re-key the tables)."""
    key = (_ptr(qs.absmax), _ptr(qs.code))
    if getattr(qs, "nested", False):
        s2, off = qs.state2, qs.offset
        key += (id(s2), _ptr(s2.absmax), _ptr(s2.code), _ptr(off) if isinstance(off, torch.Tensor) else off)
    return key


class Lean:
    """What a hook-less `.data =` / quant_state / absmax swap or requires_grad flip changes, over the
    base projections' `_parameters` dicts: weight and bias identity, weight address and
    requires_grad, quant state identity and the addresses its tables embed (_qs_ptrs). Re-read in
    C-level maps (one optimizer step bumps only the step stamp, so this runs once per block per
    step)."""
    __slots__ = ("params", "ws", "wptrs", "rgs", "qss", "aptrs", "bs", "brgs")

    def __init__(self, params):
        self.params = params
        self.ws, self.bs = list(map(_get_weight, params)), list(map(_get_bias, params))
        with torch._C.DisableTorchFunctionSubclass():
            self.wptrs = list(map(_ptr, self.ws))
            self.rgs = list(map(_get_requires_grad, self.ws))
            self.brgs = [b is not None and b.requires_grad for b in self.bs]
            qss = [w.__dict__.get("quant_state") for w in self.ws]
            # All 4-bit or none (the readiness checks decline mixes); else no quant state check.
            self.qss = qss if all(q is not None for q in qss) else None
            self.aptrs = None if self.qss is None else list(map(_qs_ptrs, qss))

    def same(self):
        try:
            params = self.params
            ws, bs = list(map(_get_weight, params)), list(map(_get_bias, params))
            if not (all(map(is_, ws, self.ws)) and all(map(is_, bs, self.bs))):
                return False
            with torch._C.DisableTorchFunctionSubclass():
                if list(map(_ptr, ws)) != self.wptrs or list(map(_get_requires_grad, ws)) != self.rgs:
                    return False
                if any(self.brgs) or bs.count(None) != len(bs):
                    if [b is not None and b.requires_grad for b in bs] != self.brgs:
                        return False
                if self.qss is not None:
                    qss = list(map(_get_quant_state, ws))
                    if not all(map(is_, qss, self.qss)) or list(map(_qs_ptrs, qss)) != self.aptrs:
                        return False
            return True
        except Exception:
            return False


class Record:
    """What a full check saw: the stamp it is valid at, the call context, the spot signature of
    the end experts (and what keeps its ids alive), the Lean storage view, non-Identity dropouts and
    their (training, p), and a token for the NF4 tables (new per full check, so a table re-keys
    over every expert once after any expert-state change; a step re-validation keeps it)."""
    __slots__ = ("stamp", "ctx", "spot", "lean", "drops", "drop_state", "gen", "refs")

    def __init__(self, ctx, spot, params, drops, refs):
        self.stamp = stamp()
        self.ctx = ctx
        self.spot = spot
        self.lean = Lean(params)
        self.drops = drops
        self.drop_state = drop_state(drops)
        self.gen = object()
        self.refs = refs


def drop_state(drops):
    return tuple((m.__dict__.get("training"), m.__dict__.get("p")) for m in drops)


def valid(rec, ctx, spot_fn):
    """True (and the record re-stamped) while `rec` still describes the experts; see Record."""
    if rec is None or rec.ctx != ctx:
        return False
    cur = (_EPOCH[0], _EPOCH[1])
    lean = False
    if rec.stamp != cur:
        if rec.stamp[0] != cur[0] or not rec.lean.same():
            return False
        lean = True
    if rec.drops and drop_state(rec.drops) != rec.drop_state:
        return False
    spot = spot_fn()
    if spot is None or spot != rec.spot:
        return False
    rec.stamp = cur
    COUNTS["lean" if lean else "cheap"] += 1
    return True


def mark_valid(experts, rec):
    """Readiness verified the experts' storage (rec.lean) at the current stamp: tables may trust it."""
    experts.__dict__[VALID] = (rec.stamp, rec.gen)


def current_gen(experts):
    """The storage generation verified at the current stamp, else None."""
    v = experts.__dict__.get(VALID)
    if v is None or v[0] != (_EPOCH[0], _EPOCH[1]):
        return None
    return v[1]
