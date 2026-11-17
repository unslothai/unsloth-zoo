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
"""Frozen base expert outputs kept from the no-grad first pass for the gradient checkpoint replay.

Under Unsloth gradient checkpointing an MoE layer's frozen base expert GEMMs run three times per step:
the no-grad first pass, the replay in backward, and dX. For a 4-bit base each of those dequantizes the
whole expert stack. The replay's base outputs only feed the LoRA terms and the activation; their values
are the first pass's (the replay reuses the first pass's routing, gradient_checkpointing #2479), and their
backward is dX through the frozen weight, which never reads the forward value. So the first pass keeps
each base output [rows, N] in the checkpoint's routing frame and the replay takes it back: 2 dequants and
2 base GEMMs per projection instead of 3 and 3. Eager, with the separated LoRA path the replay computes
the same tensors bit for bit; with the folded LoRA (moe_lora_fold) both passes add the LoRA term to the
kept base the same way.

Memory: kept outputs live from a layer's first pass to its replay, so all of them coexist at the start of
backward. They are capped at UNSLOTH_MOE_GC_KEEP_BASE_MB per device (default auto: 4% of the device's total
memory); past the cap the oldest kept outputs (earliest layers, replayed last) are dropped and recomputed
as before. UNSLOTH_MOE_GC_KEEP_BASE=0 turns it off. Only the dequant / grouped_mm base path takes part
(not the fused NF4 GEMM, MXFP4 or the selective dequant plan), and nothing engages while tracing."""

__all__ = [
    "KEEP_BASE_STATS",
    "keep_base_enabled",
    "keep_base_budget_bytes",
    "keep_base_slot",
    "kept_base_bytes",
    "kept_base_mm",
]

import collections
import os
import weakref

import torch

# stored / reused: first-pass outputs kept and taken back; evicted: dropped for the cap; missing: replay found none
# (evicted, or a retain_graph second backward); bytes_peak: most bytes kept at once.
KEEP_BASE_STATS = collections.Counter()
_AUTO_FRACTION = 0.04
_BUDGET_CACHE = {}
# (frame weakref, key) in store order, per device: the eviction order.
_ORDER = collections.OrderedDict()


def keep_base_enabled() -> bool:
    return os.environ.get("UNSLOTH_MOE_GC_KEEP_BASE", "1").strip().lower() not in ("0", "false", "off", "no")


def keep_base_budget_bytes(device) -> int:
    raw = os.environ.get("UNSLOTH_MOE_GC_KEEP_BASE_MB", "auto").strip().lower()
    if raw not in ("", "auto"):
        try:
            return max(int(float(raw) * 2**20), 0)
        except ValueError:
            pass
    key = (device.type, device.index)
    hit = _BUDGET_CACHE.get(key)
    if hit is None:
        hit = 0
        try:
            if device.type == "cuda":
                index = device.index if device.index is not None else torch.cuda.current_device()
                hit = int(torch.cuda.get_device_properties(index).total_memory * _AUTO_FRACTION)
        except Exception:
            hit = 0
        _BUDGET_CACHE[key] = hit
    return hit


def kept_base_bytes(device = None) -> int:
    total = 0
    for (ref, key), (dev, nbytes) in list(_ORDER.items()):
        frame = ref()
        if frame is None or key not in frame.kept:
            _ORDER.pop((ref, key), None)
            continue
        if device is None or dev == device:
            total += nbytes
    return total


class KeepSlot:
    """One MoE experts call inside an Unsloth checkpoint: store base outputs (first pass) or take them (replay)."""
    __slots__ = ("frame", "call", "replaying", "module")

    def __init__(self, frame, call, replaying, module):
        self.frame, self.call, self.replaying, self.module = frame, call, replaying, id(module)

    def _key(self, proj):
        return (self.call, proj, self.module)

    def store(self, proj, tensor):
        """First pass (no grad): keep `tensor` for the replay if the cap allows (evicting the oldest)."""
        if self.replaying or torch.is_grad_enabled() or not isinstance(tensor, torch.Tensor):
            return
        device = (tensor.device.type, tensor.device.index)
        budget = keep_base_budget_bytes(tensor.device)
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes > budget:
            KEEP_BASE_STATS["over_budget"] += 1
            return
        used = kept_base_bytes(device)
        if used + nbytes > budget:
            # Oldest first: the earliest layers, replayed last, give way to later ones.
            for entry, (dev, size) in list(_ORDER.items()):
                if used + nbytes <= budget:
                    break
                if dev != device:
                    continue
                _ORDER.pop(entry, None)
                frame = entry[0]()
                if frame is not None and frame.kept.pop(entry[1], None) is not None:
                    used -= size
                    KEEP_BASE_STATS["evicted"] += 1
        if used + nbytes > budget:
            return
        key = self._key(proj)
        self.frame.kept[key] = tensor
        _ORDER[(weakref.ref(self.frame), key)] = (device, nbytes)
        KEEP_BASE_STATS["stored"] += 1
        KEEP_BASE_STATS["bytes_peak"] = max(KEEP_BASE_STATS["bytes_peak"], used + nbytes)

    def take(self, proj, rows, dtype, device):
        """Replay (grad on): the first pass's output for `proj`, or None (recompute as before)."""
        if not self.replaying or not torch.is_grad_enabled():
            return None
        key = self._key(proj)
        tensor = self.frame.kept.pop(key, None)
        for k in [k for k in _ORDER if k[1] == key and k[0]() is self.frame]:
            _ORDER.pop(k, None)
        if tensor is None:
            KEEP_BASE_STATS["missing"] += 1
            return None
        if tensor.dim() != 2 or tensor.shape[0] != rows or tensor.dtype != dtype or tensor.device != device:
            KEEP_BASE_STATS["mismatch"] += 1
            return None
        KEEP_BASE_STATS["reused"] += 1
        return tensor


def keep_base_slot(module):
    """KeepSlot for this experts call when it runs inside an Unsloth checkpoint whose routing frame stored (first
    pass) or reused (replay) this call's routing; else None."""
    if not keep_base_enabled() or torch.compiler.is_compiling() or not getattr(module, "training", False):
        return None
    try:
        from unsloth_zoo.gradient_checkpointing import moe_checkpoint_call
    except ImportError:
        return None
    found = moe_checkpoint_call()
    if found is None:
        return None
    frame, call, replaying = found
    return KeepSlot(frame, call, replaying, module)


class GroupedMMKeptBase(torch.autograd.Function):
    """Returns the kept base output (handed over in a one-element list, so it is not an autograd input); backward is
    dX through the frozen weight from weight_provider, as _GroupedMMRecompute's."""

    @staticmethod
    def forward(ctx, inputs, offsets, weight_provider, box):
        ctx.weight_provider = weight_provider
        ctx.save_for_backward(offsets)
        return box.pop()

    @staticmethod
    def backward(ctx, grad_output):
        from unsloth_zoo.temporary_patches.moe_utils import _grouped_mm_with_backward_fix
        (offsets,) = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None, None, None, None
        with torch.no_grad():
            transposed = getattr(ctx.weight_provider, "transposed", None)
            weight_t = transposed() if transposed is not None else ctx.weight_provider().transpose(-2, -1)
            grad_input = _grouped_mm_with_backward_fix(grad_output.contiguous(), weight_t, offsets)
        return grad_input, None, None, None


def kept_base_mm(inputs, offsets, weight_provider, recompute, lora, source, slot, proj):
    """(output, LoRA still to add) for one projection inside a checkpoint, or None for the normal path.

    First pass (no grad): the base GEMM runs alone and its output is kept. Replay: the kept output (or, when it
    was dropped, the base recomputed without grad) stands in for the base GEMM. Both passes add a foldable LoRA
    onto the base the same way (moe_lora_fold), so the replay reproduces the first pass's values."""
    from unsloth_zoo.temporary_patches import moe_utils as mu
    if not mu._base_is_recomputable(source):
        return None   # a trainable base needs its dW from the GEMM itself
    grad = torch.is_grad_enabled()
    if slot.replaying != grad:
        return None
    try:
        from unsloth_zoo.temporary_patches.moe_lora_fold import lora_fold_onto_base, lora_fold_prepare
    except ImportError:
        lora_fold_onto_base = lora_fold_prepare = None
    prepared = lora_fold_prepare(inputs, lora, source) if (lora is not None and lora_fold_prepare is not None) else None
    if slot.replaying:
        base = slot.take(proj, inputs.shape[0], inputs.dtype, inputs.device)
        if base is None:
            with torch.no_grad():
                base = mu._base_grouped_mm(inputs, offsets, weight_provider, recompute, None)
        if prepared is not None:
            return lora_fold_onto_base(inputs, offsets, weight_provider, prepared, base), None
        return GroupedMMKeptBase.apply(inputs, offsets, weight_provider, [base]), lora
    base = mu._base_grouped_mm(inputs, offsets, weight_provider, recompute, None)
    slot.store(proj, base)
    if prepared is not None:
        return lora_fold_onto_base(inputs, offsets, weight_provider, prepared, base, keep_base = True), None
    return base, lora
