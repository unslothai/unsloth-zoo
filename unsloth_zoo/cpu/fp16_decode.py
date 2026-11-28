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

"""CPU rollouts (GRPO / online DPO generate in train mode) on x86 without native bf16: decode through fp16 copies.

Training on these CPUs computes in fp32, with the frozen projections stored in bf16 (upcast on every call) or in fp32.
A decoded token streams every frozen weight from DRAM, either as the bf16 bytes plus an fp32 copy written and read
back, or as fp32 bytes. FBGEMM's fp16 GEMM (``quantized.linear_dynamic_fp16``) reads 2 bytes per weight, converts in
registers and multiplies fp32 activations with fp32 accumulation: the 196 Qwen3-0.6B projections took 49 ms per
decoded token at 2 rows against 120 ms upcast per call and 68 ms in fp32 (Xeon 8370C, 2 threads), and 59 against 193
and 165 ms at 8 rows. Linux x86 wheels have it; Windows wheels ship only the oneDNN quantized engine, whose prepack
refuses fp16, so the availability check turns it off there.

Every bf16 value of magnitude at least 2**-17 (and at most 65504) is an fp16 value, so the packs hold the checkpoint's
own weights; smaller ones round to the fp16 subnormal grid (at most 2**-25 off: 0.017% of Qwen3-0.6B's weights), far
below the summation order difference between two fp32 GEMMs. Only weights whose values are bf16 numbers are packed
(an fp32 checkpoint's would round), never trainable ones, and only no-grad calls of at most ``_MAX_ROWS`` rows read
the packs: training forwards, gradients and the loss are unchanged. Packs are built once per weight (in parallel),
kept while the weight is unchanged at 2 bytes per packed weight when that fits in a share of the free RAM, and
dropped by for_inference. UNSLOTH_CPU_FP16_DECODE=0 turns it off, =1 skips the native bf16 check."""

import functools
import os
import platform
import weakref

import torch
import torch.nn.functional as F

__all__ = [
    "fp16_decode_available",
    "fp16_decode_wanted",
    "fp16_decode",
    "release_fp16_decode",
    "ENGAGED",
]

# Decode calls answered from a pack (tests and benchmarks read it).
ENGAGED = [0]

# Rows of one call (batch x new positions) up to which the fp16 GEMM beats the per-call path; longer prefills keep it.
_MAX_ROWS = 64
# Packs (2 bytes per packed weight) may take at most this share of the free RAM.
_RAM_SHARE = 0.25
# Output embeddings are packed in row blocks of this many vocabulary rows, in parallel.
_HEAD_BLOCK = 16384
_FP16_MAX = 65504.0
# Rows per step of the eligibility check (bounded temporaries for large output embeddings).
_CHECK_ROWS = 4096
_PROJECTIONS = frozenset((
    "q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj", "gate_proj", "up_proj", "down_proj", "gate_up_proj",
))
# module -> (weight key, [(first row, end row, packed), ...]); weak, so a deepcopy or a freed model carries no packs.
_PACKS = weakref.WeakKeyDictionary()


def _env():
    return os.environ.get("UNSLOTH_CPU_FP16_DECODE", "").strip().lower()


def _fbgemm_engine():
    """The quantized engine that builds FBGEMM fp16 packs: the current one when it does, else x86 / fbgemm when this
    build has them, else None (Windows wheels ship only oneDNN, whose prepack refuses fp16)."""
    try:
        engines = list(torch.backends.quantized.supported_engines)
        current = torch.backends.quantized.engine
    except Exception:
        return None
    if current in ("x86", "fbgemm"):
        return current
    return next((name for name in ("x86", "fbgemm") if name in engines), None)


class _quantized_engine:
    def __init__(self, name):
        self.name, self.previous = name, None

    def __enter__(self):
        self.previous = torch.backends.quantized.engine
        if self.previous != self.name:
            torch.backends.quantized.engine = self.name
        return self

    def __exit__(self, *exc):
        if torch.backends.quantized.engine != self.previous:
            torch.backends.quantized.engine = self.previous
        return False


@functools.cache
def fp16_decode_available():
    """Whether this torch build runs FBGEMM's fp16 GEMM and it agrees with F.linear on a small check."""
    if platform.machine().lower() not in ("x86_64", "amd64"):
        return False
    engine = _fbgemm_engine()
    if engine is None:
        return False
    try:
        generator = torch.Generator().manual_seed(0)
        weight = torch.randn(48, 40, generator = generator).bfloat16().float()
        bias = torch.randn(48, generator = generator)
        x = torch.randn(3, 40, generator = generator)
        with _quantized_engine(engine):
            packed = torch.ops.quantized.linear_prepack_fp16(weight, bias)
        y = torch.ops.quantized.linear_dynamic_fp16(x, packed)
        return bool(y.dtype == torch.float32 and y.shape == (3, 48)
                    and torch.allclose(y, F.linear(x, weight, bias), rtol = 1e-5, atol = 1e-5))
    except Exception:
        return False


def fp16_decode_wanted(model, rows = None):
    """True for a train-mode generate on x86 without native bf16 when the model computes in fp32 and this torch
    build has FBGEMM's fp16 GEMM, unless the rollout decodes more than _MAX_ROWS rows per token (then no decode call
    could use the packs). UNSLOTH_CPU_FP16_DECODE=0 turns it off, =1 skips the CPU check."""
    setting = _env()
    if setting in ("0", "false", "no", "off"):
        return False
    if not isinstance(model, torch.nn.Module) or not model.training:
        return False
    if rows is not None and rows > _MAX_ROWS:
        return False
    if setting not in ("1", "true", "yes", "on"):
        from .runtime import cpu_bf16_supported
        if cpu_bf16_supported():
            return False
    if platform.machine().lower() not in ("x86_64", "amd64"):
        return False
    return fp16_decode_available()


def _base_name(name):
    return name.replace(".base_layer", "").rsplit(".", 1)[-1]


def _candidates(model):
    """Frozen fp32-computing Linear projections and the output embedding: (module, is_head, adapter layer or None)."""
    head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if not isinstance(head, torch.nn.Linear):
        head = None
    owners = {id(module.base_layer): module for module in model.modules()
              if isinstance(getattr(module, "base_layer", None), torch.nn.Linear)}
    found, seen = [], set()
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or id(module) in seen:
            continue
        is_head = module is head
        if not is_head and _base_name(name) not in _PROJECTIONS:
            continue
        weight, bias = module.weight, module.bias
        if weight is None or weight.device.type != "cpu" or weight.dtype not in (torch.bfloat16, torch.float32) \
                or weight.requires_grad or weight.dim() != 2:
            continue
        if bias is not None and (bias.requires_grad or bias.dtype not in (torch.bfloat16, torch.float32)):
            continue
        seen.add(id(module))
        found.append((module, is_head, owners.get(id(module))))
    return found


def _key(module, owner):
    """Storage, version and shape of the weight (and bias), plus the adapters PEFT merged into it: merge_adapter /
    unmerge_adapter write through ``.data``, which moves neither the storage nor the version."""
    weight, bias = module.weight, module.bias
    key = (weight.data_ptr(), weight._version, weight.dtype, tuple(weight.shape))
    if bias is not None:
        key += (bias.data_ptr(), bias._version)
    if owner is not None:
        key += (tuple(getattr(owner, "merged_adapters", ()) or ()),)
    return key


def _packable(weight):
    """fp16 holds the weight's values: bf16 numbers (bf16 storage, or fp32 upcast from a bf16 checkpoint) in range.
    Checked _CHECK_ROWS rows at a time, so the temporaries stay small next to the weight."""
    w = weight.detach()
    for start in range(0, w.shape[0], _CHECK_ROWS):
        part = w[start:start + _CHECK_ROWS]
        if part.dtype == torch.float32 and not torch.equal(part.bfloat16().float(), part):
            return False
        low, high = torch.aminmax(part)
        if not bool(low >= -_FP16_MAX and high <= _FP16_MAX):  # NaN compares False
            return False
    return True


def _prepare(model):
    """Packs for every eligible module (built in parallel where missing or stale), or None when they do not fit."""
    stale, ready = [], []
    for module, is_head, owner in _candidates(model):
        key = _key(module, owner)
        cached = _PACKS.get(module, None)
        if cached is not None and cached[0] == key:
            ready.append((module, owner, key, cached[1]))
            continue
        if cached is not None:  # the weight changed (merged, reloaded, edited in place): pack it again
            del _PACKS[module]
        stale.append((module, is_head, key, owner))
    if stale:
        # RAM first: before any check or pack allocates, for every weight that would need one.
        from .runtime import available_ram
        extra = 2 * sum(module.weight.numel() for module, _, _, _ in stale)
        available = available_ram()
        if available is None or extra > _RAM_SHARE * available:
            return ready or None
    todo = [entry for entry in stale if _packable(entry[0].weight)]
    if todo:
        jobs = []
        for module, is_head, _, _ in todo:
            if is_head and module.weight.shape[0] > _HEAD_BLOCK:
                jobs.extend((module, True, start) for start in range(0, module.weight.shape[0], _HEAD_BLOCK))
            else:
                jobs.append((module, False, 0))
        packed = _pack_parallel(jobs)
        for module, _, key, owner in todo:
            parts = sorted(packed[id(module)], key = lambda part: part[0])
            _PACKS[module] = (key, parts)
            ready.append((module, owner, key, parts))
    return ready


def _pack_parallel(jobs):
    """{id(module): [(start, stop, packed), ...]}; FBGEMM packs one matrix per thread (the op releases the GIL)."""
    prepack = torch.ops.quantized.linear_prepack_fp16

    def work(job):
        module, is_block, start = job
        weight, bias = module.weight.detach(), module.bias
        stop = min(start + _HEAD_BLOCK, weight.shape[0]) if is_block else weight.shape[0]
        w = weight[start:stop].float()
        b = None if bias is None else bias.detach()[start:stop].float()
        return id(module), (start, stop, prepack(w, b))
    workers = max(1, min(len(jobs), torch.get_num_threads()))
    out = {}
    with _quantized_engine(_fbgemm_engine()):
        if workers == 1:
            results = [work(job) for job in jobs]
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(workers) as pool:
                results = list(pool.map(work, jobs))
    for key, part in results:
        out.setdefault(key, []).append(part)
    return out


def _cpu_autocast():
    try:
        return torch.is_autocast_enabled("cpu")
    except TypeError:  # torch < 2.4
        return torch.is_autocast_cpu_enabled()


def _decode_forward(module, owner, key, parts, forward):
    run = torch.ops.quantized.linear_dynamic_fp16
    out_features = module.weight.shape[0]

    def wrapped(x, *args, **kwargs):
        if args or kwargs or x.dtype != torch.float32 or x.device.type != "cpu" or torch.is_grad_enabled() \
                or x.shape[-1] != module.weight.shape[1] or _cpu_autocast() or _key(module, owner) != key:
            # Autocast would compute in its own dtype. The key again per call: PEFT unmerges a disabled adapter
            # inside its forward, right before this call.
            return forward(x, *args, **kwargs)
        rows = x.numel() // x.shape[-1] if x.shape[-1] else 0
        if rows == 0 or rows > _MAX_ROWS:
            return forward(x)
        x2 = x.reshape(rows, x.shape[-1]).contiguous()
        if len(parts) == 1:
            y = run(x2, parts[0][2])
        else:
            y = torch.cat([run(x2, packed) for _, _, packed in parts], dim = -1)
        ENGAGED[0] += 1
        return y.view(*x.shape[:-1], out_features)
    return wrapped


def release_fp16_decode(model):
    """Drops the packs of ``model``'s modules (for_inference: eval-mode generate never reads them)."""
    dropped = 0
    for module in model.modules() if isinstance(model, torch.nn.Module) else ():
        if _PACKS.pop(module, None) is not None:
            dropped += 1
    return dropped


class fp16_decode:
    """Context manager: inside, eligible modules of ``model`` answer no-grad calls of at most _MAX_ROWS rows from
    their fp16 packs; on exit each module's own forward is back. ``active`` is False when nothing was packed."""

    def __init__(self, model):
        self.model = model
        self.active = False
        self._patched = []

    def __enter__(self):
        try:
            ready = _prepare(self.model) if fp16_decode_wanted(self.model) else None
        except Exception:
            ready = None
        for module, owner, key, parts in ready or ():
            own = module.__dict__.get("forward", None)
            module.forward = _decode_forward(module, owner, key, parts, module.forward)
            self._patched.append((module, own))
        self.active = bool(self._patched)
        return self

    def __exit__(self, *exc):
        for module, own in self._patched:
            if own is None:
                module.__dict__.pop("forward", None)
            else:
                module.forward = own
        self._patched = []
        return False
