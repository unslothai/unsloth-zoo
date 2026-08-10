# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Head-aware multi-GPU device map planner.

Why this exists
---------------
When a quantised causal LM or vision-LM is split over several GPUs, the tail of
the model (final norm + output head) lands on the *last* card that the layout
walk touches. Anything that materialises a full-vocabulary logit tensor then has
to allocate `rows x vocab` on that same card:

* TRL / Unsloth GRPO calls
  `unsloth_zoo.rl_replacements.chunked_hidden_states_selective_log_softmax`,
  which builds `chunk_rows x vocab` logits in the compute dtype and then a
  float32 copy of them.
* On a 200k-token vocabulary that is hundreds of MB per chunk, on top of the
  weights already resident on that card.

`device_map="auto"` / `"balanced"` splits the weights *evenly*, so the head's
card ends up with the same weight load as every other card and no room for the
logits. `device_map="sequential"` fills greedily, but accelerate only reserves
`max_layer_size` on GPU 0 (`main_devices = [gpus[0], "cpu"]`), never on the card
that actually holds the head.

Capping the head's card with `max_memory` does not work either: accelerate's
allocator never backtracks to an earlier device, so once the capped card is
full the remaining modules (the head itself, typically) are pushed to `"cpu"` /
`"disk"`, and bitsandbytes then refuses to load with

    ValueError: Some modules are dispatched on the CPU or the disk.

This module therefore builds an **explicit** `device_map` dict: every split unit
of the model is named and pinned to a GPU, so accelerate has no freedom to spill.

Core idea
---------
Reserve headroom on whichever device holds the output head, sized from the real
vocabulary width and the real per-chunk row count, and hand out the rest of the
slack as an activation budget for every device (``free_space_policy="balanced"``,
the default). ``free_space_policy="head_max"`` instead fills the non-head cards as
far as they go, which maximises head free space but can starve GPU 0.

Caveat, measured: activations follow layers. Moving decoder layers off the head's
card raises the other card's peak by more than the weight it gained (about
0.55 GiB per layer for a 4 x 640-token GRPO step on a 30B 4-bit model, against
about 0.27 GiB saved
on the head card). On a vision-LM whose GPU 0 also holds the vision tower, the input
embedding and the generation KV cache, pass ``activation_reserve_bytes`` as a
per-device mapping built from measurements instead of trusting the equal split.

Formula (see :func:`logit_headroom_bytes`), with ``s = logit dtype itemsize``::

    per_chunk = rows_per_chunk * vocab * (s + 4)          # logits + float32 copy
              + rows_per_chunk * vocab * s   if softcapped        # tanh temporary
              + rows_per_chunk * vocab * 4   if temperature != 1  # 2nd fp32 buffer
              + rows_per_chunk * vocab * 4   if retained_rows      # backward grad buffer
    retained  = retained_rows  * vocab * (4 + (s if softcapped else 0))
    headroom  = per_chunk + retained + safety_bytes

* ``rows_per_chunk * vocab * s`` is ``chunk_hidden @ lm_head.T``.
* ``+ 4`` bytes/element is the ``chunk_logits.to(torch.float32)`` copy, which is
  live at the same time as the low-precision tensor it was copied from.
* the softcap term is the ``logit_softcapping * tanh(x / logit_softcapping)``
  temporary, which is also the tensor autograd saves for the tanh backward.
* ``retained_rows`` is the autograd case: when the log-softmax is
  differentiated, the float32 logits of *every* chunk (plus the saved tanh
  output when softcapping is on) stay alive until backward, so the retained
  term scales with ``total_rows``, not with ``rows_per_chunk``. Chunking does
  not help there. Pass ``retained_rows=0`` for a ``torch.no_grad()`` /
  reference-model pass.

Usage
-----
    from device_map_planner import plan_device_map_for_pretrained

    plan = plan_device_map_for_pretrained(
        "unsloth/Qwen3-14B-unsloth-bnb-4bit",
        max_memory     = {0: "14GiB", 1: "14GiB"},
        vocab_size     = None,          # read from the config
        rows_per_chunk = 128,
        retained_rows  = 0,
    )
    if plan is None:                    # single GPU, nothing to plan
        device_map = "sequential"
    else:
        device_map = plan.device_map
        print(plan.describe())

    model, tok = FastModel.from_pretrained(..., device_map = device_map)

Degradation contract
--------------------
* 0 or 1 usable GPU  -> returns ``None``; the caller keeps its current
  behaviour (``"auto"`` / ``"sequential"`` / single device).
* enough memory      -> a plan that is a strict superset of what
  ``"sequential"`` would have done, with the head's card deliberately
  under-filled.
* not enough memory  -> raises :class:`DeviceMapInfeasible` with the exact
  numbers. It never silently offloads to CPU or disk.

Model agnostic
--------------
Nothing here is specific to one architecture:

* split granularity comes from ``model._no_split_modules`` (falling back to a
  scan of ``nn.ModuleList`` children),
* the head comes from ``model.get_output_embeddings()`` resolved back to its
  module name by identity,
* tied embeddings are detected from the config *and* by parameter identity, and
  the tied input embedding is pinned to the head's device,
* vision towers / multimodal projectors / final norms are just ordinary split
  units and are placed by the same greedy walk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn

__all__ = [
    "DeviceMapInfeasible",
    "DeviceMapPlan",
    "logit_headroom_bytes",
    "plan_device_map",
    "plan_device_map_for_pretrained",
    "build_meta_model",
]

_GiB = 1024 ** 3


class DeviceMapInfeasible(RuntimeError):
    """The model plus the requested head headroom does not fit on the GPUs."""


# --------------------------------------------------------------------------- #
# headroom formula
# --------------------------------------------------------------------------- #
def logit_headroom_bytes(
    vocab_size: int,
    rows_per_chunk: int,
    *,
    logit_dtype: torch.dtype = torch.bfloat16,
    retained_rows: int = 0,
    softcapped: bool = True,
    temperature_scaled: bool = False,
    safety_bytes: int = 256 * 1024 ** 2,
) -> int:
    """Bytes that must stay free on the device holding the output head.

    Args:
        vocab_size: width of the output head (``lm_head.out_features``).
        rows_per_chunk: number of token rows the log-softmax materialises at
            once. For ``chunked_hidden_states_selective_log_softmax`` this is
            ``ceil(total_rows / chunks)``, or the ``TOKENS_PER_CHUNK`` cap if
            the caller enforces one.
        logit_dtype: dtype of the matmul output, i.e. the lm_head dtype.
        retained_rows: total rows whose logits stay alive until backward. ``0``
            under ``torch.no_grad()``; ``total_rows`` when the log-softmax is
            differentiated, because autograd keeps every chunk's saved tensors.
        softcapped: the caller passes a non-zero ``logit_softcapping``, adding a
            ``tanh`` temporary of the chunk shape (also saved for backward).
        temperature_scaled: the caller passes ``temperature != 1``, adding one
            more float32 buffer of the chunk shape.
        safety_bytes: allocator fragmentation / cuBLAS workspace slack. The
            default 256 MiB was chosen from measurements: the modelled terms
            below land within ~0.25 GiB of the observed allocator peak.

    Returns:
        Headroom in bytes.
    """
    if vocab_size <= 0 or rows_per_chunk <= 0:
        return int(safety_bytes)
    s = torch.empty((), dtype=logit_dtype).element_size()
    per_chunk = rows_per_chunk * vocab_size * (s + 4)
    if softcapped:
        per_chunk += rows_per_chunk * vocab_size * s
    if temperature_scaled:
        per_chunk += rows_per_chunk * vocab_size * 4
    if retained_rows:
        # backward allocates one float32 gradient buffer of the chunk shape while
        # the saved forward tensors of every chunk are still resident.
        per_chunk += rows_per_chunk * vocab_size * 4
    retained = int(retained_rows) * vocab_size * (4 + (s if softcapped else 0))
    return int(per_chunk + retained + safety_bytes)


# --------------------------------------------------------------------------- #
# plan object
# --------------------------------------------------------------------------- #
@dataclass
class DeviceMapPlan:
    """Result of :func:`plan_device_map`."""

    device_map: dict[str, int]
    head_module: str | None
    head_device: int
    headroom_bytes: int
    budgets: dict[int, int]
    """Per-device budget actually available to weights (after every reserve)."""
    raw_budgets: dict[int, int]
    """Per-device budget as requested by the caller, before any reserve."""
    weight_bytes: dict[int, int]
    activation_reserve_bytes: int
    total_weight_bytes: int
    no_split_module_classes: list[str]
    tied_to_head: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def free_bytes(self) -> dict[int, int]:
        """Budget minus placed weights, per device (what activations may use)."""
        return {d: self.raw_budgets[d] - self.weight_bytes.get(d, 0) for d in self.raw_budgets}

    def describe(self) -> str:
        lines = [
            f"total weights      : {self.total_weight_bytes / _GiB:.3f} GiB",
            f"no_split classes   : {sorted(self.no_split_module_classes)}",
            f"output head        : {self.head_module} -> cuda:{self.head_device}",
            f"head headroom      : {self.headroom_bytes / _GiB:.3f} GiB",
            f"activation reserve : {self.activation_reserve_bytes / _GiB:.3f} GiB per device",
        ]
        if self.tied_to_head:
            lines.append(f"tied to head       : {self.tied_to_head}")
        for d in sorted(self.raw_budgets):
            w = self.weight_bytes.get(d, 0)
            lines.append(
                f"  cuda:{d}  budget {self.raw_budgets[d] / _GiB:6.2f} GiB"
                f"  weights {w / _GiB:6.3f} GiB"
                f"  free {(self.raw_budgets[d] - w) / _GiB:6.3f} GiB"
                + ("   <- output head" if d == self.head_device else "")
            )
        lines.extend(f"note: {n}" for n in self.notes)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# introspection helpers (all generic)
# --------------------------------------------------------------------------- #
def _parse_size(value: Any) -> int:
    """Accept 12345, "14GiB", "14GB", "500MiB"."""
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return int(value)
    s = str(value).strip()
    units = (("GIB", _GiB), ("MIB", 1024 ** 2), ("KIB", 1024),
             ("GB", 10 ** 9), ("MB", 10 ** 6), ("KB", 10 ** 3),
             ("B", 1))
    up = s.upper().replace(" ", "")
    for suffix, mult in units:
        if up.endswith(suffix):
            return int(float(up[: -len(suffix)]) * mult)
    return int(float(s))


def _module_by_name(model: nn.Module, name: str) -> nn.Module:
    mod = model
    if name:
        for part in name.split("."):
            mod = getattr(mod, part)
    return mod


def _name_of_module(model: nn.Module, target: nn.Module) -> str | None:
    for name, mod in model.named_modules():
        if mod is target:
            return name
    return None


def resolve_no_split_classes(model: nn.Module) -> list[str]:
    """Decoder / encoder block classes that must not be split across devices."""
    classes = getattr(model, "_no_split_modules", None)
    if classes:
        return sorted(str(c) for c in classes)
    # Fallback: every distinct child class of every nn.ModuleList that holds
    # more than one entry. That is where repeated transformer blocks live.
    found: set[str] = set()
    for _, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 1:
            for child in mod:
                if any(True for _ in child.parameters(recurse=True)):
                    found.add(type(child).__name__)
    return sorted(found)


def resolve_output_head(model: nn.Module) -> tuple[str | None, nn.Module | None]:
    """Locate the output head generically.

    Prefers ``get_output_embeddings()``. Falls back to a name scan for the
    usual head attribute names, then to the widest ``nn.Linear`` in the model.
    """
    head = None
    getter = getattr(model, "get_output_embeddings", None)
    if callable(getter):
        try:
            head = getter()
        except Exception:
            head = None
    if head is not None:
        name = _name_of_module(model, head)
        if name is not None:
            return name, head
    for candidate in ("lm_head", "output", "embed_out", "score",
                      "language_model.lm_head", "model.lm_head"):
        try:
            mod = _module_by_name(model, candidate)
        except AttributeError:
            continue
        if isinstance(mod, nn.Module):
            return candidate, mod
    widest_name, widest_mod, widest = None, None, -1
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and mod.out_features > widest:
            widest_name, widest_mod, widest = name, mod, mod.out_features
    return widest_name, widest_mod


def resolve_head_width(model: nn.Module, head: nn.Module | None) -> int:
    """Vocabulary width of the output head."""
    if head is not None:
        w = getattr(head, "out_features", None)
        if isinstance(w, int) and w > 0:
            return w
        weight = getattr(head, "weight", None)
        if weight is not None and weight.dim() >= 1:
            return int(weight.shape[0])
    cfg = getattr(model, "config", None)
    for holder in (getattr(cfg, "text_config", None), cfg):
        v = getattr(holder, "vocab_size", None)
        if isinstance(v, int) and v > 0:
            return v
    return 0


def head_is_tied(model: nn.Module, head: nn.Module | None) -> bool:
    """True when the output head shares storage with the input embedding."""
    cfg = getattr(model, "config", None)
    for holder in (cfg, getattr(cfg, "text_config", None)):
        flag = getattr(holder, "tie_word_embeddings", None)
        if flag is True:
            return True
    if head is None:
        return False
    getter = getattr(model, "get_input_embeddings", None)
    if not callable(getter):
        return False
    try:
        inp = getter()
    except Exception:
        return False
    hw, iw = getattr(head, "weight", None), getattr(inp, "weight", None)
    return hw is not None and iw is not None and hw is iw


def _model_dtype(model: nn.Module) -> Any:
    """The dtype the checkpoint will be loaded in, as the config declares it."""
    cfg = getattr(model, "config", None)
    for holder in (cfg, getattr(cfg, "text_config", None)):
        for attr in ("dtype", "torch_dtype"):
            d = getattr(holder, attr, None)
            if isinstance(d, torch.dtype):
                return d
    for t in model.parameters():
        if t.dtype.is_floating_point:
            return t.dtype
    return None


def _quantized_size_kwargs(model: nn.Module, hf_quantizer: Any) -> dict[str, Any]:
    """`dtype` / `special_dtypes` for `compute_module_sizes`, quantiser-aware.

    Mirrors what ``transformers.modeling_utils._get_device_map`` does before it
    calls ``infer_auto_device_map``: the quantiser turns the load dtype into the
    storage dtype it will really allocate (``CustomDtype.INT4`` for bnb-4bit,
    half a byte per element), and names the modules it will NOT convert so those
    keep the load dtype.

    This matters a lot. ``preprocess_model`` leaves a meta ``Params4bit`` at the
    full unpacked shape in float32, so measuring the modules directly reports a
    4-bit checkpoint at roughly four times its real size, and the planner then
    refuses or badly partitions exactly the quantised models it exists for.
    """
    if hf_quantizer is None:
        return {}
    dtype = _model_dtype(model)
    if dtype is None:
        return {}
    kwargs: dict[str, Any] = {}
    adjust = getattr(hf_quantizer, "adjust_target_dtype", None)
    if callable(adjust):
        try:
            target = adjust(dtype)
        except Exception:
            target = None
        if target is not None:
            kwargs["dtype"] = target
    special = getattr(hf_quantizer, "get_special_dtypes_update", None)
    if callable(special) and getattr(hf_quantizer, "modules_to_not_convert", None):
        try:
            kwargs["special_dtypes"] = dict(special(model, dtype))
        except Exception:
            pass
    return kwargs


def _compute_module_sizes(model: nn.Module, hf_quantizer: Any = None) -> dict[str, int]:
    """Bytes per module subtree. Uses transformers'/accelerate's version when
    available so quantised (bnb / Params4bit) sizes match what the loader will
    actually allocate; otherwise falls back to a plain walk."""
    size_kwargs = _quantized_size_kwargs(model, hf_quantizer)
    for mod_path in ("transformers.integrations.accelerate", "accelerate.utils.modeling"):
        try:
            module = __import__(mod_path, fromlist=["compute_module_sizes"])
            fn = getattr(module, "compute_module_sizes")
        except Exception:
            continue
        # Drop the extras one at a time rather than all at once: an older
        # accelerate without `special_dtypes` still gets the storage dtype,
        # which is the term that actually moves the number.
        for kwargs in ([size_kwargs] if size_kwargs else []) + \
                      ([{"dtype": size_kwargs["dtype"]}] if "special_dtypes" in size_kwargs else []) + [{}]:
            try:
                out = fn(model, **kwargs)
            except Exception:
                continue
            if isinstance(out, tuple):
                out = out[0]
            if isinstance(out, Mapping) and "" in out:
                return {k: int(v) for k, v in out.items()}
    sizes: dict[str, int] = {}
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        nbytes = tensor.numel() * tensor.element_size()
        parts = name.split(".")
        for i in range(len(parts) + 1):
            sizes[".".join(parts[:i])] = sizes.get(".".join(parts[:i]), 0) + nbytes
    sizes.setdefault("", 0)
    return sizes


def _split_units(
    model: nn.Module,
    no_split_classes: Sequence[str],
    sizes: Mapping[str, int],
) -> list[tuple[str, int]]:
    """Ordered, non-overlapping list of (module name, bytes) placement units.

    Descends the module tree in definition order and stops at (a) modules whose
    class is in ``no_split_classes``, (b) modules with no child modules, and
    (c) modules with no parameters or buffers at all (they still get an entry so
    the map covers the whole model).
    """
    no_split = set(no_split_classes)
    units: list[tuple[str, int]] = []

    def walk(prefix: str, module: nn.Module) -> None:
        children = list(module.named_children())
        size = sizes.get(prefix, 0)
        if prefix and (type(module).__name__ in no_split or not children or size == 0):
            units.append((prefix, size))
            return
        if not children:
            units.append((prefix, size))
            return
        # Direct parameters/buffers owned by this module (not by a child) have
        # to travel with the module itself, so keep them as their own unit.
        own_tensors = list(module.named_parameters(recurse=False)) + \
                      list(module.named_buffers(recurse=False))
        own = sum(t.numel() * t.element_size() for _, t in own_tensors)
        if own and prefix:
            units.append((prefix, own))
        elif own:
            # State registered directly on the ROOT module. There is no module
            # name to hang it on -- the prefix is "" -- and accelerate reads ""
            # as a catch-all default for everything unlisted, which would fight
            # the explicit entries this planner exists to produce. So name each
            # tensor instead: accelerate's lookup walks a parameter name up to
            # its longest present prefix and tries the full name first, so an
            # exact key matches immediately. Without this the coverage check at
            # the end of plan_device_map reports the state as unplaced and
            # raises, so a model with a root-level scale or bias could not be
            # planned at all.
            for tensor_name, tensor in own_tensors:
                units.append((tensor_name, tensor.numel() * tensor.element_size()))
        for child_name, child in children:
            walk(f"{prefix}.{child_name}" if prefix else child_name, child)

    walk("", model)
    return units


def _usable_devices(max_memory: Mapping[Any, Any] | None) -> list[int]:
    if max_memory is not None:
        devs = sorted(int(k) for k in max_memory if isinstance(k, int) or str(k).isdigit())
        return [d for d in devs if _parse_size(max_memory[d] if d in max_memory else max_memory[str(d)]) > 0]
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


# --------------------------------------------------------------------------- #
# the planner
# --------------------------------------------------------------------------- #
def plan_device_map(
    model: nn.Module,
    *,
    max_memory: Mapping[Any, Any] | None = None,
    rows_per_chunk: int = 128,
    retained_rows: int = 0,
    vocab_size: int | None = None,
    logit_dtype: torch.dtype | None = None,
    softcapped: bool | None = None,
    temperature_scaled: bool = False,
    headroom_bytes: int | None = None,
    safety_bytes: int = 256 * 1024 ** 2,
    activation_reserve_bytes: int | Mapping[int, Any] | None = None,
    free_space_policy: str = "balanced",
    hf_quantizer: Any = None,
    no_split_module_classes: Sequence[str] | None = None,
    prefer_head_device: int | None = None,
) -> DeviceMapPlan | None:
    """Build an explicit device map that reserves logit headroom on the head's card.

    Args:
        model: the model, real or on the meta device (see :func:`build_meta_model`).
        max_memory: ``{device_index: bytes or "14GiB"}``. Defaults to the free
            memory reported by every visible CUDA device.
        rows_per_chunk: rows the log-softmax materialises at once.
        retained_rows: rows whose float32 logits survive to backward
            (``total_rows`` with autograd, ``0`` under ``no_grad``).
        vocab_size: override the detected head width.
        logit_dtype: dtype of the logits; defaults to the head's dtype.
        softcapped: whether a non-zero ``logit_softcapping`` is in play.
            ``None`` (default) reads ``final_logit_softcapping`` off the config.
        temperature_scaled: whether the caller divides by a temperature != 1.
        headroom_bytes: bypass the formula entirely.
        safety_bytes: slack added to the headroom.
        activation_reserve_bytes: bytes kept free for ordinary activations. An
            int applies to every device; a ``{device: bytes_or_str}`` mapping
            sets it per device, which is what you want when one card also holds
            a vision tower or the generation KV cache. Kept free on *every* device for ordinary
            activations, on top of the head headroom on the head's device.
            Defaults follow ``free_space_policy``. A value you pass is a **hard
            constraint**: if the model cannot be placed while keeping it free,
            :class:`DeviceMapInfeasible` is raised rather than a plan that
            quietly leaves less room than you asked for. Only the default,
            auto-derived reserve is relaxed to make a placement fit.
        free_space_policy: how the leftover memory is shared.
            ``"balanced"`` (default) gives every device an equal activation
            budget ``(total_slack - headroom) / n_devices`` and hands the head's
            device the logit headroom on top. ``"head_max"`` reserves only the
            largest split unit per device (accelerate's rule) and lets the
            greedy fill push as much weight as possible off the head's card,
            which maximises head free space but can starve GPU 0.
        hf_quantizer: pass the model's ``HfQuantizer`` so quantised sizes are
            exact.
        no_split_module_classes: override the detected block classes.
        prefer_head_device: force the head onto this device index.

    Returns:
        A :class:`DeviceMapPlan`, or ``None`` when there are fewer than two
        usable GPUs (the caller should keep its existing behaviour).

    Raises:
        DeviceMapInfeasible: when no assignment fits without CPU/disk offload.
    """
    devices = _usable_devices(max_memory)
    if len(devices) < 2:
        return None

    raw_budgets: dict[int, int] = {}
    for d in devices:
        if max_memory is not None:
            raw = max_memory[d] if d in max_memory else max_memory[str(d)]
            raw_budgets[d] = _parse_size(raw)
        else:
            free, _total = torch.cuda.mem_get_info(d)
            raw_budgets[d] = int(free)

    no_split = list(no_split_module_classes) if no_split_module_classes else resolve_no_split_classes(model)
    sizes = _compute_module_sizes(model, hf_quantizer)
    units = _split_units(model, no_split, sizes)
    total = sizes.get("", sum(s for _, s in units))

    head_name, head_mod = resolve_output_head(model)
    width = int(vocab_size) if vocab_size else resolve_head_width(model, head_mod)
    if logit_dtype is None:
        w = getattr(head_mod, "weight", None)
        logit_dtype = w.dtype if w is not None and w.dtype.is_floating_point else torch.bfloat16

    if softcapped is None:
        cfg = getattr(model, "config", None)
        softcapped = any(
            bool(getattr(h, "final_logit_softcapping", None))
            for h in (cfg, getattr(cfg, "text_config", None))
        )

    if headroom_bytes is None:
        headroom = logit_headroom_bytes(
            width, rows_per_chunk,
            logit_dtype=logit_dtype,
            retained_rows=retained_rows,
            softcapped=softcapped,
            temperature_scaled=temperature_scaled,
            safety_bytes=safety_bytes,
        )
    else:
        headroom = int(headroom_bytes)

    if free_space_policy not in ("balanced", "head_max"):
        raise ValueError(f"free_space_policy must be 'balanced' or 'head_max', got {free_space_policy!r}")

    notes: list[str] = []

    # Units that must travel with the head. Resolved BEFORE the reserve is
    # derived: the head's device pays for them on top of the headroom, so the
    # reserve cap below cannot be computed without knowing how big they are.
    pinned: list[str] = []
    if head_name is not None:
        pinned.append(head_name)
    tied_to_head: list[str] = []
    if head_is_tied(model, head_mod):
        getter = getattr(model, "get_input_embeddings", None)
        inp = getter() if callable(getter) else None
        inp_name = _name_of_module(model, inp) if inp is not None else None
        if inp_name is not None and inp_name != head_name:
            pinned.append(inp_name)
            tied_to_head.append(inp_name)
            notes.append(
                f"tied embeddings: {inp_name} pinned with the head so accelerate "
                "does not have to mirror the weight across devices"
            )

    unit_size = dict(units)
    # Expand each pinned module to every placement unit at or below it. A head
    # that is a plain nn.Linear is a leaf and is its own unit, so this is a
    # no-op there; a composite head is split by `_split_units` into descendants
    # like `lm_head.proj` and the head's own name never appears as a unit, so an
    # exact-name filter dropped it from `pinned` entirely and the greedy walk was
    # free to put it on a card that is not `head_device` -- leaving the logit
    # headroom reserved on the wrong GPU.
    pinned = list(dict.fromkeys(
        u for p in pinned for u, _ in units if u == p or u.startswith(p + ".")
    ))
    pinned_bytes = sum(unit_size[p] for p in pinned)

    # A reserve the caller measured is a constraint; a reserve we guessed is a
    # heuristic. Only the guess may be relaxed to make a placement fit (see
    # `attempt`), otherwise a per-device mapping built from measurements -- the
    # documented answer to the asymmetric-activation caveat above -- could be
    # quietly reduced to zero and the run would OOM on a card the plan still
    # claimed had the reserve free.
    reserve_is_explicit = activation_reserve_bytes is not None
    if activation_reserve_bytes is None:
        if free_space_policy == "balanced":
            # Give every device the same activation budget, then hand the head's
            # device the logit headroom on top.
            #
            # This equal split is a heuristic and it is only right when the
            # per-device activation demand is symmetric. It is NOT symmetric for
            # a vision-LM whose GPU 0 also carries the vision tower, the input
            # embedding and the generation KV cache: a measured GRPO step
            # (4 x 640 completion tokens, a 30B 4-bit model, 2 x 14.56 GiB) OOMs on
            # GPU 0 under the equal split while succeeding under accelerate's
            # own layout. Pass a per-device mapping when you have measurements.
            slack = sum(raw_budgets.values()) - total
            activation_reserve_bytes = int(max(0, (slack - headroom) // len(devices)))
            # The head's device pays the headroom on top of its share, so an
            # equal split of the global slack can exceed that one device's own
            # budget even when the totals fit. Cap by the smallest budget less
            # the headroom and less the weight that device has to hold,
            # otherwise a model that trivially fits is refused.
            #
            # That weight is at least the pinned units, which the head's device
            # takes whole. An average share alone is not enough: `attempt` only
            # relaxes the reserve on the OTHER cards, so once the head's budget
            # goes negative every step fails and the plan is refused. A four-layer
            # model whose head is larger than half the weights (share 24 KiB,
            # head 32 KiB) was rejected on 2 x 8 GiB for exactly that reason.
            share = max(-(-total // len(devices)), pinned_bytes)
            per_device_cap = min(raw_budgets.values()) - headroom - share
            activation_reserve_bytes = int(max(0, min(activation_reserve_bytes, per_device_cap)))
        else:
            # Mirror accelerate: reserve the largest single placement unit.
            activation_reserve_bytes = max((s for _, s in units), default=0)
    if isinstance(activation_reserve_bytes, Mapping):
        reserve = {d: _parse_size(activation_reserve_bytes.get(d, 0)) for d in devices}
    else:
        reserve = dict.fromkeys(devices, int(activation_reserve_bytes))
    activation_reserve_bytes = max(reserve.values()) if reserve else 0

    def attempt(head_device: int):
        """Try progressively smaller activation reserves on the non-head devices.

        The head device always keeps ``activation_reserve + headroom`` free; when
        block granularity makes the ideal split infeasible we take the slack out
        of the *other* cards, never out of the head's logit headroom.

        Only an auto-derived reserve is relaxed. A caller-supplied one is a hard
        constraint, so an explicit reserve either fits or the plan is refused.
        """
        steps = 1 if reserve_is_explicit else 21
        for step in range(steps):
            r = _fill(head_device, max(reserve.values()) * (20 - step) // 20)
            if r is not None:
                return r
        return None

    def _fill(head_device: int, other_reserve: int):
        budget = {
            d: raw_budgets[d] - (reserve[d] + headroom if d == head_device
                                 else min(reserve[d], other_reserve))
            for d in devices
        }
        budget[head_device] -= pinned_bytes
        if any(b < 0 for b in budget.values()):
            return None
        free = [(name, size) for name, size in units if name not in pinned]
        used, assign = _walk_in_order(free, budget)
        if used is None:
            # The in-order walk is next-fit with a first-fit rescue, and neither
            # backtracks, so it can run out of room while a valid assignment
            # exists: capacities 10 and 10 with units 7, 6, 3, 2, 2 build loads
            # of 9 and 9 and then reject the last unit, although 7+3 and 6+2+2
            # both fit. Differently sized units -- a vision tower, a projector
            # and decoder blocks in one model -- produce exactly that shape. So
            # before giving up, repack largest-first into the tightest device
            # that still fits. This runs ONLY after the in-order walk has
            # failed, so a plan that already worked is never rewritten: layout
            # in definition order keeps consecutive layers together, which
            # costs fewer cross-device hops than a size-sorted one.
            used, assign = _best_fit(free, budget)
        if used is None:
            return None
        for p in pinned:
            assign[p] = head_device
            used[head_device] += unit_size[p]
        weight_bytes = {d: used[d] for d in devices}
        return assign, weight_bytes, {d: budget[d] for d in devices}

    def _walk_in_order(free, budget):
        used = dict.fromkeys(devices, 0)
        assign: dict[str, int] = {}
        cursor = 0
        for name, size in free:
            placed = False
            while cursor < len(devices):
                d = devices[cursor]
                if used[d] + size <= budget[d]:
                    assign[name] = d
                    used[d] += size
                    placed = True
                    break
                cursor += 1
            if not placed:
                # Sequential cursor exhausted: try any earlier device that still
                # has room rather than falling off to CPU.
                for d in devices:
                    if used[d] + size <= budget[d]:
                        assign[name] = d
                        used[d] += size
                        placed = True
                        break
            if not placed:
                return None, None
        return used, assign

    def _best_fit(free, budget):
        """Largest unit first into the device it leaves least room on."""
        used = dict.fromkeys(devices, 0)
        assign: dict[str, int] = {}
        for name, size in sorted(free, key=lambda item: -item[1]):
            room = [(budget[d] - used[d] - size, d) for d in devices
                    if used[d] + size <= budget[d]]
            if not room:
                return None, None
            _, d = min(room)
            assign[name] = d
            used[d] += size
        return used, assign

    if prefer_head_device is not None and prefer_head_device not in devices:
        # Otherwise the candidate loop below skips every option and the failure
        # is reported as a memory shortfall, which sends the caller looking in
        # entirely the wrong place.
        raise ValueError(
            f"prefer_head_device={prefer_head_device} is not one of the usable "
            f"devices {devices}"
        )
    order = [prefer_head_device] if prefer_head_device is not None else list(reversed(devices))
    result = None
    chosen = None
    for cand in order:
        if cand not in devices:
            continue
        result = attempt(cand)
        if result is not None:
            chosen = cand
            break
    if result is None:
        slack = sum(raw_budgets.values()) - total - activation_reserve_bytes * len(devices)
        raise DeviceMapInfeasible(
            "Cannot place the model and keep "
            f"{headroom / _GiB:.3f} GiB free on the output head's device.\n"
            f"  weights                 : {total / _GiB:.3f} GiB\n"
            f"  budgets                 : "
            + ", ".join(f"cuda:{d}={raw_budgets[d] / _GiB:.2f} GiB" for d in devices)
            + "\n"
            f"  per-device act. reserve : {activation_reserve_bytes / _GiB:.3f} GiB\n"
            f"  slack after weights     : {slack / _GiB:.3f} GiB\n"
            f"  head headroom needed    : {headroom / _GiB:.3f} GiB\n"
            "Reduce rows_per_chunk, reduce retained_rows (run the log-softmax under "
            "no_grad), use a smaller quantisation, or add a GPU. Refusing to offload "
            "to CPU/disk: bitsandbytes cannot load a partially offloaded 4-bit model."
        )

    assign, weight_bytes, budgets = result
    # Sanity: the map must cover every parameter and buffer.
    uncovered = []
    for pname, _ in list(model.named_parameters()) + list(model.named_buffers()):
        if not any(pname == k or pname.startswith(k + ".") for k in assign):
            uncovered.append(pname)
    if uncovered:
        raise DeviceMapInfeasible(
            f"internal error: {len(uncovered)} parameters are not covered by the "
            f"device map, e.g. {uncovered[:5]}"
        )

    return DeviceMapPlan(
        device_map=assign,
        head_module=head_name,
        head_device=chosen,
        headroom_bytes=headroom,
        budgets=budgets,
        raw_budgets=raw_budgets,
        weight_bytes=weight_bytes,
        activation_reserve_bytes=activation_reserve_bytes,
        total_weight_bytes=total,
        no_split_module_classes=no_split,
        tied_to_head=tied_to_head,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# pre-load convenience
# --------------------------------------------------------------------------- #
def build_meta_model(model_name_or_path: str, **from_pretrained_kwargs: Any):
    """Instantiate the model on the meta device, quantiser included.

    Returns ``(model, hf_quantizer, config)``. Costs no GPU memory and no weight
    IO, so the plan can be computed before ``from_pretrained``.
    """
    from accelerate import init_empty_weights
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name_or_path, **from_pretrained_kwargs)
    trust_remote_code = bool(from_pretrained_kwargs.get("trust_remote_code", False))
    auto_cls = _auto_class_for(config, trust_remote_code=trust_remote_code)
    hf_quantizer = None
    qcfg = getattr(config, "quantization_config", None)
    if qcfg is not None:
        from transformers.quantizers import AutoHfQuantizer

        hf_quantizer = AutoHfQuantizer.from_config(qcfg)
    with init_empty_weights():
        # `trust_remote_code` has to travel to `from_config` too. AutoConfig can
        # resolve a dynamic config on its own, but the model class then lives in
        # the same Hub repo and `from_config` only fetches it when it is allowed
        # to, so without this a remote-code checkpoint raises instead of
        # honouring the option the caller already passed.
        model = auto_cls.from_config(config, trust_remote_code=True) if trust_remote_code \
            else auto_cls.from_config(config)
    model.eval()
    if hf_quantizer is not None:
        # Swap in the quantised Linear classes so the size table matches the
        # bytes the loader will really allocate.
        try:
            hf_quantizer.preprocess_model(model=model, device_map=None, keep_in_fp32_modules=[])
        except Exception:
            pass
    return model, hf_quantizer, config


def _auto_class_for(config: Any, trust_remote_code: bool = False):
    """Pick the right Auto* class without hardcoding an architecture."""
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoModel,
    )

    archs = list(getattr(config, "architectures", None) or [])
    # A dynamic (remote-code) config is not in any `_model_mapping`, so the
    # mapping walk below always falls through to AutoModel and construction
    # fails. The repo says which Auto class it registered for; `from_config`
    # looks the model class up under exactly that name, so pick the matching one.
    auto_map = getattr(config, "auto_map", None)
    if trust_remote_code and isinstance(auto_map, Mapping):
        for auto_cls in (AutoModelForImageTextToText, AutoModelForCausalLM, AutoModel):
            if auto_cls.__name__ in auto_map:
                return auto_cls
    for auto_cls in (AutoModelForImageTextToText, AutoModelForCausalLM):
        mapping = getattr(auto_cls, "_model_mapping", None)
        if mapping is None:
            continue
        try:
            model_cls = mapping[type(config)]
        except (KeyError, TypeError):
            continue
        names = {model_cls.__name__} if not isinstance(model_cls, (list, tuple)) else {
            c.__name__ for c in model_cls
        }
        if not archs or names & set(archs):
            return auto_cls
        return auto_cls
    return AutoModel


def plan_device_map_for_pretrained(
    model_name_or_path: str,
    *,
    max_memory: Mapping[Any, Any] | None = None,
    rows_per_chunk: int = 128,
    retained_rows: int = 0,
    vocab_size: int | None = None,
    softcapped: bool | None = None,
    temperature_scaled: bool = False,
    headroom_bytes: int | None = None,
    safety_bytes: int = 256 * 1024 ** 2,
    activation_reserve_bytes: int | Mapping[int, Any] | None = None,
    free_space_policy: str = "balanced",
    prefer_head_device: int | None = None,
    trust_remote_code: bool = False,
    **config_kwargs: Any,
) -> DeviceMapPlan | None:
    """Plan a device map straight from a checkpoint id or path.

    Builds the model on the meta device, so this is cheap and touches no GPU.
    Returns ``None`` when fewer than two GPUs are usable.
    """
    if len(_usable_devices(max_memory)) < 2:
        return None
    model, hf_quantizer, _config = build_meta_model(
        model_name_or_path, trust_remote_code=trust_remote_code, **config_kwargs
    )
    return plan_device_map(
        model,
        max_memory=max_memory,
        rows_per_chunk=rows_per_chunk,
        retained_rows=retained_rows,
        vocab_size=vocab_size,
        softcapped=softcapped,
        temperature_scaled=temperature_scaled,
        headroom_bytes=headroom_bytes,
        safety_bytes=safety_bytes,
        activation_reserve_bytes=activation_reserve_bytes,
        free_space_policy=free_space_policy,
        hf_quantizer=hf_quantizer,
        prefer_head_device=prefer_head_device,
    )
