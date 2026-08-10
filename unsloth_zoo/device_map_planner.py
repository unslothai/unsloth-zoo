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
              + rows_per_chunk * vocab * s   if logit_scaled      # scaled copy
              + rows_per_chunk * vocab * s   if softcapped        # tanh temporary
              + rows_per_chunk * vocab * 4   if temperature != 1  # 2nd fp32 buffer
              + rows_per_chunk * vocab * 4   if retained_rows      # backward grad buffer
    retained  = retained_rows  * vocab * (4 + (s if softcapped else 0))
    headroom  = per_chunk + retained + safety_bytes

* ``rows_per_chunk * vocab * s`` is ``chunk_hidden @ lm_head.T``.
* ``+ 4`` bytes/element is the ``chunk_logits.to(torch.float32)`` copy, which is
  live at the same time as the low-precision tensor it was copied from.
* the scaling term is the out-of-place ``chunk_logits * logit_scale_multiply``
  (or ``/ logit_scale_divide``), which is live alongside the tensor it reads.
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
    from unsloth_zoo.device_map_planner import plan_device_map_for_pretrained

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
  the tied input embedding is pinned to the head's device; any other group of
  modules sharing a parameter is placed as one unit, since the size table counts
  a shared tensor once,
* vision towers / multimodal projectors / final norms are just ordinary split
  units and are placed by the same greedy walk.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

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


class _SearchExhausted(Exception):
    """Internal: the bounded exact-packing search hit its node budget."""


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
    logit_scaled: bool = False,
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
        logit_scaled: the caller passes a non-zero ``logit_scale_multiply`` or
            ``logit_scale_divide``. Both are out-of-place, so one more buffer of
            the chunk shape in the logit dtype is live before the float32 copy.
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
    if logit_scaled:
        per_chunk += rows_per_chunk * vocab_size * s
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
    """Per-device budget before any reserve, after the quantiser's own haircut."""
    weight_bytes: dict[int, int]
    activation_reserve_bytes: int
    """Largest reserve asked for. See ``activation_reserve_by_device`` for what
    the accepted packing actually kept, which is smaller wherever a per-device
    mapping was passed or an auto-derived reserve had to be relaxed."""
    total_weight_bytes: int
    no_split_module_classes: list[str]
    activation_reserve_by_device: dict[int, int] = field(default_factory=dict)
    """Reserve the accepted packing really kept free, per device."""
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
            f"activation reserve : {self.activation_reserve_bytes / _GiB:.3f} GiB requested",
        ]
        if self.tied_to_head:
            lines.append(f"tied to head       : {self.tied_to_head}")
        for d in sorted(self.raw_budgets):
            w = self.weight_bytes.get(d, 0)
            kept = self.activation_reserve_by_device.get(d)
            lines.append(
                f"  cuda:{d}  budget {self.raw_budgets[d] / _GiB:6.2f} GiB"
                f"  weights {w / _GiB:6.3f} GiB"
                f"  free {(self.raw_budgets[d] - w) / _GiB:6.3f} GiB"
                + (f"  reserve {kept / _GiB:6.3f} GiB" if kept is not None else "")
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
    # `[]` is the model declaring that nothing is atomic, which several
    # transformers models do (camembert, colpali, colqwen2, efficientnet, fuyu).
    # Only `None`, the base-class default, means "not declared, go and detect".
    if classes is not None:
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


def _keep_in_fp32_modules(model: nn.Module, hf_quantizer: Any = None) -> list[str]:
    """Modules the loader will leave in float32, mirroring transformers.

    ``from_pretrained`` hands this list to ``preprocess_model`` so the quantiser
    skips those modules and they keep the load dtype. Passing ``[]`` instead
    sizes them at the quantised width while the loader keeps them wide, which
    undercounts the weights of a tight plan and OOMs during loading. Both
    bitsandbytes quantisers set ``use_keep_in_fp32_modules``, and real
    checkpoints do declare the list (gpt-oss keeps its layer norms, the T5
    family keeps `wo`), so this is not a rare path.

    The gate is the same as ``modeling_utils``: the plain list applies under
    float16 or when the quantiser asks for it, the strict list under float16 or
    bfloat16, both against the dtype the quantiser resolved.
    """
    dtype = _model_dtype(model)
    update = getattr(hf_quantizer, "update_dtype", None)
    if callable(update):
        try:
            dtype = update(dtype)
        except Exception:
            pass
    keep: list[str] = []
    plain = getattr(model, "_keep_in_fp32_modules", None)
    if plain and (dtype == torch.float16 or
                  getattr(hf_quantizer, "use_keep_in_fp32_modules", False)):
        keep.extend(plain)
    strict = getattr(model, "_keep_in_fp32_modules_strict", None)
    if strict and dtype in (torch.float16, torch.bfloat16):
        keep.extend(strict)
    return list(dict.fromkeys(keep))


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

    The fp32 exceptions apply with no quantiser at all: an fp16 checkpoint whose
    class declares ``_keep_in_fp32_modules`` (the T5 family keeps ``wo`` that
    way) has those tensors loaded as float32 while the meta model still holds
    them in the config dtype, so without this the planner charges half.
    """
    dtype = _model_dtype(model)
    if dtype is None:
        return {}
    fp32_modules = _keep_in_fp32_modules(model, hf_quantizer)
    fp32_dtypes = {
        name: torch.float32
        for name, _ in model.named_parameters()
        if any(m in name for m in fp32_modules)
    } if fp32_modules else {}
    if hf_quantizer is None:
        return {"dtype": dtype, "special_dtypes": fp32_dtypes} if fp32_dtypes else {}
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
    actually allocate; otherwise falls back to a plain walk.

    The two available implementations take the quantiser differently and neither
    tolerates the other's arguments, so dispatch on the signature:

    * ``transformers.integrations.accelerate.compute_module_sizes(model,
      hf_quantizer=...)`` asks the quantiser for the element size of each
      parameter, which is exact. It returns a 2-tuple.
    * ``accelerate.utils.modeling.compute_module_sizes(model, dtype=...,
      special_dtypes=...)`` has no quantiser argument, so it gets the storage
      dtype and the not-converted modules instead (see
      :func:`_quantized_size_kwargs`).

    Passing the wrong pair is not a near miss. ``hf_quantizer`` in accelerate's
    second positional slot is read as a dtype and raises, and a ``dtype`` keyword
    on the transformers version is unexpected, so either mistake ends in the
    plain walk below -- which measures a preprocessed meta ``Params4bit`` at the
    full unpacked shape in float32, several times the real size of a 4-bit
    checkpoint.
    """
    size_kwargs = _quantized_size_kwargs(model, hf_quantizer)
    for mod_path in ("transformers.integrations.accelerate", "accelerate.utils.modeling"):
        try:
            module = __import__(mod_path, fromlist=["compute_module_sizes"])
            fn = getattr(module, "compute_module_sizes")
            accepted = inspect.signature(fn).parameters
        except Exception:
            continue
        attempts: list[dict[str, Any]] = []
        if hf_quantizer is not None and "hf_quantizer" in accepted:
            attempts.append({"hf_quantizer": hf_quantizer})
        elif size_kwargs:
            supported = {k: v for k, v in size_kwargs.items() if k in accepted}
            # Drop the extras one at a time rather than all at once: an older
            # accelerate without `special_dtypes` still gets the storage dtype,
            # which is the term that actually moves the number.
            if supported:
                attempts.append(supported)
            if "dtype" in supported and len(supported) > 1:
                attempts.append({"dtype": supported["dtype"]})
        # Last resort. Correct for an unquantised model, an over-estimate for a
        # quantised one -- which refuses rather than OOMs, so it is the safe way
        # to be wrong.
        attempts.append({})
        for kwargs in attempts:
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


def _adjust_budgets_for_quantizer(
    raw_budgets: Mapping[int, int], hf_quantizer: Any
) -> tuple[dict[int, int], str | None]:
    """Apply the quantiser's own budget haircut, exactly as transformers does.

    ``transformers.modeling_utils._get_device_map`` runs
    ``hf_quantizer.adjust_max_memory(max_memory)`` before it infers a device map,
    and the bitsandbytes quantisers hold back 10% of every budget for the buffers
    they allocate while quantising. An explicit device map skips that whole path,
    so without this the planner hands out memory the loader still needs and a plan
    that measured as feasible OOMs while the checkpoint loads.

    Only ever lowers a budget: a quantiser that hands back more than was asked for
    cannot talk the planner into overcommitting a card.
    """
    budgets = dict(raw_budgets)
    adjust = getattr(hf_quantizer, "adjust_max_memory", None)
    if not callable(adjust):
        return budgets, None
    try:
        adjusted = adjust(dict(raw_budgets))
    except Exception:
        return budgets, None
    if not isinstance(adjusted, Mapping):
        return budgets, None
    lowered: list[int] = []
    for d in raw_budgets:
        if d not in adjusted:
            continue
        try:
            value = _parse_size(adjusted[d])
        except (TypeError, ValueError):
            continue
        value = min(int(value), raw_budgets[d])
        if value < budgets[d]:
            budgets[d] = value
            lowered.append(d)
    if not lowered:
        return budgets, None
    note = (
        f"{type(hf_quantizer).__name__}.adjust_max_memory lowered the budget on "
        + ", ".join(
            f"cuda:{d} {raw_budgets[d] / _GiB:.3f} -> {budgets[d] / _GiB:.3f} GiB"
            for d in sorted(lowered)
        )
        + " (memory the quantiser keeps for its own load-time buffers)"
    )
    return budgets, note


def _tied_parameter_groups(model: nn.Module) -> list[list[str]]:
    """Names of every tensor that is one shared object under several names.

    ``compute_module_sizes`` counts a shared tensor once, so two modules that
    share a parameter are only correctly sized when they land on the same device.
    Grouping by object identity (rather than ``data_ptr``) is what makes this work
    on a meta model, where every tensor reports the same null pointer.
    """
    by_object: dict[int, list[str]] = {}
    for name, tensor in list(model.named_parameters(remove_duplicate=False)) + \
                        list(model.named_buffers(remove_duplicate=False)):
        by_object.setdefault(id(tensor), []).append(name)
    return [names for names in by_object.values() if len(names) > 1]


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
        # `_modules`, not `named_children()`: that helper drops repeated
        # registrations of the SAME object, so `nn.ModuleList([block] * n)`
        # yields one name and the aliases never become placement units. The
        # coverage check enumerates parameters with `remove_duplicate=False`, so
        # the missing aliases were then reported as an internal failure on a
        # model that fits. Sizing counts the shared tensors once, so an alias
        # unit is zero bytes, and the tied-parameter grouping below co-locates
        # it with the name that carries the weight.
        children = [(n, m) for n, m in module._modules.items() if m is not None]
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
        # Size direct state from the same quantiser-aware table as everything
        # else, by subtracting the children from the subtree. `element_size()` on
        # the live tensor is the number this module exists to avoid: a
        # preprocessed meta `Params4bit` reports the full unpacked shape in
        # float32, so the units would stop summing to `total_weight_bytes` and
        # the planner would refuse a model that fits (or overcommit one that
        # does not).
        child_bytes = sum(
            sizes.get(f"{prefix}.{child_name}" if prefix else child_name, 0)
            for child_name, _ in children
        )
        # Emit the unit whenever direct state exists, even at zero bytes: a
        # tensor tied to one counted elsewhere weighs nothing here, and dropping
        # its unit would leave the parameter uncovered and fail the coverage
        # check at the end of `plan_device_map`.
        own = max(0, size - child_bytes) if own_tensors else 0
        if own_tensors and prefix:
            units.append((prefix, own))
        elif own_tensors:
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
                units.append((tensor_name, sizes.get(
                    tensor_name, tensor.numel() * tensor.element_size()
                )))
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
    logit_scaled: bool = False,
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
            ``None`` (default) reads ``final_logit_softcapping`` (or its
            RecurrentGemma alias ``logits_soft_cap``) off the config.
        logit_scaled: whether the caller passes a non-zero
            ``logit_scale_multiply`` or ``logit_scale_divide``.
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
        no_split_module_classes: override the detected block classes. ``[]``
            removes every no-split constraint, so blocks may be split at their
            children; ``None`` (default) detects the classes from the model.
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

    notes: list[str] = []

    raw_budgets: dict[int, int] = {}
    for d in devices:
        if max_memory is not None:
            raw = max_memory[d] if d in max_memory else max_memory[str(d)]
            raw_budgets[d] = _parse_size(raw)
        else:
            free, _total = torch.cuda.mem_get_info(d)
            raw_budgets[d] = int(free)
    raw_budgets, quantizer_note = _adjust_budgets_for_quantizer(raw_budgets, hf_quantizer)
    if quantizer_note is not None:
        notes.append(quantizer_note)

    # `[]` is a real override -- "split anywhere, no block is atomic" -- and is
    # not the same request as `None`, which means "detect the block classes".
    no_split = (
        list(no_split_module_classes) if no_split_module_classes is not None
        else resolve_no_split_classes(model)
    )
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
        # `logits_soft_cap` is the RecurrentGemma spelling of the same knob, and
        # the repository's own `_detect_logit_softcap` already treats them as
        # aliases. Missing it drops the tanh temporary and the retained soft-cap
        # buffer from the headroom, so the head's card is under-reserved.
        softcapped = any(
            bool(getattr(h, name, None))
            for h in (cfg, getattr(cfg, "text_config", None))
            for name in ("final_logit_softcapping", "logits_soft_cap")
        )

    if headroom_bytes is None:
        headroom = logit_headroom_bytes(
            width, rows_per_chunk,
            logit_dtype=logit_dtype,
            retained_rows=retained_rows,
            softcapped=softcapped,
            logit_scaled=logit_scaled,
            temperature_scaled=temperature_scaled,
            safety_bytes=safety_bytes,
        )
    else:
        headroom = int(headroom_bytes)

    if free_space_policy not in ("balanced", "head_max"):
        raise ValueError(f"free_space_policy must be 'balanced' or 'head_max', got {free_space_policy!r}")

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
    # ... and up as well as down. When the head sits inside a module whose class
    # is in `no_split_module_classes`, the unit is that atomic ANCESTOR and the
    # head's own name never appears, so a descendants-only filter emptied
    # `pinned` and the greedy walk was free to put the head's card's contents
    # anywhere -- headroom reserved on a GPU that does not hold the head.
    pinned = list(dict.fromkeys(
        u for p in pinned for u, _ in units
        if u == p or u.startswith(p + ".") or p.startswith(u + ".")
    ))

    # Co-locate every group of units that shares a parameter, not just the
    # input/output embedding pair. `compute_module_sizes` counts a shared tensor
    # once, so splitting its owners across devices makes accelerate materialise a
    # second copy that `weight_bytes` never accounted for, and a plan that
    # measured as feasible OOMs during dispatch. Encoder/decoder models that tie
    # `shared` to both embedding tables are the common case.
    unit_names = [u for u, _ in units]
    parent = {u: u for u in unit_names}

    def _root(u: str) -> str:
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def _owning_unit(param_name: str) -> str | None:
        best = None
        for u in unit_names:
            if (param_name == u or param_name.startswith(u + ".")) and \
               (best is None or len(u) > len(best)):
                best = u
        return best

    for shared_names in _tied_parameter_groups(model):
        owners = list(dict.fromkeys(
            u for u in (_owning_unit(n) for n in shared_names) if u is not None
        ))
        for other in owners[1:]:
            a, b = _root(owners[0]), _root(other)
            if a != b:
                parent[a] = b

    # A tied group that touches the head travels with the head; the rest just
    # have to stay together.
    pinned_roots = {_root(p) for p in pinned}
    tied_with_head = [u for u in unit_names if u not in pinned and _root(u) in pinned_roots]
    if tied_with_head:
        pinned.extend(tied_with_head)
        tied_to_head.extend(u for u in tied_with_head if u not in tied_to_head)
        notes.append(
            "tied parameters: " + ", ".join(tied_with_head) +
            " share storage with the head's group and are pinned to its device"
        )
    pinned_bytes = sum(unit_size[p] for p in pinned)

    # Placement items: one per co-location group, in first-appearance order, so
    # the walks below cannot separate units that share a tensor.
    free_groups: list[tuple[tuple[str, ...], int]] = []
    _group_index: dict[str, int] = {}
    for name, size in units:
        if name in pinned:
            continue
        root = _root(name)
        if root in _group_index:
            at = _group_index[root]
            names, total_size = free_groups[at]
            free_groups[at] = (names + (name,), total_size + size)
        else:
            _group_index[root] = len(free_groups)
            free_groups.append(((name,), size))

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
        # What this attempt really keeps free per device, which is what the plan
        # reports: a per-device mapping is not one number, and `attempt` relaxes
        # the reserve on the non-head cards, so a single scalar would promise
        # headroom the accepted packing did not keep.
        kept = {d: reserve[d] if d == head_device else min(reserve[d], other_reserve)
                for d in devices}
        budget = {
            d: raw_budgets[d] - (kept[d] + headroom if d == head_device else kept[d])
            for d in devices
        }
        budget[head_device] -= pinned_bytes
        if any(b < 0 for b in budget.values()):
            return None
        used, assign = _walk_in_order(free_groups, budget, head_device)
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
            used, assign = _best_fit(free_groups, budget)
        if used is None:
            # Best-fit is not exact either: capacities 10 and 10 with units
            # 6, 5, 3, 2, 2, 2 pack 6+3 and 5+2+2 and then reject the last 2,
            # although 6+2+2 and 5+3+2 both fit. Last resort before refusing a
            # model that demonstrably fits.
            used, assign = _exact_fit(free_groups, budget)
        if used is None:
            return None
        for p in pinned:
            assign[p] = head_device
            used[head_device] += unit_size[p]
        weight_bytes = {d: used[d] for d in devices}
        # `budget` is the *remaining* capacity the packing walks were allowed to
        # use, so the head device has already paid for the pinned units. The
        # public field means "budget available to weights after every reserve"
        # and `weight_bytes` counts those same pinned units, so hand the pinned
        # bytes back or the head device reports weights above its own budget.
        public_budgets = {d: budget[d] for d in devices}
        public_budgets[head_device] += pinned_bytes
        return assign, weight_bytes, public_budgets, kept

    def _walk_in_order(free, budget, head_device: int):
        # `head_max` means "push as much weight off the head's card as possible".
        # Walking plain device order defeats that whenever the head is not the
        # last device -- `prefer_head_device = 0`, or a higher-index candidate
        # that could not hold the pinned head -- because the walk then fills the
        # head's card first and leaves the later GPUs empty.
        order = devices
        if free_space_policy == "head_max":
            order = [d for d in devices if d != head_device] + [head_device]
        used = dict.fromkeys(devices, 0)
        assign: dict[str, int] = {}
        cursor = 0
        for names, size in free:
            placed = False
            while cursor < len(order):
                d = order[cursor]
                if used[d] + size <= budget[d]:
                    assign.update(dict.fromkeys(names, d))
                    used[d] += size
                    placed = True
                    break
                cursor += 1
            if not placed:
                # Sequential cursor exhausted: try any earlier device that still
                # has room rather than falling off to CPU.
                for d in order:
                    if used[d] + size <= budget[d]:
                        assign.update(dict.fromkeys(names, d))
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
        for names, size in sorted(free, key=lambda item: -item[1]):
            room = [(budget[d] - used[d] - size, d) for d in devices
                    if used[d] + size <= budget[d]]
            if not room:
                return None, None
            _, d = min(room)
            assign.update(dict.fromkeys(names, d))
            used[d] += size
        return used, assign

    def _exact_fit(free, budget, node_budget=20000, max_units=512):
        """Bounded depth-first packing, largest unit first.

        Runs only after both heuristics have failed, so it can turn a refusal
        into a plan and can never rewrite one that already worked. Two prunes
        keep it cheap: at each depth a device whose remaining capacity was
        already tried is skipped (interchangeable), and the whole search gives up
        after ``node_budget`` placements rather than exploring an exponential
        tree. Deep unit lists (a fine-grained ``no_split_module_classes = []``)
        are left to the heuristics, which handle uniform units well.
        """
        order = sorted(free, key=lambda item: -item[1])
        if not order or len(order) > max_units:
            return None, None
        if sum(size for _, size in order) > sum(budget[d] for d in devices):
            return None, None
        remaining = {d: budget[d] for d in devices}
        assign: dict[str, int] = {}
        visited = 0

        def place(i: int) -> bool:
            nonlocal visited
            if i == len(order):
                return True
            visited += 1
            if visited > node_budget:
                raise _SearchExhausted
            names, size = order[i]
            tried: set[int] = set()
            for d in devices:
                room = remaining[d]
                if size > room or room in tried:
                    continue
                tried.add(room)
                remaining[d] -= size
                assign.update(dict.fromkeys(names, d))
                if place(i + 1):
                    return True
                remaining[d] += size
                for n in names:
                    assign.pop(n, None)
            return False

        try:
            fitted = place(0)
        except (_SearchExhausted, RecursionError):
            return None, None
        if not fitted:
            return None, None
        return {d: budget[d] - remaining[d] for d in devices}, assign

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
        reserved_total = sum(reserve.values())
        slack = sum(raw_budgets.values()) - total - reserved_total
        raise DeviceMapInfeasible(
            "Cannot place the model and keep "
            f"{headroom / _GiB:.3f} GiB free on the output head's device.\n"
            f"  weights                 : {total / _GiB:.3f} GiB\n"
            f"  budgets                 : "
            + ", ".join(f"cuda:{d}={raw_budgets[d] / _GiB:.2f} GiB" for d in devices)
            + "\n"
            "  per-device act. reserve : "
            + ", ".join(f"cuda:{d}={reserve[d] / _GiB:.3f} GiB" for d in devices)
            + f" ({reserved_total / _GiB:.3f} GiB total)\n"
            f"  slack after weights     : {slack / _GiB:.3f} GiB\n"
            f"  head headroom needed    : {headroom / _GiB:.3f} GiB\n"
            "Reduce rows_per_chunk, reduce retained_rows (run the log-softmax under "
            "no_grad), use a smaller quantisation, or add a GPU. Refusing to offload "
            "to CPU/disk: bitsandbytes cannot load a partially offloaded 4-bit model."
        )

    assign, weight_bytes, budgets, reserve_kept = result
    # Sanity: the map must cover every parameter and buffer.
    # `remove_duplicate=False`, because accelerate's own `check_device_map`
    # walks `state_dict()`, which lists a tied tensor under every name it has.
    # Covering only the deduplicated names lets a map through that accelerate
    # then rejects with "does not give any device for the following parameters".
    uncovered = []
    for pname, _ in list(model.named_parameters(remove_duplicate=False)) + \
                    list(model.named_buffers(remove_duplicate=False)):
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
        activation_reserve_by_device=reserve_kept,
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
        model = _from_config_remote_aware(auto_cls, config, from_pretrained_kwargs) \
            if trust_remote_code else auto_cls.from_config(config)
    model.eval()
    if hf_quantizer is not None:
        # Swap in the quantised Linear classes so the size table matches the
        # bytes the loader will really allocate.
        try:
            hf_quantizer.preprocess_model(
                model=model, device_map=None,
                keep_in_fp32_modules=_keep_in_fp32_modules(model, hf_quantizer),
            )
        except Exception:
            pass
    return model, hf_quantizer, config


_HUB_KWARGS = (
    "cache_dir", "force_download", "proxies", "token", "revision",
    "local_files_only", "code_revision", "repo_type",
)


def _from_config_remote_aware(auto_cls: Any, config: Any, from_pretrained_kwargs: Mapping[str, Any]):
    """``from_config`` for a remote-code checkpoint, with the Hub options applied.

    Resolving a dynamic model class is a second Hub lookup, and the config's own
    options do not travel with it: a private repo then fails to authenticate,
    ``code_revision`` is ignored and ``local_files_only`` is violated.

    They cannot simply be handed to ``from_config``. It forwards its leftover
    kwargs to ``_from_config``, which passes them straight to ``cls(config,
    **kwargs)``, and a model ``__init__`` takes the config alone -- so
    ``token=...`` there is a ``TypeError`` on every ordinary checkpoint. Resolve
    the class explicitly instead, and fall back to plain ``from_config`` for
    anything that is not a dynamic class.

    The fallback is chosen before the lookup, never after it: retrying a *failed*
    resolution without the options would go to the network under
    ``local_files_only``, drop the token, or take a different ``code_revision``
    than the caller asked for, so a genuine Hub failure is raised as is.
    """
    hub_kwargs = {k: from_pretrained_kwargs[k] for k in _HUB_KWARGS
                  if k in from_pretrained_kwargs}
    auto_map = getattr(config, "auto_map", None)
    class_ref = auto_map.get(auto_cls.__name__) if isinstance(auto_map, Mapping) else None
    if class_ref and hub_kwargs:
        try:
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
        except ImportError:
            get_class_from_dynamic_module = None
        # Hand over the reference UNSPLIT, with the model's own path. The
        # `repo--module.Class` form names a separate code repository, and
        # `get_class_from_dynamic_module` compares the two before deciding
        # anything: `if code_revision is None and pretrained_model_name_or_path
        # == repo_id: code_revision = revision`. Splitting it here and passing
        # the code repo as the path makes those two always equal, so a model
        # `revision` would be looked for as a branch of the unrelated code repo.
        model_path = getattr(config, "_name_or_path", None) \
            or getattr(config, "name_or_path", None) or ""
        if get_class_from_dynamic_module is not None and (model_path or "--" in class_ref):
            model_cls = get_class_from_dynamic_module(class_ref, model_path, **hub_kwargs)
            return model_cls._from_config(config)
    return auto_cls.from_config(config, trust_remote_code=True)


def _auto_class_for(config: Any, trust_remote_code: bool = False):
    """Pick the right Auto* class without hardcoding an architecture."""
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoModelForSeq2SeqLM,
        AutoModel,
    )

    archs = list(getattr(config, "architectures", None) or [])
    # A dynamic (remote-code) config is not in any `_model_mapping`, so the
    # mapping walk below always falls through to AutoModel and construction
    # fails. The repo says which Auto class it registered for; `from_config`
    # looks the model class up under exactly that name, so pick the matching one.
    auto_map = getattr(config, "auto_map", None)
    if trust_remote_code and isinstance(auto_map, Mapping):
        for auto_cls in (AutoModelForImageTextToText, AutoModelForCausalLM,
                         AutoModelForSeq2SeqLM, AutoModel):
            if auto_cls.__name__ in auto_map:
                return auto_cls
    # Seq2Seq is in the walk because an encoder-decoder checkpoint (T5, mT5) is
    # registered ONLY under `AutoModelForSeq2SeqLM`; without it the walk fell
    # through to the bare `AutoModel`, whose meta model has no output head at
    # all, so the plan reserved the logit headroom around some unrelated linear
    # and undercounted the weights the loader would really place.
    first_hit = None
    for auto_cls in (AutoModelForImageTextToText, AutoModelForCausalLM,
                     AutoModelForSeq2SeqLM):
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
        # A config can appear in two mappings -- BART is both `BartForCausalLM`
        # and `BartForConditionalGeneration` -- so let the checkpoint's own
        # `architectures` break the tie, and keep the first hit otherwise.
        if archs and names & set(archs):
            return auto_cls
        if first_hit is None:
            first_hit = auto_cls
    return first_hit if first_hit is not None else AutoModel


def plan_device_map_for_pretrained(
    model_name_or_path: str,
    *,
    max_memory: Mapping[Any, Any] | None = None,
    rows_per_chunk: int = 128,
    retained_rows: int = 0,
    vocab_size: int | None = None,
    softcapped: bool | None = None,
    logit_scaled: bool = False,
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
        logit_scaled=logit_scaled,
        temperature_scaled=temperature_scaled,
        headroom_bytes=headroom_bytes,
        safety_bytes=safety_bytes,
        activation_reserve_bytes=activation_reserve_bytes,
        free_space_policy=free_space_policy,
        hf_quantizer=hf_quantizer,
        prefer_head_device=prefer_head_device,
    )
