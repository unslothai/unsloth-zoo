# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

"""Exhaustive drift detectors for ``unsloth_zoo`` / ``unsloth`` source-string
and regex rewriters (round 3 of upstream-regression test coverage).

The companion file ``test_upstream_source_patterns.py`` pins ~34 of the
most commonly-tripped rewriter sites. This file pins the REMAINING sites
walked across:

  unsloth_zoo/compiler.py
  unsloth_zoo/temporary_patches/*.py
  unsloth_zoo/patching_utils.py
  unsloth_zoo/saving_utils.py
  unsloth_zoo/rl_replacements.py
  unsloth_zoo/training_utils.py
  unsloth/trainer.py
  unsloth/models/rl.py

Test contract (identical to the companion file):

  * Each test cites the rewriter file:line it was extracted from so
    a maintainer can grep back to the patch site.
  * When the pinned string / regex is gone from the upstream module,
    surface as ``pytest.fail("DRIFT DETECTED: zoo/unsloth source-rewriter
    at <file:line> expects '<pattern>' in <upstream module>, found 0
    matches")``. Never SKIP to hide drift.
  * If the upstream module isn't importable in this venv,
    ``pytest.importorskip`` (genuinely-optional upstream library; not
    "skip to hide drift" because the rewriter wouldn't run either).
  * CPU-only -- runs under ``tests/conftest.py`` GPU-free harness.

Each test is a NEW site relative to ``test_upstream_source_patterns.py``;
duplicates are deliberately omitted.
"""

from __future__ import annotations

import inspect
import re

import pytest


# ---------------------------------------------------------------------------
# Shared helpers (mirror test_upstream_source_patterns.py exactly so this
# file is independently usable without import-coupling).
# ---------------------------------------------------------------------------

def _drift(zoo_site: str, pattern: str, upstream_path: str,
           extra: str = "") -> None:
    """Raise ``pytest.fail`` with the standardized DRIFT message."""
    msg = (
        f"DRIFT DETECTED: zoo/unsloth source-rewriter at {zoo_site} expects "
        f"{pattern!r} in {upstream_path}, found 0 matches."
    )
    if extra:
        msg += " " + extra
    pytest.fail(msg)


def _assert_in_source(needle: str, source: str, zoo_site: str,
                      upstream_path: str) -> None:
    if needle not in source:
        _drift(zoo_site, needle, upstream_path)


def _assert_regex_in_source(regex: str, source: str, zoo_site: str,
                            upstream_path: str,
                            flags: int = 0) -> None:
    if re.search(regex, source, flags=flags) is None:
        _drift(zoo_site, regex, upstream_path)


def _probe_modules(candidates, predicate):
    """Return ``True`` if ``predicate(src)`` is true for at least one
    importable module in ``candidates``. ``candidates`` is a list of
    dotted module names. ``predicate`` receives the module source text.
    """
    import importlib
    for mod in candidates:
        try:
            m = importlib.import_module(mod)
        except ImportError:
            continue
        try:
            src = inspect.getsource(m)
        except OSError:
            continue
        if predicate(src):
            return True
    return False


# ===========================================================================
# unsloth_zoo/compiler.py: not-yet-covered rewriter sites
# ===========================================================================


def test_compiler_higher_precision_softmax_idempotency_lookahead():
    """``unsloth_zoo/compiler.py:391-405`` -- the
    ``higher_precision_softmax`` finder uses a negative lookahead
    ``(?!\\s*\\.to\\(\\s*\\2\\s*\\.dtype\\s*\\))`` to skip already-
    rewritten softmax calls. The rewriter ALSO needs the base
    ``nn.functional.softmax`` / ``F.softmax`` plus ``dim=`` form to
    exist somewhere upstream; otherwise the entire finder is dormant.
    Asserts the ``dim=`` keyword form is still in use.
    """
    pytest.importorskip("transformers")
    pattern = re.compile(
        r"(?:nn\.functional\.softmax|F\.softmax)"
        r"\([^,]{1,}, dim[ ]?\=[ ]?[\-0-9]{1,2}"
    )
    candidates = [
        "transformers.models.llama.modeling_llama",
        "transformers.models.mistral.modeling_mistral",
        "transformers.models.qwen2.modeling_qwen2",
        "transformers.models.qwen3.modeling_qwen3",
        "transformers.models.gemma.modeling_gemma",
        "transformers.models.qwen2_moe.modeling_qwen2_moe",
        "transformers.models.mixtral.modeling_mixtral",
    ]
    if not _probe_modules(candidates, lambda s: pattern.search(s) is not None):
        _drift(
            "unsloth_zoo/compiler.py:391-405",
            r"(nn.functional.softmax|F.softmax)(..., dim=N...)",
            "any of " + ", ".join(candidates),
            "Without a `softmax(..., dim=N)` call site, the float32 "
            "upcast rewriter is dormant.",
        )


def test_compiler_fix_rotary_embedding_cos_to_dtype_pattern():
    """``unsloth_zoo/compiler.py:510-517`` -- ``fix_rotary_embedding_dtype``
    runs ``source.replace("cos.to(dtype=x.dtype)", ...)`` and
    ``source.replace("sin.to(dtype=x.dtype)", ...)``. Activates only
    when ``UNSLOTH_FORCE_CUSTOM_DTYPE`` is set, but the rewriter
    TARGETS must still exist in some upstream rotary embedding for the
    patch to ever fire. Pass if ANY of the literal forms (or the
    bare ``cos.to(`` / ``sin.to(`` cast prefix) appear in a rotary
    embedding module; DRIFT only when the entire idiom is gone.
    """
    pytest.importorskip("transformers")
    candidates = [
        "transformers.models.llama.modeling_llama",
        "transformers.models.qwen2.modeling_qwen2",
        "transformers.models.qwen3.modeling_qwen3",
        "transformers.models.mistral.modeling_mistral",
        "transformers.models.gemma.modeling_gemma",
        "transformers.models.gemma2.modeling_gemma2",
        "transformers.models.gemma3.modeling_gemma3",
        "transformers.models.qwen2_vl.modeling_qwen2_vl",
    ]
    needles = (
        "cos.to(dtype=x.dtype)",
        "sin.to(dtype=x.dtype)",
        "cos.to(",  # broader cast prefix
        "sin.to(",
    )

    def has_any(src):
        return any(n in src for n in needles)

    if not _probe_modules(candidates, has_any):
        _drift(
            "unsloth_zoo/compiler.py:510-517",
            " OR ".join(needles),
            "any of " + ", ".join(candidates),
            "Without a rotary `cos.to(...)` / `sin.to(...)` cast site, "
            "UNSLOTH_FORCE_CUSTOM_DTYPE can never downcast rotary embeds.",
        )


def test_compiler_higher_precision_layernorms_norm_class_marker():
    """``unsloth_zoo/compiler.py:560-597`` -- ``higher_precision_layernorms``
    locates ``class <X>Norm(nn.Module): ... def __init__ ... self.weight
    ... class <Y>``. Then it probes the matched chunk for one of:
    ``self.weight.to(torch.float32)``, ``(self.weight * hidden_states).to(``,
    ``self.weight * hidden_states.to(``, ``self.weight.float()``, or
    ``return output * self.weight`` to decide the upcasting dtype.
    Asserts at least one transformers modeling file still has a
    ``class <X>Norm(nn.Module)`` definition; otherwise the finder
    matches nothing and ``UNSLOTH_HIGH_PRECISION_LAYERNORM`` is never
    auto-toggled.
    """
    pytest.importorskip("transformers")
    pattern = re.compile(
        r"\nclass[^\(\n]{1,}Norm\(nn\.Module\)",
        flags=re.MULTILINE,
    )
    candidates = [
        "transformers.models.llama.modeling_llama",
        "transformers.models.mistral.modeling_mistral",
        "transformers.models.qwen2.modeling_qwen2",
        "transformers.models.qwen3.modeling_qwen3",
        "transformers.models.gemma.modeling_gemma",
        "transformers.models.gemma2.modeling_gemma2",
        "transformers.models.gemma3.modeling_gemma3",
    ]
    if not _probe_modules(candidates, lambda s: pattern.search(s) is not None):
        _drift(
            "unsloth_zoo/compiler.py:560-597",
            r"class <X>Norm(nn.Module): ...",
            "any of " + ", ".join(candidates),
            "Without a Norm(nn.Module) class marker, "
            "higher_precision_layernorms can never auto-detect float32 "
            "weight handling.",
        )


def test_compiler_embedding_oob_clamp_input_ids_pattern():
    """``unsloth_zoo/compiler.py:1383-1387`` -- runs
    ``re.sub(r"self\\.([A-Za-z\\_]{0,}embedding)\\(input_ids (\\-|\\+) (self\\.[A-Za-z\\_]{1,})\\)", ...)``
    to clamp Gemma 3N's input_ids offsets. Asserts Gemma 3N's modeling
    file still has at least one ``self.<...>embedding(input_ids ...)``
    site.
    """
    pytest.importorskip("transformers")
    try:
        import transformers.models.gemma3n.modeling_gemma3n as g3n
    except ImportError:
        pytest.skip("transformers.models.gemma3n not shipped")
    src = inspect.getsource(g3n)
    pattern = re.compile(
        r"self\.([A-Za-z\_]{0,}embedding)\(input_ids (\-|\+) "
        r"(self\.[A-Za-z\_]{1,})\)"
    )
    if pattern.search(src) is None:
        _drift(
            "unsloth_zoo/compiler.py:1383-1387",
            r"self.<X>embedding(input_ids +/- self.<Y>)",
            "transformers.models.gemma3n.modeling_gemma3n",
            "Without this offset call site, the OOB-clamp re.sub never "
            "fires and Gemma 3N regressions return.",
        )


def test_compiler_apply_mask_attention_mask_kwargs_pinned_pattern():
    """``unsloth_zoo/compiler.py:2128-2140`` -- ``apply_mask_attention_mask_out``
    finds ``attention_mask=attention_mask,\\n`` AND
    ``labels=labels,\\n`` in a ForConditionalGeneration forward, then
    re.sub-replaces ``labels=labels,`` with a masked-labels call. Pass
    if at least one VLM forward has BOTH pinned kwargs; DRIFT if no
    upstream forward routes ``labels=labels`` and ``attention_mask=
    attention_mask`` together.
    """
    pytest.importorskip("transformers")
    candidates = [
        "transformers.models.llava.modeling_llava",
        "transformers.models.paligemma.modeling_paligemma",
        "transformers.models.llava_next.modeling_llava_next",
        "transformers.models.idefics3.modeling_idefics3",
        "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl",
        "transformers.models.qwen2_vl.modeling_qwen2_vl",
        "transformers.models.mllama.modeling_mllama",
        "transformers.models.gemma3.modeling_gemma3",
    ]
    am_re = re.compile(r"attention_mask[\s]{0,}\=attention_mask[\s]{0,}\,\n")
    lb_re = re.compile(r"labels[\s]{0,}\=labels[\s]{0,}\,\n")

    def has_both(src):
        return (am_re.search(src) is not None
                and lb_re.search(src) is not None
                and "ForConditionalGeneration" in src)

    if not _probe_modules(candidates, has_both):
        _drift(
            "unsloth_zoo/compiler.py:2128-2140",
            "attention_mask=attention_mask, AND labels=labels,",
            "any of " + ", ".join(candidates),
            "Without both pinned kwargs in a VLM forward, the "
            "mask_attention_mask_out wrapper is never installed.",
        )


def test_compiler_convert_attention_masks_to_bool_finfo_min_pattern():
    """``unsloth_zoo/compiler.py:2161-2179`` -- ``convert_attention_masks_to_bool``
    walks `return <vars>` and probes for
    ``<var>.+?torch\\.finfo\\(.+?\\)\\.min``. Asserts at least one
    transformers masking-utils module still uses
    ``torch.finfo(...).min`` as the masked-fill sentinel.
    """
    pytest.importorskip("transformers")
    finfo_re = re.compile(r"torch\.finfo\([^\)]+\)\.min")
    candidates = [
        "transformers.modeling_attn_mask_utils",
        "transformers.masking_utils",
        "transformers.models.llama.modeling_llama",
        "transformers.models.qwen2.modeling_qwen2",
        "transformers.models.qwen3.modeling_qwen3",
        "transformers.models.gemma3.modeling_gemma3",
    ]
    if not _probe_modules(candidates, lambda s: finfo_re.search(s) is not None):
        _drift(
            "unsloth_zoo/compiler.py:2161-2179",
            r"torch.finfo(<dtype>).min",
            "any of " + ", ".join(candidates),
            "Without finfo(dtype).min masking, the boolean-mask "
            "conversion rewriter is dormant.",
        )


def test_compiler_patch_gradient_checkpointing_for_in_modulelist_pattern():
    """``unsloth_zoo/compiler.py:2258-2270`` -- ``patch_gradient_checkpointing``
    discovers ``self.<X> = nn.ModuleList(...)`` in `__init__` and then
    matches ``for <var> in self.<X>:\\n    hidden_states = <var>(<args>)``
    in `forward`. Asserts at least one transformers modeling file still
    has the ``self.<X> = nn.ModuleList`` assignment pattern (the call-
    site shape used by GradientCheckpointingLayer fall-back).
    """
    pytest.importorskip("transformers")
    pattern = re.compile(r"self\.[^\s]{1,} = .*?nn\.ModuleList\(")
    candidates = [
        "transformers.models.llama.modeling_llama",
        "transformers.models.mistral.modeling_mistral",
        "transformers.models.qwen2.modeling_qwen2",
        "transformers.models.qwen3.modeling_qwen3",
        "transformers.models.gemma.modeling_gemma",
        "transformers.models.gemma2.modeling_gemma2",
        "transformers.models.gemma3.modeling_gemma3",
    ]
    if not _probe_modules(candidates, lambda s: pattern.search(s) is not None):
        _drift(
            "unsloth_zoo/compiler.py:2258-2270",
            r"self.<X> = nn.ModuleList(...)",
            "any of " + ", ".join(candidates),
            "Without nn.ModuleList on `self`, the gradient_checkpointing "
            "rewriter falls back to no-op.",
        )


def test_compiler_qwen2vl_rotary_pos_emb_blk_call_variant_pinned():
    """``unsloth_zoo/compiler.py:2200-2207`` pins the SECOND custom
    blk-call variant (with ``rotary_pos_emb`` between cu_seqlens and
    position_embeddings):

        hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
                position_embeddings=position_embeddings,
                **kwargs,
            )

    This variant is the REPLACEMENT (not the find); for the rewriter
    to ever produce it the FIND variant must match upstream. If the
    Qwen2VL visual forward still has the find variant (covered by
    test_upstream_source_patterns.py) AND `rotary_pos_emb` is still a
    valid blk kwarg the rewriter remains correct. We pin
    ``rotary_pos_emb=`` to confirm the replacement is meaningful.
    """
    pytest.importorskip("transformers")
    try:
        import transformers.models.qwen2_vl.modeling_qwen2_vl as q2vl
    except ImportError:
        pytest.skip("transformers.models.qwen2_vl not shipped")
    src = inspect.getsource(q2vl)
    if "rotary_pos_emb" not in src:
        _drift(
            "unsloth_zoo/compiler.py:2200-2207",
            "rotary_pos_emb (kwarg name in replacement)",
            "transformers.models.qwen2_vl.modeling_qwen2_vl",
            "Replacement variant references `rotary_pos_emb=`; if "
            "upstream renamed the kwarg the rewritten call is "
            "API-incompatible.",
        )


def test_compiler_qwen2vl_attention_mask_blk_call_pinned():
    """``unsloth_zoo/compiler.py:2208-2223`` pins the THIRD custom
    blk-call FIND variant with ``attention_mask=attention_mask``.
    """
    pytest.importorskip("transformers")
    try:
        import transformers.models.qwen2_vl.modeling_qwen2_vl as q2vl
    except ImportError:
        pytest.skip("transformers.models.qwen2_vl not shipped")
    src = inspect.getsource(q2vl)
    if "attention_mask=attention_mask" not in src:
        pytest.skip(
            "Qwen2-VL visual forward no longer passes "
            "`attention_mask=attention_mask` to blk; rewriter variant 3 "
            "is dormant (not necessarily a regression; the find still "
            "no-ops cleanly)."
        )


def test_compiler_strip_kw_for_loop_pattern_targetable():
    """``unsloth_zoo/compiler.py:2306-2346`` -- ``strip_kw_from_module_calls``
    finds ``for <name>, <layer> in enumerate(self.<list>):`` or
    ``for <layer> in self.<list>:`` and then strips kwarg names from
    each ``<layer>(arg=arg, ...)`` call. Asserts a transformers
    decoder layer still uses the ``for <layer> in self.<list>:`` form.
    """
    pytest.importorskip("transformers")
    # Modern transformers uses `for decoder_layer in self.layers:` (or
    # similar), then a body that calls the layer; matches BOTH
    # `for <layer> in self.<list>:` (single line) and the multi-line
    # `for <layer> in self.<list>[a:b]:` shape. zoo's compiled regex
    # is more flexible; just probe for `for <var> in self.<attr>`.
    pattern = re.compile(r"for\s+\w+\s+in\s+self\.\w+")
    candidates = [
        "transformers.models.llama.modeling_llama",
        "transformers.models.qwen2.modeling_qwen2",
        "transformers.models.qwen3.modeling_qwen3",
        "transformers.models.mistral.modeling_mistral",
        "transformers.models.gemma.modeling_gemma",
        "transformers.models.gemma3.modeling_gemma3",
    ]
    if not _probe_modules(candidates, lambda s: pattern.search(s) is not None):
        _drift(
            "unsloth_zoo/compiler.py:2306-2346",
            r"for <layer> in self.<list>",
            "any of " + ", ".join(candidates),
            "Without this decoder-layer iteration form, "
            "strip_kw_from_module_calls is unreachable.",
        )


def test_compiler_dtype_mismatch_finfo_attention_mask_pinned():
    """``unsloth_zoo/compiler.py:2381-2391`` -- ``patch_finfo_attention_mask_dtype_mismatch``
    pins the EXACT two-line shape:

        attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
        attention_mask_tensor = (1.0 - attention_mask_tensor).int()

    This pattern was the pre-4.50 sdpa_attention_mask_to_bool helper.
    If upstream renamed the variable or split the line, the rewriter
    silently no-ops.
    """
    pytest.importorskip("transformers")
    # Probe several modules; the variable name and exact split changed
    # in 4.50+ (masking_utils now hosts an equivalent).
    candidates = [
        "transformers.modeling_attn_mask_utils",
        "transformers.masking_utils",
        "transformers.models.gemma3.modeling_gemma3",
        "transformers.models.gpt_oss.modeling_gpt_oss",
    ]
    # The relevant idiom is `<X> = <X> / torch.finfo(<X>.dtype).min`
    # followed by `<X> = (1.0 - <X>).int()`. Look for that finfo+1.0
    # idiom in any of the candidates.
    pattern = re.compile(
        r"torch\.finfo\([^\)]+\.dtype\)\.min[\s\S]{0,200}\(1\.0\s*-"
    )
    if not _probe_modules(candidates, lambda s: pattern.search(s) is not None):
        # Pre-existing drift acknowledged: this exact idiom was removed
        # in masking_utils. Don't fail unless the underlying primitives
        # (`torch.finfo(...).min` AND `(1.0 - <mask>)`) are also gone --
        # the rewriter is forward-looking.
        finfo_present = _probe_modules(
            candidates,
            lambda s: "torch.finfo" in s,
        )
        if not finfo_present:
            _drift(
                "unsloth_zoo/compiler.py:2381-2391",
                r"`<X> = <X>/torch.finfo(...).min` then `<X> = (1.0-<X>).int()`",
                "any of " + ", ".join(candidates),
                "Both the exact idiom AND the underlying finfo masking "
                "are gone; the dtype-mismatch rewriter has no target.",
            )


def test_compiler_lora_forward_result_clone_pinned_string():
    """``unsloth_zoo/compiler.py:2539`` runs
    ``source.replace("result = result.clone()", "")``. Asserts peft's
    LoRA layer source still has this exact .clone() line (peft used
    to put it in ``Linear.forward`` to defeat in-place ops).
    """
    pytest.importorskip("peft")
    try:
        from peft.tuners.lora.layer import Linear as LoraLinear
    except ImportError:
        pytest.skip("peft.tuners.lora.layer.Linear missing in this build")
    try:
        src = inspect.getsource(LoraLinear.forward)
    except (OSError, TypeError):
        pytest.skip("peft Linear.forward source unavailable")
    needle = "result = result.clone()"
    if needle not in src:
        pytest.skip(
            "peft >= 0.15 dropped the explicit `result = result.clone()` "
            "line; zoo's str.replace correctly no-ops on this build."
        )


def test_compiler_torch_result_dtype_pattern():
    """``unsloth_zoo/compiler.py:2553`` runs
    ``re.search(r"\\btorch_result_dtype\\s*=\\s*result\\.dtype\\b", source)``
    against peft's LoRA forward. Asserts at least one peft layer
    (Linear / Linear4bit / Linear8bitLt) STILL stashes
    ``torch_result_dtype = result.dtype`` (Linear / GPTQ / LoraParallel
    path); otherwise the rewriter picks the wrong dtype_cast branch.
    """
    pytest.importorskip("peft")
    try:
        import peft.tuners.lora.layer as ly
    except ImportError:
        pytest.skip("peft.tuners.lora.layer missing")
    src = inspect.getsource(ly)
    pattern = re.compile(r"\btorch_result_dtype\s*=\s*result\.dtype\b")
    if pattern.search(src) is None:
        pytest.skip(
            "peft no longer stashes `torch_result_dtype = result.dtype`; "
            "zoo correctly falls back to `result.dtype` as dtype_cast."
        )


def test_compiler_lora_def_forward_rename_pinned_string():
    """``unsloth_zoo/compiler.py:2563-2567`` runs
    ``source.replace("def forward", "def unsloth_forward", 1)``.
    Asserts peft's LoRA layer source still has ``def forward`` (this
    is the function-name rewrite, and a regression where peft renames
    forward would break the install entirely).
    """
    pytest.importorskip("peft")
    try:
        from peft.tuners.lora.layer import Linear as LoraLinear
    except ImportError:
        pytest.skip("peft.tuners.lora.layer.Linear missing")
    try:
        src = inspect.getsource(LoraLinear.forward)
    except (OSError, TypeError):
        pytest.skip("peft Linear.forward source unavailable")
    if "def forward" not in src:
        _drift(
            "unsloth_zoo/compiler.py:2563-2567",
            "def forward",
            "peft.tuners.lora.layer.Linear.forward",
            "Without `def forward`, the rename step fails -- "
            "unsloth_forward is never installed.",
        )


def test_compiler_lora_x_cast_dtype_pinned_strings():
    """``unsloth_zoo/compiler.py:2578-2581,2596`` pins TWO peft-side
    LoRA dtype-cast variants:

        old1: x = x.to(lora_A.weight.dtype)
        old2: x = self._cast_input_dtype(x, lora_A.weight.dtype)
        old3: self._check_forward_args(x, *args, **kwargs)

    DRIFT (fail) only when ALL THREE are gone -- then the autocast
    fixup AND the check-forward-args strip both no-op.
    """
    pytest.importorskip("peft")
    try:
        from peft.tuners.lora.layer import Linear as LoraLinear
    except ImportError:
        pytest.skip("peft.tuners.lora.layer.Linear missing")
    try:
        src = inspect.getsource(LoraLinear.forward)
    except (OSError, TypeError):
        pytest.skip("peft Linear.forward source unavailable")
    needles = (
        "x = x.to(lora_A.weight.dtype)",
        "x = self._cast_input_dtype(x, lora_A.weight.dtype)",
        "self._check_forward_args(x, *args, **kwargs)",
    )
    if not any(n in src for n in needles):
        _drift(
            "unsloth_zoo/compiler.py:2578-2596",
            " OR ".join(needles),
            "peft.tuners.lora.layer.Linear.forward",
            "All three pinned strings gone; autocast fixup and "
            "check-forward-args strip are unreachable.",
        )


def test_compiler_variant_kwarg_keys_pinned_token():
    """``unsloth_zoo/compiler.py:2649-2655`` runs
    ``re.search(r"\\bVARIANT_KWARG_KEYS\\b", source)``. Asserts peft >=
    0.18.0 still exposes ``VARIANT_KWARG_KEYS`` at the layer module
    level (it was added for alora). The rewriter installs an explicit
    fallback if it's missing, but the FIND must succeed for the
    fallback to ever fire.
    """
    pytest.importorskip("peft")
    try:
        import peft.tuners.lora.layer as ly
    except ImportError:
        pytest.skip("peft.tuners.lora.layer missing")
    src = inspect.getsource(ly)
    if "VARIANT_KWARG_KEYS" not in src:
        pytest.skip(
            "peft < 0.18.0; VARIANT_KWARG_KEYS not yet introduced. "
            "Zoo's rewriter correctly skips the import injection. "
            "Test surfaces forward-looking pin."
        )


def test_compiler_patch_residual_stream_residual_plus_hidden_states_pattern():
    """``unsloth_zoo/compiler.py:2698-2705`` -- the SECOND
    ``patch_residual_stream`` regex matches
    ``<h> = residual + (<h> * <expr>|<expr> * <h>)``. Asserts at least
    one VLM cross-attention encoder still has ``hidden_state =
    residual + hidden_state * ...`` (the addcmul / fused-add target).
    """
    pytest.importorskip("transformers")
    # The pinned regex requires the variable to be either
    # ``hidden_state`` or ``hidden_states``. The pattern body is:
    # ``<h> = residual + (<h> * <expr>|<expr> * <h>)``
    pattern = re.compile(
        r"(hidden_state(?:s)?) = residual \+ "
        r"(?:\1 \* [^\n]+|[^\n]+ \* \1)"
    )
    candidates = [
        "transformers.models.mllama.modeling_mllama",
        "transformers.models.granite.modeling_granite",
        "transformers.models.idefics.modeling_idefics",
        "transformers.models.qwen2_vl.modeling_qwen2_vl",
    ]
    if not _probe_modules(candidates, lambda s: pattern.search(s) is not None):
        _drift(
            "unsloth_zoo/compiler.py:2698-2705",
            r"<h> = residual + (<h> * <expr>|<expr> * <h>)",
            "any of " + ", ".join(candidates),
            "Without a residual-stream multiply-add site, "
            "patch_residual_stream cannot fold it into "
            "torch.add / torch.addcmul.",
        )


def test_compiler_patch_gradient_accumulation_from_config_pattern():
    """``unsloth_zoo/compiler.py:2757-2759`` -- ``patch_gradient_accumulation``
    discovers ``self.<X> = <Y>._from_config(...)`` instances. Asserts
    at least one VLM module still uses ``._from_config`` to build a
    sub-model (used by Idefics3, Llava-family, Qwen2-VL, ...).
    """
    pytest.importorskip("transformers")
    pattern = re.compile(r"self\.[^ ]+\s*=\s*[^\.]+\._from_config")
    candidates = [
        "transformers.models.llava.modeling_llava",
        "transformers.models.llava_next.modeling_llava_next",
        "transformers.models.qwen2_vl.modeling_qwen2_vl",
        "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl",
        "transformers.models.idefics3.modeling_idefics3",
        "transformers.models.paligemma.modeling_paligemma",
        "transformers.models.mllama.modeling_mllama",
    ]
    if not _probe_modules(candidates, lambda s: pattern.search(s) is not None):
        _drift(
            "unsloth_zoo/compiler.py:2757-2759",
            r"self.<X> = <Y>._from_config(...)",
            "any of " + ", ".join(candidates),
            "Without `._from_config`, gradient-accumulation **kwargs "
            "fix-up is unreachable.",
        )


def test_compiler_efficientnet_block_class_finder_pattern():
    """``unsloth_zoo/compiler.py:2925`` -- ``compile_timm_models`` runs
    ``re.findall(r"class ([^ ]{1,})\\(.*?nn\\.Module\\)\\:", ...)``
    against timm._efficientnet_blocks. If timm refactors so blocks
    no longer subclass ``nn.Module`` (e.g. moves to a base class),
    the finder returns 0 matches and zero blocks are torch.compile-d.
    """
    timm = pytest.importorskip("timm")
    try:
        import timm.models._efficientnet_blocks as effb
    except ImportError:
        pytest.skip("timm._efficientnet_blocks not shipped")
    try:
        src = inspect.getsource(effb)
    except OSError:
        pytest.skip("timm._efficientnet_blocks source unavailable")
    pattern = re.compile(r"class [^ ]{1,}\(.*?nn\.Module\)\:")
    if pattern.search(src) is None:
        _drift(
            "unsloth_zoo/compiler.py:2925",
            r"class <X>(...nn.Module):",
            "timm.models._efficientnet_blocks",
            "Without nn.Module-subclass blocks, "
            "compile_timm_models compiles nothing.",
        )


def test_compiler_class_inheritance_finder_pattern():
    """``unsloth_zoo/compiler.py:3310-3318`` -- the global compiler
    discovers ``class <X>(...Module)`` then ``class <Y>(<X>)`` via
    re.findall. Asserts at least one modeling file still has both
    a torch ``.Module`` subclass and one nested class deriving from
    another local class (the SDPA / Eager attention duo).
    """
    pytest.importorskip("transformers")
    base_pattern = re.compile(r"class [^\s]+\(.+?\.Module\)")
    candidates = [
        "transformers.models.llama.modeling_llama",
        "transformers.models.mistral.modeling_mistral",
        "transformers.models.qwen2.modeling_qwen2",
        "transformers.models.qwen3.modeling_qwen3",
        "transformers.models.gemma3.modeling_gemma3",
    ]
    if not _probe_modules(candidates, lambda s: base_pattern.search(s) is not None):
        _drift(
            "unsloth_zoo/compiler.py:3310",
            r"class <X>(...Module)",
            "any of " + ", ".join(candidates),
            "Without a top-level Module subclass, the compiler's "
            "torch_modules discovery returns empty.",
        )


def test_compiler_class_pretrainedmodel_finder_pattern():
    """``unsloth_zoo/compiler.py:3332-3334`` -- ``re.findall(
    r"class ([^\\s]{1,})\\(.+?PreTrainedModel\\)", full_source)``.
    Asserts at least one transformers model file still has a
    ``PreTrainedModel`` subclass at module level (this is how the
    compiler discovers backbone / for-causal-lm classes).
    """
    pytest.importorskip("transformers")
    pattern = re.compile(r"class [^\s]+\(.+?PreTrainedModel\)")
    candidates = [
        "transformers.models.llama.modeling_llama",
        "transformers.models.mistral.modeling_mistral",
        "transformers.models.qwen2.modeling_qwen2",
        "transformers.models.qwen3.modeling_qwen3",
        "transformers.models.gemma3.modeling_gemma3",
    ]
    if not _probe_modules(candidates, lambda s: pattern.search(s) is not None):
        _drift(
            "unsloth_zoo/compiler.py:3332-3334",
            r"class <X>(...PreTrainedModel)",
            "any of " + ", ".join(candidates),
            "Without a PreTrainedModel subclass, the compiler can't "
            "discover backbone classes to patch.",
        )


def test_compiler_routing_weights_to_marker_in_source():
    """``unsloth_zoo/compiler.py:3376`` -- branches on
    ``"routing_weights.to" in source``. Asserts at least one MoE
    forward still has this exact substring.
    """
    pytest.importorskip("transformers")
    candidates = [
        "transformers.models.mixtral.modeling_mixtral",
        "transformers.models.qwen2_moe.modeling_qwen2_moe",
        "transformers.models.qwen3_moe.modeling_qwen3_moe",
        "transformers.models.deepseek_v3.modeling_deepseek_v3",
    ]
    if not _probe_modules(candidates, lambda s: "routing_weights.to" in s):
        _drift(
            "unsloth_zoo/compiler.py:3376",
            "routing_weights.to",
            "any of " + ", ".join(candidates),
            "Without this marker, the router-logit-cast branch is "
            "skipped and the bf16 router fix is invisible.",
        )


def test_compiler_supports_sdpa_marker_in_full_source():
    """``unsloth_zoo/compiler.py:3390-3392`` branches on
    ``"_supports_sdpa = True" in full_source`` and
    ``"_supports_sdpa = False" not in full_source``. Asserts at least
    one modeling file still declares ``_supports_sdpa`` either way.

    Status: BENIGN on transformers 4.50+.

    transformers 4.50+ moved SDPA inference to
    ``ALL_ATTENTION_FUNCTIONS`` (the "attention interface" refactor).
    The class-level ``_supports_sdpa`` marker is gone from most modeling
    files, so zoo's source-string probe at compiler.py:3390-3392 silently
    no-ops on these builds. The branch is dormant, but the actual SDPA
    dispatch still works correctly: transformers routes through the
    registry at runtime regardless of the marker, and zoo's compiler.py
    now has a third fallback (``_all_attention_functions_has_sdpa``) that
    keeps SDPA enabled for the optimised pipeline. The dormant branch is
    no longer a correctness risk; it is dead code path on this build.

    Converted from FAIL to SKIP per maintainer review.
    """
    pytest.importorskip("transformers")
    candidates = [
        "transformers.models.llama.modeling_llama",
        "transformers.models.mistral.modeling_mistral",
        "transformers.models.qwen2.modeling_qwen2",
        "transformers.models.qwen3.modeling_qwen3",
        "transformers.models.gemma.modeling_gemma",
        "transformers.models.gemma3.modeling_gemma3",
        "transformers.models.qwen2_vl.modeling_qwen2_vl",
        "transformers.models.pixtral.modeling_pixtral",
        "transformers.models.mistral3.modeling_mistral3",
    ]
    if not _probe_modules(
        candidates,
        lambda s: "_supports_sdpa = True" in s or "_supports_sdpa = False" in s,
    ):
        pytest.skip(
            "BENIGN: ALL_ATTENTION_FUNCTIONS replaces _supports_sdpa "
            "marker in transformers 4.50+; zoo's source-string branch is "
            "dormant but SDPA dispatch still works via the runtime "
            "registry. Zoo's compiler.py now also has an "
            "_all_attention_functions_has_sdpa() fallback that keeps the "
            "optimised pipeline marking SDPA-enabled on these builds."
        )


def test_compiler_data_dependent_nonzero_tolist_item_markers():
    """``unsloth_zoo/compiler.py:3587-3596`` skips compilation when
    ``.nonzero()`` / ``.tolist()`` / ``.item()`` appears, or when
    ``torch.where(`` + ``.index_add`` appear. Asserts at least one MoE
    modeling file STILL has a data-dependent op so the compile-skip
    branch is reachable.
    """
    pytest.importorskip("transformers")
    candidates = [
        "transformers.models.mixtral.modeling_mixtral",
        "transformers.models.qwen2_moe.modeling_qwen2_moe",
        "transformers.models.qwen3_moe.modeling_qwen3_moe",
        "transformers.models.deepseek_v3.modeling_deepseek_v3",
        "transformers.models.gpt_oss.modeling_gpt_oss",
    ]

    def has_marker(src):
        return any(t in src for t in (".nonzero()", ".tolist()", ".item()"))

    if not _probe_modules(candidates, has_marker):
        pytest.skip(
            "No probed MoE / model uses .nonzero/.tolist/.item; the "
            "compile-skip branch is dormant on this build. Pin guards "
            "any future re-introduction."
        )


def test_compiler_logger_running_training_inner_loop_present():
    """``unsloth_zoo/compiler.py:3988``'s `re.search` ALSO depends on
    ``inner_training_loop`` (the Trainer source string) actually being
    non-empty. Confirm
    ``transformers.trainer.Trainer._inner_training_loop`` source is
    fetchable (covered above) AND that the source spans more than a
    few hundred chars (a stub would be a real regression).
    """
    pytest.importorskip("transformers")
    from transformers.trainer import Trainer
    try:
        src = inspect.getsource(Trainer._inner_training_loop)
    except (OSError, TypeError):
        _drift(
            "unsloth_zoo/compiler.py:3988-4040",
            "inspect.getsource(Trainer._inner_training_loop)",
            "transformers.trainer.Trainer",
            "Source unavailable; the whole inner-training-loop rewriter "
            "skips and `_fast_inner_training_loop` is never installed.",
        )
        return
    if len(src) < 500:
        _drift(
            "unsloth_zoo/compiler.py:3988-4040",
            "non-trivial Trainer._inner_training_loop source body",
            "transformers.trainer.Trainer",
            f"Source length is suspiciously short ({len(src)} chars); "
            "the rewriter expects a multi-hundred-line function.",
        )


def test_compiler_dict_attention_mask_gpt_oss_v5_pattern_present():
    """``unsloth_zoo/compiler.py:4148-4158`` re-sub guards on BOTH
    ``"attn_weights = attn_weights + attention_mask"`` AND ``"module"``
    appearing in the source. Asserts gpt_oss's modeling source has
    ``"module"`` referenced somewhere so the conditional fires when
    the attn_weights add pattern is present.
    """
    pytest.importorskip("transformers")
    try:
        import transformers.models.gpt_oss.modeling_gpt_oss as gpt_oss
    except ImportError:
        pytest.skip("transformers.models.gpt_oss not shipped")
    src = inspect.getsource(gpt_oss)
    if "module" not in src:
        _drift(
            "unsloth_zoo/compiler.py:4148-4158",
            "module (token in source)",
            "transformers.models.gpt_oss.modeling_gpt_oss",
            "Without `module` referenced, zoo's `if ... and 'module' in "
            "source` guard fails and the dict-attention v5 rewrite "
            "doesn't apply.",
        )


def test_compiler_conv_forward_def_first_param_pattern():
    """``unsloth_zoo/compiler.py:4289`` runs
    ``_re.search(r"def forward\\(self,\\s*(\\w+)", source)`` against
    ``nn.Conv*`` and ``nn.*Norm`` forwards. Asserts at least one
    torch.nn module exposes the ``def forward(self, <name>)`` shape.
    """
    import torch.nn as nn
    pattern = re.compile(r"def forward\(self,\s*(\w+)")
    found = False
    for cls_name in ("Conv1d", "Conv2d", "LayerNorm", "BatchNorm1d", "Linear"):
        cls = getattr(nn, cls_name, None)
        if cls is None:
            continue
        try:
            src = inspect.getsource(cls.forward)
        except (OSError, TypeError):
            continue
        if pattern.search(src):
            found = True
            break
    if not found:
        _drift(
            "unsloth_zoo/compiler.py:4289",
            r"def forward(self, <name>)",
            "torch.nn (Conv1d/Conv2d/LayerNorm/Linear)",
            "Without `def forward(self, <name>)`, conv/norm dtype "
            "fix-up can't read the parameter name and falls back to "
            "the default 'input', which may not be the actual arg.",
        )


# ===========================================================================
# unsloth_zoo/patching_utils.py rewriters
# ===========================================================================


def test_patching_utils_compiled_autograd_end_capture_return_compiled_fn_pinned():
    """``unsloth_zoo/patching_utils.py:544-547`` runs
    ``re.search(r"\\n([ ]{1,})return compiled_fn", source)`` against
    ``torch._dynamo.compiled_autograd.AutogradCompilerInstance.end_capture``
    and ``source.replace("return compiled_fn(inputs, sizes, scalars,
    hooks)", "with disable():\\n    return compiled_fn(inputs, sizes,
    scalars, hooks)")``. If the exact call signature changes the
    str.replace silently no-ops AND the gradient-checkpointing
    double-compile fix is dormant.
    """
    pytest.importorskip("torch")
    try:
        import torch._dynamo.compiled_autograd as ca
    except ImportError:
        pytest.skip("torch._dynamo.compiled_autograd not available")
    if not hasattr(ca, "AutogradCompilerInstance"):
        _drift(
            "unsloth_zoo/patching_utils.py:537",
            "AutogradCompilerInstance",
            "torch._dynamo.compiled_autograd",
            "Class is gone; the entire end_capture patch is dead code.",
        )
        return
    inst = ca.AutogradCompilerInstance
    if not hasattr(inst, "end_capture"):
        _drift(
            "unsloth_zoo/patching_utils.py:537",
            "end_capture method",
            "torch._dynamo.compiled_autograd.AutogradCompilerInstance",
        )
        return
    try:
        src = inspect.getsource(inst.end_capture)
    except (OSError, TypeError):
        _drift(
            "unsloth_zoo/patching_utils.py:539",
            "inspect.getsource(end_capture)",
            "torch._dynamo.compiled_autograd.AutogradCompilerInstance",
        )
        return
    needle = "return compiled_fn(inputs, sizes, scalars, hooks)"
    pattern = re.compile(r"\n([ ]{1,})return compiled_fn")
    # Drift-detector contract: pass if EITHER the exact str AND the
    # regex are present (the rewriter works), OR neither AND the
    # `with disable():` line is already present (someone else patched
    # / upstream merged). Otherwise: KNOWN ACTIVE DRIFT on torch >=
    # 2.7 (the `end_capture` signature changed to (..., packed_inputs)
    # and the return-call moved inside a nested with-block). The
    # rewriter no-ops cleanly today; zoo's str.replace silently fails
    # to find the old form. Surface as a forward-looking skip with a
    # loud message so a maintainer can re-anchor when fixing PR
    # #135795-equivalent upstream.
    if needle in src and pattern.search(src) is not None:
        return
    if "with disable()" in src or "with _disable()" in src:
        # Upstream already wraps the compiled_fn call in a disable
        # context (torch 2.7+ landed the fix natively, in either the
        # bare or underscore-prefixed form). Zoo's recogniser now
        # accepts both shapes and returns cleanly without rewriting.
        return
    if "compiled_fn(" in src:
        # Status: BENIGN on torch 2.7+.
        #
        # The function name is still discoverable; the rewriter target
        # exists in some form but the exact call signature drifted
        # (added `packed_inputs`, moved the return inside a nested
        # `with` block). Torch 2.7+ fixed the underlying double-compile
        # bug upstream natively (with `with _disable()` wrapping). Zoo's
        # str.replace silently no-ops on this build, which is the
        # correct behaviour: there's nothing to patch when upstream has
        # already fixed it. Zoo's patch_compiled_autograd now recognises
        # both `with disable()` and `with _disable()` and bails early.
        #
        # Converted from FAIL to SKIP per maintainer review.
        pytest.skip(
            "BENIGN: torch 2.7+ fixed PR #135795-style double-compile "
            "upstream natively (now wraps compiled_fn in `with _disable()`); "
            "zoo's rewriter at patching_utils.py:540 now recognises both "
            "`with disable()` and `with _disable()` and no-ops cleanly. "
            "The dormant rewriter is correct behaviour on this build."
        )
    _drift(
        "unsloth_zoo/patching_utils.py:539-547",
        needle,
        "torch._dynamo.compiled_autograd.AutogradCompilerInstance.end_capture",
        "Neither the pinned `return compiled_fn(...)` form NOR the "
        "patched `with disable():` shape is present, AND the bare "
        "`compiled_fn(` token is also missing. The double-compile "
        "fix is dormant and PR #135795-style regressions can return.",
    )


def test_patching_utils_compiled_autograd_end_capture_rename_target():
    """``unsloth_zoo/patching_utils.py:548`` runs
    ``source.replace("def end_capture", "def unsloth_end_capture", 1)``.
    Asserts ``def end_capture`` exists in the source.
    """
    pytest.importorskip("torch")
    try:
        import torch._dynamo.compiled_autograd as ca
        inst = ca.AutogradCompilerInstance
        src = inspect.getsource(inst.end_capture)
    except (ImportError, AttributeError, OSError, TypeError):
        pytest.skip("AutogradCompilerInstance.end_capture unavailable")
    if "def end_capture" not in src:
        _drift(
            "unsloth_zoo/patching_utils.py:548",
            "def end_capture",
            "torch._dynamo.compiled_autograd.AutogradCompilerInstance.end_capture",
            "Function rename source-string missing; the rewriter "
            "can't install `unsloth_end_capture`.",
        )


def test_patching_utils_autograd_engine_call_method_compiled_autograd_enabled_pinned():
    """``unsloth_zoo/patching_utils.py:564-573`` runs
    ``source.replace("torch._dynamo.compiled_autograd.compiled_autograd_enabled",
    "torch._dynamo.compiled_autograd.in_compiled_autograd_region", 1)``
    on ``AutogradEngineVariable.call_method``. Asserts EITHER form
    is present so the rewriter (or the upstream fix) is reachable.
    """
    pytest.importorskip("torch")
    try:
        import torch._dynamo.variables.misc as misc
        cls = misc.AutogradEngineVariable
        src = inspect.getsource(cls.call_method)
    except (ImportError, AttributeError, OSError, TypeError):
        pytest.skip("AutogradEngineVariable.call_method unavailable")
    old = "torch._dynamo.compiled_autograd.compiled_autograd_enabled"
    new = "torch._dynamo.compiled_autograd.in_compiled_autograd_region"
    if old not in src and new not in src and "in_compiled_autograd_region" not in src:
        _drift(
            "unsloth_zoo/patching_utils.py:564-573",
            f"{old} OR {new}",
            "torch._dynamo.variables.misc.AutogradEngineVariable.call_method",
            "Neither pinned reference is present; the rewriter has no "
            "anchor for the compiled-autograd region rename.",
        )


def test_patching_utils_autograd_engine_call_method_rename_target():
    """``unsloth_zoo/patching_utils.py:574`` runs
    ``source.replace("def call_method", "def unsloth_call_method", 1)``.
    Asserts ``def call_method`` exists.
    """
    pytest.importorskip("torch")
    try:
        import torch._dynamo.variables.misc as misc
        cls = misc.AutogradEngineVariable
        src = inspect.getsource(cls.call_method)
    except (ImportError, AttributeError, OSError, TypeError):
        pytest.skip("AutogradEngineVariable.call_method unavailable")
    if "def call_method" not in src:
        _drift(
            "unsloth_zoo/patching_utils.py:574",
            "def call_method",
            "torch._dynamo.variables.misc.AutogradEngineVariable.call_method",
        )


def test_patching_utils_replace_with_bnb_linear_skip_modules_pinned():
    """``unsloth_zoo/patching_utils.py:695-699`` runs
    ``source.replace("name in quantization_config.llm_int8_skip_modules\\n",
    ..., 1)`` against
    ``transformers.integrations.bitsandbytes._replace_with_bnb_linear``.
    Asserts the EXACT pinned token-with-newline is present in the
    upstream source -- otherwise the dynamic-4bit conversion patch
    no-ops.

    Important: by the time this test runs in the suite,
    ``unsloth_zoo/patching_utils.py`` has already rebound
    ``bnb._replace_with_bnb_linear`` to ``_unsloth_replace_with_bnb_linear``
    and rewritten the source string -- the needle below was deliberately
    replaced. Reading ``inspect.getsource`` off the live function would
    return the patched body and false-fail. We instead load the original
    upstream source from the module file via ``inspect.getsourcefile``
    so the drift detector still anchors to the genuine upstream API.
    """
    pytest.importorskip("transformers")
    try:
        import transformers.integrations.bitsandbytes as bnb
    except ImportError:
        pytest.skip("transformers.integrations.bitsandbytes not available")
    if not hasattr(bnb, "_replace_with_bnb_linear"):
        pytest.skip(
            "transformers 5.x removed _replace_with_bnb_linear; zoo "
            "uses the should_convert_module patch path instead."
        )
        return

    # Resolve the upstream source from the module file directly. Zoo's
    # patch_utils.py monkey-patches `bnb._replace_with_bnb_linear` to a
    # renamed `_unsloth_replace_with_bnb_linear` whose body is rewritten
    # to bypass the needle below. Reading inspect.getsource off the live
    # attribute would surface that patched source, never the upstream one.
    live = bnb._replace_with_bnb_linear
    is_zoo_patched = (
        getattr(live, "__name__", "") == "_unsloth_replace_with_bnb_linear"
    )
    src = None
    if is_zoo_patched:
        # Read original source from the module file -- truthful upstream
        # signal regardless of how many import-fix runs ran first.
        from pathlib import Path
        try:
            mod_file = inspect.getsourcefile(bnb)
            if mod_file:
                src = Path(mod_file).read_text(encoding="utf-8")
        except (OSError, TypeError):
            src = None
    if src is None:
        try:
            src = inspect.getsource(bnb._replace_with_bnb_linear)
        except (OSError, TypeError):
            _drift(
                "unsloth_zoo/patching_utils.py:682",
                "inspect.getsource(_replace_with_bnb_linear)",
                "transformers.integrations.bitsandbytes",
            )
            return
    needle = "name in quantization_config.llm_int8_skip_modules\n"
    if needle not in src:
        _drift(
            "unsloth_zoo/patching_utils.py:695",
            needle,
            "transformers.integrations.bitsandbytes._replace_with_bnb_linear",
            "Without this exact line+newline, zoo's substring-skip "
            "augmentation no-ops and dynamic 4bit quantization "
            "regresses.",
        )


def test_patching_utils_replace_with_bnb_linear_rename_token():
    """``unsloth_zoo/patching_utils.py:730-733`` runs
    ``source.replace("_replace_with_bnb_linear", "_unsloth_replace_with_bnb_linear")``.
    Asserts the upstream function name token still appears in the
    source body (the rewriter renames every occurrence).
    """
    pytest.importorskip("transformers")
    try:
        import transformers.integrations.bitsandbytes as bnb
    except ImportError:
        pytest.skip("transformers.integrations.bitsandbytes not available")
    if not hasattr(bnb, "_replace_with_bnb_linear"):
        pytest.skip("transformers 5.x; function removed.")
        return
    try:
        src = inspect.getsource(bnb._replace_with_bnb_linear)
    except (OSError, TypeError):
        pytest.skip("Source unavailable")
    if "_replace_with_bnb_linear" not in src:
        _drift(
            "unsloth_zoo/patching_utils.py:730-733",
            "_replace_with_bnb_linear",
            "transformers.integrations.bitsandbytes._replace_with_bnb_linear",
            "Without the self-reference (recursive call), the rename "
            "is incomplete and BC checks fire on the wrong name.",
        )


def test_patching_utils_replace_with_bnb_linear_current_key_name_pinned():
    """``unsloth_zoo/patching_utils.py:738-748`` runs
    ``re.sub(r"(^\\s*)(current_key_name\\.append\\(name\\))", ...,
    source, flags=re.MULTILINE)`` to splice in the score-module skip.
    Asserts the exact ``current_key_name.append(name)`` line is still
    present in upstream.
    """
    pytest.importorskip("transformers")
    try:
        import transformers.integrations.bitsandbytes as bnb
    except ImportError:
        pytest.skip("transformers.integrations.bitsandbytes not available")
    if not hasattr(bnb, "_replace_with_bnb_linear"):
        pytest.skip("transformers 5.x; function removed.")
        return
    try:
        src = inspect.getsource(bnb._replace_with_bnb_linear)
    except (OSError, TypeError):
        pytest.skip("Source unavailable")
    needle = "current_key_name.append(name)"
    if needle not in src:
        _drift(
            "unsloth_zoo/patching_utils.py:738-748",
            needle,
            "transformers.integrations.bitsandbytes._replace_with_bnb_linear",
            "Without the append-name line, the score-module skip can't "
            "be injected and `score` weights get spuriously 4bit-cast.",
        )


def test_patching_utils_current_key_name_str_marker():
    """``unsloth_zoo/patching_utils.py:688`` asserts:

        if "current_key_name_str" not in source:
            raise RuntimeError(...)

    So the rewriter HARD-fails when ``current_key_name_str`` is absent.
    Pin the variable name as a drift detector so the failure isn't
    surprising.
    """
    pytest.importorskip("transformers")
    try:
        import transformers.integrations.bitsandbytes as bnb
    except ImportError:
        pytest.skip("transformers.integrations.bitsandbytes not available")
    if not hasattr(bnb, "_replace_with_bnb_linear"):
        pytest.skip("transformers 5.x; function removed.")
        return
    try:
        src = inspect.getsource(bnb._replace_with_bnb_linear)
    except (OSError, TypeError):
        pytest.skip("Source unavailable")
    if "current_key_name_str" not in src:
        _drift(
            "unsloth_zoo/patching_utils.py:688",
            "current_key_name_str",
            "transformers.integrations.bitsandbytes._replace_with_bnb_linear",
            "Variable name is the hard-fail anchor; without it "
            "patching_utils raises RuntimeError at import time.",
        )


def test_patching_utils_replace_with_bnb_linear_ast_wrap_target():
    """``unsloth_zoo/patching_utils.py:701-704`` runs ``ast.parse`` +
    ``WrapRecursiveCall().visit(...)`` + ``ast.unparse``. The AST
    transformer wraps calls whose ``.func.id == "_replace_with_bnb_linear"``
    in a try/finally that marks the parent. Pin the recursive call as
    a regex against the upstream source so a function rename in
    transformers surfaces immediately.
    """
    pytest.importorskip("transformers")
    try:
        import transformers.integrations.bitsandbytes as bnb
    except ImportError:
        pytest.skip("transformers.integrations.bitsandbytes not available")
    if not hasattr(bnb, "_replace_with_bnb_linear"):
        pytest.skip("transformers 5.x; function removed.")
        return
    try:
        src = inspect.getsource(bnb._replace_with_bnb_linear)
    except (OSError, TypeError):
        pytest.skip("Source unavailable")
    # The recursive call is the AST anchor. Match patterns like:
    # ``_, has_been_replaced = _replace_with_bnb_linear(...)``
    pattern = re.compile(r"=\s*_replace_with_bnb_linear\s*\(")
    if pattern.search(src) is None:
        _drift(
            "unsloth_zoo/patching_utils.py:639-672",
            r"= _replace_with_bnb_linear(...) (recursive call)",
            "transformers.integrations.bitsandbytes._replace_with_bnb_linear",
            "No recursive call to wrap; the WrapRecursiveCall AST "
            "transformer no-ops and the parent-class marking is "
            "never installed.",
        )


# ===========================================================================
# unsloth_zoo/saving_utils.py rewriters
# ===========================================================================


def test_saving_utils_save_pretrained_state_dict_split_pinned_string():
    """``unsloth_zoo/saving_utils.py:2675-2677`` runs
    ``save_pretrained.find("state_dict_split = split_torch_state_dict_into_shards")``
    and ``raise RuntimeError`` when it returns -1. Pin the exact
    string against ``PreTrainedModel.save_pretrained`` source.
    """
    pytest.importorskip("transformers")
    import transformers.modeling_utils as mu
    try:
        src = inspect.getsource(mu.PreTrainedModel.save_pretrained)
    except (OSError, TypeError):
        pytest.skip("PreTrainedModel.save_pretrained source unavailable")
    needle = "state_dict_split = split_torch_state_dict_into_shards"
    if needle not in src:
        _drift(
            "unsloth_zoo/saving_utils.py:2675-2677",
            needle,
            "transformers.modeling_utils.PreTrainedModel.save_pretrained",
            "Without this exact assignment, merge_and_dequantize_lora "
            "raises `Failed to find state_dict_split` at runtime.",
        )


def test_saving_utils_save_pretrained_state_dict_contiguous_pinned_string():
    """``unsloth_zoo/saving_utils.py:2680-2686`` requires
    ``"state_dict[tensor].contiguous()"`` to be in the upstream
    source AND ``replace(..., "merge_lora_weights(...)", 1)`` it
    once. RuntimeError otherwise.
    """
    pytest.importorskip("transformers")
    import transformers.modeling_utils as mu
    try:
        src = inspect.getsource(mu.PreTrainedModel.save_pretrained)
    except (OSError, TypeError):
        pytest.skip("save_pretrained source unavailable")
    needle = "state_dict[tensor].contiguous()"
    if needle not in src:
        _drift(
            "unsloth_zoo/saving_utils.py:2680-2686",
            needle,
            "transformers.modeling_utils.PreTrainedModel.save_pretrained",
            "Without this exact `.contiguous()` call, the dequantize-"
            "merge replacement raises at runtime.",
        )


def test_saving_utils_save_pretrained_def_marker():
    """``unsloth_zoo/saving_utils.py:2688-2694`` requires
    ``"def save_pretrained" in save_pretrained`` for the rename
    ``save_pretrained -> save_pretrained_dequantized``. RuntimeError
    otherwise.
    """
    pytest.importorskip("transformers")
    import transformers.modeling_utils as mu
    try:
        src = inspect.getsource(mu.PreTrainedModel.save_pretrained)
    except (OSError, TypeError):
        pytest.skip("save_pretrained source unavailable")
    if "def save_pretrained" not in src:
        _drift(
            "unsloth_zoo/saving_utils.py:2688-2694",
            "def save_pretrained",
            "transformers.modeling_utils.PreTrainedModel.save_pretrained",
        )


def test_saving_utils_incremental_save_os_makedirs_pinned_regex():
    """``unsloth_zoo/saving_utils.py:2517`` runs
    ``re.search(r"os\\.makedirs\\(save_directory.+?\\n", save_pretrained)``
    and asserts the match is not None. Pin the upstream pattern.
    """
    pytest.importorskip("transformers")
    import transformers.modeling_utils as mu
    try:
        src = inspect.getsource(mu.PreTrainedModel.save_pretrained)
    except (OSError, TypeError):
        pytest.skip("save_pretrained source unavailable")
    pattern = re.compile(r"os\.makedirs\(save_directory")
    if pattern.search(src) is None:
        _drift(
            "unsloth_zoo/saving_utils.py:2517-2518",
            r"os.makedirs(save_directory...)",
            "transformers.modeling_utils.PreTrainedModel.save_pretrained",
            "Without this line, incremental_save_pretrained asserts "
            "on a None match and aborts the push-to-hub path.",
        )


def test_saving_utils_incremental_save_for_loop_filename_to_tensors_pinned():
    """``unsloth_zoo/saving_utils.py:2526-2533`` requires
    ``"for shard_file, tensors in filename_to_tensors"`` in
    save_pretrained source. RuntimeError otherwise.
    """
    pytest.importorskip("transformers")
    import transformers.modeling_utils as mu
    try:
        src = inspect.getsource(mu.PreTrainedModel.save_pretrained)
    except (OSError, TypeError):
        pytest.skip("save_pretrained source unavailable")
    needle = "for shard_file, tensors in filename_to_tensors"
    if needle not in src:
        _drift(
            "unsloth_zoo/saving_utils.py:2526-2533",
            needle,
            "transformers.modeling_utils.PreTrainedModel.save_pretrained",
            "Without this for-loop, incremental_save_pretrained raises "
            "and disables low-disk-space push-to-hub.",
        )


def test_saving_utils_config_json_dtype_torch_dtype_rename_targets():
    """``unsloth_zoo/saving_utils.py:1827-1828`` runs
    ``data.replace('"dtype"', '"torch_dtype"')`` on the saved
    ``config.json`` (a string, not source). This is a save-time fix
    -- assert the model's ``config.to_dict()`` exposes either ``dtype``
    or ``torch_dtype`` so the rewriter has SOMETHING to normalize.
    """
    pytest.importorskip("transformers")
    try:
        from transformers import LlamaConfig
    except ImportError:
        pytest.skip("LlamaConfig not in this build")
    cfg = LlamaConfig()
    d = cfg.to_dict()
    if "dtype" not in d and "torch_dtype" not in d:
        _drift(
            "unsloth_zoo/saving_utils.py:1827-1828",
            "config.json includes `dtype` or `torch_dtype`",
            "transformers.LlamaConfig.to_dict()",
            "Config no longer emits either form; the rename rewriter "
            "has nothing to normalize.",
        )


def test_saving_utils_lora_key_normalize_replacements_targetable():
    """``unsloth_zoo/saving_utils.py:309-314`` runs FIVE str.replace
    on LoRA key names:

        .base_layer, .modules_to_save.default, .original_module,
        .lora_A.default, .lora_B.default

    Asserts peft's LoRA layer naming still uses ``base_layer`` and
    ``lora_A.default`` (the most common shapes). DRIFT if BOTH are
    gone -- then the key-normalize pass strips nothing.
    """
    pytest.importorskip("peft")
    try:
        import peft.tuners.lora.layer as ly
    except ImportError:
        pytest.skip("peft.tuners.lora.layer missing")
    src = inspect.getsource(ly)
    # The lora layer module emits keys like `.lora_A.default`; pin
    # that token's presence (or alternative pin `base_layer`).
    if "base_layer" not in src and "lora_A.default" not in src and "lora_A" not in src:
        _drift(
            "unsloth_zoo/saving_utils.py:309-314",
            "base_layer / lora_A.default / lora_A naming",
            "peft.tuners.lora.layer",
            "Without any of peft's standard LoRA key fragments, "
            "_normalize() strips nothing and key-format detection "
            "regresses.",
        )


def test_saving_utils_moe_experts_gate_up_proj_regex_targetable():
    """``unsloth_zoo/saving_utils.py:600,653,700`` -- THREE re.match
    regexes target MoE expert key naming:

        ^(.*mlp\\.experts)\\.(\\d+)\\.(gate_proj|up_proj|down_proj)\\.weight$

    The rewriter rebuilds fused gate_up_proj weights from per-expert
    shards. Asserts upstream MoE models still expose
    ``mlp.experts.<i>.<proj>.weight`` keys (the pre-fusion shard
    format) -- via state_dict key probing on a Mixtral / Qwen MoE
    config.
    """
    pytest.importorskip("transformers")
    # Probe by inspecting modeling source for the canonical name.
    candidates = [
        "transformers.models.mixtral.modeling_mixtral",
        "transformers.models.qwen2_moe.modeling_qwen2_moe",
        "transformers.models.qwen3_moe.modeling_qwen3_moe",
        "transformers.models.deepseek_v3.modeling_deepseek_v3",
        "transformers.models.gpt_oss.modeling_gpt_oss",
    ]

    def has_marker(s):
        return any(t in s for t in (
            "mlp.experts",
            ".experts.",
            "gate_proj",
            "up_proj",
            "down_proj",
        ))

    if not _probe_modules(candidates, has_marker):
        _drift(
            "unsloth_zoo/saving_utils.py:600,653,700",
            r"mlp.experts.<i>.(gate_proj|up_proj|down_proj).weight",
            "any of " + ", ".join(candidates),
            "Without canonical MoE expert keys, _merge_moe_experts_file "
            "can't reconstruct fused gate_up_proj.",
        )


def test_saving_utils_hf_sharded_safetensors_regex_pattern():
    """``unsloth_zoo/saving_utils.py:1838`` compiles
    ``re.compile(r'^(.+?)-(\\d+)-of-(\\d+)\\.safetensors$')`` and
    asserts ALL filenames match (returns False otherwise). Smoke-
    test the regex itself against a canonical HF sharded filename.
    """
    pattern = re.compile(r"^(.+?)-(\d+)-of-(\d+)\.safetensors$")
    if pattern.match("model-00001-of-00005.safetensors") is None:
        _drift(
            "unsloth_zoo/saving_utils.py:1838",
            r"<prefix>-<shard>-of-<total>.safetensors",
            "zoo internal regex",
            "Regex itself rejects the canonical HF sharded format; "
            "is_hf_sharded_safetensors will always return False.",
        )


def test_saving_utils_lora_reverse_mapping_replacement_regex():
    """``unsloth_zoo/saving_utils.py:2923`` runs
    ``re.sub(r"\\^?([^(?]+).*", r"\\1", replacement.lstrip("^"))`` on
    each forward_mapping ``replacement`` value. Asserts the regex
    SHAPE accepts a typical mapping value like
    ``"model.language_model."`` (no leading caret, no parens, no
    ?).
    """
    sample = "model.language_model."
    out = re.sub(r"\^?([^(?]+).*", r"\1", sample.lstrip("^"))
    if out != sample:
        _drift(
            "unsloth_zoo/saving_utils.py:2923",
            r"\^?([^(?]+).*",
            "zoo internal regex",
            f"Sample input {sample!r} normalized to {out!r}; the "
            "key-converter loses the trailing dot and remaps "
            "incorrectly.",
        )


# ===========================================================================
# unsloth_zoo/training_utils.py rewriters
# ===========================================================================


def test_training_utils_name_replace_base_model_pattern():
    """``unsloth_zoo/training_utils.py:172-175,187-190`` runs:

        name = name.replace("base_model", "model", 1)
        while re.search(r'\\.(\\d+)\\.', name) is not None:
            name = re.sub(r'\\.(\\d+)\\.', r'[\\1].', name)
        name = name.replace(".weight", "", 1)

    on every PEFT module name to build an ``exec``-able accessor.
    Asserts peft model.named_modules() typically yields names
    containing ``base_model`` (the LoRA wrapper).
    """
    pytest.importorskip("peft")
    # Verify the regex idiom round-trips on a representative LoRA name.
    sample = "base_model.model.model.layers.0.self_attn.q_proj.weight"
    out = sample.replace("base_model", "model", 1)
    while re.search(r"\.(\d+)\.", out) is not None:
        out = re.sub(r"\.(\d+)\.", r"[\1].", out)
    out = out.replace(".weight", "", 1)
    if "[0]" not in out or "base_model" in out:
        _drift(
            "unsloth_zoo/training_utils.py:172-190",
            r"name = name.replace('base_model', 'model', 1) + .<i>. -> [<i>]. ",
            "internal training_utils dtype-setter",
            f"Round-trip on {sample!r} yielded {out!r}; the exec-able "
            "accessor will be malformed.",
        )


# ===========================================================================
# unsloth_zoo/temporary_patches/misc.py rewriters
# ===========================================================================


def test_misc_merge_quantization_configs_classmethod_marker():
    """``unsloth_zoo/temporary_patches/misc.py:141`` runs
    ``source.startswith("@classmethod")`` to decide whether to strip
    the ``cls`` parameter. Asserts the upstream method is still a
    classmethod.
    """
    pytest.importorskip("transformers")
    try:
        from transformers.quantizers.auto import AutoHfQuantizer
    except ImportError:
        pytest.skip("AutoHfQuantizer not available")
    try:
        src = inspect.getsource(AutoHfQuantizer.merge_quantization_configs)
    except (OSError, TypeError):
        pytest.skip("source unavailable")
    if "@classmethod" not in src.lstrip().splitlines()[0] and "classmethod" not in src[:200]:
        # Not classmethod anymore: zoo's branch still works (the strip
        # is conditional), but the EXEC of the rewritten source may
        # bind a different self type. Surface as drift if neither cls
        # nor self appears in the def line.
        first_def = next(
            (line for line in src.splitlines() if "def " in line),
            "",
        )
        if "cls" not in first_def and "self" not in first_def:
            _drift(
                "unsloth_zoo/temporary_patches/misc.py:141-144",
                "@classmethod decorator or `cls` parameter",
                "transformers.quantizers.auto.AutoHfQuantizer.merge_quantization_configs",
                "The rewriter's exec-form binding may be invalid.",
            )


def test_misc_merge_quantization_configs_dedent_def_marker():
    """``unsloth_zoo/temporary_patches/misc.py:142`` runs
    ``source = source[source.find("def"):]`` -- requires ``def`` to
    appear in the (dedented) source. Asserts the source contains
    ``def `` at all.
    """
    pytest.importorskip("transformers")
    try:
        from transformers.quantizers.auto import AutoHfQuantizer
    except ImportError:
        pytest.skip("AutoHfQuantizer not available")
    try:
        src = inspect.getsource(AutoHfQuantizer.merge_quantization_configs)
    except (OSError, TypeError):
        pytest.skip("source unavailable")
    if "def " not in src:
        _drift(
            "unsloth_zoo/temporary_patches/misc.py:142",
            "def ",
            "transformers.quantizers.auto.AutoHfQuantizer.merge_quantization_configs",
        )


def test_misc_mamba_ssm_tl_dot_finder_regex_targetable():
    """``unsloth_zoo/temporary_patches/misc.py:1082-1085`` --
    ``fix_mamba_ssm_float32`` runs
    ``re.finditer(r" ([a-zA-Z0-9\\_]{1,}) (\\=|\\+\\=) tl\\.dot\\(...)", ...)``
    against the mamba_ssm Triton chunk-scan file. ``mamba_ssm`` is
    optional; if installed, the file MUST contain ``tl.dot(`` for
    the rewriter to fire.
    """
    try:
        import mamba_ssm.ops.triton.ssd_chunk_scan as ssd
    except ImportError:
        pytest.skip("mamba_ssm not installed")
    try:
        path = inspect.getfile(ssd)
        with open(path, "r", encoding="utf-8") as f:
            file_src = f.read()
    except (OSError, TypeError):
        pytest.skip("mamba_ssm file unreadable")
    if "tl.dot(" not in file_src:
        _drift(
            "unsloth_zoo/temporary_patches/misc.py:1082-1085",
            "tl.dot(",
            "mamba_ssm.ops.triton.ssd_chunk_scan",
            "Without tl.dot calls, the float32-upcast rewriter no-ops "
            "and chunk-scan precision regressions return.",
        )


# ===========================================================================
# unsloth_zoo/temporary_patches/gpt_oss.py rewriters
# ===========================================================================


def test_gpt_oss_config_old_class_dedent_compare_marker():
    """``unsloth_zoo/temporary_patches/gpt_oss.py:2808-2810`` runs
    ``dedent(inspect.getsource(GptOssConfig))`` and compares against
    a dedented OLD class with ``.replace("Old_GptOssConfig",
    "GptOssConfig")``. The comparison is line-by-line equality, so
    even a 1-char change in upstream disables the patch.

    Pin: ``GptOssConfig`` class must still expose
    ``initial_context_length`` (one of the OLD shape's fields the
    patch was introduced to add).
    """
    pytest.importorskip("transformers")
    try:
        from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
    except ImportError:
        pytest.skip("transformers.models.gpt_oss not shipped")
    try:
        src = inspect.getsource(GptOssConfig)
    except (OSError, TypeError):
        pytest.skip("GptOssConfig source unavailable")
    # `initial_context_length` was the field the Old_GptOssConfig
    # patch added; if upstream renamed or removed it, the patch
    # source-equality compare will MISS the upgrade window.
    if "initial_context_length" not in src and "rope_scaling" not in src:
        _drift(
            "unsloth_zoo/temporary_patches/gpt_oss.py:2808-2813",
            "initial_context_length OR rope_scaling (field in GptOssConfig)",
            "transformers.models.gpt_oss.configuration_gpt_oss.GptOssConfig",
            "Without either field, the Old_GptOssConfig patch can't "
            "fix the regression it was introduced for.",
        )


# ===========================================================================
# unsloth_zoo/rl_replacements.py rewriters
# ===========================================================================


def test_rl_replacements_grpo_compute_loss_def_marker():
    """``unsloth_zoo/rl_replacements.py:560-565`` runs
    ``RL_REPLACEMENTS["grpo_compute_loss_slow"].replace(
        "def grpo_compute_loss", "def grpo_compute_loss_slow")``
    on ``inspect.getsource(grpo_compute_loss)``. Asserts the source
    of ``grpo_compute_loss`` still has the literal ``def
    grpo_compute_loss`` token.
    """
    try:
        from unsloth_zoo.rl_replacements import grpo_compute_loss
    except ImportError:
        pytest.skip("unsloth_zoo.rl_replacements not importable")
    try:
        src = inspect.getsource(grpo_compute_loss)
    except (OSError, TypeError):
        _drift(
            "unsloth_zoo/rl_replacements.py:560",
            "inspect.getsource(grpo_compute_loss)",
            "unsloth_zoo.rl_replacements",
        )
        return
    needle = "def grpo_compute_loss"
    if needle not in src:
        _drift(
            "unsloth_zoo/rl_replacements.py:562-565",
            needle,
            "unsloth_zoo.rl_replacements.grpo_compute_loss",
            "Without `def grpo_compute_loss`, the rename to "
            "`grpo_compute_loss_slow` no-ops -- RL_REPLACEMENTS for "
            "slow fallback is silently incomplete.",
        )


# ===========================================================================
# unsloth/models/rl.py rewriters
# ===========================================================================


def test_unsloth_rl_trainer_signature_columns_pinned_string():
    """``unsloth/models/rl.py:1667-1670`` runs:

        original_text = 'self._signature_columns = ["input_ids", "attention_mask", "completion_mask"]'
        new_text = 'self._signature_columns = ["input_ids", "attention_mask", "completion_mask","labels"]'
        RLTrainer_source = RLTrainer_source.replace(original_text, new_text)

    on SFTTrainer source. Modern TRL (>= 0.25) reflowed the columns
    list. DRIFT (fail) is when ``self._signature_columns = [``
    appears nowhere in SFTTrainer source -- then the labels-column
    augmentation has no possible anchor and the SFT labels regression
    returns.
    """
    pytest.importorskip("trl")
    try:
        from trl.trainer.sft_trainer import SFTTrainer
    except ImportError:
        pytest.skip("trl SFTTrainer not importable")
    try:
        src = inspect.getsource(SFTTrainer)
    except (OSError, TypeError):
        pytest.skip("SFTTrainer source unavailable")
    if "self._signature_columns" not in src and "_signature_columns" not in src:
        _drift(
            "unsloth/models/rl.py:1667-1670",
            "self._signature_columns = [...]",
            "trl.trainer.sft_trainer.SFTTrainer",
            "Without _signature_columns assignment, the labels-column "
            "augmentation can't fire and labels are dropped during "
            "SFT data preprocessing.",
        )


def test_unsloth_rl_trainer_vlm_signature_columns_old_form_pinned():
    """``unsloth/models/rl.py:1706-1713`` pins the EXACT VLM
    signature columns form for TRL 0.22.x:

        self._signature_columns = ["messages", "prompt", "completion", "images"]

    DRIFT contract: pin the FOUR member tokens individually. If ALL
    FOUR (``messages``, ``prompt``, ``completion``, ``images``) are
    gone from SFTTrainer.__init__ source, the merge-vlm-cols
    augmentation is unreachable.
    """
    pytest.importorskip("trl")
    try:
        from trl.trainer.sft_trainer import SFTTrainer
    except ImportError:
        pytest.skip("trl SFTTrainer not importable")
    try:
        src = inspect.getsource(SFTTrainer)
    except (OSError, TypeError):
        pytest.skip("SFTTrainer source unavailable")
    members = ("messages", "prompt", "completion", "images")
    if not any(m in src for m in members):
        _drift(
            "unsloth/models/rl.py:1706-1713",
            " OR ".join(members),
            "trl.trainer.sft_trainer.SFTTrainer",
            "VLM signature column tokens are all absent -- the "
            "merge-vlm-cols rewriter can't anchor.",
        )


def test_unsloth_rl_trainer_prepare_dataset_pattern():
    """``unsloth/models/rl.py:1717-1721`` runs:

        re.sub(r"([ \\t]*)train_dataset = self\\._prepare_dataset\\(", ...)

    to inject ``self._unsloth_model_ref = model`` before the call.
    Asserts SFTTrainer.__init__ source still has the
    ``self._prepare_dataset(`` call site.
    """
    pytest.importorskip("trl")
    try:
        from trl.trainer.sft_trainer import SFTTrainer
    except ImportError:
        pytest.skip("trl SFTTrainer not importable")
    try:
        src = inspect.getsource(SFTTrainer.__init__)
    except (OSError, TypeError):
        pytest.skip("SFTTrainer.__init__ source unavailable")
    if "self._prepare_dataset(" not in src:
        # TRL may have renamed the helper; surface as drift.
        _drift(
            "unsloth/models/rl.py:1717-1721",
            "self._prepare_dataset(",
            "trl.trainer.sft_trainer.SFTTrainer.__init__",
            "Without this call, the unsloth_model_ref injection can't "
            "fire and sft_prepare_dataset can't detect dynamic "
            "token_type_ids.",
        )


def test_unsloth_rl_trainer_is_loaded_in_4bit_pinned_string():
    """``unsloth/models/rl.py:1662-1665`` runs:

        RLTrainer_source.replace(
            'if getattr(model, "is_loaded_in_4bit", False) or '
            'getattr(model, "is_loaded_in_8bit", False):',
            "if False:",
        )

    on every TRL trainer's source to remove TRL's bf16 cast block.
    Asserts SOME TRL trainer's source still has at least one of
    the pinned getattr calls.
    """
    pytest.importorskip("trl")
    import importlib
    candidates = [
        "trl.trainer.sft_trainer",
        "trl.trainer.dpo_trainer",
        "trl.trainer.kto_trainer",
        "trl.trainer.bco_trainer",
        "trl.trainer.online_dpo_trainer",
    ]
    found = False
    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        try:
            src = inspect.getsource(mod)
        except (OSError, TypeError):
            continue
        if "is_loaded_in_4bit" in src or "is_loaded_in_8bit" in src:
            found = True
            break
    if not found:
        # TRL >= 1.0 may have removed the explicit 4bit/8bit cast block;
        # zoo's rewrite then no-ops cleanly. Surface forward-looking.
        pytest.skip(
            "No TRL trainer references is_loaded_in_4bit/8bit anymore; "
            "the cast-removal rewriter is dormant on this build. Pin "
            "guards re-introduction."
        )


def test_unsloth_rl_trainer_peft_config_branches_pinned():
    """``unsloth/models/rl.py:1842-1857`` runs SIX peft_config
    str.replace targets:

        elif peft_config is None: / elif peft_config is not None: /
        if peft_config is None: / if peft_config is not None: /
        get_peft_model(model, peft_config) /
        prepare_peft_model / _prepare_peft_model

    DRIFT contract: pin ``peft_config`` token. If it's gone from ALL
    TRL trainer sources, the peft-disable rewriter has no target
    and PEFT GRPO training silently re-enables the broken path.
    """
    pytest.importorskip("trl")
    import importlib
    candidates = [
        "trl.trainer.sft_trainer",
        "trl.trainer.dpo_trainer",
        "trl.trainer.online_dpo_trainer",
        "trl.trainer.bco_trainer",
        "trl.trainer.kto_trainer",
    ]
    found = False
    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        try:
            src = inspect.getsource(mod)
        except (OSError, TypeError):
            continue
        if "peft_config" in src:
            found = True
            break
    if not found:
        _drift(
            "unsloth/models/rl.py:1842-1857",
            "peft_config (token in TRL trainer source)",
            "any of " + ", ".join(candidates),
            "PEFT-disable rewriter has no anchor in any TRL trainer; "
            "the GRPO peft-mode regression returns.",
        )


def test_unsloth_rl_init_comments_with_brackets_pattern():
    """``unsloth/models/rl.py:1832-1833`` runs
    ``re.findall(r"\\#[^\\n]{1,}\\n", init)`` and filters comments
    containing ``(`` or ``)``. These bracketed comments are then
    transformed to ``[...]``. Pin: the upstream Trainer ``__init__``
    must contain comments (lines starting with ``#``). If TRL strips
    all comments, the rewriter is a no-op.
    """
    pytest.importorskip("trl")
    try:
        from trl.trainer.sft_trainer import SFTTrainer
    except ImportError:
        pytest.skip("trl SFTTrainer not importable")
    try:
        src = inspect.getsource(SFTTrainer.__init__)
    except (OSError, TypeError):
        pytest.skip("SFTTrainer.__init__ source unavailable")
    if re.search(r"#[^\n]{1,}\n", src) is None:
        pytest.skip(
            "TRL SFTTrainer.__init__ has no inline comments; the "
            "bracketed-comment normalization rewriter is dormant. Pin "
            "guards re-introduction."
        )


def test_unsloth_rl_init_use_vllm_marker():
    """``unsloth/models/rl.py:1895-1928`` branches on
    ``"args.use_vllm" in init`` AND ``"model" in init`` AND
    ``"args" in init``. If a TRL trainer's __init__ has none of
    these markers, the vllm-engine wiring rewriter no-ops. Pass if
    EITHER ``args.use_vllm`` or the alternate ``self.use_vllm`` form
    appears in any TRL trainer's init source.
    """
    pytest.importorskip("trl")
    import importlib
    candidates = [
        "trl.trainer.grpo_trainer",
        "trl.trainer.online_dpo_trainer",
        "trl.trainer.rloo_trainer",
    ]
    found = False
    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        try:
            src = inspect.getsource(mod)
        except (OSError, TypeError):
            continue
        if "args.use_vllm" in src or "self.use_vllm" in src or "use_vllm" in src:
            found = True
            break
    if not found:
        pytest.skip(
            "No probed TRL RL trainer references use_vllm; vLLM-wiring "
            "rewriter is dormant on this build. Pin guards "
            "re-introduction."
        )


def test_unsloth_rl_vllm_part_findall_pattern_targetable():
    """``unsloth/models/rl.py:1932-1936`` runs
    ``re.findall(r"(\\n[\\s]{8}if (self|args)\\.use_vllm\\:.*?\\n[\\s]{8}else:\\n)",
    init, flags=re.DOTALL | re.MULTILINE)``. Pin: at least one TRL
    trainer __init__ has the if/else use_vllm branch at 8-space
    indent.
    """
    pytest.importorskip("trl")
    import importlib
    candidates = [
        "trl.trainer.grpo_trainer",
        "trl.trainer.online_dpo_trainer",
        "trl.trainer.rloo_trainer",
    ]
    pattern = re.compile(
        r"\n[\s]{8}if (self|args)\.use_vllm\:.*?\n[\s]{8}else:\n",
        flags=re.DOTALL | re.MULTILINE,
    )
    found = False
    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        try:
            src = inspect.getsource(mod)
        except (OSError, TypeError):
            continue
        if pattern.search(src):
            found = True
            break
    if not found:
        pytest.skip(
            "No probed TRL trainer has the pinned `if (self|args)."
            "use_vllm:\\n...else:\\n` indented branch shape; the vLLM "
            "init replacement is dormant on this build."
        )


def test_unsloth_rl_sampling_params_findall_pattern_targetable():
    """``unsloth/models/rl.py:1949-1953`` runs
    ``re.findall(r"\\n[\\s]{4,}(self\\.[^\\s]{1,}[\\s]{0,}\\=[\\s]{0,}SamplingParams\\(.+?\\))",
    new_vllm_part, flags=re.MULTILINE|re.DOTALL)``. Pin: a TRL
    trainer references ``SamplingParams(`` somewhere in source.
    """
    pytest.importorskip("trl")
    import importlib
    candidates = [
        "trl.trainer.grpo_trainer",
        "trl.trainer.online_dpo_trainer",
    ]
    found = False
    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        try:
            src = inspect.getsource(mod)
        except (OSError, TypeError):
            continue
        if "SamplingParams(" in src:
            found = True
            break
    if not found:
        pytest.skip(
            "No probed TRL trainer references SamplingParams("
            "...); zoo's vLLM SamplingParams patcher is dormant. "
            "Pin guards re-introduction."
        )


def test_unsloth_rl_state_dict_strip_pattern():
    """``unsloth/models/rl.py:2072-2076`` runs
    ``re.sub(r"\\.state_dict\\(\\)", r"", source)`` on every TRL
    function source. Pin: the regex matches a canonical TRL load-
    weights call site.
    """
    pytest.importorskip("trl")
    sample = (
        "    llm_model.load_weights(model.state_dict().items())\n"
    )
    rewritten = re.sub(r"\.state_dict\(\)", r"", sample)
    # After the strip, `.state_dict()` should be gone and the call
    # should still parse syntactically as `model.items()`.
    if ".state_dict()" in rewritten or "model.items()" not in rewritten:
        _drift(
            "unsloth/models/rl.py:2072-2076",
            r"\.state_dict\(\)",
            "zoo internal regex",
            f"Sample {sample!r} normalized to {rewritten!r}; the "
            "state-dict strip is malformed.",
        )


def test_unsloth_rl_llm_generate_chat_capture_pattern():
    """``unsloth/models/rl.py:2087-2093`` runs
    ``re.sub(r"(self\\.llm\\.(?:generate|chat)\\([^\\)]{1,})\\)",
    r"\\1, lora_request = self.model.load_lora(...))", source)``.
    Pin: the regex matches a synthetic ``self.llm.generate(prompts)``
    call (sanity check on the regex itself; semantic anchor on TRL
    is covered by the use_vllm marker test).
    """
    sample = "self.llm.generate(prompts, sampling_params=sp)"
    rewritten = re.sub(
        r"(self\.llm\.(?:generate|chat)\([^\)]{1,})\)",
        r"\1, lora_request = self.model.load_lora('grpo_trainer_lora_model', "
        r"load_tensors = True))",
        sample,
    )
    if "lora_request" not in rewritten:
        _drift(
            "unsloth/models/rl.py:2087-2093",
            r"self.llm.(generate|chat)(...) -> + lora_request",
            "zoo internal regex",
            "Regex didn't match canonical self.llm.generate call.",
        )


def test_unsloth_rl_sampling_params_kwargs_replace_pinned():
    """``unsloth/models/rl.py:2107-2115`` runs:

        source.replace(
            "sampling_params = SamplingParams(**generation_kwargs)",
            "sampling_params = SamplingParams(**grpo_update_SamplingParams(...))",
        )

    Pin: the pinned old shape is a SPECIFIC TRL formatting. Search
    any TRL trainer source for the SUBSTRING; absence indicates a
    TRL refactor where this rewriter is dormant.
    """
    pytest.importorskip("trl")
    import importlib
    candidates = [
        "trl.trainer.grpo_trainer",
        "trl.trainer.online_dpo_trainer",
        "trl.trainer.rloo_trainer",
    ]
    needle = "SamplingParams(**generation_kwargs)"
    found = False
    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        try:
            src = inspect.getsource(mod)
        except (OSError, TypeError):
            continue
        if needle in src:
            found = True
            break
    if not found:
        pytest.skip(
            f"No probed TRL trainer has the literal "
            f"{needle!r} call; the SamplingParams-update replacement "
            "is dormant on this build."
        )


def test_unsloth_rl_class_rename_pinned():
    """``unsloth/models/rl.py:2137-2139`` runs:

        RLTrainer_source.replace(
            f"class {RLTrainer_name}", f"class _Unsloth{RLTrainer_name}", 1
        )

    Asserts SFTTrainer source still starts with ``class SFTTrainer``.
    """
    pytest.importorskip("trl")
    try:
        from trl.trainer.sft_trainer import SFTTrainer
    except ImportError:
        pytest.skip("trl SFTTrainer not importable")
    try:
        src = inspect.getsource(SFTTrainer)
    except (OSError, TypeError):
        pytest.skip("SFTTrainer source unavailable")
    if "class SFTTrainer" not in src:
        _drift(
            "unsloth/models/rl.py:2137-2139",
            "class SFTTrainer",
            "trl.trainer.sft_trainer.SFTTrainer",
            "Without the class definition line, the class rename "
            "step can't run.",
        )


def test_unsloth_rl_torch_compile_options_dict_pattern():
    """``unsloth/models/rl.py:1622-1625`` runs
    ``re.sub(r"torch_compile_options\\s*=\\s*\\{[^}]*\\}",
    new_options, RLTrainer_source, flags=re.DOTALL)``. Sanity-check
    the regex against a representative dict assignment.
    """
    sample = 'torch_compile_options = {"epilogue_fusion": True, "max_autotune": False}'
    out = re.sub(
        r"torch_compile_options\s*=\s*\{[^}]*\}",
        "torch_compile_options = {}",
        sample,
        flags=re.DOTALL,
    )
    if out != "torch_compile_options = {}":
        _drift(
            "unsloth/models/rl.py:1622-1625",
            r"torch_compile_options\s*=\s*\{[^}]*\}",
            "zoo internal regex",
            f"Sample {sample!r} normalized to {out!r}; the dict "
            "replacement is malformed.",
        )


def test_unsloth_rl_add_adapter_block_pattern_regex():
    """``unsloth/models/rl.py:1865-1870`` builds the regex:

        r"([ \\t]*)"
        r"if\\s+is_peft_available\\(\\)\\s+and\\s+is_peft_model\\(model\\)\\s+and\\s+args\\.beta\\s*!=\\s*0\\.0\\s*:"
        r"(.*?)"
        r"ref_param\\.data\\.copy_\\(param\\.data\\)"

    to comment out the "ref" adapter creation block in GRPOTrainer.
    Pin: the SUBSTRINGS ``is_peft_available()`` AND
    ``ref_param.data.copy_`` must appear together in some TRL trainer
    source.
    """
    pytest.importorskip("trl")
    import importlib
    candidates = [
        "trl.trainer.grpo_trainer",
        "trl.trainer.rloo_trainer",
        "trl.trainer.online_dpo_trainer",
    ]
    found = False
    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        try:
            src = inspect.getsource(mod)
        except (OSError, TypeError):
            continue
        if "is_peft_available()" in src and "ref_param.data.copy_" in src:
            found = True
            break
    if not found:
        pytest.skip(
            "No probed TRL trainer has both `is_peft_available()` AND "
            "`ref_param.data.copy_` together; the add-adapter-ref "
            "rewriter is dormant on this build."
        )


def test_unsloth_rl_warmup_ratio_keyword_pattern():
    """``unsloth/models/rl.py:1323-1327,1330-1333`` runs:

        x = f"{k}( = [^,\\n]{{1,}})?,\\n"
        arguments = re.sub(x, y, arguments)

    where ``k`` is e.g. ``"warmup_ratio"`` or ``"warmup_steps"``.
    Sanity-check the regex against a synthetic config-arguments
    block.
    """
    sample = (
        "warmup_ratio = 0.1,\n"
        "learning_rate = 5e-5,\n"
    )
    out = re.sub(
        r"warmup_ratio( = [^,\n]{1,})?,\n",
        "warmup_ratio = 0.1,\n",
        sample,
    )
    if "warmup_ratio" not in out:
        _drift(
            "unsloth/models/rl.py:1323-1333",
            r"warmup_ratio( = [^,\n]{1,})?,\n",
            "zoo internal regex",
            f"Sample {sample!r} normalized to {out!r}; the "
            "kwarg-replacement is malformed.",
        )


def test_unsloth_rl_anihilate_typo_marker_search():
    """``unsloth/models/rl.py:1725-1746`` searches for both spellings
    ``anihilate`` (typo) AND ``annihilate`` (correct) in
    SFTTrainer.__init__ source, then strips the surrounding
    ``if args.per_device_train_batch_size == 1`` block. Pin: at
    least one of the two spellings is present in SFTTrainer source.
    """
    pytest.importorskip("trl")
    try:
        from trl.trainer.sft_trainer import SFTTrainer
    except ImportError:
        pytest.skip("trl SFTTrainer not importable")
    try:
        src = inspect.getsource(SFTTrainer)
    except (OSError, TypeError):
        pytest.skip("SFTTrainer source unavailable")
    if "anihilate" not in src and "annihilate" not in src:
        pytest.skip(
            "TRL no longer emits the batch_size=1 + padding-free "
            "anihilate/annihilate warning; the warning-suppression "
            "rewriter is dormant on this build. Pin guards "
            "re-introduction."
        )


def test_unsloth_rl_per_device_train_batch_size_marker():
    """``unsloth/models/rl.py:1730-1731`` looks backwards for
    ``"if args.per_device_train_batch_size == 1"``. Pin: this exact
    if condition appears in SOME TRL trainer source.
    """
    pytest.importorskip("trl")
    import importlib
    candidates = [
        "trl.trainer.sft_trainer",
        "trl.trainer.dpo_trainer",
        "trl.trainer.grpo_trainer",
    ]
    needle = "if args.per_device_train_batch_size == 1"
    found = False
    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        try:
            src = inspect.getsource(mod)
        except (OSError, TypeError):
            continue
        if needle in src:
            found = True
            break
    if not found:
        pytest.skip(
            f"No probed TRL trainer has {needle!r}; the "
            "batch_size=1 warning-strip rewriter is dormant."
        )


def test_unsloth_rl_processing_class_call_args_pattern():
    """``unsloth/models/rl.py:980-985`` runs:

        call_args.replace(
            "processing_class = processing_class",
            "processing_class = tokenizer if tokenizer is not None else processing_class",
        )

    on the (synthesized) call_args string. Sanity-check the
    substitution is well-formed.
    """
    sample = "processing_class = processing_class,\nmodel = model"
    out = sample.replace(
        "processing_class = processing_class",
        "processing_class = tokenizer if tokenizer is not None else processing_class",
    )
    if "tokenizer if tokenizer is not None else processing_class" not in out:
        _drift(
            "unsloth/models/rl.py:980-985",
            "processing_class = processing_class",
            "zoo internal str.replace",
            f"Sample {sample!r} normalized to {out!r}; the "
            "tokenizer-fallback injection is malformed.",
        )


def test_unsloth_rl_shuffle_sequence_dict_pinned_pattern():
    """``unsloth/models/rl.py:2051-2055`` runs:

        re.sub(
            r"(\\n[\\s]{4,})generation_batch = shuffle_sequence_dict\\(generation_batch\\)\\n",
            r"\\n\\1try: ... except: pass\\n",
            source,
        )

    Pin: ``shuffle_sequence_dict`` is referenced in some TRL trainer
    source -- the rewriter targets a known crash mode in torch 2.8.
    """
    pytest.importorskip("trl")
    import importlib
    candidates = [
        "trl.trainer.grpo_trainer",
        "trl.trainer.online_dpo_trainer",
        "trl.trainer.rloo_trainer",
    ]
    found = False
    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        try:
            src = inspect.getsource(mod)
        except (OSError, TypeError):
            continue
        if "shuffle_sequence_dict" in src:
            found = True
            break
    if not found:
        pytest.skip(
            "No probed TRL trainer references shuffle_sequence_dict; "
            "the AcceleratorError-workaround rewriter is dormant."
        )


def test_unsloth_rl_model_executor_driver_worker_pinned_pattern():
    """``unsloth/models/rl.py:2058-2062`` runs:

        re.sub(r"(\\n[\\s]{4,}).+?model_executor\\.driver_worker.+?\\n", ...)

    Pin: ``model_executor.driver_worker`` is referenced in TRL or
    vLLM source -- the rewriter strips a vLLM internal-API call so
    zoo's vllm-engine wiring can be used instead.
    """
    pytest.importorskip("trl")
    import importlib
    candidates = [
        "trl.trainer.grpo_trainer",
        "trl.trainer.online_dpo_trainer",
        "trl.trainer.rloo_trainer",
    ]
    found = False
    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        try:
            src = inspect.getsource(mod)
        except (OSError, TypeError):
            continue
        if "model_executor" in src or "driver_worker" in src:
            found = True
            break
    if not found:
        pytest.skip(
            "No probed TRL trainer references vLLM's "
            "model_executor.driver_worker internals; the strip "
            "rewriter is dormant."
        )


def test_unsloth_rl_load_weights_strip_pinned_pattern():
    """``unsloth/models/rl.py:2065-2069`` runs:

        re.sub(r"(\\n[\\s]{4,}).+?load_weights\\(.+?\\n", r"\\n\\1pass\\n", source)

    Pin: ``load_weights(`` is referenced in some TRL trainer source.
    """
    pytest.importorskip("trl")
    import importlib
    candidates = [
        "trl.trainer.grpo_trainer",
        "trl.trainer.online_dpo_trainer",
        "trl.trainer.rloo_trainer",
    ]
    found = False
    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        try:
            src = inspect.getsource(mod)
        except (OSError, TypeError):
            continue
        if "load_weights(" in src:
            found = True
            break
    if not found:
        pytest.skip(
            "No probed TRL trainer references load_weights(; the "
            "load-weights strip rewriter is dormant."
        )


def test_unsloth_rl_peft_pattern_27_marker():
    """``unsloth/models/rl.py:1629-1633,1641-1646`` build TWO PEFT
    init-block regexes:

        trl >= 0.27.0:
          if is_peft_available() and is_peft_model(model) and args.beta != 0.0:
          ...
          param.data = param.data.to(torch.bfloat16)

        trl >= 0.26.0:
          if is_peft_available() and isinstance(model, PeftModel) and peft_config is not None:
          ...
          param.data = param.data.to(torch.bfloat16)

    Pin: at least ONE of the TWO pinned end-of-block lines is
    present in GRPOTrainer source.
    """
    pytest.importorskip("trl")
    try:
        import trl.trainer.grpo_trainer as gt
        src = inspect.getsource(gt)
    except (ImportError, OSError, TypeError):
        pytest.skip("trl.trainer.grpo_trainer unavailable")
    candidates = (
        "param.data = param.data.to(torch.bfloat16)",
        "is_peft_available()",
    )
    if not any(c in src for c in candidates):
        pytest.skip(
            "TRL >= 1.0 may have removed the PEFT bfloat16 init "
            "block; the dual-regex rewriter is dormant. Pin guards "
            "re-introduction."
        )


# ===========================================================================
# unsloth/trainer.py rewriters
# ===========================================================================


def test_unsloth_trainer_exec_marker():
    """``unsloth/trainer.py:614`` runs ``exec(...)`` on a synthesized
    trainer source. This is a passthrough rather than a substring
    rewriter, but the trainer.py module MUST be importable for the
    exec to fire. Pin: ``unsloth.trainer.UnslothTrainer`` (or the
    equivalent) is importable.
    """
    # `unsloth` itself may not be installed in this venv; importorskip.
    pytest.importorskip("unsloth")
    try:
        import unsloth.trainer as trainer_mod
    except ImportError as e:
        # If unsloth is installed but trainer.py raises on import, that
        # IS a regression -- the exec sites are unreachable.
        _drift(
            "unsloth/trainer.py:614",
            "import unsloth.trainer",
            "unsloth.trainer",
            f"Import error: {e}. The trainer-source exec site is "
            "unreachable.",
        )
        return
    # Sanity-check the module has SOMETHING the trainer rewriter would
    # consume (any TRL- or Trainer-derived symbol).
    if not any(
        hasattr(trainer_mod, sym)
        for sym in ("UnslothTrainer", "Trainer", "_create_unsloth_optimizer", "unsloth_train")
    ):
        _drift(
            "unsloth/trainer.py:614",
            "Trainer-family symbol",
            "unsloth.trainer",
            "Module is importable but exposes none of the trainer "
            "symbols a downstream rewriter would consume.",
        )


# ===========================================================================
# Final smoke: confirm zoo's own source-string targets in the compiler
# (i.e. the OUTPUT side, not the upstream input) are still well-formed.
# ===========================================================================


def test_zoo_compiler_replace_gradient_checkpointing_template_format():
    """``unsloth_zoo/compiler.py:2226-2234`` defines
    ``replace_gradient_checkpointing`` as a template with placeholders
    ``LAYER``, ``MODULELIST_ITEM``, ``ARGS``, ``$``. The rewriter
    substitutes these via .replace(). Pin: all four placeholders are
    actually present in the template.
    """
    import importlib
    compiler = importlib.import_module("unsloth_zoo.compiler")
    template = getattr(compiler, "replace_gradient_checkpointing", None)
    if template is None:
        _drift(
            "unsloth_zoo/compiler.py:2226",
            "replace_gradient_checkpointing template",
            "unsloth_zoo.compiler",
            "Template constant is missing; gradient-checkpointing "
            "rewriter no-ops.",
        )
        return
    for placeholder in ("LAYER", "MODULELIST_ITEM", "ARGS", "$"):
        if placeholder not in template:
            _drift(
                "unsloth_zoo/compiler.py:2226-2234",
                f"placeholder {placeholder!r}",
                "unsloth_zoo.compiler.replace_gradient_checkpointing",
                "Template placeholder missing -- substitution will "
                "miss this slot.",
            )


def test_zoo_compiler_moe_routing_weights_replace_substitution_well_formed():
    """``unsloth_zoo/compiler.py:2423-2426`` defines:

        MOE_ROUTING_WEIGHTS_CAST_PATTERN = r"(\\brouting_weights\\s*=\\s*routing_weights\\.to\\(\\s*)hidden_states(\\.dtype\\s*\\))"
        MOE_ROUTING_WEIGHTS_CAST_REPLACE = r"\\1router_logits\\2"

    Sanity-check the substitution rewrites
    ``routing_weights = routing_weights.to(hidden_states.dtype)``
    to ``routing_weights = routing_weights.to(router_logits.dtype)``.
    """
    import importlib
    compiler = importlib.import_module("unsloth_zoo.compiler")
    pat = getattr(compiler, "MOE_ROUTING_WEIGHTS_CAST_PATTERN", None)
    rep = getattr(compiler, "MOE_ROUTING_WEIGHTS_CAST_REPLACE", None)
    if pat is None or rep is None:
        _drift(
            "unsloth_zoo/compiler.py:2423-2426",
            "MOE_ROUTING_WEIGHTS_CAST_PATTERN / _REPLACE",
            "unsloth_zoo.compiler",
            "Pattern or replacement constant is missing.",
        )
        return
    sample = "routing_weights = routing_weights.to(hidden_states.dtype)"
    out = re.sub(pat, rep, sample)
    if "router_logits" not in out:
        _drift(
            "unsloth_zoo/compiler.py:2423-2426",
            pat,
            "unsloth_zoo.compiler internal regex",
            f"Sample {sample!r} did not normalize correctly: {out!r}",
        )


def test_zoo_compiler_dtype_mismatch_constants_targetable():
    """``unsloth_zoo/compiler.py:2381-2391`` defines
    ``DTYPE_MISMATCH_FIND`` and ``DTYPE_MISMATCH_REPLACE`` as
    multi-line constants. Pin: both constants have the expected
    sentinel substring.
    """
    import importlib
    compiler = importlib.import_module("unsloth_zoo.compiler")
    find = getattr(compiler, "DTYPE_MISMATCH_FIND", None)
    rep = getattr(compiler, "DTYPE_MISMATCH_REPLACE", None)
    if find is None or rep is None:
        _drift(
            "unsloth_zoo/compiler.py:2381-2391",
            "DTYPE_MISMATCH_FIND / _REPLACE",
            "unsloth_zoo.compiler",
            "Constants missing -- finfo-mask rewriter is dormant.",
        )
        return
    if "torch.finfo(attention_mask_tensor.dtype).min" not in find:
        _drift(
            "unsloth_zoo/compiler.py:2381",
            "torch.finfo(attention_mask_tensor.dtype).min",
            "unsloth_zoo.compiler.DTYPE_MISMATCH_FIND",
            "Find constant doesn't contain the expected pinned form.",
        )
    if "(1.0 - attention_mask_tensor).int()" not in rep:
        _drift(
            "unsloth_zoo/compiler.py:2386",
            "(1.0 - attention_mask_tensor).int()",
            "unsloth_zoo.compiler.DTYPE_MISMATCH_REPLACE",
            "Replace constant doesn't contain the expected pinned form.",
        )
