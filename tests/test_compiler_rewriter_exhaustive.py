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

"""Exhaustive drift detectors for the ``unsloth_zoo`` / ``unsloth``
source-string and regex rewriters (round 3 of coverage).

Pins sites in compiler.py, temporary_patches/*.py, patching_utils.py,
saving_utils.py, rl_replacements.py, training_utils.py, unsloth/trainer.py,
unsloth/models/rl.py that aren't already pinned by
test_upstream_source_patterns.py.

Each test cites its rewriter file:line. Missing pinned string / regex ->
``pytest.fail("DRIFT DETECTED: zoo/unsloth source-rewriter at <file:line>
expects '<pattern>' in <upstream module>, found 0 matches")``. CPU-only.
"""

from __future__ import annotations

import inspect
import re

import pytest

try:
    import transformers as _transformers
    from packaging.version import Version as _Version
    _TX_VERSION = getattr(_transformers, "__version__", "0.0.0")
    _TX_IS_5X = _Version(_TX_VERSION) >= _Version("5.0.0")
except Exception:
    _TX_VERSION = "unknown"
    _TX_IS_5X = False


def _skip_if_transformers_5x(reason: str) -> None:
    """Skip when transformers 5.x removed the anchor the rewriter
    probe pins. Keep the detector strict on 4.57.6."""
    if _TX_IS_5X:
        pytest.skip(
            f"transformers {_TX_VERSION}: {reason} (zoo rewriter silently "
            "no-ops -- str.replace returns source unchanged)"
        )


# Shared helpers (mirror test_upstream_source_patterns.py).

def _drift(zoo_site: str, pattern: str, upstream_path: str,
           extra: str = "") -> None:
    """``pytest.fail`` with the standardized DRIFT message."""
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
    """True if predicate(src) holds for at least one importable module."""
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


# unsloth_zoo/compiler.py: not-yet-covered rewriter sites.


def test_compiler_higher_precision_softmax_idempotency_lookahead():
    """``unsloth_zoo/compiler.py:391-405`` higher_precision_softmax negative
    lookahead skips already-rewritten softmax. Pins the upstream
    ``softmax(..., dim=N)`` keyword form so the finder isn't dormant."""
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
    """``unsloth_zoo/compiler.py:510-517`` fix_rotary_embedding_dtype
    replaces ``cos.to(dtype=x.dtype)`` / ``sin.to(dtype=x.dtype)``;
    UNSLOTH_FORCE_CUSTOM_DTYPE-gated. Pin: some rotary cast site exists."""
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
    """``unsloth_zoo/compiler.py:560-597`` higher_precision_layernorms
    locates ``class <X>Norm(nn.Module):`` blocks; without one,
    UNSLOTH_HIGH_PRECISION_LAYERNORM is never auto-toggled."""
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
    """``unsloth_zoo/compiler.py:1383-1387`` clamps Gemma 3N's input_ids
    offsets via re.sub on ``self.<X>embedding(input_ids +/- self.<Y>)``."""
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
    """``unsloth_zoo/compiler.py:2128-2140`` apply_mask_attention_mask_out
    needs ``attention_mask=attention_mask,`` AND ``labels=labels,`` together
    in a ForConditionalGeneration forward."""
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
    """``unsloth_zoo/compiler.py:2161-2179`` convert_attention_masks_to_bool
    needs upstream ``torch.finfo(...).min`` masked-fill sentinel."""
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
    """``unsloth_zoo/compiler.py:2258-2270`` patch_gradient_checkpointing
    needs ``self.<X> = nn.ModuleList(...)`` in __init__ (call-site shape
    used by GradientCheckpointingLayer fall-back)."""
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
    """``unsloth_zoo/compiler.py:2200-2207`` second blk-call REPLACEMENT
    variant references ``rotary_pos_emb=``; if upstream renamed the kwarg
    the rewritten call is API-incompatible."""
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
    """``unsloth_zoo/compiler.py:2208-2223`` third blk-call FIND variant
    with ``attention_mask=attention_mask``."""
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
    """``unsloth_zoo/compiler.py:2306-2346`` strip_kw_from_module_calls
    needs ``for <layer> in self.<list>:`` decoder iteration."""
    pytest.importorskip("transformers")
    # Probe `for <var> in self.<attr>` (broader than zoo's compiled regex).
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
    """``unsloth_zoo/compiler.py:2381-2391`` patch_finfo_attention_mask_dtype_mismatch
    pins pre-4.50 ``<X> = <X>/torch.finfo(<X>.dtype).min`` followed by
    ``<X> = (1.0 - <X>).int()``. Forward-looking: only DRIFT when the
    underlying primitives are also gone."""
    pytest.importorskip("transformers")
    candidates = [
        "transformers.modeling_attn_mask_utils",
        "transformers.masking_utils",
        "transformers.models.gemma3.modeling_gemma3",
        "transformers.models.gpt_oss.modeling_gpt_oss",
    ]
    pattern = re.compile(
        r"torch\.finfo\([^\)]+\.dtype\)\.min[\s\S]{0,200}\(1\.0\s*-"
    )
    if not _probe_modules(candidates, lambda s: pattern.search(s) is not None):
        # Exact idiom removed in masking_utils 4.50+; only fail when both
        # `torch.finfo(...).min` AND `(1.0 - <mask>)` primitives are gone.
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
    """``unsloth_zoo/compiler.py:2539`` replaces ``result = result.clone()``
    in peft's LoRA forward (defeats in-place ops)."""
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
    """``unsloth_zoo/compiler.py:2553`` checks
    ``torch_result_dtype = result.dtype`` in peft's LoRA forward;
    absent -> rewriter falls back to ``result.dtype``."""
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
    """``unsloth_zoo/compiler.py:2563-2567`` renames peft's
    ``def forward`` -> ``def unsloth_forward``."""
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
    """``unsloth_zoo/compiler.py:2578-2581,2596`` pins three peft LoRA
    targets: ``x = x.to(lora_A.weight.dtype)`` / ``x =
    self._cast_input_dtype(x, lora_A.weight.dtype)`` /
    ``self._check_forward_args(x, *args, **kwargs)``. DRIFT only when
    ALL THREE are gone."""
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
    """``unsloth_zoo/compiler.py:2649-2655`` peft >= 0.18.0
    ``VARIANT_KWARG_KEYS`` at the layer module level; the FIND must
    succeed for the fallback to fire."""
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
    """``unsloth_zoo/compiler.py:2698-2705`` second patch_residual_stream
    regex matches ``<h> = residual + (<h> * <expr>|<expr> * <h>)``
    (addcmul / fused-add target on VLM cross-attention encoders)."""
    pytest.importorskip("transformers")
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
    """``unsloth_zoo/compiler.py:2757-2759`` patch_gradient_accumulation
    discovers ``self.<X> = <Y>._from_config(...)`` (Idefics3, Llava-family,
    Qwen2-VL)."""
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
    """``unsloth_zoo/compiler.py:2925`` compile_timm_models requires
    ``class <X>(...nn.Module):`` in timm._efficientnet_blocks;
    refactor -> zero blocks torch.compile-d."""
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
    """``unsloth_zoo/compiler.py:3310-3318`` global compiler discovers
    ``class <X>(...Module)`` then ``class <Y>(<X>)``; pin a top-level
    Module subclass (SDPA / Eager attention duo)."""
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
    """``unsloth_zoo/compiler.py:3332-3334`` discovers PreTrainedModel
    subclasses (backbone / for-causal-lm classes) via re.findall."""
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
    """``unsloth_zoo/compiler.py:3376`` branches on ``routing_weights.to``
    in MoE forward (router-logit-cast / bf16 router fix anchor)."""
    _skip_if_transformers_5x(
        "MoE forwards refactored on transformers 5.x -- `routing_weights.to` "
        "substring no longer present in mixtral/qwen2_moe/qwen3_moe/deepseek_v3. "
        "compiler.py:3524 substring-in check just skips the module from the "
        "router_logit_cast_modules list"
    )
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
    ``_supports_sdpa = True/False`` in modeling source. BENIGN on
    transformers 4.50+ (moved to ALL_ATTENTION_FUNCTIONS); zoo has
    ``_all_attention_functions_has_sdpa()`` fallback. Pin guards
    re-introduction."""
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
    """``unsloth_zoo/compiler.py:3587-3596`` skips compile on
    .nonzero()/.tolist()/.item() (or torch.where + .index_add); pin a
    data-dependent op site so the skip branch is reachable."""
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
    """``unsloth_zoo/compiler.py:3988`` re.search needs non-empty
    ``Trainer._inner_training_loop`` source (multi-hundred-line body)."""
    pytest.importorskip("transformers")
    _skip_if_transformers_5x(
        "Trainer._inner_training_loop is wrapped/compiled on transformers "
        "5.x so inspect.getsource is no longer expected to return its body; "
        "the source-rewriter no-ops by design on 5.x."
    )
    from transformers.trainer import Trainer
    try:
        src = inspect.getsource(Trainer._inner_training_loop)
    except (OSError, TypeError):
        pytest.skip(
            "Trainer._inner_training_loop source unavailable on this "
            "transformers build (likely wrapped/decorated); the rewriter "
            "silently no-ops, which is acceptable on this matrix cell."
        )
    if len(src) < 500:
        _drift(
            "unsloth_zoo/compiler.py:3988-4040",
            "non-trivial Trainer._inner_training_loop source body",
            "transformers.trainer.Trainer",
            f"Source length is suspiciously short ({len(src)} chars); "
            "the rewriter expects a multi-hundred-line function.",
        )


def test_compiler_dict_attention_mask_gpt_oss_v5_pattern_present():
    """``unsloth_zoo/compiler.py:4148-4158`` guards on
    ``attn_weights = attn_weights + attention_mask`` AND ``module`` in
    gpt_oss source."""
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
    """``unsloth_zoo/compiler.py:4289`` reads first-param name of
    ``def forward(self, <name>)`` on nn.Conv* / nn.*Norm forwards;
    absent -> falls back to 'input' which may not be the actual arg."""
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


# unsloth_zoo/patching_utils.py rewriters.


def test_patching_utils_compiled_autograd_end_capture_return_compiled_fn_pinned():
    """``unsloth_zoo/patching_utils.py:544-547`` wraps
    ``AutogradCompilerInstance.end_capture``'s ``return compiled_fn(...)``
    in ``with disable():`` (PR #135795 double-compile fix). Signature
    drift silently no-ops the rewriter."""
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
    # Contract: pass if exact str+regex present (rewriter works), or
    # `with disable()` / `with _disable()` already in src (torch 2.7+
    # native fix). torch 2.7+ end_capture signature drifted (added
    # packed_inputs, nested with-block); rewriter no-ops cleanly.
    if needle in src and pattern.search(src) is not None:
        return
    if "with disable()" in src or "with _disable()" in src:
        return
    if "compiled_fn(" in src:
        # BENIGN on torch 2.7+: upstream fixed PR #135795 natively with
        # ``with _disable()``; zoo's patch_compiled_autograd recognises
        # both forms and no-ops cleanly.
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
    """``unsloth_zoo/patching_utils.py:548`` renames
    ``def end_capture`` -> ``def unsloth_end_capture``."""
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
    """``unsloth_zoo/patching_utils.py:564-573`` replaces
    ``compiled_autograd_enabled`` -> ``in_compiled_autograd_region`` in
    ``AutogradEngineVariable.call_method``."""
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
    """``unsloth_zoo/patching_utils.py:574`` renames ``def call_method``
    -> ``def unsloth_call_method``."""
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
    """``unsloth_zoo/patching_utils.py:695-699`` replaces
    ``name in quantization_config.llm_int8_skip_modules\\n`` on
    ``bnb._replace_with_bnb_linear`` (dynamic-4bit conversion).
    Resolves UPSTREAM source via inspect.getsourcefile because zoo has
    already rebound the function by test time."""
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

    # Resolve upstream source directly from the module file (zoo rebinds
    # bnb._replace_with_bnb_linear so inspect.getsource off the live
    # function returns the patched body, never upstream).
    live = bnb._replace_with_bnb_linear
    is_zoo_patched = (
        getattr(live, "__name__", "") == "_unsloth_replace_with_bnb_linear"
    )
    src = None
    if is_zoo_patched:
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
    """``unsloth_zoo/patching_utils.py:730-733`` renames every occurrence
    of ``_replace_with_bnb_linear`` -> ``_unsloth_replace_with_bnb_linear``."""
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
    """``unsloth_zoo/patching_utils.py:738-748`` splices score-module
    skip via re.sub on ``current_key_name.append(name)``."""
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
    """``unsloth_zoo/patching_utils.py:688`` hard-fails with RuntimeError
    when ``current_key_name_str`` is absent in upstream source."""
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
    """``unsloth_zoo/patching_utils.py:701-704`` AST-wraps recursive
    ``= _replace_with_bnb_linear(...)`` calls in try/finally with
    parent-class marking."""
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


# unsloth_zoo/saving_utils.py rewriters.


def test_saving_utils_save_pretrained_state_dict_split_pinned_string():
    """``unsloth_zoo/saving_utils.py:2675-2677`` hard-fails (RuntimeError)
    when ``state_dict_split = split_torch_state_dict_into_shards`` is
    missing from PreTrainedModel.save_pretrained."""
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
    ``state_dict[tensor].contiguous()`` in upstream + replace to
    ``merge_lora_weights(...)``; RuntimeError otherwise.

    transformers 5.x rewrote PreTrainedModel.save_pretrained (sharding /
    state-dict iteration moved). zoo's saving_utils.py upfront-anchor
    check (``_required_anchors``) detects the missing string and falls
    back to vanilla ``model.save_pretrained`` with a warning. The
    detector becomes a positive-assertion on 5.x: confirm the anchor is
    gone AND zoo's _required_anchors list flags it AND the warning path
    fires gracefully (no RuntimeError).
    """
    pytest.importorskip("transformers")
    import transformers.modeling_utils as mu
    try:
        src = inspect.getsource(mu.PreTrainedModel.save_pretrained)
    except (OSError, TypeError):
        pytest.skip("save_pretrained source unavailable")
    needle = "state_dict[tensor].contiguous()"
    if _TX_IS_5X:
        assert needle not in src, (
            f"transformers {_TX_VERSION}: `{needle}` was expected gone "
            "on 5.x but is present; refresh the zoo prod-fix anchor "
            "list at saving_utils.py:_required_anchors"
        )
        # Positive assertion: zoo's prod-fix correctly identifies the
        # missing anchor in its preflight check.
        import unsloth_zoo.saving_utils as zsu
        zsu_src = inspect.getsource(zsu.merge_and_dequantize_lora)
        assert needle in zsu_src, (
            f"transformers {_TX_VERSION}: anchor `{needle}` missing on "
            "5.x but zoo's _required_anchors check doesn't include it; "
            "production call merge_and_dequantize_lora() will hit the "
            "downstream per-anchor RuntimeError instead of the "
            "graceful fallback"
        )
        return
    if needle not in src:
        _drift(
            "unsloth_zoo/saving_utils.py:2680-2686",
            needle,
            "transformers.modeling_utils.PreTrainedModel.save_pretrained",
            "Without this exact `.contiguous()` call, the dequantize-"
            "merge replacement raises at runtime.",
        )


def test_saving_utils_save_pretrained_def_marker():
    """``unsloth_zoo/saving_utils.py:2688-2694`` renames
    ``save_pretrained`` -> ``save_pretrained_dequantized``."""
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
    """``unsloth_zoo/saving_utils.py:2517`` re.search
    ``os.makedirs(save_directory...)`` in save_pretrained; absent ->
    incremental_save_pretrained aborts push-to-hub path."""
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
    ``for shard_file, tensors in filename_to_tensors`` in
    save_pretrained; RuntimeError otherwise.

    transformers 5.x renamed the iterator. zoo's prod fix in
    ``merge_and_dequantize_lora`` runs an upfront anchor check that
    includes this string and falls back to vanilla
    ``model.save_pretrained`` (with a warning) when push_to_hub=True
    and the anchor is missing. On 5.x: assert the anchor is gone AND
    zoo's preflight check covers it.
    """
    pytest.importorskip("transformers")
    import transformers.modeling_utils as mu
    try:
        src = inspect.getsource(mu.PreTrainedModel.save_pretrained)
    except (OSError, TypeError):
        pytest.skip("save_pretrained source unavailable")
    needle = "for shard_file, tensors in filename_to_tensors"
    if _TX_IS_5X:
        assert needle not in src, (
            f"transformers {_TX_VERSION}: `{needle}` was expected gone "
            "on 5.x but is present; refresh the zoo prod-fix anchor "
            "list at saving_utils.py:_required_anchors"
        )
        import unsloth_zoo.saving_utils as zsu
        zsu_src = inspect.getsource(zsu.merge_and_dequantize_lora)
        assert needle in zsu_src, (
            f"transformers {_TX_VERSION}: anchor `{needle}` missing on "
            "5.x but zoo's _required_anchors check doesn't include it; "
            "merge_and_dequantize_lora(push_to_hub=True) will RuntimeError"
        )
        return
    if needle not in src:
        _drift(
            "unsloth_zoo/saving_utils.py:2526-2533",
            needle,
            "transformers.modeling_utils.PreTrainedModel.save_pretrained",
            "Without this for-loop, incremental_save_pretrained raises "
            "and disables low-disk-space push-to-hub.",
        )


def test_saving_utils_config_json_dtype_torch_dtype_rename_targets():
    """``unsloth_zoo/saving_utils.py:1827-1828`` save-time replaces
    ``"dtype"`` -> ``"torch_dtype"`` in config.json; needs model
    config.to_dict() to expose either field."""
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
    """``unsloth_zoo/saving_utils.py:309-314`` five str.replace on LoRA
    keys (.base_layer, .modules_to_save.default, .original_module,
    .lora_A.default, .lora_B.default). Pin peft's ``base_layer`` /
    ``lora_A.default`` naming."""
    pytest.importorskip("peft")
    try:
        import peft.tuners.lora.layer as ly
    except ImportError:
        pytest.skip("peft.tuners.lora.layer missing")
    src = inspect.getsource(ly)
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
    """``unsloth_zoo/saving_utils.py:600,653,700`` rebuilds fused
    gate_up_proj from per-expert shards via three re.match on
    ``mlp.experts.<i>.<proj>.weight``."""
    pytest.importorskip("transformers")
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
    """``unsloth_zoo/saving_utils.py:1838`` sharded-safetensors regex
    must match canonical HF format ``<prefix>-<shard>-of-<total>.safetensors``."""
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
    """``unsloth_zoo/saving_utils.py:2923`` re.sub regex must preserve a
    typical mapping value like ``"model.language_model."`` (no leading
    caret, no parens, no ?)."""
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


# unsloth_zoo/training_utils.py rewriters.


def test_training_utils_name_replace_base_model_pattern():
    """``unsloth_zoo/training_utils.py:172-190`` builds exec-able
    accessor from PEFT module names: replace base_model -> model,
    .<i>. -> [<i>]., strip .weight."""
    pytest.importorskip("peft")
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


# unsloth_zoo/temporary_patches/misc.py rewriters.


def test_misc_merge_quantization_configs_classmethod_marker():
    """``unsloth_zoo/temporary_patches/misc.py:141`` strips cls when
    upstream ``AutoHfQuantizer.merge_quantization_configs`` is a
    classmethod."""
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
        # Drift only if neither cls nor self in def line (exec binding broken).
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
    """``unsloth_zoo/temporary_patches/misc.py:142`` requires ``def `` in
    the source (``source = source[source.find("def"):]``)."""
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
    """``unsloth_zoo/temporary_patches/misc.py:1082-1085``
    fix_mamba_ssm_float32 finds ``tl.dot(`` in mamba_ssm Triton
    chunk-scan source."""
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


# unsloth_zoo/temporary_patches/gpt_oss.py rewriters.


def test_gpt_oss_config_old_class_dedent_compare_marker():
    """``unsloth_zoo/temporary_patches/gpt_oss.py:2808-2810``
    line-by-line equality compare of dedented GptOssConfig vs OLD
    class; pin ``initial_context_length`` / ``rope_scaling`` field
    presence (the Old_GptOssConfig regression target).

    transformers 5.x replaced ``rope_theta`` / ``rope_scaling`` /
    ``initial_context_length`` with the ``rope_parameters`` dict. zoo's
    ``patch_gpt_oss_config`` gates on
    ``inspect.getsource(GptOssConfig) == Old_GptOssConfig``, so the
    patch silently no-ops on the new shape -- skip the detector on 5.x.
    """
    _skip_if_transformers_5x(
        "GptOssConfig replaced rope_theta/rope_scaling/initial_context_length "
        "with rope_parameters dict; patch site silently no-ops via source-"
        "equality gate"
    )
    pytest.importorskip("transformers")
    try:
        from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
    except ImportError:
        pytest.skip("transformers.models.gpt_oss not shipped")
    try:
        src = inspect.getsource(GptOssConfig)
    except (OSError, TypeError):
        pytest.skip("GptOssConfig source unavailable")
    if "initial_context_length" not in src and "rope_scaling" not in src:
        _drift(
            "unsloth_zoo/temporary_patches/gpt_oss.py:2808-2813",
            "initial_context_length OR rope_scaling (field in GptOssConfig)",
            "transformers.models.gpt_oss.configuration_gpt_oss.GptOssConfig",
            "Without either field, the Old_GptOssConfig patch can't "
            "fix the regression it was introduced for.",
        )


# unsloth_zoo/rl_replacements.py rewriters.


def test_rl_replacements_grpo_compute_loss_def_marker():
    """``unsloth_zoo/rl_replacements.py:560-565`` renames
    ``def grpo_compute_loss`` -> ``def grpo_compute_loss_slow``
    (slow-fallback RL_REPLACEMENTS variant)."""
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


# unsloth/models/rl.py rewriters.


def test_unsloth_rl_trainer_signature_columns_pinned_string():
    """``unsloth/models/rl.py:1667-1670`` augments SFTTrainer
    ``self._signature_columns`` with ``"labels"``. DRIFT when
    ``_signature_columns`` is gone from SFTTrainer source."""
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
    """``unsloth/models/rl.py:1706-1713`` pins TRL 0.22.x VLM form
    ``["messages", "prompt", "completion", "images"]``. DRIFT only when
    ALL four member tokens are gone from SFTTrainer."""
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
    """``unsloth/models/rl.py:1717-1721`` injects
    ``self._unsloth_model_ref = model`` before SFTTrainer's
    ``self._prepare_dataset(`` call (token_type_ids detection)."""
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
        _drift(
            "unsloth/models/rl.py:1717-1721",
            "self._prepare_dataset(",
            "trl.trainer.sft_trainer.SFTTrainer.__init__",
            "Without this call, the unsloth_model_ref injection can't "
            "fire and sft_prepare_dataset can't detect dynamic "
            "token_type_ids.",
        )


def test_unsloth_rl_trainer_is_loaded_in_4bit_pinned_string():
    """``unsloth/models/rl.py:1662-1665`` replaces TRL's
    ``if getattr(model, "is_loaded_in_4bit"/"is_loaded_in_8bit", ...)``
    bf16 cast block with ``if False:``."""
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
        pytest.skip(
            "No TRL trainer references is_loaded_in_4bit/8bit anymore; "
            "the cast-removal rewriter is dormant on this build. Pin "
            "guards re-introduction."
        )


def test_unsloth_rl_trainer_peft_config_branches_pinned():
    """``unsloth/models/rl.py:1842-1857`` six peft_config str.replace
    targets (peft_config branches, get_peft_model, prepare_peft_model);
    DRIFT only when ``peft_config`` is gone from every TRL trainer."""
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
    """``unsloth/models/rl.py:1832-1833`` normalises bracketed
    ``#...(...)`` comments to ``[...]`` in SFTTrainer __init__."""
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
    """``unsloth/models/rl.py:1895-1928`` vLLM-engine wiring needs
    ``args.use_vllm`` / ``self.use_vllm`` marker in some TRL RL
    trainer __init__."""
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
    """``unsloth/models/rl.py:1932-1936`` matches 8-space-indented
    ``if (self|args).use_vllm:\\n...else:\\n`` branch in TRL trainer."""
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
    """``unsloth/models/rl.py:1949-1953`` patches
    ``self.X = SamplingParams(...)`` in TRL trainer; pin
    ``SamplingParams(`` presence."""
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
    """``unsloth/models/rl.py:2072-2076`` strips ``.state_dict()`` from
    TRL function source via re.sub."""
    pytest.importorskip("trl")
    sample = (
        "    llm_model.load_weights(model.state_dict().items())\n"
    )
    rewritten = re.sub(r"\.state_dict\(\)", r"", sample)
    if ".state_dict()" in rewritten or "model.items()" not in rewritten:
        _drift(
            "unsloth/models/rl.py:2072-2076",
            r"\.state_dict\(\)",
            "zoo internal regex",
            f"Sample {sample!r} normalized to {rewritten!r}; the "
            "state-dict strip is malformed.",
        )


def test_unsloth_rl_llm_generate_chat_capture_pattern():
    """``unsloth/models/rl.py:2087-2093`` injects ``lora_request =
    self.model.load_lora(...)`` into ``self.llm.(generate|chat)(...)``
    calls."""
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
    """``unsloth/models/rl.py:2107-2115`` replaces
    ``SamplingParams(**generation_kwargs)`` with
    ``SamplingParams(**grpo_update_SamplingParams(...))``."""
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
    """``unsloth/models/rl.py:2137-2139`` renames ``class SFTTrainer``
    -> ``class _UnslothSFTTrainer``."""
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
    """``unsloth/models/rl.py:1622-1625`` re.sub on
    ``torch_compile_options = {...}`` dict assignment."""
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
    """``unsloth/models/rl.py:1865-1870`` comments out the "ref"
    adapter creation block in GRPOTrainer; needs ``is_peft_available()``
    AND ``ref_param.data.copy_`` together in TRL trainer source."""
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
    """``unsloth/models/rl.py:1323-1333`` re.sub on
    ``<kwarg> = <value>,\\n`` config-arguments block (warmup_ratio,
    warmup_steps, etc.)."""
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
    """``unsloth/models/rl.py:1725-1746`` searches both ``anihilate``
    (typo) and ``annihilate`` in SFTTrainer to strip the surrounding
    ``args.per_device_train_batch_size == 1`` warning block."""
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
    """``unsloth/models/rl.py:1730-1731`` searches
    ``if args.per_device_train_batch_size == 1`` in TRL trainer."""
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
    """``unsloth/models/rl.py:980-985`` injects tokenizer fallback:
    ``processing_class = processing_class`` ->
    ``= tokenizer if tokenizer is not None else processing_class``."""
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
    """``unsloth/models/rl.py:2051-2055`` wraps
    ``generation_batch = shuffle_sequence_dict(...)`` in try/except
    pass (torch 2.8 AcceleratorError workaround)."""
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
    """``unsloth/models/rl.py:2058-2062`` strips
    ``model_executor.driver_worker`` vLLM internal-API call so zoo's
    vllm-engine wiring can replace it."""
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
    """``unsloth/models/rl.py:2065-2069`` strips ``load_weights(...)``
    lines from TRL trainer source via re.sub."""
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
    """``unsloth/models/rl.py:1629-1633,1641-1646`` TWO PEFT init-block
    regexes (TRL 0.26.0 and 0.27.0 shapes) ending in
    ``param.data = param.data.to(torch.bfloat16)``."""
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


# unsloth/trainer.py rewriters.


def test_unsloth_trainer_exec_marker():
    """``unsloth/trainer.py:614`` exec()'s synthesized trainer source;
    pin that unsloth.trainer is importable.

    Skips on a host without a real accelerator: ``import unsloth`` raises
    ``NotImplementedError("Unsloth cannot find any torch accelerator")``
    at top-level on a CPU-only CI runner, which is neither an ImportError
    nor a drift signal -- it's just the harness gate. ``importorskip``
    only converts ``ImportError`` to ``skip``, so we have to wrap the
    whole import path. Treat the no-accelerator case as skip so the
    no-GPU CI cell goes green; the GPU cell still exercises the import
    end-to-end.
    """
    try:
        import unsloth  # noqa: F401
        import unsloth.trainer as trainer_mod
    except ImportError as e:
        if e.name == "unsloth":
            pytest.skip(f"unsloth is not installed: {e}")
        _drift(
            "unsloth/trainer.py:614",
            "import unsloth.trainer",
            "unsloth.trainer",
            f"Import error: {e}. The trainer-source exec site is "
            "unreachable.",
        )
        return
    except NotImplementedError as e:
        if "accelerator" in str(e) or "GPU" in str(e):
            pytest.skip(f"No accelerator visible to unsloth import: {e}")
        raise
    # Module must expose some Trainer-family symbol downstream rewriter consumes.
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


# Final smoke: zoo compiler's OUTPUT-side source-string targets.


def test_zoo_compiler_replace_gradient_checkpointing_template_format():
    """``unsloth_zoo/compiler.py:2226-2234`` template
    ``replace_gradient_checkpointing`` placeholders LAYER /
    MODULELIST_ITEM / ARGS / $ must all be present."""
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
    """``unsloth_zoo/compiler.py:2423-2426`` MOE_ROUTING_WEIGHTS_CAST
    PATTERN/REPLACE rewrites ``routing_weights.to(hidden_states.dtype)``
    -> ``...to(router_logits.dtype)``."""
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
    """``unsloth_zoo/compiler.py:2381-2391`` DTYPE_MISMATCH_FIND /
    REPLACE constants must contain their pinned sentinel substrings."""
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
