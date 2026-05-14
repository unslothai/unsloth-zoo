# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team.
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at
# your option) any later version.
"""Pinned-symbol matrix tests for the transformers / peft / datasets
symbols that ``unsloth_zoo`` reaches into.

Motivating zoo PRs (each test docstring names the precise PR):

  #569 Guard transformers caching_allocator_warmup on low-memory GPUs
  #549 Fix VRAM regression with transformers 5.2+ gradient checkpointing
  #491 Patch should_convert_module for transformers 5.x substring matching
  #488 Fix Gemma3 + Gemma3N on transformers 5.x
  #618 Fix qwen lora extractor for diff peft versions
  #572 Fix forward compatibility with transformers 5.x
  #571 fix gemma3 and csm transformers v5.3 patches
  #560 fix: support prompt/completion datasets with completion_only_loss
  #472 Fix tokenizer guard, ModernBERT attention, gpt_oss MoE unwrap
  #635 Mask for gemma3 attn
"""

from __future__ import annotations

import os
import re
import urllib.error
import urllib.request

import pytest


# transformers anchors per project spec: 4.57.6 (lower-bound) and 5.5.0.
TRANSFORMERS_TAGS = [
    "v4.57.6",
    "v5.0.0",
    "v5.1.0",
    "v5.2.0",
    "v5.3.0",
    "v5.5.0",
    "main",
]

# peft tags covering the API change that motivated zoo PR #618.
PEFT_TAGS = [
    "v0.17.0",
    "v0.18.0",
    "v0.19.1",
    "main",
]


def _fetch_text(repo: str, ref: str, path: str) -> str | None:
    """GET raw.githubusercontent.com/{repo}/{ref}/{path}. Returns None on
    404; pytest.skip on transient errors."""
    url = f"https://raw.githubusercontent.com/{repo}/{ref}/{path}"
    req = urllib.request.Request(url)
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        pytest.skip(f"GitHub fetch failed ({e.code}) for {url}")
    except (urllib.error.URLError, TimeoutError) as e:
        pytest.skip(f"GitHub fetch failed ({e}) for {url}")


def _has_def(src: str, name: str, kind: str = "any") -> bool:
    """Grep-based AST-equivalent; accepts indented matches."""
    if kind in ("any", "class") and re.search(
        rf"^\s*class\s+{re.escape(name)}\b", src, re.MULTILINE
    ):
        return True
    if kind in ("any", "func") and re.search(
        rf"^\s*(?:async\s+)?def\s+{re.escape(name)}\b", src, re.MULTILINE
    ):
        return True
    if kind == "any" and re.search(rf"^\s*{re.escape(name)}\s*[:=]", src, re.MULTILINE):
        return True
    return False


def _first_match(repo: str, ref: str, paths: list[str]) -> tuple[str, str] | None:
    for p in paths:
        src = _fetch_text(repo, ref, p)
        if src is not None:
            return (p, src)
    return None


# Gemma3 attention surface, zoo PR #635 / #488 / #571.
# unsloth_zoo/temporary_patches/gemma.py imports Gemma3Attention,
# Gemma3RMSNorm, Gemma3MLP, Gemma3TextScaledWordEmbedding plus
# apply_rotary_pos_emb / ALL_ATTENTION_FUNCTIONS / eager_attention_forward.


@pytest.mark.parametrize("tag", TRANSFORMERS_TAGS)
def test_gemma3_modeling_required_classes(tag: str):
    """Zoo PR #635 + #488: classes referenced by
    unsloth_zoo/temporary_patches/gemma.py must remain at
    transformers.models.gemma3.modeling_gemma3."""
    src = _fetch_text(
        "huggingface/transformers",
        tag,
        "src/transformers/models/gemma3/modeling_gemma3.py",
    )
    if src is None:
        pytest.skip(f"{tag}: modeling_gemma3.py not present")
    required = (
        "Gemma3Attention",
        "Gemma3RMSNorm",
        "Gemma3MLP",
        "Gemma3TextScaledWordEmbedding",
    )
    missing = [c for c in required if not _has_def(src, c, "class")]
    assert not missing, (
        f"{tag}: classes missing from gemma3 modeling source: {missing}; "
        f"unsloth_zoo/temporary_patches/gemma.py would silently no-op via "
        f"raise_error() and Gemma3 fp16 / SDPA mask fixes would not apply"
    )


@pytest.mark.parametrize("tag", TRANSFORMERS_TAGS)
def test_gemma3_apply_rotary_pos_emb_and_attention_funcs(tag: str):
    """Zoo PR #488 / #571 / #635: gemma.py:399 imports apply_rotary_pos_emb,
    ALL_ATTENTION_FUNCTIONS, eager_attention_forward from the gemma3
    module; must stay reachable on that exact path."""
    src = _fetch_text(
        "huggingface/transformers",
        tag,
        "src/transformers/models/gemma3/modeling_gemma3.py",
    )
    if src is None:
        pytest.skip(f"{tag}: modeling_gemma3.py not present")
    for name in ("apply_rotary_pos_emb", "ALL_ATTENTION_FUNCTIONS", "eager_attention_forward"):
        assert name in src, (
            f"{tag}: `{name}` not reachable from "
            f"transformers.models.gemma3.modeling_gemma3 -- "
            f"unsloth_zoo/temporary_patches/gemma.py:399 import fails"
        )


# ministral / mistral-3 forward signature, zoo PR #571 / #509 / #465.
# ministral.py:35-103 imports apply_rotary_pos_emb / eager_attention_forward
# / ALL_ATTENTION_FUNCTIONS from modeling_ministral, then rebinds .forward.


@pytest.mark.parametrize("tag", TRANSFORMERS_TAGS)
def test_ministral_attention_module_present(tag: str):
    """Zoo PR #571 / #509 / #465: MinistralAttention must remain at
    transformers.models.ministral.modeling_ministral.MinistralAttention."""
    src = _fetch_text(
        "huggingface/transformers",
        tag,
        "src/transformers/models/ministral/modeling_ministral.py",
    )
    if src is None:
        pytest.skip(f"{tag}: modeling_ministral.py not present (legacy/unreleased)")
    assert _has_def(src, "MinistralAttention", "class"), (
        f"{tag}: class MinistralAttention missing; "
        f"unsloth_zoo/temporary_patches/ministral.py:35-103 patch breaks"
    )
    for name in ("apply_rotary_pos_emb", "eager_attention_forward", "ALL_ATTENTION_FUNCTIONS"):
        assert name in src, (
            f"{tag}: `{name}` not reachable from "
            f"transformers.models.ministral.modeling_ministral; "
            f"ministral.py:36-40 import line crashes"
        )


# gpt_oss MoE patch surface, zoo PR #525 / #472 / #471 / #470 / #467.
# gpt_oss.py reads transformers.models.gpt_oss.modeling_gpt_oss.{
# GptOssExperts, GptOssTopKRouter, GptOssAttention, GptOssModel,
# GptOssPreTrainedModel} and reassigns .GptOssExperts.forward.


@pytest.mark.parametrize("tag", TRANSFORMERS_TAGS)
def test_gpt_oss_modeling_classes(tag: str):
    """Zoo PR #525 / #472 / #471: temporary_patches/gpt_oss.py touches all
    five classes by attribute on the module."""
    src = _fetch_text(
        "huggingface/transformers",
        tag,
        "src/transformers/models/gpt_oss/modeling_gpt_oss.py",
    )
    if src is None:
        pytest.skip(f"{tag}: modeling_gpt_oss.py not present (legacy)")
    required = (
        "GptOssExperts",
        "GptOssTopKRouter",
        "GptOssAttention",
        "GptOssModel",
        "GptOssPreTrainedModel",
    )
    missing = [c for c in required if not _has_def(src, c, "class")]
    assert not missing, (
        f"{tag}: gpt_oss modeling missing classes {missing}; "
        f"unsloth_zoo/temporary_patches/gpt_oss.py reassigns these by name "
        f"(.GptOssExperts = ..., .GptOssExperts.forward = ...) — rename "
        f"makes the MoE patches silently no-op"
    )


# qwen3_moe MoE patch surface, zoo PR #601 / #605 / #607 / #574 / #618.
# qwen3_moe.py rebinds Qwen3MoeSparseMoeBlock.forward and
# Qwen3MoeExperts.forward via attribute assignment.


@pytest.mark.parametrize("tag", TRANSFORMERS_TAGS)
def test_qwen3_moe_required_classes(tag: str):
    """Zoo PR #601 / #605 / #607 / #618: Qwen3MoeSparseMoeBlock (all tags)
    and Qwen3MoeExperts (5.x only) must exist on
    transformers.models.qwen3_moe.modeling_qwen3_moe."""
    src = _fetch_text(
        "huggingface/transformers",
        tag,
        "src/transformers/models/qwen3_moe/modeling_qwen3_moe.py",
    )
    if src is None:
        pytest.skip(f"{tag}: modeling_qwen3_moe.py not present")
    # Qwen3MoeSparseMoeBlock: stable across the entire support window
    # (zoo qwen3_moe.py:215 + L337-L340 read it).
    assert _has_def(src, "Qwen3MoeSparseMoeBlock", "class"), (
        f"{tag}: class Qwen3MoeSparseMoeBlock missing; "
        f"unsloth_zoo/temporary_patches/qwen3_moe.py forward / LoRA "
        f"extractor patch becomes a silent no-op"
    )
    # Qwen3MoeExperts: 5.x-only. Zoo qwen3_moe.py:222 gates on this.
    if tag.startswith("v4."):
        return
    assert _has_def(src, "Qwen3MoeExperts", "class"), (
        f"{tag}: class Qwen3MoeExperts missing on transformers 5.x; "
        f"unsloth_zoo/temporary_patches/qwen3_moe.py:326 LoRA-extractor "
        f"registration on .Qwen3MoeExperts is a silent no-op -> Qwen MoE "
        f"grouped-mm LoRA breakage (zoo PR #601 / #605 / #607 / #618)"
    )


# transformers.modeling_utils must expose `checkpoint` and PushToHubMixin.
# Zoo PR #549 patches modeling_utils.checkpoint directly
# (gradient_checkpointing.py:923); unsloth_zoo/saving_utils.py:76 imports
# PushToHubMixin and calls ._upload_modified_files / ._get_files_timestamps.


@pytest.mark.parametrize("tag", TRANSFORMERS_TAGS)
def test_modeling_utils_checkpoint_and_pushtohubmixin(tag: str):
    """Zoo PR #549 + saving_utils.py:76: ``transformers.modeling_utils``
    must bind ``checkpoint`` and reach ``PushToHubMixin``."""
    src = _fetch_text(
        "huggingface/transformers",
        tag,
        "src/transformers/modeling_utils.py",
    )
    if src is None:
        pytest.skip(f"{tag}: modeling_utils.py missing")
    has_checkpoint = bool(
        re.search(r"^from\s+torch\.utils\.checkpoint\s+import\s+checkpoint", src, re.MULTILINE)
        or re.search(r"^import\s+torch\.utils\.checkpoint\b", src, re.MULTILINE)
        or "checkpoint = torch.utils.checkpoint.checkpoint" in src
    )
    assert has_checkpoint, (
        f"{tag}: transformers.modeling_utils.checkpoint not reachable; "
        f"unsloth_zoo/gradient_checkpointing.py:923 reassignment silently "
        f"no-ops and the PR #549 VRAM regression returns on transformers 5.2+"
    )
    # PushToHubMixin: class-defined locally (4.x) OR re-imported from
    # transformers.utils.hub (5.x); either satisfies the zoo import.
    has_pushtohub = bool(
        re.search(r"^\s*class\s+PushToHubMixin\b", src, re.MULTILINE)
        or re.search(r"\bPushToHubMixin\b", src)
    )
    assert has_pushtohub, (
        f"{tag}: PushToHubMixin not reachable from "
        f"transformers.modeling_utils; unsloth_zoo/saving_utils.py:76 ImportError"
    )


# transformers.quantizers.quantizers_utils.should_convert_module, Zoo PR
# #491 / #488 patches this on 5.x; rename silently no-ops the
# vision_tower / audio_tower quantization-skip regression fix.


@pytest.mark.parametrize("tag", TRANSFORMERS_TAGS)
def test_quantizers_should_convert_module_present_on_v5(tag: str):
    """Zoo PR #491 / #488 (patching_utils.py): on transformers 5.x,
    `should_convert_module` must live at
    transformers.quantizers.quantizers_utils. 4.x uses
    `_replace_with_bnb_linear` via a separate zoo path."""
    src = _fetch_text(
        "huggingface/transformers",
        tag,
        "src/transformers/quantizers/quantizers_utils.py",
    )
    if src is None:
        pytest.skip(f"{tag}: quantizers_utils.py not present (legacy 4.x layout)")
    if tag.startswith("v4."):
        pytest.skip(f"{tag}: 4.x line — function not expected here")
    assert _has_def(src, "should_convert_module", "func"), (
        f"{tag}: function should_convert_module missing on transformers 5.x; "
        f"unsloth_zoo/patching_utils.py PR #491 substring-matching patch "
        f"silently no-ops -> vision_tower / audio_tower modules get "
        f"quantized to Linear4bit (PR #488 regression)"
    )


# transformers.integrations.bitsandbytes._replace_with_bnb_linear, 4.x
# companion. unsloth_zoo/patching_utils.py:678 wraps it.


@pytest.mark.parametrize("tag", TRANSFORMERS_TAGS)
def test_integrations_bitsandbytes_legacy_replace_fn(tag: str):
    """patching_utils.py:678 wraps `_replace_with_bnb_linear`; on 4.x must
    be present, on 5.x allowed absent (PR #491 path via
    should_convert_module)."""
    src = _fetch_text(
        "huggingface/transformers",
        tag,
        "src/transformers/integrations/bitsandbytes.py",
    )
    if src is None:
        pytest.skip(f"{tag}: integrations/bitsandbytes.py not present")
    has_legacy = _has_def(src, "_replace_with_bnb_linear", "func")
    has_new = _has_def(src, "replace_with_bnb_linear", "func")
    if tag.startswith("v4."):
        assert has_legacy, (
            f"{tag}: _replace_with_bnb_linear missing on 4.x; "
            f"unsloth_zoo/patching_utils.py:678 hasattr() check fails and "
            f"the substring-match Linear4bit skip-list breaks"
        )
    else:
        assert has_legacy or has_new, (
            f"{tag}: neither _replace_with_bnb_linear (4.x) nor "
            f"replace_with_bnb_linear (5.x) present in "
            f"integrations/bitsandbytes.py"
        )


# transformers.modeling_utils.caching_allocator_warmup, Zoo PR #569:
# hasattr-guarded wrap; fail only on likely rename (not removal).


@pytest.mark.parametrize("tag", TRANSFORMERS_TAGS)
def test_caching_allocator_warmup_reachable(tag: str):
    """Zoo PR #569 wraps modeling_utils.caching_allocator_warmup with a
    <=24 GiB skip-guard via hasattr(). Removal degrades gracefully; only
    a rename (function vanished AND a `*warmup*` sibling appeared)
    silently drops the OOM-before-load guard."""
    src = _fetch_text(
        "huggingface/transformers",
        tag,
        "src/transformers/modeling_utils.py",
    )
    if src is None:
        pytest.skip(f"{tag}: modeling_utils.py missing")
    has_exact = _has_def(src, "caching_allocator_warmup", "func")
    if has_exact:
        return
    # Rename detection: any other `def *_warmup(` in the file.
    other_warmup = re.findall(r"^def\s+(\w*warmup\w*)\s*\(", src, re.MULTILINE)
    other_warmup = [n for n in other_warmup if n != "caching_allocator_warmup"]
    if other_warmup:
        pytest.fail(
            f"{tag}: caching_allocator_warmup missing but found likely rename "
            f"candidates: {other_warmup}. Zoo PR #569 hasattr() guard would "
            f"silently skip the wrap, reintroducing the low-VRAM OOM regression."
        )
    pytest.skip(f"{tag}: caching_allocator_warmup removed (graceful, hasattr-guarded)")


# transformers.masking_utils.{create_causal_mask, create_sliding_window_causal_mask},
# zoo PR #488 / #525; misc.py:382 + gpt_oss.py:2182-2183 rebind these.


@pytest.mark.parametrize("tag", TRANSFORMERS_TAGS)
def test_masking_utils_create_causal_mask_names(tag: str):
    """temporary_patches/misc.py:414 + gpt_oss.py:2182 + ministral.py:144
    read masking_utils.create_causal_mask /
    .create_sliding_window_causal_mask."""
    src = _fetch_text(
        "huggingface/transformers",
        tag,
        "src/transformers/masking_utils.py",
    )
    if src is None:
        pytest.skip(f"{tag}: masking_utils.py not present")
    assert _has_def(src, "create_causal_mask", "func"), (
        f"{tag}: transformers.masking_utils.create_causal_mask missing; "
        f"unsloth_zoo/temporary_patches/misc.py:414 and "
        f"gpt_oss.py:2182 patch breaks"
    )
    assert _has_def(src, "create_sliding_window_causal_mask", "func"), (
        f"{tag}: transformers.masking_utils.create_sliding_window_causal_mask "
        f"missing; misc.py:419 + gpt_oss.py:2183 + ministral.py:144 break"
    )


# peft LoraLayer 3D-parameter (MoE) attribute surface, zoo PR #618.
# Qwen MoE LoRA extractor reads wrapper.get_base_layer() + .parameter_name
# / .hidden_dim / .intermediate_dim. Pin: peft keeps emitting ParamWrapper
# (LoraLayer subclass) in peft/tuners/lora/layer.py.


@pytest.mark.parametrize("tag", PEFT_TAGS)
def test_peft_lora_layer_paramwrapper_present(tag: str):
    """Zoo PR #618: temporary_patches/qwen3_moe.py:46-77 walks
    wrapper.get_base_layer() / wrapper.parameter_name on peft.tuners.lora
    .layer.{LoraLayer,ParamWrapper}; missing -> MoE LoRA extractor
    returns (None, None) and grouped-mm crashes."""
    src = _fetch_text("huggingface/peft", tag, "src/peft/tuners/lora/layer.py")
    if src is None:
        pytest.skip(f"{tag}: peft/tuners/lora/layer.py missing")
    assert _has_def(src, "LoraLayer", "class"), (
        f"{tag}: class LoraLayer missing in peft.tuners.lora.layer; "
        f"unsloth_zoo qwen LoRA extractor and saving_utils.py break"
    )
    # ParamWrapper: peft 0.17 lacked it; 0.18+ has it. Either name accepted.
    has_param_wrapper = bool(
        re.search(r"^\s*class\s+ParamWrapper\b", src, re.MULTILINE)
        or "ParamWrapper" in src
    )
    has_get_base = "get_base_layer" in src
    assert has_get_base, (
        f"{tag}: peft.tuners.lora.layer.LoraLayer no longer exposes "
        f"`get_base_layer`; unsloth qwen MoE extractor returns (None, None) "
        f"and grouped-mm crashes (PR #618 silently disabled)."
    )
    if tag in ("v0.18.0", "v0.19.1", "main") and not has_param_wrapper:
        pytest.fail(
            f"{tag}: peft 0.18+ should expose ParamWrapper in lora/layer.py "
            f"(zoo PR #618 dispatches on 3D weight shape semantics this "
            f"class introduced); class is missing on this tag."
        )
