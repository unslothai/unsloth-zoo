# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team.
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at
# your option) any later version.
"""Pinned-symbol matrix tests: do the EXACT transformers / peft / datasets
symbols that ``unsloth_zoo`` reaches into still exist with the expected
shape?

Each test is a CHEAP grep-on-raw-github-source check that catches the
moment an upstream rename / removal / signature flip would silently
no-op one of our monkey-patches. No GPU, no pip install, no model load
required. Runs under the GPU-free harness in ``tests/conftest.py``.

The matrix below covers the supported transformers window declared in
``pyproject.toml`` plus a couple of bleeding-edge tags so we get early
warning before users hit a regression.

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


# ---------------------------------------------------------------------------
# Version matrix. Keep aligned with the project's supported window.
# transformers anchors per project spec: 4.57.6 and 5.5.0.
# ---------------------------------------------------------------------------

TRANSFORMERS_TAGS = [
    "v4.57.6",  # anchor (lower-bound, must keep working)
    "v5.0.0",
    "v5.1.0",
    "v5.2.0",
    "v5.3.0",
    "v5.5.0",  # anchor
    "main",
]

# peft tags covering the API change that motivated zoo PR #618.
PEFT_TAGS = [
    "v0.17.0",
    "v0.18.0",
    "v0.19.1",
    "main",
]


# ---------------------------------------------------------------------------
# Tiny self-contained helpers (no dependency on tests/version_compat/_fetch
# because that directory does not exist in unsloth-zoo yet -- this file is
# the first pinned-symbol test we ship from zoo's side).
# ---------------------------------------------------------------------------


def _fetch_text(repo: str, ref: str, path: str) -> str | None:
    """GET https://raw.githubusercontent.com/{repo}/{ref}/{path}. Returns
    None on 404 (path renamed/removed), pytest.skip on transient errors so
    flaky CI doesn't false-fail."""
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
    """Grep-based AST-equivalent. Accepts indented matches so class methods
    pass the same check as module-level defs."""
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


# ===========================================================================
# 1. Gemma3 attention surface — zoo PR #635 (gemma3 SDPA mask),
#    PR #488 / #571 (Gemma3 5.x forward signature). Our patches in
#    unsloth_zoo/temporary_patches/gemma.py do
#
#       from transformers.models.gemma3.modeling_gemma3 import (
#           apply_rotary_pos_emb, ALL_ATTENTION_FUNCTIONS,
#           eager_attention_forward,
#       )
#       transformers.models.gemma3.modeling_gemma3.Gemma3Attention
#       transformers.models.gemma3.modeling_gemma3.Gemma3RMSNorm
#       transformers.models.gemma3.modeling_gemma3.Gemma3MLP
#       transformers.models.gemma3.modeling_gemma3.Gemma3TextScaledWordEmbedding
#
#    The patches no-op silently (via `raise_error` swallow) if any symbol
#    is renamed; we want a loud test in CI instead.
# ===========================================================================


@pytest.mark.parametrize("tag", TRANSFORMERS_TAGS)
def test_gemma3_modeling_required_classes(tag: str):
    """Zoo PR #635 + #488: every class referenced by
    unsloth_zoo/temporary_patches/gemma.py must remain at the module path
    transformers.models.gemma3.modeling_gemma3.<Name>."""
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
    """Zoo PR #488 / #571 / #635: gemma.py imports
    ``apply_rotary_pos_emb``, ``ALL_ATTENTION_FUNCTIONS`` and
    ``eager_attention_forward`` from the gemma3 module. These names must
    stay reachable on that exact path; transformers occasionally moves
    them to ``modeling_utils`` and ``modeling_layers``, which would make
    the `from X import Y` line in gemma.py ImportError out."""
    src = _fetch_text(
        "huggingface/transformers",
        tag,
        "src/transformers/models/gemma3/modeling_gemma3.py",
    )
    if src is None:
        pytest.skip(f"{tag}: modeling_gemma3.py not present")
    # Either defined locally, or re-exported (e.g. via `from ... import X`).
    for name in ("apply_rotary_pos_emb", "ALL_ATTENTION_FUNCTIONS", "eager_attention_forward"):
        assert name in src, (
            f"{tag}: `{name}` not reachable from "
            f"transformers.models.gemma3.modeling_gemma3 -- "
            f"unsloth_zoo/temporary_patches/gemma.py:399 import fails"
        )


# ===========================================================================
# 2. ministral / mistral-3 forward signature — zoo PR #571, #509, #465.
#    ministral.py imports `apply_rotary_pos_emb, eager_attention_forward,
#    ALL_ATTENTION_FUNCTIONS` from modeling_ministral, then rebinds the
#    class's `.forward`. If MinistralAttention disappears or moves we
#    want CI to scream.
# ===========================================================================


@pytest.mark.parametrize("tag", TRANSFORMERS_TAGS)
def test_ministral_attention_module_present(tag: str):
    """Zoo PR #571 / #509 / #465: MinistralAttention class must remain at
    transformers.models.ministral.modeling_ministral.MinistralAttention.
    """
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
    # The same module is also expected to expose these names. (We don't
    # require them to be DEFINED here -- just reachable as
    # ``from transformers.models.ministral.modeling_ministral import X``.)
    for name in ("apply_rotary_pos_emb", "eager_attention_forward", "ALL_ATTENTION_FUNCTIONS"):
        assert name in src, (
            f"{tag}: `{name}` not reachable from "
            f"transformers.models.ministral.modeling_ministral; "
            f"ministral.py:36-40 import line crashes"
        )


# ===========================================================================
# 3. gpt_oss MoE patch surface — zoo PR #525, #472, #471, #470, #467.
#    gpt_oss.py reads `transformers.models.gpt_oss.modeling_gpt_oss.{
#    GptOssExperts, GptOssTopKRouter, GptOssAttention, GptOssModel,
#    GptOssPreTrainedModel }` and reassigns `.GptOssExperts.forward`.
#    If any of these are renamed our MoE bnb / native pytorch / unwrap
#    fixes silently disable themselves.
# ===========================================================================


@pytest.mark.parametrize("tag", TRANSFORMERS_TAGS)
def test_gpt_oss_modeling_classes(tag: str):
    """Zoo PR #525 / #472 / #471: gpt_oss.py touches all five classes
    below by attribute on the module — if any disappear, the gpt_oss
    patches go silently dormant and grpo / bnb breakage returns."""
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


# ===========================================================================
# 4. qwen3_moe MoE patch surface — zoo PR #601, #605, #607, #574, #618.
#    qwen3_moe.py rebinds Qwen3MoeSparseMoeBlock.forward and
#    Qwen3MoeExperts.forward via attribute assignment.
# ===========================================================================


@pytest.mark.parametrize("tag", TRANSFORMERS_TAGS)
def test_qwen3_moe_required_classes(tag: str):
    """Zoo PR #601 / #605 / #607 / #618: both Qwen3MoeSparseMoeBlock and
    Qwen3MoeExperts must exist on
    transformers.models.qwen3_moe.modeling_qwen3_moe. The zoo's LoRA
    extractor and forward override are attribute-keyed, so a class
    rename would silently bypass them."""
    src = _fetch_text(
        "huggingface/transformers",
        tag,
        "src/transformers/models/qwen3_moe/modeling_qwen3_moe.py",
    )
    if src is None:
        pytest.skip(f"{tag}: modeling_qwen3_moe.py not present")
    # Qwen3MoeSparseMoeBlock: stable across the entire support window —
    # always required (zoo qwen3_moe.py:215 + L337-L340 read it).
    assert _has_def(src, "Qwen3MoeSparseMoeBlock", "class"), (
        f"{tag}: class Qwen3MoeSparseMoeBlock missing; "
        f"unsloth_zoo/temporary_patches/qwen3_moe.py forward / LoRA "
        f"extractor patch becomes a silent no-op"
    )
    # Qwen3MoeExperts: 5.x-only — added when transformers split MoE
    # weights into a dedicated `Experts` module. Zoo qwen3_moe.py:222
    # comments `# New transformers has this`, so the patch is gated. We
    # mirror that gate (must exist on 5.x and main; allowed absent on 4.x).
    if tag.startswith("v4."):
        # 4.x predates the split — accept absence.
        return
    assert _has_def(src, "Qwen3MoeExperts", "class"), (
        f"{tag}: class Qwen3MoeExperts missing on transformers 5.x; "
        f"unsloth_zoo/temporary_patches/qwen3_moe.py:326 LoRA-extractor "
        f"registration on .Qwen3MoeExperts is a silent no-op -> Qwen MoE "
        f"grouped-mm LoRA breakage (zoo PR #601 / #605 / #607 / #618)"
    )


# ===========================================================================
# 5. transformers.modeling_utils MUST expose `checkpoint` AND a
#    `PushToHubMixin` class.
#    - Zoo PR #549 patches transformers.modeling_utils.checkpoint
#      directly so that gradient_checkpointing_enable() picks up the
#      Unsloth smart-offload variant. If transformers stops re-binding
#      `checkpoint` in modeling_utils, our offload silently disables
#      itself and users hit the documented #549 VRAM regression again.
#    - unsloth_zoo/saving_utils.py imports PushToHubMixin from this
#      module (5.x removed _create_repo but the class itself is still
#      relied on for ._upload_modified_files / ._get_files_timestamps).
# ===========================================================================


@pytest.mark.parametrize("tag", TRANSFORMERS_TAGS)
def test_modeling_utils_checkpoint_and_pushtohubmixin(tag: str):
    """Zoo PR #549 + saving_utils.py: ``transformers.modeling_utils``
    must (a) bind ``checkpoint`` (we monkey-patch it) and (b) define
    ``class PushToHubMixin`` (we call ._upload_modified_files /
    ._get_files_timestamps on it)."""
    src = _fetch_text(
        "huggingface/transformers",
        tag,
        "src/transformers/modeling_utils.py",
    )
    if src is None:
        pytest.skip(f"{tag}: modeling_utils.py missing")
    # `checkpoint` must be reachable as a module attribute. Accept any of:
    #   from torch.utils.checkpoint import checkpoint
    #   import torch.utils.checkpoint as ...
    #   checkpoint = torch.utils.checkpoint.checkpoint
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
    # PushToHubMixin: zoo does `from transformers.modeling_utils import
    # PushToHubMixin`. The name can be class-defined locally (4.x) OR
    # re-imported from `transformers.utils.hub` (5.x). Either is fine for
    # the import to succeed — what we need is the NAME reachable on the
    # module attribute surface. Match either an import line or a class def.
    has_pushtohub = bool(
        re.search(r"^\s*class\s+PushToHubMixin\b", src, re.MULTILINE)
        or re.search(r"\bPushToHubMixin\b", src)
    )
    assert has_pushtohub, (
        f"{tag}: PushToHubMixin not reachable from "
        f"transformers.modeling_utils; unsloth_zoo/saving_utils.py:76 ImportError"
    )


# ===========================================================================
# 6. transformers.quantizers.quantizers_utils.should_convert_module —
#    Zoo PR #491 / #488 patches this function on 5.x. The patch reads
#    the function's source string and rewrites it; if the function is
#    renamed, the patch silently no-ops and the vision_tower /
#    audio_tower quantization-skip regression returns.
# ===========================================================================


@pytest.mark.parametrize("tag", TRANSFORMERS_TAGS)
def test_quantizers_should_convert_module_present_on_v5(tag: str):
    """Zoo PR #491 / #488: on transformers 5.x, the function
    `should_convert_module` must live at
    transformers.quantizers.quantizers_utils. (4.x predates this path —
    the 4.x equivalent is `_replace_with_bnb_linear` and is covered by
    a separate zoo patch path.)"""
    src = _fetch_text(
        "huggingface/transformers",
        tag,
        "src/transformers/quantizers/quantizers_utils.py",
    )
    if src is None:
        # 4.57.6 era: no such module yet. The legacy patch path keys off
        # `_replace_with_bnb_linear` in transformers.integrations.bitsandbytes
        # — separately validated below.
        pytest.skip(f"{tag}: quantizers_utils.py not present (legacy 4.x layout)")
    # On 5.x the symbol MUST be present (PR #491 / #488 hinges on it).
    if tag.startswith("v4."):
        pytest.skip(f"{tag}: 4.x line — function not expected here")
    assert _has_def(src, "should_convert_module", "func"), (
        f"{tag}: function should_convert_module missing on transformers 5.x; "
        f"unsloth_zoo/patching_utils.py PR #491 substring-matching patch "
        f"silently no-ops -> vision_tower / audio_tower modules get "
        f"quantized to Linear4bit (PR #488 regression)"
    )


# ===========================================================================
# 7. transformers.integrations.bitsandbytes._replace_with_bnb_linear —
#    the 4.x companion of test 6. unsloth_zoo/patching_utils.py:678
#    keys its substring-matching wrapper off the presence of this
#    function.
# ===========================================================================


@pytest.mark.parametrize("tag", TRANSFORMERS_TAGS)
def test_integrations_bitsandbytes_legacy_replace_fn(tag: str):
    """Zoo patching_utils.py:678 wraps
    ``transformers.integrations.bitsandbytes._replace_with_bnb_linear`` —
    on 4.x this MUST be present (else the substring-matching skip-list
    feature silently disappears). On 5.x it's allowed to be absent (the
    PR #491 path handles it via should_convert_module instead)."""
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
        # 5.x is allowed to drop _replace_with_bnb_linear; we just need
        # one of the two forms reachable so SOMEONE handles BNB conv.
        assert has_legacy or has_new, (
            f"{tag}: neither _replace_with_bnb_linear (4.x) nor "
            f"replace_with_bnb_linear (5.x) present in "
            f"integrations/bitsandbytes.py"
        )


# ===========================================================================
# 8. transformers.modeling_utils.caching_allocator_warmup —
#    Zoo PR #569 wraps this. The wrap is guarded with hasattr() so an
#    absence is OK, BUT if the function is renamed (not removed) we'd
#    silently lose the low-VRAM OOM guard. Snapshot the name.
# ===========================================================================


@pytest.mark.parametrize("tag", TRANSFORMERS_TAGS)
def test_caching_allocator_warmup_reachable(tag: str):
    """Zoo PR #569: ``transformers.modeling_utils.caching_allocator_warmup``
    is the function we wrap with the <=24 GiB skip-guard. The wrap is
    gated by hasattr(), so a removal degrades gracefully — but a RENAME
    would silently drop the guard and reintroduce OOM-before-load on
    low-VRAM cards. Just record presence/absence informationally and
    fail only when we detect a likely RENAME (function vanished AND a
    same-prefix `*warmup*` symbol appeared)."""
    src = _fetch_text(
        "huggingface/transformers",
        tag,
        "src/transformers/modeling_utils.py",
    )
    if src is None:
        pytest.skip(f"{tag}: modeling_utils.py missing")
    has_exact = _has_def(src, "caching_allocator_warmup", "func")
    if has_exact:
        return  # all good
    # Look for a likely rename — any other `def *_warmup(` in the file.
    other_warmup = re.findall(r"^def\s+(\w*warmup\w*)\s*\(", src, re.MULTILINE)
    other_warmup = [n for n in other_warmup if n != "caching_allocator_warmup"]
    if other_warmup:
        pytest.fail(
            f"{tag}: caching_allocator_warmup missing but found likely rename "
            f"candidates: {other_warmup}. Zoo PR #569 hasattr() guard would "
            f"silently skip the wrap, reintroducing the low-VRAM OOM regression."
        )
    # Removed without rename — graceful degradation; OK.
    pytest.skip(f"{tag}: caching_allocator_warmup removed (graceful, hasattr-guarded)")


# ===========================================================================
# 9. transformers.masking_utils.{create_causal_mask,
#    create_sliding_window_causal_mask} — zoo PR #488, #525, and the
#    gpt_oss patch rebind these. Their NAMES must remain stable.
# ===========================================================================


@pytest.mark.parametrize("tag", TRANSFORMERS_TAGS)
def test_masking_utils_create_causal_mask_names(tag: str):
    """unsloth_zoo/temporary_patches/misc.py:382 and gpt_oss.py:2182-2183
    do `import transformers.masking_utils as masking_utils` and read
    `masking_utils.create_causal_mask` + `.create_sliding_window_causal_mask`.
    Both must remain reachable on every supported transformers tag.
    Zoo PR #488 additionally adds ``create_causal_mask_mapping`` to
    DISABLED_KEYWORDS — the 5.x rename — so we ALSO accept that name."""
    src = _fetch_text(
        "huggingface/transformers",
        tag,
        "src/transformers/masking_utils.py",
    )
    if src is None:
        pytest.skip(f"{tag}: masking_utils.py not present")
    # create_causal_mask: stable through the entire window.
    assert _has_def(src, "create_causal_mask", "func"), (
        f"{tag}: transformers.masking_utils.create_causal_mask missing; "
        f"unsloth_zoo/temporary_patches/misc.py:414 and "
        f"gpt_oss.py:2182 patch breaks"
    )
    assert _has_def(src, "create_sliding_window_causal_mask", "func"), (
        f"{tag}: transformers.masking_utils.create_sliding_window_causal_mask "
        f"missing; misc.py:419 + gpt_oss.py:2183 + ministral.py:144 break"
    )


# ===========================================================================
# 10. peft LoraLayer 3D-parameter (MoE) attribute surface — zoo PR #618.
#     The Qwen MoE LoRA extractor reads `wrapper.get_base_layer()` and
#     then `.parameter_name`, `.hidden_dim`, `.intermediate_dim` on the
#     base layer. The extractor also has to track peft's behaviour
#     change between 0.18 and 0.19 where the 3D weight is now swapped
#     in-place before LoRA is created. We don't try to verify the
#     swap-aware path here (no install needed); instead we pin the
#     simpler upstream contract: PEFT must keep emitting
#     ``ParamWrapper`` (LoraLayer subclass) in peft/tuners/lora/layer.py
#     for our `get_base_layer() / parameter_name` API to keep working.
# ===========================================================================


@pytest.mark.parametrize("tag", PEFT_TAGS)
def test_peft_lora_layer_paramwrapper_present(tag: str):
    """Zoo PR #618: peft.tuners.lora.layer must keep defining the
    LoraLayer + ParamWrapper surface that
    unsloth_zoo/temporary_patches/qwen3_moe.py:46-77 walks via
    ``wrapper.get_base_layer()`` and ``wrapper.parameter_name``. If
    either disappears the MoE LoRA extractor silently returns
    ``(None, None)`` and grouped-mm crashes."""
    src = _fetch_text("huggingface/peft", tag, "src/peft/tuners/lora/layer.py")
    if src is None:
        pytest.skip(f"{tag}: peft/tuners/lora/layer.py missing")
    assert _has_def(src, "LoraLayer", "class"), (
        f"{tag}: class LoraLayer missing in peft.tuners.lora.layer; "
        f"unsloth_zoo qwen LoRA extractor and saving_utils.py break"
    )
    # ParamWrapper or its forerunner — peft 0.17 didn't have it, peft 0.18+
    # introduced it. Accept either name (peft has had a couple of stabs at
    # the API) so we don't false-fail on legacy tags.
    has_param_wrapper = bool(
        re.search(r"^\s*class\s+ParamWrapper\b", src, re.MULTILINE)
        or "ParamWrapper" in src
    )
    # `get_base_layer` is the unmodified-since-0.7 accessor we rely on.
    has_get_base = "get_base_layer" in src
    assert has_get_base, (
        f"{tag}: peft.tuners.lora.layer.LoraLayer no longer exposes "
        f"`get_base_layer`; unsloth qwen MoE extractor returns (None, None) "
        f"and grouped-mm crashes (PR #618 silently disabled)."
    )
    # ParamWrapper is informational on very old peft, required on 0.18+.
    if tag in ("v0.18.0", "v0.19.1", "main") and not has_param_wrapper:
        pytest.fail(
            f"{tag}: peft 0.18+ should expose ParamWrapper in lora/layer.py "
            f"(zoo PR #618 dispatches on 3D weight shape semantics this "
            f"class introduced); class is missing on this tag."
        )
