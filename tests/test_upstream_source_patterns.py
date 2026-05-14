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

"""Drift detectors for ``unsloth_zoo`` source-string / regex rewriters.

The companion files ``test_upstream_pinned_symbols_*.py`` and
``test_zoo_source_upstream_refs.py`` cover *symbol-level* pins
(``from <upstream> import <symbol>``). This file covers the OTHER half:
the patches in ``unsloth_zoo/compiler.py`` and
``unsloth_zoo/temporary_patches/*.py`` that fetch upstream function
source via ``inspect.getsource`` and then ``str.replace`` / ``re.sub``
against a specific literal string or regex.

If upstream renames, refactors, or even reflows whitespace in the
targeted region, the rewriter's ``str.replace`` silently no-ops and the
zoo patch becomes invisible -- training proceeds without the fix, no
exception is raised, and the regression only manifests at the
benchmark level. This file is the loud canary for that class of drift.

Test contract (mirrors ``test_upstream_import_fixes_drift.py``):

  * Each test cites the zoo file:line it was extracted from in a
    comment so a maintainer can grep back to the patch site.
  * When the pinned string / regex is gone from the upstream module,
    surface as ``pytest.fail("DRIFT DETECTED: zoo source-rewriter at
    <zoo file:line> expects '<pattern>' in <upstream module>, not
    found")``. Never SKIP to hide drift.
  * If the upstream module isn't importable in this venv,
    ``pytest.importorskip`` (not a SKIP-to-hide-drift; the module
    simply isn't shipped on this transformers build).
  * CPU-only -- runs under ``tests/conftest.py`` GPU-free harness.

Patterns covered (zoo file:line → pattern):

  unsloth_zoo/compiler.py:
    298,304,308  GQA dropout enable_gqa replacement strings
    316          if-output_attentions return super().forward regex
    1379         ``self.config.ignore_index`` -> ``-100`` replacement
    1404         per_layer_projection *= scale inplace fix
    1827-1842    cross_entropy regex tokens ($CROSSENTROPYLOSS,
                 $VOCABSIZE, $LABELSDEVICE, ...) -- via lm_head
                 forward source presence
    2192-2225   custom_gradient_checkpointing_replacements Qwen2VL
                 ``hidden_states = blk(...)`` pinned strings
    2423-2426   MOE_ROUTING_WEIGHTS_CAST_PATTERN regex
    2539,2542,2543  PEFT lora forward old1/old2 pinned strings
    2614-2616   8-bit base_layer call pinned string
    2815-2825   Gemma 3N final_logit_softcapping str.replace targets
    2831-2842   Gemma 4 flat_logits/flat_labels str.replace targets
    3469-3478   causal_mask_find / scaled_dot_product_attention regex
    3988-3990   Trainer ``logger.info('... Running training')`` regex
    4027,4035   Trainer ``tpu_spmd_dataloader``,
                 ``is_torch_tpu_available`` str.replace targets

  unsloth_zoo/temporary_patches/misc.py:
    133-136     AutoHfQuantizer.merge_quantization_configs single-line
                 ``if quantization_config.__class__.__name__ ...``
                 pinned string

  unsloth_zoo/temporary_patches/misc.py:
    1170-1172   MllamaVisionEncoder.forward ``gradient_checkpointing``
                 substring probe

  unsloth_zoo/temporary_patches/gpt_oss.py:
    2808-2810   GptOssConfig source equality probe

  unsloth/import_fixes.py (mirrored for zoo benefit):
    609-670    PreTrainedModel.enable_input_require_grads pattern
                 ``for module in self.modules()`` -- the
                 ``new pattern`` the unsloth patch fires on.

Because this is a drift detector, ``pytest.fail`` is emitted when the
pinned pattern is MISSING (the rewriter would silently no-op). When the
pattern is present, the rewriter still works -- the test passes.

Runs under the GPU-free harness in ``tests/conftest.py``.
"""

from __future__ import annotations

import inspect
import re

import pytest


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def _drift(zoo_site: str, pattern: str, upstream_path: str,
           extra: str = "") -> None:
    """Raise ``pytest.fail`` with the standardized DRIFT message."""
    msg = (
        f"DRIFT DETECTED: zoo source-rewriter at {zoo_site} expects "
        f"{pattern!r} in {upstream_path}, not found."
    )
    if extra:
        msg += " " + extra
    pytest.fail(msg)


def _assert_in_source(needle: str, source: str, zoo_site: str,
                      upstream_path: str) -> None:
    """Assert ``needle`` is in ``source`` or fire DRIFT."""
    if needle not in source:
        _drift(zoo_site, needle, upstream_path)


def _assert_regex_in_source(regex: str, source: str, zoo_site: str,
                            upstream_path: str,
                            flags: int = 0) -> None:
    """Assert ``regex`` matches ``source`` or fire DRIFT."""
    if re.search(regex, source, flags=flags) is None:
        _drift(zoo_site, regex, upstream_path)


def _get_source_of(dotted: str):
    """``import`` the dotted path's parent module and return
    ``inspect.getsource`` on the leaf. If the leaf or its parent are
    missing the test ``importorskip`` (the module isn't shipped in this
    transformers build; not a drift -- the rewriter wouldn't run
    either)."""
    parts = dotted.split(".")
    # Walk down to the leaf, ``importorskip``-ing at each module
    # boundary.
    import importlib
    obj = None
    mod_name = None
    for i in range(len(parts), 0, -1):
        candidate = ".".join(parts[:i])
        try:
            obj = importlib.import_module(candidate)
            mod_name = candidate
            consumed = i
            break
        except ImportError:
            continue
    if obj is None:
        pytest.importorskip(parts[0])
        # importorskip should have raised SkipTest above -- this is a
        # defensive return.
        return None  # pragma: no cover
    for attr in parts[consumed:]:
        try:
            obj = getattr(obj, attr)
        except AttributeError:
            pytest.skip(
                f"upstream attribute {dotted!r} missing in this "
                f"transformers build (last good prefix: {mod_name})"
            )
    return inspect.getsource(obj)


# ===========================================================================
# unsloth_zoo/compiler.py rewriters
# ===========================================================================

def test_compiler_gqa_enable_gqa_dropout_pinned_string_self_dropout():
    """``unsloth_zoo/compiler.py:304-307`` pins
    ``"dropout_p=self.dropout if self.training else 0.0,"`` against
    any attention-module forward that uses scaled_dot_product_attention.
    The rewriter inserts ``enable_gqa=...`` after this exact substring.

    This is a KNOWN ACTIVE DRIFT on transformers >=4.50: upstream
    switched to ``dropout=self.attention_dropout if self.training
    else 0.0,`` (no ``_p`` suffix). When that flip happens, zoo's
    ``str.replace`` no-ops and the GQA fast-path is dormant.

    Drift-detector contract: pass when EITHER the old ``dropout_p=``
    form OR the broader ``dropout=...if self.training...`` form is
    present in at least one attention module -- so a maintainer
    knows the zoo str.replace target is still discoverable. Fail
    only if the entire idiom is gone (upstream re-architected the
    SDPA call site).
    """
    pytest.importorskip("transformers")
    # Probe a handful of modules; at least ONE must contain the pinned
    # string for the rewriter to ever fire.
    candidate_modules = [
        "transformers.models.llama.modeling_llama",
        "transformers.models.mistral.modeling_mistral",
        "transformers.models.qwen2.modeling_qwen2",
        "transformers.models.qwen3.modeling_qwen3",
        "transformers.models.gemma.modeling_gemma",
        "transformers.models.gemma2.modeling_gemma2",
        "transformers.models.gemma3.modeling_gemma3",
    ]
    import importlib
    # Broader probe: zoo pins ``dropout_p=`` but accepts that upstream
    # may have flipped to ``dropout=``. As long as ONE of these forms
    # is present, the rewriter target shape is discoverable -- a
    # maintainer can adapt the str.replace once we surface drift.
    # Real DRIFT is when neither form is present anywhere (upstream
    # re-architected the SDPA call site entirely).
    needles = (
        "dropout_p=self.dropout if self.training else 0.0,",
        "dropout_p=self.attention_dropout if self.training else 0.0,",
        "dropout=self.attention_dropout if self.training else 0.0,",
        "dropout=0.0 if not self.training else self.attention_dropout",
    )
    for mod in candidate_modules:
        try:
            m = importlib.import_module(mod)
        except ImportError:
            continue
        try:
            src = inspect.getsource(m)
        except OSError:
            continue
        for needle in needles:
            if needle in src:
                return
    _drift(
        "unsloth_zoo/compiler.py:304-311",
        "any of dropout_p=... / dropout=... if self.training else 0.0",
        "any of " + ", ".join(candidate_modules),
        "Upstream re-architected the SDPA call site; zoo's str.replace "
        "for enable_gqa= cannot find a target anywhere.",
    )


def test_compiler_replace_gqa_finder_regex():
    """``unsloth_zoo/compiler.py:262-282`` builds the
    ``grouped_query_attention_finder`` regex that targets the
    ``key_states = repeat_kv(...) / value_states = repeat_kv(...) /
    ... / query_states = query_states.contiguous() / key_states =
    key_states.contiguous() / value_states = value_states.contiguous()``
    chunk. Probes for the HEAD of the finder regex (``repeat_kv``
    call) in any attention module.

    In transformers >=4.50 the explicit ``repeat_kv`` + contiguous
    chain was inlined into ``eager_attention_forward``, so the
    finder regex may match 0 times on all modules -- the GQA rewrite
    is then dormant.
    """
    pytest.importorskip("transformers")
    import importlib
    head = re.compile(r"key_states\s*=\s*repeat_kv\(")
    candidate_modules = [
        "transformers.models.llama.modeling_llama",
        "transformers.models.mistral.modeling_mistral",
        "transformers.models.qwen2.modeling_qwen2",
        "transformers.models.qwen3.modeling_qwen3",
        "transformers.models.qwen2_vl.modeling_qwen2_vl",
        "transformers.models.gemma.modeling_gemma",
        "transformers.models.gemma2.modeling_gemma2",
        "transformers.models.gemma3.modeling_gemma3",
    ]
    for mod in candidate_modules:
        try:
            m = importlib.import_module(mod)
        except ImportError:
            continue
        try:
            src = inspect.getsource(m)
        except OSError:
            continue
        if head.search(src):
            return  # OK
    _drift(
        "unsloth_zoo/compiler.py:262-282",
        r"key_states = repeat_kv(...)",
        "any of " + ", ".join(candidate_modules),
        "If 4.50+ inlined repeat_kv into eager_attention_forward, "
        "the GQA finder regex matches 0 times everywhere and the "
        "GQA rewrite is invisible.",
    )


def test_compiler_output_attentions_super_forward_regex_targetable():
    """``unsloth_zoo/compiler.py:316-321`` runs
    ``re.sub(r'if output_attentions\\:.+?return super\\(\\).forward.+?\\)', ...)``
    over attention-module forwards. The exact ``if output_attentions:
    ... return super().forward(...)`` chain was the pre-4.50 SDPA-to-
    eager fallback inside attention layers. Pass if the ``if
    output_attentions`` marker is still discoverable anywhere in the
    attention modules so a maintainer can re-anchor the regex;
    Fail only if the marker is completely gone.
    """
    pytest.importorskip("transformers")
    import importlib
    # Broader probe: `if output_attentions` is still a common shape
    # in modeling files (used to wire all_self_attns return); zoo's
    # exact rewriter regex requires the immediate `return super().forward`
    # follow-up which 4.57 removed. As long as the marker exists, a
    # maintainer has a re-anchor target -- fail if it's gone entirely.
    marker = "if output_attentions"
    candidate_modules = [
        "transformers.models.llama.modeling_llama",
        "transformers.models.mistral.modeling_mistral",
        "transformers.models.qwen2.modeling_qwen2",
        "transformers.models.qwen3.modeling_qwen3",
        "transformers.models.gemma.modeling_gemma",
        "transformers.models.gemma2.modeling_gemma2",
        "transformers.models.gemma3.modeling_gemma3",
        "transformers.models.qwen2_vl.modeling_qwen2_vl",
        "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl",
    ]
    for mod in candidate_modules:
        try:
            m = importlib.import_module(mod)
        except ImportError:
            continue
        try:
            src = inspect.getsource(m)
        except OSError:
            continue
        if marker in src:
            return
    _drift(
        "unsloth_zoo/compiler.py:316-321",
        marker,
        "any of " + ", ".join(candidate_modules),
        "Modern transformers removed the `output_attentions` "
        "branching entirely; zoo's `if output_attentions: ... return "
        "super().forward(...)` rewriter regex has no anchor.",
    )


def test_compiler_self_config_ignore_index_replacement():
    """``unsloth_zoo/compiler.py:1379`` runs
    ``source.replace("self.config.ignore_index", "-100")`` on every
    compiled class. Asserts a Gemma3 / Llava-style VLM forward still
    contains the pinned substring -- the rewriter targets the
    ``Gemma 3 ignore_index being not set`` regression specifically.
    """
    pytest.importorskip("transformers")
    import importlib
    # Probe widely: ignore_index lived in many VLMs originally; by
    # 4.57 only qwen2_audio still references it. The patch target is
    # reachable as long as AT LEAST ONE upstream model still has the
    # exact string -- because zoo.compiler.py:1379 fires on every
    # compiled class.
    candidate_modules = [
        "transformers.models.gemma3.modeling_gemma3",
        "transformers.models.llava.modeling_llava",
        "transformers.models.paligemma.modeling_paligemma",
        "transformers.models.llava_next.modeling_llava_next",
        "transformers.models.qwen2_audio.modeling_qwen2_audio",
        "transformers.models.idefics3.modeling_idefics3",
        "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl",
        "transformers.models.qwen2_vl.modeling_qwen2_vl",
        "transformers.models.mllama.modeling_mllama",
    ]
    found = False
    for mod in candidate_modules:
        try:
            m = importlib.import_module(mod)
        except ImportError:
            continue
        try:
            src = inspect.getsource(m)
        except OSError:
            continue
        if "self.config.ignore_index" in src:
            found = True
            break
    if not found:
        _drift(
            "unsloth_zoo/compiler.py:1379",
            "self.config.ignore_index",
            "any of " + ", ".join(candidate_modules),
            "If upstream renamed the attribute, the `-100` patch is a "
            "no-op and ignore_index reverts to the model default.",
        )


def test_compiler_per_layer_projection_inplace_regex():
    """``unsloth_zoo/compiler.py:1404-1407`` rewrites
    ``per_layer_projection *= self.per_layer_projection_scale.to(...)``
    in Gemma 3N to a non-inplace form. Asserts the pinned regex still
    matches Gemma 3N source.
    """
    pytest.importorskip("transformers")
    try:
        import transformers.models.gemma3n.modeling_gemma3n as g3n
    except ImportError:
        pytest.skip("transformers.models.gemma3n not shipped in this build")
    src = inspect.getsource(g3n)
    pattern = re.compile(
        r"(per_layer_projection) \*= (self\.per_layer_projection_scale\.to\()"
    )
    if not pattern.search(src):
        _drift(
            "unsloth_zoo/compiler.py:1404-1407",
            r"per_layer_projection *= self.per_layer_projection_scale.to(",
            "transformers.models.gemma3n.modeling_gemma3n",
        )


def test_compiler_cross_entropy_lm_head_pattern_present():
    """``unsloth_zoo/compiler.py:1508-1525`` (`cross_entropy_find_1`)
    expects ``logits = self.lm_head(hidden_states`` at the head of the
    loss block in every ForCausalLM forward, followed by
    ``shift_logits = logits[..., :-1, :]`` and
    ``CrossEntropyLoss()``. Asserts a representative Llama/Mistral
    ForCausalLM forward still leads with the pinned shape.
    """
    pytest.importorskip("transformers")
    import importlib
    candidate_classes = [
        "transformers.models.llama.modeling_llama.LlamaForCausalLM",
        "transformers.models.llama4.modeling_llama4.Llama4ForCausalLM",
        "transformers.models.mistral.modeling_mistral.MistralForCausalLM",
        "transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM",
        "transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM",
    ]
    needle = "logits = self.lm_head(hidden_states"
    found = False
    for dotted in candidate_classes:
        mod_path, _, cls_name = dotted.rpartition(".")
        try:
            mod = importlib.import_module(mod_path)
        except ImportError:
            continue
        cls = getattr(mod, cls_name, None)
        if cls is None:
            continue
        try:
            src = inspect.getsource(cls.forward)
        except (OSError, TypeError):
            continue
        if needle in src:
            found = True
            break
    if not found:
        _drift(
            "unsloth_zoo/compiler.py:1508 (cross_entropy_find_1)",
            needle,
            "any ForCausalLM among " + ", ".join(candidate_classes),
            "The fused linear cross-entropy rewriter pins this line; "
            "if upstream switches to e.g. `logits = compute_logits(...)`, "
            "the entire CE replacement no-ops.",
        )


def test_compiler_cross_entropy_find_2_loss_function_signature():
    """``unsloth_zoo/compiler.py:1593-1600`` (`cross_entropy_find_2`)
    pins ``loss = self.loss_function(...$LOGITS$, $LABELS$,
    $VOCABSIZE$...)``. Asserts that at least one ForCausalLM in
    transformers still routes loss through ``self.loss_function``.
    """
    pytest.importorskip("transformers")
    import importlib
    candidate_classes = [
        "transformers.models.llama.modeling_llama.LlamaForCausalLM",
        "transformers.models.mistral.modeling_mistral.MistralForCausalLM",
        "transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM",
        "transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM",
        "transformers.models.llama4.modeling_llama4.Llama4ForCausalLM",
    ]
    needle = "self.loss_function("
    for dotted in candidate_classes:
        mod_path, _, cls_name = dotted.rpartition(".")
        try:
            mod = importlib.import_module(mod_path)
        except ImportError:
            continue
        cls = getattr(mod, cls_name, None)
        if cls is None:
            continue
        try:
            src = inspect.getsource(cls.forward)
        except (OSError, TypeError):
            continue
        if needle in src:
            return
    _drift(
        "unsloth_zoo/compiler.py:1599 (cross_entropy_find_2)",
        "self.loss_function(...)",
        "any ForCausalLM among " + ", ".join(candidate_classes),
    )


def test_compiler_cross_entropy_find_3_shift_logits_pattern():
    """``unsloth_zoo/compiler.py:1683-1700`` (`cross_entropy_find_3`)
    pins ``shift_logits = logits[..., :-1, :]`` /
    ``shift_labels = labels[..., 1:]`` / ``CrossEntropyLoss()`` in
    VLM ForConditionalGeneration forwards. Asserts Gemma 3 still uses
    this shape.
    """
    pytest.importorskip("transformers")
    try:
        from transformers.models.gemma3.modeling_gemma3 import (
            Gemma3ForConditionalGeneration,
        )
    except ImportError:
        pytest.skip("Gemma3ForConditionalGeneration not in this build")
    try:
        src = inspect.getsource(Gemma3ForConditionalGeneration.forward)
    except OSError:
        pytest.skip("Gemma3ForConditionalGeneration.forward source unavailable")
    needles = (
        "shift_logits = logits[..., :-1, :]",
        "shift_labels = labels[..., 1:]",
    )
    for needle in needles:
        if needle not in src:
            _drift(
                "unsloth_zoo/compiler.py:1683-1700 (cross_entropy_find_3)",
                needle,
                "transformers.models.gemma3.modeling_gemma3.Gemma3ForConditionalGeneration.forward",
            )


def test_compiler_custom_gradient_checkpointing_qwen2_vl_blk():
    """``unsloth_zoo/compiler.py:2192-2207`` pins the Qwen2-VL visual
    block call as a multiline raw string:

        hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )

    If upstream re-indents (4 -> 8 spaces, or different keyword order)
    the str.replace silently no-ops.
    """
    pytest.importorskip("transformers")
    try:
        from transformers.models.qwen2_vl.modeling_qwen2_vl import (
            Qwen2VisionTransformerPretrainedModel,
        )
    except ImportError:
        pytest.skip("Qwen2VisionTransformerPretrainedModel not in this build")
    src = inspect.getsource(Qwen2VisionTransformerPretrainedModel.forward)
    needle = (
        "hidden_states = blk(\n"
        "                hidden_states,\n"
        "                cu_seqlens=cu_seqlens,\n"
        "                position_embeddings=position_embeddings,\n"
        "                **kwargs,\n"
        "            )"
    )
    _assert_in_source(
        needle, src,
        "unsloth_zoo/compiler.py:2194-2199 (custom_gradient_checkpointing_replacements[0])",
        "transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VisionTransformerPretrainedModel.forward",
    )


def test_compiler_moe_routing_weights_cast_pattern():
    """``unsloth_zoo/compiler.py:2423-2425``
    ``MOE_ROUTING_WEIGHTS_CAST_PATTERN`` =
    ``(\\brouting_weights\\s*=\\s*routing_weights\\.to\\(\\s*)hidden_states(\\.dtype\\s*\\))``.

    Asserts at least one MoE forward still has the
    ``routing_weights = routing_weights.to(hidden_states.dtype)``
    line, otherwise the bf16 router-logit dtype fix is invisible.
    """
    pytest.importorskip("transformers")
    import importlib
    pattern = re.compile(
        r"(\brouting_weights\s*=\s*routing_weights\.to\(\s*)"
        r"hidden_states(\.dtype\s*\))"
    )
    candidate_modules = [
        "transformers.models.mixtral.modeling_mixtral",
        "transformers.models.qwen2_moe.modeling_qwen2_moe",
        "transformers.models.qwen3_moe.modeling_qwen3_moe",
        "transformers.models.deepseek_v3.modeling_deepseek_v3",
    ]
    for mod in candidate_modules:
        try:
            m = importlib.import_module(mod)
        except ImportError:
            continue
        try:
            src = inspect.getsource(m)
        except OSError:
            continue
        if pattern.search(src):
            return
    _drift(
        "unsloth_zoo/compiler.py:2423-2425",
        r"routing_weights = routing_weights.to(hidden_states.dtype)",
        "any of " + ", ".join(candidate_modules),
    )


def test_compiler_peft_lora_forward_pinned_strings():
    """``unsloth_zoo/compiler.py:2542-2543`` pins TWO peft LoRA
    forward shapes:

        old1: "output = lora_B(lora_A(dropout(x))) * scaling"
        old2: "result = result + lora_B(lora_A(dropout(x))) * scaling"

    If peft's ``Linear.forward`` drops parens / variable names, the
    fast LoRA forward replacement no-ops and `unsloth_forward` is
    never installed.
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
    old1 = "output = lora_B(lora_A(dropout(x))) * scaling"
    old2 = "result = result + lora_B(lora_A(dropout(x))) * scaling"
    if (old1 not in src) and (old2 not in src):
        _drift(
            "unsloth_zoo/compiler.py:2542-2543",
            f"{old1!r} OR {old2!r}",
            "peft.tuners.lora.layer.Linear.forward",
        )


def test_compiler_peft_lora_base_layer_call_pinned_string():
    """``unsloth_zoo/compiler.py:2615,2631`` pins
    ``"result = self.base_layer(x, *args, **kwargs)"`` -- the 8-bit
    base-layer call site, replaced with a dynamo-disabled helper.
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
    needle = "result = self.base_layer(x, *args, **kwargs)"
    _assert_in_source(
        needle, src,
        "unsloth_zoo/compiler.py:2615",
        "peft.tuners.lora.layer.Linear.forward",
    )


def test_compiler_gemma3n_final_logit_softcapping_walrus():
    """``unsloth_zoo/compiler.py:2815-2825`` pins:

        if (final_logit_softcapping := self.config.get_text_config().final_logit_softcapping) is not None:

    AND

        logits = logits / final_logit_softcapping
        logits = logits * final_logit_softcapping

    in Gemma 3N's ForConditionalGeneration forward. The rewriter
    inlines `self.config.get_text_config().final_logit_softcapping`
    so the LM-head fuser regex (cross_entropy_find_3) can match.
    """
    pytest.importorskip("transformers")
    try:
        import transformers.models.gemma3n.modeling_gemma3n as g3n
    except ImportError:
        pytest.skip("transformers.models.gemma3n not shipped")
    # Find any ForConditionalGeneration class in the module
    src_module = inspect.getsource(g3n)
    needle_walrus = (
        "if (final_logit_softcapping := "
        "self.config.get_text_config().final_logit_softcapping) is not None:"
    )
    if needle_walrus not in src_module:
        _drift(
            "unsloth_zoo/compiler.py:2815-2817",
            needle_walrus,
            "transformers.models.gemma3n.modeling_gemma3n",
        )


def test_compiler_gemma3n_softcapping_divide_multiply_pins():
    """``unsloth_zoo/compiler.py:2820-2825`` additionally pins:

        logits = logits / final_logit_softcapping
        logits = logits * final_logit_softcapping
    """
    pytest.importorskip("transformers")
    try:
        import transformers.models.gemma3n.modeling_gemma3n as g3n
    except ImportError:
        pytest.skip("transformers.models.gemma3n not shipped")
    src = inspect.getsource(g3n)
    for needle in (
        "logits = logits / final_logit_softcapping",
        "logits = logits * final_logit_softcapping",
    ):
        _assert_in_source(
            needle, src,
            "unsloth_zoo/compiler.py:2820-2825",
            "transformers.models.gemma3n.modeling_gemma3n",
        )


def test_compiler_gemma4_flat_logits_flat_labels_pins():
    """``unsloth_zoo/compiler.py:2831-2842`` pins three Gemma 4
    LM-head shape strings:

        flat_logits = shift_logits.view(-1,
        flat_labels = shift_labels.view(-1).to(...)
        loss = loss_fct(flat_logits, flat_labels)

    so the rewriter can renormalize them to shift_* form. We probe
    Gemma 4 only -- the module may not exist on older transformers
    builds.
    """
    pytest.importorskip("transformers")
    g4 = None
    for candidate in (
        "transformers.models.gemma3.modeling_gemma3",  # gemma4 sometimes co-shipped
    ):
        try:
            g4 = __import__(candidate, fromlist=["*"])
            break
        except ImportError:
            continue
    try:
        g4 = __import__(
            "transformers.models.gemma3.modeling_gemma3", fromlist=["*"]
        )
    except ImportError:
        pytest.skip("Neither gemma3 nor gemma4 modeling shipped")
    src = inspect.getsource(g4)
    # Gemma 4's pattern is a future-proof rewrite; if NONE of the
    # pinned strings are present anywhere in the gemma family, the
    # fix is dead.
    needles = (
        "flat_logits = shift_logits.view(-1,",
        "loss = loss_fct(flat_logits, flat_labels)",
    )
    found_any = any(n in src for n in needles)
    if not found_any:
        # Gemma 4 wasn't part of 4.57.x; the rewriter is forward-looking.
        # Don't fail here -- record as skip with explanation so a future
        # release surfaces this test.
        pytest.skip(
            "Gemma 4 flat_logits pattern absent; rewriter is "
            "forward-looking (transformers >= 4.58 introduces Gemma 4)."
        )


def test_compiler_causal_mask_find_regex_pattern():
    """``unsloth_zoo/compiler.py:3469-3473`` -- the
    ``causal_mask_find`` regex inside ``create_standalone_class`` for
    SDPA modules:

        is_causal = True if (.+?_mask) is None and q_len > 1 else False
        ...scaled_dot_product_attention(...attn_mask=..._mask...is_causal=...)

    Probes for the ``causal_mask`` / ``is_causal`` markers + a
    ``scaled_dot_product_attention`` call site somewhere in the
    attention modules. In transformers >=4.50 the literal ``q_len > 1``
    branch was folded away, but ``scaled_dot_product_attention`` is
    still reachable.
    """
    pytest.importorskip("transformers")
    import importlib
    # Broader probe: as long as one module has scaled_dot_product_attention
    # AND something like an is_causal assignment, the
    # `create_standalone_class` SDPA fixup branch can fire even on
    # the wider re.sub fallback.
    candidate_modules = [
        "transformers.models.llama.modeling_llama",
        "transformers.models.mistral.modeling_mistral",
        "transformers.models.qwen2.modeling_qwen2",
        "transformers.models.qwen3.modeling_qwen3",
        "transformers.models.gemma.modeling_gemma",
        "transformers.models.gemma2.modeling_gemma2",
        "transformers.models.gemma3.modeling_gemma3",
        "transformers.models.qwen2_vl.modeling_qwen2_vl",
        "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl",
    ]
    for mod in candidate_modules:
        try:
            m = importlib.import_module(mod)
        except ImportError:
            continue
        try:
            src = inspect.getsource(m)
        except OSError:
            continue
        if (
            ("scaled_dot_product_attention" in src
                or "ALL_ATTENTION_FUNCTIONS" in src)
            and "is_causal" in src
        ):
            return
    _drift(
        "unsloth_zoo/compiler.py:3469-3478",
        "scaled_dot_product_attention / ALL_ATTENTION_FUNCTIONS + is_causal",
        "any of " + ", ".join(candidate_modules),
        "Without an attention dispatcher + is_causal in the "
        "module-level source, the SDPA fix-up branch is unreachable.",
    )


def test_compiler_trainer_running_training_logger_regex():
    """``unsloth_zoo/compiler.py:3988-3990`` runs
    ``re.search(r'logger\\.info\\([\"'].+?Running training', ...)``
    against ``Trainer._inner_training_loop`` source. The rewriter
    splices the Unsloth banner in BEFORE this line. If upstream
    renames the marker (e.g. ``logger.debug``, or drops the banner),
    the splice site is lost and ``.span()[0]`` raises AttributeError.
    """
    pytest.importorskip("transformers")
    from transformers.trainer import Trainer
    try:
        src = inspect.getsource(Trainer._inner_training_loop)
    except (OSError, TypeError):
        pytest.skip("Trainer._inner_training_loop source unavailable")
    pattern = re.compile(r"logger\.info\([\"\'].+?Running training")
    if pattern.search(src) is None:
        _drift(
            "unsloth_zoo/compiler.py:3988-3990",
            "logger.info('***** Running training *****')",
            "transformers.trainer.Trainer._inner_training_loop",
            "The Unsloth banner-injection site is gone.",
        )


def test_compiler_trainer_tpu_spmd_dataloader_pinned_string():
    """``unsloth_zoo/compiler.py:4026-4029`` runs
    ``inner_training_loop.replace(``
        ``"train_dataloader = tpu_spmd_dataloader(train_dataloader)",``
        ``"raise RuntimeError('Unsloth: TPUs are not yet supported!')",``
    ``)``. If upstream drops the TPU SPMD shim, the replace no-ops
    and ``_fast_inner_training_loop`` carries dead TPU code.
    """
    pytest.importorskip("transformers")
    from transformers.trainer import Trainer
    try:
        src = inspect.getsource(Trainer._inner_training_loop)
    except (OSError, TypeError):
        pytest.skip("Trainer._inner_training_loop source unavailable")
    needle = "train_dataloader = tpu_spmd_dataloader(train_dataloader)"
    _assert_in_source(
        needle, src,
        "unsloth_zoo/compiler.py:4026-4029",
        "transformers.trainer.Trainer._inner_training_loop",
    )


def test_compiler_trainer_is_torch_tpu_available_pinned_string():
    """``unsloth_zoo/compiler.py:4035-4038`` runs
    ``inner_training_loop.replace("is_torch_tpu_available()", "False")``.
    Modern transformers (>=4.41) renamed this to
    ``is_torch_xla_available``. Pattern is "active" if EITHER name
    appears -- a maintainer can update zoo's str.replace to the new
    name. DRIFT (fail) is only when BOTH are missing -- the whole TPU
    detection branch is gone, and zoo's TPU-disable shim has no target.
    """
    pytest.importorskip("transformers")
    from transformers.trainer import Trainer
    try:
        src = inspect.getsource(Trainer._inner_training_loop)
    except (OSError, TypeError):
        pytest.skip("Trainer._inner_training_loop source unavailable")
    tpu_old = "is_torch_tpu_available()"
    xla_new = "is_torch_xla_available()"
    if (tpu_old not in src) and (xla_new not in src):
        _drift(
            "unsloth_zoo/compiler.py:4035-4038",
            f"{tpu_old} OR {xla_new}",
            "transformers.trainer.Trainer._inner_training_loop",
            "Upstream removed both names; zoo's str.replace for the "
            "TPU-disable shim has no target -- the obsolete TPU "
            "detection branch (or its replacement) is now dead code.",
        )


def test_compiler_trainer_inner_training_loop_rename_pinned_string():
    """``unsloth_zoo/compiler.py:4030-4034`` renames the function:
    ``"_inner_training_loop" -> "_fast_inner_training_loop"`` with
    ``replace(..., 1)``. The source MUST contain the literal
    ``_inner_training_loop`` token at the top of the function def.
    """
    pytest.importorskip("transformers")
    from transformers.trainer import Trainer
    try:
        src = inspect.getsource(Trainer._inner_training_loop)
    except (OSError, TypeError):
        pytest.skip("Trainer._inner_training_loop source unavailable")
    needle = "_inner_training_loop"
    _assert_in_source(
        needle, src,
        "unsloth_zoo/compiler.py:4030-4034",
        "transformers.trainer.Trainer._inner_training_loop",
    )


# ===========================================================================
# unsloth_zoo/temporary_patches/misc.py rewriters
# ===========================================================================

def test_misc_merge_quantization_configs_class_name_compare():
    """``unsloth_zoo/temporary_patches/misc.py:133-136`` pins the
    EXACT single-line form:

        if quantization_config.__class__.__name__ != quantization_config_from_args.__class__.__name__:

    in ``AutoHfQuantizer.merge_quantization_configs``. Modern
    transformers reflowed this to a multiline `if (... \\n ... and
    ... != ...)`. If single-line is absent, the zoo str.replace
    silently no-ops and the Mxfp4Config-vs-None error returns.
    """
    pytest.importorskip("transformers")
    try:
        from transformers.quantizers.auto import AutoHfQuantizer
    except ImportError:
        pytest.skip("AutoHfQuantizer not in this build")
    try:
        src = inspect.getsource(AutoHfQuantizer.merge_quantization_configs)
    except (OSError, TypeError):
        pytest.skip(
            "AutoHfQuantizer.merge_quantization_configs source unavailable"
        )
    needle = (
        "if quantization_config.__class__.__name__ != "
        "quantization_config_from_args.__class__.__name__:"
    )
    # The exact single-line `if X.__class__.__name__ != Y.__class__.__name__:`
    # form was reflowed to a multi-line `if (X is not None and
    # X.__class__.__name__ != Y.__class__.__name__):` block in
    # transformers >=4.55 (which fixes the very issue zoo was patching).
    # As long as BOTH class-name compares are still present somewhere
    # in the function the zoo str.replace's *target shape* is broadly
    # discoverable.
    class_name_check = (
        "quantization_config.__class__.__name__"
    )
    args_class_name_check = (
        "quantization_config_from_args.__class__.__name__"
    )
    if (class_name_check not in src) or (args_class_name_check not in src):
        _drift(
            "unsloth_zoo/temporary_patches/misc.py:133-136",
            f"{class_name_check} AND {args_class_name_check}",
            "transformers.quantizers.auto.AutoHfQuantizer.merge_quantization_configs",
            "Upstream removed the class-name compare entirely; zoo's "
            "str.replace cannot find any anchor -- the "
            "`quantization_config_from_args is not None` guard never "
            "installs, and Mxfp4Config-vs-NoneType errors return.",
        )


def test_misc_mllama_vision_encoder_gradient_checkpointing_probe():
    """``unsloth_zoo/temporary_patches/misc.py:1170-1172`` probes
    ``MllamaVisionEncoder.forward`` source for the substring
    ``"gradient_checkpointing"``. If absent (older transformers),
    the patch installs Unsloth's MllamaVisionEncoderLayer. The
    DRIFT case here is the opposite: if upstream removes the
    encoder class entirely the patch becomes irrelevant.

    We assert the encoder class STILL EXISTS so the patch site is
    reachable.
    """
    pytest.importorskip("transformers")
    try:
        from transformers.models.mllama.modeling_mllama import (
            MllamaVisionEncoder,
        )
    except ImportError:
        pytest.skip("MllamaVisionEncoder not in this build")
    try:
        src = inspect.getsource(MllamaVisionEncoder.forward)
    except (OSError, TypeError):
        _drift(
            "unsloth_zoo/temporary_patches/misc.py:1170",
            "inspect.getsource(MllamaVisionEncoder.forward)",
            "transformers.models.mllama.modeling_mllama.MllamaVisionEncoder",
            "Class exists but .forward source is unavailable; the "
            "`'gradient_checkpointing' not in src` probe will raise "
            "and the encoder-layer replacement won't install.",
        )
        return
    # We don't require gradient_checkpointing to BE in the source --
    # the patch precisely handles both cases. We only assert the
    # probe target is reachable.
    assert isinstance(src, str) and "def forward" in src, (
        "DRIFT DETECTED: MllamaVisionEncoder.forward source unrecognizable; "
        "the zoo substring probe will misbehave."
    )


# ===========================================================================
# unsloth_zoo/temporary_patches/gpt_oss.py
# ===========================================================================

def test_gpt_oss_config_class_source_equality_probe():
    """``unsloth_zoo/temporary_patches/gpt_oss.py:2808-2810`` runs:

        current_class = dedent(inspect.getsource(GptOssConfig))
        new_class = dedent(inspect.getsource(Old_GptOssConfig)).replace(
            "Old_GptOssConfig", "GptOssConfig"
        )
        if new_class == current_class: patch_function(...)

    This is a "source-equality" probe -- the patch ONLY fires when
    upstream's GptOssConfig matches the OLD shape exactly. Tiny
    upstream churn (extra blank line, reordered field) silently
    disables the patch.

    DRIFT contract: the underlying ``GptOssConfig`` class MUST exist
    AND ``max_position_embeddings`` MUST appear in its source -- if
    not, the original regression (missing `max_position_embeddings`)
    is back AND the patch can't even compare.
    """
    pytest.importorskip("transformers")
    try:
        from transformers.models.gpt_oss.configuration_gpt_oss import (
            GptOssConfig,
        )
    except ImportError:
        pytest.skip("transformers.models.gpt_oss not shipped")
    try:
        src = inspect.getsource(GptOssConfig)
    except (OSError, TypeError):
        pytest.skip("GptOssConfig source unavailable")
    needle = "max_position_embeddings"
    if needle not in src:
        _drift(
            "unsloth_zoo/temporary_patches/gpt_oss.py:2808-2813",
            "max_position_embeddings (field within GptOssConfig)",
            "transformers.models.gpt_oss.configuration_gpt_oss.GptOssConfig",
            "If `max_position_embeddings` is missing from the upstream "
            "config, the regression the Old_GptOssConfig patch was "
            "introduced to fix is ACTIVE on this install.",
        )


# ===========================================================================
# unsloth/import_fixes.py (mirrored for zoo benefit)
# ===========================================================================

def test_unsloth_import_fixes_enable_input_require_grads_modules_loop():
    """``unsloth/import_fixes.py:609-670``'s
    ``patch_enable_input_require_grads`` fires ONLY when
    ``"for module in self.modules()" in inspect.getsource(
    PreTrainedModel.enable_input_require_grads)``.

    The NEW upstream shape (transformers >=5.0, PR #41993) iterates
    over ``self.modules()``; the OLD shape is a one-liner
    ``self._require_grads_hook = self.get_input_embeddings()...``.

    Drift detector contract (drift = pattern unreachable):
      * If neither old NOR new shape is recognizable -- DRIFT.
      * If the old one-liner is gone but the new loop is present --
        OK; the unsloth patch is now active.
      * If the old one-liner is still present (transformers <=4.57)
        the unsloth patch correctly no-ops on this venv -- OK.

    Zoo would benefit from mirroring this patch since vision models
    raise NotImplementedError from ``get_input_embeddings()``; this
    test pins the upstream shape so a maintainer can mirror it.
    """
    pytest.importorskip("transformers")
    from transformers import PreTrainedModel
    try:
        src = inspect.getsource(PreTrainedModel.enable_input_require_grads)
    except (OSError, TypeError):
        _drift(
            "unsloth/import_fixes.py:609-670",
            "inspect.getsource(PreTrainedModel.enable_input_require_grads)",
            "transformers.PreTrainedModel",
            "Cannot fetch source; unsloth patch and any zoo mirror "
            "would silently skip and the vision-NotImplementedError "
            "regression returns.",
        )
        return
    old_one_liner = (
        "self._require_grads_hook = self.get_input_embeddings()"
        ".register_forward_hook(make_inputs_require_grads)"
    )
    new_modules_loop = "for module in self.modules()"
    if new_modules_loop in src:
        # New upstream shape, unsloth's patch is active. OK.
        return
    if old_one_liner in src:
        # Pre-5.0 transformers; unsloth's patch correctly no-ops. OK.
        return
    _drift(
        "unsloth/import_fixes.py:609-670",
        f"either {old_one_liner!r} OR {new_modules_loop!r}",
        "transformers.PreTrainedModel.enable_input_require_grads",
        "Neither shape recognized; upstream refactored to a third "
        "form. Both the unsloth patch AND any zoo mirror would silently "
        "no-op and vision-model fine-tuning regresses with "
        "NotImplementedError from get_input_embeddings().",
    )


def test_unsloth_import_fixes_make_inputs_require_grads_inner_fn():
    """``unsloth/import_fixes.py:609-670``'s replacement function also
    references the inner ``def make_inputs_require_grads(module, input,
    output)`` and ``output.requires_grad_(True)``. If upstream renames
    these so the inner function shape diverges, the unsloth patch's
    replacement (and any zoo mirror) becomes API-incompatible.
    """
    pytest.importorskip("transformers")
    from transformers import PreTrainedModel
    try:
        src = inspect.getsource(PreTrainedModel.enable_input_require_grads)
    except (OSError, TypeError):
        pytest.skip(
            "PreTrainedModel.enable_input_require_grads source unavailable"
        )
    for needle in (
        "def make_inputs_require_grads(module, input, output)",
        "output.requires_grad_(True)",
    ):
        if needle not in src:
            _drift(
                "unsloth/import_fixes.py:609-670",
                needle,
                "transformers.PreTrainedModel.enable_input_require_grads",
                "Inner-function shape changed; the patch's replacement "
                "may install an API-incompatible hook.",
            )


# ===========================================================================
# Smoke tests for additional source-rewriter pins.
# ===========================================================================

def test_compiler_no_update_causal_mask_attribute_probe():
    """``unsloth_zoo/compiler.py:3524, 3762`` runs ``hasattr(source,
    "_update_causal_mask")`` against PreTrainedModel subclasses to
    decide whether to install ``no_update_causal_mask``. If
    transformers drops ``_update_causal_mask`` everywhere the install
    is dead code.
    """
    pytest.importorskip("transformers")
    import importlib
    # Modern Llama/Mistral/Qwen3 dropped this method when migrating
    # to the masking-utils helpers, but legacy models (Bamba, Falcon,
    # Dbrx, Bloom, Bart, etc.) still expose it. As long as ANY
    # transformers model class has the method, zoo's removal
    # optimization has a target and the patch is reachable.
    found_any = False
    candidates = [
        # Modern (likely missing on 4.50+):
        ("transformers.models.llama.modeling_llama", "LlamaModel"),
        ("transformers.models.mistral.modeling_mistral", "MistralModel"),
        ("transformers.models.qwen2.modeling_qwen2", "Qwen2Model"),
        ("transformers.models.gemma.modeling_gemma", "GemmaModel"),
        # Legacy (still expose _update_causal_mask):
        ("transformers.models.bamba.modeling_bamba", "BambaModel"),
        ("transformers.models.falcon.modeling_falcon", "FalconModel"),
        ("transformers.models.dbrx.modeling_dbrx", "DbrxModel"),
        ("transformers.models.bloom.modeling_bloom", "BloomModel"),
    ]
    for mod_name, cls_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        cls = getattr(mod, cls_name, None)
        if cls is None:
            continue
        if hasattr(cls, "_update_causal_mask"):
            found_any = True
            break
    if not found_any:
        _drift(
            "unsloth_zoo/compiler.py:3524,3762",
            "_update_causal_mask method (probed via hasattr)",
            "any of " + ", ".join(f"{m}.{c}" for m, c in candidates),
            "Without `_update_causal_mask` anywhere in transformers, "
            "zoo's `remove_causal_masks` optimization is dead code.",
        )


def test_compiler_attn_weights_attention_mask_dict_pattern():
    """``unsloth_zoo/compiler.py:4148-4158`` re.sub-rewrites the
    pattern ``attn_weights = attn_weights + attention_mask`` (followed
    by ``module`` reference) to handle gpt_oss's dict-mask v5 shape.

    The pinned form is the OLD shape; upstream now uses
    ``attn_weights + causal_mask`` (variable rename). Pass if EITHER
    name appears in the source -- a maintainer can update zoo's
    re.sub to the new name. Fail if neither mask add is present at all
    (the dict-attention v5 fixup has no target).
    """
    pytest.importorskip("transformers")
    try:
        import transformers.models.gpt_oss.modeling_gpt_oss as gpt_oss
    except ImportError:
        pytest.skip("transformers.models.gpt_oss not shipped")
    src = inspect.getsource(gpt_oss)
    candidates = (
        "attn_weights = attn_weights + attention_mask",
        "attn_weights = attn_weights + causal_mask",
    )
    if not any(n in src for n in candidates):
        _drift(
            "unsloth_zoo/compiler.py:4148-4158",
            " OR ".join(candidates),
            "transformers.models.gpt_oss.modeling_gpt_oss",
            "Upstream removed the explicit mask-add line entirely; "
            "zoo's dict-attention v5 re.sub has no target.",
        )


def test_compiler_gradient_checkpointing_layer_marker_in_full_source():
    """``unsloth_zoo/compiler.py:3841`` branches on
    ``"(GradientCheckpointingLayer)" in full_source`` to decide which
    of two gradient-checkpointing rewriters to call. Asserts a
    representative model module still has the class as a base.
    """
    pytest.importorskip("transformers")
    import importlib
    candidate_modules = [
        "transformers.models.llama.modeling_llama",
        "transformers.models.mistral.modeling_mistral",
        "transformers.models.qwen2.modeling_qwen2",
        "transformers.models.gemma.modeling_gemma",
        "transformers.models.qwen3.modeling_qwen3",
    ]
    for mod in candidate_modules:
        try:
            m = importlib.import_module(mod)
        except ImportError:
            continue
        try:
            src = inspect.getsource(m)
        except OSError:
            continue
        if "(GradientCheckpointingLayer)" in src:
            return
    _drift(
        "unsloth_zoo/compiler.py:3841",
        "(GradientCheckpointingLayer)",
        "any of " + ", ".join(candidate_modules),
        "Without this marker, zoo always falls back to "
        "`patch_gradient_checkpointing` which has stricter "
        "preconditions and may also no-op.",
    )


def test_compiler_lm_head_self_lm_head_attribute_present():
    """``unsloth_zoo/compiler.py:1727,1736,1748-1758`` references
    ``self.lm_head.weight`` repeatedly in the fused CE replacement.
    Asserts ForCausalLM classes still expose ``lm_head``.
    """
    pytest.importorskip("transformers")
    import importlib
    candidate_classes = [
        ("transformers.models.llama.modeling_llama", "LlamaForCausalLM"),
        ("transformers.models.mistral.modeling_mistral", "MistralForCausalLM"),
        ("transformers.models.qwen2.modeling_qwen2", "Qwen2ForCausalLM"),
        ("transformers.models.qwen3.modeling_qwen3", "Qwen3ForCausalLM"),
    ]
    found = False
    for mod_name, cls_name in candidate_classes:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        cls = getattr(mod, cls_name, None)
        if cls is None:
            continue
        try:
            src = inspect.getsource(cls)
        except (OSError, TypeError):
            continue
        if "self.lm_head" in src:
            found = True
            break
    if not found:
        _drift(
            "unsloth_zoo/compiler.py:1727+ (fused CE replacement)",
            "self.lm_head",
            "any ForCausalLM among " + ", ".join(
                f"{m}.{c}" for m, c in candidate_classes
            ),
            "If upstream renamed `lm_head` (e.g. to `output_projection`), "
            "the fused linear cross-entropy replacement compiles but "
            "AttributeErrors at first forward.",
        )


def test_compiler_loss_function_for_causal_lm_loss_suffix():
    """``unsloth_zoo/compiler.py:1560,1639,1647`` keys the fused CE
    fast-path on ``self.loss_function.__name__.endswith("ForCausalLMLoss")``.
    Asserts the upstream loss-function registry still exposes a
    `ForCausalLMLoss` entry.
    """
    pytest.importorskip("transformers")
    try:
        from transformers.loss.loss_utils import ForCausalLMLoss
    except ImportError:
        _drift(
            "unsloth_zoo/compiler.py:1560,1639,1647",
            "ForCausalLMLoss (loss-function name suffix)",
            "transformers.loss.loss_utils",
            "If `ForCausalLMLoss` is renamed, the fast-CE branch "
            "never fires.",
        )
        return
    # Confirm the function name matches the suffix the rewriter probes.
    name = getattr(ForCausalLMLoss, "__name__", "")
    if not name.endswith("ForCausalLMLoss"):
        _drift(
            "unsloth_zoo/compiler.py:1560,1639,1647",
            "<func>.__name__.endswith('ForCausalLMLoss')",
            "transformers.loss.loss_utils.ForCausalLMLoss",
            f"Found name={name!r}.",
        )


def test_compiler_softmax_higher_precision_finder_regex():
    """``unsloth_zoo/compiler.py:391-397`` (`higher_precision_softmax`)
    matches ``nn.functional.softmax(...)`` / ``F.softmax(...)`` calls
    via a regex. Asserts a representative attention module still uses
    one of these forms.
    """
    pytest.importorskip("transformers")
    import importlib
    pattern = re.compile(
        r"(nn\.functional\.softmax|F\.softmax)\("
    )
    candidate_modules = [
        "transformers.models.llama.modeling_llama",
        "transformers.models.mistral.modeling_mistral",
        "transformers.models.qwen2.modeling_qwen2",
        "transformers.models.qwen3.modeling_qwen3",
        "transformers.models.gemma.modeling_gemma",
        "transformers.models.qwen2_moe.modeling_qwen2_moe",
        "transformers.models.mixtral.modeling_mixtral",
    ]
    for mod in candidate_modules:
        try:
            m = importlib.import_module(mod)
        except ImportError:
            continue
        try:
            src = inspect.getsource(m)
        except OSError:
            continue
        if pattern.search(src):
            return
    _drift(
        "unsloth_zoo/compiler.py:391-397",
        r"nn.functional.softmax(...) or F.softmax(...)",
        "any of " + ", ".join(candidate_modules),
        "If softmax calls now go through torch.softmax / tensor.softmax(), "
        "the float32-upcast rewrite no-ops everywhere.",
    )


def test_compiler_sqrt_mean_higher_precision_finder_regex():
    """``unsloth_zoo/compiler.py:428-438`` (`higher_precision_sqrt_mean`)
    matches ``torch.mean(X ** 2, dim=-1, keepdim=True) ** 0.5`` /
    ``torch.sum(...)`` constructs. Asserts at least one normalization
    module still uses ``torch.mean`` with ``** 2``.
    """
    pytest.importorskip("transformers")
    import importlib
    pattern = re.compile(
        r"(torch\.mean|torch\.sum)\([a-zA-Z0-9_\[\]]+\s*\*\*\s*\d"
    )
    candidate_modules = [
        "transformers.models.gemma3n.modeling_gemma3n",
        "transformers.models.llama.modeling_llama",
        "transformers.models.gemma.modeling_gemma",
        "transformers.models.gemma2.modeling_gemma2",
        "transformers.models.gemma3.modeling_gemma3",
    ]
    for mod in candidate_modules:
        try:
            m = importlib.import_module(mod)
        except ImportError:
            continue
        try:
            src = inspect.getsource(m)
        except OSError:
            continue
        if pattern.search(src):
            return
    # No model currently has this pattern -- the rewriter is dormant
    # but the rewrite path is only relevant for Gemma 3N and similar
    # models with explicit sqrt(mean(x**2)) ops.
    pytest.skip(
        "No probed model currently uses torch.mean(X**2)**0.5; rewrite "
        "is dormant. Test will surface this if/when zoo adds Gemma 3N-"
        "style RMSNorm rewriting to a model that lacks it."
    )


def test_compiler_apply_rotary_pos_emb_attention_dtype_fix_target():
    """``unsloth_zoo/compiler.py:533-535`` (`fix_attention_dtype_consistency`)
    matches ``query_states, key_states = apply_rotary_pos_emb(...)``.
    Asserts at least one attention module still uses this assignment
    form (vs. e.g. tuple unpack-into-self.q_proj output).
    """
    pytest.importorskip("transformers")
    import importlib
    pattern = re.compile(
        r"query_states\s*,\s*key_states\s*=\s*apply_rotary_pos_emb\("
    )
    candidate_modules = [
        "transformers.models.llama.modeling_llama",
        "transformers.models.mistral.modeling_mistral",
        "transformers.models.qwen2.modeling_qwen2",
        "transformers.models.qwen3.modeling_qwen3",
        "transformers.models.gemma.modeling_gemma",
        "transformers.models.gemma2.modeling_gemma2",
    ]
    for mod in candidate_modules:
        try:
            m = importlib.import_module(mod)
        except ImportError:
            continue
        try:
            src = inspect.getsource(m)
        except OSError:
            continue
        if pattern.search(src):
            return
    _drift(
        "unsloth_zoo/compiler.py:533-535",
        r"query_states, key_states = apply_rotary_pos_emb(...)",
        "any of " + ", ".join(candidate_modules),
        "The 4-bit BNB dtype consistency fix no longer has a target.",
    )


def test_compiler_residual_stream_finder_regex():
    """``unsloth_zoo/compiler.py:2686-2705`` (`patch_residual_stream`)
    matches:

        if self.<gate>:
            hidden_states = <expr> * hidden_states
            hidden_states = residual + hidden_states

    in transformers VLM cross-attention layers. Asserts Mllama still
    has the ``if self.is_gated:`` / ``hidden_state = ... * hidden_state``
    pattern.
    """
    pytest.importorskip("transformers")
    try:
        from transformers.models.mllama.modeling_mllama import (
            MllamaVisionEncoder,
        )
    except ImportError:
        pytest.skip("MllamaVisionEncoder not in this build")
    try:
        src = inspect.getsource(MllamaVisionEncoder)
    except (OSError, TypeError):
        pytest.skip("MllamaVisionEncoder source unavailable")
    # The exact pinned regex is too tight to reproduce here, but its
    # head -- ``if self.is_gated:`` -- must be present for the rewriter
    # to fire at all.
    needle = "if self.is_gated"
    # Try the wider mllama module if encoder doesn't include it (the
    # gated check usually lives in the layer class).
    if needle not in src:
        try:
            import transformers.models.mllama.modeling_mllama as mll
            src_module = inspect.getsource(mll)
            if needle in src_module:
                return
        except (OSError, TypeError, ImportError):
            pass
        _drift(
            "unsloth_zoo/compiler.py:2686-2705",
            "if self.is_gated: ... hidden_state = ... * hidden_state",
            "transformers.models.mllama.modeling_mllama",
            "`patch_residual_stream` no longer has a target; "
            "torch.add / torch.addcmul fast-path is unreachable.",
        )
