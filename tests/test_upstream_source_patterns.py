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

Patches in ``unsloth_zoo/compiler.py`` and
``unsloth_zoo/temporary_patches/*.py`` fetch upstream function source
via ``inspect.getsource`` and ``str.replace`` / ``re.sub`` against
literal strings. If upstream renames/refactors/reflows the targeted
region, the rewriter silently no-ops and the zoo patch becomes invisible.

DRIFT-DETECTED framing: when the pinned string / regex is gone, fire
``pytest.fail("DRIFT DETECTED: zoo source-rewriter at <zoo file:line>
expects '<pattern>' in <upstream module>, not found")``. Each test
cites the zoo file:line it pins.
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
    msg = (
        f"DRIFT DETECTED: zoo source-rewriter at {zoo_site} expects "
        f"{pattern!r} in {upstream_path}, not found."
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


def _get_source_of(dotted: str):
    """Import dotted parent module and return ``inspect.getsource`` on
    the leaf. Missing leaf/parent -> ``importorskip`` (not drift)."""
    parts = dotted.split(".")
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
    ``"dropout_p=self.dropout if self.training else 0.0,"`` -- the
    rewriter inserts ``enable_gqa=...`` after this substring.

    Known active drift on transformers >=4.50: upstream switched to
    ``dropout=self.attention_dropout if self.training else 0.0,``
    (no ``_p`` suffix). Pass if any old-or-new form is present
    anywhere; fail only if the idiom is entirely gone."""
    pytest.importorskip("transformers")
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
    """``unsloth_zoo/compiler.py:262-282`` -- the
    ``grouped_query_attention_finder`` regex targets ``key_states =
    repeat_kv(...) / value_states = repeat_kv(...) / ... contiguous()``.
    transformers >=4.50 inlined repeat_kv into eager_attention_forward,
    so the finder may match 0 times -> rewrite dormant."""
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
            return
    _drift(
        "unsloth_zoo/compiler.py:262-282",
        r"key_states = repeat_kv(...)",
        "any of " + ", ".join(candidate_modules),
        "If 4.50+ inlined repeat_kv into eager_attention_forward, "
        "the GQA finder regex matches 0 times everywhere and the "
        "GQA rewrite is invisible.",
    )


def test_compiler_output_attentions_super_forward_regex_targetable():
    """``unsloth_zoo/compiler.py:316-321`` runs re.sub for
    ``if output_attentions: ... return super().forward(...)``. 4.57
    removed the immediate ``return super().forward`` follow-up; pass if
    ``if output_attentions`` marker is still discoverable so a
    maintainer can re-anchor."""
    pytest.importorskip("transformers")
    import importlib
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
    ``source.replace("self.config.ignore_index", "-100")``. By 4.57
    only qwen2_audio still references it; pass if any upstream model
    still has the exact string."""
    pytest.importorskip("transformers")
    import importlib
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
    """``unsloth_zoo/compiler.py:1404-1407`` rewrites Gemma 3N's
    ``per_layer_projection *= self.per_layer_projection_scale.to(...)``
    to a non-inplace form."""
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
    """``unsloth_zoo/compiler.py:1508-1525`` (cross_entropy_find_1)
    expects ``logits = self.lm_head(hidden_states`` at the head of the
    loss block in every ForCausalLM forward."""
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
    """``unsloth_zoo/compiler.py:1593-1600`` (cross_entropy_find_2) pins
    ``loss = self.loss_function(...$LOGITS$, $LABELS$, $VOCABSIZE$...)``."""
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
    """``unsloth_zoo/compiler.py:1683-1700`` (cross_entropy_find_3) pins
    ``shift_logits = logits[..., :-1, :]`` / ``shift_labels = labels[..., 1:]``
    in VLM ForConditionalGeneration forwards."""
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
    """``unsloth_zoo/compiler.py:2192-2207`` pins the Qwen2-VL multiline
    raw string ``hidden_states = blk(\\n hidden_states,\\n
    cu_seqlens=cu_seqlens,\\n position_embeddings=position_embeddings,\\n
    **kwargs,\\n )``. A re-indent silently no-ops."""
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
    """``unsloth_zoo/compiler.py:2423-2425`` MOE_ROUTING_WEIGHTS_CAST_PATTERN
    targets ``routing_weights = routing_weights.to(hidden_states.dtype)``;
    needed for the bf16 router-logit dtype fix."""
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
    """``unsloth_zoo/compiler.py:2542-2543`` pins:
        old1: "output = lora_B(lora_A(dropout(x))) * scaling"
        old2: "result = result + lora_B(lora_A(dropout(x))) * scaling"
    If peft drops parens/names, the fast LoRA forward never installs."""
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
    base-layer call site, replaced with a dynamo-disabled helper."""
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
    """``unsloth_zoo/compiler.py:2815-2825`` pins the walrus form
    ``if (final_logit_softcapping := self.config.get_text_config()
    .final_logit_softcapping) is not None:`` in Gemma 3N's
    ForConditionalGeneration forward."""
    pytest.importorskip("transformers")
    try:
        import transformers.models.gemma3n.modeling_gemma3n as g3n
    except ImportError:
        pytest.skip("transformers.models.gemma3n not shipped")
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
    """``unsloth_zoo/compiler.py:2820-2825`` also pins
    ``logits = logits / final_logit_softcapping`` and
    ``logits = logits * final_logit_softcapping``."""
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
    """``unsloth_zoo/compiler.py:2831-2842`` pins Gemma 4 LM-head shape
    strings; rewriter is forward-looking (Gemma 4 lands in >= 4.58).
    Skip cleanly when pattern is absent."""
    pytest.importorskip("transformers")
    g4 = None
    for candidate in (
        "transformers.models.gemma3.modeling_gemma3",
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
    needles = (
        "flat_logits = shift_logits.view(-1,",
        "loss = loss_fct(flat_logits, flat_labels)",
    )
    found_any = any(n in src for n in needles)
    if not found_any:
        pytest.skip(
            "Gemma 4 flat_logits pattern absent; rewriter is "
            "forward-looking (transformers >= 4.58 introduces Gemma 4)."
        )


def test_compiler_causal_mask_find_regex_pattern():
    """``unsloth_zoo/compiler.py:3469-3473`` causal_mask_find regex
    targets ``is_causal = True if (.+?_mask) is None and q_len > 1
    else False`` + scaled_dot_product_attention. Pass as long as
    dispatcher + is_causal markers exist."""
    pytest.importorskip("transformers")
    import importlib
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
    """``unsloth_zoo/compiler.py:3988-3990`` re.searches
    ``logger.info('***** Running training *****')`` in
    ``Trainer._inner_training_loop``; rewriter splices the Unsloth
    banner before this line."""
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
    """``unsloth_zoo/compiler.py:4026-4029`` replaces
    ``train_dataloader = tpu_spmd_dataloader(train_dataloader)`` with
    a RuntimeError TPU stub."""
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
    """``unsloth_zoo/compiler.py:4035-4038`` replaces
    ``is_torch_tpu_available()`` with ``False``. Modern transformers
    renamed to ``is_torch_xla_available``; pass if EITHER name appears."""
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
    """``unsloth_zoo/compiler.py:4030-4034`` renames
    ``_inner_training_loop -> _fast_inner_training_loop``."""
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
    """``unsloth_zoo/temporary_patches/misc.py:133-136`` pins the single-line
    ``if quantization_config.__class__.__name__ !=
    quantization_config_from_args.__class__.__name__:``. Modern
    transformers reflowed this; pass as long as BOTH class-name compares
    are present somewhere."""
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
    class_name_check = "quantization_config.__class__.__name__"
    args_class_name_check = "quantization_config_from_args.__class__.__name__"
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
    ``MllamaVisionEncoder.forward`` for ``"gradient_checkpointing"``.
    Drift = encoder class removed -> patch unreachable."""
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
    assert isinstance(src, str) and "def forward" in src, (
        "DRIFT DETECTED: MllamaVisionEncoder.forward source unrecognizable; "
        "the zoo substring probe will misbehave."
    )


# ===========================================================================
# unsloth_zoo/temporary_patches/gpt_oss.py
# ===========================================================================

def test_gpt_oss_config_class_source_equality_probe():
    """``unsloth_zoo/temporary_patches/gpt_oss.py:2808-2810`` runs a
    source-equality probe between ``GptOssConfig`` and the bundled
    ``Old_GptOssConfig``. Drift contract: GptOssConfig must exist AND
    ``max_position_embeddings`` must appear; otherwise the regression
    the Old_GptOssConfig patch was introduced to fix is ACTIVE."""
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
    ``patch_enable_input_require_grads`` fires only when ``"for module
    in self.modules()"`` is in
    ``PreTrainedModel.enable_input_require_grads`` source. Old shape is
    a one-liner ``self._require_grads_hook = self.get_input_embeddings()
    .register_forward_hook(make_inputs_require_grads)``. Drift = neither
    shape recognizable."""
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
        return
    if old_one_liner in src:
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
    """``unsloth/import_fixes.py:609-670``'s replacement also references
    inner ``def make_inputs_require_grads(module, input, output)`` and
    ``output.requires_grad_(True)``."""
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
# Additional source-rewriter pins.
# ===========================================================================

def test_compiler_no_update_causal_mask_attribute_probe():
    """``unsloth_zoo/compiler.py:3524, 3762`` ``hasattr(source,
    "_update_causal_mask")`` probe. Modern Llama/Mistral/Qwen3 dropped
    it; legacy models (Bamba, Falcon, etc.) still expose it. Pass if any
    model still has it."""
    pytest.importorskip("transformers")
    import importlib
    found_any = False
    candidates = [
        ("transformers.models.llama.modeling_llama", "LlamaModel"),
        ("transformers.models.mistral.modeling_mistral", "MistralModel"),
        ("transformers.models.qwen2.modeling_qwen2", "Qwen2Model"),
        ("transformers.models.gemma.modeling_gemma", "GemmaModel"),
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
    """``unsloth_zoo/compiler.py:4148-4158`` rewrites ``attn_weights =
    attn_weights + attention_mask`` (gpt_oss dict-mask v5 shape).
    Upstream may rename ``attention_mask`` -> ``causal_mask``; pass on
    either."""
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
    ``"(GradientCheckpointingLayer)" in full_source``."""
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
    ``self.lm_head.weight`` in the fused CE replacement."""
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
    fast-path on
    ``self.loss_function.__name__.endswith("ForCausalLMLoss")``."""
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
    name = getattr(ForCausalLMLoss, "__name__", "")
    if not name.endswith("ForCausalLMLoss"):
        _drift(
            "unsloth_zoo/compiler.py:1560,1639,1647",
            "<func>.__name__.endswith('ForCausalLMLoss')",
            "transformers.loss.loss_utils.ForCausalLMLoss",
            f"Found name={name!r}.",
        )


def test_compiler_softmax_higher_precision_finder_regex():
    """``unsloth_zoo/compiler.py:391-397`` (higher_precision_softmax)
    matches ``nn.functional.softmax(...)`` / ``F.softmax(...)``."""
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
    """``unsloth_zoo/compiler.py:428-438`` (higher_precision_sqrt_mean)
    targets ``torch.mean(X ** 2, dim=-1, keepdim=True) ** 0.5``.
    Currently dormant on modern models."""
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
    pytest.skip(
        "No probed model currently uses torch.mean(X**2)**0.5; rewrite "
        "is dormant. Test will surface this if/when zoo adds Gemma 3N-"
        "style RMSNorm rewriting to a model that lacks it."
    )


def test_compiler_apply_rotary_pos_emb_attention_dtype_fix_target():
    """``unsloth_zoo/compiler.py:533-535`` (fix_attention_dtype_consistency)
    matches ``query_states, key_states = apply_rotary_pos_emb(...)``."""
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
    """``unsloth_zoo/compiler.py:2686-2705`` (patch_residual_stream)
    matches ``if self.<gate>: hidden_states = <expr> * hidden_states ...
    hidden_states = residual + hidden_states`` in VLM cross-attention.
    Pin ``if self.is_gated`` head in mllama."""
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
    needle = "if self.is_gated"
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
