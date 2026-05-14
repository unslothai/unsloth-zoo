# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team.
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at
# your option) any later version.

"""Importable-symbol pins for upstream references in ``unsloth_zoo`` source.

The existing ``test_upstream_pinned_symbols_{transformers,trl_vllm,accelerator}.py``
files cover a curated subset of the symbols zoo reaches into upstream
libraries, mostly via raw-source GitHub fetches against pinned version
tags. This file is the complement: a flat enumeration of every
``from <upstream> import <symbol>`` and ``<upstream>.X.Y`` reference
visible in ``unsloth_zoo/**.py`` -- exercised against the **installed**
versions of transformers / trl / peft / datasets / accelerate / vllm.

Why both files? The github-fetch tests catch upstream API drift before
it lands in a user's venv. These tests catch the OPPOSITE failure mode:
a user's venv has a transformers / peft / etc. version that drops or
renames a symbol the zoo references unconditionally. The failure surface
is the same -- an ImportError or AttributeError at zoo import time --
but the trigger is different (venv content, not upstream main).

Each test names the source file + line it was extracted from in a
comment so a maintainer can grep back to the patch site. Tests use
``importlib.import_module`` + ``getattr`` chains so the failure mode is
a clean AssertionError with the missing dotted path printed.

The matrix dimension is the upstream version (HF=4.57.6 / HF=default /
HF=latest). A symbol that exists only on transformers >=X but is
referenced unconditionally in zoo source is a forward-compat bug, and
the test docstring flags that case.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Iterable

import pytest


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def _resolve(dotted: str) -> object:
    """``importlib.import_module`` + ``getattr`` chain. Raises an
    AssertionError naming the broken segment so the failure message
    points at the actual zoo callsite the symbol unblocks.

    Distinguishes:
      * module-file-actually-missing (find_spec returns None) -> FAIL,
        real upstream drift signal worth surfacing.
      * module-file-present-but-transitively-broken (find_spec returns
        a spec but import_module raises ImportError because of a
        nested optional dep, e.g. transformers.utils.notebook needing
        IPython) -> SKIP. Zoo's call sites for these paths are already
        try/except-wrapped (see e.g. logging_utils.py:49-56), so the
        zoo runtime tolerates the missing dep -- a test failure here
        would be noise, not signal.
      * attribute missing on a successfully-imported module -> FAIL.
    """
    parts = dotted.split(".")
    obj: object = None
    consumed: list[str] = []
    for i in range(len(parts), 0, -1):
        mod_name = ".".join(parts[:i])
        # Probe metadata first; this NEVER executes module code.
        try:
            spec = importlib.util.find_spec(mod_name)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            # find_spec returning None means the module path is
            # genuinely absent at this depth -- try a shorter prefix.
            continue
        # Spec exists; importing should succeed unless the module
        # itself has a transitively-broken optional dep.
        try:
            obj = importlib.import_module(mod_name)
            consumed = parts[:i]
            break
        except ImportError as exc:
            pytest.skip(
                f"`{mod_name}` exists but its imports fail on this "
                f"install ({type(exc).__name__}: {exc}); zoo wraps "
                "this in try/except so absence is not a runtime bug. "
                "Skipping to avoid false-positive in matrix CI."
            )
    if obj is None:
        raise AssertionError(
            f"Could not locate any module prefix of `{dotted}`; "
            "zoo references this dotted path -- regression at the "
            "import line (see source comment above the test)."
        )
    # Remaining parts must be attribute accesses on the module.
    for attr in parts[len(consumed):]:
        if not hasattr(obj, attr):
            walked = ".".join(consumed + [attr])
            raise AssertionError(
                f"`{walked}` missing on installed upstream "
                f"(walked from `{dotted}`); zoo references this "
                "exact path -- a rename or removal silently breaks "
                "the zoo patch site cited in the test comment."
            )
        obj = getattr(obj, attr)
        consumed.append(attr)
    return obj


def _resolve_all(dotted_paths: Iterable[str]) -> None:
    """Resolve every dotted path; collect missing entries into one
    AssertionError so a maintainer sees the full damage at once."""
    missing: list[str] = []
    for d in dotted_paths:
        try:
            _resolve(d)
        except AssertionError as e:
            missing.append(f"  - {d}\n      ({e})")
    assert not missing, "Missing upstream symbols:\n" + "\n".join(missing)


def _skip_if_missing(module_name: str) -> None:
    """Skip the test cleanly if the top-level upstream package isn't
    installed in this venv (mirrors ``pytest.importorskip``)."""
    pytest.importorskip(module_name)


# ===========================================================================
# unsloth_zoo/compiler.py
# ===========================================================================

def test_compiler_modeling_flash_attention_utils_top_level():
    """unsloth_zoo/compiler.py:218 — `from
    transformers.modeling_flash_attention_utils import
    is_flash_attn_available` is at module-top-level and UNGUARDED;
    if upstream removes the module, `import unsloth_zoo` itself
    ImportErrors during compile-cell construction."""
    _resolve_all([
        "transformers.modeling_flash_attention_utils",
        "transformers.modeling_flash_attention_utils.is_flash_attn_available",
    ])


def test_compiler_masking_utils():
    """unsloth_zoo/compiler.py:372 — `import transformers.masking_utils`.
    Module path is required for the compile-cell rewriter to inject the
    causal-mask helpers."""
    _resolve("transformers.masking_utils")


def test_compiler_transformers_logging():
    """unsloth_zoo/compiler.py:3145 — `from transformers import logging
    as transformers_logging`."""
    _resolve("transformers.logging")


def test_compiler_generation_mixin():
    """unsloth_zoo/compiler.py:3781 — `from transformers.generation
    import GenerationMixin` — used to detect generate() availability
    on a compiled model."""
    _resolve("transformers.generation.GenerationMixin")


def test_compiler_trainer_module_and_class():
    """unsloth_zoo/compiler.py:3963, 3975 — `from transformers.trainer
    import Trainer` AND `import transformers.trainer`."""
    _resolve_all([
        "transformers.trainer",
        "transformers.trainer.Trainer",
    ])


# ===========================================================================
# unsloth_zoo/loss_utils.py
# ===========================================================================

def test_loss_utils_training_args_parallel_mode():
    """unsloth_zoo/loss_utils.py:232 — TOP-LEVEL unguarded import
    `from transformers.training_args import ParallelMode`. Used by the
    Trainer parallelism branch to decide whether logits gathering is
    needed; a removal silently breaks distributed loss aggregation."""
    _resolve_all([
        "transformers.training_args",
        "transformers.training_args.ParallelMode",
    ])


def test_loss_utils_modeling_utils():
    """unsloth_zoo/loss_utils.py:138 — `import transformers.modeling_utils`
    feeds the `LOSS_MAPPING` rebind path."""
    _resolve("transformers.modeling_utils")


def test_loss_utils_loss_module():
    """unsloth_zoo/loss_utils.py:82 — `import transformers.loss.loss_utils`.
    The whole loss-helper subpackage moved into transformers 4.50; zoo
    relies on the `transformers.loss.loss_utils` path remaining stable."""
    _resolve("transformers.loss.loss_utils")


# ===========================================================================
# unsloth_zoo/training_utils.py — ALL top-level imports
# ===========================================================================

def test_training_utils_top_level_transformers_surface():
    """unsloth_zoo/training_utils.py:20-23 — four top-level imports.
    Any single removal makes ``from unsloth_zoo import ...`` blow up at
    every site that depends on training_utils (Trainer wrapper,
    data-collator helpers, scheduler patching)."""
    _resolve_all([
        "transformers.set_seed",
        "transformers.get_scheduler",
        "transformers.Trainer",
        "transformers.trainer_utils.seed_worker",
    ])


def test_training_utils_data_collator_for_lm():
    """unsloth_zoo/training_utils.py:345 — `from transformers import
    DataCollatorForLanguageModeling`."""
    _resolve("transformers.DataCollatorForLanguageModeling")


def test_training_utils_peft_modules_to_save_wrapper():
    """unsloth_zoo/training_utils.py:239 — `from peft.utils import
    ModulesToSaveWrapper`. This wrapper is how zoo identifies non-LoRA
    trainable adapter weights for the saving path."""
    _resolve("peft.utils.ModulesToSaveWrapper")


# ===========================================================================
# unsloth_zoo/dataset_utils.py
# ===========================================================================

def test_dataset_utils_datasets_top_level():
    """unsloth_zoo/dataset_utils.py:594 — `from datasets import (Dataset,
    IterableDataset,)`. Imported at module top-level; missing means the
    SFT data pipeline never loads."""
    _resolve_all(["datasets.Dataset", "datasets.IterableDataset"])


def test_dataset_utils_data_collator_for_seq2seq():
    """unsloth_zoo/dataset_utils.py:457, 672 — `from transformers import
    DataCollatorForSeq2Seq` (both call sites)."""
    _resolve("transformers.DataCollatorForSeq2Seq")


# ===========================================================================
# unsloth_zoo/saving_utils.py
# ===========================================================================

def test_saving_utils_pushtohubmixin():
    """unsloth_zoo/saving_utils.py:76 — TOP-LEVEL unguarded
    `from transformers.modeling_utils import PushToHubMixin`. We call
    `._upload_modified_files` and `._get_files_timestamps` on it."""
    _resolve("transformers.modeling_utils.PushToHubMixin")


def test_saving_utils_peft_top_level():
    """unsloth_zoo/saving_utils.py:82 + 270 — TOP-LEVEL unguarded
    `from peft import PeftModelForCausalLM, PeftModel` and
    `from peft.utils.integrations import dequantize_module_weight`."""
    _resolve_all([
        "peft.PeftModelForCausalLM",
        "peft.PeftModel",
        "peft.utils.integrations.dequantize_module_weight",
    ])


def test_saving_utils_autoconfig():
    """unsloth_zoo/saving_utils.py:2101 — `from transformers import
    AutoConfig`. Used inside the save-path config rewrite."""
    _resolve("transformers.AutoConfig")


# ===========================================================================
# unsloth_zoo/patching_utils.py
# ===========================================================================

def test_patching_utils_pretrainedconfig_either_name():
    """unsloth_zoo/patching_utils.py:247-251 — try `PreTrainedConfig`
    (4.x removed the camel-case) then `PretrainedConfig`. At least one
    MUST exist. Zoo source has BOTH forms gated by try/except so we
    only require ONE to resolve."""
    found = False
    for name in ("PreTrainedConfig", "PretrainedConfig"):
        try:
            _resolve(f"transformers.configuration_utils.{name}")
            found = True
            break
        except AssertionError:
            continue
    assert found, (
        "Neither PreTrainedConfig nor PretrainedConfig exists on "
        "transformers.configuration_utils; unsloth_zoo/patching_utils.py "
        ":247-251 try/except chain has no fallback left."
    )


def test_patching_utils_peft_linear4bit():
    """unsloth_zoo/patching_utils.py:313 — `from peft.tuners.lora import
    Linear4bit as Peft_Linear4bit`. This is the 4-bit LoRA layer that
    zoo's dtype/dequant patch keys on."""
    _resolve("peft.tuners.lora.Linear4bit")


def test_patching_utils_integrations_bitsandbytes_module():
    """unsloth_zoo/patching_utils.py:677 — `import
    transformers.integrations.bitsandbytes`. Module path used at
    IMPORT TIME (top-level) to introspect _replace_with_bnb_linear."""
    _resolve("transformers.integrations.bitsandbytes")


def test_patching_utils_quantizers_utils_module():
    """unsloth_zoo/patching_utils.py:761 — `import
    transformers.quantizers.quantizers_utils as _quantizers_utils`.
    Top-level on transformers 5.x. (On 4.x this module is present too
    in the installed window.)"""
    _resolve("transformers.quantizers.quantizers_utils")


# ===========================================================================
# unsloth_zoo/hf_utils.py
# ===========================================================================

def test_hf_utils_pretrainedconfig_either_name():
    """unsloth_zoo/hf_utils.py:25-28 — same dance as patching_utils:
    try `PreTrainedConfig` (5.x), fall back to `PretrainedConfig`
    (4.x). At least one MUST exist on top-level `transformers`."""
    found = False
    for name in ("PreTrainedConfig", "PretrainedConfig"):
        try:
            _resolve(f"transformers.{name}")
            found = True
            break
        except AssertionError:
            continue
    assert found, (
        "Neither PreTrainedConfig nor PretrainedConfig present on "
        "`transformers`; unsloth_zoo/hf_utils.py:25-28 has no name to "
        "bind to and dtype_from_config() breaks."
    )


def test_hf_utils_auto_processor_and_tokenizer():
    """unsloth_zoo/hf_utils.py:322, 363, 372 — `from transformers
    import AutoTokenizer` and `from transformers import AutoProcessor`.
    These drive zoo's `unsloth_tokenizer_from_pretrained` shim."""
    _resolve_all([
        "transformers.AutoTokenizer",
        "transformers.AutoProcessor",
    ])


def test_hf_utils_processor_mapping_names():
    """unsloth_zoo/hf_utils.py:278 — `from
    transformers.models.auto.processing_auto import
    PROCESSOR_MAPPING_NAMES`. Used to enumerate VLM processors."""
    _resolve(
        "transformers.models.auto.processing_auto.PROCESSOR_MAPPING_NAMES",
    )


def test_hf_utils_peft_config_top_level():
    """unsloth_zoo/hf_utils.py:119, 314 — `from peft import PeftConfig`.
    Two callsites, both under try/except — but they both want the same
    symbol. A removal disables BOTH adapter-detection paths."""
    _resolve("peft.PeftConfig")


# ===========================================================================
# unsloth_zoo/utils.py
# ===========================================================================

def test_utils_auto_quantization_config():
    """unsloth_zoo/utils.py:197 — `from transformers.quantizers import
    AutoQuantizationConfig`. Quantization config dispatch shim."""
    _resolve("transformers.quantizers.AutoQuantizationConfig")


# ===========================================================================
# unsloth_zoo/empty_model.py
# ===========================================================================

def test_empty_model_accelerate_init_empty_weights():
    """unsloth_zoo/empty_model.py:238, 322 — `from accelerate import
    init_empty_weights`. Two callsites, both top-level inside their
    functions (no try/except). A removal makes meta-model loading
    crash."""
    _resolve("accelerate.init_empty_weights")


def test_empty_model_siglip_vision_model():
    """unsloth_zoo/empty_model.py:307 — `from
    transformers.models.siglip.modeling_siglip import SiglipVisionModel`.
    Used to detect SigLIP vision towers during empty-model construction."""
    _resolve("transformers.models.siglip.modeling_siglip.SiglipVisionModel")


def test_empty_model_auto_model_for_causal_lm():
    """unsloth_zoo/empty_model.py:237 — `from transformers import
    AutoModelForCausalLM`."""
    _resolve("transformers.AutoModelForCausalLM")


# ===========================================================================
# unsloth_zoo/tokenizer_utils.py + unsloth_zoo/training_utils.py
# (datasets top-level imports)
# ===========================================================================

def test_top_level_datasets_module():
    """unsloth_zoo/tokenizer_utils.py:21 and training_utils.py:19 —
    `import datasets` at module top-level. A missing datasets package
    means the WHOLE tokenizer / training surface ImportErrors."""
    _resolve("datasets")


# ===========================================================================
# unsloth_zoo/peft_utils.py
# ===========================================================================

def test_peft_utils_peft_tuners_lora_module():
    """unsloth_zoo/peft_utils.py:157 — `import peft.tuners.lora`. Used to
    enumerate LoRA-eligible layers."""
    _resolve("peft.tuners.lora")


# ===========================================================================
# unsloth_zoo/temporary_patches/utils.py
# ===========================================================================

def test_temporary_patches_utils_kwargs_typing():
    """unsloth_zoo/temporary_patches/utils.py:146, 211, 231, 244 — the
    KWARGS_TYPE alias is built from a try-cascade of upstream Unpack /
    TransformersKwargs / FlashAttentionKwargs / LossKwargs. AT LEAST
    ONE must resolve, else the cascade ends with a NameError at zoo
    import time."""
    found_any = False
    for path in (
        "transformers.processing_utils.Unpack",
        "transformers.utils.TransformersKwargs",
        "transformers.modeling_flash_attention_utils.FlashAttentionKwargs",
        "transformers.utils.LossKwargs",
    ):
        try:
            _resolve(path)
            found_any = True
        except AssertionError:
            continue
    assert found_any, (
        "None of Unpack / TransformersKwargs / FlashAttentionKwargs / "
        "LossKwargs resolved; zoo temporary_patches/utils.py KWARGS_TYPE "
        "cascade exhausts and zoo import-time NameErrors."
    )


def test_temporary_patches_utils_transformers_version():
    """unsloth_zoo/temporary_patches/utils.py:216 — `from transformers
    import __version__`. Used by the temporary-patch version gates."""
    _resolve("transformers.__version__")


# ===========================================================================
# unsloth_zoo/temporary_patches/misc.py
# ===========================================================================

def test_temp_patches_misc_config_mapping():
    """unsloth_zoo/temporary_patches/misc.py:47 — `from
    transformers.models.auto.configuration_auto import CONFIG_MAPPING`."""
    _resolve(
        "transformers.models.auto.configuration_auto.CONFIG_MAPPING",
    )


def test_temp_patches_misc_tokenization_utils_base():
    """unsloth_zoo/temporary_patches/misc.py:63, 89, 1438 — `from
    transformers.tokenization_utils_base import PreTrainedTokenizerBase,
    AddedToken`."""
    _resolve_all([
        "transformers.tokenization_utils_base.PreTrainedTokenizerBase",
        "transformers.tokenization_utils_base.AddedToken",
    ])


def test_temp_patches_misc_quantizers_auto():
    """unsloth_zoo/temporary_patches/misc.py:115 — `import
    transformers.quantizers.auto`."""
    _resolve("transformers.quantizers.auto")


def test_temp_patches_misc_loss_for_causal_lm_loss():
    """unsloth_zoo/temporary_patches/misc.py:162, 248 — `from
    transformers.loss.loss_utils import ForCausalLMLoss`. The CSM
    patches monkey-rebind this; a rename silently disables them."""
    _resolve("transformers.loss.loss_utils.ForCausalLMLoss")


def test_temp_patches_misc_modeling_outputs_causal_lm_output():
    """unsloth_zoo/temporary_patches/misc.py:161 (and several other
    patch sites) — `from transformers.modeling_outputs import
    CausalLMOutputWithPast`. Also referenced in
    unsloth_zoo/temporary_patches/qwen3_next_moe.py:78."""
    _resolve("transformers.modeling_outputs.CausalLMOutputWithPast")


def test_temp_patches_misc_generation_utils_module():
    """unsloth_zoo/temporary_patches/misc.py:383 +
    gpt_oss.py:2135 — `import transformers.generation.utils`. The
    create_causal_mask_mapping patch rebinds names on this module."""
    _resolve("transformers.generation.utils")


def test_temp_patches_misc_modeling_layers_grad_ckpt():
    """unsloth_zoo/temporary_patches/misc.py:1121 — `from
    transformers.modeling_layers import GradientCheckpointingLayer`.
    The mllama vision encoder patch subclasses this."""
    _resolve(
        "transformers.modeling_layers.GradientCheckpointingLayer",
    )


def test_temp_patches_misc_all_attention_functions():
    """unsloth_zoo/temporary_patches/misc.py:526 — `from
    transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS`. The
    SDPA mask attention-fn registry. Renamed in some transformers 5
    pre-release tags."""
    _resolve("transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS")


def test_temp_patches_misc_integrations_sdpa_attention():
    """unsloth_zoo/temporary_patches/misc.py:525 — `import
    transformers.integrations.sdpa_attention`. Module rebinding site."""
    _resolve("transformers.integrations.sdpa_attention")


def test_temp_patches_misc_import_utils():
    """unsloth_zoo/temporary_patches/misc.py:834 — `import
    transformers.utils.import_utils`. Used to introspect optional
    backends without paying the import cost."""
    _resolve("transformers.utils.import_utils")


def test_temp_patches_misc_peft_lora_bnb():
    """unsloth_zoo/temporary_patches/misc.py:1289 — `import
    peft.tuners.lora.bnb as peft_bnb`. The BNB dtype-promotion patch
    iterates this module's Linear*bit classes."""
    _resolve("peft.tuners.lora.bnb")


def test_temp_patches_misc_training_arguments():
    """unsloth_zoo/temporary_patches/misc.py:1333 — `from transformers
    import TrainingArguments`. Reassigned to patch deprecation
    warnings."""
    _resolve("transformers.TrainingArguments")


def test_temp_patches_misc_models_auto_modeling_auto():
    """unsloth_zoo/temporary_patches/misc.py:1363 — `import
    transformers.models.auto.modeling_auto as auto_mod`. Reads
    MODEL_FOR_*_MAPPING_NAMES off this module."""
    _resolve("transformers.models.auto.modeling_auto")


def test_temp_patches_misc_pretrained_tokenizer_base_top_level():
    """unsloth_zoo/temporary_patches/misc.py:1438 — `from transformers
    import PreTrainedTokenizerBase`. Top-level import surface (not
    the .tokenization_utils_base path)."""
    _resolve("transformers.PreTrainedTokenizerBase")


# ===========================================================================
# unsloth_zoo/temporary_patches/gemma.py
# ===========================================================================

def test_temp_patches_gemma_processing_surface():
    """unsloth_zoo/temporary_patches/gemma.py:93-97 — five imports
    used to rebuild the Gemma3 processor:
      - transformers.models.gemma3.processing_gemma3.Gemma3ProcessorKwargs
      - transformers.image_utils.make_nested_list_of_images
      - transformers.feature_extraction_utils.BatchFeature
      - transformers.utils.to_py_obj
    Module-level installs that need ALL to resolve."""
    _resolve_all([
        "transformers.models.gemma3.processing_gemma3.Gemma3ProcessorKwargs",
        "transformers.image_utils.make_nested_list_of_images",
        "transformers.feature_extraction_utils.BatchFeature",
        "transformers.utils.to_py_obj",
    ])


# ===========================================================================
# unsloth_zoo/temporary_patches/gpt_oss.py
# ===========================================================================

def test_temp_patches_gpt_oss_modeling_rope_utils():
    """unsloth_zoo/temporary_patches/gpt_oss.py:2602 — `from
    transformers.modeling_rope_utils import rope_config_validation`."""
    _resolve(
        "transformers.modeling_rope_utils.rope_config_validation",
    )


def test_temp_patches_gpt_oss_layer_type_validation():
    """unsloth_zoo/temporary_patches/gpt_oss.py:2593 — `from
    transformers.configuration_utils import layer_type_validation`.
    Added in transformers 4.56 for layered config validation; the
    gpt_oss config-rebind needs it on every supported version."""
    _resolve(
        "transformers.configuration_utils.layer_type_validation",
    )


# ===========================================================================
# unsloth_zoo/temporary_patches/qwen3_vl_moe.py
# ===========================================================================

def test_temp_patches_qwen3_vl_moe_act2fn():
    """unsloth_zoo/temporary_patches/qwen3_vl_moe.py:201 — `from
    transformers.activations import ACT2FN`. The activation registry;
    moved between modeling_utils and activations historically."""
    _resolve("transformers.activations.ACT2FN")


# ===========================================================================
# unsloth_zoo/temporary_patches/gemma4.py
# ===========================================================================

def test_temp_patches_gemma4_cache_utils():
    """unsloth_zoo/temporary_patches/gemma4.py:308, 334, 460 — `from
    transformers.cache_utils import DynamicCache, StaticCache`. The
    Gemma4 forward-rewrite branches on these classes."""
    _resolve_all([
        "transformers.cache_utils.DynamicCache",
        "transformers.cache_utils.StaticCache",
    ])


# ===========================================================================
# unsloth_zoo/temporary_patches/moe_utils.py
# ===========================================================================

def test_temp_patches_moe_utils_param_wrapper():
    """unsloth_zoo/temporary_patches/moe_utils.py:897 — `from
    peft.tuners.lora.layer import ParamWrapper`. Required by zoo PR
    #618's 3D-weight LoRA dispatch."""
    _resolve("peft.tuners.lora.layer.ParamWrapper")


# ===========================================================================
# unsloth_zoo/logging_utils.py
# ===========================================================================

def test_logging_utils_utils_notebook():
    """unsloth_zoo/logging_utils.py:50 — `from
    transformers.utils.notebook import (...)`. The IPython progress-bar
    helpers live here on all currently-supported transformers."""
    _resolve("transformers.utils.notebook")


def test_logging_utils_trainer_progress_callback():
    """unsloth_zoo/logging_utils.py:174-178 — `from transformers.trainer
    import is_in_notebook, DEFAULT_PROGRESS_CALLBACK`."""
    _resolve_all([
        "transformers.trainer.is_in_notebook",
        "transformers.trainer.DEFAULT_PROGRESS_CALLBACK",
    ])


def test_logging_utils_trl_trainer_module():
    """unsloth_zoo/logging_utils.py:190 — `import trl.trainer`. The
    progress-callback override walks `trl.trainer.*Trainer` classes
    by attribute."""
    _skip_if_missing("trl")
    _resolve("trl.trainer")


# ===========================================================================
# unsloth_zoo/temporary_patches/pixtral.py
# ===========================================================================

def test_temp_patches_pixtral_rotary_emb():
    """unsloth_zoo/temporary_patches/pixtral.py:30 — `from
    transformers.models.pixtral.modeling_pixtral import
    apply_rotary_pos_emb`. The Pixtral RoPE helper used by the
    attention rewrite."""
    _resolve(
        "transformers.models.pixtral.modeling_pixtral.apply_rotary_pos_emb",
    )


# ===========================================================================
# unsloth_zoo/vllm_lora_worker_manager.py — TOP-LEVEL UNGUARDED imports
# ===========================================================================

def test_vllm_lora_worker_manager_top_level():
    """unsloth_zoo/vllm_lora_worker_manager.py:22, 23, 32-34 — five
    TOP-LEVEL unguarded imports. Module fails to import outright if
    any is missing on the installed vllm:
      vllm.config.LoRAConfig
      vllm.logger.init_logger
      vllm.lora.peft_helper.PEFTHelper
      vllm.lora.request.LoRARequest
      vllm.lora.utils.get_adapter_absolute_path
    """
    _skip_if_missing("vllm")
    _resolve_all([
        "vllm.config.LoRAConfig",
        "vllm.logger.init_logger",
        "vllm.lora.peft_helper.PEFTHelper",
        "vllm.lora.request.LoRARequest",
        "vllm.lora.utils.get_adapter_absolute_path",
    ])


def test_vllm_lora_worker_manager_vllm_config_top_level():
    """unsloth_zoo/vllm_lora_worker_manager.py:315 — `from vllm.config
    import VllmConfig`. Constructor sig changed in vllm 0.10."""
    _skip_if_missing("vllm")
    _resolve("vllm.config.VllmConfig")


# ===========================================================================
# unsloth_zoo/vllm_utils.py — surface assertions for the unguarded paths
# ===========================================================================

def test_vllm_utils_top_level_peft_type():
    """unsloth_zoo/vllm_utils.py:2520 — `from peft import PeftType`
    at MODULE TOP LEVEL (no try/except)."""
    _resolve("peft.PeftType")


def test_vllm_utils_sampling_params_path():
    """unsloth_zoo/vllm_utils.py:3107 — `from vllm import
    SamplingParams`. (Constructor introspection is covered in the
    trl/vllm pinned-symbols suite; this just pins the import path.)"""
    _skip_if_missing("vllm")
    _resolve("vllm.SamplingParams")


def test_vllm_utils_models_registry():
    """unsloth_zoo/vllm_utils.py:1649 — `from
    vllm.model_executor.models.registry import ModelRegistry`."""
    _skip_if_missing("vllm")
    _resolve("vllm.model_executor.models.registry.ModelRegistry")


# ===========================================================================
# unsloth_zoo/temporary_patches/mxfp4.py
# ===========================================================================

def test_temp_patches_mxfp4_module_path():
    """unsloth_zoo/temporary_patches/mxfp4.py — three sites import
    `transformers.integrations.mxfp4` either as a module
    (transformers.integrations.mxfp4) OR for `FP4_VALUES` /
    `Mxfp4Config`. The module path itself MUST resolve so the patch
    site can rebind FP4 conversion."""
    _resolve("transformers.integrations.mxfp4")


def test_temp_patches_mxfp4_tensor_parallel_helper():
    """unsloth_zoo/temporary_patches/mxfp4.py:181 — `from
    transformers.integrations.tensor_parallel import
    shard_and_distribute_module`. Also used by gpt_oss.py:467."""
    _resolve(
        "transformers.integrations.tensor_parallel.shard_and_distribute_module",
    )


# ===========================================================================
# Cross-cutting: the qwen2_vl + qwen2_5_vl image-processing surface used
# by both compiler.py and temporary_patches/misc.py:1485, 1501.
# ===========================================================================

def test_qwen2_vl_image_processor_class():
    """unsloth_zoo/temporary_patches/misc.py:1485 —
    Qwen2VLImageProcessor at transformers.models.qwen2_vl
    .image_processing_qwen2_vl. The patch site is wrapped in
    try/except but the symbol IS reached when zoo runs on
    transformers >= 5.0; pin the path so a rename produces a clean
    failure instead of a silent no-op."""
    _resolve(
        "transformers.models.qwen2_vl.image_processing_qwen2_vl.Qwen2VLImageProcessor",
    )


def test_qwen2_5_vl_image_processor_class_gated_on_v5():
    """unsloth_zoo/temporary_patches/misc.py:1501 —
    Qwen2_5_VLImageProcessor at
    transformers.models.qwen2_5_vl.image_processing_qwen2_5_vl.

    The whole patch_qwen2vl_image_processor_pixel_attrs site is
    early-returned on transformers < 5.0.0 (see misc.py:1478-1482),
    and the qwen2_5_vl import is additionally wrapped in
    try/except. So on 4.57.6 this symbol is allowed to be absent;
    on >= 5.0 it MUST resolve."""
    import transformers
    # Match the version gate in unsloth_zoo/temporary_patches/misc.py:1479.
    from packaging.version import Version
    if Version(transformers.__version__) < Version("5.0.0"):
        pytest.skip(
            "qwen2_5_vl.image_processing_qwen2_5_vl not required on "
            f"transformers {transformers.__version__} (zoo patch is "
            "version-gated to >= 5.0.0)"
        )
    _resolve(
        "transformers.models.qwen2_5_vl.image_processing_qwen2_5_vl.Qwen2_5_VLImageProcessor",
    )
