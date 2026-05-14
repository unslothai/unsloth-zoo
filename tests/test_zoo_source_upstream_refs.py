# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team.
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at
# your option) any later version.

"""Importable-symbol pins for upstream references in ``unsloth_zoo`` source.

Flat enumeration of every ``from <upstream> import <symbol>`` /
``<upstream>.X.Y`` reference visible in ``unsloth_zoo/**.py``, exercised
against the INSTALLED versions of transformers / trl / peft / datasets
/ accelerate / vllm. Each test cites the zoo file:line it pins.
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
    """``importlib.import_module`` + ``getattr`` chain.

    DRIFT-DETECTED policy: any failure to resolve is reported as an
    AssertionError -- never a SKIP. Three failure modes all surface as
    DRIFT: module-file missing, module-file present but import raises
    (transitively-broken optional dep), or attribute missing on a
    successfully-imported module.
    """
    parts = dotted.split(".")
    obj: object = None
    consumed: list[str] = []
    last_import_error: Exception | None = None
    for i in range(len(parts), 0, -1):
        mod_name = ".".join(parts[:i])
        # Probe metadata first; this NEVER executes module code.
        try:
            spec = importlib.util.find_spec(mod_name)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            continue
        try:
            obj = importlib.import_module(mod_name)
            consumed = parts[:i]
            break
        except ImportError as exc:
            last_import_error = exc
            raise AssertionError(
                f"DRIFT DETECTED: `{mod_name}` exists but its imports "
                f"fail on this install ({type(exc).__name__}: {exc}). "
                "zoo references this dotted path -- a transitively-"
                "missing dep here is exactly the regression class this "
                "suite catches. Either install the dep in CI or remove "
                "the zoo reference."
            )
    if obj is None:
        raise AssertionError(
            f"DRIFT DETECTED: could not locate any module prefix of "
            f"`{dotted}`; zoo references this dotted path -- regression "
            "at the import line (see source comment above the test)."
            + (f" Last ImportError: {last_import_error!r}"
               if last_import_error is not None else "")
        )
    for attr in parts[len(consumed):]:
        if not hasattr(obj, attr):
            walked = ".".join(consumed + [attr])
            raise AssertionError(
                f"DRIFT DETECTED: `{walked}` missing on installed "
                f"upstream (walked from `{dotted}`); zoo references "
                "this exact path -- a rename or removal silently "
                "breaks the zoo patch site cited in the test comment."
            )
        obj = getattr(obj, attr)
        consumed.append(attr)
    return obj


def _resolve_all(dotted_paths: Iterable[str]) -> None:
    """Resolve every dotted path; collect misses into one AssertionError."""
    missing: list[str] = []
    for d in dotted_paths:
        try:
            _resolve(d)
        except AssertionError as e:
            missing.append(f"  - {d}\n      ({e})")
    assert not missing, "Missing upstream symbols:\n" + "\n".join(missing)


def _skip_if_missing(module_name: str) -> None:
    pytest.importorskip(module_name)


# ===========================================================================
# unsloth_zoo/compiler.py
# ===========================================================================

def test_compiler_modeling_flash_attention_utils_top_level():
    """unsloth_zoo/compiler.py:218 -- TOP-LEVEL unguarded
    ``from transformers.modeling_flash_attention_utils import
    is_flash_attn_available``; a removal ImportErrors ``import unsloth_zoo``."""
    _resolve_all([
        "transformers.modeling_flash_attention_utils",
        "transformers.modeling_flash_attention_utils.is_flash_attn_available",
    ])


def test_compiler_masking_utils():
    """unsloth_zoo/compiler.py:372 -- ``import transformers.masking_utils``."""
    _resolve("transformers.masking_utils")


def test_compiler_transformers_logging():
    """unsloth_zoo/compiler.py:3145 -- ``from transformers import logging``."""
    _resolve("transformers.logging")


def test_compiler_generation_mixin():
    """unsloth_zoo/compiler.py:3781 -- ``from transformers.generation import
    GenerationMixin``."""
    _resolve("transformers.generation.GenerationMixin")


def test_compiler_trainer_module_and_class():
    """unsloth_zoo/compiler.py:3963, 3975 -- ``from transformers.trainer
    import Trainer`` AND ``import transformers.trainer``."""
    _resolve_all([
        "transformers.trainer",
        "transformers.trainer.Trainer",
    ])


# ===========================================================================
# unsloth_zoo/loss_utils.py
# ===========================================================================

def test_loss_utils_training_args_parallel_mode():
    """unsloth_zoo/loss_utils.py:232 -- TOP-LEVEL unguarded ``from
    transformers.training_args import ParallelMode``; a removal silently
    breaks distributed loss aggregation."""
    _resolve_all([
        "transformers.training_args",
        "transformers.training_args.ParallelMode",
    ])


def test_loss_utils_modeling_utils():
    """unsloth_zoo/loss_utils.py:138 -- ``import transformers.modeling_utils``
    feeds the LOSS_MAPPING rebind path."""
    _resolve("transformers.modeling_utils")


def test_loss_utils_loss_module():
    """unsloth_zoo/loss_utils.py:82 -- ``import transformers.loss.loss_utils``.
    The loss-helper subpackage moved into transformers 4.50."""
    _resolve("transformers.loss.loss_utils")


# ===========================================================================
# unsloth_zoo/training_utils.py
# ===========================================================================

def test_training_utils_top_level_transformers_surface():
    """unsloth_zoo/training_utils.py:20-23 -- four top-level imports; any
    removal makes ``from unsloth_zoo import ...`` blow up at every site
    depending on training_utils."""
    _resolve_all([
        "transformers.set_seed",
        "transformers.get_scheduler",
        "transformers.Trainer",
        "transformers.trainer_utils.seed_worker",
    ])


def test_training_utils_data_collator_for_lm():
    """unsloth_zoo/training_utils.py:345 -- ``from transformers import
    DataCollatorForLanguageModeling``."""
    _resolve("transformers.DataCollatorForLanguageModeling")


def test_training_utils_peft_modules_to_save_wrapper():
    """unsloth_zoo/training_utils.py:239 -- ``from peft.utils import
    ModulesToSaveWrapper``. Identifies non-LoRA trainable adapter weights
    for the saving path."""
    _resolve("peft.utils.ModulesToSaveWrapper")


# ===========================================================================
# unsloth_zoo/dataset_utils.py
# ===========================================================================

def test_dataset_utils_datasets_top_level():
    """unsloth_zoo/dataset_utils.py:594 -- ``from datasets import (Dataset,
    IterableDataset,)`` at module top-level."""
    _resolve_all(["datasets.Dataset", "datasets.IterableDataset"])


def test_dataset_utils_data_collator_for_seq2seq():
    """unsloth_zoo/dataset_utils.py:457, 672 -- ``from transformers import
    DataCollatorForSeq2Seq``."""
    _resolve("transformers.DataCollatorForSeq2Seq")


# ===========================================================================
# unsloth_zoo/saving_utils.py
# ===========================================================================

def test_saving_utils_pushtohubmixin():
    """unsloth_zoo/saving_utils.py:76 -- TOP-LEVEL unguarded ``from
    transformers.modeling_utils import PushToHubMixin``. Calls
    ``._upload_modified_files`` / ``._get_files_timestamps``."""
    _resolve("transformers.modeling_utils.PushToHubMixin")


def test_saving_utils_peft_top_level():
    """unsloth_zoo/saving_utils.py:82 + 270 -- TOP-LEVEL unguarded
    ``from peft import PeftModelForCausalLM, PeftModel`` and ``from
    peft.utils.integrations import dequantize_module_weight``."""
    _resolve_all([
        "peft.PeftModelForCausalLM",
        "peft.PeftModel",
        "peft.utils.integrations.dequantize_module_weight",
    ])


def test_saving_utils_autoconfig():
    """unsloth_zoo/saving_utils.py:2101 -- ``from transformers import
    AutoConfig``."""
    _resolve("transformers.AutoConfig")


# ===========================================================================
# unsloth_zoo/patching_utils.py
# ===========================================================================

def test_patching_utils_pretrainedconfig_either_name():
    """unsloth_zoo/patching_utils.py:247-251 -- try ``PreTrainedConfig``
    (5.x) then ``PretrainedConfig`` (4.x). Zoo gates both with
    try/except, so only one needs to resolve."""
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
    """unsloth_zoo/patching_utils.py:313 -- ``from peft.tuners.lora import
    Linear4bit as Peft_Linear4bit``. The 4-bit LoRA layer zoo's
    dtype/dequant patch keys on."""
    _resolve("peft.tuners.lora.Linear4bit")


def test_patching_utils_integrations_bitsandbytes_module():
    """unsloth_zoo/patching_utils.py:677 -- ``import
    transformers.integrations.bitsandbytes``. Used at IMPORT TIME
    (top-level) to introspect _replace_with_bnb_linear."""
    _resolve("transformers.integrations.bitsandbytes")


def test_patching_utils_quantizers_utils_module():
    """unsloth_zoo/patching_utils.py:761 -- ``import
    transformers.quantizers.quantizers_utils``. Top-level on transformers 5.x."""
    _resolve("transformers.quantizers.quantizers_utils")


# ===========================================================================
# unsloth_zoo/hf_utils.py
# ===========================================================================

def test_hf_utils_pretrainedconfig_either_name():
    """unsloth_zoo/hf_utils.py:25-28 -- try ``PreTrainedConfig`` (5.x),
    fall back to ``PretrainedConfig`` (4.x). At least one must exist on
    top-level ``transformers``."""
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
    """unsloth_zoo/hf_utils.py:322, 363, 372 -- ``AutoTokenizer`` and
    ``AutoProcessor`` drive zoo's ``unsloth_tokenizer_from_pretrained``."""
    _resolve_all([
        "transformers.AutoTokenizer",
        "transformers.AutoProcessor",
    ])


def test_hf_utils_processor_mapping_names():
    """unsloth_zoo/hf_utils.py:278 -- ``from
    transformers.models.auto.processing_auto import PROCESSOR_MAPPING_NAMES``.
    Enumerates VLM processors."""
    _resolve(
        "transformers.models.auto.processing_auto.PROCESSOR_MAPPING_NAMES",
    )


def test_hf_utils_peft_config_top_level():
    """unsloth_zoo/hf_utils.py:119, 314 -- ``from peft import PeftConfig``.
    Two callsites, both try/except, same symbol."""
    _resolve("peft.PeftConfig")


# ===========================================================================
# unsloth_zoo/utils.py
# ===========================================================================

def test_utils_auto_quantization_config():
    """unsloth_zoo/utils.py:197 -- ``from transformers.quantizers import
    AutoQuantizationConfig``."""
    _resolve("transformers.quantizers.AutoQuantizationConfig")


# ===========================================================================
# unsloth_zoo/empty_model.py
# ===========================================================================

def test_empty_model_accelerate_init_empty_weights():
    """unsloth_zoo/empty_model.py:238, 322 -- ``from accelerate import
    init_empty_weights``. Two callsites, no try/except; removal makes
    meta-model loading crash."""
    _resolve("accelerate.init_empty_weights")


def test_empty_model_siglip_vision_model():
    """unsloth_zoo/empty_model.py:307 -- ``from
    transformers.models.siglip.modeling_siglip import SiglipVisionModel``."""
    _resolve("transformers.models.siglip.modeling_siglip.SiglipVisionModel")


def test_empty_model_auto_model_for_causal_lm():
    """unsloth_zoo/empty_model.py:237 -- ``from transformers import
    AutoModelForCausalLM``."""
    _resolve("transformers.AutoModelForCausalLM")


# ===========================================================================
# unsloth_zoo/tokenizer_utils.py + unsloth_zoo/training_utils.py
# ===========================================================================

def test_top_level_datasets_module():
    """unsloth_zoo/tokenizer_utils.py:21, training_utils.py:19 --
    ``import datasets`` at module top-level."""
    _resolve("datasets")


# ===========================================================================
# unsloth_zoo/peft_utils.py
# ===========================================================================

def test_peft_utils_peft_tuners_lora_module():
    """unsloth_zoo/peft_utils.py:157 -- ``import peft.tuners.lora``.
    Enumerates LoRA-eligible layers."""
    _resolve("peft.tuners.lora")


# ===========================================================================
# unsloth_zoo/temporary_patches/utils.py
# ===========================================================================

def test_temporary_patches_utils_kwargs_typing():
    """unsloth_zoo/temporary_patches/utils.py:146, 211, 231, 244 --
    KWARGS_TYPE is built from a try-cascade of upstream Unpack /
    TransformersKwargs / FlashAttentionKwargs / LossKwargs. At least one
    must resolve, else cascade ends with NameError at zoo import."""
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
    """unsloth_zoo/temporary_patches/utils.py:216 -- ``from transformers
    import __version__`` drives version gates."""
    _resolve("transformers.__version__")


# ===========================================================================
# unsloth_zoo/temporary_patches/misc.py
# ===========================================================================

def test_temp_patches_misc_config_mapping():
    """unsloth_zoo/temporary_patches/misc.py:47 -- ``from
    transformers.models.auto.configuration_auto import CONFIG_MAPPING``."""
    _resolve(
        "transformers.models.auto.configuration_auto.CONFIG_MAPPING",
    )


def test_temp_patches_misc_tokenization_utils_base():
    """unsloth_zoo/temporary_patches/misc.py:63, 89, 1438 -- ``from
    transformers.tokenization_utils_base import PreTrainedTokenizerBase,
    AddedToken``."""
    _resolve_all([
        "transformers.tokenization_utils_base.PreTrainedTokenizerBase",
        "transformers.tokenization_utils_base.AddedToken",
    ])


def test_temp_patches_misc_quantizers_auto():
    """unsloth_zoo/temporary_patches/misc.py:115 -- ``import
    transformers.quantizers.auto``."""
    _resolve("transformers.quantizers.auto")


def test_temp_patches_misc_loss_for_causal_lm_loss():
    """unsloth_zoo/temporary_patches/misc.py:162, 248 -- ``from
    transformers.loss.loss_utils import ForCausalLMLoss``. CSM patches
    monkey-rebind this; a rename silently disables them."""
    _resolve("transformers.loss.loss_utils.ForCausalLMLoss")


def test_temp_patches_misc_modeling_outputs_causal_lm_output():
    """unsloth_zoo/temporary_patches/misc.py:161 + qwen3_next_moe.py:78
    -- ``from transformers.modeling_outputs import CausalLMOutputWithPast``."""
    _resolve("transformers.modeling_outputs.CausalLMOutputWithPast")


def test_temp_patches_misc_generation_utils_module():
    """unsloth_zoo/temporary_patches/misc.py:383 + gpt_oss.py:2135 --
    ``import transformers.generation.utils``. The
    create_causal_mask_mapping patch rebinds names on this module."""
    _resolve("transformers.generation.utils")


def test_temp_patches_misc_modeling_layers_grad_ckpt():
    """unsloth_zoo/temporary_patches/misc.py:1121 -- ``from
    transformers.modeling_layers import GradientCheckpointingLayer``.
    Mllama vision encoder patch subclasses this."""
    _resolve(
        "transformers.modeling_layers.GradientCheckpointingLayer",
    )


def test_temp_patches_misc_all_attention_functions():
    """unsloth_zoo/temporary_patches/misc.py:526 -- ``from
    transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS``.
    Renamed in some transformers 5 pre-release tags."""
    _resolve("transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS")


def test_temp_patches_misc_integrations_sdpa_attention():
    """unsloth_zoo/temporary_patches/misc.py:525 -- ``import
    transformers.integrations.sdpa_attention``. Module rebinding site."""
    _resolve("transformers.integrations.sdpa_attention")


def test_temp_patches_misc_import_utils():
    """unsloth_zoo/temporary_patches/misc.py:834 -- ``import
    transformers.utils.import_utils``."""
    _resolve("transformers.utils.import_utils")


def test_temp_patches_misc_peft_lora_bnb():
    """unsloth_zoo/temporary_patches/misc.py:1289 -- ``import
    peft.tuners.lora.bnb as peft_bnb``. BNB dtype-promotion patch."""
    _resolve("peft.tuners.lora.bnb")


def test_temp_patches_misc_training_arguments():
    """unsloth_zoo/temporary_patches/misc.py:1333 -- ``from transformers
    import TrainingArguments``. Reassigned to patch deprecation warnings."""
    _resolve("transformers.TrainingArguments")


def test_temp_patches_misc_models_auto_modeling_auto():
    """unsloth_zoo/temporary_patches/misc.py:1363 -- ``import
    transformers.models.auto.modeling_auto as auto_mod``. Reads
    MODEL_FOR_*_MAPPING_NAMES off this module."""
    _resolve("transformers.models.auto.modeling_auto")


def test_temp_patches_misc_pretrained_tokenizer_base_top_level():
    """unsloth_zoo/temporary_patches/misc.py:1438 -- ``from transformers
    import PreTrainedTokenizerBase`` (top-level surface)."""
    _resolve("transformers.PreTrainedTokenizerBase")


# ===========================================================================
# unsloth_zoo/temporary_patches/gemma.py
# ===========================================================================

def test_temp_patches_gemma_processing_surface():
    """unsloth_zoo/temporary_patches/gemma.py:93-97 -- five module-level
    imports used to rebuild the Gemma3 processor; all must resolve."""
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
    """unsloth_zoo/temporary_patches/gpt_oss.py:2602 -- ``from
    transformers.modeling_rope_utils import rope_config_validation``."""
    _resolve(
        "transformers.modeling_rope_utils.rope_config_validation",
    )


def test_temp_patches_gpt_oss_layer_type_validation():
    """unsloth_zoo/temporary_patches/gpt_oss.py:2593 -- ``from
    transformers.configuration_utils import layer_type_validation``
    (added in transformers 4.56)."""
    _resolve(
        "transformers.configuration_utils.layer_type_validation",
    )


# ===========================================================================
# unsloth_zoo/temporary_patches/qwen3_vl_moe.py
# ===========================================================================

def test_temp_patches_qwen3_vl_moe_act2fn():
    """unsloth_zoo/temporary_patches/qwen3_vl_moe.py:201 -- ``from
    transformers.activations import ACT2FN``. Moved between modeling_utils
    and activations historically."""
    _resolve("transformers.activations.ACT2FN")


# ===========================================================================
# unsloth_zoo/temporary_patches/gemma4.py
# ===========================================================================

def test_temp_patches_gemma4_cache_utils():
    """unsloth_zoo/temporary_patches/gemma4.py:308, 334, 460 -- ``from
    transformers.cache_utils import DynamicCache, StaticCache``. Gemma4
    forward-rewrite branches on these classes."""
    _resolve_all([
        "transformers.cache_utils.DynamicCache",
        "transformers.cache_utils.StaticCache",
    ])


# ===========================================================================
# unsloth_zoo/temporary_patches/moe_utils.py
# ===========================================================================

def test_temp_patches_moe_utils_param_wrapper():
    """unsloth_zoo/temporary_patches/moe_utils.py:897 -- ``from
    peft.tuners.lora.layer import ParamWrapper``. Required by zoo PR
    #618's 3D-weight LoRA dispatch."""
    _resolve("peft.tuners.lora.layer.ParamWrapper")


# ===========================================================================
# unsloth_zoo/logging_utils.py
# ===========================================================================

def test_logging_utils_utils_notebook():
    """unsloth_zoo/logging_utils.py:50 -- ``from
    transformers.utils.notebook import (...)``. IPython progress-bar
    helpers."""
    _resolve("transformers.utils.notebook")


def test_logging_utils_trainer_progress_callback():
    """unsloth_zoo/logging_utils.py:174-178 -- ``from transformers.trainer
    import is_in_notebook, DEFAULT_PROGRESS_CALLBACK``."""
    _resolve_all([
        "transformers.trainer.is_in_notebook",
        "transformers.trainer.DEFAULT_PROGRESS_CALLBACK",
    ])


def test_logging_utils_trl_trainer_module():
    """unsloth_zoo/logging_utils.py:190 -- ``import trl.trainer``. The
    progress-callback override walks ``trl.trainer.*Trainer`` classes."""
    _skip_if_missing("trl")
    _resolve("trl.trainer")


# ===========================================================================
# unsloth_zoo/temporary_patches/pixtral.py
# ===========================================================================

def test_temp_patches_pixtral_rotary_emb():
    """unsloth_zoo/temporary_patches/pixtral.py:30 -- ``from
    transformers.models.pixtral.modeling_pixtral import
    apply_rotary_pos_emb``. Used by the attention rewrite."""
    _resolve(
        "transformers.models.pixtral.modeling_pixtral.apply_rotary_pos_emb",
    )


# ===========================================================================
# unsloth_zoo/vllm_lora_worker_manager.py
# ===========================================================================

def test_vllm_lora_worker_manager_top_level():
    """unsloth_zoo/vllm_lora_worker_manager.py:22, 23, 32-34 -- five
    TOP-LEVEL unguarded imports; module fails to import if any is missing:
      vllm.config.LoRAConfig, vllm.logger.init_logger,
      vllm.lora.peft_helper.PEFTHelper, vllm.lora.request.LoRARequest,
      vllm.lora.utils.get_adapter_absolute_path."""
    _skip_if_missing("vllm")
    _resolve_all([
        "vllm.config.LoRAConfig",
        "vllm.logger.init_logger",
        "vllm.lora.peft_helper.PEFTHelper",
        "vllm.lora.request.LoRARequest",
        "vllm.lora.utils.get_adapter_absolute_path",
    ])


def test_vllm_lora_worker_manager_vllm_config_top_level():
    """unsloth_zoo/vllm_lora_worker_manager.py:315 -- ``from vllm.config
    import VllmConfig``. Constructor sig changed in vllm 0.10."""
    _skip_if_missing("vllm")
    _resolve("vllm.config.VllmConfig")


# ===========================================================================
# unsloth_zoo/vllm_utils.py
# ===========================================================================

def test_vllm_utils_top_level_peft_type():
    """unsloth_zoo/vllm_utils.py:2520 -- ``from peft import PeftType`` at
    MODULE TOP LEVEL (no try/except)."""
    _resolve("peft.PeftType")


def test_vllm_utils_sampling_params_path():
    """unsloth_zoo/vllm_utils.py:3107 -- ``from vllm import SamplingParams``."""
    _skip_if_missing("vllm")
    _resolve("vllm.SamplingParams")


def test_vllm_utils_models_registry():
    """unsloth_zoo/vllm_utils.py:1649 -- ``from
    vllm.model_executor.models.registry import ModelRegistry``."""
    _skip_if_missing("vllm")
    _resolve("vllm.model_executor.models.registry.ModelRegistry")


# ===========================================================================
# unsloth_zoo/temporary_patches/mxfp4.py
# ===========================================================================

def test_temp_patches_mxfp4_module_path():
    """unsloth_zoo/temporary_patches/mxfp4.py -- three sites import
    ``transformers.integrations.mxfp4``. The module path must resolve so
    the patch site can rebind FP4 conversion."""
    _resolve("transformers.integrations.mxfp4")


def test_temp_patches_mxfp4_tensor_parallel_helper():
    """unsloth_zoo/temporary_patches/mxfp4.py:181 + gpt_oss.py:467 --
    ``from transformers.integrations.tensor_parallel import
    shard_and_distribute_module``."""
    _resolve(
        "transformers.integrations.tensor_parallel.shard_and_distribute_module",
    )


# ===========================================================================
# qwen2_vl + qwen2_5_vl image-processing surface
# ===========================================================================

def test_qwen2_vl_image_processor_class():
    """unsloth_zoo/temporary_patches/misc.py:1485 -- Qwen2VLImageProcessor.
    The patch site is wrapped in try/except, but the symbol IS reached on
    transformers >= 5.0; pin the path so a rename produces a clean
    failure instead of a silent no-op."""
    _resolve(
        "transformers.models.qwen2_vl.image_processing_qwen2_vl.Qwen2VLImageProcessor",
    )


def test_qwen2_5_vl_image_processor_class_gated_on_v5():
    """unsloth_zoo/temporary_patches/misc.py:1501 --
    Qwen2_5_VLImageProcessor. Version-gated on transformers >= 5.0.0
    (misc.py:1478-1482)."""
    import transformers
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
