# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Signature pins for upstream symbols that unsloth_zoo patches, wraps or calls positionally; drift fails as DRIFT
DETECTED, while missing optional deps skip via importorskip."""

from __future__ import annotations

import inspect
from typing import Iterable

import importlib

import pytest

try:
    import transformers as _transformers
    from packaging.version import Version as _Version
    _TX_IS_5X = _Version(getattr(_transformers, "__version__", "0.0.0")) >= _Version("5.0.0")
    _TX_VERSION = getattr(_transformers, "__version__", "0.0.0")
except Exception:
    _TX_IS_5X = False
    _TX_VERSION = "unknown"


def _skip_if_transformers_5x(reason: str) -> None:
    """Skips on transformers 5.x, where the zoo patch wraps via a relaxed **kwargs match so runtime still works; the check stays strict on 4.57.6."""
    if _TX_IS_5X:
        pytest.skip(
            f"transformers {_TX_VERSION}: {reason} (zoo patch silently "
            "no-ops via relaxed patch_function)"
        )


def _names_the_target(exc: ModuleNotFoundError, dotted_module: str) -> bool:
    """exc.name is the deepest missing package (the parent for a removed model package), so any package prefix of the
    target counts; the trailing dot keeps siglip from matching siglipx."""
    name = exc.name or ""
    return bool(name) and (dotted_module == name or dotted_module.startswith(name + "."))


def _import_or_skip(dotted_module: str, *names):
    """Imports names from dotted_module, skipping when an existing module fails on a dependency (e.g. timm in gemma3n)
    rather than reporting drift; a genuinely removed module still raises, as drift."""
    try:
        mod = importlib.import_module(dotted_module)
    except ModuleNotFoundError as exc:
        if _names_the_target(exc, dotted_module):
            raise
        pytest.skip(
            f"{dotted_module!r} raised on import, so nothing can be said about "
            f"signature drift here: {type(exc).__name__}: {exc}"
        )
    except Exception as exc:
        pytest.skip(
            f"{dotted_module!r} raised on import, so nothing can be said about "
            f"signature drift here: {type(exc).__name__}: {exc}"
        )
    got = tuple(getattr(mod, name) for name in names)
    return got[0] if len(got) == 1 else got


def _param_names(func) -> list[str]:
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError) as exc:
        pytest.fail(
            f"DRIFT DETECTED: cannot inspect signature of {func!r}: {exc}"
        )
    return [name for name in sig.parameters.keys()]


def _assert_params_superset(
    func,
    required: Iterable[str],
    zoo_callsite: str,
):
    """Asserts every name in required is a parameter of func; upstream may add params but must not drop one zoo forwards by name."""
    got = _param_names(func)
    missing = [name for name in required if name not in got]
    if missing:
        pytest.fail(
            f"DRIFT DETECTED: {zoo_callsite}: "
            f"zoo forwards by-name params {sorted(missing)} but installed "
            f"{func!r} signature is {got}"
        )


def _assert_positional_arity_at_least(
    func,
    arity: int,
    zoo_callsite: str,
):
    """Asserts func accepts at least arity non-self positionals, catching a positional dropped upstream (e.g. in super().forward(...))."""
    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    if params and params[0].name in ("self", "cls"):
        params = params[1:]
    positional = 0
    for p in params:
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                      inspect.Parameter.POSITIONAL_ONLY):
            positional += 1
        elif p.kind is inspect.Parameter.VAR_POSITIONAL:
            return  # *args -> unlimited
    if positional < arity:
        pytest.fail(
            f"DRIFT DETECTED: {zoo_callsite}: zoo calls with >= {arity} "
            f"positional args but installed {func!r} only accepts "
            f"{positional} positional ({[p.name for p in params]})"
        )


# Module-level importorskip: one clean skip if transformers is missing.
pytest.importorskip("transformers")


# transformers.modeling_utils.checkpoint (gradient_checkpointing.py:232/234/246)

def test_torch_checkpoint_function_first_positional_arg():
    """gradient_checkpointing.py:222 assigns unsloth_gradient_checkpoint to torch.utils.checkpoint.checkpoint; upstream
    must keep function as first positional and use_reentrant as kwarg."""
    import torch.utils.checkpoint as tuc
    sig = inspect.signature(tuc.checkpoint)
    params = list(sig.parameters.keys())
    if not params or params[0] != "function":
        pytest.fail(
            f"DRIFT DETECTED: torch.utils.checkpoint.checkpoint: zoo "
            f"unsloth_gradient_checkpoint(function, *args, use_reentrant) "
            f"expects first positional to be 'function' but got {params}"
        )
    if "use_reentrant" not in params:
        pytest.fail(
            f"DRIFT DETECTED: torch.utils.checkpoint.checkpoint: zoo "
            f"unsloth_gradient_checkpoint takes use_reentrant kwarg but "
            f"installed signature dropped it: {params}"
        )


def test_transformers_modeling_utils_checkpoint_symbol_present():
    """gradient_checkpointing.py:234/246/924 sets transformers.modeling_utils.checkpoint; a removed re-export silently
    no-ops the patch, causing a long-context VRAM bug."""
    import transformers.modeling_utils as mu
    if not hasattr(mu, "checkpoint"):
        pytest.fail(
            "DRIFT DETECTED: transformers.modeling_utils.checkpoint: "
            "symbol removed upstream. zoo monkey-patches this attribute "
            "in gradient_checkpointing.py:232/246/924. Patch is now a no-op."
        )


# transformers.integrations.bitsandbytes._replace_with_bnb_linear

def test_replace_with_bnb_linear_signature():
    """patching_utils.py:682 ``inspect.getsource(_replace_with_bnb_linear)``
    + source rewrite -> re-installed as ``_unsloth_replace_with_bnb_linear``.
    Pins ``(model, modules_to_not_convert, current_key_name,
    quantization_config, has_been_replaced)``."""
    pytest.importorskip("bitsandbytes")
    try:
        from transformers.integrations.bitsandbytes import (
            _replace_with_bnb_linear,
        )
    except ImportError:
        # transformers 5.x removed this private symbol; zoo guards it at patching_utils.py:678 and falls back to should_convert_module, which must exist.
        try:
            import transformers.quantizers.quantizers_utils as qu  # noqa
        except ImportError:
            pytest.fail(
                "DRIFT DETECTED: neither "
                "transformers.integrations.bitsandbytes._replace_with_bnb_linear "
                "nor transformers.quantizers.quantizers_utils is importable. "
                "patching_utils.py:678-783 has no fallback path."
            )
        return
    _assert_params_superset(
        _replace_with_bnb_linear,
        required=[
            "model",
            "modules_to_not_convert",
            "current_key_name",
            "quantization_config",
        ],
        zoo_callsite="patching_utils.py:682 inspect.getsource(_replace_with_bnb_linear) + rewrite",
    )


# transformers.modeling_utils.PreTrainedModel.loss_function (loss_utils.py:145)

def test_pretrained_model_loss_function_exists():
    """loss_utils.py:143-146 unwraps PreTrainedModel.loss_function.fget.__wrapped__; removing the property silently aborts the patch, so no fused CE."""
    import transformers.modeling_utils as mu
    if not hasattr(mu.PreTrainedModel, "loss_function"):
        pytest.fail(
            "DRIFT DETECTED: transformers.modeling_utils.PreTrainedModel.loss_function: "
            "attribute removed upstream. loss_utils.py:143 patch silently aborts."
        )


def test_LOSS_MAPPING_ForCausalLM_signature_compatible():
    """loss_utils.py:140 sets LOSS_MAPPING['ForCausalLM'] to UnslothForCausalLMLoss, which takes (logits, labels, vocab_size, ...) as its first three positionals."""
    from transformers.loss.loss_utils import LOSS_MAPPING
    if "ForCausalLM" not in LOSS_MAPPING:
        pytest.fail(
            "DRIFT DETECTED: transformers.loss.loss_utils.LOSS_MAPPING: "
            "'ForCausalLM' key removed. loss_utils.py:140 monkey-patch no-ops."
        )
    upstream = LOSS_MAPPING["ForCausalLM"]
    _assert_params_superset(
        upstream,
        required=["logits", "labels", "vocab_size"],
        zoo_callsite="loss_utils.py:113 UnslothForCausalLMLoss positional contract",
    )


def test_fixed_cross_entropy_signature():
    """loss_utils.py:99 UnslothFixedCrossEntropy keyword-forwards num_items_in_batch and ignore_index; fixed_cross_entropy must keep both params."""
    from transformers.loss.loss_utils import fixed_cross_entropy
    _assert_params_superset(
        fixed_cross_entropy,
        required=["num_items_in_batch", "ignore_index"],
        zoo_callsite="loss_utils.py:99 unsloth_fixed_cross_entropy forwards "
                     "num_items_in_batch and ignore_index by name",
    )


# transformers Trainer (training_utils.py:354-355 and compiler.py:4040)

def test_Trainer_get_optimizer_cls_and_kwargs_signature():
    """training_utils.py:354 calls Trainer.get_optimizer_cls_and_kwargs(training_args) with one positional argument."""
    from transformers import Trainer
    _assert_positional_arity_at_least(
        Trainer.get_optimizer_cls_and_kwargs,
        arity=1,
        zoo_callsite="training_utils.py:354 Trainer.get_optimizer_cls_and_kwargs(training_args)",
    )


def test_Trainer_get_decay_parameter_names_signature():
    """training_utils.py:355 calls Trainer.get_decay_parameter_names(None, model), passing self=None plus a model positional."""
    from transformers import Trainer
    sig = inspect.signature(Trainer.get_decay_parameter_names)
    params = list(sig.parameters.keys())
    if "model" not in params:
        pytest.fail(
            f"DRIFT DETECTED: Trainer.get_decay_parameter_names: zoo "
            f"training_utils.py:355 passes a model positionally as second "
            f"arg, but installed signature is {params}"
        )


def test_Trainer_inner_training_loop_signature_preserved():
    """compiler.py:3966-4040 replaces Trainer._inner_training_loop with _fast_inner_training_loop via inspect.getsource; pins params the rewriter assumes exist."""
    from transformers import Trainer
    _assert_params_superset(
        Trainer._inner_training_loop,
        required=[
            "self",
            "batch_size",
            "args",
            "resume_from_checkpoint",
            "trial",
            "ignore_keys_for_eval",
        ],
        zoo_callsite="compiler.py:3966-4040 Trainer._inner_training_loop rewrite",
    )


# transformers.set_seed / get_scheduler / seed_worker / DataCollator*

def test_set_seed_signature():
    """training_utils.py:20 -- first positional must be ``seed``."""
    from transformers import set_seed
    sig = inspect.signature(set_seed)
    params = list(sig.parameters.keys())
    if not params or params[0] != "seed":
        pytest.fail(
            f"DRIFT DETECTED: transformers.set_seed: zoo uses positional "
            f"seed arg, but installed signature is {params}"
        )


def test_get_scheduler_signature():
    """training_utils.py:377 calls transformers_get_scheduler with name, optimizer, num_warmup_steps and num_training_steps as keywords."""
    from transformers import get_scheduler
    _assert_params_superset(
        get_scheduler,
        required=["name", "optimizer", "num_warmup_steps", "num_training_steps"],
        zoo_callsite="training_utils.py:377 transformers_get_scheduler(name, optimizer, "
                     "num_warmup_steps, num_training_steps)",
    )


def test_seed_worker_imported_as_trainer_utils_seed_worker():
    """training_utils.py:23 imports seed_worker from transformers.trainer_utils with no fallback."""
    try:
        from transformers.trainer_utils import seed_worker  # noqa: F401
    except ImportError as exc:
        pytest.fail(
            f"DRIFT DETECTED: transformers.trainer_utils.seed_worker "
            f"import failed: {exc}. training_utils.py:23 has no fallback."
        )


def test_DataCollatorForLanguageModeling_signature():
    """training_utils.py:346 and dataset_utils.py:686 construct DataCollatorForLanguageModeling(tokenizer=..., mlm=False, pad_to_multiple_of=4)."""
    from transformers import DataCollatorForLanguageModeling
    _assert_params_superset(
        DataCollatorForLanguageModeling.__init__,
        required=["tokenizer", "mlm"],
        zoo_callsite="training_utils.py:346 + dataset_utils.py:686 + 838 "
                     "DataCollatorForLanguageModeling(tokenizer, mlm, pad_to_multiple_of)",
    )


def test_DataCollatorForSeq2Seq_signature():
    """dataset_utils.py:464 / 678 -- ``DataCollatorForSeq2Seq(tokenizer=...)``."""
    from transformers import DataCollatorForSeq2Seq
    _assert_params_superset(
        DataCollatorForSeq2Seq.__init__,
        required=["tokenizer"],
        zoo_callsite="dataset_utils.py:464 / 678 DataCollatorForSeq2Seq(tokenizer)",
    )


# TrainingArguments (temporary_patches/misc.py:1334)

def test_TrainingArguments_to_dict_signature():
    """temporary_patches/misc.py:1334-1343 wraps TrainingArguments.to_dict with _patched_to_dict(self) and does not forward args, so upstream must stay self-only."""
    from transformers import TrainingArguments
    sig = inspect.signature(TrainingArguments.to_dict)
    params = list(sig.parameters.keys())
    if params != ["self"]:
        pytest.fail(
            f"DRIFT DETECTED: TrainingArguments.to_dict: zoo wrapper at "
            f"temporary_patches/misc.py:1337 is `def _patched_to_dict(self)` "
            f"with NO *args/**kwargs forwarding, but installed signature "
            f"is {params}. Any caller-passed kwarg is silently dropped."
        )


def test_TrainingArguments_get_warmup_steps_signature():
    """training_utils.py:380 -- ``training_args.get_warmup_steps(max_steps)``."""
    from transformers import TrainingArguments
    _assert_positional_arity_at_least(
        TrainingArguments.get_warmup_steps,
        arity=1,
        zoo_callsite="training_utils.py:380 training_args.get_warmup_steps(max_steps)",
    )


# PretrainedConfig (patching_utils.py:244-273)

def test_PretrainedConfig_to_dict_signature():
    """patching_utils.py:256-259 wraps PretrainedConfig.to_dict with *args/**kwargs; to_dict and to_diff_dict must remain methods."""
    try:
        from transformers.configuration_utils import PreTrainedConfig as Cfg
    except ImportError:
        from transformers.configuration_utils import PretrainedConfig as Cfg
    if not hasattr(Cfg, "to_dict"):
        pytest.fail(
            "DRIFT DETECTED: PretrainedConfig.to_dict: method removed. "
            "patching_utils.py:253 wrap target gone."
        )
    if not hasattr(Cfg, "to_diff_dict"):
        pytest.fail(
            "DRIFT DETECTED: PretrainedConfig.to_diff_dict: method removed. "
            "patching_utils.py:254 wrap target gone."
        )


# PushToHubMixin.push_to_hub (saving_utils.py:76)

def test_PushToHubMixin_push_to_hub_signature():
    """saving_utils.py:76 uses PushToHubMixin as a mixin base; pins the push_to_hub kwargs zoo expects (repo_id, commit_message, token, private, revision)."""
    from transformers.modeling_utils import PushToHubMixin
    _assert_params_superset(
        PushToHubMixin.push_to_hub,
        required=["repo_id", "commit_message", "token", "private", "revision"],
        zoo_callsite="saving_utils.py + utilities calling PushToHubMixin.push_to_hub",
    )


# accelerate.init_empty_weights (empty_model.py:238, 322)

def test_accelerate_init_empty_weights_signature():
    """empty_model.py:252 and 329 call init_empty_weights(include_buffers=False); pins that kwarg name."""
    pytest.importorskip("accelerate")
    from accelerate import init_empty_weights
    _assert_params_superset(
        init_empty_weights,
        required=["include_buffers"],
        zoo_callsite="empty_model.py:252 with init_empty_weights(include_buffers=False)",
    )


# Masking-utils + GPT-OSS overrides

def test_masking_utils_create_causal_mask_signature():
    """gpt_oss.py:2178-2182 wraps transformers.masking_utils.create_causal_mask with *args/**kwargs, so only the symbol itself must exist."""
    try:
        from transformers.masking_utils import create_causal_mask  # noqa
    except ImportError as exc:
        pytest.fail(
            f"DRIFT DETECTED: transformers.masking_utils.create_causal_mask "
            f"removed: {exc}. gpt_oss.py:2178 wrap target gone."
        )


def test_masking_utils_create_sliding_window_causal_mask_signature():
    """gpt_oss.py:2179 + ministral.py also depend on this being importable."""
    try:
        from transformers.masking_utils import (
            create_sliding_window_causal_mask,  # noqa
        )
    except ImportError as exc:
        pytest.fail(
            f"DRIFT DETECTED: "
            f"transformers.masking_utils.create_sliding_window_causal_mask "
            f"removed: {exc}. gpt_oss.py:2179 + ministral.py wrap target gone."
        )


def test_masking_utils_create_masks_for_generate_signature():
    """gpt_oss.py:2184-2185 wraps transformers.masking_utils.create_masks_for_generate and its re-export in transformers.generation.utils."""
    try:
        from transformers.masking_utils import create_masks_for_generate
    except ImportError as exc:
        pytest.fail(
            f"DRIFT DETECTED: "
            f"transformers.masking_utils.create_masks_for_generate "
            f"removed: {exc}. gpt_oss.py:2184 wrap target gone."
        )
    import transformers.generation.utils as gu
    if not hasattr(gu, "create_masks_for_generate"):
        pytest.fail(
            "DRIFT DETECTED: "
            "transformers.generation.utils.create_masks_for_generate "
            "re-export missing. gpt_oss.py:2185 patch silently no-ops."
        )


# Gemma3 forward / norm / mlp overrides (temporary_patches/gemma.py)

def test_gemma3_apply_rotary_pos_emb_signature():
    """gemma.py:399 calls apply_rotary_pos_emb(query_states, key_states, cos, sin) with 4 positionals."""
    from transformers.models.gemma3.modeling_gemma3 import apply_rotary_pos_emb
    _assert_positional_arity_at_least(
        apply_rotary_pos_emb,
        arity=4,
        zoo_callsite="gemma.py:399+639 apply_rotary_pos_emb(q, k, cos, sin)",
    )


def test_gemma3_eager_attention_forward_signature():
    """gemma.py:399 and ministral.py:38: the relaxed-mode patch passes module, query, key, value, attention_mask, dropout and scaling."""
    from transformers.models.gemma3.modeling_gemma3 import eager_attention_forward
    _assert_params_superset(
        eager_attention_forward,
        required=["module", "query", "key", "value", "attention_mask"],
        zoo_callsite="gemma.py + ministral.py eager_attention_forward forward chain",
    )


def test_gemma3_ALL_ATTENTION_FUNCTIONS_present():
    """gemma.py:399 / ministral.py:39 -- ``from gemma3.modeling_gemma3
    import ALL_ATTENTION_FUNCTIONS``. Zoo subscripts it via
    ``ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]``."""
    from transformers.models.gemma3.modeling_gemma3 import ALL_ATTENTION_FUNCTIONS  # noqa
    if not hasattr(ALL_ATTENTION_FUNCTIONS, "__getitem__"):
        pytest.fail(
            f"DRIFT DETECTED: gemma3.ALL_ATTENTION_FUNCTIONS: zoo subscripts "
            f"this object via ``ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]`` "
            f"but installed type {type(ALL_ATTENTION_FUNCTIONS)} has no __getitem__"
        )


def test_Gemma3Processor_call_signature():
    """gemma.py:224 patches Gemma3Processor.__call__(self, images, text, videos, audio, **kwargs) with match_level='relaxed'."""
    from transformers.models.gemma3.processing_gemma3 import Gemma3Processor
    _assert_params_superset(
        Gemma3Processor.__call__,
        required=["images", "text"],
        zoo_callsite="gemma.py:224 Gemma3Processor.__call__ patch",
    )


def test_Gemma3RMSNorm_forward_signature():
    """gemma.py:361 / 628 patches ``Gemma3RMSNorm.forward(self,
    hidden_states)`` with fullgraph=True. One-tensor forward."""
    from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm
    sig = inspect.signature(Gemma3RMSNorm.forward)
    params = [p.name for p in sig.parameters.values() if p.name != "self"]
    if len(params) != 1:
        pytest.fail(
            f"DRIFT DETECTED: Gemma3RMSNorm.forward: zoo replacement at "
            f"gemma.py:361/628 takes (self, hidden_states) but installed "
            f"signature has params {params}"
        )


def test_Gemma3MLP_forward_signature():
    """gemma.py:389 patches Gemma3MLP.forward with a single-tensor forward."""
    from transformers.models.gemma3.modeling_gemma3 import Gemma3MLP
    sig = inspect.signature(Gemma3MLP.forward)
    params = [p.name for p in sig.parameters.values() if p.name != "self"]
    if len(params) != 1:
        pytest.fail(
            f"DRIFT DETECTED: Gemma3MLP.forward: zoo replacement at "
            f"gemma.py:389 takes (self, x) but installed signature has "
            f"params {params}"
        )


def test_Gemma3TextScaledWordEmbedding_forward_signature():
    """gemma.py:331 patches Gemma3TextScaledWordEmbedding.forward(self, input_ids)."""
    from transformers.models.gemma3.modeling_gemma3 import (
        Gemma3TextScaledWordEmbedding,
    )
    sig = inspect.signature(Gemma3TextScaledWordEmbedding.forward)
    params = [p.name for p in sig.parameters.values() if p.name != "self"]
    if len(params) != 1:
        pytest.fail(
            f"DRIFT DETECTED: Gemma3TextScaledWordEmbedding.forward: zoo "
            f"replacement at gemma.py:331 takes (self, input_ids) but "
            f"installed signature has params {params}"
        )


def test_Gemma3Attention_forward_signature():
    """gemma.py:607 and 849 patch Gemma3Attention.forward via patch_function_past_key_values with match_level='relaxed'."""
    from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention
    _assert_params_superset(
        Gemma3Attention.forward,
        required=["hidden_states", "position_embeddings", "attention_mask"],
        zoo_callsite="gemma.py:607/849 Gemma3Attention.forward patch",
    )


# Gemma3n overrides (temporary_patches/gemma3n.py)

def test_Gemma3nMultimodalEmbedder_forward_signature():
    """gemma3n.py:88 patches Gemma3nMultimodalEmbedder.forward(self, input_ids, inputs_embeds)."""
    Gemma3nMultimodalEmbedder = _import_or_skip(
        "transformers.models.gemma3n.modeling_gemma3n", "Gemma3nMultimodalEmbedder",
    )
    _assert_params_superset(
        Gemma3nMultimodalEmbedder.forward,
        required=["input_ids", "inputs_embeds"],
        zoo_callsite="gemma3n.py:88 Gemma3nMultimodalEmbedder.forward patch",
    )


def test_Gemma3nTextAltUp_predict_signature():
    """gemma3n.py:122 patches Gemma3nTextAltUp.predict(self, hidden_states)."""
    Gemma3nTextAltUp = _import_or_skip(
        "transformers.models.gemma3n.modeling_gemma3n", "Gemma3nTextAltUp",
    )
    sig = inspect.signature(Gemma3nTextAltUp.predict)
    params = [p.name for p in sig.parameters.values() if p.name != "self"]
    if "hidden_states" not in params:
        pytest.fail(
            f"DRIFT DETECTED: Gemma3nTextAltUp.predict: zoo replacement at "
            f"gemma3n.py:122 takes (self, hidden_states) but installed "
            f"signature has params {params}"
        )


def test_Gemma3nTextAltUp_correct_signature():
    """gemma3n.py:146 patches Gemma3nTextAltUp.correct(self, predictions, activated)."""
    Gemma3nTextAltUp = _import_or_skip(
        "transformers.models.gemma3n.modeling_gemma3n", "Gemma3nTextAltUp",
    )
    _assert_params_superset(
        Gemma3nTextAltUp.correct,
        required=["predictions", "activated"],
        zoo_callsite="gemma3n.py:146 Gemma3nTextAltUp.correct patch",
    )


def test_Gemma3nModel_get_placeholder_mask_signature():
    """gemma3n.py:201 patches Gemma3nModel.get_placeholder_mask with match_level='relaxed'."""
    Gemma3nModel = _import_or_skip(
        "transformers.models.gemma3n.modeling_gemma3n", "Gemma3nModel",
    )
    _assert_params_superset(
        Gemma3nModel.get_placeholder_mask,
        required=["input_ids", "inputs_embeds"],
        zoo_callsite="gemma3n.py:201 Gemma3nModel.get_placeholder_mask patch",
    )


# Ministral overrides (temporary_patches/ministral.py)

def test_MinistralAttention_forward_signature():
    """ministral.py:99 patches MinistralAttention.forward (relaxed wrap hides its signature); probes hidden_states,
    position_embeddings, attention_mask on the stashed _original_modeling_ministral_MinistralAttention_forward; if
    absent, probe skipped, runtime still works via kwargs."""
    try:
        from transformers.models.ministral.modeling_ministral import (
            MinistralAttention,
        )
    except ImportError:
        pytest.skip("transformers.models.ministral not installed (added in 4.57)")
    stash_attr = "_original_modeling_ministral_MinistralAttention_forward"
    candidate = getattr(MinistralAttention, stash_attr, MinistralAttention.forward)
    candidate_params = list(inspect.signature(candidate).parameters.keys())
    if candidate_params == ["self", "args", "kwargs"]:
        pytest.skip(
            "MinistralAttention.forward is zoo's relaxed (self, *args, "
            "**kwargs) wrapper and no _original_ stash is available on "
            "this run; the wrapper forwards via kwargs so the named-"
            "param contract is enforced at runtime, not via "
            "inspect.signature"
        )
    _assert_params_superset(
        candidate,
        required=["hidden_states", "position_embeddings", "attention_mask"],
        zoo_callsite="ministral.py:99 MinistralAttention.forward patch",
    )


def test_MinistralModel_forward_signature():
    """ministral.py:179 patches MinistralModel.forward with match_level='relaxed'; zoo forwards input_ids,
    attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, cache_position by name; skipped on 5.x,
    where cache_position moved into **kwargs."""
    _skip_if_transformers_5x(
        "MinistralModel.forward moved cache_position into "
        "**kwargs: Unpack[TransformersKwargs]"
    )
    try:
        from transformers.models.ministral.modeling_ministral import (
            MinistralModel,
        )
    except ImportError:
        pytest.skip("transformers.models.ministral not installed (added in 4.57)")
    _assert_params_superset(
        MinistralModel.forward,
        required=[
            "input_ids",
            "attention_mask",
            "position_ids",
            "past_key_values",
            "inputs_embeds",
            "use_cache",
            "cache_position",
        ],
        zoo_callsite="ministral.py:179 MinistralModel.forward patch",
    )


def test_ministral_apply_rotary_pos_emb_signature():
    """ministral.py:37 calls apply_rotary_pos_emb(query_states, key_states, cos, sin) with 4 positionals."""
    try:
        from transformers.models.ministral.modeling_ministral import (
            apply_rotary_pos_emb,
        )
    except ImportError:
        pytest.skip("transformers.models.ministral not installed")
    _assert_positional_arity_at_least(
        apply_rotary_pos_emb,
        arity=4,
        zoo_callsite="ministral.py:61 apply_rotary_pos_emb(q, k, cos, sin)",
    )


# GPT-OSS class-level monkey-patches (temporary_patches/gpt_oss.py)

def test_GptOssExperts_class_present_and_init_takes_config():
    """gpt_oss.py:1060, 1070, 1849 and 1858 monkey-patch GptOssExperts; zoo's override defines __init__(self, config)."""
    try:
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts
    except ImportError:
        pytest.skip("transformers.models.gpt_oss not installed")
    _assert_params_superset(
        GptOssExperts.__init__,
        required=["config"],
        zoo_callsite="gpt_oss.py:1070 transformers...GptOssExperts replacement (self, config)",
    )


def test_GptOssExperts_forward_signature():
    """gpt_oss.py:1845 and 1852 replace GptOssExperts.forward(self, hidden_states, router_indices=None, routing_weights=None)."""
    try:
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts
    except ImportError:
        pytest.skip("transformers.models.gpt_oss not installed")
    _assert_params_superset(
        GptOssExperts.forward,
        required=["hidden_states", "router_indices", "routing_weights"],
        zoo_callsite="gpt_oss.py:1845/1852 GptOssExperts.forward replacement",
    )


def test_GptOssTopKRouter_present():
    """gpt_oss.py:1062/1077 monkey-patches ``GptOssTopKRouter``."""
    try:
        from transformers.models.gpt_oss.modeling_gpt_oss import (
            GptOssTopKRouter,
        )
    except ImportError:
        pytest.skip("transformers.models.gpt_oss not installed")
    _assert_params_superset(
        GptOssTopKRouter.__init__,
        required=["config"],
        zoo_callsite="gpt_oss.py:1077 GptOssTopKRouter replacement (self, config)",
    )


def test_GptOssAttention_forward_signature():
    """gpt_oss.py patches GptOssAttention.forward(self, hidden_states, position_embeddings, attention_mask, ...)."""
    try:
        from transformers.models.gpt_oss.modeling_gpt_oss import (
            GptOssAttention,
        )
    except ImportError:
        pytest.skip("transformers.models.gpt_oss not installed")
    _assert_params_superset(
        GptOssAttention.forward,
        required=["hidden_states", "position_embeddings", "attention_mask"],
        zoo_callsite="gpt_oss.py GptOssAttention.forward patch",
    )


def test_GptOssModel_forward_signature():
    """gpt_oss.py:2481 patches GptOssModel.forward, match_level='relaxed'."""
    try:
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssModel
    except ImportError:
        pytest.skip("transformers.models.gpt_oss not installed")
    _assert_params_superset(
        GptOssModel.forward,
        required=[
            "input_ids",
            "attention_mask",
            "position_ids",
            "past_key_values",
            "inputs_embeds",
        ],
        zoo_callsite="gpt_oss.py:2481 GptOssModel.forward patch",
    )


def test_GptOssPreTrainedModel_init_weights_signature():
    """gpt_oss.py:2853 patches ``GptOssPreTrainedModel._init_weights``."""
    try:
        from transformers.models.gpt_oss.modeling_gpt_oss import (
            GptOssPreTrainedModel,
        )
    except ImportError:
        pytest.skip("transformers.models.gpt_oss not installed")
    _assert_params_superset(
        GptOssPreTrainedModel._init_weights,
        required=["module"],
        zoo_callsite="gpt_oss.py:2853 _init_weights patch (self, module)",
    )


# Mxfp4 integrations (temporary_patches/gpt_oss.py)

def test_mxfp4_swizzle_mxfp4_signature():
    """gpt_oss.py:190 patches transformers.integrations.mxfp4.swizzle_mxfp4 with match_level='relaxed'."""
    try:
        from transformers.integrations.mxfp4 import swizzle_mxfp4
    except (ImportError, AttributeError):
        pytest.skip("transformers.integrations.mxfp4.swizzle_mxfp4 not available")
    _assert_params_superset(
        swizzle_mxfp4,
        required=["w", "w_scale"],
        zoo_callsite="gpt_oss.py:190 swizzle_mxfp4 patch",
    )


def test_mxfp4_load_and_swizzle_mxfp4_signature():
    """gpt_oss.py:540 patches load_and_swizzle_mxfp4 with match_level='relaxed'."""
    try:
        from transformers.integrations.mxfp4 import load_and_swizzle_mxfp4
    except (ImportError, AttributeError):
        pytest.skip("transformers.integrations.mxfp4.load_and_swizzle_mxfp4 not available")
    _assert_params_superset(
        load_and_swizzle_mxfp4,
        required=["module", "param_name", "param_value"],
        zoo_callsite="gpt_oss.py:540 load_and_swizzle_mxfp4 patch",
    )


def test_mxfp4_replace_with_mxfp4_linear_signature():
    """gpt_oss.py:569 patches ``replace_with_mxfp4_linear``."""
    try:
        from transformers.integrations.mxfp4 import replace_with_mxfp4_linear
    except (ImportError, AttributeError):
        pytest.skip("transformers.integrations.mxfp4.replace_with_mxfp4_linear not available")
    _assert_params_superset(
        replace_with_mxfp4_linear,
        required=["model", "modules_to_not_convert", "quantization_config"],
        zoo_callsite="gpt_oss.py:569 replace_with_mxfp4_linear patch",
    )


def test_mxfp4_mlp_forward_signature():
    """gpt_oss.py:454 patches ``mlp_forward`` -- (self, hidden_states)."""
    try:
        from transformers.integrations.mxfp4 import mlp_forward
    except (ImportError, AttributeError):
        pytest.skip("transformers.integrations.mxfp4.mlp_forward not available")
    _assert_params_superset(
        mlp_forward,
        required=["hidden_states"],
        zoo_callsite="gpt_oss.py:454 mlp_forward patch",
    )


# AutoHfQuantizer.merge_quantization_configs (misc.py:153)

def test_AutoHfQuantizer_merge_quantization_configs_signature():
    """misc.py:153 patches AutoHfQuantizer.merge_quantization_configs(quantization_config, quantization_config_from_args)."""
    from transformers.quantizers.auto import AutoHfQuantizer
    _assert_params_superset(
        AutoHfQuantizer.merge_quantization_configs,
        required=["quantization_config", "quantization_config_from_args"],
        zoo_callsite="misc.py:153 AutoHfQuantizer.merge_quantization_configs patch",
    )


# Granitemoehybrid + CSM (misc.py:1061 / 770)

def test_GraniteMoeHybridMambaLayer_cuda_kernels_forward_signature():
    """misc.py:1061 patches GraniteMoeHybridMambaLayer.cuda_kernels_forward(self, hidden_states, cache_params,
    cache_position, attention_mask, seq_idx); skipped on 5.x."""
    _skip_if_transformers_5x(
        "GraniteMoeHybridMambaLayer.cuda_kernels_forward moved cache_position "
        "into **kwargs: Unpack[TransformersKwargs]"
    )
    try:
        from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
            GraniteMoeHybridMambaLayer,
        )
    except ImportError:
        pytest.skip("transformers.models.granitemoehybrid not installed")
    _assert_params_superset(
        GraniteMoeHybridMambaLayer.cuda_kernels_forward,
        required=["hidden_states", "cache_params", "cache_position", "attention_mask"],
        zoo_callsite="misc.py:1061 GraniteMoeHybridMambaLayer.cuda_kernels_forward patch",
    )


def test_CsmForConditionalGeneration_merge_input_ids_signature():
    """misc.py:770 patches CsmForConditionalGeneration._merge_input_ids_with_input_values(self, input_ids, input_values, input_values_cutoffs, labels)."""
    try:
        from transformers.models.csm.modeling_csm import (
            CsmForConditionalGeneration,
        )
    except ImportError:
        pytest.skip("transformers.models.csm not installed")
    _assert_params_superset(
        CsmForConditionalGeneration._merge_input_ids_with_input_values,
        required=["input_ids", "input_values", "input_values_cutoffs", "labels"],
        zoo_callsite="misc.py:770 CsmForConditionalGeneration."
                     "_merge_input_ids_with_input_values patch",
    )


# Mllama vision encoder layer (misc.py:1172)

def test_MllamaVisionEncoderLayer_forward_signature():
    """misc.py:1146-1172 uses MllamaVisionEncoderLayer.forward(self, hidden_state, attention_mask=None); upstream's name is hidden_state, not hidden_states."""
    try:
        from transformers.models.mllama.modeling_mllama import (
            MllamaVisionEncoderLayer,
        )
    except ImportError:
        pytest.skip("transformers.models.mllama not installed")
    got = _param_names(MllamaVisionEncoderLayer.forward)
    if "hidden_state" not in got and "hidden_states" not in got:
        pytest.fail(
            f"DRIFT DETECTED: MllamaVisionEncoderLayer.forward: zoo "
            f"replacement at misc.py:1146 takes (self, hidden_state, "
            f"attention_mask) but installed has neither 'hidden_state' nor "
            f"'hidden_states' in {got}"
        )


# Siglip encoder layer (misc.py:1228)

def test_SiglipEncoderLayer_forward_signature():
    """misc.py:1187-1228 patch body still references output_attentions, so removing it from
    SiglipEncoderLayer.forward(self, hidden_states, attention_mask, output_attentions=False) breaks it."""
    from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer
    _assert_params_superset(
        SiglipEncoderLayer.forward,
        required=["hidden_states", "attention_mask"],
        zoo_callsite="misc.py:1187-1228 SiglipEncoderLayer.forward patch",
    )


# Qwen3 MoE (qwen3_moe / qwen3_vl_moe / qwen3_next_moe)

def test_Qwen3MoeSparseMoeBlock_forward_signature():
    """qwen3_moe.py patches Qwen3MoeSparseMoeBlock.forward(self, hidden_states)."""
    try:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            Qwen3MoeSparseMoeBlock,
        )
    except ImportError:
        pytest.skip("transformers.models.qwen3_moe not installed")
    _assert_params_superset(
        Qwen3MoeSparseMoeBlock.forward,
        required=["hidden_states"],
        zoo_callsite="qwen3_moe.py Qwen3MoeSparseMoeBlock.forward patch",
    )


def test_Qwen3VLMoeTextSparseMoeBlock_forward_signature():
    """qwen3_vl_moe.py:362-383 patches Qwen3VLMoeTextSparseMoeBlock.forward(self, hidden_states)."""
    try:
        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
            Qwen3VLMoeTextSparseMoeBlock,
        )
    except ImportError:
        pytest.skip("transformers.models.qwen3_vl_moe not installed")
    _assert_params_superset(
        Qwen3VLMoeTextSparseMoeBlock.forward,
        required=["hidden_states"],
        zoo_callsite="qwen3_vl_moe.py:362 Qwen3VLMoeTextSparseMoeBlock.forward patch",
    )


def test_Qwen3VLMoeTextExperts_forward_signature():
    """qwen3_vl_moe.py:376 patches Qwen3VLMoeTextExperts.forward; zoo's replacement renames params, so only positional arity (3 after self) is pinned."""
    try:
        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
            Qwen3VLMoeTextExperts,
        )
    except ImportError:
        pytest.skip("transformers.models.qwen3_vl_moe not installed")
    _assert_positional_arity_at_least(
        Qwen3VLMoeTextExperts.forward,
        arity=3,
        zoo_callsite="qwen3_vl_moe.py:376 Qwen3VLMoeTextExperts.forward patch "
                     "(3 positional after self)",
    )


def test_Qwen3VLMoeTextExperts_init_signature():
    """qwen3_vl_moe.py:242 patches Qwen3VLMoeTextExperts.__init__(self, config)."""
    try:
        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
            Qwen3VLMoeTextExperts,
        )
    except ImportError:
        pytest.skip("transformers.models.qwen3_vl_moe not installed")
    _assert_params_superset(
        Qwen3VLMoeTextExperts.__init__,
        required=["config"],
        zoo_callsite="qwen3_vl_moe.py:242 Qwen3VLMoeTextExperts.__init__ patch",
    )


def test_Qwen3NextSparseMoeBlock_forward_signature():
    """qwen3_next_moe.py:67 patches ``Qwen3NextSparseMoeBlock.forward``."""
    try:
        from transformers.models.qwen3_next.modeling_qwen3_next import (
            Qwen3NextSparseMoeBlock,
        )
    except ImportError:
        pytest.skip("transformers.models.qwen3_next not installed")
    _assert_params_superset(
        Qwen3NextSparseMoeBlock.forward,
        required=["hidden_states"],
        zoo_callsite="qwen3_next_moe.py:67 Qwen3NextSparseMoeBlock.forward patch",
    )


# Deepseek-V3 MoE (deepseek_v3_moe.py)

def test_DeepseekV3MoE_forward_signature():
    """deepseek_v3_moe.py:125 patches DeepseekV3MoE.forward(self, hidden_states)."""
    try:
        from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
            DeepseekV3MoE,
        )
    except ImportError:
        pytest.skip("transformers.models.deepseek_v3 not installed")
    _assert_params_superset(
        DeepseekV3MoE.forward,
        required=["hidden_states"],
        zoo_callsite="deepseek_v3_moe.py:125 DeepseekV3MoE.forward patch",
    )


def test_DeepseekV3ForCausalLM_forward_signature():
    """deepseek_v3_moe.py:213 patches DeepseekV3ForCausalLM.forward; zoo forwards input_ids, attention_mask,
    position_ids, past_key_values, inputs_embeds, labels, use_cache by name."""
    try:
        from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
            DeepseekV3ForCausalLM,
        )
    except ImportError:
        pytest.skip("transformers.models.deepseek_v3 not installed")
    _assert_params_superset(
        DeepseekV3ForCausalLM.forward,
        required=[
            "input_ids",
            "attention_mask",
            "position_ids",
            "past_key_values",
            "inputs_embeds",
            "labels",
            "use_cache",
        ],
        zoo_callsite="deepseek_v3_moe.py:213 DeepseekV3ForCausalLM.forward patch",
    )


# PEFT (temporary_patches/misc.py:1281 dispatch_bnb_4bit wrap)

def test_peft_dispatch_bnb_4bit_signature():
    """misc.py:1297 wraps peft.tuners.lora.bnb.dispatch_bnb_4bit with safe_dispatch_bnb_4bit(target, adapter_name,
    **kwargs); upstream must keep the two positionals and **kwargs."""
    pytest.importorskip("peft")
    try:
        import peft.tuners.lora.bnb as peft_bnb
        dispatch_bnb_4bit = peft_bnb.dispatch_bnb_4bit
    except (ImportError, AttributeError) as exc:
        pytest.fail(
            f"DRIFT DETECTED: peft.tuners.lora.bnb.dispatch_bnb_4bit "
            f"removed: {exc}. misc.py:1281 wrap target gone."
        )
    _assert_params_superset(
        dispatch_bnb_4bit,
        required=["target", "adapter_name"],
        zoo_callsite="misc.py:1297 safe_dispatch_bnb_4bit(target, adapter_name, **kwargs)",
    )


def test_peft_Linear4bit_importable():
    """patching_utils.py:313 imports peft.tuners.lora Linear4bit as Peft_Linear4bit for isinstance checks."""
    pytest.importorskip("peft")
    try:
        from peft.tuners.lora import Linear4bit  # noqa
    except ImportError as exc:
        pytest.fail(
            f"DRIFT DETECTED: peft.tuners.lora.Linear4bit import: {exc}. "
            f"patching_utils.py:313 hard import."
        )


def test_peft_get_peft_model_signature():
    """peft.get_peft_model is the primary attach point after get_peft_regex in peft_utils.py; it must accept model and peft_config as positionals."""
    pytest.importorskip("peft")
    from peft import get_peft_model
    _assert_positional_arity_at_least(
        get_peft_model,
        arity=2,
        zoo_callsite="peft_utils.py get_peft_regex output -> get_peft_model(model, peft_config)",
    )


# Cache utilities (gemma4.py, qwen3_moe etc)

def test_DynamicCache_importable():
    """gemma4.py:308 and 460 import DynamicCache and StaticCache from transformers.cache_utils; zoo instantiates them with zero args."""
    from transformers.cache_utils import DynamicCache, StaticCache  # noqa
    if not callable(DynamicCache):
        pytest.fail("DRIFT DETECTED: DynamicCache is no longer callable.")
    if not callable(StaticCache):
        pytest.fail("DRIFT DETECTED: StaticCache is no longer callable.")


# Bitsandbytes patch (bitsandbytes.py:108)

def test_bnb_Linear4bit_forward_signature():
    """bitsandbytes.py:108 patches bitsandbytes.nn.modules.Linear4bit.forward(self, x)."""
    bitsandbytes = pytest.importorskip("bitsandbytes")
    Linear4bit = getattr(bitsandbytes.nn.modules, "Linear4bit", None)
    if Linear4bit is None:
        pytest.fail(
            "DRIFT DETECTED: bitsandbytes.nn.modules.Linear4bit removed. "
            "bitsandbytes.py:108 patch target gone."
        )
    _assert_positional_arity_at_least(
        Linear4bit.forward,
        arity=1,
        zoo_callsite="bitsandbytes.py:108 Linear4bit.forward(self, x) patch",
    )


# vllm (vllm_utils.py + temporary_patches/misc.py:1402)

def test_vllm_SamplingParams_constructor():
    """vllm_utils.py's grpo_update_SamplingParams filters kwargs by inspect.signature(vllm.SamplingParams).parameters;
    an *args-only shape silently drops every kwarg."""
    vllm = pytest.importorskip("vllm")
    SamplingParams = getattr(vllm, "SamplingParams", None)
    if SamplingParams is None:
        pytest.fail(
            "DRIFT DETECTED: vllm.SamplingParams removed. "
            "vllm_utils.py grpo_update_SamplingParams target gone."
        )
    sig = inspect.signature(SamplingParams)
    if "temperature" not in sig.parameters and "top_p" not in sig.parameters:
        pytest.fail(
            f"DRIFT DETECTED: vllm.SamplingParams: zoo expects standard "
            f"sampling kwargs (temperature/top_p) but installed signature "
            f"has only {list(sig.parameters.keys())}"
        )
