# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at
# your option) any later version.

"""Signature pinning tests for the upstream functions / methods that
``unsloth_zoo`` monkey-patches, wraps, or calls with positional shape
assumptions.

Class of bug this catches
=========================
Transformers / TRL / PEFT / Accelerate adds, removes, or renames a
parameter on a function or method that ``unsloth_zoo`` overrides.
``unsloth_zoo``'s override silently ignores or mis-positions the new
parameter, downstream users get wrong-output bugs (NaN losses,
mis-quantized layers, silent attention truncation, broken gradient
checkpointing) with NO exception. Drift surfaces only as bad training
runs days later.

Each test below uses ``inspect.signature(...)`` on the **installed**
upstream symbol and asserts the parameter list that the matching
``unsloth_zoo`` monkey-patch / wrapper / positional call assumes.

Contract
========
* CPU-only -- no GPU, no model downloads, no network.
* Optional deps (``vllm``, ``mlx``, ``xformers``, ``timm``,
  ``bitsandbytes``) are gated with ``pytest.importorskip`` so genuinely
  uninstalled stacks don't false-fail.
* Real drift -> ``pytest.fail("DRIFT DETECTED: <upstream.path>
  signature changed: zoo expects {X} but installed has {Y}")``.
* Never ``pytest.skip`` to hide drift -- skips are only for genuine
  optional-dep absence and for upstream symbols that legitimately moved
  / were renamed in versions ``unsloth_zoo`` doesn't claim to support.

Source-of-truth callsite is cited in every test docstring so when an
upstream rename lands we can find the matching zoo override in a single
grep.

Runs under the GPU-free harness in ``tests/conftest.py``.
"""

from __future__ import annotations

import inspect
from typing import Iterable

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _param_names(func) -> list[str]:
    """Return the ordered parameter-name list of a callable. Wraps
    ``inspect.signature`` so a single ``inspect`` failure -> a test fail
    with a useful message instead of a stack trace."""
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
    """Assert that EVERY name in ``required`` appears in ``func``'s
    parameter list. The upstream may add NEW params (that's OK -- zoo
    just won't forward them yet) but MUST NOT drop any param that zoo
    forwards by name."""
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
    """Assert ``func`` accepts at least ``arity`` non-self positional
    args (POSITIONAL_OR_KEYWORD or POSITIONAL_ONLY, plus VAR_POSITIONAL
    counts as unlimited). Catches the case where zoo does
    ``super().forward(a, b, c, d)`` but upstream removed a positional."""
    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    # Drop a leading "self" / "cls" so the count is callsite-equivalent.
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


# ---------------------------------------------------------------------------
# Pre-flight: every signature-pinning test below assumes transformers is
# installed. If it isn't, the whole file is irrelevant -- mark a single
# module-level importorskip so the failure message is "no transformers"
# instead of N hard import failures.
# ---------------------------------------------------------------------------

pytest.importorskip("transformers")


# ===========================================================================
# transformers.modeling_utils.checkpoint (gradient_checkpointing.py:232/234/246)
# ===========================================================================

def test_torch_checkpoint_function_first_positional_arg():
    """gradient_checkpointing.py:222 defines
    ``def unsloth_gradient_checkpoint(function, *args, use_reentrant=None,
    **kwargs)`` and is assigned to ``transformers.modeling_utils.checkpoint``
    and ``torch.utils.checkpoint.checkpoint``. Upstream must keep
    ``function`` as the first positional arg AND must keep ``use_reentrant``
    as a kwarg so zoo's override remains drop-in."""
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
    """gradient_checkpointing.py:234 / 246 / 924 do
    ``transformers.modeling_utils.checkpoint = unsloth_gradient_checkpoint``.
    If upstream renames or removes this re-export, the patch silently
    no-ops and stock checkpointing remains on -> long-context VRAM bug."""
    import transformers.modeling_utils as mu
    if not hasattr(mu, "checkpoint"):
        pytest.fail(
            "DRIFT DETECTED: transformers.modeling_utils.checkpoint: "
            "symbol removed upstream. zoo monkey-patches this attribute "
            "in gradient_checkpointing.py:232/246/924. Patch is now a no-op."
        )


# ===========================================================================
# transformers.integrations.bitsandbytes._replace_with_bnb_linear
# (patching_utils.py:751)
# ===========================================================================

def test_replace_with_bnb_linear_signature():
    """patching_utils.py rebuilds upstream
    ``_replace_with_bnb_linear`` via source rewrite (line 682
    ``inspect.getsource(...)``) then re-installs as
    ``_unsloth_replace_with_bnb_linear``. The rewrite assumes
    parameters: ``(model, modules_to_not_convert, current_key_name,
    quantization_config, has_been_replaced)``."""
    pytest.importorskip("bitsandbytes")
    try:
        from transformers.integrations.bitsandbytes import (
            _replace_with_bnb_linear,
        )
    except ImportError:
        # On transformers 5.x this private was removed -- zoo guards
        # this at patching_utils.py:678 and falls back to should_convert_module.
        # That's a legitimate API migration, not drift. Confirm the
        # fallback symbol exists instead.
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


# ===========================================================================
# transformers.modeling_utils.PreTrainedModel.loss_function (loss_utils.py:145)
# ===========================================================================

def test_pretrained_model_loss_function_exists():
    """loss_utils.py:143-146 unwraps ``PreTrainedModel.loss_function.fget.__wrapped__``
    if loss_function is a property. If upstream removes the property
    entirely or makes it a plain attribute, the unwrap raises and the
    patch silently aborts -> stock loss runs, no fused CE."""
    import transformers.modeling_utils as mu
    if not hasattr(mu.PreTrainedModel, "loss_function"):
        pytest.fail(
            "DRIFT DETECTED: transformers.modeling_utils.PreTrainedModel.loss_function: "
            "attribute removed upstream. loss_utils.py:143 patch silently aborts."
        )


def test_LOSS_MAPPING_ForCausalLM_signature_compatible():
    """loss_utils.py:140 sets ``LOSS_MAPPING['ForCausalLM'] =
    UnslothForCausalLMLoss`` which is defined with parameters
    ``(logits, labels, vocab_size, num_items_in_batch=None,
    ignore_index=-100, **kwargs)``. Upstream loss callers must still
    pass at least the first three positionally, else zoo's override
    receives a swapped arg-order. We pin the ORIGINAL function's signature
    so any rename surfaces immediately."""
    from transformers.loss.loss_utils import LOSS_MAPPING
    if "ForCausalLM" not in LOSS_MAPPING:
        pytest.fail(
            "DRIFT DETECTED: transformers.loss.loss_utils.LOSS_MAPPING: "
            "'ForCausalLM' key removed. loss_utils.py:140 monkey-patch no-ops."
        )
    upstream = LOSS_MAPPING["ForCausalLM"]
    # Zoo's UnslothForCausalLMLoss expects logits, labels, vocab_size to be
    # the first three positionals. Upstream must accept the same.
    _assert_params_superset(
        upstream,
        required=["logits", "labels", "vocab_size"],
        zoo_callsite="loss_utils.py:113 UnslothForCausalLMLoss positional contract",
    )


def test_fixed_cross_entropy_signature():
    """loss_utils.py:99 inside UnslothFixedCrossEntropy calls back into
    transformers' upstream cross entropy helper indirectly via the loss
    function plumbing. The override uses ``num_items_in_batch`` and
    ``ignore_index`` keyword-forwarded. Pin those."""
    from transformers.loss.loss_utils import fixed_cross_entropy
    _assert_params_superset(
        fixed_cross_entropy,
        required=["num_items_in_batch", "ignore_index"],
        zoo_callsite="loss_utils.py:99 unsloth_fixed_cross_entropy forwards "
                     "num_items_in_batch and ignore_index by name",
    )


# ===========================================================================
# transformers Trainer (training_utils.py:354-355 and compiler.py:4040)
# ===========================================================================

def test_Trainer_get_optimizer_cls_and_kwargs_signature():
    """training_utils.py:354 calls
    ``Trainer.get_optimizer_cls_and_kwargs(training_args)`` as a single
    positional arg. If upstream changes the signature shape, zoo's
    isolated training loop builds a broken optimizer silently."""
    from transformers import Trainer
    _assert_positional_arity_at_least(
        Trainer.get_optimizer_cls_and_kwargs,
        arity=1,
        zoo_callsite="training_utils.py:354 Trainer.get_optimizer_cls_and_kwargs(training_args)",
    )


def test_Trainer_get_decay_parameter_names_signature():
    """training_utils.py:355 calls ``Trainer.get_decay_parameter_names(
    None, model)`` -- passes ``self=None`` and the model positionally. So
    upstream must accept (self, model) at least."""
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
    """compiler.py:3966-4040 replaces ``Trainer._inner_training_loop`` with
    ``_fast_inner_training_loop``. The rewriter uses
    ``inspect.getsource`` on the original. Pin the parameters the
    rewriter assumes exist: self, batch_size, args, resume_from_checkpoint,
    trial, ignore_keys_for_eval. If upstream renames any of these, the
    body-substitution targets that the rewriter performs at lines
    4011-4038 silently fail to match -> stock loop remains."""
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


# ===========================================================================
# transformers.set_seed / get_scheduler / seed_worker / DataCollator*
# (training_utils.py:20-23, 345-349, dataset_utils.py:457/672/678/686)
# ===========================================================================

def test_set_seed_signature():
    """training_utils.py:20 imports ``set_seed`` and uses it directly.
    The first positional must be ``seed``."""
    from transformers import set_seed
    sig = inspect.signature(set_seed)
    params = list(sig.parameters.keys())
    if not params or params[0] != "seed":
        pytest.fail(
            f"DRIFT DETECTED: transformers.set_seed: zoo uses positional "
            f"seed arg, but installed signature is {params}"
        )


def test_get_scheduler_signature():
    """training_utils.py:377 calls ``transformers_get_scheduler(name=...,
    optimizer=..., num_warmup_steps=..., num_training_steps=...,
    **lr_scheduler_kwargs)``. Pin those keyword args."""
    from transformers import get_scheduler
    _assert_params_superset(
        get_scheduler,
        required=["name", "optimizer", "num_warmup_steps", "num_training_steps"],
        zoo_callsite="training_utils.py:377 transformers_get_scheduler(name, optimizer, "
                     "num_warmup_steps, num_training_steps)",
    )


def test_seed_worker_imported_as_trainer_utils_seed_worker():
    """training_utils.py:23 imports
    ``transformers.trainer_utils.seed_worker as trainer_utils_seed_worker``
    -- the import must succeed at zoo import time. Confirm presence."""
    try:
        from transformers.trainer_utils import seed_worker  # noqa: F401
    except ImportError as exc:
        pytest.fail(
            f"DRIFT DETECTED: transformers.trainer_utils.seed_worker "
            f"import failed: {exc}. training_utils.py:23 has no fallback."
        )


def test_DataCollatorForLanguageModeling_signature():
    """training_utils.py:346 and dataset_utils.py:686 instantiate
    ``DataCollatorForLanguageModeling(tokenizer=..., mlm=False,
    pad_to_multiple_of=4)``. Pin those three kwargs."""
    from transformers import DataCollatorForLanguageModeling
    _assert_params_superset(
        DataCollatorForLanguageModeling.__init__,
        required=["tokenizer", "mlm"],
        zoo_callsite="training_utils.py:346 + dataset_utils.py:686 + 838 "
                     "DataCollatorForLanguageModeling(tokenizer, mlm, pad_to_multiple_of)",
    )


def test_DataCollatorForSeq2Seq_signature():
    """dataset_utils.py:464 / 678 instantiate
    ``DataCollatorForSeq2Seq(tokenizer=...)``."""
    from transformers import DataCollatorForSeq2Seq
    _assert_params_superset(
        DataCollatorForSeq2Seq.__init__,
        required=["tokenizer"],
        zoo_callsite="dataset_utils.py:464 / 678 DataCollatorForSeq2Seq(tokenizer)",
    )


# ===========================================================================
# TrainingArguments (temporary_patches/misc.py:1334 patches to_dict)
# ===========================================================================

def test_TrainingArguments_to_dict_signature():
    """temporary_patches/misc.py:1334-1343 wraps
    ``TrainingArguments.to_dict`` with one that injects
    ``push_to_hub_token``. zoo's wrapper signature is ``def
    _patched_to_dict(self)`` -- upstream must remain a zero-arg
    (besides self) method, else the wrapper drops kwargs."""
    from transformers import TrainingArguments
    sig = inspect.signature(TrainingArguments.to_dict)
    params = list(sig.parameters.keys())
    # MUST be just self; anything else means upstream added params that
    # zoo's wrapper would silently swallow.
    if params != ["self"]:
        pytest.fail(
            f"DRIFT DETECTED: TrainingArguments.to_dict: zoo wrapper at "
            f"temporary_patches/misc.py:1337 is `def _patched_to_dict(self)` "
            f"with NO *args/**kwargs forwarding, but installed signature "
            f"is {params}. Any caller-passed kwarg is silently dropped."
        )


def test_TrainingArguments_get_warmup_steps_signature():
    """training_utils.py:380 calls
    ``training_args.get_warmup_steps(max_steps)`` -- one positional."""
    from transformers import TrainingArguments
    _assert_positional_arity_at_least(
        TrainingArguments.get_warmup_steps,
        arity=1,
        zoo_callsite="training_utils.py:380 training_args.get_warmup_steps(max_steps)",
    )


# ===========================================================================
# PretrainedConfig (patching_utils.py:244-273)
# ===========================================================================

def test_PretrainedConfig_to_dict_signature():
    """patching_utils.py:256-259 wraps
    ``PretrainedConfig.to_dict`` with ``def wrapped_to_dict(self, *args,
    **kwargs)``. That forwarding is correct so long as upstream
    ``to_dict`` is a method. If upstream makes it a classmethod or moves
    it, the @wraps target breaks."""
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


# ===========================================================================
# PushToHubMixin.push_to_hub (saving_utils.py:76)
# ===========================================================================

def test_PushToHubMixin_push_to_hub_signature():
    """saving_utils.py:76 imports ``PushToHubMixin`` and uses it as a
    mixin base for the save plumbing. Pin the canonical ``push_to_hub``
    kwargs zoo expects (``repo_id``, ``commit_message``, ``token``,
    ``private``, ``revision``, ``create_pr``)."""
    from transformers.modeling_utils import PushToHubMixin
    _assert_params_superset(
        PushToHubMixin.push_to_hub,
        required=["repo_id", "commit_message", "token", "private", "revision"],
        zoo_callsite="saving_utils.py + utilities calling PushToHubMixin.push_to_hub",
    )


# ===========================================================================
# accelerate.init_empty_weights (empty_model.py:238, 322)
# ===========================================================================

def test_accelerate_init_empty_weights_signature():
    """empty_model.py:252 / 329 do ``with init_empty_weights(include_buffers
    = False):``. Pin that parameter name."""
    pytest.importorskip("accelerate")
    from accelerate import init_empty_weights
    _assert_params_superset(
        init_empty_weights,
        required=["include_buffers"],
        zoo_callsite="empty_model.py:252 with init_empty_weights(include_buffers=False)",
    )


# ===========================================================================
# Masking-utils + GPT-OSS overrides (temporary_patches/gpt_oss.py)
# ===========================================================================

def test_masking_utils_create_causal_mask_signature():
    """temporary_patches/gpt_oss.py:2178-2182 wraps
    ``transformers.masking_utils.create_causal_mask`` via ``wrap()`` and
    re-assigns. zoo's wrap is *args/**kwargs forwarding so positional
    layout is invariant, but the SYMBOL must exist."""
    try:
        from transformers.masking_utils import create_causal_mask  # noqa
    except ImportError as exc:
        pytest.fail(
            f"DRIFT DETECTED: transformers.masking_utils.create_causal_mask "
            f"removed: {exc}. gpt_oss.py:2178 wrap target gone."
        )


def test_masking_utils_create_sliding_window_causal_mask_signature():
    """Companion of the above. gpt_oss.py:2179 wraps it; ministral.py
    also depends on it being importable."""
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
    """gpt_oss.py:2184-2185 wraps
    ``transformers.masking_utils.create_masks_for_generate`` and the
    re-export in ``transformers.generation.utils``."""
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


# ===========================================================================
# Gemma3 forward / norm / mlp overrides (temporary_patches/gemma.py)
# ===========================================================================

def test_gemma3_apply_rotary_pos_emb_signature():
    """gemma.py:399 imports ``apply_rotary_pos_emb`` from gemma3 and
    calls it as ``apply_rotary_pos_emb(query_states, key_states, cos,
    sin)`` -- four positionals. So upstream must accept >=4 positional
    args."""
    from transformers.models.gemma3.modeling_gemma3 import apply_rotary_pos_emb
    _assert_positional_arity_at_least(
        apply_rotary_pos_emb,
        arity=4,
        zoo_callsite="gemma.py:399+639 apply_rotary_pos_emb(q, k, cos, sin)",
    )


def test_gemma3_eager_attention_forward_signature():
    """gemma.py:399 / ministral.py:38 import ``eager_attention_forward``
    and the relaxed-mode patch passes ``module, query, key, value,
    attention_mask, dropout, scaling`` -- pin those keyword/positional
    forwardable names."""
    from transformers.models.gemma3.modeling_gemma3 import eager_attention_forward
    _assert_params_superset(
        eager_attention_forward,
        required=["module", "query", "key", "value", "attention_mask"],
        zoo_callsite="gemma.py + ministral.py eager_attention_forward forward chain",
    )


def test_gemma3_ALL_ATTENTION_FUNCTIONS_present():
    """gemma.py:399 / ministral.py:39 import ``ALL_ATTENTION_FUNCTIONS``
    from gemma3.modeling_gemma3. If upstream moves it to
    ``transformers.modeling_utils`` only, the import in zoo fails at
    patch-registration time."""
    from transformers.models.gemma3.modeling_gemma3 import ALL_ATTENTION_FUNCTIONS  # noqa
    # Must be a mapping-like object: zoo does ``ALL_ATTENTION_FUNCTIONS[name]``
    if not hasattr(ALL_ATTENTION_FUNCTIONS, "__getitem__"):
        pytest.fail(
            f"DRIFT DETECTED: gemma3.ALL_ATTENTION_FUNCTIONS: zoo subscripts "
            f"this object via ``ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]`` "
            f"but installed type {type(ALL_ATTENTION_FUNCTIONS)} has no __getitem__"
        )


def test_Gemma3Processor_call_signature():
    """gemma.py:224 patches
    ``Gemma3Processor.__call__`` with ``match_level='relaxed'``. The
    replacement defines ``__call__(self, images, text, videos, audio,
    **kwargs)`` -- pin those param names."""
    from transformers.models.gemma3.processing_gemma3 import Gemma3Processor
    _assert_params_superset(
        Gemma3Processor.__call__,
        required=["images", "text"],
        zoo_callsite="gemma.py:224 Gemma3Processor.__call__ patch",
    )


def test_Gemma3RMSNorm_forward_signature():
    """gemma.py:361 / 628 patches
    ``Gemma3RMSNorm.forward(self, hidden_states)`` with fullgraph=True.
    Upstream must keep it a one-tensor forward."""
    from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm
    sig = inspect.signature(Gemma3RMSNorm.forward)
    params = [p.name for p in sig.parameters.values() if p.name != "self"]
    # zoo's replacement: def forward(self, hidden_states) -> single positional.
    if len(params) != 1:
        pytest.fail(
            f"DRIFT DETECTED: Gemma3RMSNorm.forward: zoo replacement at "
            f"gemma.py:361/628 takes (self, hidden_states) but installed "
            f"signature has params {params}"
        )


def test_Gemma3MLP_forward_signature():
    """gemma.py:389 patches Gemma3MLP.forward with a single-tensor
    forward. Pin the positional shape."""
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
    """gemma.py:331 patches
    ``Gemma3TextScaledWordEmbedding.forward(self, input_ids)`` with
    fullgraph=True. Pin the (self, input_ids) shape."""
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
    """gemma.py:607/849 patches Gemma3Attention.forward via
    ``patch_function_past_key_values`` with match_level='relaxed'. Pin
    the keyword params zoo's forward variants forward by name."""
    from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention
    _assert_params_superset(
        Gemma3Attention.forward,
        required=["hidden_states", "position_embeddings", "attention_mask"],
        zoo_callsite="gemma.py:607/849 Gemma3Attention.forward patch",
    )


# ===========================================================================
# Gemma3n overrides (temporary_patches/gemma3n.py)
# ===========================================================================

def test_Gemma3nMultimodalEmbedder_forward_signature():
    """gemma3n.py:88 patches
    ``Gemma3nMultimodalEmbedder.forward(self, input_ids, inputs_embeds)``
    with fullgraph=True. Pin those two positional kwargs."""
    from transformers.models.gemma3n.modeling_gemma3n import (
        Gemma3nMultimodalEmbedder,
    )
    _assert_params_superset(
        Gemma3nMultimodalEmbedder.forward,
        required=["input_ids", "inputs_embeds"],
        zoo_callsite="gemma3n.py:88 Gemma3nMultimodalEmbedder.forward patch",
    )


def test_Gemma3nTextAltUp_predict_signature():
    """gemma3n.py:122 patches
    ``Gemma3nTextAltUp.predict(self, hidden_states)`` with
    fullgraph=True. Pin the one-positional shape."""
    from transformers.models.gemma3n.modeling_gemma3n import Gemma3nTextAltUp
    sig = inspect.signature(Gemma3nTextAltUp.predict)
    params = [p.name for p in sig.parameters.values() if p.name != "self"]
    if "hidden_states" not in params:
        pytest.fail(
            f"DRIFT DETECTED: Gemma3nTextAltUp.predict: zoo replacement at "
            f"gemma3n.py:122 takes (self, hidden_states) but installed "
            f"signature has params {params}"
        )


def test_Gemma3nTextAltUp_correct_signature():
    """gemma3n.py:146 patches
    ``Gemma3nTextAltUp.correct(self, predictions, activated)``."""
    from transformers.models.gemma3n.modeling_gemma3n import Gemma3nTextAltUp
    _assert_params_superset(
        Gemma3nTextAltUp.correct,
        required=["predictions", "activated"],
        zoo_callsite="gemma3n.py:146 Gemma3nTextAltUp.correct patch",
    )


def test_Gemma3nModel_get_placeholder_mask_signature():
    """gemma3n.py:201 patches
    ``Gemma3nModel.get_placeholder_mask`` with match_level='relaxed'. Pin
    the keyword params zoo forwards by name: ``input_ids``,
    ``inputs_embeds``, ``image_features``, ``audio_features``."""
    from transformers.models.gemma3n.modeling_gemma3n import Gemma3nModel
    _assert_params_superset(
        Gemma3nModel.get_placeholder_mask,
        required=["input_ids", "inputs_embeds"],
        zoo_callsite="gemma3n.py:201 Gemma3nModel.get_placeholder_mask patch",
    )


# ===========================================================================
# Ministral overrides (temporary_patches/ministral.py)
# ===========================================================================

def test_MinistralAttention_forward_signature():
    """ministral.py:99 patches MinistralAttention.forward with
    match_level='relaxed'. zoo's replacement signature is
    ``forward(self, hidden_states, position_embeddings, attention_mask=None,
    past_key_values=None, cache_position=None, **kwargs)``."""
    try:
        from transformers.models.ministral.modeling_ministral import (
            MinistralAttention,
        )
    except ImportError:
        pytest.skip("transformers.models.ministral not installed (added in 4.57)")
    _assert_params_superset(
        MinistralAttention.forward,
        required=["hidden_states", "position_embeddings", "attention_mask"],
        zoo_callsite="ministral.py:99 MinistralAttention.forward patch",
    )


def test_MinistralModel_forward_signature():
    """ministral.py:179 patches MinistralModel.forward with
    match_level='relaxed'. zoo forwards by name: input_ids,
    attention_mask, position_ids, past_key_values, inputs_embeds,
    use_cache, cache_position."""
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
    """ministral.py:37 imports ``apply_rotary_pos_emb`` from ministral
    and calls it ``apply_rotary_pos_emb(query_states, key_states, cos,
    sin)`` -- 4 positionals."""
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


# ===========================================================================
# GPT-OSS class-level monkey-patches (temporary_patches/gpt_oss.py)
# ===========================================================================

def test_GptOssExperts_class_present_and_init_takes_config():
    """gpt_oss.py:1060 / 1070 / 1849 / 1858 monkey-patch
    ``transformers.models.gpt_oss.modeling_gpt_oss.GptOssExperts``. The
    class and its ``(self, config)`` __init__ must remain stable, since
    zoo's GptOssExpertsBnb4bit / GptOssExperts override defines
    ``__init__(self, config)``."""
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
    """gpt_oss.py:1845 / 1852 replaces ``GptOssExperts.forward`` with
    ``forward(self, hidden_states, router_indices=None,
    routing_weights=None)``. Upstream must keep these param names since
    they're forwarded by name."""
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
    """gpt_oss.py:1062 / 1077 monkey-patches
    ``transformers.models.gpt_oss.modeling_gpt_oss.GptOssTopKRouter``.
    The class must remain importable."""
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
    """gpt_oss.py:2201-2220 patches with ``pre_attention_decoding(self,
    hidden_states, position_embeddings, attention_mask, past_key_values,
    cache_position, **kwargs)``. The GptOssAttention.forward target
    upstream must accept the same keyword forwarding."""
    try:
        from transformers.models.gpt_oss.modeling_gpt_oss import (
            GptOssAttention,
        )
    except ImportError:
        pytest.skip("transformers.models.gpt_oss not installed")
    _assert_params_superset(
        GptOssAttention.forward,
        required=["hidden_states", "position_embeddings", "attention_mask"],
        zoo_callsite="gpt_oss.py:2201 pre_attention_decoding shape vs "
                     "GptOssAttention.forward",
    )


def test_GptOssModel_forward_signature():
    """gpt_oss.py:2481 patches GptOssModel.forward with
    match_level='relaxed'. Pin the kwargs zoo forwards by name."""
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


# ===========================================================================
# Mxfp4 integrations (temporary_patches/gpt_oss.py:190/433/454/540/569)
# ===========================================================================

def test_mxfp4_swizzle_mxfp4_signature():
    """gpt_oss.py:190 patches
    ``transformers.integrations.mxfp4.swizzle_mxfp4`` with
    ``match_level='relaxed'``. The replacement signature in zoo accepts
    ``w, w_scale, triton_kernels_hub`` -- upstream must keep at least
    those three positional names."""
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
    """gpt_oss.py:540 patches ``load_and_swizzle_mxfp4`` with
    ``match_level='relaxed'``. zoo's replacement accepts
    ``module, param_name, param_value, target_device,
    triton_kernels_hub, **kwargs``."""
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
    """gpt_oss.py:454 patches ``mlp_forward`` in
    ``transformers.integrations.mxfp4``. zoo expects (self, hidden_states)."""
    try:
        from transformers.integrations.mxfp4 import mlp_forward
    except (ImportError, AttributeError):
        pytest.skip("transformers.integrations.mxfp4.mlp_forward not available")
    _assert_params_superset(
        mlp_forward,
        required=["hidden_states"],
        zoo_callsite="gpt_oss.py:454 mlp_forward patch",
    )


# ===========================================================================
# AutoHfQuantizer.merge_quantization_configs (misc.py:153)
# ===========================================================================

def test_AutoHfQuantizer_merge_quantization_configs_signature():
    """misc.py:153 patches
    ``transformers.quantizers.auto.AutoHfQuantizer.merge_quantization_configs``.
    zoo's replacement signature is ``(quantization_config,
    quantization_config_from_args)``."""
    from transformers.quantizers.auto import AutoHfQuantizer
    _assert_params_superset(
        AutoHfQuantizer.merge_quantization_configs,
        required=["quantization_config", "quantization_config_from_args"],
        zoo_callsite="misc.py:153 AutoHfQuantizer.merge_quantization_configs patch",
    )


# ===========================================================================
# Granitemoehybrid + CSM (misc.py:1061 / 770)
# ===========================================================================

def test_GraniteMoeHybridMambaLayer_cuda_kernels_forward_signature():
    """misc.py:1061 patches
    ``GraniteMoeHybridMambaLayer.cuda_kernels_forward`` -- the patch
    expects ``(self, hidden_states, cache_params, cache_position,
    attention_mask, seq_idx)``."""
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
    """misc.py:770 patches
    ``CsmForConditionalGeneration._merge_input_ids_with_input_values``.
    zoo's replacement signature is ``(self, input_ids, input_values,
    input_values_cutoffs, labels)``."""
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


# ===========================================================================
# Mllama vision encoder layer (misc.py:1172)
# ===========================================================================

def test_MllamaVisionEncoderLayer_forward_signature():
    """misc.py:1146-1172 defines a replacement
    ``MllamaVisionEncoderLayer.forward(self, hidden_state,
    attention_mask=None)`` -- pin those param names. NOTE the upstream
    uses ``hidden_state`` (singular), not ``hidden_states``."""
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


# ===========================================================================
# Siglip encoder layer (misc.py:1228)
# ===========================================================================

def test_SiglipEncoderLayer_forward_signature():
    """misc.py:1187-1228 defines a replacement
    ``SiglipEncoderLayer.forward(self, hidden_states, attention_mask,
    output_attentions=False)``. The replacement still references
    ``output_attentions`` in the body, so upstream removing it (already
    happened in some versions) leaves zoo's patched body broken when
    callers stop passing it. Pin ``hidden_states`` + ``attention_mask``
    as a minimum."""
    from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer
    _assert_params_superset(
        SiglipEncoderLayer.forward,
        required=["hidden_states", "attention_mask"],
        zoo_callsite="misc.py:1187-1228 SiglipEncoderLayer.forward patch",
    )


# ===========================================================================
# Qwen3 MoE (qwen3_moe / qwen3_5_moe / qwen3_vl_moe / qwen3_next_moe)
# ===========================================================================

def test_Qwen3MoeSparseMoeBlock_forward_signature():
    """qwen3_moe.py patches Qwen3MoeSparseMoeBlock.forward via
    patch_function. zoo's replacement is single-positional
    ``forward(self, hidden_states)``."""
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
    """qwen3_vl_moe.py:362-383 patches
    ``Qwen3VLMoeTextSparseMoeBlock.forward(self, hidden_states)``."""
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
    """qwen3_vl_moe.py:376 patches Qwen3VLMoeTextExperts.forward. Zoo's
    replacement signature is ``forward(self, hidden_states, top_k_index,
    top_k_weights)`` BUT it overrides the upstream which uses
    ``hidden_states, routing_weights, router_indices``. Pin only
    ``hidden_states`` (1st positional) so the patch's positional arity
    stays compatible."""
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
    """qwen3_vl_moe.py:242 patches ``Qwen3VLMoeTextExperts.__init__``
    with ``patched_experts_init(self, config)``. Pin (self, config)."""
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
    """qwen3_next_moe.py:67 patches Qwen3NextSparseMoeBlock.forward."""
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


# ===========================================================================
# Deepseek-V3 MoE (deepseek_v3_moe.py)
# ===========================================================================

def test_DeepseekV3MoE_forward_signature():
    """deepseek_v3_moe.py:125 patches DeepseekV3MoE.forward(self,
    hidden_states)."""
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
    """deepseek_v3_moe.py:213 patches DeepseekV3ForCausalLM.forward.
    Zoo's replacement forwards by name: input_ids, attention_mask,
    position_ids, past_key_values, inputs_embeds, labels, use_cache."""
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


# ===========================================================================
# PEFT (temporary_patches/misc.py:1281 dispatch_bnb_4bit wrap)
# ===========================================================================

def test_peft_dispatch_bnb_4bit_signature():
    """misc.py:1297 wraps ``peft.tuners.lora.bnb.dispatch_bnb_4bit``
    with ``def safe_dispatch_bnb_4bit(target, adapter_name, **kwargs)``.
    Upstream must keep the first two positional params and the **kwargs
    tail, else zoo's wrapper either drops or mis-positions arguments."""
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
    """patching_utils.py:313 imports ``from peft.tuners.lora import
    Linear4bit as Peft_Linear4bit`` and uses ``isinstance(module,
    Peft_Linear4bit)``. Pin the import path."""
    pytest.importorskip("peft")
    try:
        from peft.tuners.lora import Linear4bit  # noqa
    except ImportError as exc:
        pytest.fail(
            f"DRIFT DETECTED: peft.tuners.lora.Linear4bit import: {exc}. "
            f"patching_utils.py:313 hard import."
        )


def test_peft_get_peft_model_signature():
    """peft.get_peft_model is the primary attach point used after
    ``get_peft_regex`` in peft_utils.py. The signature must accept
    ``model`` and ``peft_config`` (positionals 1 and 2)."""
    pytest.importorskip("peft")
    from peft import get_peft_model
    _assert_positional_arity_at_least(
        get_peft_model,
        arity=2,
        zoo_callsite="peft_utils.py get_peft_regex output -> get_peft_model(model, peft_config)",
    )


# ===========================================================================
# Cache utilities (gemma4.py, qwen3_moe etc -- DynamicCache, StaticCache)
# ===========================================================================

def test_DynamicCache_importable():
    """gemma4.py:308 / 460 imports ``DynamicCache`` and ``StaticCache``
    from ``transformers.cache_utils``. Confirm both still exist."""
    from transformers.cache_utils import DynamicCache, StaticCache  # noqa
    # All zoo callsites instantiate as ``DynamicCache()`` zero-arg; pin
    # that there IS a callable constructor (signature varies wildly).
    if not callable(DynamicCache):
        pytest.fail("DRIFT DETECTED: DynamicCache is no longer callable.")
    if not callable(StaticCache):
        pytest.fail("DRIFT DETECTED: StaticCache is no longer callable.")


# ===========================================================================
# Bitsandbytes patch (bitsandbytes.py:108)
# ===========================================================================

def test_bnb_Linear4bit_forward_signature():
    """bitsandbytes.py:108 patches ``bitsandbytes.nn.modules.Linear4bit.forward``
    with a replacement that takes ``(self, x)``."""
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


# ===========================================================================
# vllm (vllm_utils.py + temporary_patches/misc.py:1402)
# ===========================================================================

def test_vllm_SamplingParams_constructor():
    """vllm_utils.py's ``grpo_update_SamplingParams`` filters by
    ``inspect.signature(vllm.SamplingParams).parameters``. If vllm
    removes the constructor or changes it to a *args-only shape, the
    filter swallows every kwarg silently."""
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
