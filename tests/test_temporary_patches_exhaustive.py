# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at
# your option) any later version.

"""Exhaustive upstream-signature pinning for the (class, method) pairs
that ``unsloth_zoo/temporary_patches/<file>.py`` rebinds (tail not
covered by the sibling test_upstream_signatures /
test_upstream_pinned_symbols_transformers / test_zoo_source_upstream_refs
/ test_upstream_source_patterns files).

Each (model_class, method) pair below maps 1:1 to a
``patch_function(...)`` call or attribute reassignment in zoo. Upstream
rename / drop -> zoo's patch silently no-ops via ``raise_error()``; this
file makes that drift loud.

CPU-only; ``pytest.importorskip`` for optional libs (timm, bitsandbytes).
Drift -> ``pytest.fail("DRIFT DETECTED: zoo temporary_patches/<file>.py
expects <class>.<method>(<params>) but installed transformers has
<signature>")``.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
from typing import Iterable

import pytest


pytest.importorskip("transformers")
import transformers  # noqa: E402

_TX_VERSION = getattr(transformers, "__version__", "0.0.0")


def _try_get_class(dotted_module: str, class_name: str):
    """Import ``dotted_module`` and return ``class_name`` off it (or None).
    Used to skip 5.0+-gated tests on a 4.x install."""
    try:
        mod = importlib.import_module(dotted_module)
    except Exception:
        return None
    return getattr(mod, class_name, None)


def _require_class(dotted_module: str, class_name: str, zoo_file: str):
    """Like ``_try_get_class`` but DRIFT-fail when the parent module
    exists but the class is missing. Skip when parent module is missing
    (legitimate version gate)."""
    try:
        mod = importlib.import_module(dotted_module)
    except Exception as exc:
        pytest.skip(
            f"{zoo_file}: parent module {dotted_module!r} unavailable on "
            f"transformers {_TX_VERSION}: {exc}"
        )
    cls = getattr(mod, class_name, None)
    if cls is None:
        pytest.fail(
            f"DRIFT DETECTED: zoo temporary_patches/{zoo_file} expects "
            f"{dotted_module}.{class_name} but installed transformers "
            f"{_TX_VERSION} has no such attribute on the module"
        )
    return cls


def _param_names(func) -> list[str]:
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError) as exc:
        pytest.fail(f"DRIFT DETECTED: cannot inspect {func!r}: {exc}")
    return [name for name in sig.parameters.keys()]


def _original_attr_name(cls, attr: str) -> str:
    """Storage key used by patch_function to stash the original method
    body: ``_original_<last-module-component>_<class>_<attr>``. Mirrors
    _get_unique_storage_name in temporary_patches/utils.py."""
    module_tail = getattr(cls, "__module__", "").rsplit(".", 1)[-1]
    class_name = getattr(cls, "__name__", "") or cls.__class__.__name__
    return f"_original_{module_tail}_{class_name}_{attr}"


def _resolve_upstream_method(cls, method_name: str):
    """Return UPSTREAM (unpatched) method body for cls.method_name.

    apply_import_fixes() and temporary_patches runner monkey-patch at
    import time, so naive ``cls.method_name`` returns zoo's
    ``(self, *args, **kwargs)`` wrapper. Resolution order:
      1. ``_original_<module>_<class>_<method>`` stash from patch_function.
      2. Live attribute (caller checks _maybe_skip_if_patched if needed).
    """
    if not hasattr(cls, method_name):
        return None
    live = getattr(cls, method_name)
    storage_key = _original_attr_name(cls, method_name)
    original = getattr(cls, storage_key, None)
    if original is not None:
        return original
    qualname = getattr(live, "__qualname__", "") or ""
    if ".<locals>." in qualname and qualname.split(".", 1)[0].startswith("patch_"):
        # zoo patch wrapper with no _original_ stash (rare: force=True +
        # store_original=False); _maybe_skip_if_patched handles cleanly.
        return live
    return live


def _maybe_skip_if_patched(cls, method_name: str, zoo_file: str) -> None:
    """Skip when live method is a zoo patch wrapper with no original
    stash (avoid false-failing on ``(self, *args, **kwargs)``)."""
    if not hasattr(cls, method_name):
        return
    live = getattr(cls, method_name)
    storage_key = _original_attr_name(cls, method_name)
    original = getattr(cls, storage_key, None)
    if original is not None:
        # We have the upstream original stashed; tests use it directly.
        return
    qualname = getattr(live, "__qualname__", "") or ""
    if ".<locals>." in qualname and qualname.split(".", 1)[0].startswith("patch_"):
        pytest.skip(
            f"{zoo_file}: {cls.__module__}.{cls.__name__}.{method_name} "
            f"is already overwritten by zoo's patch wrapper "
            f"{qualname!r}; no upstream-original stash is available on "
            f"this run, so the upstream signature pin can't be probed "
            f"directly. The patch itself is exercised by the temporary_"
            f"patches integration tests."
        )


def _assert_method_exists(cls, method_name: str, zoo_file: str):
    if not hasattr(cls, method_name):
        pytest.fail(
            f"DRIFT DETECTED: zoo temporary_patches/{zoo_file} expects "
            f"{cls.__module__}.{cls.__name__}.{method_name} but installed "
            f"transformers {_TX_VERSION} has no such method on the class"
        )
    return _resolve_upstream_method(cls, method_name)


def _assert_params_superset(
    func,
    required: Iterable[str],
    zoo_file: str,
    label: str,
):
    got = _param_names(func)
    missing = [name for name in required if name not in got]
    if missing:
        try:
            sig = inspect.signature(func)
        except Exception:
            sig = "<uninspectable>"
        pytest.fail(
            f"DRIFT DETECTED: zoo temporary_patches/{zoo_file} expects "
            f"{label}({sorted(required)}) but installed transformers "
            f"{_TX_VERSION} has signature {sig} (missing {sorted(missing)})"
        )


def _has_var_keyword(func) -> bool:
    try:
        sig = inspect.signature(func)
    except Exception:
        return False
    return any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )


# bitsandbytes.py: patches bitsandbytes.nn.modules.Linear4bit.forward
# (covered elsewhere) + bitsandbytes.nn.Linear4bit.forward top-level
# re-export alias.

def test_bitsandbytes_top_level_Linear4bit_alias():
    """bitsandbytes.py:110 patches top-level ``bitsandbytes.nn.Linear4bit``
    alias; pin presence + .forward method."""
    bnb = pytest.importorskip("bitsandbytes")
    top_level = getattr(bnb.nn, "Linear4bit", None)
    inner = getattr(bnb.nn.modules, "Linear4bit", None)
    if top_level is None or inner is None:
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/bitsandbytes.py expects "
            "both bitsandbytes.nn.Linear4bit and bitsandbytes.nn.modules.Linear4bit "
            "but at least one is missing"
        )
    if not hasattr(top_level, "forward"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/bitsandbytes.py:110 expects "
            "bitsandbytes.nn.Linear4bit.forward but it has no forward attribute"
        )


def test_bitsandbytes_Params4bit_class_present():
    """bitsandbytes.py:47-67 reads ``Params4bit`` and conditionally
    deletes ``__torch_function__`` (torch.compile recursion fix)."""
    bnb = pytest.importorskip("bitsandbytes")
    p4 = getattr(bnb.nn.modules, "Params4bit", None)
    if p4 is None:
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/bitsandbytes.py:47 expects "
            "bitsandbytes.nn.modules.Params4bit but it is missing"
        )


def test_bitsandbytes_fix_4bit_weight_quant_state_from_module_present():
    """bitsandbytes.py:48 / :73 calls
    ``fix_4bit_weight_quant_state_from_module(self)``."""
    bnb = pytest.importorskip("bitsandbytes")
    fn = getattr(bnb.nn.modules, "fix_4bit_weight_quant_state_from_module", None)
    if fn is None:
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/bitsandbytes.py:48 expects "
            "bitsandbytes.nn.modules.fix_4bit_weight_quant_state_from_module "
            "but it is missing"
        )
    # Patched forward calls fn(self); reject zero-arity.
    sig = inspect.signature(fn)
    params = [
        p for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                      inspect.Parameter.POSITIONAL_ONLY,
                      inspect.Parameter.VAR_POSITIONAL)
    ]
    if not params:
        pytest.fail(
            f"DRIFT DETECTED: zoo temporary_patches/bitsandbytes.py:73 calls "
            f"fix_4bit_weight_quant_state_from_module(self) but installed "
            f"signature {sig} accepts no positional args"
        )


def test_bitsandbytes_matmul_4bit_present():
    """bitsandbytes.py:106 calls top-level ``bitsandbytes.matmul_4bit(...)``."""
    bnb = pytest.importorskip("bitsandbytes")
    if not hasattr(bnb, "matmul_4bit"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/bitsandbytes.py:106 expects "
            "bitsandbytes.matmul_4bit() but it is missing"
        )


# deepseek_v3_moe.py: patches DeepseekV3{NaiveMoe,MoE,ForCausalLM}.forward.

def test_deepseek_v3_naive_moe_class_gated_5x():
    """deepseek_v3_moe.py:56-61 imports DeepseekV3NaiveMoe (5.x-only;
    bails via try/except on 4.x)."""
    cls = _try_get_class(
        "transformers.models.deepseek_v3.modeling_deepseek_v3",
        "DeepseekV3NaiveMoe",
    )
    if cls is None:
        pytest.skip(
            f"DeepseekV3NaiveMoe absent on transformers {_TX_VERSION} -- "
            "5.x-only class, zoo gracefully no-ops via try/except"
        )
    _assert_method_exists(cls, "forward", "deepseek_v3_moe.py")


def test_deepseek_v3_topk_router_class_present():
    """deepseek_v3_moe.py:59 imports DeepseekV3TopkRouter as a gate;
    missing -> whole patch entry silently no-ops."""
    mod = importlib.import_module(
        "transformers.models.deepseek_v3.modeling_deepseek_v3"
    )
    cls = getattr(mod, "DeepseekV3TopkRouter", None)
    if cls is None:
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/deepseek_v3_moe.py:59 imports "
            "DeepseekV3TopkRouter as a gate condition but it is missing on "
            f"transformers {_TX_VERSION}"
        )


def test_deepseek_v3_config_class_present():
    """deepseek_v3_moe.py:60 imports DeepseekV3Config as a gate. Pin."""
    mod = importlib.import_module(
        "transformers.models.deepseek_v3.modeling_deepseek_v3"
    )
    if getattr(mod, "DeepseekV3Config", None) is None:
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/deepseek_v3_moe.py:60 imports "
            "DeepseekV3Config but it is missing on transformers "
            f"{_TX_VERSION}"
        )


def test_deepseek_v3_moe_forward_single_positional():
    """deepseek_v3_moe.py:125 patches DeepseekV3MoE.forward(self,
    hidden_states); pin single required positional."""
    cls = _try_get_class(
        "transformers.models.deepseek_v3.modeling_deepseek_v3",
        "DeepseekV3MoE",
    )
    if cls is None:
        pytest.skip(f"DeepseekV3MoE absent on transformers {_TX_VERSION}")
    fwd = _assert_method_exists(cls, "forward", "deepseek_v3_moe.py")
    params = _param_names(fwd)
    # Drop "self".
    params = [p for p in params if p != "self"]
    required = [p for p in inspect.signature(fwd).parameters.values()
                if p.name != "self" and p.default is inspect.Parameter.empty
                and p.kind not in (inspect.Parameter.VAR_POSITIONAL,
                                    inspect.Parameter.VAR_KEYWORD)]
    if len(required) != 1:
        pytest.fail(
            f"DRIFT DETECTED: zoo temporary_patches/deepseek_v3_moe.py:105 patches "
            f"DeepseekV3MoE.forward(self, hidden_states) -- single required arg "
            f"-- but installed signature has {len(required)} required positionals: "
            f"{[p.name for p in required]}"
        )


def test_deepseek_v3_for_causal_lm_forward_named_params():
    """deepseek_v3_moe.py:142 patches DeepseekV3ForCausalLM.forward with
    by-name kwargs (input_ids, attention_mask, ..., output_router_logits,
    cache_position, logits_to_keep); output_router_logits may be in
    **kwargs (TransformersKwargs catch-all)."""
    cls = _try_get_class(
        "transformers.models.deepseek_v3.modeling_deepseek_v3",
        "DeepseekV3ForCausalLM",
    )
    if cls is None:
        pytest.skip(f"DeepseekV3ForCausalLM absent on transformers {_TX_VERSION}")
    fwd = _assert_method_exists(cls, "forward", "deepseek_v3_moe.py")
    _assert_params_superset(
        fwd,
        required=[
            "input_ids", "attention_mask", "position_ids", "past_key_values",
            "inputs_embeds", "labels", "use_cache",
            "cache_position", "logits_to_keep",
        ],
        zoo_file="deepseek_v3_moe.py",
        label="DeepseekV3ForCausalLM.forward",
    )
    # output_router_logits: explicit param OR **kwargs passthrough.
    if "output_router_logits" not in _param_names(fwd) and not _has_var_keyword(fwd):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/deepseek_v3_moe.py:171 "
            "forwards output_router_logits=output_router_logits but installed "
            f"DeepseekV3ForCausalLM.forward on transformers {_TX_VERSION} "
            f"has neither an explicit output_router_logits param nor a "
            f"**kwargs catch-all: {inspect.signature(fwd)}"
        )


# gemma.py: UNSLOTH_FORCE_FLOAT32-gated Gemma3Model._update_causal_mask
# patch (gemma.py:308); upstream removed in 4.55+, patch no-op on modern.

def test_gemma3_force_fp32_update_causal_mask_gated():
    """gemma.py:308-310 patches Gemma3{Model,ForConditionalGeneration}.
    _update_causal_mask under UNSLOTH_FORCE_FLOAT32=1."""
    mod = importlib.import_module(
        "transformers.models.gemma3.modeling_gemma3"
    )
    model = getattr(mod, "Gemma3Model", None)
    for_cond = getattr(mod, "Gemma3ForConditionalGeneration", None)
    if model is None or for_cond is None:
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/gemma.py:233-234 expects "
            "Gemma3Model and Gemma3ForConditionalGeneration but at least one is "
            f"missing on transformers {_TX_VERSION}"
        )
    # Class presence required; method removal OK (zoo's patch_function no-ops).
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1":
        if not hasattr(model, "_update_causal_mask"):
            pytest.fail(
                "DRIFT DETECTED: zoo temporary_patches/gemma.py:308 with "
                "UNSLOTH_FORCE_FLOAT32=1 patches Gemma3Model._update_causal_mask, "
                f"but transformers {_TX_VERSION} has dropped this method. The "
                "patch silently no-ops and the FORCE_FLOAT32 mask fix never lands."
            )


# gemma3n.py: AltUp.scale_corrected_output (tail not covered by siblings).

def test_gemma3n_text_alt_up_scale_corrected_output_signature():
    """gemma3n.py:148 patches Gemma3nTextAltUp.scale_corrected_output
    (fullgraph=True); shape: ``(self, corrected)``."""
    cls = _require_class(
        "transformers.models.gemma3n.modeling_gemma3n",
        "Gemma3nTextAltUp",
        "gemma3n.py",
    )
    fn = _assert_method_exists(cls, "scale_corrected_output", "gemma3n.py")
    params = [
        p for p in inspect.signature(fn).parameters.values()
        if p.name != "self"
    ]
    required = [p for p in params
                if p.default is inspect.Parameter.empty
                and p.kind not in (inspect.Parameter.VAR_POSITIONAL,
                                    inspect.Parameter.VAR_KEYWORD)]
    if len(required) != 1:
        pytest.fail(
            f"DRIFT DETECTED: zoo temporary_patches/gemma3n.py:148 patches "
            f"Gemma3nTextAltUp.scale_corrected_output(self, corrected) -- "
            f"single required arg -- but installed signature has "
            f"{len(required)} required positionals: "
            f"{[p.name for p in required]}"
        )


def test_gemma3n_text_alt_up_three_methods_present():
    """gemma3n.py:143-148 hasattr-guards predict / correct /
    scale_corrected_output; loud-fail when ALL three are gone."""
    cls = _require_class(
        "transformers.models.gemma3n.modeling_gemma3n",
        "Gemma3nTextAltUp",
        "gemma3n.py",
    )
    present = [
        m for m in ("predict", "correct", "scale_corrected_output")
        if hasattr(cls, m)
    ]
    if not present:
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/gemma3n.py:143-148 "
            "expects at least one of Gemma3nTextAltUp.{predict,correct,"
            "scale_corrected_output} on installed transformers "
            f"{_TX_VERSION} but none are present"
        )


def test_gemma3n_RMSNorm_helper_target_present():
    """gemma3n.py:53-85 Gemma3nRMSNorm_forward helper dereferences
    self.soft_embedding_norm / self.hard_embedding_norm /
    self.embedding_projection / self.embedding_post_projection_norm on
    Gemma3nMultimodalEmbedder."""
    embedder = _require_class(
        "transformers.models.gemma3n.modeling_gemma3n",
        "Gemma3nMultimodalEmbedder",
        "gemma3n.py",
    )
    # Read __init__ source: zoo's patched forward dereferences
    # self.soft_embedding_norm and self.hard_embedding_norm. If they
    # were renamed in upstream, the patched forward AttributeError-s.
    try:
        src = inspect.getsource(embedder.__init__)
    except (OSError, TypeError):
        pytest.skip("Cannot read Gemma3nMultimodalEmbedder.__init__ source")
    for attr in ("soft_embedding_norm", "hard_embedding_norm",
                  "embedding_projection", "embedding_post_projection_norm"):
        if attr not in src:
            pytest.fail(
                f"DRIFT DETECTED: zoo temporary_patches/gemma3n.py:74-85 "
                f"dereferences self.{attr} on Gemma3nMultimodalEmbedder, "
                f"but the upstream __init__ source on transformers "
                f"{_TX_VERSION} doesn't mention {attr}"
            )


# gemma4.py: Gemma4TextMLP.forward (gemma4.py:655); 5.0+-only.

def test_gemma4_text_mlp_forward_signature():
    """gemma4.py:655 patches Gemma4TextMLP.forward(self, x); single
    positional arg."""
    cls = _try_get_class(
        "transformers.models.gemma4.modeling_gemma4", "Gemma4TextMLP",
    )
    if cls is None:
        pytest.skip(
            f"Gemma4TextMLP absent on transformers {_TX_VERSION} "
            "(Gemma4 is 5.0+-only, zoo gracefully no-ops via try/except)"
        )
    fwd = _assert_method_exists(cls, "forward", "gemma4.py")
    required = [p for p in inspect.signature(fwd).parameters.values()
                if p.name != "self"
                and p.default is inspect.Parameter.empty
                and p.kind not in (inspect.Parameter.VAR_POSITIONAL,
                                    inspect.Parameter.VAR_KEYWORD)]
    if len(required) != 1:
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/gemma4.py:655 patches "
            "Gemma4TextMLP.forward(self, x) -- single required arg -- but "
            f"installed signature has {len(required)} required positionals: "
            f"{[p.name for p in required]}"
        )


def test_gemma4_text_mlp_has_required_attrs():
    """gemma4.py:644-652 patched forward dereferences self.{gate_proj,
    up_proj, down_proj, act_fn}; pin presence in __init__ source."""
    cls = _try_get_class(
        "transformers.models.gemma4.modeling_gemma4", "Gemma4TextMLP",
    )
    if cls is None:
        pytest.skip(f"Gemma4TextMLP absent on transformers {_TX_VERSION}")
    try:
        src = inspect.getsource(cls.__init__)
    except (OSError, TypeError):
        pytest.skip("Cannot read Gemma4TextMLP.__init__ source")
    for attr in ("gate_proj", "up_proj", "down_proj", "act_fn"):
        if attr not in src:
            pytest.fail(
                f"DRIFT DETECTED: zoo temporary_patches/gemma4.py:644-652 "
                f"dereferences self.{attr} on Gemma4TextMLP, but the upstream "
                f"__init__ source on transformers {_TX_VERSION} doesn't "
                f"mention {attr}"
            )


# gemma4_moe.py: Gemma4Text{Experts,DecoderLayer,MoEBlock}{.forward,
# .__init__}, Gemma4ForConditionalGeneration.forward. 5.0+-gated.

def test_gemma4_text_experts_forward_signature():
    """gemma4_moe.py:239 patches Gemma4TextExperts.forward(self,
    hidden_states, top_k_index, top_k_weights)."""
    cls = _try_get_class(
        "transformers.models.gemma4.modeling_gemma4", "Gemma4TextExperts",
    )
    if cls is None:
        pytest.skip(
            f"Gemma4TextExperts absent on transformers {_TX_VERSION} "
            "(5.0+-only)"
        )
    fwd = _assert_method_exists(cls, "forward", "gemma4_moe.py")
    _assert_params_superset(
        fwd,
        required=["hidden_states"],
        zoo_file="gemma4_moe.py",
        label="Gemma4TextExperts.forward",
    )


def test_gemma4_text_decoder_layer_init_signature():
    """gemma4_moe.py:287 patches Gemma4TextDecoderLayer.__init__(self,
    config, layer_idx)."""
    cls = _try_get_class(
        "transformers.models.gemma4.modeling_gemma4", "Gemma4TextDecoderLayer",
    )
    if cls is None:
        pytest.skip(
            f"Gemma4TextDecoderLayer absent on transformers {_TX_VERSION} "
            "(5.0+-only)"
        )
    _assert_params_superset(
        cls.__init__,
        required=["config", "layer_idx"],
        zoo_file="gemma4_moe.py",
        label="Gemma4TextDecoderLayer.__init__",
    )


def test_gemma4_text_moe_block_forward_signature():
    """gemma4_moe.py:301 patches Gemma4TextMoEBlock.forward(self,
    hidden_states, top_k_index, top_k_weights)."""
    cls = _try_get_class(
        "transformers.models.gemma4.modeling_gemma4", "Gemma4TextMoEBlock",
    )
    if cls is None:
        pytest.skip(
            f"Gemma4TextMoEBlock absent on transformers {_TX_VERSION} "
            "(5.0+-only legacy MoE layout)"
        )
    fwd = _assert_method_exists(cls, "forward", "gemma4_moe.py")
    _assert_params_superset(
        fwd,
        required=["hidden_states"],
        zoo_file="gemma4_moe.py",
        label="Gemma4TextMoEBlock.forward",
    )


def test_gemma4_for_conditional_generation_forward_named_params():
    """gemma4_moe.py:208 patches
    Gemma4ForConditionalGeneration.forward; pin by-name kwargs
    (input_ids, attention_mask, ..., logits_to_keep)."""
    cls = _try_get_class(
        "transformers.models.gemma4.modeling_gemma4",
        "Gemma4ForConditionalGeneration",
    )
    if cls is None:
        pytest.skip(
            f"Gemma4ForConditionalGeneration absent on transformers "
            f"{_TX_VERSION} (5.0+-only)"
        )
    fwd = _assert_method_exists(cls, "forward", "gemma4_moe.py")
    _assert_params_superset(
        fwd,
        required=[
            "input_ids", "attention_mask", "position_ids", "past_key_values",
            "inputs_embeds", "labels", "use_cache", "logits_to_keep",
        ],
        zoo_file="gemma4_moe.py",
        label="Gemma4ForConditionalGeneration.forward",
    )


def test_gemma4_causal_lm_output_with_past_kwargs():
    """gemma4_moe.py:189 constructs Gemma4CausalLMOutputWithPast (loss,
    logits, past_key_values, hidden_states, attentions, ...)."""
    mod = _try_get_class("transformers.models.gemma4", "modeling_gemma4")
    if mod is None:
        pytest.skip(f"gemma4 absent on transformers {_TX_VERSION}")
    cls = _try_get_class(
        "transformers.models.gemma4.modeling_gemma4",
        "Gemma4CausalLMOutputWithPast",
    )
    if cls is None:
        pytest.skip(
            f"Gemma4CausalLMOutputWithPast absent on transformers {_TX_VERSION}"
        )
    sig = inspect.signature(cls)
    field_names = list(sig.parameters.keys())
    for req in ("loss", "logits", "past_key_values", "hidden_states", "attentions"):
        if req not in field_names:
            pytest.fail(
                f"DRIFT DETECTED: zoo temporary_patches/gemma4_moe.py:189 "
                f"constructs Gemma4CausalLMOutputWithPast({req}=...) but "
                f"installed dataclass on transformers {_TX_VERSION} has fields "
                f"{field_names}"
            )


# glm4_moe.py: Glm4MoeLite{NaiveMoe,MoE}.forward (5.0+-only).

def test_glm4_moe_lite_naive_moe_forward_signature():
    """glm4_moe.py:97 patches Glm4MoeLiteNaiveMoe.forward via
    ``get_forward_moe_backend()`` (self, hidden_states, ...)."""
    cls = _try_get_class(
        "transformers.models.glm4_moe_lite.modeling_glm4_moe_lite",
        "Glm4MoeLiteNaiveMoe",
    )
    if cls is None:
        pytest.skip(
            f"Glm4MoeLiteNaiveMoe absent on transformers {_TX_VERSION} "
            "(glm4_moe_lite is 5.0+-only)"
        )
    fwd = _assert_method_exists(cls, "forward", "glm4_moe.py")
    _assert_params_superset(
        fwd,
        required=["hidden_states"],
        zoo_file="glm4_moe.py",
        label="Glm4MoeLiteNaiveMoe.forward",
    )


def test_glm4_moe_lite_moe_forward_signature():
    """glm4_moe.py:98 patches Glm4MoeLiteMoE.forward(self,
    hidden_states)."""
    cls = _try_get_class(
        "transformers.models.glm4_moe_lite.modeling_glm4_moe_lite",
        "Glm4MoeLiteMoE",
    )
    if cls is None:
        pytest.skip(
            f"Glm4MoeLiteMoE absent on transformers {_TX_VERSION} "
            "(glm4_moe_lite is 5.0+-only)"
        )
    fwd = _assert_method_exists(cls, "forward", "glm4_moe.py")
    _assert_params_superset(
        fwd,
        required=["hidden_states"],
        zoo_file="glm4_moe.py",
        label="Glm4MoeLiteMoE.forward",
    )


# gpt_oss.py: Mxfp4GptOssExperts, GptOssConfig (source-only),
# GptOssForCausalLM.forward (tail not covered by siblings).

def test_mxfp4_gpt_oss_experts_class_present_and_init_signature():
    """gpt_oss.py:433 replaces
    transformers.integrations.mxfp4.Mxfp4GptOssExperts; pin class +
    __init__(self, config)."""
    cls = _try_get_class(
        "transformers.integrations.mxfp4", "Mxfp4GptOssExperts",
    )
    if cls is None:
        pytest.skip(
            f"Mxfp4GptOssExperts absent on transformers {_TX_VERSION} "
            "(mxfp4 integrations gated)"
        )
    _assert_params_superset(
        cls.__init__,
        required=["config"],
        zoo_file="gpt_oss.py",
        label="Mxfp4GptOssExperts.__init__",
    )


def test_gpt_oss_config_class_construction_signature():
    """gpt_oss.py:2813 replaces GptOssConfig with Old_GptOssConfig; pin
    kwarg names (num_hidden_layers, num_local_experts, vocab_size, ...)."""
    cls = _try_get_class(
        "transformers.models.gpt_oss.configuration_gpt_oss", "GptOssConfig",
    )
    if cls is None:
        pytest.skip(
            f"GptOssConfig absent on transformers {_TX_VERSION}"
        )
    _assert_params_superset(
        cls.__init__,
        required=[
            "num_hidden_layers", "num_local_experts", "vocab_size",
            "hidden_size", "intermediate_size", "head_dim",
            "num_attention_heads", "num_key_value_heads",
            "sliding_window", "rope_theta",
            "max_position_embeddings", "attention_dropout",
            "num_experts_per_tok", "router_aux_loss_coef",
            "output_router_logits", "use_cache", "layer_types",
        ],
        zoo_file="gpt_oss.py",
        label="GptOssConfig.__init__",
    )


def test_gpt_oss_for_causal_lm_forward_named_params():
    """gpt_oss.py:2890 patches GptOssForCausalLM.forward with by-name
    kwargs (input_ids, attention_mask, ..., logits_to_keep)."""
    cls = _try_get_class(
        "transformers.models.gpt_oss.modeling_gpt_oss", "GptOssForCausalLM",
    )
    if cls is None:
        pytest.skip(f"GptOssForCausalLM absent on transformers {_TX_VERSION}")
    fwd = _assert_method_exists(cls, "forward", "gpt_oss.py")
    # output_attentions / output_hidden_states may be folded into
    # **kwargs on newer transformers; pin only guaranteed-stable params.
    _assert_params_superset(
        fwd,
        required=[
            "input_ids", "attention_mask", "position_ids",
            "past_key_values", "inputs_embeds", "labels", "use_cache",
            "cache_position", "logits_to_keep",
        ],
        zoo_file="gpt_oss.py",
        label="GptOssForCausalLM.forward",
    )


def test_gpt_oss_moe_causal_lm_output_kwargs():
    """gpt_oss.py:2949 constructs MoeCausalLMOutputWithPast (loss,
    aux_loss, logits, past_key_values, ...)."""
    cls = _try_get_class(
        "transformers.models.gpt_oss.modeling_gpt_oss",
        "MoeCausalLMOutputWithPast",
    )
    if cls is None:
        pytest.skip(
            f"MoeCausalLMOutputWithPast absent on transformers {_TX_VERSION}"
        )
    sig = inspect.signature(cls)
    field_names = list(sig.parameters.keys())
    for req in ("loss", "aux_loss", "logits", "past_key_values",
                "hidden_states", "attentions", "router_logits"):
        if req not in field_names:
            pytest.fail(
                f"DRIFT DETECTED: zoo temporary_patches/gpt_oss.py:2949 "
                f"constructs MoeCausalLMOutputWithPast({req}=...) but "
                f"installed dataclass on transformers {_TX_VERSION} has fields "
                f"{field_names}"
            )


def test_gpt_oss_dynamic_cache_re_export():
    """gpt_oss.py:2126 imports DynamicCache from modeling_gpt_oss (soft
    try/except). Zoo falls back to a no-op lambda silently; surface
    rename loudly."""
    mod = importlib.import_module(
        "transformers.models.gpt_oss.modeling_gpt_oss"
    )
    if not hasattr(mod, "DynamicCache"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/gpt_oss.py:2126 expects "
            "transformers.models.gpt_oss.modeling_gpt_oss.DynamicCache but "
            f"installed transformers {_TX_VERSION} has dropped the re-export. "
            "Zoo silently falls back to a lambda that returns None -- caches "
            "stop working."
        )


def test_gpt_oss_apply_rotary_pos_emb_re_export():
    """gpt_oss.py:2122 imports apply_rotary_pos_emb from
    modeling_gpt_oss; re-export pin."""
    mod = importlib.import_module(
        "transformers.models.gpt_oss.modeling_gpt_oss"
    )
    if not hasattr(mod, "apply_rotary_pos_emb"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/gpt_oss.py:2122 expects "
            "transformers.models.gpt_oss.modeling_gpt_oss.apply_rotary_pos_emb "
            "but it is missing"
        )


def test_gpt_oss_moe_model_output_with_past_present():
    """gpt_oss.py:2121 imports MoeModelOutputWithPast (patched
    GptOssModel.forward return type)."""
    mod = importlib.import_module(
        "transformers.models.gpt_oss.modeling_gpt_oss"
    )
    if not hasattr(mod, "MoeModelOutputWithPast"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/gpt_oss.py:2121 expects "
            "transformers.models.gpt_oss.modeling_gpt_oss.MoeModelOutputWithPast "
            "but it is missing"
        )


# misc.py: CsmDepthDecoderForCausalLM.forward,
# CsmForConditionalGeneration.forward (tail not covered by siblings).

def test_csm_depth_decoder_for_causal_lm_forward_named_params():
    """misc.py:239 patches CsmDepthDecoderForCausalLM.forward with named
    params (input_ids, backbone_last_hidden_state, ..., logits_to_keep).
    Resolves via ``_original_*`` stash past zoo's wrapper."""
    cls = _try_get_class(
        "transformers.models.csm.modeling_csm",
        "CsmDepthDecoderForCausalLM",
    )
    if cls is None:
        pytest.skip(
            f"CsmDepthDecoderForCausalLM absent on transformers {_TX_VERSION}"
        )
    _maybe_skip_if_patched(cls, "forward", "misc.py")
    fwd = _assert_method_exists(cls, "forward", "misc.py")
    _assert_params_superset(
        fwd,
        required=[
            "input_ids", "backbone_last_hidden_state", "attention_mask",
            "position_ids", "past_key_values", "inputs_embeds", "labels",
            "use_cache", "cache_position", "logits_to_keep",
        ],
        zoo_file="misc.py",
        label="CsmDepthDecoderForCausalLM.forward",
    )


def test_csm_for_conditional_generation_forward_named_params():
    """misc.py:373 patches CsmForConditionalGeneration.forward (input_ids,
    input_values, ..., logits_to_keep). Resolves via ``_original_*``
    stash past zoo's wrapper."""
    cls = _try_get_class(
        "transformers.models.csm.modeling_csm",
        "CsmForConditionalGeneration",
    )
    if cls is None:
        pytest.skip(
            f"CsmForConditionalGeneration absent on transformers {_TX_VERSION}"
        )
    _maybe_skip_if_patched(cls, "forward", "misc.py")
    fwd = _assert_method_exists(cls, "forward", "misc.py")
    _assert_params_superset(
        fwd,
        required=[
            "input_ids", "input_values", "attention_mask",
            "input_values_cutoffs", "position_ids", "past_key_values",
            "inputs_embeds", "labels", "use_cache", "cache_position",
            "logits_to_keep",
        ],
        zoo_file="misc.py",
        label="CsmForConditionalGeneration.forward",
    )


def test_csm_output_with_past_kwargs():
    """misc.py constructs CsmOutputWithPast; pin field set."""
    cls = _try_get_class(
        "transformers.models.csm.modeling_csm", "CsmOutputWithPast",
    )
    if cls is None:
        pytest.skip(f"CsmOutputWithPast absent on transformers {_TX_VERSION}")
    sig = inspect.signature(cls)
    field_names = list(sig.parameters.keys())
    for req in ("loss", "logits", "past_key_values"):
        if req not in field_names:
            pytest.fail(
                f"DRIFT DETECTED: zoo temporary_patches/misc.py constructs "
                f"CsmOutputWithPast({req}=...) but installed dataclass on "
                f"transformers {_TX_VERSION} has fields {field_names}"
            )


def test_csm_for_causal_lm_loss_signature():
    """misc.py:221 calls ForCausalLMLoss(logits, labels, vocab_size,
    shift_labels)."""
    fn = None
    try:
        from transformers.loss.loss_utils import ForCausalLMLoss
        fn = ForCausalLMLoss
    except Exception as exc:
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/misc.py:162 imports "
            "transformers.loss.loss_utils.ForCausalLMLoss but it is missing: "
            f"{exc}"
        )
    _assert_params_superset(
        fn,
        required=["logits", "labels", "vocab_size", "shift_labels"],
        zoo_file="misc.py",
        label="ForCausalLMLoss",
    )


def test_csm_merge_input_ids_with_input_values_param_count_realistic():
    """misc.py:770 patches CsmForConditionalGeneration.
    _merge_input_ids_with_input_values; pin method existence."""
    cls = _try_get_class(
        "transformers.models.csm.modeling_csm",
        "CsmForConditionalGeneration",
    )
    if cls is None:
        pytest.skip(f"CsmForConditionalGeneration absent on transformers {_TX_VERSION}")
    _assert_method_exists(cls, "_merge_input_ids_with_input_values", "misc.py")


def test_misc_quantizers_auto_module_present():
    """misc.py:153 patches transformers.quantizers.auto.AutoHfQuantizer."""
    try:
        mod = importlib.import_module("transformers.quantizers.auto")
    except Exception as exc:
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/misc.py:153 imports "
            "transformers.quantizers.auto but it is missing: " + str(exc)
        )
    if not hasattr(mod, "AutoHfQuantizer"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/misc.py:153 expects "
            "transformers.quantizers.auto.AutoHfQuantizer but it is missing"
        )


def test_misc_granitemoehybrid_class_present():
    """misc.py:1061 patches GraniteMoeHybridMambaLayer; pin class."""
    cls = _try_get_class(
        "transformers.models.granitemoehybrid.modeling_granitemoehybrid",
        "GraniteMoeHybridMambaLayer",
    )
    if cls is None:
        pytest.skip(
            f"GraniteMoeHybridMambaLayer absent on transformers {_TX_VERSION}"
        )


def test_misc_siglip_encoder_layer_class_present():
    """misc.py:1228 patches SiglipEncoderLayer."""
    cls = _try_get_class(
        "transformers.models.siglip.modeling_siglip", "SiglipEncoderLayer",
    )
    if cls is None:
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/misc.py:1228 expects "
            "transformers.models.siglip.modeling_siglip.SiglipEncoderLayer but "
            f"it is missing on transformers {_TX_VERSION}"
        )


def test_misc_mllama_vision_classes_present():
    """misc.py:1116-1119 imports MllamaVision{Config,Attention,MLP,Encoder,
    EncoderLayer}."""
    mod_name = "transformers.models.mllama.modeling_mllama"
    try:
        mod = importlib.import_module(mod_name)
    except Exception as exc:
        pytest.fail(
            f"DRIFT DETECTED: zoo temporary_patches/misc.py:1110 imports "
            f"{mod_name} but it is missing: {exc}"
        )
    missing = [name for name in (
        "MllamaVisionConfig", "MllamaVisionAttention", "MllamaVisionMLP",
        "MllamaVisionEncoder", "MllamaVisionEncoderLayer",
    ) if not hasattr(mod, name)]
    if missing:
        pytest.fail(
            f"DRIFT DETECTED: zoo temporary_patches/misc.py:1116-1119 imports "
            f"{missing} from {mod_name} but at least one is missing on "
            f"transformers {_TX_VERSION}"
        )


# mxfp4.py: convert_moe_packed_tensors, dequantize (tail).

def test_mxfp4_convert_moe_packed_tensors_signature():
    """mxfp4.py:173 patches
    ``transformers.integrations.mxfp4.convert_moe_packed_tensors``
    (blocks, scales, *, dtype, rows_per_chunk)."""
    mod_name = "transformers.integrations.mxfp4"
    try:
        mod = importlib.import_module(mod_name)
    except Exception as exc:
        pytest.skip(f"mxfp4 integrations unavailable: {exc}")
    fn = getattr(mod, "convert_moe_packed_tensors", None)
    if fn is None:
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/mxfp4.py:173 expects "
            "transformers.integrations.mxfp4.convert_moe_packed_tensors but "
            f"it is missing on transformers {_TX_VERSION}"
        )
    _assert_params_superset(
        fn,
        required=["blocks", "scales"],
        zoo_file="mxfp4.py",
        label="convert_moe_packed_tensors",
    )


def test_mxfp4_dequantize_signature():
    """mxfp4.py:220 patches ``mxfp4.dequantize`` (module, param_name,
    param_value, target_device, dq_param_name, **kwargs)."""
    mod_name = "transformers.integrations.mxfp4"
    try:
        mod = importlib.import_module(mod_name)
    except Exception as exc:
        pytest.skip(f"mxfp4 integrations unavailable: {exc}")
    fn = getattr(mod, "dequantize", None)
    if fn is None:
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/mxfp4.py:220 expects "
            "transformers.integrations.mxfp4.dequantize but it is missing on "
            f"transformers {_TX_VERSION}"
        )
    _assert_params_superset(
        fn,
        required=[
            "module", "param_name", "param_value", "target_device",
            "dq_param_name",
        ],
        zoo_file="mxfp4.py",
        label="dequantize",
    )
    if not _has_var_keyword(fn):
        pytest.fail(
            f"DRIFT DETECTED: zoo temporary_patches/mxfp4.py:185 forwards "
            f"by name (model=..., empty_param=..., casting_dtype=..., "
            f"to_contiguous=..., rank=..., device_mesh=...) via **kwargs but "
            f"upstream transformers.integrations.mxfp4.dequantize lost its "
            f"**kwargs catch-all on {_TX_VERSION}: {inspect.signature(fn)}"
        )


def test_mxfp4_fp4_values_constant_present():
    """mxfp4.py:113 / :227 imports ``FP4_VALUES`` constant."""
    try:
        mod = importlib.import_module("transformers.integrations.mxfp4")
    except Exception as exc:
        pytest.skip(f"mxfp4 integrations unavailable: {exc}")
    if not hasattr(mod, "FP4_VALUES"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/mxfp4.py:113 expects "
            "transformers.integrations.mxfp4.FP4_VALUES but it is missing"
        )


def test_mxfp4_shard_and_distribute_module_present():
    """mxfp4.py:181 imports
    ``transformers.integrations.tensor_parallel.shard_and_distribute_module``;
    patched dequantize delegates when device_mesh != None. Positional
    arity must be 8 for zoo's call to land."""
    try:
        mod = importlib.import_module("transformers.integrations.tensor_parallel")
    except Exception as exc:
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/mxfp4.py:181 imports "
            f"transformers.integrations.tensor_parallel but it is missing: {exc}"
        )
    fn = getattr(mod, "shard_and_distribute_module", None)
    if fn is None:
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/mxfp4.py:181 expects "
            "shard_and_distribute_module but it is missing on transformers "
            f"{_TX_VERSION}"
        )
    # 8 positionals: model, param_value, empty_param, dq_param_name,
    # casting_dtype, to_contiguous, rank, device_mesh.
    params = [
        p for p in inspect.signature(fn).parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                      inspect.Parameter.POSITIONAL_ONLY)
    ]
    if len(params) < 8:
        pytest.fail(
            f"DRIFT DETECTED: zoo temporary_patches/mxfp4.py:196 calls "
            f"shard_and_distribute_module with 8 positionals but installed "
            f"signature on transformers {_TX_VERSION} accepts only "
            f"{len(params)}: {inspect.signature(fn)}"
        )


def test_mxfp4_shard_and_distribute_set_param_kwarg_or_4x_compat():
    """mxfp4.py:196 passes ``set_param=False`` (5.x kwarg). 4.x without
    **kwargs catch-all TypeErrors on the TP path; surface as skip."""
    mod = importlib.import_module("transformers.integrations.tensor_parallel")
    fn = mod.shard_and_distribute_module
    if "set_param" in _param_names(fn):
        return  # 5.x
    if _has_var_keyword(fn):
        return  # **kwargs swallows set_param
    pytest.skip(
        f"transformers {_TX_VERSION} predates set_param kwarg on "
        "shard_and_distribute_module; zoo's TP path (device_mesh != None) "
        "requires 5.x. Non-TP users unaffected."
    )


def test_mxfp4_mxfp4_config_top_level_class():
    """mxfp4.py:93 imports top-level ``transformers.Mxfp4Config``."""
    if not hasattr(transformers, "Mxfp4Config"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/mxfp4.py:93 expects "
            "transformers.Mxfp4Config (top-level re-export) but it is "
            f"missing on transformers {_TX_VERSION}"
        )


# pixtral.py: PixtralAttention.{__init__, forward}.

def test_pixtral_attention_init_signature():
    """pixtral.py:91 patches PixtralAttention.__init__(self, config)."""
    cls = _require_class(
        "transformers.models.pixtral.modeling_pixtral",
        "PixtralAttention",
        "pixtral.py",
    )
    _assert_params_superset(
        cls.__init__,
        required=["config"],
        zoo_file="pixtral.py",
        label="PixtralAttention.__init__",
    )


def test_pixtral_attention_forward_signature():
    """pixtral.py:97 patches PixtralAttention.forward(self,
    hidden_states, attention_mask, position_embeddings, ...). Resolves
    via ``_original_*`` stash to read upstream past zoo's wrapper."""
    cls = _require_class(
        "transformers.models.pixtral.modeling_pixtral",
        "PixtralAttention",
        "pixtral.py",
    )
    _maybe_skip_if_patched(cls, "forward", "pixtral.py")
    upstream_fwd = _resolve_upstream_method(cls, "forward")
    _assert_params_superset(
        upstream_fwd,
        required=["hidden_states", "attention_mask", "position_embeddings"],
        zoo_file="pixtral.py",
        label="PixtralAttention.forward",
    )


def test_pixtral_apply_rotary_pos_emb_present():
    """pixtral.py:30 imports apply_rotary_pos_emb from modeling_pixtral;
    re-export pin."""
    mod = importlib.import_module(
        "transformers.models.pixtral.modeling_pixtral"
    )
    if not hasattr(mod, "apply_rotary_pos_emb"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/pixtral.py:30 expects "
            "transformers.models.pixtral.modeling_pixtral.apply_rotary_pos_emb "
            f"but it is missing on transformers {_TX_VERSION}"
        )


def test_pixtral_attention_init_attrs_present():
    """pixtral.py:36-47 patched __init__ reads self.config.{hidden_size,
    num_attention_heads, attention_dropout}."""
    cls = _require_class(
        "transformers.models.pixtral.configuration_pixtral",
        "PixtralVisionConfig",
        "pixtral.py",
    )
    sig = inspect.signature(cls.__init__)
    field_names = list(sig.parameters.keys())
    for req in ("hidden_size", "num_attention_heads", "attention_dropout"):
        if req not in field_names:
            pytest.fail(
                f"DRIFT DETECTED: zoo temporary_patches/pixtral.py:37-42 "
                f"reads self.config.{req} but installed PixtralVisionConfig "
                f"on transformers {_TX_VERSION} has __init__ params "
                f"{field_names}"
            )


# qwen3_5_moe.py: Qwen3_5Moe{Experts,SparseMoeBlock,ForCausalLM}.forward
# (5.0+-only).

def test_qwen3_5_moe_sparse_moe_block_forward_signature():
    """qwen3_5_moe.py:66 patches Qwen3_5MoeSparseMoeBlock.forward."""
    cls = _try_get_class(
        "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
        "Qwen3_5MoeSparseMoeBlock",
    )
    if cls is None:
        pytest.skip(
            f"Qwen3_5MoeSparseMoeBlock absent on transformers {_TX_VERSION} "
            "(qwen3_5_moe is 5.0+-only)"
        )
    fwd = _assert_method_exists(cls, "forward", "qwen3_5_moe.py")
    _assert_params_superset(
        fwd,
        required=["hidden_states"],
        zoo_file="qwen3_5_moe.py",
        label="Qwen3_5MoeSparseMoeBlock.forward",
    )


def test_qwen3_5_moe_experts_forward_signature():
    """qwen3_5_moe.py:56 patches Qwen3_5MoeExperts.forward."""
    cls = _try_get_class(
        "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
        "Qwen3_5MoeExperts",
    )
    if cls is None:
        pytest.skip(
            f"Qwen3_5MoeExperts absent on transformers {_TX_VERSION} "
            "(qwen3_5_moe is 5.0+-only)"
        )
    fwd = _assert_method_exists(cls, "forward", "qwen3_5_moe.py")
    _assert_params_superset(
        fwd,
        required=["hidden_states"],
        zoo_file="qwen3_5_moe.py",
        label="Qwen3_5MoeExperts.forward",
    )


def test_qwen3_5_moe_for_causal_lm_class_present():
    """qwen3_5_moe.py:77-78 reads Qwen3_5MoeForCausalLM +
    MoeCausalLMOutputWithPast (GRPO hidden-states patch)."""
    mod = _try_get_class(
        "transformers.models.qwen3_5_moe", "modeling_qwen3_5_moe",
    )
    if mod is None:
        pytest.skip(
            f"qwen3_5_moe absent on transformers {_TX_VERSION}"
        )
    cls = _try_get_class(
        "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
        "Qwen3_5MoeForCausalLM",
    )
    if cls is None:
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/qwen3_5_moe.py:77 expects "
            "Qwen3_5MoeForCausalLM but it is missing despite the parent module "
            "existing -- this is a real drift, not a version gate"
        )
    moe_out = _try_get_class(
        "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
        "MoeCausalLMOutputWithPast",
    )
    if moe_out is None:
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/qwen3_5_moe.py:78 expects "
            "Qwen3_5MoeForCausalLM.MoeCausalLMOutputWithPast but it is missing"
        )


# qwen3_moe.py: Qwen3MoeExperts.forward (5.0+),
# Qwen3MoeForCausalLM.forward (tail).

def test_qwen3_moe_experts_forward_signature_5x():
    """qwen3_moe.py:339 patches Qwen3MoeExperts.forward(hidden_states,
    top_k_index, top_k_weights) on 5.0+ stacked-experts branch."""
    cls = _try_get_class(
        "transformers.models.qwen3_moe.modeling_qwen3_moe", "Qwen3MoeExperts",
    )
    if cls is None:
        pytest.skip(
            f"Qwen3MoeExperts absent on transformers {_TX_VERSION} "
            "(5.0+-only; zoo gracefully patches the old SparseMoeBlock instead)"
        )
    fwd = _assert_method_exists(cls, "forward", "qwen3_moe.py")
    _assert_params_superset(
        fwd,
        required=["hidden_states"],
        zoo_file="qwen3_moe.py",
        label="Qwen3MoeExperts.forward",
    )


def test_qwen3_moe_for_causal_lm_forward_named_params():
    """qwen3_moe.py:351 patches Qwen3MoeForCausalLM.forward via
    ``_patch_causal_lm_forward_for_hidden_states`` (qwen3_moe.py:138)."""
    cls = _try_get_class(
        "transformers.models.qwen3_moe.modeling_qwen3_moe",
        "Qwen3MoeForCausalLM",
    )
    if cls is None:
        pytest.skip(
            f"Qwen3MoeForCausalLM absent on transformers {_TX_VERSION}"
        )
    fwd = _assert_method_exists(cls, "forward", "qwen3_moe.py")
    _assert_params_superset(
        fwd,
        required=[
            "input_ids", "attention_mask", "position_ids", "past_key_values",
            "inputs_embeds", "labels", "use_cache", "output_router_logits",
            "cache_position", "logits_to_keep",
        ],
        zoo_file="qwen3_moe.py",
        label="Qwen3MoeForCausalLM.forward",
    )


def test_qwen3_moe_for_causal_lm_output_class_present():
    """qwen3_moe.py:349 imports MoeCausalLMOutputWithPast from
    modeling_qwen3_moe."""
    mod = importlib.import_module(
        "transformers.models.qwen3_moe.modeling_qwen3_moe"
    )
    if not hasattr(mod, "MoeCausalLMOutputWithPast"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/qwen3_moe.py:349 expects "
            "transformers.models.qwen3_moe.modeling_qwen3_moe."
            f"MoeCausalLMOutputWithPast but it is missing on transformers "
            f"{_TX_VERSION}"
        )


# qwen3_next_moe.py: Qwen3Next{Experts,ForCausalLM}.forward tail.

def test_qwen3_next_experts_forward_signature():
    """qwen3_next_moe.py:57 patches Qwen3NextExperts.forward(self,
    hidden_states, ...) on 5.0+."""
    cls = _try_get_class(
        "transformers.models.qwen3_next.modeling_qwen3_next",
        "Qwen3NextExperts",
    )
    if cls is None:
        pytest.skip(
            f"Qwen3NextExperts absent on transformers {_TX_VERSION} "
            "(5.0+-only)"
        )
    fwd = _assert_method_exists(cls, "forward", "qwen3_next_moe.py")
    _assert_params_superset(
        fwd,
        required=["hidden_states"],
        zoo_file="qwen3_next_moe.py",
        label="Qwen3NextExperts.forward",
    )


def test_qwen3_next_for_causal_lm_forward_named_params():
    """qwen3_next_moe.py:79 patches Qwen3NextForCausalLM.forward via
    ``_patch_causal_lm_forward_for_hidden_states``."""
    cls = _try_get_class(
        "transformers.models.qwen3_next.modeling_qwen3_next",
        "Qwen3NextForCausalLM",
    )
    if cls is None:
        pytest.skip(
            f"Qwen3NextForCausalLM absent on transformers {_TX_VERSION}"
        )
    fwd = _assert_method_exists(cls, "forward", "qwen3_next_moe.py")
    _assert_params_superset(
        fwd,
        required=[
            "input_ids", "attention_mask", "position_ids", "past_key_values",
            "inputs_embeds", "labels", "use_cache", "output_router_logits",
            "cache_position", "logits_to_keep",
        ],
        zoo_file="qwen3_next_moe.py",
        label="Qwen3NextForCausalLM.forward",
    )


# qwen3_vl_moe.py: Qwen3VLMoeForConditionalGeneration.forward (tail).

def test_qwen3_vl_moe_for_conditional_generation_forward_named_params():
    """qwen3_vl_moe.py:401 patches
    Qwen3VLMoeForConditionalGeneration.forward with by-name kwargs
    (input_ids, attention_mask, ..., logits_to_keep)."""
    cls = _try_get_class(
        "transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe",
        "Qwen3VLMoeForConditionalGeneration",
    )
    if cls is None:
        pytest.skip(
            f"Qwen3VLMoeForConditionalGeneration absent on transformers "
            f"{_TX_VERSION}"
        )
    fwd = _assert_method_exists(cls, "forward", "qwen3_vl_moe.py")
    _assert_params_superset(
        fwd,
        required=[
            "input_ids", "attention_mask", "position_ids", "past_key_values",
            "inputs_embeds", "labels", "pixel_values", "pixel_values_videos",
            "image_grid_thw", "video_grid_thw", "cache_position",
            "logits_to_keep",
        ],
        zoo_file="qwen3_vl_moe.py",
        label="Qwen3VLMoeForConditionalGeneration.forward",
    )


def test_qwen3_vl_moe_causal_lm_output_with_past_kwargs():
    """qwen3_vl_moe.py:466 constructs Qwen3VLMoeCausalLMOutputWithPast
    (loss, aux_loss, ..., rope_deltas)."""
    cls = _try_get_class(
        "transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe",
        "Qwen3VLMoeCausalLMOutputWithPast",
    )
    if cls is None:
        pytest.skip(
            f"Qwen3VLMoeCausalLMOutputWithPast absent on transformers "
            f"{_TX_VERSION}"
        )
    sig = inspect.signature(cls)
    field_names = list(sig.parameters.keys())
    for req in ("loss", "logits", "past_key_values", "hidden_states",
                "attentions", "rope_deltas"):
        if req not in field_names:
            pytest.fail(
                f"DRIFT DETECTED: zoo temporary_patches/qwen3_vl_moe.py:466 "
                f"constructs Qwen3VLMoeCausalLMOutputWithPast({req}=...) but "
                f"installed dataclass on transformers {_TX_VERSION} has "
                f"fields {field_names}"
            )


def test_qwen3_vl_moe_text_top_k_router_class_present():
    """qwen3_vl_moe.py:326 expects ``self.gate ==
    Qwen3VLMoeTextTopKRouter`` on 5.x; tuple-unpack fallback otherwise."""
    cls = _try_get_class(
        "transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe",
        "Qwen3VLMoeTextTopKRouter",
    )
    if cls is None:
        # 5.x-only; zoo's tuple-unpack fallback (qwen3_vl_moe.py:333) handles.
        pytest.skip(
            f"Qwen3VLMoeTextTopKRouter absent on transformers {_TX_VERSION} "
            "(zoo gracefully falls back to old-style logit gate)"
        )


def test_qwen3_vl_moe_text_experts_class_present():
    """qwen3_vl_moe.py:73 imports Qwen3VLMoeTextExperts."""
    cls = _try_get_class(
        "transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe",
        "Qwen3VLMoeTextExperts",
    )
    if cls is None:
        pytest.skip(
            f"Qwen3VLMoeTextExperts absent on transformers {_TX_VERSION}"
        )


def test_qwen3_vl_moe_act2fn_dict_present():
    """qwen3_vl_moe.py:201 imports ACT2FN; patched __init__ does
    ``self.act_fn = ACT2FN[config.hidden_act]``."""
    from transformers.activations import ACT2FN  # noqa: F401
    if "silu" not in ACT2FN:
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/qwen3_vl_moe.py:236 expects "
            "transformers.activations.ACT2FN['silu'] but the key is missing"
        )


# moe_utils.py / moe_bnb.py / flex_attention_bwd.py: helper modules
# (no patch_function call sites); consumer sites covered elsewhere.

def test_moe_utils_param_wrapper_target_present():
    """moe_utils.py registers patches against
    peft.tuners.lora.layer.ParamWrapper (split-LoRA grouped-GEMM)."""
    peft = pytest.importorskip("peft")
    try:
        from peft.tuners.lora.layer import ParamWrapper  # noqa: F401
    except Exception as exc:
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/moe_utils.py expects "
            "peft.tuners.lora.layer.ParamWrapper but it is missing: " + str(exc)
        )


# misc.py additional patch sites: config-mapping, tokenizer attrs,
# mask-utils, modernbert, lfm2, peft dispatch, trl, vllm, qwen2-vl shims.

def test_misc_config_mapping_present_for_ministral3_register():
    """misc.py:47-53 imports CONFIG_MAPPING from
    transformers.models.auto.configuration_auto + calls
    ``.register(...)``."""
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    except Exception as exc:
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/misc.py:47 expects "
            f"transformers.models.auto.configuration_auto.CONFIG_MAPPING "
            f"but it is missing: {exc}"
        )
    if not hasattr(CONFIG_MAPPING, "register"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/misc.py:53 calls "
            "CONFIG_MAPPING.register(...) but the installed CONFIG_MAPPING "
            f"on transformers {_TX_VERSION} has no register attribute "
            f"(type {type(CONFIG_MAPPING).__name__})"
        )


def test_misc_ministral_config_top_level_import():
    """misc.py:48 imports top-level ``transformers.MinistralConfig``
    (ministral3 alias target)."""
    if not hasattr(transformers, "MinistralConfig"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/misc.py:48 expects "
            "transformers.MinistralConfig at top level but it is missing on "
            f"{_TX_VERSION}"
        )


def test_misc_pretrained_tokenizer_base_convert_added_tokens_method():
    """misc.py:67 expects
    ``PreTrainedTokenizerBase.convert_added_tokens`` classmethod."""
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    if not hasattr(PreTrainedTokenizerBase, "convert_added_tokens"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/misc.py:67 expects "
            "PreTrainedTokenizerBase.convert_added_tokens but it is missing "
            f"on transformers {_TX_VERSION}"
        )


def test_misc_added_token_class_present():
    """misc.py:63 / :75 imports AddedToken; patched convert_added_tokens
    constructs ``AddedToken(content=...)``."""
    from transformers.tokenization_utils_base import AddedToken
    sig = inspect.signature(AddedToken)
    field_names = list(sig.parameters.keys())
    for req in ("content",):
        if req not in field_names:
            pytest.fail(
                f"DRIFT DETECTED: zoo temporary_patches/misc.py:75 constructs "
                f"AddedToken(content=...) but installed AddedToken on "
                f"transformers {_TX_VERSION} has __init__ params {field_names}"
            )


def test_misc_pretrained_tokenizer_base_init_takes_kwargs():
    """misc.py:97 wraps PreTrainedTokenizerBase.__init__(self, **kwargs)
    (rejects / coerces extra_special_tokens)."""
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    if not _has_var_keyword(PreTrainedTokenizerBase.__init__):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/misc.py:97 patched "
            "PreTrainedTokenizerBase.__init__(self, **kwargs) but installed "
            f"signature has no VAR_KEYWORD: "
            f"{inspect.signature(PreTrainedTokenizerBase.__init__)}"
        )


def test_misc_masking_utils_create_block_mask_available_or_compile_flag():
    """misc.py:391-445 imports BlockMask / create_block_mask and rewrites
    masking_utils.{create_causal_mask, create_sliding_window_causal_mask,
    create_masks_for_generate}."""
    masking = importlib.import_module("transformers.masking_utils")
    for name in ("create_causal_mask", "create_sliding_window_causal_mask",
                 "create_masks_for_generate"):
        if not hasattr(masking, name):
            pytest.fail(
                f"DRIFT DETECTED: zoo temporary_patches/misc.py:414-445 "
                f"reads transformers.masking_utils.{name} but it is missing "
                f"on transformers {_TX_VERSION}"
            )


def test_misc_generation_utils_create_masks_for_generate():
    """misc.py:447 reassigns
    ``transformers.generation.utils.create_masks_for_generate``."""
    gu = importlib.import_module("transformers.generation.utils")
    if not hasattr(gu, "create_masks_for_generate"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/misc.py:447 expects "
            "transformers.generation.utils.create_masks_for_generate but it "
            f"is missing on transformers {_TX_VERSION}"
        )


def test_misc_masking_utils_padding_and_packed_helpers():
    """misc.py:472 / :490 wraps padding_mask_function /
    packed_sequence_mask_function on masking_utils (hasattr-gated)."""
    masking = importlib.import_module("transformers.masking_utils")
    if hasattr(masking, "padding_mask_function"):
        if not callable(masking.padding_mask_function):
            pytest.fail(
                "DRIFT DETECTED: zoo temporary_patches/misc.py:472 expects "
                "callable transformers.masking_utils.padding_mask_function "
                "but it is not callable"
            )
    if hasattr(masking, "packed_sequence_mask_function"):
        if not callable(masking.packed_sequence_mask_function):
            pytest.fail(
                "DRIFT DETECTED: zoo temporary_patches/misc.py:490 expects "
                "callable transformers.masking_utils.packed_sequence_mask_function"
            )


def test_misc_sdpa_attention_forward_present():
    """misc.py:525-530 patches
    ``transformers.integrations.sdpa_attention.sdpa_attention_forward``."""
    try:
        mod = importlib.import_module("transformers.integrations.sdpa_attention")
    except Exception as exc:
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/misc.py:525 imports "
            f"transformers.integrations.sdpa_attention but it is missing: {exc}"
        )
    if not hasattr(mod, "sdpa_attention_forward"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/misc.py:530 expects "
            "transformers.integrations.sdpa_attention.sdpa_attention_forward "
            f"but it is missing on transformers {_TX_VERSION}"
        )


def test_misc_all_attention_functions_modeling_utils_top_level():
    """misc.py:526 imports
    ``transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS``."""
    mu = importlib.import_module("transformers.modeling_utils")
    if not hasattr(mu, "ALL_ATTENTION_FUNCTIONS"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/misc.py:526 expects "
            "transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS but it is "
            f"missing on transformers {_TX_VERSION}"
        )


def test_misc_modernbert_model_update_attention_mask_present():
    """misc.py:662 patches
    ``ModernBertModel._update_attention_mask`` (SDPA-stride fix)."""
    cls = _try_get_class(
        "transformers.models.modernbert.modeling_modernbert",
        "ModernBertModel",
    )
    if cls is None:
        pytest.skip(f"ModernBertModel absent on transformers {_TX_VERSION}")
    if not hasattr(cls, "_update_attention_mask"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/misc.py:662 expects "
            "ModernBertModel._update_attention_mask but it is missing on "
            f"transformers {_TX_VERSION}; the modernbert SDPA-stride fix no-ops"
        )
    sig = inspect.signature(cls._update_attention_mask)
    _assert_params_superset(
        cls._update_attention_mask,
        required=["attention_mask"],
        zoo_file="misc.py",
        label="ModernBertModel._update_attention_mask",
    )


def test_misc_csm_for_conditional_generation_merge_input_ids_target_present():
    """misc.py:687 patches
    ``CsmForConditionalGeneration._merge_input_ids_with_input_values``
    (input_ids, input_values, input_values_cutoffs, labels)."""
    cls = _try_get_class(
        "transformers.models.csm.modeling_csm", "CsmForConditionalGeneration",
    )
    if cls is None:
        pytest.skip(f"CsmForConditionalGeneration absent on transformers {_TX_VERSION}")
    method = _assert_method_exists(
        cls, "_merge_input_ids_with_input_values", "misc.py",
    )
    _assert_params_superset(
        method,
        required=["input_ids", "input_values", "input_values_cutoffs", "labels"],
        zoo_file="misc.py",
        label="CsmForConditionalGeneration._merge_input_ids_with_input_values",
    )


def test_misc_lfm2_vl_multimodal_projector_class_present():
    """misc.py:1247 patches Lfm2VlMultiModalProjector.{__init__,
    forward}; transformers pre-5.0.0-gated."""
    cls = _try_get_class(
        "transformers.models.lfm2_vl.modeling_lfm2_vl",
        "Lfm2VlMultiModalProjector",
    )
    if cls is None:
        pytest.skip(
            f"Lfm2VlMultiModalProjector absent on transformers {_TX_VERSION}"
        )
    _assert_params_superset(
        cls.__init__,
        required=["config"],
        zoo_file="misc.py",
        label="Lfm2VlMultiModalProjector.__init__",
    )


def test_misc_peft_dispatch_bnb_4bit_target_present():
    """misc.py:1289-1290 patches
    ``peft.tuners.lora.bnb.dispatch_bnb_4bit`` (target, adapter_name)."""
    peft = pytest.importorskip("peft")
    try:
        import peft.tuners.lora.bnb as peft_bnb
    except Exception as exc:
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/misc.py:1289 imports "
            f"peft.tuners.lora.bnb but it is missing: {exc}"
        )
    if not hasattr(peft_bnb, "dispatch_bnb_4bit"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/misc.py:1290 expects "
            "peft.tuners.lora.bnb.dispatch_bnb_4bit but it is missing"
        )
    sig = inspect.signature(peft_bnb.dispatch_bnb_4bit)
    _assert_params_superset(
        peft_bnb.dispatch_bnb_4bit,
        required=["target", "adapter_name"],
        zoo_file="misc.py",
        label="peft.tuners.lora.bnb.dispatch_bnb_4bit",
    )


def test_misc_trl_push_to_hub_target_training_arguments_to_dict():
    """misc.py:1333-1334 patches ``TrainingArguments.to_dict`` on 5.0+."""
    if not hasattr(transformers, "TrainingArguments"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/misc.py:1333 expects "
            f"transformers.TrainingArguments but it is missing on {_TX_VERSION}"
        )
    if not hasattr(transformers.TrainingArguments, "to_dict"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/misc.py:1334 expects "
            "TrainingArguments.to_dict() but it is missing on transformers "
            f"{_TX_VERSION}"
        )


def test_misc_trl_vision_model_mapping_target_module_present():
    """misc.py:1363-1371 reads
    ``MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES`` (5.0+) or
    ``MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES`` (legacy); at least one
    must exist."""
    auto_mod = importlib.import_module(
        "transformers.models.auto.modeling_auto"
    )
    new_name = getattr(
        auto_mod, "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES", None,
    )
    old_name = getattr(
        auto_mod, "MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES", None,
    )
    if new_name is None and old_name is None:
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/misc.py:1363-1371 reads "
            "either MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES (5.0+) or "
            "MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES (legacy) but BOTH are "
            f"missing on transformers {_TX_VERSION} -- DPO + vision broken"
        )


def test_misc_apply_chat_template_signature_has_return_dict():
    """misc.py:1446-1455 sets ``kwargs['return_dict']=False`` when
    tokenize=True; needs ``return_dict`` in
    PreTrainedTokenizerBase.apply_chat_template signature."""
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    sig = inspect.signature(PreTrainedTokenizerBase.apply_chat_template)
    if "return_dict" not in sig.parameters and not _has_var_keyword(
        PreTrainedTokenizerBase.apply_chat_template
    ):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/misc.py:1455 sets "
            "kwargs['return_dict']=False when tokenize=True, but installed "
            f"apply_chat_template signature on {_TX_VERSION} has neither "
            f"return_dict nor **kwargs: {sig}"
        )


def test_misc_qwen2_vl_image_processor_class_present():
    """misc.py:1485 imports Qwen2VLImageProcessor (max_pixels /
    min_pixels properties)."""
    cls = _try_get_class(
        "transformers.models.qwen2_vl.image_processing_qwen2_vl",
        "Qwen2VLImageProcessor",
    )
    if cls is None:
        pytest.skip(
            f"Qwen2VLImageProcessor absent on transformers {_TX_VERSION}"
        )


# gpt_oss.py additional patch sites (beyond existing tests).

def test_gpt_oss_mxfp4_quantizer_class_present():
    """gpt_oss.py:124-127 patches
    ``transformers.quantizers.quantizer_mxfp4.Mxfp4HfQuantizer.is_trainable``."""
    try:
        mod = importlib.import_module("transformers.quantizers.quantizer_mxfp4")
    except Exception as exc:
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/gpt_oss.py:124 imports "
            f"transformers.quantizers.quantizer_mxfp4 but it is missing: {exc}"
        )
    if not hasattr(mod, "Mxfp4HfQuantizer"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/gpt_oss.py:127 expects "
            "transformers.quantizers.quantizer_mxfp4.Mxfp4HfQuantizer but it "
            f"is missing on transformers {_TX_VERSION}"
        )


def test_gpt_oss_mxfp4_quantizer_is_kernels_available_present():
    """gpt_oss.py:136 reassigns
    ``transformers.quantizers.quantizer_mxfp4.is_kernels_available``."""
    mod = importlib.import_module("transformers.quantizers.quantizer_mxfp4")
    if not hasattr(mod, "is_kernels_available"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/gpt_oss.py:136 expects "
            "transformers.quantizers.quantizer_mxfp4.is_kernels_available "
            f"but it is missing on transformers {_TX_VERSION}"
        )


def test_gpt_oss_modeling_module_top_level_classes_present():
    """gpt_oss.py:1060-1063 reassigns ``modeling_gpt_oss.GptOssExperts``
    and ``.GptOssTopKRouter`` (BnB 4-bit GPT-OSS shim)."""
    mod = importlib.import_module(
        "transformers.models.gpt_oss.modeling_gpt_oss"
    )
    for name in ("GptOssExperts", "GptOssTopKRouter"):
        if not hasattr(mod, name):
            pytest.fail(
                f"DRIFT DETECTED: zoo temporary_patches/gpt_oss.py:1060-1063 "
                f"reassigns modeling_gpt_oss.{name} but the symbol is missing "
                f"on transformers {_TX_VERSION}; the BnB 4-bit GPT-OSS shim "
                f"silently no-ops"
            )


def test_gpt_oss_layer_type_validation_module_path():
    """gpt_oss.py:2801-2803 reads
    ``transformers.models.gpt_oss.configuration_gpt_oss.GptOssConfig``."""
    try:
        cfg_mod = importlib.import_module(
            "transformers.models.gpt_oss.configuration_gpt_oss"
        )
    except Exception as exc:
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/gpt_oss.py:2801 imports "
            f"transformers.models.gpt_oss.configuration_gpt_oss but it is "
            f"missing: {exc}"
        )
    if not hasattr(cfg_mod, "GptOssConfig"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/gpt_oss.py:2803 expects "
            "transformers.models.gpt_oss.configuration_gpt_oss.GptOssConfig "
            f"but it is missing on {_TX_VERSION}"
        )


def test_gpt_oss_pretrained_model_present():
    """gpt_oss.py:2832 reads ``GptOssPreTrainedModel`` for
    _init_weights patch."""
    mod = importlib.import_module(
        "transformers.models.gpt_oss.modeling_gpt_oss"
    )
    if not hasattr(mod, "GptOssPreTrainedModel"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/gpt_oss.py:2832 expects "
            "GptOssPreTrainedModel but it is missing on transformers "
            f"{_TX_VERSION}"
        )


def test_gpt_oss_model_module_dynamic_cache_present():
    """gpt_oss.py:2126 fallback path pins
    ``transformers.cache_utils.DynamicCache``."""
    cu = importlib.import_module("transformers.cache_utils")
    if not hasattr(cu, "DynamicCache"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/gpt_oss.py expects "
            "transformers.cache_utils.DynamicCache but it is missing on "
            f"transformers {_TX_VERSION}"
        )


def test_gpt_oss_attention_apply_rotary_pos_emb_imported_at_attention():
    """gpt_oss.py:1875 imports ``apply_rotary_pos_emb(q, k, cos, sin)``
    (4 positionals) from modeling_gpt_oss."""
    mod = importlib.import_module(
        "transformers.models.gpt_oss.modeling_gpt_oss"
    )
    apply = getattr(mod, "apply_rotary_pos_emb", None)
    if apply is None:
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/gpt_oss.py:1875 expects "
            "modeling_gpt_oss.apply_rotary_pos_emb but it is missing"
        )
    params = [
        p for p in inspect.signature(apply).parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                      inspect.Parameter.POSITIONAL_ONLY)
    ]
    if len(params) < 4:
        pytest.fail(
            f"DRIFT DETECTED: zoo temporary_patches/gpt_oss.py calls "
            f"apply_rotary_pos_emb(q, k, cos, sin) -- 4 positionals -- but "
            f"installed signature accepts only {len(params)}: "
            f"{inspect.signature(apply)}"
        )


def test_gpt_oss_eager_attention_forward_present():
    """gpt_oss.py:2063 calls ``eager_attention_forward(self, q, k, v,
    mask, ...)`` (5+ positionals)."""
    mod = importlib.import_module(
        "transformers.models.gpt_oss.modeling_gpt_oss"
    )
    fn = getattr(mod, "eager_attention_forward", None)
    if fn is None:
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/gpt_oss.py:2063 expects "
            "modeling_gpt_oss.eager_attention_forward but it is missing on "
            f"transformers {_TX_VERSION}"
        )
    # Lenient: only positional arity (kwarg names change across releases).
    params = [
        p for p in inspect.signature(fn).parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                      inspect.Parameter.POSITIONAL_ONLY)
    ]
    if len(params) < 5:  # module + q + k + v + mask
        pytest.fail(
            f"DRIFT DETECTED: zoo temporary_patches/gpt_oss.py:2063 calls "
            f"eager_attention_forward(self, q, k, v, mask, ...) -- 5+ "
            f"positionals -- but installed signature accepts only "
            f"{len(params)}: {inspect.signature(fn)}"
        )


# gemma.py transitive deps: Gemma3DecoderLayer, Gemma3TextModel,
# Gemma3PreTrainedModel.

def test_gemma3_decoder_layer_class_present():
    """gemma.py imports Gemma3Attention; the decoder-layer parent
    Gemma3DecoderLayer must exist."""
    cls = _try_get_class(
        "transformers.models.gemma3.modeling_gemma3", "Gemma3DecoderLayer",
    )
    if cls is None:
        pytest.fail(
            "DRIFT DETECTED: gemma.py expects Gemma3DecoderLayer (parent of "
            f"Gemma3Attention) but it is missing on transformers {_TX_VERSION}"
        )


def test_gemma3_text_model_class_present():
    """gemma.py:233 references Gemma3Model; pin Gemma3TextModel
    (LM head backbone)."""
    cls = _try_get_class(
        "transformers.models.gemma3.modeling_gemma3", "Gemma3TextModel",
    )
    if cls is None:
        pytest.fail(
            "DRIFT DETECTED: gemma.py expects Gemma3TextModel but it is "
            f"missing on transformers {_TX_VERSION}"
        )


def test_gemma3_pre_trained_model_class_present():
    """gemma.py touches Gemma3 surfaces; pin Gemma3PreTrainedModel base."""
    cls = _try_get_class(
        "transformers.models.gemma3.modeling_gemma3", "Gemma3PreTrainedModel",
    )
    if cls is None:
        pytest.fail(
            "DRIFT DETECTED: gemma.py expects Gemma3PreTrainedModel but it "
            f"is missing on transformers {_TX_VERSION}"
        )


def test_gemma3_processor_kwargs_class_present():
    """gemma.py:218 reads ``Gemma3ProcessorKwargs`` (Unpack type for
    __call__)."""
    mod = importlib.import_module(
        "transformers.models.gemma3.processing_gemma3"
    )
    if not hasattr(mod, "Gemma3ProcessorKwargs"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/gemma.py:218 expects "
            "Gemma3ProcessorKwargs but it is missing on transformers "
            f"{_TX_VERSION}"
        )


# gemma3n.py additional pins.

def test_gemma3n_for_conditional_generation_class_present():
    """gemma3n.py patches Gemma3nModel.get_placeholder_mask; pin
    Gemma3nForConditionalGeneration."""
    cls = _try_get_class(
        "transformers.models.gemma3n.modeling_gemma3n",
        "Gemma3nForConditionalGeneration",
    )
    if cls is None:
        pytest.fail(
            "DRIFT DETECTED: gemma3n.py expects Gemma3nForConditionalGeneration "
            f"but it is missing on transformers {_TX_VERSION}"
        )


def test_gemma3n_RMSNorm_class_present():
    """gemma3n.py:53 Gemma3nRMSNorm_forward delegate needs upstream
    Gemma3nRMSNorm class (self.weight / self._norm references)."""
    cls = _try_get_class(
        "transformers.models.gemma3n.modeling_gemma3n", "Gemma3nRMSNorm",
    )
    if cls is None:
        pytest.fail(
            "DRIFT DETECTED: gemma3n.py:53 helper expects Gemma3nRMSNorm but "
            f"it is missing on transformers {_TX_VERSION}"
        )


# qwen3_moe.py / qwen3_5_moe.py / qwen3_next_moe.py shared deps.

def test_qwen3_moe_rms_norm_class_present():
    """qwen3_moe.py patched forward calls .gate / .experts; pin
    Qwen3MoeRMSNorm sibling so namespace rename surfaces."""
    cls = _try_get_class(
        "transformers.models.qwen3_moe.modeling_qwen3_moe", "Qwen3MoeRMSNorm",
    )
    if cls is None:
        pytest.fail(
            "DRIFT DETECTED: qwen3_moe.py expects Qwen3MoeRMSNorm class "
            f"namespace on transformers {_TX_VERSION}"
        )


def test_qwen3_moe_pre_trained_model_present():
    """qwen3_moe.py patches Qwen3MoeForCausalLM.forward; pin
    Qwen3MoePreTrainedModel base."""
    cls = _try_get_class(
        "transformers.models.qwen3_moe.modeling_qwen3_moe",
        "Qwen3MoePreTrainedModel",
    )
    if cls is None:
        pytest.fail(
            "DRIFT DETECTED: qwen3_moe.py expects Qwen3MoePreTrainedModel "
            f"on transformers {_TX_VERSION}"
        )


def test_qwen3_moe_model_present():
    """qwen3_moe.py:170-179 calls self.model(...); self.model is
    Qwen3MoeModel."""
    cls = _try_get_class(
        "transformers.models.qwen3_moe.modeling_qwen3_moe", "Qwen3MoeModel",
    )
    if cls is None:
        pytest.fail(
            "DRIFT DETECTED: qwen3_moe.py:170 calls self.model(...) where "
            "self is Qwen3MoeForCausalLM -- Qwen3MoeModel is missing on "
            f"transformers {_TX_VERSION}"
        )


def test_qwen3_next_model_class_present():
    """qwen3_next_moe.py needs inner ``Qwen3NextModel``."""
    cls = _try_get_class(
        "transformers.models.qwen3_next.modeling_qwen3_next", "Qwen3NextModel",
    )
    if cls is None:
        pytest.fail(
            "DRIFT DETECTED: qwen3_next_moe.py expects Qwen3NextModel on "
            f"transformers {_TX_VERSION}"
        )


def test_qwen3_vl_moe_text_model_class_present():
    """qwen3_vl_moe.py patches Qwen3VLMoeTextSparseMoeBlock; pin parent
    Qwen3VLMoeTextModel."""
    cls = _try_get_class(
        "transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe",
        "Qwen3VLMoeTextModel",
    )
    if cls is None:
        pytest.fail(
            "DRIFT DETECTED: qwen3_vl_moe.py expects Qwen3VLMoeTextModel on "
            f"transformers {_TX_VERSION}"
        )


# Cache-output-class signature pins (constructed by zoo patch wrappers).

def test_modeling_outputs_causal_lm_output_with_past_kwargs():
    """deepseek_v3_moe.py:200 + qwen3_next_moe.py construct
    ``transformers.modeling_outputs.CausalLMOutputWithPast(loss, logits,
    past_key_values, hidden_states, attentions)``."""
    from transformers.modeling_outputs import CausalLMOutputWithPast
    sig = inspect.signature(CausalLMOutputWithPast)
    field_names = list(sig.parameters.keys())
    for req in ("loss", "logits", "past_key_values", "hidden_states",
                "attentions"):
        if req not in field_names:
            pytest.fail(
                f"DRIFT DETECTED: zoo constructs CausalLMOutputWithPast"
                f"({req}=...) but installed dataclass has fields {field_names}"
            )


def test_modeling_outputs_moe_causal_lm_output_with_past_kwargs():
    """qwen3_moe.py:191 constructs ``MoeCausalLMOutputWithPast(loss,
    logits, past_key_values, hidden_states, attentions, aux_loss,
    router_logits)`` via top-level modeling_outputs."""
    from transformers.modeling_outputs import MoeCausalLMOutputWithPast
    sig = inspect.signature(MoeCausalLMOutputWithPast)
    field_names = list(sig.parameters.keys())
    for req in ("loss", "logits", "past_key_values", "hidden_states",
                "attentions", "aux_loss", "router_logits"):
        if req not in field_names:
            pytest.fail(
                f"DRIFT DETECTED: zoo constructs MoeCausalLMOutputWithPast"
                f"({req}=...) but installed dataclass has fields {field_names}"
            )


# Caches the patches require.

def test_static_cache_class_present():
    """gemma.py:255 isinstance(past_key_values, StaticCache)."""
    cu = importlib.import_module("transformers.cache_utils")
    if not hasattr(cu, "StaticCache"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/gemma.py:255 uses "
            "transformers.cache_utils.StaticCache via isinstance but it is "
            f"missing on transformers {_TX_VERSION}"
        )


def test_hybrid_cache_class_present():
    """gemma.py:260 isinstance(past_key_values, HybridCache)."""
    cu = importlib.import_module("transformers.cache_utils")
    if not hasattr(cu, "HybridCache"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/gemma.py:260 uses "
            "transformers.cache_utils.HybridCache but it is missing on "
            f"transformers {_TX_VERSION}"
        )


# bitsandbytes.py: Linear4bit __init__ signature pin.

def test_bitsandbytes_linear4bit_init_signature():
    """bitsandbytes.py:46-47 needs
    ``bitsandbytes.nn.modules.Linear4bit.__init__(input_features,
    output_features, ...)`` (zoo's patched forward dereferences
    self.weight / self.bias)."""
    bnb = pytest.importorskip("bitsandbytes")
    cls = getattr(bnb.nn.modules, "Linear4bit", None)
    if cls is None:
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/bitsandbytes.py:46 expects "
            "bitsandbytes.nn.modules.Linear4bit but it is missing"
        )
    _assert_params_superset(
        cls.__init__,
        required=["input_features", "output_features"],
        zoo_file="bitsandbytes.py",
        label="bitsandbytes.nn.modules.Linear4bit.__init__",
    )


# pixtral.py: PixtralVisionConfig.

def test_pixtral_vision_config_class_present():
    """pixtral.py:36 reads self.config attrs; pin PixtralVisionConfig."""
    cls = _try_get_class(
        "transformers.models.pixtral.configuration_pixtral",
        "PixtralVisionConfig",
    )
    if cls is None:
        pytest.fail(
            "DRIFT DETECTED: pixtral.py:36 reads self.config attrs but "
            "PixtralVisionConfig is missing on transformers "
            f"{_TX_VERSION}"
        )


# gemma3n.py: Gemma3nTextConfig pin (AltUp.predict config typing).

def test_gemma3n_text_config_class_present():
    """gemma3n.py:101-114 reads
    self.config.{altup_num_inputs, altup_active_idx} in AltUp.predict."""
    cls = _try_get_class(
        "transformers.models.gemma3n.configuration_gemma3n",
        "Gemma3nTextConfig",
    )
    if cls is None:
        pytest.fail(
            "DRIFT DETECTED: gemma3n.py:101 reads self.config.altup_active_idx "
            "but Gemma3nTextConfig is missing on transformers "
            f"{_TX_VERSION}"
        )
    sig = inspect.signature(cls.__init__)
    field_names = list(sig.parameters.keys())
    for req in ("altup_num_inputs", "altup_active_idx"):
        if req not in field_names:
            pytest.fail(
                f"DRIFT DETECTED: gemma3n.py:101-114 reads self.config.{req} "
                f"but installed Gemma3nTextConfig __init__ has params "
                f"{field_names}"
            )


# Auto-attention function dictionary for gemma3 patch chain.

def test_gemma3_eager_attention_forward_kwargs_supported():
    """gemma.py:407-412 calls ``eager_attention_forward(...,
    dropout, scaling, sliding_window, **kwargs)``."""
    from transformers.models.gemma3.modeling_gemma3 import eager_attention_forward
    if not _has_var_keyword(eager_attention_forward):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/gemma.py:412 calls "
            "eager_attention_forward(..., **kwargs) but installed signature "
            f"on transformers {_TX_VERSION} has no VAR_KEYWORD: "
            f"{inspect.signature(eager_attention_forward)}"
        )


# Sanity: temporary_patches/ inventory.

def test_temporary_patches_directory_has_expected_files():
    """Pin the floor set of patch files; new files OK, missing -> DRIFT."""
    pkg_spec = importlib.util.find_spec("unsloth_zoo.temporary_patches")
    if pkg_spec is None or not pkg_spec.submodule_search_locations:
        pytest.skip("unsloth_zoo.temporary_patches not importable as a package")
    root = pkg_spec.submodule_search_locations[0]
    files = {
        f for f in os.listdir(root)
        if f.endswith(".py")
        and f not in ("__init__.py", "utils.py", "common.py")
    }
    must_have = {
        "bitsandbytes.py", "deepseek_v3_moe.py", "gemma.py", "gemma3n.py",
        "gemma4.py", "gemma4_moe.py", "glm4_moe.py", "gpt_oss.py",
        "ministral.py", "misc.py", "mxfp4.py", "pixtral.py",
        "qwen3_moe.py", "qwen3_next_moe.py", "qwen3_vl_moe.py",
    }
    missing = sorted(must_have - files)
    if missing:
        pytest.fail(
            f"DRIFT DETECTED: temporary_patches/ is missing files {missing}; "
            f"either they were renamed or dropped without updating the test"
        )


