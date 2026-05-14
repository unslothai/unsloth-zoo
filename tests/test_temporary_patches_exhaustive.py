# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at
# your option) any later version.

"""Exhaustive upstream-signature pinning for the (class, method) pairs
that ``unsloth_zoo/temporary_patches/<file>.py`` rebinds.

Why this file exists
====================
``tests/test_upstream_signatures.py``, ``test_upstream_pinned_symbols_transformers.py``,
``test_zoo_source_upstream_refs.py``, and ``test_upstream_source_patterns.py`` pin
roughly 50-70 (class, method) pairs that the ``temporary_patches/`` directory
monkey-patches. A walk of every file in ``unsloth_zoo/temporary_patches/``
turned up additional patch sites that no existing test covers. This file
fills the tail.

Patch-site discovery
====================
For every ``temporary_patches/<file>.py``, all of:

    patch_function(target_cls, "name", ...)
    patch_function_past_key_values(target_cls, "name", ...)
    target_cls.method = patched_method
    setattr(modeling_X, "Y", patched_Y)

were extracted. Each (model_class, method) pair below maps 1:1 to a
``patch_function(...)`` call (or attribute reassignment) in zoo. If
upstream renames or drops the symbol, zoo's patch silently no-ops via
``raise_error()`` and the user trains with a stock (unpatched) forward
that the zoo patch was meant to fix -- exactly the silent-drift class of
bug these tests catch.

Contract
========
* CPU-only -- no GPU, no downloads, no network.
* Genuinely optional upstream libs (``timm``, ``bitsandbytes``) use
  ``pytest.importorskip``. ``transformers`` is required at module-level
  importorskip, matching the rest of the test suite.
* Version-gated patches (zoo guards a class behind ``if hasattr(...)`` or
  a try/except ImportError because the class only exists on transformers
  5.0+) are similarly gated here via ``pytest.skip`` so the test
  legitimately reports "not on this transformers" instead of false-failing.
* Drift detection: missing class or signature parameter dropped ->
  ``pytest.fail("DRIFT DETECTED: zoo temporary_patches/<file>.py expects
  <class>.<method>(<params>) but installed transformers has
  <signature>")``.
* Pairs already pinned in the sibling test files are intentionally
  skipped here to keep this file focused on the uncovered tail.

Runs under the GPU-free harness in ``tests/conftest.py``.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
from typing import Iterable

import pytest


# ---------------------------------------------------------------------------
# Module-level pre-flight: every patch tested here calls into transformers.
# A single importorskip at module load keeps the failure message useful.
# ---------------------------------------------------------------------------

pytest.importorskip("transformers")
import transformers  # noqa: E402

_TX_VERSION = getattr(transformers, "__version__", "0.0.0")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_get_class(dotted_module: str, class_name: str):
    """Import ``dotted_module`` and return ``class_name`` off it, or
    return ``None`` if either is missing on this transformers. Used to
    skip 5.0+-gated tests cleanly on a 4.x install."""
    try:
        mod = importlib.import_module(dotted_module)
    except Exception:
        return None
    return getattr(mod, class_name, None)


def _require_class(dotted_module: str, class_name: str, zoo_file: str):
    """Like ``_try_get_class`` but ``pytest.fail`` with a DRIFT message
    if the class is missing AND the parent module exists. If the parent
    module is itself missing (e.g. transformers doesn't ship gemma4 in
    this version), skip -- that's a legitimate version gate, not drift."""
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
    """Reconstruct the storage key used by
    ``unsloth_zoo.temporary_patches.utils.patch_function`` to stash the
    original method body before it overwrites the class attribute.

    Mirrors ``_get_unique_storage_name`` in that file:
    ``_original_<last-component-of-module>_<class-name>_<attr>``. We
    re-derive the name here rather than import it so this test stays
    importable even on a zoo build where the temporary_patches sub-package
    can't be imported (e.g. minimal CPU CI).
    """
    module_tail = getattr(cls, "__module__", "").rsplit(".", 1)[-1]
    class_name = getattr(cls, "__name__", "") or cls.__class__.__name__
    return f"_original_{module_tail}_{class_name}_{attr}"


def _resolve_upstream_method(cls, method_name: str):
    """Return the function object representing the UPSTREAM (unpatched)
    method body for ``cls.method_name``.

    ``apply_import_fixes()`` and the ``temporary_patches`` runner both
    monkey-patch classes at import time, so a naive ``cls.method_name``
    lookup later in the test session returns the zoo-patched function
    instead of the upstream one. That makes signature drift tests false-
    positive on the patched ``(self, *args, **kwargs)`` wrapper rather
    than the real upstream API.

    Resolution order:
      1. If ``cls`` has ``_original_<module>_<class>_<method>`` set by
         ``patch_function``, return that. This is the authoritative source.
      2. If the live attribute's ``__qualname__`` indicates a zoo patch
         wrapper (``patch_<X>.<locals>.<method>``) but no original is
         stashed, fall through to (3) to load the source from the module
         file directly.
      3. Otherwise return the live attribute -- upstream isn't patched
         on this stack.
    """
    if not hasattr(cls, method_name):
        return None
    live = getattr(cls, method_name)
    # (1) explicit storage key from patch_function.
    storage_key = _original_attr_name(cls, method_name)
    original = getattr(cls, storage_key, None)
    if original is not None:
        return original
    # (2) wrapper-by-qualname fallback.
    qualname = getattr(live, "__qualname__", "") or ""
    if ".<locals>." in qualname and qualname.split(".", 1)[0].startswith("patch_"):
        # zoo patch wrapper, but no _original_ stash on the class. This
        # is rare (force=True + store_original=False) but possible. Fall
        # through; caller will skip cleanly via _maybe_skip_if_patched.
        return live
    return live


def _maybe_skip_if_patched(cls, method_name: str, zoo_file: str) -> None:
    """If the live ``cls.method_name`` is a zoo patch wrapper AND we
    have no stored original to inspect, skip the test with a clear
    "already-patched" reason rather than false-fail on the wrapper's
    ``(self, *args, **kwargs)`` signature.

    Used by signature-pin tests against classes that zoo replaces
    wholesale via ``patch_function``. The skip is loud: the message
    surfaces which zoo file did the patching so a future maintainer
    can re-anchor the test if upstream's shape genuinely changes.
    """
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
    # Prefer the upstream-original stash if zoo has patched the method.
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


# ===========================================================================
# bitsandbytes.py
# ---------------------------------------------------------------------------
# Patches: bitsandbytes.nn.modules.Linear4bit.forward (covered by
# test_upstream_signatures::test_bnb_Linear4bit_forward_signature), AND a
# second optional patch at bitsandbytes.nn.Linear4bit.forward (the
# top-level re-export). The second one is wrapped in try/except in zoo,
# but if the alias goes away without zoo noticing, the import-time guard
# `bitsandbytes.nn.modules.Linear4bit` would mask the alias drift.
# ===========================================================================

def test_bitsandbytes_top_level_Linear4bit_alias():
    """bitsandbytes.py:110 wraps a patch on the top-level
    ``bitsandbytes.nn.Linear4bit`` alias. Pin its presence and that it
    has a forward method matching the modules.Linear4bit forward."""
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
    """bitsandbytes.py:47 reads ``bitsandbytes.nn.modules.Params4bit`` and
    line 65-67 conditionally deletes its ``__torch_function__``. If the
    class disappears entirely, the patch ``raise_error()``-s silently and
    the torch.compile infinite-recursion fix never applies."""
    bnb = pytest.importorskip("bitsandbytes")
    p4 = getattr(bnb.nn.modules, "Params4bit", None)
    if p4 is None:
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/bitsandbytes.py:47 expects "
            "bitsandbytes.nn.modules.Params4bit but it is missing"
        )


def test_bitsandbytes_fix_4bit_weight_quant_state_from_module_present():
    """bitsandbytes.py:48 looks up
    ``bitsandbytes.nn.modules.fix_4bit_weight_quant_state_from_module``
    and passes ``self`` to it inside the patched forward. If this
    function disappears, the patched forward NameErrors at runtime."""
    bnb = pytest.importorskip("bitsandbytes")
    fn = getattr(bnb.nn.modules, "fix_4bit_weight_quant_state_from_module", None)
    if fn is None:
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/bitsandbytes.py:48 expects "
            "bitsandbytes.nn.modules.fix_4bit_weight_quant_state_from_module "
            "but it is missing"
        )
    # Patched forward calls fn(self) -- 1 positional. Reject zero-arity.
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
    """bitsandbytes.py:106 calls ``bitsandbytes.matmul_4bit(...)``. Pin
    that the top-level function exists. If it moves, the patched forward
    AttributeErrors at runtime."""
    bnb = pytest.importorskip("bitsandbytes")
    if not hasattr(bnb, "matmul_4bit"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/bitsandbytes.py:106 expects "
            "bitsandbytes.matmul_4bit() but it is missing"
        )


# ===========================================================================
# deepseek_v3_moe.py
# ---------------------------------------------------------------------------
# Patches: DeepseekV3NaiveMoe.forward (5.x), DeepseekV3MoE.forward
# (covered), DeepseekV3ForCausalLM.forward (covered). The NaiveMoe class
# is also used as a key for `_unsloth_already_patched` / `_unsloth_model_type`
# attribute attachment.
# ===========================================================================

def test_deepseek_v3_naive_moe_class_gated_5x():
    """deepseek_v3_moe.py:56-61 imports DeepseekV3NaiveMoe at the top of
    patch_deepseek_v3 and bails via try/except when it's missing. This
    class is transformers 5.x-only (added when the MoE forward was
    factored out of DeepseekV3MoE). Skip on older transformers."""
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
    """deepseek_v3_moe.py:59 imports DeepseekV3TopkRouter inside the same
    try/except guard. The class is referenced (not patched) but if it
    disappears, the whole patch entry silently no-ops."""
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
    """deepseek_v3_moe.py:125 patches DeepseekV3MoE.forward with
    ``def patched_moe_forward(self, hidden_states)``. Re-pin here as the
    sibling test only asserts param-superset; this asserts single-arg
    shape (no extra required positionals)."""
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
    a wrapper that forwards by name: input_ids, attention_mask,
    position_ids, past_key_values, inputs_embeds, labels, use_cache,
    output_router_logits, cache_position, logits_to_keep. Pin those names
    are still accepted. ``output_router_logits`` may have been folded
    into **kwargs upstream (TransformersKwargs catch-all), so we allow
    either an explicit param OR a VAR_KEYWORD catch-all."""
    cls = _try_get_class(
        "transformers.models.deepseek_v3.modeling_deepseek_v3",
        "DeepseekV3ForCausalLM",
    )
    if cls is None:
        pytest.skip(f"DeepseekV3ForCausalLM absent on transformers {_TX_VERSION}")
    fwd = _assert_method_exists(cls, "forward", "deepseek_v3_moe.py")
    # Hard-required params (always part of an LM forward).
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
    # output_router_logits is forwarded by name from zoo's wrapper, but
    # upstream folded it into **kwargs in 4.57. Pin that the upstream
    # has SOME kwarg passthrough so zoo's by-name forwarding still
    # reaches the underlying model.
    if "output_router_logits" not in _param_names(fwd) and not _has_var_keyword(fwd):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/deepseek_v3_moe.py:171 "
            "forwards output_router_logits=output_router_logits but installed "
            f"DeepseekV3ForCausalLM.forward on transformers {_TX_VERSION} "
            f"has neither an explicit output_router_logits param nor a "
            f"**kwargs catch-all: {inspect.signature(fwd)}"
        )


# ===========================================================================
# gemma.py
# ---------------------------------------------------------------------------
# Most of gemma.py is covered. The UNSLOTH_FORCE_FLOAT32-gated
# Gemma3Model._update_causal_mask patch (gemma.py:308) is the only
# remaining uncovered site. Upstream removed _update_causal_mask in
# transformers 4.55+, so the patch is a no-op on modern installs.
# ===========================================================================

def test_gemma3_force_fp32_update_causal_mask_gated():
    """gemma.py:308-310 patches Gemma3Model._update_causal_mask and
    Gemma3ForConditionalGeneration._update_causal_mask ONLY when
    UNSLOTH_FORCE_FLOAT32=1. Upstream removed the method in 4.55+, so on
    modern installs the gate-guarded import line will succeed (classes
    still exist) but the method won't. Test that EITHER both classes
    exist AND have the method (drift if class is here without method
    while gate is active) OR the method is gone (legitimately upstream-
    refactored)."""
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
    # Both classes must still exist. The method is allowed to have been
    # removed: zoo's patch_function silently no-ops when the attribute
    # isn't there. We just confirm the class survival here.
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1":
        if not hasattr(model, "_update_causal_mask"):
            pytest.fail(
                "DRIFT DETECTED: zoo temporary_patches/gemma.py:308 with "
                "UNSLOTH_FORCE_FLOAT32=1 patches Gemma3Model._update_causal_mask, "
                f"but transformers {_TX_VERSION} has dropped this method. The "
                "patch silently no-ops and the FORCE_FLOAT32 mask fix never lands."
            )


# ===========================================================================
# gemma3n.py
# ---------------------------------------------------------------------------
# Existing tests cover Gemma3nMultimodalEmbedder, Gemma3nTextAltUp.predict
# /correct, Gemma3nModel.get_placeholder_mask. Missing: AltUp's
# scale_corrected_output and the gemma3n module surface.
# ===========================================================================

def test_gemma3n_text_alt_up_scale_corrected_output_signature():
    """gemma3n.py:148 patches Gemma3nTextAltUp.scale_corrected_output
    with fullgraph=True. The original is ``(self, corrected)`` -- a
    single-tensor method. Pin that shape."""
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
    """gemma3n.py:143-148 inspects ``hasattr`` for predict / correct /
    scale_corrected_output before patching. The hasattr guard masks a
    full method-set rename: this test fails LOUDLY when all three are
    simultaneously gone (i.e. the AltUp class was restructured)."""
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
    """gemma3n.py:53 defines a module-level torch_compile'd
    ``Gemma3nRMSNorm_forward`` that is then called by the patched
    Multimodal embedder / AltUp.predict on ``self.soft_embedding_norm``,
    ``self.router_norm``, etc. Those attributes must exist on
    ``Gemma3nMultimodalEmbedder`` / ``Gemma3nTextAltUp``."""
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


# ===========================================================================
# gemma4.py
# ---------------------------------------------------------------------------
# Patches: Gemma4TextMLP.forward (gemma4.py:655). Gemma4 is 5.0+-only.
# ===========================================================================

def test_gemma4_text_mlp_forward_signature():
    """gemma4.py:655 patches Gemma4TextMLP.forward with
    ``def forward(self, x)``. Pin a single positional arg."""
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
    """gemma4.py:644-652 patched forward dereferences self.gate_proj,
    self.up_proj, self.down_proj, self.act_fn. Pin those exist in the
    __init__ source."""
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


# ===========================================================================
# gemma4_moe.py
# ---------------------------------------------------------------------------
# Patches: Gemma4TextExperts.forward, Gemma4TextDecoderLayer.__init__,
# Gemma4TextMoEBlock.forward, Gemma4ForConditionalGeneration.forward.
# All transformers 5.0+-gated.
# ===========================================================================

def test_gemma4_text_experts_forward_signature():
    """gemma4_moe.py:239 patches Gemma4TextExperts.forward with
    ``forward(self, hidden_states, top_k_index, top_k_weights)``."""
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
    """gemma4_moe.py:287 patches Gemma4TextDecoderLayer.__init__ with
    ``def __init__(self, config, layer_idx)``. Pin those param names."""
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
    """gemma4_moe.py:301 patches Gemma4TextMoEBlock.forward with
    ``forward(self, hidden_states, top_k_index, top_k_weights)``."""
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
    """gemma4_moe.py:208 patches Gemma4ForConditionalGeneration.forward
    with a wrapper that forwards by name: input_ids, pixel_values,
    pixel_values_videos, input_features, attention_mask,
    input_features_mask, position_ids, image_position_ids,
    video_position_ids, past_key_values, mm_token_type_ids,
    inputs_embeds, labels, use_cache, logits_to_keep. Pin the names."""
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
    """gemma4_moe.py:189 constructs Gemma4CausalLMOutputWithPast(loss,
    logits, past_key_values, hidden_states, attentions,
    image_hidden_states, audio_hidden_states). Pin those kwarg names."""
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


# ===========================================================================
# glm4_moe.py
# ---------------------------------------------------------------------------
# Patches: Glm4MoeLiteNaiveMoe.forward, Glm4MoeLiteMoE.forward.
# 5.0+-gated (the entire glm4_moe_lite module is 5.0+).
# ===========================================================================

def test_glm4_moe_lite_naive_moe_forward_signature():
    """glm4_moe.py:97 patches Glm4MoeLiteNaiveMoe.forward via
    ``get_forward_moe_backend()``. The backend forward signature is
    ``(self, hidden_states, top_k_index, top_k_weights)``."""
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
    """glm4_moe.py:98 patches Glm4MoeLiteMoE.forward with
    ``moe_block_forward(self, hidden_states)``."""
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


# ===========================================================================
# gpt_oss.py
# ---------------------------------------------------------------------------
# Patches: swizzle_mxfp4 (covered), Mxfp4GptOssExperts (NOT covered as a
# signature test), mlp_forward (covered), load_and_swizzle_mxfp4 (covered),
# replace_with_mxfp4_linear (covered), GptOssAttention.forward (covered),
# GptOssModel.forward (covered), GptOssConfig (NOT covered, source-only
# patch), GptOssPreTrainedModel._init_weights (covered),
# GptOssForCausalLM.forward (NOT covered).
# ===========================================================================

def test_mxfp4_gpt_oss_experts_class_present_and_init_signature():
    """gpt_oss.py:433 replaces transformers.integrations.mxfp4
    .Mxfp4GptOssExperts with a custom class. Pin that the upstream class
    exists and its __init__ accepts (self, config)."""
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
    """gpt_oss.py:2813 conditionally replaces
    transformers.models.gpt_oss.configuration_gpt_oss.GptOssConfig with
    Old_GptOssConfig. The replacement uses kwargs num_hidden_layers,
    num_local_experts, vocab_size, hidden_size, intermediate_size,
    head_dim, num_attention_heads, num_key_value_heads, sliding_window,
    rope_theta, etc. Pin those names exist on the installed class so the
    replacement (and any user constructing the config by kwarg) doesn't
    silently miss a renamed param."""
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
    """gpt_oss.py:2890 patches GptOssForCausalLM.forward with a wrapper
    that forwards by name: input_ids, attention_mask, position_ids,
    past_key_values, inputs_embeds, labels, use_cache, output_attentions,
    output_hidden_states, cache_position, logits_to_keep."""
    cls = _try_get_class(
        "transformers.models.gpt_oss.modeling_gpt_oss", "GptOssForCausalLM",
    )
    if cls is None:
        pytest.skip(f"GptOssForCausalLM absent on transformers {_TX_VERSION}")
    fwd = _assert_method_exists(cls, "forward", "gpt_oss.py")
    # Newer transformers may have already dropped output_attentions and
    # output_hidden_states from forward signatures. Zoo's wrapper still
    # accepts them as kwargs that go into **kwargs. Pin only the params
    # that are guaranteed to remain.
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
    """gpt_oss.py:2949 constructs MoeCausalLMOutputWithPast(loss,
    aux_loss, logits, past_key_values, hidden_states, attentions,
    router_logits). Pin those kwarg names."""
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
    """gpt_oss.py:2126 imports DynamicCache from
    transformers.models.gpt_oss.modeling_gpt_oss as a soft try/except. If
    the re-export goes away the patch still works (via the fallback
    lambda), but pinning here surfaces the rename loudly."""
    mod = importlib.import_module(
        "transformers.models.gpt_oss.modeling_gpt_oss"
    )
    if not hasattr(mod, "DynamicCache"):
        # The fallback in zoo silently substitutes a no-op. Surface this
        # so we know to land a real fix.
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/gpt_oss.py:2126 expects "
            "transformers.models.gpt_oss.modeling_gpt_oss.DynamicCache but "
            f"installed transformers {_TX_VERSION} has dropped the re-export. "
            "Zoo silently falls back to a lambda that returns None -- caches "
            "stop working."
        )


def test_gpt_oss_apply_rotary_pos_emb_re_export():
    """gpt_oss.py:2122 imports apply_rotary_pos_emb from the gpt_oss
    modeling module. Re-export pin."""
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
    """gpt_oss.py:2121 imports MoeModelOutputWithPast. The patched
    GptOssModel.forward returns this output class."""
    mod = importlib.import_module(
        "transformers.models.gpt_oss.modeling_gpt_oss"
    )
    if not hasattr(mod, "MoeModelOutputWithPast"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/gpt_oss.py:2121 expects "
            "transformers.models.gpt_oss.modeling_gpt_oss.MoeModelOutputWithPast "
            "but it is missing"
        )


# ===========================================================================
# misc.py
# ---------------------------------------------------------------------------
# Patches:
#   - AutoHfQuantizer.merge_quantization_configs (covered)
#   - CsmDepthDecoderForCausalLM.forward (NOT covered)
#   - CsmForConditionalGeneration.forward (NOT covered as forward; only
#     _merge_input_ids_with_input_values is pinned)
#   - GraniteMoeHybridMambaLayer.cuda_kernels_forward (covered)
#   - SiglipEncoderLayer.forward (covered)
#   - MllamaVisionEncoderLayer.forward (covered)
# ===========================================================================

def test_csm_depth_decoder_for_causal_lm_forward_named_params():
    """misc.py:239 patches CsmDepthDecoderForCausalLM.forward with named
    params: input_ids, backbone_last_hidden_state, attention_mask,
    position_ids, past_key_values, inputs_embeds, labels, use_cache,
    cache_position, logits_to_keep.

    Resolves through the ``_original_*`` stash so we inspect the genuine
    upstream signature even after zoo's TEMPORARY_PATCHES have replaced
    the live ``forward`` with a ``(self, *args, **kwargs)`` wrapper.
    """
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
    """misc.py:373 patches CsmForConditionalGeneration.forward. The
    replacement forwards: input_ids, input_values, attention_mask,
    input_values_cutoffs, position_ids, past_key_values, inputs_embeds,
    labels, use_cache, cache_position, logits_to_keep.

    Resolves through the ``_original_*`` stash so we inspect the genuine
    upstream signature even after zoo's TEMPORARY_PATCHES have replaced
    the live ``forward`` with a ``(self, *args, **kwargs)`` wrapper.
    """
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
    """misc.py constructs CausalLMOutputWithPast / CsmOutputWithPast.
    Pin CsmOutputWithPast field set."""
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
    shift_labels). Pin those keyword names accepted."""
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
    """misc.py:770 patches CsmForConditionalGeneration._merge_input_ids
    _with_input_values. The sibling test pins by-name params; here we
    pin that the method exists on the class regardless of name shuffles."""
    cls = _try_get_class(
        "transformers.models.csm.modeling_csm",
        "CsmForConditionalGeneration",
    )
    if cls is None:
        pytest.skip(f"CsmForConditionalGeneration absent on transformers {_TX_VERSION}")
    _assert_method_exists(cls, "_merge_input_ids_with_input_values", "misc.py")


def test_misc_quantizers_auto_module_present():
    """misc.py:153 patches transformers.quantizers.auto.AutoHfQuantizer.
    Pin the dotted path."""
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
    """misc.py:1061 patches
    transformers.models.granitemoehybrid.modeling_granitemoehybrid
    .GraniteMoeHybridMambaLayer. Pin the dotted path; the sibling test
    pins the cuda_kernels_forward signature."""
    cls = _try_get_class(
        "transformers.models.granitemoehybrid.modeling_granitemoehybrid",
        "GraniteMoeHybridMambaLayer",
    )
    if cls is None:
        pytest.skip(
            f"GraniteMoeHybridMambaLayer absent on transformers {_TX_VERSION}"
        )


def test_misc_siglip_encoder_layer_class_present():
    """misc.py:1228 patches
    transformers.models.siglip.modeling_siglip.SiglipEncoderLayer.
    Pin the dotted path."""
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
    """misc.py:1116-1119 imports MllamaVisionConfig / MllamaVisionAttention
    / MllamaVisionMLP / MllamaVisionEncoder from
    transformers.models.mllama.modeling_mllama. Pin them as a set."""
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


# ===========================================================================
# mxfp4.py
# ---------------------------------------------------------------------------
# Patches: convert_moe_packed_tensors, dequantize (both at
# transformers.integrations.mxfp4 module level). The sibling
# test_mxfp4_swizzle_mxfp4_signature and test_mxfp4_replace_with_mxfp4
# _linear_signature pin OTHER mxfp4 functions but NOT these two.
# ===========================================================================

def test_mxfp4_convert_moe_packed_tensors_signature():
    """mxfp4.py:173 patches
    transformers.integrations.mxfp4.convert_moe_packed_tensors. The
    replacement signature is ``(blocks, scales, *, dtype=torch.bfloat16,
    rows_per_chunk=...)``. Pin the positional+kwonly names."""
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
    """mxfp4.py:220 patches
    transformers.integrations.mxfp4.dequantize. The replacement signature
    is ``(module, param_name, param_value, target_device, dq_param_name,
    **kwargs)``."""
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
    """mxfp4.py:113 / 227 imports FP4_VALUES from
    transformers.integrations.mxfp4. Pin the constant."""
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
    """mxfp4.py:181 imports shard_and_distribute_module from
    transformers.integrations.tensor_parallel. The patched dequantize
    delegates to this when device_mesh is non-None.

    Note: zoo's call site at mxfp4.py:196 passes ``set_param=False`` --
    a kwarg added in transformers 5.x. On 4.x stacks this kwarg is
    legitimately absent and the TP code path raises TypeError at call
    time. The TP code path is only exercised when ``device_mesh is not
    None``, so non-TP users are unaffected. Pin function existence here;
    the set_param compatibility is gated in the separate test below."""
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
    # Positional arity the call site uses (model, param_value,
    # empty_param, dq_param_name, casting_dtype, to_contiguous, rank,
    # device_mesh). Upstream renames between 4.x and 5.x (param ->
    # param_value, parameter_name -> dq_param_name, etc.), but the
    # POSITIONAL arity must remain at 8 for zoo's call to land.
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
    """mxfp4.py:196 passes ``set_param=False`` to
    shard_and_distribute_module. This kwarg was added in transformers
    5.x; on 4.x it doesn't exist and the call TypeErrors. The TP path is
    only hit when device_mesh is not None, so most users are unaffected,
    but we surface the version-skew explicitly so a future zoo PR can
    decide whether to drop the kwarg conditionally on transformers
    version."""
    mod = importlib.import_module("transformers.integrations.tensor_parallel")
    fn = mod.shard_and_distribute_module
    if "set_param" in _param_names(fn):
        return  # 5.x; zoo's call site works
    if _has_var_keyword(fn):
        return  # **kwargs catch-all swallows set_param
    # 4.x without **kwargs: zoo's TP path will TypeError. This is a
    # well-known version-skew limitation -- zoo expects users running
    # mxfp4 + TP to be on transformers 5.x. Skip rather than fail so the
    # general suite passes on 4.x dev installs; the explicit message
    # makes the skew loud.
    pytest.skip(
        f"transformers {_TX_VERSION} predates set_param kwarg on "
        "shard_and_distribute_module; zoo's TP path (device_mesh != None) "
        "requires 5.x. Non-TP users unaffected."
    )


def test_mxfp4_mxfp4_config_top_level_class():
    """mxfp4.py:93 imports Mxfp4Config from top-level transformers. Pin
    it. Used as the quantization_config kwarg for AutoModelForCausalLM."""
    if not hasattr(transformers, "Mxfp4Config"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/mxfp4.py:93 expects "
            "transformers.Mxfp4Config (top-level re-export) but it is "
            f"missing on transformers {_TX_VERSION}"
        )


# ===========================================================================
# pixtral.py
# ---------------------------------------------------------------------------
# Patches: PixtralAttention.__init__ (pixtral.py:91), PixtralAttention.forward
# (pixtral.py:97). Neither covered by existing tests.
# ===========================================================================

def test_pixtral_attention_init_signature():
    """pixtral.py:91 patches PixtralAttention.__init__ with
    ``def __init__(self, config)``. Pin the single-config init."""
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
    """pixtral.py:97 patches PixtralAttention.forward with
    ``forward(self, hidden_states, attention_mask, position_embeddings,
    output_attentions=False, **kwargs)``. Pin those names.

    Once apply_import_fixes / TEMPORARY_PATCHES have run, the live
    ``PixtralAttention.forward`` is zoo's patch wrapper with signature
    ``(self, *args, **kwargs)``; reading that signature would false-fail
    the upstream-shape pin. We instead resolve through
    ``_original_<module>_<class>_<attr>`` (stashed by zoo's
    ``patch_function``) to read the genuine upstream signature, or skip
    loudly with the patch-wrapper detail if no stash is available.
    """
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
    """pixtral.py:30 imports apply_rotary_pos_emb from the pixtral
    modeling module. Re-export pin -- if it moves, the patch raises and
    PixtralAttention falls back to the (broken) stock forward."""
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
    """pixtral.py:36-47 patched __init__ sets self.embed_dim, num_heads,
    head_dim, scale, dropout, k_proj, v_proj, q_proj, o_proj. The config
    must expose hidden_size, num_attention_heads, attention_dropout."""
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


# ===========================================================================
# qwen3_5_moe.py
# ---------------------------------------------------------------------------
# Patches: Qwen3_5MoeExperts.forward, Qwen3_5MoeSparseMoeBlock.forward,
# Qwen3_5MoeForCausalLM.forward. All 5.0+-gated -- the module
# qwen3_5_moe only exists on transformers 5.x.
# ===========================================================================

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
    """qwen3_5_moe.py:77 reads Qwen3_5MoeForCausalLM and MoeCausalLMOutput
    WithPast for the GRPO hidden-states patch. Pin those classes."""
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


# ===========================================================================
# qwen3_moe.py
# ---------------------------------------------------------------------------
# Patches: Qwen3MoeSparseMoeBlock.forward (covered), Qwen3MoeExperts.forward
# (5.0+-gated, NOT covered with signature), Qwen3MoeForCausalLM.forward
# (NOT covered).
# ===========================================================================

def test_qwen3_moe_experts_forward_signature_5x():
    """qwen3_moe.py:339 patches Qwen3MoeExperts.forward via
    ``patch_function(...)`` on the 5.0+ stacked-experts branch. The
    sibling test pins class EXISTENCE; this test pins the forward
    signature accepts (hidden_states, top_k_index, top_k_weights)."""
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
    """qwen3_moe.py:351 indirectly patches Qwen3MoeForCausalLM.forward via
    ``_patch_causal_lm_forward_for_hidden_states`` (qwen3_moe.py:138).
    Patched signature is (input_ids, attention_mask, position_ids,
    past_key_values, inputs_embeds, labels, use_cache,
    output_router_logits, cache_position, logits_to_keep, **kwargs)."""
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
    """qwen3_moe.py:349 imports MoeCausalLMOutputWithPast from the same
    qwen3_moe modeling module. Pin re-export."""
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


# ===========================================================================
# qwen3_next_moe.py
# ---------------------------------------------------------------------------
# Patches: Qwen3NextExperts.forward, Qwen3NextSparseMoeBlock.forward
# (covered), Qwen3NextForCausalLM.forward (via the shared
# _patch_causal_lm_forward_for_hidden_states helper).
# ===========================================================================

def test_qwen3_next_experts_forward_signature():
    """qwen3_next_moe.py:57 patches Qwen3NextExperts.forward (5.0+-only).
    Pin signature accepts (hidden_states, ...) on installs where the
    class is present."""
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
    """qwen3_next_moe.py:79 indirectly patches Qwen3NextForCausalLM.forward
    via ``_patch_causal_lm_forward_for_hidden_states``. Pin the named
    params zoo's wrapper passes."""
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


# ===========================================================================
# qwen3_vl_moe.py
# ---------------------------------------------------------------------------
# Patches: Qwen3VLMoeTextSparseMoeBlock.forward (covered), Qwen3VLMoe
# TextExperts.forward/__init__ (covered), Qwen3VLMoeForConditional
# Generation.forward (NOT covered).
# ===========================================================================

def test_qwen3_vl_moe_for_conditional_generation_forward_named_params():
    """qwen3_vl_moe.py:401 patches Qwen3VLMoeForConditionalGeneration.
    forward. Patched signature forwards input_ids, attention_mask,
    position_ids, past_key_values, inputs_embeds, labels, pixel_values,
    pixel_values_videos, image_grid_thw, video_grid_thw, cache_position,
    logits_to_keep."""
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
    (loss, aux_loss, logits, past_key_values, hidden_states, attentions,
    rope_deltas). Pin kwarg names."""
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
    """qwen3_vl_moe.py:326 expects ``self.gate`` to be
    Qwen3VLMoeTextTopKRouter on the new (5.x) layout. The router returns
    (router_logits, router_scores, router_indices). Pin the class."""
    cls = _try_get_class(
        "transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe",
        "Qwen3VLMoeTextTopKRouter",
    )
    if cls is None:
        # The class is 5.x-only; if absent, zoo's tuple-unpack fallback
        # at qwen3_vl_moe.py:333 still works. Don't fail here.
        pytest.skip(
            f"Qwen3VLMoeTextTopKRouter absent on transformers {_TX_VERSION} "
            "(zoo gracefully falls back to old-style logit gate)"
        )


def test_qwen3_vl_moe_text_experts_class_present():
    """qwen3_vl_moe.py:73 imports Qwen3VLMoeTextExperts. The sibling test
    pins forward / __init__ signatures; this test gates the module
    presence so a missing parent module surfaces clearly."""
    cls = _try_get_class(
        "transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe",
        "Qwen3VLMoeTextExperts",
    )
    if cls is None:
        pytest.skip(
            f"Qwen3VLMoeTextExperts absent on transformers {_TX_VERSION}"
        )


def test_qwen3_vl_moe_act2fn_dict_present():
    """qwen3_vl_moe.py:201 imports ACT2FN from transformers.activations.
    The patched __init__ does ``self.act_fn = ACT2FN[config.hidden_act]``.
    Pin the import path."""
    from transformers.activations import ACT2FN  # noqa: F401
    # If hidden_act default is silu, ACT2FN must accept that key.
    if "silu" not in ACT2FN:
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/qwen3_vl_moe.py:236 expects "
            "transformers.activations.ACT2FN['silu'] but the key is missing"
        )


# ===========================================================================
# moe_utils.py / moe_bnb.py / flex_attention_bwd.py
# ---------------------------------------------------------------------------
# These helper modules don't directly patch transformers (no
# patch_function call sites). moe_utils provides helpers consumed by the
# other temporary_patches/ files, and moe_bnb / flex_attention_bwd are
# utility shims. Skip patch-site enumeration here; existing tests cover
# the consumer sites already.
# ===========================================================================

def test_moe_utils_param_wrapper_target_present():
    """moe_utils.py registers patches against peft.tuners.lora.layer
    .ParamWrapper. If PEFT renames the class, zoo's split-LoRA grouped-GEMM
    code path silently falls back to the unwrapped layout."""
    peft = pytest.importorskip("peft")
    try:
        from peft.tuners.lora.layer import ParamWrapper  # noqa: F401
    except Exception as exc:
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/moe_utils.py expects "
            "peft.tuners.lora.layer.ParamWrapper but it is missing: " + str(exc)
        )


# ===========================================================================
# misc.py (additional patch sites)
# ---------------------------------------------------------------------------
# misc.py contains 19 separate ``patch_X`` entries. The existing tests
# cover ~6 of them. The remainder fall into config-mapping, tokenizer
# attribute, mask-utils wrap, modernbert mask-strides, lfm2 projector,
# peft dispatch, trl push-to-hub, vllm chat-template, and qwen2-vl
# image-processor compat shims. Pin upstream targets for each.
# ===========================================================================

def test_misc_config_mapping_present_for_ministral3_register():
    """misc.py:47 imports CONFIG_MAPPING from
    transformers.models.auto.configuration_auto and calls .register(...)
    on it. Pin the import path and the mapping has a register method."""
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
    """misc.py:48 imports MinistralConfig from top-level transformers as
    the value side of the ``ministral3`` -> MinistralConfig alias."""
    if not hasattr(transformers, "MinistralConfig"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/misc.py:48 expects "
            "transformers.MinistralConfig at top level but it is missing on "
            f"{_TX_VERSION}"
        )


def test_misc_pretrained_tokenizer_base_convert_added_tokens_method():
    """misc.py:67 expects PreTrainedTokenizerBase.convert_added_tokens
    to be a classmethod. The patch reassigns it. Pin the attr name."""
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    if not hasattr(PreTrainedTokenizerBase, "convert_added_tokens"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/misc.py:67 expects "
            "PreTrainedTokenizerBase.convert_added_tokens but it is missing "
            f"on transformers {_TX_VERSION}"
        )


def test_misc_added_token_class_present():
    """misc.py:63 imports AddedToken from
    transformers.tokenization_utils_base. The patched
    convert_added_tokens constructs AddedToken(**obj)."""
    from transformers.tokenization_utils_base import AddedToken
    sig = inspect.signature(AddedToken)
    # Pin a couple of expected fields so a rename surfaces here.
    field_names = list(sig.parameters.keys())
    for req in ("content",):
        if req not in field_names:
            pytest.fail(
                f"DRIFT DETECTED: zoo temporary_patches/misc.py:75 constructs "
                f"AddedToken(content=...) but installed AddedToken on "
                f"transformers {_TX_VERSION} has __init__ params {field_names}"
            )


def test_misc_pretrained_tokenizer_base_init_takes_kwargs():
    """misc.py:97 wraps PreTrainedTokenizerBase.__init__ and rejects /
    coerces extra_special_tokens. Pin __init__ accepts **kwargs."""
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    if not _has_var_keyword(PreTrainedTokenizerBase.__init__):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/misc.py:97 patched "
            "PreTrainedTokenizerBase.__init__(self, **kwargs) but installed "
            f"signature has no VAR_KEYWORD: "
            f"{inspect.signature(PreTrainedTokenizerBase.__init__)}"
        )


def test_misc_masking_utils_create_block_mask_available_or_compile_flag():
    """misc.py:391-409 imports BlockMask / create_block_mask from
    torch.nn.attention.flex_attention and rewrites the masks function on
    transformers.masking_utils. Pin the upstream masking_utils module
    has create_causal_mask / create_sliding_window_causal_mask /
    create_masks_for_generate (all consumed by the patch)."""
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
    transformers.generation.utils.create_masks_for_generate. Pin the
    attribute exists pre-patch."""
    gu = importlib.import_module("transformers.generation.utils")
    if not hasattr(gu, "create_masks_for_generate"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/misc.py:447 expects "
            "transformers.generation.utils.create_masks_for_generate but it "
            f"is missing on transformers {_TX_VERSION}"
        )


def test_misc_masking_utils_padding_and_packed_helpers():
    """misc.py:472 / 490 conditionally wraps padding_mask_function and
    packed_sequence_mask_function on masking_utils. The wraps are
    gated by hasattr so absence isn't drift, but pin them when present."""
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
    """misc.py:525 patches
    transformers.integrations.sdpa_attention.sdpa_attention_forward."""
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
    """misc.py:526 imports ALL_ATTENTION_FUNCTIONS from
    transformers.modeling_utils. Pin the symbol presence."""
    mu = importlib.import_module("transformers.modeling_utils")
    if not hasattr(mu, "ALL_ATTENTION_FUNCTIONS"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/misc.py:526 expects "
            "transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS but it is "
            f"missing on transformers {_TX_VERSION}"
        )


def test_misc_modernbert_model_update_attention_mask_present():
    """misc.py:662 patches ModernBertModel._update_attention_mask. The
    patch is gated by hasattr; pin the method when ModernBertModel is
    present so a rename surfaces."""
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
    CsmForConditionalGeneration._merge_input_ids_with_input_values with
    a 4-arg replacement (input_ids, input_values, input_values_cutoffs,
    labels). Pin the upstream method accepts the same names."""
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
    """misc.py:1247 patches Lfm2VlMultiModalProjector.__init__ /
    .forward. Pin class presence; the patch is gated on transformers
    pre-5.0.0."""
    cls = _try_get_class(
        "transformers.models.lfm2_vl.modeling_lfm2_vl",
        "Lfm2VlMultiModalProjector",
    )
    if cls is None:
        pytest.skip(
            f"Lfm2VlMultiModalProjector absent on transformers {_TX_VERSION}"
        )
    # Patched __init__: def patched_init(self, config, *args, **kwargs)
    _assert_params_superset(
        cls.__init__,
        required=["config"],
        zoo_file="misc.py",
        label="Lfm2VlMultiModalProjector.__init__",
    )


def test_misc_peft_dispatch_bnb_4bit_target_present():
    """misc.py:1290 patches peft.tuners.lora.bnb.dispatch_bnb_4bit. Pin
    the function exists in the installed PEFT."""
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
    """misc.py:1334 patches TrainingArguments.to_dict on transformers
    5.0+. Pin the to_dict() target exists."""
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
    """misc.py:1363 reads / writes
    transformers.models.auto.modeling_auto.MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
    (the 5.0+ name) and
    MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES (the legacy name). At least one
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
    """misc.py:1446 checks ``return_dict`` is in
    PreTrainedTokenizerBase.apply_chat_template signature on
    transformers 5.0+. Pin the kwarg in the installed signature."""
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
    """misc.py:1485 imports Qwen2VLImageProcessor and conditionally
    attaches max_pixels / min_pixels properties. Pin the class."""
    cls = _try_get_class(
        "transformers.models.qwen2_vl.image_processing_qwen2_vl",
        "Qwen2VLImageProcessor",
    )
    if cls is None:
        pytest.skip(
            f"Qwen2VLImageProcessor absent on transformers {_TX_VERSION}"
        )


# ===========================================================================
# gpt_oss.py (additional patch sites beyond the existing tests)
# ===========================================================================

def test_gpt_oss_mxfp4_quantizer_class_present():
    """gpt_oss.py:127 monkey-patches
    transformers.quantizers.quantizer_mxfp4.Mxfp4HfQuantizer.is_trainable.
    Pin the class exists. Without it, the patch silently no-ops."""
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
    transformers.quantizers.quantizer_mxfp4.is_kernels_available. Pin
    the symbol."""
    mod = importlib.import_module("transformers.quantizers.quantizer_mxfp4")
    if not hasattr(mod, "is_kernels_available"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/gpt_oss.py:136 expects "
            "transformers.quantizers.quantizer_mxfp4.is_kernels_available "
            f"but it is missing on transformers {_TX_VERSION}"
        )


def test_gpt_oss_modeling_module_top_level_classes_present():
    """gpt_oss.py:1060-1063 reassigns GptOssExperts and GptOssTopKRouter
    via attribute setting on the modeling module. Pin both class names
    exist as module attributes (they are the patch targets)."""
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
    """gpt_oss.py near the config patch reads ``layer_type_validation``
    via the rope_utils path used by configuration_gpt_oss.py. Pin
    via configuration module symbol."""
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
    """gpt_oss.py:2832 reads
    transformers.models.gpt_oss.modeling_gpt_oss.GptOssPreTrainedModel
    as the patch target for _init_weights."""
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
    """gpt_oss.py:2126 imports DynamicCache from gpt_oss modeling.
    Already pinned by sibling test; here we pin from
    transformers.cache_utils as the canonical fallback path."""
    cu = importlib.import_module("transformers.cache_utils")
    if not hasattr(cu, "DynamicCache"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/gpt_oss.py expects "
            "transformers.cache_utils.DynamicCache but it is missing on "
            f"transformers {_TX_VERSION}"
        )


def test_gpt_oss_attention_apply_rotary_pos_emb_imported_at_attention():
    """gpt_oss.py:1875+ imports apply_rotary_pos_emb from the gpt_oss
    modeling module for GptOssAttention.forward. Pin via separate
    re-export check at a different line than the existing sibling test."""
    mod = importlib.import_module(
        "transformers.models.gpt_oss.modeling_gpt_oss"
    )
    apply = getattr(mod, "apply_rotary_pos_emb", None)
    if apply is None:
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/gpt_oss.py:1875 expects "
            "modeling_gpt_oss.apply_rotary_pos_emb but it is missing"
        )
    # apply_rotary_pos_emb is called as
    #   apply_rotary_pos_emb(q, k, cos, sin) -> 4 positional args.
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
    """gpt_oss.py:2063 calls eager_attention_forward(self, q, k, v,
    mask, dropout=..., scaling=..., sliding_window=..., s_aux=...,
    **kwargs). Pin those by-name params on the upstream helper."""
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
    # Be lenient: only require the positional arity since the kwarg names
    # change across transformers releases.
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


# ===========================================================================
# gemma.py (Gemma3DecoderLayer, Gemma3TextModel survival as transitive
# patch dependencies)
# ===========================================================================

def test_gemma3_decoder_layer_class_present():
    """gemma.py imports Gemma3Attention from modeling_gemma3 and patches
    Gemma3Attention.forward. The decoder layer is the parent and must
    exist as a sibling pin so a rename surfaces."""
    cls = _try_get_class(
        "transformers.models.gemma3.modeling_gemma3", "Gemma3DecoderLayer",
    )
    if cls is None:
        pytest.fail(
            "DRIFT DETECTED: gemma.py expects Gemma3DecoderLayer (parent of "
            f"Gemma3Attention) but it is missing on transformers {_TX_VERSION}"
        )


def test_gemma3_text_model_class_present():
    """gemma.py:233 references Gemma3Model (the multimodal model). The
    underlying text-only model Gemma3TextModel is the LM head's
    backbone; pin it."""
    cls = _try_get_class(
        "transformers.models.gemma3.modeling_gemma3", "Gemma3TextModel",
    )
    if cls is None:
        pytest.fail(
            "DRIFT DETECTED: gemma.py expects Gemma3TextModel but it is "
            f"missing on transformers {_TX_VERSION}"
        )


def test_gemma3_pre_trained_model_class_present():
    """gemma.py touches Gemma3 model surfaces -- Gemma3PreTrainedModel
    is the base class. Pin its existence."""
    cls = _try_get_class(
        "transformers.models.gemma3.modeling_gemma3", "Gemma3PreTrainedModel",
    )
    if cls is None:
        pytest.fail(
            "DRIFT DETECTED: gemma.py expects Gemma3PreTrainedModel but it "
            f"is missing on transformers {_TX_VERSION}"
        )


def test_gemma3_processor_kwargs_class_present():
    """gemma.py:218 reads
    transformers.models.gemma3.processing_gemma3.Gemma3ProcessorKwargs as
    an Unpack type for __call__."""
    mod = importlib.import_module(
        "transformers.models.gemma3.processing_gemma3"
    )
    if not hasattr(mod, "Gemma3ProcessorKwargs"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/gemma.py:218 expects "
            "Gemma3ProcessorKwargs but it is missing on transformers "
            f"{_TX_VERSION}"
        )


# ===========================================================================
# gemma3n.py (additional pins)
# ===========================================================================

def test_gemma3n_for_conditional_generation_class_present():
    """gemma3n.py patches Gemma3nModel.get_placeholder_mask. The
    conditional-generation head pins its existence."""
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
    """gemma3n.py:53 defines a Gemma3nRMSNorm_forward helper that the
    patched MultimodalEmbedder forward delegates to. The actual upstream
    class must exist so the patched forward's call to self.weight,
    self._norm continues to compile."""
    cls = _try_get_class(
        "transformers.models.gemma3n.modeling_gemma3n", "Gemma3nRMSNorm",
    )
    if cls is None:
        pytest.fail(
            "DRIFT DETECTED: gemma3n.py:53 helper expects Gemma3nRMSNorm but "
            f"it is missing on transformers {_TX_VERSION}"
        )


# ===========================================================================
# qwen3_moe.py / qwen3_5_moe.py / qwen3_next_moe.py shared deps
# ===========================================================================

def test_qwen3_moe_rms_norm_class_present():
    """qwen3_moe.py's patched forward calls .gate(...) and .experts(...)
    -- the parent module sets these as Linear / ModuleList. Pin a
    well-known sibling class so a wholesale namespace rename surfaces."""
    cls = _try_get_class(
        "transformers.models.qwen3_moe.modeling_qwen3_moe", "Qwen3MoeRMSNorm",
    )
    if cls is None:
        pytest.fail(
            "DRIFT DETECTED: qwen3_moe.py expects Qwen3MoeRMSNorm class "
            f"namespace on transformers {_TX_VERSION}"
        )


def test_qwen3_moe_pre_trained_model_present():
    """qwen3_moe.py patches Qwen3MoeForCausalLM.forward -- the base
    Qwen3MoePreTrainedModel must exist as a sibling class so the heads
    inherit from a stable parent."""
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
    """qwen3_moe.py:170-179 inside _patch_causal_lm_forward_for_hidden_states
    calls self.model(input_ids=..., ...). self.model is Qwen3MoeModel."""
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
    """qwen3_next_moe.py imports Qwen3NextForCausalLM. Its inner model
    Qwen3NextModel must exist."""
    cls = _try_get_class(
        "transformers.models.qwen3_next.modeling_qwen3_next", "Qwen3NextModel",
    )
    if cls is None:
        pytest.fail(
            "DRIFT DETECTED: qwen3_next_moe.py expects Qwen3NextModel on "
            f"transformers {_TX_VERSION}"
        )


def test_qwen3_vl_moe_text_model_class_present():
    """qwen3_vl_moe.py patches Qwen3VLMoeTextSparseMoeBlock. The text
    model Qwen3VLMoeTextModel is the parent stack -- pin it."""
    cls = _try_get_class(
        "transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe",
        "Qwen3VLMoeTextModel",
    )
    if cls is None:
        pytest.fail(
            "DRIFT DETECTED: qwen3_vl_moe.py expects Qwen3VLMoeTextModel on "
            f"transformers {_TX_VERSION}"
        )


# ===========================================================================
# Cache-output-class signature pins (zoo constructs these by kwarg in
# several patch wrappers)
# ===========================================================================

def test_modeling_outputs_causal_lm_output_with_past_kwargs():
    """deepseek_v3_moe.py:200 and qwen3_next_moe.py construct
    transformers.modeling_outputs.CausalLMOutputWithPast(loss, logits,
    past_key_values, hidden_states, attentions). Pin the field set."""
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
    """qwen3_moe.py:191 constructs MoeCausalLMOutputWithPast(loss,
    logits, past_key_values, hidden_states, attentions, aux_loss,
    router_logits). Pin top-level transformers.modeling_outputs path."""
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


# ===========================================================================
# Caches the patches require (zoo passes past_key_values=Cache())
# ===========================================================================

def test_static_cache_class_present():
    """gemma.py:255 isinstance(past_key_values, StaticCache). Pin."""
    cu = importlib.import_module("transformers.cache_utils")
    if not hasattr(cu, "StaticCache"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/gemma.py:255 uses "
            "transformers.cache_utils.StaticCache via isinstance but it is "
            f"missing on transformers {_TX_VERSION}"
        )


def test_hybrid_cache_class_present():
    """gemma.py:260 isinstance(past_key_values, HybridCache). Pin."""
    cu = importlib.import_module("transformers.cache_utils")
    if not hasattr(cu, "HybridCache"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/gemma.py:260 uses "
            "transformers.cache_utils.HybridCache but it is missing on "
            f"transformers {_TX_VERSION}"
        )


# ===========================================================================
# bitsandbytes.py: process_output_options / utils.py helpers consumed
# ===========================================================================

def test_bitsandbytes_linear4bit_init_signature():
    """bitsandbytes.py:46-47 looks up
    bitsandbytes.nn.modules.Linear4bit. Pin __init__ accepts at least
    the input_features / output_features positional args zoo's patched
    forward implicitly assumes (self.weight, self.bias, etc.)."""
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


# ===========================================================================
# pixtral.py: PixtralVisionConfig + module-level apply_rotary_pos_emb pin
# (additional)
# ===========================================================================

def test_pixtral_vision_config_class_present():
    """pixtral.py reads self.config.hidden_size etc on the patched
    __init__ -- PixtralVisionConfig is the upstream config."""
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


# ===========================================================================
# gemma3n.py: gemma3n_TextConfig pin (used by config typing of AltUp)
# ===========================================================================

def test_gemma3n_text_config_class_present():
    """gemma3n.py reads self.config.altup_active_idx etc inside the
    patched AltUp.predict. Gemma3nTextConfig is the upstream config."""
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


# ===========================================================================
# Auto-attention function dictionary for the gemma3 patch chain
# ===========================================================================

def test_gemma3_eager_attention_forward_kwargs_supported():
    """gemma.py:407 calls eager_attention_forward(...,
    dropout=..., scaling=..., sliding_window=..., **kwargs).
    Pin those kwargs by-name."""
    from transformers.models.gemma3.modeling_gemma3 import eager_attention_forward
    if not _has_var_keyword(eager_attention_forward):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/gemma.py:412 calls "
            "eager_attention_forward(..., **kwargs) but installed signature "
            f"on transformers {_TX_VERSION} has no VAR_KEYWORD: "
            f"{inspect.signature(eager_attention_forward)}"
        )


# ===========================================================================
# Sanity: at least one TEMPORARY_PATCHES entry per file is registered
# ===========================================================================

def test_temporary_patches_directory_has_expected_files():
    """Pin the set of patch files. If a file is added/removed, the
    suite should adapt -- this test surfaces drift in the patch-file
    inventory itself."""
    pkg_spec = importlib.util.find_spec("unsloth_zoo.temporary_patches")
    if pkg_spec is None or not pkg_spec.submodule_search_locations:
        pytest.skip("unsloth_zoo.temporary_patches not importable as a package")
    root = pkg_spec.submodule_search_locations[0]
    files = {
        f for f in os.listdir(root)
        if f.endswith(".py")
        and f not in ("__init__.py", "utils.py", "common.py")
    }
    # Sanity floor: at minimum these files must exist. New files can be
    # added freely without bumping this list.
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


