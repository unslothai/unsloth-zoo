# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at
# your option) any later version.

"""Exhaustive upstream signature pins for (class, method) pairs rebound by unsloth_zoo/temporary_patches; a rename or
drop makes that patch silently no-op via raise_error(); CPU-only, optional libs skip via pytest.importorskip."""

from __future__ import annotations

import dis
import importlib
import importlib.util
import types

from _pytest import outcomes as _pytest_outcomes
import inspect
import os
from typing import Iterable

import pytest


pytest.importorskip("transformers")
import transformers  # noqa: E402
from packaging.version import Version  # noqa: E402

_TX_VERSION = getattr(transformers, "__version__", "0.0.0")
_TX_IS_5X = Version(_TX_VERSION) >= Version("5.0.0")
# 5.16.0 (upstream PR #47579, DTensor tensor-parallel rewrite) deleted mxfp4.dequantize and left
# shard_and_distribute_module as a tombstone that raises RuntimeError when called.
_TX_GE_5_16 = Version(_TX_VERSION) >= Version("5.16.0")
# transformers 5.10.0 is a yanked stale-branch release (superseded by 5.10.1) whose tree moved
# integrations/tensor_parallel.py to distributed/, so shard_and_distribute_module is absent; a retracted artifact, not
# an API decision.
_TX_IS_YANKED_5_10_0 = Version(_TX_VERSION) == Version("5.10.0")


def _skip_if_transformers_5x(reason: str) -> None:
    """On transformers 5.x forward params fold into **kwargs and patches no-op gracefully, so the detector only blocks CI; 4.57.6 still surfaces real drift."""
    if _TX_IS_5X:
        pytest.skip(
            f"transformers {_TX_VERSION}: {reason} (zoo patch silently "
            "no-ops via try/except + relaxed patch_function)"
        )


def _unimportable(dotted_module: str, exc: BaseException) -> str:
    return (
        f"{dotted_module!r} raised on import, so nothing can be said about "
        f"upstream drift here: {type(exc).__name__}: {exc}"
    )


def _names_the_target(exc: ModuleNotFoundError, dotted_module: str) -> bool:
    """exc.name is the deepest missing package (the parent for a removed model), so any package prefix of the target
    counts as gone; the trailing dot stops transformers.models.siglip also matching transformers.models.siglipx."""
    name = exc.name or ""
    return bool(name) and (dotted_module == name or dotted_module.startswith(name + "."))


def _try_get_class(dotted_module: str, class_name: str):
    """None means drift; a module that raises on import skips instead, since a newer timm dropping ImageNetInfo once made every gemma3n class look deleted."""
    try:
        mod = importlib.import_module(dotted_module)
    except ModuleNotFoundError as exc:
        # An absent target module is drift and returns None; a ModuleNotFoundError naming another module is a broken dependency and is skipped.
        if _names_the_target(exc, dotted_module):
            return None
        pytest.skip(_unimportable(dotted_module, exc))
    except Exception as exc:
        pytest.skip(_unimportable(dotted_module, exc))
    return getattr(mod, class_name, None)


def _bitsandbytes_or_skip():
    """Imports bitsandbytes, skipping when it is the unsloth stub from the MLX-on-torch simulation, since pinning the real signature against it would false-DRIFT."""
    bnb = pytest.importorskip("bitsandbytes")
    if getattr(bnb, "IS_UNSLOTH_STUB", False):
        pytest.skip("bitsandbytes is stubbed (MLX simulation active); cannot pin real upstream signature")
    return bnb


def _require_class(dotted_module: str, class_name: str, zoo_file: str):
    """Like _try_get_class but fails on drift when the parent module exists without the class; skips when the module is absent (a legitimate version gate)."""
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
    """Storage key patch_function uses for the original method: _original_<last-module-component>_<class>_<attr>;
    mirrors _get_unique_storage_name in temporary_patches/utils.py."""
    module_tail = getattr(cls, "__module__", "").rsplit(".", 1)[-1]
    class_name = getattr(cls, "__name__", "") or cls.__class__.__name__
    return f"_original_{module_tail}_{class_name}_{attr}"


def _resolve_upstream_method(cls, method_name: str):
    """Returns the upstream method: zoo patches at import time, so prefer the _original_<module>_<class>_<method> stash
    over the live wrapper; callers may guard via _maybe_skip_if_patched."""
    if not hasattr(cls, method_name):
        return None
    live = getattr(cls, method_name)
    storage_key = _original_attr_name(cls, method_name)
    original = getattr(cls, storage_key, None)
    if original is not None:
        return original
    qualname = getattr(live, "__qualname__", "") or ""
    if ".<locals>." in qualname and qualname.split(".", 1)[0].startswith("patch_"):
        # A zoo patch wrapper with no _original_ stash (rare: force=True + store_original=False) is handled by _maybe_skip_if_patched.
        return live
    return live


def _maybe_skip_if_patched(cls, method_name: str, zoo_file: str) -> None:
    """Skips when the live method is a zoo patch wrapper with no original stash, to avoid false failures on a (self, *args, **kwargs) signature."""
    if not hasattr(cls, method_name):
        return
    live = getattr(cls, method_name)
    storage_key = _original_attr_name(cls, method_name)
    original = getattr(cls, storage_key, None)
    if original is not None:
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
    if missing and _has_var_keyword(func):
        # Newer transformers wrap forwards as (self, *args, **kwargs) and kwargs pass through verbatim, so static
        # parameter-name checks are not meaningful; skip.
        pytest.skip(
            f"{label} is wrapped as `(self, *args, **kwargs)` on "
            f"transformers {_TX_VERSION}; static param-name verification "
            "is not meaningful for this passthrough signature."
        )
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


# Opcodes naming module-level bindings (what a setattr(module, name, ...) patch can reach); LOAD_ATTR and STORE_ATTR are
# excluded because co_names mixes attribute names with globals, so self.quantization_config.dequantize would masquerade
# as module-level dequantize.
_GLOBAL_NAME_OPS = frozenset((
    "LOAD_GLOBAL", "STORE_GLOBAL", "DELETE_GLOBAL",
    "LOAD_NAME", "STORE_NAME", "DELETE_NAME",
    "IMPORT_NAME", "IMPORT_FROM",
))


def _referenced_global_names(obj) -> set[str]:
    """Nested code is walked via co_consts; globals are filtered by opcode, not taken from co_names wholesale, so
    attribute names cannot make the drift check pass vacuously."""
    code = getattr(obj, "__code__", None) or getattr(obj, "__func__", None)
    code = getattr(code, "__code__", code)
    if code is None or not hasattr(code, "co_names"):
        return set()
    found: set[str] = set()
    stack = [code]
    while stack:
        current = stack.pop()
        for instruction in dis.get_instructions(current):
            if instruction.opname in _GLOBAL_NAME_OPS and isinstance(instruction.argval, str):
                found.add(instruction.argval)
        stack += [c for c in current.co_consts if hasattr(c, "co_names")]
    return found


def _assert_on_live_path(callee_name: str, caller, caller_label: str, zoo_file: str):
    """Signature pins missed a dead patch: mxfp4.dequantize kept its signature to 5.15 but 5.x loaded it via
    Mxfp4Dequantize -> dequantize_convertops; check the callee is still referenced."""
    if callee_name not in _referenced_global_names(caller):
        pytest.fail(
            f"DRIFT DETECTED: zoo temporary_patches/{zoo_file} patches "
            f"{callee_name}, but {caller_label} no longer references it on "
            f"transformers {_TX_VERSION}. The patch would still apply cleanly "
            f"and silently reach nothing, which is exactly how the un-transposed "
            f"MXFP4 layout survived from 5.0.0 to 5.16.0."
        )


def _has_var_keyword(func) -> bool:
    try:
        sig = inspect.signature(func)
    except Exception:
        return False
    return any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )


def _is_passthrough(func) -> bool:
    """Whether a signature is a bare (*args, **kwargs) passthrough; such a signature carries no arity, so an arity probe can neither confirm nor deny drift."""
    try:
        sig = inspect.signature(func)
    except Exception:
        return False
    kinds = [p.kind for p in sig.parameters.values()]
    return (
        inspect.Parameter.VAR_POSITIONAL in kinds
        and inspect.Parameter.VAR_KEYWORD in kinds
        and not any(
            k in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                  inspect.Parameter.POSITIONAL_ONLY,
                  inspect.Parameter.KEYWORD_ONLY)
            for k in kinds
        )
    )


def _unwrap_kernel_hub_func(obj, expected_name = None):
    """transformers swaps functions for kernels.layer.layer.Func modules whose forward closes over the original; with
    several closure functions prefer a name match, else one candidate."""
    found = []
    for candidate in (obj, getattr(type(obj), "forward", None)):
        if candidate is None:
            continue
        for cell in getattr(candidate, "__closure__", None) or ():
            try:
                inner = cell.cell_contents
            except ValueError:
                continue
            if not (inspect.isfunction(inner) or inspect.isbuiltin(inner)):
                continue
            if expected_name is not None and getattr(inner, "__name__", None) == expected_name:
                return inner
            found.append(inner)
    if len(found) == 1:
        # On a name miss (upstream rename or decorator), a lone closed-over function is still what the wrapper forwards to.
        return found[0]
    return obj


def _positional_arity_from_source(module, name):
    """Counts positionals from the module's own source, a last resort when the live attribute is an opaque (*args, **kwargs) passthrough."""
    import ast

    try:
        source = inspect.getsource(module)
        tree = ast.parse(source)
    except Exception:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return len(node.args.posonlyargs) + len(node.args.args)
    return None


def _resolve_for_arity(func, expected_name = None):
    """Strip functools.wraps chains and kernels-hub wrappers before an arity probe."""
    seen = set()
    while hasattr(func, "__wrapped__") and id(func) not in seen:
        seen.add(id(func))
        func = func.__wrapped__
    if _is_passthrough(func):
        func = _unwrap_kernel_hub_func(func, expected_name)
    return func


# bitsandbytes.py patches the top-level bitsandbytes.nn.Linear4bit alias and its .forward; nn.modules.Linear4bit.forward is covered elsewhere.

def test_bitsandbytes_top_level_Linear4bit_alias():
    """bitsandbytes.py:110 patches the top-level bitsandbytes.nn.Linear4bit alias; pins its presence and .forward method."""
    bnb = _bitsandbytes_or_skip()
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
    """bitsandbytes.py:47-67 conditionally deletes __torch_function__ from Params4bit to fix torch.compile recursion; pins that the class is present."""
    bnb = _bitsandbytes_or_skip()
    p4 = getattr(bnb.nn.modules, "Params4bit", None)
    if p4 is None:
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/bitsandbytes.py:47 expects "
            "bitsandbytes.nn.modules.Params4bit but it is missing"
        )


def test_bitsandbytes_fix_4bit_weight_quant_state_from_module_present():
    """bitsandbytes.py:48 and :73 call fix_4bit_weight_quant_state_from_module(self), so that function must exist in bitsandbytes.nn.modules."""
    bnb = _bitsandbytes_or_skip()
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
    bnb = _bitsandbytes_or_skip()
    if not hasattr(bnb, "matmul_4bit"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/bitsandbytes.py:106 expects "
            "bitsandbytes.matmul_4bit() but it is missing"
        )


# deepseek_v3_moe.py: patches DeepseekV3{NaiveMoe,MoE,ForCausalLM}.forward.

def test_deepseek_v3_naive_moe_class_gated_5x():
    """deepseek_v3_moe.py:56-61 imports DeepseekV3NaiveMoe, which exists only on 5.x; the patch bails via try/except on 4.x."""
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
    """deepseek_v3_moe.py:59 imports DeepseekV3TopkRouter as a gate; if it is missing the whole patch entry silently no-ops."""
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
    """deepseek_v3_moe.py:125 patches DeepseekV3MoE.forward(self, hidden_states); pins a single required positional argument."""
    cls = _try_get_class(
        "transformers.models.deepseek_v3.modeling_deepseek_v3",
        "DeepseekV3MoE",
    )
    if cls is None:
        pytest.skip(f"DeepseekV3MoE absent on transformers {_TX_VERSION}")
    fwd = _assert_method_exists(cls, "forward", "deepseek_v3_moe.py")
    params = _param_names(fwd)
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
    """deepseek_v3_moe.py:142 patches DeepseekV3ForCausalLM.forward by name (input_ids, ..., output_router_logits,
    cache_position, logits_to_keep); output_router_logits may sit in **kwargs."""
    _skip_if_transformers_5x(
        "DeepseekV3ForCausalLM.forward moved cache_position into "
        "**kwargs: Unpack[TransformersKwargs]"
    )
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
    if "output_router_logits" not in _param_names(fwd) and not _has_var_keyword(fwd):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/deepseek_v3_moe.py:171 "
            "forwards output_router_logits=output_router_logits but installed "
            f"DeepseekV3ForCausalLM.forward on transformers {_TX_VERSION} "
            f"has neither an explicit output_router_logits param nor a "
            f"**kwargs catch-all: {inspect.signature(fwd)}"
        )


# UNSLOTH_FORCE_FLOAT32-gated Gemma3Model._update_causal_mask patch in gemma.py; upstream removed it in 4.55+, so the patch is a no-op on modern transformers.

def test_gemma3_force_fp32_update_causal_mask_gated():
    """gemma.py:308-310 patches Gemma3Model and Gemma3ForConditionalGeneration._update_causal_mask under UNSLOTH_FORCE_FLOAT32=1."""
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
    """gemma3n.py:148 patches Gemma3nTextAltUp.scale_corrected_output for fullgraph=True; its signature is (self, corrected)."""
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
    """gemma3n.py:143-148 hasattr-guards predict, correct and scale_corrected_output; fails loudly only when all three are gone."""
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
    """gemma3n.py:53-85 Gemma3nRMSNorm_forward helper reads soft_embedding_norm, hard_embedding_norm,
    embedding_projection and embedding_post_projection_norm on Gemma3nMultimodalEmbedder."""
    embedder = _require_class(
        "transformers.models.gemma3n.modeling_gemma3n",
        "Gemma3nMultimodalEmbedder",
        "gemma3n.py",
    )
    # zoo's patched forward dereferences self.soft_embedding_norm and self.hard_embedding_norm, so an upstream rename makes it raise AttributeError.
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
    """gemma4.py:655 patches Gemma4TextMLP.forward(self, x), which takes a single positional argument."""
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
    """gemma4.py:644-652 patched forward reads self.gate_proj, up_proj, down_proj and act_fn; pins their presence in __init__ source."""
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


# gemma4_moe.py patches Gemma4Text{Experts,DecoderLayer,MoEBlock}{.forward,.__init__} and Gemma4ForConditionalGeneration.forward; all 5.0+-gated.

# Each target is a 5.0+ stacked-experts class whose forward zoo rebinds; absent on 4.x.
@pytest.mark.parametrize("module, cls_name, zoo_file, absent_note", [
    pytest.param("transformers.models.gemma4.modeling_gemma4", "Gemma4TextExperts", "gemma4_moe.py", "5.0+-only", id = "gemma4_text_experts_forward_signature"),
    pytest.param("transformers.models.gemma4.modeling_gemma4", "Gemma4TextMoEBlock", "gemma4_moe.py", "5.0+-only legacy MoE layout", id = "gemma4_text_moe_block_forward_signature"),
    pytest.param("transformers.models.glm4_moe_lite.modeling_glm4_moe_lite", "Glm4MoeLiteNaiveMoe", "glm4_moe.py", "glm4_moe_lite is 5.0+-only", id = "glm4_moe_lite_naive_moe_forward_signature"),
    pytest.param("transformers.models.glm4_moe_lite.modeling_glm4_moe_lite", "Glm4MoeLiteMoE", "glm4_moe.py", "glm4_moe_lite is 5.0+-only", id = "glm4_moe_lite_moe_forward_signature"),
    pytest.param("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe", "Qwen3_5MoeSparseMoeBlock", "qwen3_5_moe.py", "qwen3_5_moe is 5.0+-only", id = "qwen3_5_moe_sparse_moe_block_forward_signature"),
    pytest.param("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe", "Qwen3_5MoeExperts", "qwen3_5_moe.py", "qwen3_5_moe is 5.0+-only", id = "qwen3_5_moe_experts_forward_signature"),
    pytest.param("transformers.models.qwen3_moe.modeling_qwen3_moe", "Qwen3MoeExperts", "qwen3_moe.py", "5.0+-only; zoo gracefully patches the old SparseMoeBlock instead", id = "qwen3_moe_experts_forward_signature_5x"),
    pytest.param("transformers.models.qwen3_next.modeling_qwen3_next", "Qwen3NextExperts", "qwen3_next_moe.py", "5.0+-only", id = "qwen3_next_experts_forward_signature"),
])
def test_moe_forward_signature(module, cls_name, zoo_file, absent_note):
    cls = _try_get_class(module, cls_name)
    if cls is None:
        pytest.skip(f"{cls_name} absent on transformers {_TX_VERSION} ({absent_note})")
    fwd = _assert_method_exists(cls, "forward", zoo_file)
    _assert_params_superset(
        fwd, required = ["hidden_states"], zoo_file = zoo_file, label = f"{cls_name}.forward",
    )


def test_gemma4_text_decoder_layer_init_signature():
    """gemma4_moe.py:287 patches Gemma4TextDecoderLayer.__init__(self, config, layer_idx)."""
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


def test_gemma4_for_conditional_generation_forward_named_params():
    """gemma4_moe.py:208 patches Gemma4ForConditionalGeneration.forward; pins its by-name kwargs (input_ids, attention_mask, ..., logits_to_keep)."""
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
    """gemma4_moe.py:189 constructs Gemma4CausalLMOutputWithPast with loss, logits, past_key_values, hidden_states and attentions kwargs."""
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


# gpt_oss.py patch sites Mxfp4GptOssExperts and GptOssConfig (source-only) and GptOssForCausalLM.forward; the forward's tail is not covered by sibling tests.

def test_mxfp4_gpt_oss_experts_class_present_and_init_signature():
    """gpt_oss.py:433 replaces transformers.integrations.mxfp4.Mxfp4GptOssExperts; pins the class and its __init__(self, config)."""
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
    """gpt_oss.py:2813 replaces GptOssConfig with Old_GptOssConfig and pins kwarg names; on 5.x (rope_theta became
    rope_parameters) it gates on a source match and skips; strict on 4.57.6."""
    _skip_if_transformers_5x(
        "GptOssConfig.__init__ renamed rope_theta -> rope_parameters"
    )
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
    """gpt_oss.py:2890 patches GptOssForCausalLM.forward with by-name kwargs (input_ids, attention_mask, ..., logits_to_keep)."""
    _skip_if_transformers_5x(
        "GptOssForCausalLM.forward moved cache_position into "
        "**kwargs: Unpack[TransformersKwargs]"
    )
    cls = _try_get_class(
        "transformers.models.gpt_oss.modeling_gpt_oss", "GptOssForCausalLM",
    )
    if cls is None:
        pytest.skip(f"GptOssForCausalLM absent on transformers {_TX_VERSION}")
    fwd = _assert_method_exists(cls, "forward", "gpt_oss.py")
    # output_attentions and output_hidden_states may be folded into **kwargs on newer transformers, so only guaranteed-stable params are pinned.
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
    """gpt_oss.py:2949 constructs MoeCausalLMOutputWithPast with loss, aux_loss, logits and past_key_values kwargs."""
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
    """gpt_oss.py:2126 imports DynamicCache from modeling_gpt_oss in a soft try/except that silently falls back to a no-op lambda, so a rename must fail loudly."""
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
    """gpt_oss.py:2122 imports apply_rotary_pos_emb from modeling_gpt_oss; this pins that re-export."""
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
    """gpt_oss.py:2121 imports MoeModelOutputWithPast, the return type of the patched GptOssModel.forward."""
    mod = importlib.import_module(
        "transformers.models.gpt_oss.modeling_gpt_oss"
    )
    if not hasattr(mod, "MoeModelOutputWithPast"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/gpt_oss.py:2121 expects "
            "transformers.models.gpt_oss.modeling_gpt_oss.MoeModelOutputWithPast "
            "but it is missing"
        )


# misc.py patches CsmDepthDecoderForCausalLM.forward and CsmForConditionalGeneration.forward; the tail is not covered by sibling tests.

def test_csm_depth_decoder_for_causal_lm_forward_named_params():
    """misc.py:239 patches CsmDepthDecoderForCausalLM.forward with named params (input_ids, backbone_last_hidden_state,
    ..., logits_to_keep); resolves via the _original_* stash."""
    _skip_if_transformers_5x(
        "CsmDepthDecoderForCausalLM.forward moved cache_position into "
        "**kwargs: Unpack[TransformersKwargs]"
    )
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


def _tiny_csm_depth_decoder(torch):
    csm = pytest.importorskip("transformers.models.csm.modeling_csm")
    config_mod = pytest.importorskip("transformers.models.csm.configuration_csm")
    config = config_mod.CsmDepthDecoderConfig(
        vocab_size=11, num_codebooks=4, backbone_hidden_size=8,
        hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=8, head_dim=4, use_cache=False,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    model = csm.CsmDepthDecoderForCausalLM(config).train()
    model.model.embed_tokens.weight.requires_grad_(False)
    return model


@pytest.mark.parametrize("backbone_requires_grad", [True, False])
def test_csm_depth_decoder_leaf_embedding_hook_backward(backbone_requires_grad):
    """CSM patch handles GC hook leaf embeddings without losing grad input."""
    torch = pytest.importorskip("torch")
    from unsloth_zoo.temporary_patches.misc import patch_CsmDepthDecoderForCausalLM_forward

    patch_CsmDepthDecoderForCausalLM_forward()
    model = _tiny_csm_depth_decoder(torch)
    seen = {}
    original_forward = model.model.forward

    def wrapped_model_forward(*args, **kwargs):
        seen["inputs_embeds_requires_grad"] = kwargs["inputs_embeds"].requires_grad
        return original_forward(*args, **kwargs)

    model.model.forward = wrapped_model_forward
    hook = model.model.embed_tokens.register_forward_hook(
        lambda module, args, output: output.requires_grad_(True)
    )
    try:
        backbone_state = torch.randn(1, 8, requires_grad=backbone_requires_grad)
        out = model(
            input_ids=torch.tensor([[0, 2, 3, 4]], dtype=torch.long),
            backbone_last_hidden_state=backbone_state,
            labels=torch.tensor([[5, 6, 7, 8]], dtype=torch.long),
            return_dict=True,
        )
        out.loss.backward()

        assert seen["inputs_embeds_requires_grad"] is True
        if backbone_requires_grad:
            assert backbone_state.grad is not None
            assert torch.count_nonzero(backbone_state.grad) > 0
    finally:
        hook.remove()
        model.model.forward = original_forward


def test_csm_audio_embeddings_tie_helper():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from unsloth_zoo.temporary_patches.misc import _tie_csm_audio_embeddings

    audio = torch.nn.Embedding(4, 3)
    depth = torch.nn.Embedding(4, 3)
    model = SimpleNamespace(
        backbone_model=SimpleNamespace(
            embed_tokens=SimpleNamespace(embed_audio_tokens=audio),
        ),
        depth_decoder=SimpleNamespace(
            model=SimpleNamespace(embed_tokens=depth),
        ),
    )

    assert audio.weight.data_ptr() != depth.weight.data_ptr()
    _tie_csm_audio_embeddings(model)
    assert audio.weight.data_ptr() == depth.weight.data_ptr()


def test_csm_audio_embeddings_tie_helper_respects_config():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from unsloth_zoo.temporary_patches.misc import _tie_csm_audio_embeddings

    audio = torch.nn.Embedding(4, 3)
    depth = torch.nn.Embedding(4, 3)
    model = SimpleNamespace(
        config=SimpleNamespace(tie_codebooks_embeddings=False),
        backbone_model=SimpleNamespace(
            embed_tokens=SimpleNamespace(embed_audio_tokens=audio),
        ),
        depth_decoder=SimpleNamespace(
            model=SimpleNamespace(embed_tokens=depth),
        ),
    )

    _tie_csm_audio_embeddings(model)
    assert audio.weight.data_ptr() != depth.weight.data_ptr()


def test_csm_audio_eos_label_helper_keeps_trainer_counts_aligned():
    torch = pytest.importorskip("torch")
    from unsloth_zoo.temporary_patches.misc import _label_csm_audio_eos_tokens

    labels = torch.tensor([[-100, 128002, -100, -100]])
    input_ids = torch.tensor([[128000, 128002, 128003, 0]])

    labels = _label_csm_audio_eos_tokens(input_ids, labels, 128003)

    assert labels.tolist() == [[-100, 128002, 128003, -100]]


def test_csm_audio_eos_label_helper_supports_numpy():
    np = pytest.importorskip("numpy")
    from unsloth_zoo.temporary_patches.misc import _label_csm_audio_eos_tokens

    labels = np.array([[-100, 128002, -100, -100]])
    input_ids = np.array([[128000, 128002, 128003, 0]])

    labels = _label_csm_audio_eos_tokens(input_ids, labels, 128003)

    assert labels.tolist() == [[-100, 128002, 128003, -100]]


def test_csm_processor_output_uses_configured_audio_eos_id():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from unsloth_zoo.temporary_patches.misc import _label_csm_processor_output

    processor = SimpleNamespace(audio_eos_token_id=42)
    output = {
        "input_ids": torch.tensor([[128000, 128002, 42, 0]]),
        "labels": torch.tensor([[-100, 128002, -100, -100]]),
    }

    output = _label_csm_processor_output(processor, output)

    assert output["labels"].tolist() == [[-100, 128002, 42, -100]]


def test_csm_processor_output_ignores_non_mappings():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from unsloth_zoo.temporary_patches.misc import _label_csm_processor_output

    tensor_output = torch.tensor([[128000, 128002]])
    text_output = "input_ids and labels are just rendered text"

    assert _label_csm_processor_output(SimpleNamespace(), tensor_output) is tensor_output
    assert _label_csm_processor_output(SimpleNamespace(), text_output) is text_output


def test_csm_for_conditional_generation_forward_named_params():
    """misc.py:373 patches CsmForConditionalGeneration.forward (input_ids, input_values, ..., logits_to_keep); resolves via the _original_* stash past zoo's wrapper."""
    _skip_if_transformers_5x(
        "CsmForConditionalGeneration.forward moved cache_position into "
        "**kwargs: Unpack[TransformersKwargs]"
    )
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
    """misc.py:221 calls ForCausalLMLoss(logits, labels, vocab_size, shift_labels); pins that signature."""
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
    """misc.py:770 patches CsmForConditionalGeneration._merge_input_ids_with_input_values; pins that the method exists."""
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


@pytest.mark.parametrize("module, name, zoo_site", [
    pytest.param("transformers.models.siglip.modeling_siglip", "SiglipEncoderLayer", "misc.py:1228", id = "misc_siglip_encoder_layer_class_present"),
    pytest.param("transformers.models.gemma3.modeling_gemma3", "Gemma3DecoderLayer", "gemma.py", id = "gemma3_decoder_layer_class_present"),
    pytest.param("transformers.models.gemma3.modeling_gemma3", "Gemma3TextModel", "gemma.py:233", id = "gemma3_text_model_class_present"),
    pytest.param("transformers.models.gemma3.modeling_gemma3", "Gemma3PreTrainedModel", "gemma.py", id = "gemma3_pre_trained_model_class_present"),
    pytest.param("transformers.models.gemma3n.modeling_gemma3n", "Gemma3nForConditionalGeneration", "gemma3n.py", id = "gemma3n_for_conditional_generation_class_present"),
    pytest.param("transformers.models.gemma3n.modeling_gemma3n", "Gemma3nRMSNorm", "gemma3n.py:53", id = "gemma3n_RMSNorm_class_present"),
    pytest.param("transformers.models.qwen3_moe.modeling_qwen3_moe", "Qwen3MoeRMSNorm", "qwen3_moe.py", id = "qwen3_moe_rms_norm_class_present"),
    pytest.param("transformers.models.qwen3_moe.modeling_qwen3_moe", "Qwen3MoePreTrainedModel", "qwen3_moe.py", id = "qwen3_moe_pre_trained_model_present"),
    pytest.param("transformers.models.qwen3_moe.modeling_qwen3_moe", "Qwen3MoeModel", "qwen3_moe.py:170", id = "qwen3_moe_model_present"),
    pytest.param("transformers.models.qwen3_next.modeling_qwen3_next", "Qwen3NextModel", "qwen3_next_moe.py", id = "qwen3_next_model_class_present"),
    pytest.param("transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe", "Qwen3VLMoeTextModel", "qwen3_vl_moe.py", id = "qwen3_vl_moe_text_model_class_present"),
    pytest.param("transformers.models.pixtral.configuration_pixtral", "PixtralVisionConfig", "pixtral.py:36", id = "pixtral_vision_config_class_present"),
])
def test_patched_class_present(module, name, zoo_site):
    if _try_get_class(module, name) is None:
        pytest.fail(
            f"DRIFT DETECTED: zoo temporary_patches/{zoo_site} expects {module}.{name} "
            f"but it is missing on transformers {_TX_VERSION}"
        )


def test_misc_mllama_vision_classes_present():
    """misc.py:1116-1119 imports MllamaVisionConfig, MllamaVisionAttention, MllamaVisionMLP, MllamaVisionEncoder and MllamaVisionEncoderLayer."""
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
    """mxfp4.py:173 patches transformers.integrations.mxfp4.convert_moe_packed_tensors(blocks, scales, *, dtype, rows_per_chunk)."""
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
    """mxfp4.dequantize is the live loader hook only on 4.x; from 5.1.0 Mxfp4Dequantize and dequantize_convertops take
    over (dequantize deleted in 5.16.0), so zoo gates its patch below 5.0.0."""
    if _TX_IS_5X:
        pytest.skip(
            f"transformers {_TX_VERSION}: mxfp4.dequantize is not the live "
            "loader hook on 5.x and zoo gates its patch to < 5.0.0; the "
            "ConversionOps replacement is pinned separately"
        )
    mod_name = "transformers.integrations.mxfp4"
    try:
        mod = importlib.import_module(mod_name)
    except Exception as exc:
        pytest.skip(f"mxfp4 integrations unavailable: {exc}")
    fn = getattr(mod, "dequantize", None)
    if fn is None:
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/mxfp4.py expects "
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
            f"DRIFT DETECTED: zoo temporary_patches/mxfp4.py forwards "
            f"by name (model=..., empty_param=..., casting_dtype=..., "
            f"to_contiguous=..., rank=..., device_mesh=...) via **kwargs but "
            f"upstream transformers.integrations.mxfp4.dequantize lost its "
            f"**kwargs catch-all on {_TX_VERSION}: {inspect.signature(fn)}"
        )


# Lists every upstream shape zoo's mxfp4.py can replace (patch_function needs an exact parameter match, so keep in
# sync); 5.0.0 alone takes target_device, 5.1.0 and newer leave placement to the caller; drop the 3-tuple if the 5.0.0
# arm is removed.
_CONVERTOPS_HANDLED_SIGNATURES = (
    ("blocks", "scales"),
    ("blocks", "scales", "target_device"),
)


def test_mxfp4_dequantize_convertops_signature():
    """Zoo rebinds dequantize_convertops (absent on 4.x) to re-apply transpose(1, 2); it needs an exact parameter tuple,
    since can_safely_patch refuses other arities; a superset check passed 5.0.0's 3-arg form."""
    try:
        mod = importlib.import_module("transformers.integrations.mxfp4")
    except Exception as exc:
        pytest.skip(f"mxfp4 integrations unavailable: {exc}")
    fn = getattr(mod, "dequantize_convertops", None)
    if not _TX_IS_5X:
        if fn is None:
            pytest.skip(
                f"transformers {_TX_VERSION} predates the ConversionOps mxfp4 "
                "loading path; zoo patches mxfp4.dequantize instead"
            )
        return
    if fn is None:
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/mxfp4.py expects "
            "transformers.integrations.mxfp4.dequantize_convertops but it is "
            f"missing on transformers {_TX_VERSION}. Without it the loader "
            "keeps zoo's un-transposed [E, D, G*B*2] layout and GPT-OSS loads "
            "with dims 1 and 2 swapped."
        )
    got = tuple(_param_names(fn))
    if got not in _CONVERTOPS_HANDLED_SIGNATURES:
        pytest.fail(
            f"DRIFT DETECTED: zoo temporary_patches/mxfp4.py replaces "
            f"transformers.integrations.mxfp4.dequantize_convertops and has arms "
            f"for {list(_CONVERTOPS_HANDLED_SIGNATURES)}, but transformers "
            f"{_TX_VERSION} declares {inspect.signature(fn)}. patch_function will "
            f"refuse a replacement whose parameters do not match exactly, so the "
            f"transpose(1, 2) that zoo's un-transposed convert_moe_packed_tensors "
            f"leaves off would never be restored and GPT-OSS would load with dims "
            f"1 and 2 swapped."
        )
    # Mxfp4Dequantize and its call site must both still exist for the patch to reach the loader.
    cls = getattr(mod, "Mxfp4Dequantize", None)
    if cls is None:
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/mxfp4.py relies on "
            "transformers.integrations.mxfp4.Mxfp4Dequantize calling "
            f"dequantize_convertops, but the class is missing on {_TX_VERSION}"
        )
    convert = getattr(cls, "convert", None)
    if convert is None:
        pytest.fail(
            "DRIFT DETECTED: transformers.integrations.mxfp4.Mxfp4Dequantize "
            f"lost its .convert method on {_TX_VERSION}; zoo's "
            "dequantize_convertops patch has no live caller"
        )
    _assert_on_live_path(
        "dequantize_convertops", convert, "Mxfp4Dequantize.convert", "mxfp4.py",
    )


def test_mxfp4_legacy_dequantize_is_on_the_live_path_on_4x():
    """On 4.x quantizer_mxfp4 must still call mxfp4.dequantize, since a patch on an uncalled function is a silent no-op;
    an unimportable caller skips, as broken optional backends say nothing about drift."""
    if _TX_IS_5X:
        pytest.skip(
            f"transformers {_TX_VERSION}: the legacy dequantize hook is not on "
            "the live path on 5.x; test_mxfp4_dequantize_convertops_signature "
            "pins the ConversionOps replacement instead"
        )
    try:
        mod = importlib.import_module("transformers.integrations.mxfp4")
        quantizer = importlib.import_module(
            "transformers.quantizers.quantizer_mxfp4"
        )
    except Exception as exc:
        pytest.skip(_unimportable("transformers.quantizers.quantizer_mxfp4", exc))
    if getattr(mod, "dequantize", None) is None:
        pytest.skip(
            f"transformers {_TX_VERSION} has no mxfp4.dequantize; covered by "
            "test_mxfp4_dequantize_signature"
        )
    cls = getattr(quantizer, "Mxfp4HfQuantizer", None)
    if cls is None:
        pytest.fail(
            "DRIFT DETECTED: transformers.quantizers.quantizer_mxfp4."
            f"Mxfp4HfQuantizer is missing on {_TX_VERSION}"
        )
    callers = [
        name for name, fn in vars(cls).items()
        if isinstance(fn, types.FunctionType)
        and "dequantize" in _referenced_global_names(fn)
    ]
    if not callers:
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/mxfp4.py patches "
            "transformers.integrations.mxfp4.dequantize, but no method on "
            f"Mxfp4HfQuantizer calls it on transformers {_TX_VERSION}. The "
            "patch would apply cleanly and reach nothing."
        )


def test_referenced_global_names_ignores_attribute_accesses():
    """co_names mixes attribute names with globals and Mxfp4HfQuantizer reads self.quantization_config.dequantize in
    several methods, so attribute reads must not satisfy the live-call check."""
    def attribute_only(cfg):
        cfg.dequantize = True
        return cfg.dequantize

    def real_global_call(x):
        return dequantize(x)  # noqa: F821

    def real_import_from(x):
        from transformers.integrations.mxfp4 import dequantize
        return dequantize(x)

    assert "dequantize" not in _referenced_global_names(attribute_only)
    assert "dequantize" in _referenced_global_names(real_global_call)
    assert "dequantize" in _referenced_global_names(real_import_from)


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
    """Positional arity must be 8; 5.16.0 made it a RuntimeError tombstone so both call sites are gated off; import
    failure is a hard fail except yanked 5.10.0 (xfail not skip, so it stays reported); widening the gate below
    5.16.0 would not help since 5.10.0 is inside it"""
    try:
        mod = importlib.import_module("transformers.integrations.tensor_parallel")
    except Exception as exc:
        if _TX_IS_YANKED_5_10_0:
            pytest.xfail(
                "transformers 5.10.0 is YANKED upstream (published from a stale "
                "branch, replaced by 5.10.1 minutes later). It moved "
                "integrations/tensor_parallel.py to distributed/tensor_parallel.py "
                "and shipped no shard_and_distribute_module at all, so "
                "patch_gpt_oss aborts at its gated import and silently drops "
                "load_and_swizzle_mxfp4 and replace_with_mxfp4_linear on installs "
                f"that have triton_kernels. Upgrade to 5.10.1 or newer. ({exc})"
            )
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/mxfp4.py and gpt_oss.py import "
            f"transformers.integrations.tensor_parallel but it is missing: {exc}"
        )
    fn = getattr(mod, "shard_and_distribute_module", None)
    if fn is None:
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/mxfp4.py expects "
            "shard_and_distribute_module but it is missing on transformers "
            f"{_TX_VERSION}"
        )
    if _TX_GE_5_16:
        pytest.skip(
            f"transformers {_TX_VERSION}: shard_and_distribute_module is a "
            "deprecation tombstone that raises RuntimeError; zoo gates both "
            "call sites to < 5.16.0 and uses tp_plan= above it"
        )
    # The 8 positionals are: model, param_value, empty_param, dq_param_name, casting_dtype, to_contiguous, rank, device_mesh.
    params = [
        p for p in inspect.signature(fn).parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                      inspect.Parameter.POSITIONAL_ONLY)
    ]
    if len(params) < 8:
        sig = inspect.signature(fn)
        # A bare (*args, **kwargs) is a passthrough, not a narrowed signature; upstream's 5.16.0 tombstone accepts any
        # call and raises RuntimeError, so do not report 0 positionals.
        if _has_var_keyword(fn) and any(
            p.kind is inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()
        ):
            detail = (
                f"accepts any call and forwards nothing: {sig}. That is a "
                "deprecation tombstone or passthrough, not a narrowed "
                "signature, so the call site cannot be repaired by dropping "
                "arguments and must be gated off or ported"
            )
        else:
            detail = f"accepts only {len(params)}: {sig}"
        pytest.fail(
            f"DRIFT DETECTED: zoo temporary_patches/mxfp4.py calls "
            f"shard_and_distribute_module with 8 positionals but the installed "
            f"signature on transformers {_TX_VERSION} {detail}"
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
    """pixtral.py patches PixtralAttention.forward(self, hidden_states, attention_mask, position_embeddings, ...);
    resolves via the _original_* stash to read upstream past zoo's wrapper."""
    cls = _require_class(
        "transformers.models.pixtral.modeling_pixtral",
        "PixtralAttention",
        "pixtral.py",
    )
    _maybe_skip_if_patched(cls, "forward", "pixtral.py")
    upstream_fwd = _resolve_upstream_method(cls, "forward")
    # Newer transformers (>=5.x) wrap forwards as (self, *args, **kwargs); zoo's patch passes kwargs through and static
    # param names are not introspectable, so skip instead of reporting drift.
    if _has_var_keyword(upstream_fwd):
        pytest.skip(
            "PixtralAttention.forward is now `(self, *args, **kwargs)` on "
            f"transformers {_TX_VERSION}; static param-name verification "
            "is not meaningful for this wrapped signature."
        )
    _assert_params_superset(
        upstream_fwd,
        required=["hidden_states", "attention_mask", "position_embeddings"],
        zoo_file="pixtral.py",
        label="PixtralAttention.forward",
    )


@pytest.mark.parametrize("module, name, zoo_site", [
    pytest.param("transformers.models.pixtral.modeling_pixtral", "apply_rotary_pos_emb", "pixtral.py:30", id = "pixtral_apply_rotary_pos_emb_present"),
    pytest.param("transformers.models.qwen3_moe.modeling_qwen3_moe", "MoeCausalLMOutputWithPast", "qwen3_moe.py:349", id = "qwen3_moe_for_causal_lm_output_class_present"),
    pytest.param("transformers.quantizers.quantizer_mxfp4", "is_kernels_available", "gpt_oss.py:136", id = "gpt_oss_mxfp4_quantizer_is_kernels_available_present"),
    pytest.param("transformers.models.gpt_oss.modeling_gpt_oss", "GptOssPreTrainedModel", "gpt_oss.py:2832", id = "gpt_oss_pretrained_model_present"),
])
def test_patched_module_attribute_present(module, name, zoo_site):
    if not hasattr(importlib.import_module(module), name):
        pytest.fail(
            f"DRIFT DETECTED: zoo temporary_patches/{zoo_site} expects {module}.{name} "
            f"but it is missing on transformers {_TX_VERSION}"
        )


def test_pixtral_attention_init_attrs_present():
    """pixtral.py's patched __init__ reads self.config.hidden_size, self.config.num_attention_heads and self.config.attention_dropout."""
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


# qwen3_5_moe.py: Qwen3_5MoeForCausalLM (5.0+-only).

def test_qwen3_5_moe_for_causal_lm_class_present():
    """qwen3_5_moe.py reads Qwen3_5MoeForCausalLM and MoeCausalLMOutputWithPast for the GRPO hidden-states patch."""
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


# qwen3_moe.py: Qwen3MoeForCausalLM.forward (tail).

def test_qwen3_moe_for_causal_lm_forward_named_params():
    """qwen3_moe.py patches Qwen3MoeForCausalLM.forward via _patch_causal_lm_forward_for_hidden_states; 5.x-skipped since cache_position moved into **kwargs."""
    _skip_if_transformers_5x(
        "Qwen3MoeForCausalLM.forward moved cache_position into "
        "**kwargs: Unpack[TransformersKwargs]"
    )
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


# qwen3_next_moe.py: Qwen3NextForCausalLM.forward tail.

def test_qwen3_next_for_causal_lm_forward_named_params():
    """qwen3_next_moe.py patches Qwen3NextForCausalLM.forward via _patch_causal_lm_forward_for_hidden_states; 5.x-skipped since cache_position moved into **kwargs."""
    _skip_if_transformers_5x(
        "Qwen3NextForCausalLM.forward moved cache_position into "
        "**kwargs: Unpack[TransformersKwargs]"
    )
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
    """qwen3_vl_moe.py patches Qwen3VLMoeForConditionalGeneration.forward with by-name kwargs incl. logits_to_keep;
    5.x-skipped since cache_position moved into **kwargs."""
    _skip_if_transformers_5x(
        "Qwen3VLMoeForConditionalGeneration.forward moved cache_position "
        "into **kwargs: Unpack[TransformersKwargs]"
    )
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
    """qwen3_vl_moe.py constructs Qwen3VLMoeCausalLMOutputWithPast (loss, aux_loss, ..., rope_deltas)."""
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


def test_qwen3_vl_moe_act2fn_dict_present():
    """qwen3_vl_moe's patched __init__ indexes ACT2FN[config.hidden_act], so ACT2FN must still contain 'silu'."""
    from transformers.activations import ACT2FN  # noqa: F401
    if "silu" not in ACT2FN:
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/qwen3_vl_moe.py:236 expects "
            "transformers.activations.ACT2FN['silu'] but the key is missing"
        )


# misc.py additional patch sites: config-mapping, tokenizer attrs,
# mask-utils, modernbert, lfm2, peft dispatch, trl, vllm, qwen2-vl shims.

def test_misc_config_mapping_present_for_ministral3_register():
    """misc.py imports CONFIG_MAPPING from transformers.models.auto.configuration_auto and calls .register() on it for ministral3."""
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
    """misc.py's ministral3 alias targets transformers.MinistralConfig at top level, so that attribute must exist."""
    if not hasattr(transformers, "MinistralConfig"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/misc.py:48 expects "
            "transformers.MinistralConfig at top level but it is missing on "
            f"{_TX_VERSION}"
        )


def test_misc_pretrained_tokenizer_base_convert_added_tokens_method():
    """misc.py expects PreTrainedTokenizerBase.convert_added_tokens to be a classmethod."""
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    if not hasattr(PreTrainedTokenizerBase, "convert_added_tokens"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/misc.py:67 expects "
            "PreTrainedTokenizerBase.convert_added_tokens but it is missing "
            f"on transformers {_TX_VERSION}"
        )


def test_misc_added_token_class_present():
    """misc.py's patched convert_added_tokens constructs AddedToken(content=...), so AddedToken must accept a content field."""
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
    """misc.py wraps PreTrainedTokenizerBase.__init__(self, **kwargs) to reject or coerce extra_special_tokens, so __init__ must accept **kwargs."""
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    if not _has_var_keyword(PreTrainedTokenizerBase.__init__):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/misc.py:97 patched "
            "PreTrainedTokenizerBase.__init__(self, **kwargs) but installed "
            f"signature has no VAR_KEYWORD: "
            f"{inspect.signature(PreTrainedTokenizerBase.__init__)}"
        )


def test_misc_masking_utils_create_block_mask_available_or_compile_flag():
    """misc.py imports BlockMask and create_block_mask and rewrites masking_utils create_causal_mask,
    create_sliding_window_causal_mask and create_masks_for_generate."""
    masking = importlib.import_module("transformers.masking_utils")
    for name in ("create_causal_mask", "create_sliding_window_causal_mask",
                 "create_masks_for_generate"):
        if not hasattr(masking, name):
            pytest.fail(
                f"DRIFT DETECTED: zoo temporary_patches/misc.py:414-445 "
                f"reads transformers.masking_utils.{name} but it is missing "
                f"on transformers {_TX_VERSION}"
            )


def test_misc_masking_utils_padding_and_packed_helpers():
    """misc.py wraps masking_utils padding_mask_function and packed_sequence_mask_function, each only if hasattr finds it, so they must be callable when present."""
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
    """misc.py patches transformers.integrations.sdpa_attention.sdpa_attention_forward."""
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


def test_misc_modernbert_model_update_attention_mask_present():
    """misc.py patches ModernBertModel._update_attention_mask (SDPA-stride fix); removed on 5.x where masks moved to
    masking-utils, so the test skips there and is strict on 4.57.6."""
    _skip_if_transformers_5x(
        "ModernBertModel._update_attention_mask removed (mask construction "
        "moved into the central masking-utils path)"
    )
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
    """misc.py patches CsmForConditionalGeneration._merge_input_ids_with_input_values(input_ids, input_values, input_values_cutoffs, labels)."""
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
    """misc.py patches Lfm2VlMultiModalProjector.__init__ and forward; zoo gates the patch to transformers before 5.0.0."""
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
    """misc.py patches peft.tuners.lora.bnb.dispatch_bnb_4bit(target, adapter_name)."""
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
    """misc.py reads MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES (5.0+) or legacy MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES; at least one must exist."""
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
    """misc.py sets kwargs['return_dict']=False when tokenize=True, so apply_chat_template must accept return_dict or **kwargs."""
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


# gpt_oss.py additional patch sites (beyond existing tests).

def test_gpt_oss_mxfp4_quantizer_class_present():
    """gpt_oss.py patches transformers.quantizers.quantizer_mxfp4.Mxfp4HfQuantizer.is_trainable."""
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


def test_gpt_oss_modeling_module_top_level_classes_present():
    """gpt_oss.py reassigns modeling_gpt_oss.GptOssExperts and GptOssTopKRouter for the BnB 4-bit GPT-OSS shim."""
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
    """gpt_oss.py reads GptOssConfig from transformers.models.gpt_oss.configuration_gpt_oss for layer type validation."""
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


def test_gpt_oss_attention_apply_rotary_pos_emb_imported_at_attention():
    """gpt_oss.py imports apply_rotary_pos_emb(q, k, cos, sin) (4 positionals) from modeling_gpt_oss."""
    mod = importlib.import_module(
        "transformers.models.gpt_oss.modeling_gpt_oss"
    )
    apply = getattr(mod, "apply_rotary_pos_emb", None)
    if apply is None:
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/gpt_oss.py:1875 expects "
            "modeling_gpt_oss.apply_rotary_pos_emb but it is missing"
        )
    # With kernels installed (transformers needs it for gpt-oss MXFP4), apply_rotary_pos_emb is a Func nn.Module
    # forwarding (*args, **kwargs); probing it reports 0 positionals, which is false drift.
    resolved = _resolve_for_arity(apply, "apply_rotary_pos_emb")
    if _is_passthrough(resolved):
        count = _positional_arity_from_source(mod, "apply_rotary_pos_emb")
        if count is None:
            pytest.skip(
                f"installed apply_rotary_pos_emb is a {type(apply).__name__} "
                f"passthrough {inspect.signature(resolved)} whose target could "
                "not be resolved, and the module source carries no def to read, "
                "so positional arity says nothing about drift"
            )
        if count < 4:
            pytest.fail(
                f"DRIFT DETECTED: zoo temporary_patches/gpt_oss.py calls "
                f"apply_rotary_pos_emb(q, k, cos, sin) -- 4 positionals -- but "
                f"{mod.__name__} defines it with {count}"
            )
        return
    params = [
        p for p in inspect.signature(resolved).parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                      inspect.Parameter.POSITIONAL_ONLY)
    ]
    if len(params) < 4:
        pytest.fail(
            f"DRIFT DETECTED: zoo temporary_patches/gpt_oss.py calls "
            f"apply_rotary_pos_emb(q, k, cos, sin) -- 4 positionals -- but "
            f"installed signature accepts only {len(params)}: "
            f"{inspect.signature(resolved)}"
        )


def test_gpt_oss_eager_attention_forward_present():
    """gpt_oss.py calls eager_attention_forward(self, q, k, v, mask, ...) with at least 5 positionals."""
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


# Cache-output-class signature pins (constructed by zoo patch wrappers).

def test_modeling_outputs_causal_lm_output_with_past_kwargs():
    """deepseek_v3_moe.py and qwen3_next_moe.py construct transformers.modeling_outputs.CausalLMOutputWithPast(loss,
    logits, past_key_values, hidden_states, attentions)."""
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
    """qwen3_moe.py constructs MoeCausalLMOutputWithPast via top-level modeling_outputs; aux_loss and router_logits must also be fields."""
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


def test_hybrid_cache_class_present():
    """gemma.py's isinstance(past_key_values, HybridCache) must exist on 4.57.6; on 5.x zoo's utils.py sets
    HAS_HYBRID_CACHE False and gemma.py gates the isinstance on it."""
    cu = importlib.import_module("transformers.cache_utils")
    from unsloth_zoo.temporary_patches.utils import HAS_HYBRID_CACHE
    if _TX_IS_5X:
        if hasattr(cu, "HybridCache"):
            # If transformers 5.x reintroduces HybridCache, zoo's HAS_HYBRID_CACHE probe must be refreshed so the gemma isinstance mask shortcut fires again.
            assert HAS_HYBRID_CACHE is True, (
                "transformers.cache_utils.HybridCache exists on "
                f"transformers {_TX_VERSION} but zoo's HAS_HYBRID_CACHE "
                "is False; refresh the utils.py probe."
            )
        else:
            assert HAS_HYBRID_CACHE is False, (
                "transformers.cache_utils.HybridCache missing on "
                f"transformers {_TX_VERSION} but zoo's HAS_HYBRID_CACHE "
                "is True; gemma.py:260 isinstance will raise TypeError."
            )
        return
    if not hasattr(cu, "HybridCache"):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/gemma.py:260 uses "
            "transformers.cache_utils.HybridCache but it is missing on "
            f"transformers {_TX_VERSION}"
        )
    assert HAS_HYBRID_CACHE is True, (
        "transformers.cache_utils.HybridCache exists on transformers "
        f"{_TX_VERSION} but zoo's HAS_HYBRID_CACHE flag is False; refresh "
        "the utils.py probe so gemma.py isinstance fires."
    )


def test_bitsandbytes_linear4bit_init_signature():
    """bitsandbytes.nn.modules.Linear4bit.__init__(input_features, output_features, ...) is needed because zoo's patched
    forward dereferences self.weight and self.bias."""
    bnb = _bitsandbytes_or_skip()
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


def test_gemma3n_text_config_class_present():
    """gemma3n.py's AltUp.predict reads self.config.altup_num_inputs and self.config.altup_active_idx."""
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


def test_gemma3_eager_attention_forward_kwargs_supported():
    """gemma.py calls eager_attention_forward(..., dropout, scaling, sliding_window, **kwargs), so it must accept **kwargs."""
    from transformers.models.gemma3.modeling_gemma3 import eager_attention_forward
    if not _has_var_keyword(eager_attention_forward):
        pytest.fail(
            "DRIFT DETECTED: zoo temporary_patches/gemma.py:412 calls "
            "eager_attention_forward(..., **kwargs) but installed signature "
            f"on transformers {_TX_VERSION} has no VAR_KEYWORD: "
            f"{inspect.signature(eager_attention_forward)}"
        )


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


def test_a_broken_dependency_is_not_reported_as_upstream_drift():
    """A module gone upstream is drift and must return None; a module that raises on its own dependency (timm.data
    ImageNetInfo for gemma3n) must skip, not report drift."""
    assert _try_get_class("transformers.models.not_a_real_model_xyz", "Whatever") is None

    module_name = "_zoo_probe_broken_import"
    module = types.ModuleType(module_name)

    def _explode():
        raise ImportError("cannot import name 'ImageNetInfo' from 'timm.data'")

    module.__getattr__ = lambda name: _explode()
    real_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == module_name:
            raise ImportError("cannot import name 'ImageNetInfo' from 'timm.data'")
        return real_import(name, *args, **kwargs)

    importlib.import_module = fake_import
    try:
        # Skipped derives from BaseException, so pytest.raises(Exception) would let it through and silently skip this test.
        with pytest.raises(_pytest_outcomes.Skipped) as caught:
            _try_get_class(module_name, "Anything")
        assert "nothing can be said about upstream drift" in str(caught.value)
    finally:
        importlib.import_module = real_import
