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

"""Drift detectors for unsloth/import_fixes.py workarounds, one test per fix_* / patch_*; an active pathology fails with DRIFT DETECTED, optional libs importorskip."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import re
import sys
from importlib.metadata import version as importlib_version

import pytest


# Mirrors Version() in import_fixes.py: strips dev/post/local suffixes so packaging.Version parses e.g. 0.0.33.post2 and 0.15.1+cu130.
from packaging.version import Version as _PkgVersion


def _safe_version(raw):
    raw_str = str(raw)
    base = raw_str.split("+", 1)[0]
    try:
        return _PkgVersion(base)
    except Exception:
        match = re.match(r"[0-9]+(?:\.[0-9]+)*", base)
        if not match:
            raise
        return _PkgVersion(match.group(0))




def test_protobuf_message_factory_get_prototype_or_get_message_class_present():
    """fix_message_factory_issue (import_fixes.py): healthy install has MessageFactory.GetPrototype OR module-level GetMessageClass."""
    mf = pytest.importorskip("google.protobuf.message_factory")
    has_mf_class = hasattr(mf, "MessageFactory")
    has_get_prototype = has_mf_class and hasattr(
        mf.MessageFactory, "GetPrototype"
    )
    has_get_message_class = hasattr(mf, "GetMessageClass")
    if not has_mf_class:
        pytest.fail(
            "DRIFT DETECTED: google.protobuf.message_factory.MessageFactory is "
            "missing entirely -- fix_message_factory_issue would inject a stub."
        )
    if not (has_get_prototype or has_get_message_class):
        pytest.fail(
            "DRIFT DETECTED: neither MessageFactory.GetPrototype nor "
            "module-level GetMessageClass is present; fix_message_factory_issue "
            "would inject the GetPrototype/GetMessageClass shim."
        )
    assert has_get_prototype or has_get_message_class




def test_datasets_version_not_in_broken_recursion_range():
    """patch_datasets: datasets 4.4.0-4.5.0 triggers _thread.RLock_recursion_count errors in the Arrow loader, so the installed version must be outside that range."""
    pytest.importorskip("datasets")
    ds_v = _safe_version(importlib_version("datasets"))
    lo = _PkgVersion("4.4.0")
    hi = _PkgVersion("4.5.0")
    assert not (lo <= ds_v <= hi), (
        f"datasets=={ds_v} lies in the 4.4.0-4.5.0 recursion-error "
        f"range that patch_datasets explicitly forbids. Downgrade to "
        f"datasets==4.3.0 or upgrade past 4.5.0."
    )




def test_trl_is_x_available_returns_bool_not_tuple():
    """fix_trl_vllm_ascend: transformers >=4.48 _is_package_available returns (bool, version); TRL caches that tuple as
    truthy and fires unconditional import vllm, so every is_*_available() must return a real bool."""
    pytest.importorskip("trl")
    try:
        import trl.import_utils as tiu
    except Exception as exc:
        pytest.skip(f"trl.import_utils not importable: {exc!r}")

    accessor_names = [
        n
        for n in dir(tiu)
        if n.startswith("is_")
        and n.endswith("_available")
        and callable(getattr(tiu, n, None))
    ]
    assert accessor_names, "trl.import_utils has no is_*_available accessors"

    bad = {}
    for name in accessor_names:
        accessor = getattr(tiu, name)
        try:
            sig = inspect.signature(accessor)
            required = [
                p
                for p in sig.parameters.values()
                if p.default is inspect.Parameter.empty
                and p.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
            if required:
                continue
            result = accessor()
        except Exception:
            continue
        if not isinstance(result, bool):
            bad[name] = (type(result).__name__, result)

    if bad:
        pytest.fail(
            "DRIFT DETECTED: fix_trl_vllm_ascend coerces these accessors "
            f"from tuple-cached values to bool: {bad}"
        )


def test_trl_cached_available_flags_are_not_tuples():
    """Same tuple pathology as fix_trl_vllm_ascend, checked on the module-level cached _*_available attrs where the drift actually lives."""
    pytest.importorskip("trl")
    try:
        import trl.import_utils as tiu
    except Exception as exc:
        pytest.skip(f"trl.import_utils not importable: {exc!r}")

    tuple_flags = {
        name: value
        for name, value in vars(tiu).items()
        if name.startswith("_")
        and name.endswith("_available")
        and isinstance(value, tuple)
    }
    if tuple_flags:
        pytest.fail(
            "DRIFT DETECTED: fix_trl_vllm_ascend needs to coerce these tuple-"
            f"cached flags to bool: {sorted(tuple_flags)}"
        )




def test_pretrained_model_enable_input_require_grads_uses_old_pattern():
    """patch_enable_input_require_grads: HF PR 41993 made it call get_input_embeddings() on every module, raising
    NotImplementedError on vision submodules; healthy if the code predates HF PR 41993 or the try/except
    NotImplementedError wrapper is installed."""
    pytest.importorskip("transformers")
    from transformers import PreTrainedModel

    try:
        src = inspect.getsource(PreTrainedModel.enable_input_require_grads)
    except Exception as exc:
        pytest.skip(f"could not getsource(enable_input_require_grads): {exc!r}")

    if "for module in self.modules()" not in src:
        return  # healthy: pre-HF#41993 shape
    if "NotImplementedError" in src:
        return  # healthy: unsloth's tolerant replacement is installed

    pytest.fail(
        "DRIFT DETECTED: PreTrainedModel.enable_input_require_grads now "
        "iterates self.modules() (post HF#41993) and has NOT been "
        "wrapped by patch_enable_input_require_grads; vision submodules "
        "(e.g. GLM V4.6's self.visual) will raise NotImplementedError "
        "from get_input_embeddings and crash the whole call."
    )


def test_transformers_torchcodec_available_flag_is_present():
    """disable_torchcodec_if_broken needs the pre-5.x _torchcodec_available flag or the 5.x is_torchcodec_available
    function; that is the patch site for missing FFmpeg."""
    tf_iu = pytest.importorskip("transformers.utils.import_utils")
    has_flag = hasattr(tf_iu, "_torchcodec_available")
    has_func = callable(getattr(tf_iu, "is_torchcodec_available", None))
    assert has_flag or has_func, (
        "transformers.utils.import_utils dropped both "
        "``_torchcodec_available`` (pre-5.x) AND "
        "``is_torchcodec_available`` (>=5.x); "
        "disable_torchcodec_if_broken can no longer disable a broken "
        "torchcodec install."
    )


def test_transformers_is_causal_conv1d_available_symbol_present():
    """_disable_transformers_causal_conv1d needs one of is_causal_conv1d_available, _causal_conv1d_available or _is_causal_conv1d_available to mask a broken binary."""
    tf_iu = pytest.importorskip("transformers.utils.import_utils")
    candidates = [
        "is_causal_conv1d_available",
        "_causal_conv1d_available",
        "_is_causal_conv1d_available",
    ]
    present = [name for name in candidates if hasattr(tf_iu, name)]
    if not present:
        pytest.fail(
            "DRIFT DETECTED: transformers.utils.import_utils dropped every "
            f"hook in {candidates}; _disable_transformers_causal_conv1d "
            "can no longer mask a broken causal_conv1d binary."
        )




def test_transformers_and_accelerate_is_wandb_available_callable():
    """disable_broken_wandb patches is_wandb_available in three modules (transformers.integrations.integration_utils,
    accelerate.utils.imports, accelerate.utils); all must exist."""
    pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    from transformers.integrations import integration_utils as tf_integration
    import accelerate.utils.imports as acc_imports
    import accelerate.utils as acc_utils

    assert callable(getattr(tf_integration, "is_wandb_available", None)), (
        "transformers.integrations.integration_utils.is_wandb_available "
        "was removed/renamed; disable_broken_wandb can no longer mask a "
        "broken wandb install for trl trainers."
    )
    assert callable(getattr(acc_imports, "is_wandb_available", None)), (
        "accelerate.utils.imports.is_wandb_available removed; "
        "disable_broken_wandb cannot patch the source module."
    )
    assert callable(getattr(acc_utils, "is_wandb_available", None)), (
        "accelerate.utils.is_wandb_available removed; "
        "disable_broken_wandb cannot patch the re-export namespace "
        "consulted by trl/trainer/callbacks.py."
    )




def test_peft_transformers_weight_conversion_importable_and_signature():
    """patch_peft_weight_converter_compatibility wraps build_peft_weight_mapping; module must import and keep
    (weight_conversions, adapter_name, peft_config=None); unimportable is drift, since the bare except in the fix
    silently no-ops"""
    pytest.importorskip("peft")
    try:
        from peft.utils import transformers_weight_conversion as twc
    except Exception as exc:
        pytest.fail(
            "DRIFT DETECTED: peft.utils.transformers_weight_conversion "
            f"is unimportable on this stack ({exc!r}). "
            "patch_peft_weight_converter_compatibility will silently no-op."
        )

    assert hasattr(twc, "build_peft_weight_mapping"), (
        "build_peft_weight_mapping vanished from "
        "peft.utils.transformers_weight_conversion."
    )
    sig = inspect.signature(twc.build_peft_weight_mapping)
    expected_params = {"weight_conversions", "adapter_name"}
    actual_params = set(sig.parameters)
    assert expected_params.issubset(actual_params), (
        f"build_peft_weight_mapping signature drifted: expected at "
        f"least {sorted(expected_params)}, got {sorted(actual_params)}."
    )




def test_triton_compiled_kernel_has_num_ctas_and_cluster_dims():
    """triton 3.6.0+ dropped num_ctas and cluster_dims from CompiledKernel, but torch 2.9.x Inductor still evaluates
    them eagerly, so torch.compile fails without the fix."""
    pytest.importorskip("torch")
    triton_mod = pytest.importorskip("triton")  # noqa: F841
    tc = pytest.importorskip("triton.compiler.compiler")

    ck_cls = tc.CompiledKernel
    # Healthy if the pre-3.6 class attr exists, or unsloth wrapped __init__ to install num_ctas and cluster_dims per instance.
    if hasattr(ck_cls, "num_ctas"):
        return
    init = getattr(ck_cls, "__init__", None)
    if init is not None:
        code = getattr(init, "__code__", None)
        freevars = set(getattr(code, "co_freevars", ()) or ())
        co_names = set(getattr(code, "co_names", ()) or ())
        if "_orig_init" in freevars or {"num_ctas", "cluster_dims"}.issubset(
            co_names
        ):
            return

    pytest.fail(
        "DRIFT DETECTED: triton.CompiledKernel lacks the `num_ctas` "
        "class attribute and ``__init__`` has not been wrapped by "
        "fix_triton_compiled_kernel_missing_attrs; torch Inductor's "
        "``make_launcher`` will crash on the eager "
        "``binary.metadata.num_ctas, *binary.metadata.cluster_dims`` "
        "unpack under torch.compile."
    )


# torch + torchvision pairing table


# Mirrors TORCH_TORCHVISION_COMPAT in import_fixes.py torchvision_compatibility_check.
_TORCH_TORCHVISION_COMPAT = {
    (2, 9): (0, 24),
    (2, 8): (0, 23),
    (2, 7): (0, 22),
    (2, 6): (0, 21),
    (2, 5): (0, 20),
    (2, 4): (0, 19),
}


def _is_custom_torch_build(raw_version_str):
    """Same logic as import_fixes._is_custom_torch_build (673-689)."""
    if "+" not in raw_version_str:
        return False
    local = raw_version_str.split("+", 1)[1]
    if not local:
        return False
    return not re.fullmatch(
        r"cu\d[\d.]*|rocm\d[\d.]*|cpu|xpu", local, re.IGNORECASE
    )


def test_installed_torch_torchvision_pair_is_compatible():
    """Mirrors the upstream torch/torchvision compat table; custom and prerelease builds are skipped, only warned about in torchvision_compatibility_check."""
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    torch_raw = importlib_version("torch")
    tv_raw = importlib_version("torchvision")
    torch_v = _safe_version(torch_raw)
    tv_v = _safe_version(tv_raw)

    torch_major = torch_v.release[0]
    torch_minor = torch_v.release[1] if len(torch_v.release) > 1 else 0

    required = _TORCH_TORCHVISION_COMPAT.get((torch_major, torch_minor))
    if required is None:
        pytest.skip(
            f"torch=={torch_raw} is outside the pinned compatibility "
            f"table (entries cover 2.4-2.9). The formula fallback "
            f"in _infer_required_torchvision handles it at runtime."
        )

    pre_tags = (".dev", "a0", "b0", "rc", "alpha", "beta", "nightly")
    is_prerelease = any(t in torch_raw for t in pre_tags) or any(
        t in tv_raw for t in pre_tags
    )
    is_custom = _is_custom_torch_build(torch_raw) or _is_custom_torch_build(
        tv_raw
    )
    if is_prerelease or is_custom:
        pytest.skip(
            f"torch=={torch_raw} torchvision=={tv_raw} is a custom/"
            f"prerelease build; the runtime check downgrades to warning."
        )

    required_str = f"{required[0]}.{required[1]}.0"
    assert tv_v >= _PkgVersion(required_str), (
        f"DRIFT DETECTED: torch=={torch_raw} requires "
        f"torchvision>={required_str}, but torchvision=={tv_raw} is "
        f"installed. torchvision_compatibility_check would raise."
    )




def test_vllm_guided_decoding_params_or_structured_outputs_present():
    """vLLM PR 22772 renamed GuidedDecodingParams to StructuredOutputsParams; trl still imports the old name, so the fix re-aliases it and at least one must exist."""
    pytest.importorskip("vllm")
    try:
        sp = importlib.import_module("vllm.sampling_params")
    except Exception as exc:
        pytest.skip(f"vllm.sampling_params unimportable: {exc!r}")

    has_guided = hasattr(sp, "GuidedDecodingParams")
    has_structured = hasattr(sp, "StructuredOutputsParams")
    assert has_guided or has_structured, (
        "vllm.sampling_params has neither GuidedDecodingParams nor "
        "StructuredOutputsParams; fix_vllm_guided_decoding_params "
        "cannot re-alias. trl import path will break."
    )
    if not has_guided:
        pytest.fail(
            "DRIFT DETECTED: vllm.sampling_params only exposes "
            "StructuredOutputsParams (post PR #22772); "
            "fix_vllm_guided_decoding_params injects a GuidedDecodingParams "
            "alias so trl keeps importing."
        )


def test_vllm_aimv2_ovis_config_is_past_fix_version():
    """vLLM < 0.10.1's Ovis config unconditionally calls AutoConfig.register('aimv2', ...), raising ValueError when Transformers already uses 'aimv2'."""
    pytest.importorskip("vllm")
    vllm_v = _safe_version(importlib_version("vllm"))
    cutoff = _PkgVersion("0.10.1")
    if vllm_v < cutoff:
        pytest.fail(
            f"DRIFT DETECTED: vllm=={vllm_v} < {cutoff}; "
            "fix_vllm_aimv2_issue rewrites ovis.py to skip the duplicate "
            'AutoConfig.register("aimv2", ...) call.'
        )




def test_huggingface_hub_is_offline_mode_or_hf_hub_offline_present():
    """huggingface_hub removed top-level is_offline_mode(); unsloth re-injects it from huggingface_hub.constants.HF_HUB_OFFLINE, so at least one must exist."""
    hub = pytest.importorskip("huggingface_hub")
    has_top_level = False
    try:
        has_top_level = callable(getattr(hub, "is_offline_mode", None))
    except Exception:
        # huggingface_hub may use __getattr__ that raises AttributeError.
        has_top_level = False

    has_constant = False
    try:
        constants_mod = importlib.import_module("huggingface_hub.constants")
        has_constant = hasattr(constants_mod, "HF_HUB_OFFLINE")
    except Exception:
        has_constant = False

    assert has_top_level or has_constant, (
        "huggingface_hub dropped both ``is_offline_mode`` AND "
        "``huggingface_hub.constants.HF_HUB_OFFLINE``; "
        "fix_huggingface_hub can no longer re-inject the helper."
    )




def test_torch_nn_init_trunc_normal_exists():
    """patch_trunc_normal_precision_issue: the fp16/bf16 wrapper monkey-patches torch.nn.init.trunc_normal_, and a rename silently fails the install."""
    pytest.importorskip("torch")
    import torch.nn.init as init_mod

    assert callable(getattr(init_mod, "trunc_normal_", None)), (
        "torch.nn.init.trunc_normal_ removed/renamed; "
        "patch_trunc_normal_precision_issue cannot wrap it."
    )




def test_xformers_is_post_num_splits_key_fix_or_not_installed():
    """xformers < 0.0.29 has the num_splits_key=-1 perf bug that unsloth rewrites at install time."""
    if importlib.util.find_spec("xformers") is None:
        pytest.skip("xformers not installed -- nothing to drift-check.")
    x_v = _safe_version(importlib_version("xformers"))
    cutoff = _PkgVersion("0.0.29")
    if x_v < cutoff:
        pytest.fail(
            f"DRIFT DETECTED: xformers=={x_v} < {cutoff}; "
            "fix_xformers_performance_issue rewrites "
            "ops/fmha/cutlass.py num_splits_key=-1 -> None."
        )


# transformers (PreTrainedModel base import sanity)


def test_transformers_pretrained_model_has_get_input_embeddings():
    """The enable_input_require_grads replacement calls get_input_embeddings() on every submodule, so renaming that method breaks the patch."""
    pytest.importorskip("transformers")
    from transformers import PreTrainedModel

    assert hasattr(PreTrainedModel, "get_input_embeddings"), (
        "PreTrainedModel.get_input_embeddings was renamed or removed; "
        "patch_enable_input_require_grads's replacement no longer compiles."
    )


# accelerate -- ``is_X_available`` API stability used across the fixes


def test_accelerate_utils_imports_module_present():
    """disable_broken_wandb and fix_trl_vllm_ascend both reach into accelerate.utils.imports; a restructuring there silently no-ops their monkey-patches."""
    pytest.importorskip("accelerate")
    mod = pytest.importorskip("accelerate.utils.imports")
    # is_wandb_available is the representative symbol disable_broken_wandb targets.
    assert hasattr(mod, "is_wandb_available"), (
        "accelerate.utils.imports.is_wandb_available is gone; "
        "disable_broken_wandb cannot patch the source module."
    )
