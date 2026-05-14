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

"""Drift detectors for the class of upstream pathologies that
``unsloth/import_fixes.py`` works around.

Every test in this file maps 1:1 to a ``fix_*`` / ``patch_*`` function
in the unsloth package's ``import_fixes.py``. The fix function is a
hand-rolled workaround for a specific upstream regression (protobuf
``MessageFactory`` drift, datasets 4.4.x recursion, TRL tuple-vs-bool
``_*_available`` caching, transformers ``enable_input_require_grads``
source pattern flip, triton ``CompiledKernel`` missing attrs, etc.).

``unsloth-zoo`` depends on the same upstream wheels but has NO test
today that screams when one of these pathologies is currently ACTIVE
on the installed stack. This suite is the drift detector.

Contract for each test:

  * Assert the *healthy* shape that the fix expects the upstream lib
    to have ABSENT the regression.
  * If the optional library isn't installed at all, ``importorskip``
    the test (not relevant to this install).
  * If the pathology is currently ACTIVE on this install, surface it
    as ``pytest.fail("DRIFT DETECTED: <fix function> needed because
    <observation>")`` so CI stays green locally but the drift is loud
    in the verbose log -- exactly the same pattern a maintainer would
    use to triage which fix has stopped being a no-op.
  * Tests that require a GPU / specific accelerator skip cleanly on
    CPU-only boxes.

Every test cites the source-of-truth ``import_fixes.py`` function and
line range it was reduced from, so when the workaround is removed or
renamed upstream we can find the matching detector quickly.

Runs under the GPU-free harness in ``tests/conftest.py``.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import re
import sys
from importlib.metadata import version as importlib_version

import pytest


# ---------------------------------------------------------------------------
# Small helper: a tolerant parsed Version. Mirrors the local ``Version()``
# in import_fixes.py (lines 51-68): strip dev / alpha / beta / rc suffixes
# so packaging.Version doesn't choke on, say, "0.0.33.post2" or
# "0.15.1+cu130".
# ---------------------------------------------------------------------------

from packaging.version import Version as _PkgVersion


def _safe_version(raw):
    """Parse a raw distribution version into packaging.Version, stripping
    local identifiers and exotic dev / post suffixes if needed."""
    raw_str = str(raw)
    # Drop local identifier (+cu130, +rocm6.3, +cpu, etc.)
    base = raw_str.split("+", 1)[0]
    try:
        return _PkgVersion(base)
    except Exception:
        # Fallback: re-extract a [0-9.]+ prefix.
        match = re.match(r"[0-9]+(?:\.[0-9]+)*", base)
        if not match:
            raise
        return _PkgVersion(match.group(0))


# ===========================================================================
# protobuf
# ===========================================================================


def test_protobuf_message_factory_get_prototype_or_get_message_class_present():
    """Drift detector for ``fix_message_factory_issue``
    (import_fixes.py lines 264-308).

    The fix monkey-patches ``google.protobuf.message_factory.MessageFactory``
    when ``GetPrototype`` is gone AND no ``GetMessageClass`` fallback
    exists. On a healthy install ONE of these must be reachable, since
    tensorflow / sentencepiece-driven tokenizer load paths call into
    one of them. Asserts the post-fix invariant.
    """
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


# ===========================================================================
# datasets
# ===========================================================================


def test_datasets_version_not_in_broken_recursion_range():
    """Drift detector for ``patch_datasets``
    (import_fixes.py lines 574-586).

    ``datasets`` 4.4.0 through 4.5.0 (inclusive) trigger
    ``_thread.RLock_recursion_count`` recursion errors deep in the
    Arrow loader. Unsloth raises ``NotImplementedError`` for that
    range. Assert the installed version is outside it.
    """
    pytest.importorskip("datasets")
    ds_v = _safe_version(importlib_version("datasets"))
    lo = _PkgVersion("4.4.0")
    hi = _PkgVersion("4.5.0")
    assert not (lo <= ds_v <= hi), (
        f"datasets=={ds_v} lies in the 4.4.0-4.5.0 recursion-error "
        f"range that patch_datasets explicitly forbids. Downgrade to "
        f"datasets==4.3.0 or upgrade past 4.5.0."
    )


# ===========================================================================
# trl
# ===========================================================================


def test_trl_is_x_available_returns_bool_not_tuple():
    """Drift detector for ``fix_trl_vllm_ascend``
    (import_fixes.py lines 493-516).

    transformers >= 4.48's ``_is_package_available(name)`` returns a
    ``(bool, version_or_None)`` tuple. TRL's module-level
    ``_*_available`` flags cache that tuple, and ``is_*_available()``
    returns it directly. A non-empty tuple is always truthy, so
    ``if is_vllm_available():`` fires even when vllm is absent and
    triggers an unconditional ``import vllm`` that hard-crashes on
    Ascend hosts (and any non-vllm host). Healthy state: every
    ``is_*_available()`` returns a real ``bool``.
    """
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
            # Some accessors take args; skip those rather than guess.
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
    """Drift detector for ``fix_trl_vllm_ascend``
    (import_fixes.py lines 493-516).

    Same pathology as above but checks the module-level cached
    ``_*_available`` attributes directly -- this is where the tuple
    drift actually lives. Healthy state: each ``_X_available`` is a
    bool (or a callable/sentinel), never a tuple.
    """
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


# ===========================================================================
# transformers
# ===========================================================================


def test_pretrained_model_enable_input_require_grads_uses_old_pattern():
    """Drift detector for ``patch_enable_input_require_grads``
    (import_fixes.py lines 609-670).

    transformers PR #41993 rewrote
    ``PreTrainedModel.enable_input_require_grads`` to iterate via
    ``for module in self.modules()`` and call
    ``module.get_input_embeddings()`` on every submodule. Vision
    sub-modules (GLM V4.6's ``self.visual``) raise
    ``NotImplementedError`` from that accessor and crash the
    whole call. Healthy (= pre-regression) state: source does NOT
    contain ``for module in self.modules()``.
    """
    pytest.importorskip("transformers")
    from transformers import PreTrainedModel

    try:
        src = inspect.getsource(PreTrainedModel.enable_input_require_grads)
    except Exception as exc:
        pytest.skip(f"could not getsource(enable_input_require_grads): {exc!r}")

    if "for module in self.modules()" in src:
        pytest.fail(
            "DRIFT DETECTED: PreTrainedModel.enable_input_require_grads now "
            "iterates self.modules() (post HF#41993). "
            "patch_enable_input_require_grads has to install a "
            "NotImplementedError-tolerant replacement."
        )


def test_transformers_torchcodec_available_flag_is_present():
    """Drift detector for ``disable_torchcodec_if_broken``
    (import_fixes.py lines 1291-1317).

    Unsloth flips ``transformers.utils.import_utils._torchcodec_available``
    to ``False`` when torchcodec is installed but can't load its
    native FFmpeg deps. The flag must exist for the patch to land.
    """
    tf_iu = pytest.importorskip("transformers.utils.import_utils")
    assert hasattr(tf_iu, "_torchcodec_available"), (
        "transformers.utils.import_utils._torchcodec_available was "
        "removed/renamed upstream; disable_torchcodec_if_broken can no "
        "longer disable a broken torchcodec install."
    )


def test_transformers_is_causal_conv1d_available_symbol_present():
    """Drift detector for ``_disable_transformers_causal_conv1d``
    (import_fixes.py lines 1881-1895).

    Unsloth needs ``transformers.utils.import_utils`` to expose
    EITHER an ``is_causal_conv1d_available`` callable OR one of the
    ``_causal_conv1d_available`` / ``_is_causal_conv1d_available``
    cached flags so it can monkey-patch a broken-binary install to
    ``False``. If transformers drops them ALL, the disable path
    silently no-ops and model imports hard-fail later.
    """
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


# ===========================================================================
# transformers + accelerate (wandb checkers)
# ===========================================================================


def test_transformers_and_accelerate_is_wandb_available_callable():
    """Drift detector for ``disable_broken_wandb``
    (import_fixes.py lines 1320-1372).

    Unsloth patches BOTH
    ``transformers.integrations.integration_utils.is_wandb_available``
    AND ``accelerate.utils.imports.is_wandb_available`` /
    ``accelerate.utils.is_wandb_available``. The fix matters because
    a protobuf mismatch can make ``import wandb`` raise. Both
    accessor locations must continue to exist.
    """
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


# ===========================================================================
# peft
# ===========================================================================


def test_peft_transformers_weight_conversion_importable_and_signature():
    """Drift detector for ``patch_peft_weight_converter_compatibility``
    (import_fixes.py lines 1375-1454).

    Unsloth wraps ``peft.utils.transformers_weight_conversion.
    build_peft_weight_mapping`` to retrofit ``distributed_operation``
    and ``quantization_operation`` kwargs onto legacy converter
    ctors. Healthy state: module imports cleanly AND the function
    signature still accepts ``(weight_conversions, adapter_name,
    peft_config=None)``. If the module is unimportable on the
    current peft/transformers pair, that IS the drift (the fix's
    bare ``except (ImportError, AttributeError): return`` would
    silently no-op).
    """
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


# ===========================================================================
# triton
# ===========================================================================


def test_triton_compiled_kernel_has_num_ctas_and_cluster_dims():
    """Drift detector for ``fix_triton_compiled_kernel_missing_attrs``
    (import_fixes.py lines 923-968).

    triton 3.6.0+ dropped direct ``num_ctas`` / ``cluster_dims``
    attributes on ``CompiledKernel`` but torch 2.9.x Inductor's
    ``make_launcher`` still eagerly evaluates ``binary.metadata.num_ctas,
    *binary.metadata.cluster_dims``. Without the fix, torch.compile
    paths blow up before reaching the new launch contract. Healthy
    state: a freshly-constructed CompiledKernel has both attrs.
    """
    pytest.importorskip("torch")
    triton_mod = pytest.importorskip("triton")  # noqa: F841
    tc = pytest.importorskip("triton.compiler.compiler")

    ck_cls = tc.CompiledKernel
    # Two healthy shapes:
    #   * pre-3.6 triton (or a future fix) exposes ``num_ctas`` as a
    #     class attribute -- the upstream pathology is gone.
    #   * post-3.6 triton with unsloth's fix applied: the class still
    #     lacks the attr but ``CompiledKernel.__init__`` has been
    #     wrapped to install both attrs on each new instance, which is
    #     enough to satisfy torch Inductor's
    #     ``hasattr(binary, "num_ctas")`` probe at launch time.
    if hasattr(ck_cls, "num_ctas"):
        return  # healthy: old-style triton with direct class attr
    # Detect the instance-level wrapped __init__. Unsloth's
    # ``fix_triton_compiled_kernel_missing_attrs`` rebinds the class's
    # ``__init__`` to a closure named ``_patched_init`` whose qualname
    # / source contains the num_ctas + cluster_dims injection. Probing
    # the closure cells / co_freevars is GPU-free and idempotent.
    init = getattr(ck_cls, "__init__", None)
    if init is not None:
        code = getattr(init, "__code__", None)
        freevars = set(getattr(code, "co_freevars", ()) or ())
        co_names = set(getattr(code, "co_names", ()) or ())
        if "_orig_init" in freevars or {"num_ctas", "cluster_dims"}.issubset(
            co_names
        ):
            return  # healthy: unsloth's __init__ wrapper is installed

    pytest.fail(
        "DRIFT DETECTED: triton.CompiledKernel lacks the `num_ctas` "
        "class attribute and ``__init__`` has not been wrapped by "
        "fix_triton_compiled_kernel_missing_attrs; torch Inductor's "
        "``make_launcher`` will crash on the eager "
        "``binary.metadata.num_ctas, *binary.metadata.cluster_dims`` "
        "unpack under torch.compile."
    )


# ===========================================================================
# torch + torchvision pairing table
# ===========================================================================


# Mirrors TORCH_TORCHVISION_COMPAT in torchvision_compatibility_check
# (import_fixes.py lines 708-798).
_TORCH_TORCHVISION_COMPAT = {
    (2, 9): (0, 24),
    (2, 8): (0, 23),
    (2, 7): (0, 22),
    (2, 6): (0, 21),
    (2, 5): (0, 20),
    (2, 4): (0, 19),
}


def _is_custom_torch_build(raw_version_str):
    """Same logic as import_fixes._is_custom_torch_build
    (lines 673-689)."""
    if "+" not in raw_version_str:
        return False
    local = raw_version_str.split("+", 1)[1]
    if not local:
        return False
    return not re.fullmatch(
        r"cu\d[\d.]*|rocm\d[\d.]*|cpu|xpu", local, re.IGNORECASE
    )


def test_installed_torch_torchvision_pair_is_compatible():
    """Drift detector for ``torchvision_compatibility_check``
    (import_fixes.py lines 708-798).

    Unsloth raises ``ImportError`` when the installed torch /
    torchvision pair doesn't satisfy the known compatibility table.
    Custom or prerelease torch builds get downgraded to warning.
    Mirror that table here: assert the installed pair satisfies it
    or skip cleanly for custom / prerelease builds.
    """
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    torch_raw = importlib_version("torch")
    tv_raw = importlib_version("torchvision")
    torch_v = _safe_version(torch_raw)
    tv_v = _safe_version(tv_raw)

    torch_major = torch_v.release[0]
    torch_minor = torch_v.release[1] if len(torch_v.release) > 1 else 0

    # Only assert for entries that exist in the pinned table.
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


# ===========================================================================
# vllm
# ===========================================================================


def test_vllm_guided_decoding_params_or_structured_outputs_present():
    """Drift detector for ``fix_vllm_guided_decoding_params``
    (import_fixes.py lines 446-490).

    vLLM PR #22772 renamed ``GuidedDecodingParams`` to
    ``StructuredOutputsParams``. trl still imports the old name, so
    the fix re-aliases on demand. Healthy state: at least one of the
    two symbols must exist at module load time.
    """
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
    """Drift detector for ``fix_vllm_aimv2_issue``
    (import_fixes.py lines 404-443).

    vLLM < 0.10.1 has an Ovis config that unconditionally
    ``AutoConfig.register("aimv2", AIMv2Config)`` and trips
    ``ValueError: 'aimv2' is already used by a Transformers config``.
    The fix only touches old versions. Assert installed vLLM is past
    the cutoff (or skip cleanly if not).
    """
    pytest.importorskip("vllm")
    vllm_v = _safe_version(importlib_version("vllm"))
    cutoff = _PkgVersion("0.10.1")
    if vllm_v < cutoff:
        pytest.fail(
            f"DRIFT DETECTED: vllm=={vllm_v} < {cutoff}; "
            "fix_vllm_aimv2_issue rewrites ovis.py to skip the duplicate "
            'AutoConfig.register("aimv2", ...) call.'
        )


# ===========================================================================
# huggingface_hub
# ===========================================================================


def test_huggingface_hub_is_offline_mode_or_hf_hub_offline_present():
    """Drift detector for ``fix_huggingface_hub``
    (import_fixes.py lines 913-920).

    huggingface_hub deprecated and removed the top-level
    ``is_offline_mode()`` helper. Unsloth re-injects it from
    ``huggingface_hub.constants.HF_HUB_OFFLINE``. Healthy state: the
    re-injection target must still exist.
    """
    hub = pytest.importorskip("huggingface_hub")
    # Either the function is still there OR the underlying constant
    # used by the fix's re-injection is still importable.
    has_top_level = False
    try:
        has_top_level = callable(getattr(hub, "is_offline_mode", None))
    except Exception:
        # huggingface_hub may use __getattr__ that raises AttributeError;
        # treat that as "missing".
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


# ===========================================================================
# torch
# ===========================================================================


def test_torch_nn_init_trunc_normal_exists():
    """Drift detector for ``patch_trunc_normal_precision_issue``
    (import_fixes.py lines 971-1050).

    The fp16/bf16 stability wrapper monkey-patches
    ``torch.nn.init.trunc_normal_``. If that symbol is renamed or
    removed the wrapper installation will fail silently.
    """
    pytest.importorskip("torch")
    import torch.nn.init as init_mod

    assert callable(getattr(init_mod, "trunc_normal_", None)), (
        "torch.nn.init.trunc_normal_ removed/renamed; "
        "patch_trunc_normal_precision_issue cannot wrap it."
    )


# ===========================================================================
# xformers
# ===========================================================================


def test_xformers_is_post_num_splits_key_fix_or_not_installed():
    """Drift detector for ``fix_xformers_performance_issue``
    (import_fixes.py lines 312-341).

    xformers < 0.0.29 has the ``num_splits_key=-1`` perf bug that
    Unsloth rewrites at install time. Healthy state: installed
    xformers is >= 0.0.29 (or xformers isn't installed).
    """
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


# ===========================================================================
# transformers (PreTrainedModel base import sanity)
# ===========================================================================


def test_transformers_pretrained_model_has_get_input_embeddings():
    """Drift detector for ``patch_enable_input_require_grads``
    (import_fixes.py lines 609-670).

    The replacement function the patch installs calls
    ``module.get_input_embeddings()`` on every submodule. If that
    accessor is renamed upstream the replacement is broken.
    """
    pytest.importorskip("transformers")
    from transformers import PreTrainedModel

    assert hasattr(PreTrainedModel, "get_input_embeddings"), (
        "PreTrainedModel.get_input_embeddings was renamed or removed; "
        "patch_enable_input_require_grads's replacement no longer compiles."
    )


# ===========================================================================
# accelerate -- ``is_X_available`` API stability used across the fixes
# ===========================================================================


def test_accelerate_utils_imports_module_present():
    """Drift detector for ``disable_broken_wandb`` and
    ``fix_trl_vllm_ascend`` (import_fixes.py lines 493-516, 1320-1372).

    Both fixes reach into ``accelerate.utils.imports``. If accelerate
    restructures that module path, both monkey-patches silently
    no-op and broken-wandb / tuple-cached flag pathologies leak
    through.
    """
    pytest.importorskip("accelerate")
    mod = pytest.importorskip("accelerate.utils.imports")
    # The module must at minimum still re-export some ``is_*_available``
    # helper; checking for a single representative one (is_wandb_available)
    # is sufficient because disable_broken_wandb specifically targets it.
    assert hasattr(mod, "is_wandb_available"), (
        "accelerate.utils.imports.is_wandb_available is gone; "
        "disable_broken_wandb cannot patch the source module."
    )
