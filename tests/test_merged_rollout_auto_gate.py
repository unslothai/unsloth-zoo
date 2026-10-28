# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Decision table for the automatic default of UNSLOTH_VLLM_MERGED_ROLLOUT (CPU only).

Unset / "auto": on for one synchronous 16-bit LoRA text engine on NVIDIA CUDA under Linux with the
exact Triton fold and vLLM >= 0.11.2. "0" is always off, "1" forces it on every engine as before.
"""

import types

import pytest
import torch

import unsloth_zoo.vllm_utils as vllm_utils


_PLATFORM_GATE = vllm_utils._merged_rollout_auto_platform


@pytest.fixture(autouse = True)
def _fresh_platform_cache():
    # The real gate is cached per process; clear it around every test (tests may patch the name).
    _PLATFORM_GATE.cache_clear()
    yield
    _PLATFORM_GATE.cache_clear()


def _platform(monkeypatch, *, platform = "linux", release = "6.8.0-generic", hip = None, cuda = True,
              triton = True, vllm = "0.19.1"):
    monkeypatch.setattr(vllm_utils.sys, "platform", platform)
    # raising = False: native Windows has no os.uname.
    monkeypatch.setattr(vllm_utils.os, "uname", lambda: types.SimpleNamespace(release = release), raising = False)
    monkeypatch.setattr(vllm_utils.torch.version, "hip", hip)
    monkeypatch.setattr(vllm_utils.torch.cuda, "is_available", lambda: cuda)
    import unsloth_zoo.merged_rollout_exact as exact
    monkeypatch.setattr(exact, "triton", object() if triton else None)
    monkeypatch.setattr(vllm_utils, "vllm_version", vllm, raising = False)
    _PLATFORM_GATE.cache_clear()
    return _PLATFORM_GATE()


@pytest.mark.parametrize("kwargs, reason", [
    ({}, None),
    ({"vllm": "0.11.2"}, None),
    ({"vllm": "0.32.0"}, None),
    ({"vllm": "0.11.1"}, "older than 0.11.2"),
    ({"vllm": "0.10.2"}, "older than 0.11.2"),
    ({"hip": "7.2.0"}, "ROCm"),
    ({"cuda": False}, "CUDA is not available"),
    ({"platform": "win32"}, "platform is win32"),
    ({"platform": "darwin"}, "platform is darwin"),
    ({"release": "5.15.167.4-microsoft-standard-WSL2"}, "WSL"),
    ({"triton": False}, "Triton"),
    ({"vllm": "not-a-version"}, "unknown"),
])
def test_platform_gate(monkeypatch, kwargs, reason):
    got = _platform(monkeypatch, **kwargs)
    if reason is None:
        assert got is None
    else:
        assert got is not None and reason in got


@pytest.mark.parametrize("value, platform_ok, expected", [
    (None, True, "all"), ("", True, "all"), ("auto", True, "all"), ("AUTO", True, "all"),
    (None, False, None), ("auto", False, None),
    ("0", True, None), ("0", False, None), ("off", True, None), ("false", True, None),
    ("1", True, "all"), ("1", False, "all"),
    ("attention", True, "attention"), ("attention", False, "attention"),
])
def test_mode_table(monkeypatch, value, platform_ok, expected):
    monkeypatch.setattr(vllm_utils, "_merged_rollout_auto_platform", lambda: None if platform_ok else "this is ROCm")
    if value is None:
        monkeypatch.delenv("UNSLOTH_VLLM_MERGED_ROLLOUT", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_VLLM_MERGED_ROLLOUT", value)
    assert vllm_utils._merged_rollout_mode() == expected
    assert vllm_utils._merged_rollout_enabled() is (expected is not None)


_OK = dict(enable_lora = True, use_async = False, use_engine = False, use_bitsandbytes = False, is_fp8 = False,
           engine_args = {"tensor_parallel_size": 1}, is_vision_model = False)


@pytest.mark.parametrize("change, ok", [
    ({}, True),
    ({"enable_lora": False}, False),
    ({"use_async": True}, False),
    ({"use_engine": True}, False),
    ({"use_bitsandbytes": True}, False),
    ({"is_fp8": True}, False),
    ({"is_vision_model": True}, False),
    ({"engine_args": {"quantization": "bitsandbytes"}}, False),
    ({"engine_args": {"quantization": "fp8"}}, False),
    ({"engine_args": {"quantization": "torchao"}}, False),
    ({"engine_args": {"tensor_parallel_size": 2}}, False),
    ({"engine_args": {"pipeline_parallel_size": 2}}, False),
    ({"engine_args": {}}, True),
])
def test_engine_gate_automatic(monkeypatch, change, ok):
    monkeypatch.delenv("UNSLOTH_VLLM_MERGED_ROLLOUT", raising = False)
    assert vllm_utils._merged_rollout_auto_engine_ok(**{**_OK, **change}) is ok


@pytest.mark.parametrize("value", ["1", "attention"])
def test_engine_gate_forced_wraps_every_engine(monkeypatch, value):
    # Forced modes keep the pre-default behaviour: every engine is wrapped, prepare decides per model.
    monkeypatch.setenv("UNSLOTH_VLLM_MERGED_ROLLOUT", value)
    for change in ({"use_bitsandbytes": True}, {"is_vision_model": True}, {"engine_args": {"tensor_parallel_size": 2}}):
        assert vllm_utils._merged_rollout_auto_engine_ok(**{**_OK, **change}) is True


def test_stats_counters_exist():
    assert set(vllm_utils._MERGED_ROLLOUT_STATS) == {"folds", "fallbacks"}
