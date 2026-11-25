# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""patch_torch_compile keeps a user's cache opt-outs, does not force the Redis-only remote autotune cache,
skips deprecated Inductor keys like it skips deprecated Dynamo ones, and leaves cuda.use_fast_math off."""

import pytest

torch = pytest.importorskip("torch")
import torch._inductor.config as inductor_config

from unsloth_zoo.patching_utils import patch_torch_compile
from unsloth_zoo.temporary_patches.common import get_torch_compile_options


class _DeprecatedEntry:
    deprecated = True

    def __init__(self, entry):
        self._entry = entry

    def __getattr__(self, name):
        return getattr(self._entry, name)


def _enter(request, patch):
    patch.__enter__()
    request.addfinalizer(lambda: patch.__exit__(None, None, None))


def test_cache_env_respects_user_and_skips_remote(monkeypatch):
    monkeypatch.setenv("TORCHINDUCTOR_FX_GRAPH_CACHE", "0")
    monkeypatch.setenv("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")
    monkeypatch.setenv("ENABLE_AOT_AUTOGRAD_CACHE", "0")
    monkeypatch.delenv("TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE", raising = False)
    patch_torch_compile(debug = False, O3 = False, ignore_errors = True)
    import os
    assert os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] == "0"
    assert os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] == "0"
    assert os.environ["ENABLE_AOT_AUTOGRAD_CACHE"] == "0"
    assert "TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE" not in os.environ


def test_cache_env_defaults_on(monkeypatch):
    for name in ("TORCHINDUCTOR_FX_GRAPH_CACHE", "TORCHINDUCTOR_AUTOGRAD_CACHE", "ENABLE_AOT_AUTOGRAD_CACHE"):
        monkeypatch.delenv(name, raising = False)
    patch_torch_compile(debug = False, O3 = False, ignore_errors = True)
    import os
    assert os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] == "1"
    assert os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] == "1"


def test_deprecated_inductor_key_is_not_written(monkeypatch, request):
    key = "efficient_conv_bn_eval_fx_passes"
    assert key in inductor_config._config
    _enter(request, inductor_config.patch({key: False}))
    monkeypatch.setitem(inductor_config._config, key, _DeprecatedEntry(inductor_config._config[key]))
    patch_torch_compile(debug = False, O3 = False, ignore_errors = True)
    assert getattr(inductor_config, key) is False


def test_cuda_fast_math_stays_off(request):
    if "cuda.use_fast_math" not in inductor_config._config:
        pytest.skip("no cuda.use_fast_math on this torch")
    try:
        _enter(request, inductor_config.patch({"cuda.use_fast_math": False}))
    except Exception as exc:
        pytest.skip(f"no cuda.use_fast_math on this torch: {exc!r}")
    patch_torch_compile(debug = False, O3 = False, ignore_errors = True)
    assert inductor_config.cuda.use_fast_math is False


def test_compile_options_drop_deprecated_group_fusion():
    assert "group_fusion" not in get_torch_compile_options(group_fusion = True)
