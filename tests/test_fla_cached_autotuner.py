# SPDX-License-Identifier: LGPL-3.0-or-later
"""CachedAutotuner launch-path regressions (issue unslothai/unsloth#10806)."""

import importlib.util
import pathlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]
_CACHE_PATH = _REPO / "unsloth_zoo/_vendored/fla/ops/utils/cache.py"


def _load_cache_module():
    if "fla_cache_under_test" in sys.modules:
        return sys.modules["fla_cache_under_test"]

    triton = ModuleType("triton")
    triton.__version__ = "3.6.0"
    triton.Config = lambda *a, **k: SimpleNamespace(
        pre_hook=None, all_kwargs=lambda: {},
    )
    autotuner = ModuleType("triton.runtime.autotuner")

    class Autotuner:
        def __init__(
            self, fn, arg_names, configs, key, reset_to_zero, restore_value, **kwargs,
        ):
            self.fn = fn
            self.arg_names = arg_names
            self.configs = configs
            self.keys = key
            self.cache = {}
            self.reset_to_zero = reset_to_zero or []
            self.restore_value = restore_value or []
            self.user_defined_pre_hook = True
            self.user_defined_post_hook = True

        def run(self, *args, **kwargs):
            return "parent"

    autotuner.Autotuner = Autotuner
    sys.modules.setdefault("triton", triton)
    sys.modules.setdefault("triton.runtime", ModuleType("triton.runtime"))
    sys.modules["triton.runtime.autotuner"] = autotuner
    sys.modules.setdefault("torch", MagicMock())

    spec = importlib.util.spec_from_file_location("fla_cache_under_test", _CACHE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fla_cache_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


_cache = _load_cache_module()
AutotuneKey = _cache.AutotuneKey
CachedAutotuner = _cache.CachedAutotuner
FlaCacheMode = _cache.FlaCacheMode


class _ReuseBestCache(dict):
    _unsloth_reuse_best = True

    def __contains__(self, key):
        return len(self) > 0 or super().__contains__(key)

    def __getitem__(self, key):
        if not super().__contains__(key) and len(self) > 0:
            return next(iter(self.values()))
        return super().__getitem__(key)


def _make_autotuner(cache, configs=(MagicMock(), MagicMock())):
    fn = MagicMock()
    fn.run = MagicMock(return_value="ok")
    fn.arg_names = ["x"]
    fn.__name__ = "test_kernel"
    inner = SimpleNamespace(fn=fn, __name__="test_kernel")
    tuner = CachedAutotuner(
        inner,
        ["x"],
        list(configs),
        ["x"],
        [],
        [],
    )
    tuner.cache = cache
    tuner.fn = fn
    return tuner, fn


def test_runtime_autotune_tuple_matches_autotune_key_build():
    built = AutotuneKey.build(["x"], ["x"], (3,), {})
    tup = CachedAutotuner._runtime_autotune_tuple(["x"], ["x"], (3,), {})
    assert built.autotune_key == tup


def test_run_skips_autotune_key_when_fla_cache_disabled(monkeypatch):
    monkeypatch.setattr(_cache, "FLA_CACHE_MODE", FlaCacheMode.DISABLED)
    cache = {}
    tuner, _fn = _make_autotuner(cache)
    with patch.object(AutotuneKey, "build") as build:
        with patch(
            "triton.runtime.autotuner.Autotuner.run",
            return_value="parent",
        ) as parent_run:
            assert tuner.run(1) == "parent"
            build.assert_not_called()
            parent_run.assert_called_once()


def test_reuse_best_cache_fast_path_skips_parent_run():
    cfg = MagicMock()
    cfg.pre_hook = None
    cfg.all_kwargs.return_value = {}
    cache = _ReuseBestCache({(1,): cfg})
    tuner, fn = _make_autotuner(cache)
    with patch("triton.runtime.autotuner.Autotuner.run") as parent_run:
        assert tuner.run(1) == "ok"
        parent_run.assert_not_called()
        fn.run.assert_called_once()
