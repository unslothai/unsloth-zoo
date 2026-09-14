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

_STUB_MODULE_KEYS = (
    "torch",
    "triton",
    "triton.runtime",
    "triton.runtime.autotuner",
    "fla_cache_under_test",
)


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
            self.user_defined_pre_hook = kwargs.get("pre_hook") is not None
            self.user_defined_post_hook = kwargs.get("post_hook") is not None
            self.pre_hook = kwargs.get("pre_hook") or (lambda kw, reset_only=False: None)
            self.post_hook = kwargs.get("post_hook") or (lambda kw, exception: None)
            self.restore_copies = {}

        def run(self, *args, **kwargs):
            return "parent"

    autotuner.Autotuner = Autotuner
    sys.modules["triton"] = triton
    sys.modules["triton.runtime"] = ModuleType("triton.runtime")
    sys.modules["triton.runtime.autotuner"] = autotuner
    sys.modules["torch"] = MagicMock()

    spec = importlib.util.spec_from_file_location("fla_cache_under_test", _CACHE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fla_cache_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cache_mod():
    saved = {k: sys.modules.get(k) for k in _STUB_MODULE_KEYS}
    try:
        yield _load_cache_module()
    finally:
        for key, previous in saved.items():
            if previous is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = previous


class _ReuseBestCache(dict):
    _unsloth_reuse_best = True

    def __contains__(self, key):
        return len(self) > 0 or super().__contains__(key)

    def __getitem__(self, key):
        if not super().__contains__(key) and len(self) > 0:
            return next(iter(self.values()))
        return super().__getitem__(key)


def _make_autotuner(cache_mod, cache, configs=(MagicMock(), MagicMock())):
    CachedAutotuner = cache_mod.CachedAutotuner
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


def test_triton_runtime_autotune_key_matches_autotune_key_build(cache_mod):
    AutotuneKey = cache_mod.AutotuneKey
    built = AutotuneKey.build(["x"], ["x"], (3,), {})
    tup = cache_mod.triton_runtime_autotune_key(["x"], ["x"], (3,), {})
    assert built.autotune_key == tup


def test_run_skips_autotune_key_when_fla_cache_disabled(cache_mod, monkeypatch):
    monkeypatch.setattr(cache_mod, "FLA_CACHE_MODE", cache_mod.FlaCacheMode.DISABLED)
    cache = {}
    tuner, _fn = _make_autotuner(cache_mod, cache)
    with patch.object(cache_mod.AutotuneKey, "build") as build:
        with patch(
            "triton.runtime.autotuner.Autotuner.run",
            return_value="parent",
        ) as parent_run:
            assert tuner.run(1) == "parent"
            build.assert_not_called()
            parent_run.assert_called_once()


def test_reuse_best_cache_fast_path_skips_parent_run(cache_mod):
    cfg = MagicMock()
    cfg.pre_hook = None
    cfg.all_kwargs.return_value = {}
    cache = _ReuseBestCache({(1,): cfg})
    tuner, fn = _make_autotuner(cache_mod, cache)
    with patch("triton.runtime.autotuner.Autotuner.run") as parent_run:
        assert tuner.run(1) == "ok"
        parent_run.assert_not_called()
        fn.run.assert_called_once()


def test_reuse_best_fast_path_disabled_for_fla_cache_always(cache_mod, monkeypatch):
    monkeypatch.setattr(cache_mod, "FLA_CACHE_MODE", cache_mod.FlaCacheMode.ALWAYS)
    cfg = MagicMock()
    cfg.pre_hook = None
    cfg.all_kwargs.return_value = {}
    cache = _ReuseBestCache({(1,): cfg})
    tuner, fn = _make_autotuner(cache_mod, cache)
    with patch.object(tuner, "maybe_load_cached_config") as load_cfg:
        with patch("triton.runtime.autotuner.Autotuner.run", return_value="parent") as parent_run:
            assert tuner.run(1) == "parent"
            parent_run.assert_called_once()
            fn.run.assert_not_called()
            load_cfg.assert_called_once()


def test_reuse_best_fast_path_runs_autotuner_pre_hook(cache_mod):
    cfg = MagicMock()
    cfg.pre_hook = None
    cfg.all_kwargs.return_value = {}
    cache = _ReuseBestCache({(1,): cfg})
    buf = MagicMock()
    tuner, fn = _make_autotuner(cache_mod, cache)
    tuner.reset_to_zero = ["x"]
    pre = MagicMock()
    tuner.pre_hook = pre
    tuner.post_hook = MagicMock()
    with patch("triton.runtime.autotuner.Autotuner.run") as parent_run:
        assert tuner.run(buf) == "ok"
        parent_run.assert_not_called()
        pre.assert_called_once()
        fn.run.assert_called_once()


def test_reuse_best_fast_path_zeros_reset_to_zero_from_init(cache_mod):
    cfg = MagicMock()
    cfg.pre_hook = None
    cfg.all_kwargs.return_value = {}
    cache = _ReuseBestCache({(1,): cfg})
    fn = MagicMock()
    fn.configure_mock(run=MagicMock(return_value="ok"), arg_names=["x"], __name__="test_kernel")
    inner = SimpleNamespace(fn=fn, __name__="test_kernel")
    buf = MagicMock()
    tuner = cache_mod.CachedAutotuner(
        inner,
        ["x"],
        [MagicMock(), MagicMock()],
        ["x"],
        ["x"],
        [],
    )
    tuner.cache = cache
    tuner.fn = fn
    with patch("triton.runtime.autotuner.Autotuner.run"):
        tuner.run(buf)
    buf.zero_.assert_called_once()
