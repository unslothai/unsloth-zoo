# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The ArraysCache.advance patch, on any platform.

Everything that decides *whether* to rewrite a third-party class -- the body
matcher, the install latch, the retry policy, the deferred arithmetic and the
adoption of caches built before installation -- is plain Python. None of it needs
mlx, Metal or Apple Silicon, and the companion Metal suite is opt-in behind a
label, so this is the coverage that runs on every pull request.

The Metal-only assertions (live buffer counts, real decoding) stay in
tests/test_mlx_generate_metal.py.
"""

import importlib.util
import sys
import types

import pytest

if importlib.util.find_spec("mlx") is None:
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()

import unsloth_zoo.mlx.generate as engine  # noqa: E402


def stock_bodied_cache():
    """A class carrying mlx-lm 0.31.3's advance, byte for byte."""

    class ArraysCache:
        lengths = left_padding = None

        def advance(self, N):
            if self.lengths is not None:
                self.lengths -= N
            if self.left_padding is not None:
                self.left_padding -= N

    return ArraysCache


@pytest.fixture
def fresh(monkeypatch):
    """Un-latch the installer and let a test aim it at a class it owns."""

    monkeypatch.setattr(engine, "_ARRAYS_CACHE_ADVANCE_RESOLVED", False)
    monkeypatch.setattr(engine, "_VLM_ARRAYS_CACHE_ADVANCE_RESOLVED", False)

    def aim(cls, module_name="mlx_lm.models.cache"):
        module = types.ModuleType(module_name)
        module.ArraysCache = cls
        monkeypatch.setitem(sys.modules, module_name, module)
        return cls

    return aim




def test_matcher_accepts_the_body_it_reproduces():
    assert engine._has_replaceable_advance(stock_bodied_cache())


def test_matcher_rejects_the_opposite_sign():
    class Incrementing:
        lengths = left_padding = None

        def advance(self, N):
            if self.lengths is not None:
                self.lengths += N
            if self.left_padding is not None:
                self.left_padding += N

    assert not engine._has_replaceable_advance(Incrementing)


def test_matcher_rejects_the_same_decrements_plus_a_side_effect():
    class SideEffecting:
        lengths = left_padding = None

        def advance(self, N):
            self.touched = True
            if self.lengths is not None:
                self.lengths -= N
            if self.left_padding is not None:
                self.left_padding -= N

    assert not engine._has_replaceable_advance(SideEffecting)


def test_matcher_rejects_a_defaulted_step():
    class Defaulted:
        lengths = left_padding = None

        def advance(self, N=1):
            if self.lengths is not None:
                self.lengths -= N
            if self.left_padding is not None:
                self.left_padding -= N

    assert not engine._has_replaceable_advance(Defaulted)


def test_matcher_rejects_reads_already_mediated_by_descriptors():
    class Descriptored:
        lengths = property(lambda self: None, lambda self, v: None)
        left_padding = property(lambda self: None, lambda self, v: None)

        def advance(self, N):
            if self.lengths is not None:
                self.lengths -= N
            if self.left_padding is not None:
                self.left_padding -= N

    assert not engine._has_replaceable_advance(Descriptored)


def test_matcher_rejects_a_staticmethod_carrying_the_same_body():
    # A plain read unwraps staticmethod, comparing a body bound differently from ours.
    cls = stock_bodied_cache()
    cls.advance = staticmethod(cls.advance)
    assert not engine._has_replaceable_advance(cls)




def test_install_replaces_a_matching_class(fresh):
    cls = fresh(stock_bodied_cache())
    engine._install_arrays_cache_advance_fix()

    assert isinstance(cls.__dict__["lengths"], property)
    assert isinstance(cls.__dict__["left_padding"], property)
    assert cls.advance is engine._deferred_advance
    assert cls._unsloth_advance_patched is True
    assert engine._ARRAYS_CACHE_ADVANCE_RESOLVED


def test_install_leaves_a_non_matching_class_untouched(fresh):
    class Clamping:
        lengths = left_padding = None

        def advance(self, N):
            self.lengths = max(0, (self.lengths or 0) - N)

    before = dict(vars(Clamping))
    fresh(Clamping)
    engine._install_arrays_cache_advance_fix()

    assert Clamping.advance is before["advance"]
    assert not hasattr(Clamping, "_unsloth_advance_patched")
    assert not isinstance(vars(Clamping).get("lengths"), property)
    # Declining is still a decision, so it is not retried on every call.
    assert engine._ARRAYS_CACHE_ADVANCE_RESOLVED


def test_install_is_idempotent(fresh, monkeypatch):
    cls = fresh(stock_bodied_cache())
    engine._install_arrays_cache_advance_fix()
    installed = (cls.__dict__["lengths"], cls.__dict__["left_padding"], cls.advance)

    monkeypatch.setattr(engine, "_ARRAYS_CACHE_ADVANCE_RESOLVED", False)
    engine._install_arrays_cache_advance_fix()

    assert (cls.__dict__["lengths"], cls.__dict__["left_padding"], cls.advance) == installed


def test_a_failed_attempt_is_not_a_decision(fresh, monkeypatch):
    cls = fresh(stock_bodied_cache())
    real_matcher = engine._has_replaceable_advance

    def explode(arrays_cache):
        raise RuntimeError("transient")

    monkeypatch.setattr(engine, "_has_replaceable_advance", explode)
    engine._install_arrays_cache_advance_fix()
    assert not engine._ARRAYS_CACHE_ADVANCE_RESOLVED

    # Not undo(): it reverts the injected module too, so the retry could not tell
    # "declined to latch" from "nothing to look at".
    monkeypatch.setattr(engine, "_has_replaceable_advance", real_matcher)
    engine._install_arrays_cache_advance_fix()
    assert engine._ARRAYS_CACHE_ADVANCE_RESOLVED
    assert cls._unsloth_advance_patched is True


def test_a_missing_mlx_lm_is_inert(fresh, monkeypatch):
    # Every non-Apple platform: mlx-lm cannot be imported and nothing may raise.
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", None)
    engine._install_arrays_cache_advance_fix()
    assert not engine._ARRAYS_CACHE_ADVANCE_RESOLVED


def test_mlx_vlm_vendored_cache_is_patched_too(fresh):
    # mlx-vlm vendors its own copy of this body from 0.6.4, which patching mlx-lm
    # alone never reaches.
    text_cache = fresh(stock_bodied_cache())
    vision_cache = fresh(stock_bodied_cache(), "mlx_vlm.models.cache")
    sys.modules.setdefault("mlx_vlm", types.ModuleType("mlx_vlm"))

    engine._install_arrays_cache_advance_fix()

    assert text_cache._unsloth_advance_patched is True
    assert vision_cache._unsloth_advance_patched is True
    assert vision_cache is not text_cache


def test_mlx_vlm_is_not_imported_by_a_text_only_run(fresh, monkeypatch):
    fresh(stock_bodied_cache())
    monkeypatch.delitem(sys.modules, "mlx_vlm", raising=False)
    engine._install_arrays_cache_advance_fix()
    assert "mlx_vlm" not in sys.modules




def test_advance_defers_and_folds_on_read(fresh):
    cls = fresh(stock_bodied_cache())
    engine._install_arrays_cache_advance_fix()

    cache = cls()
    cache.lengths, cache.left_padding = 10, 4
    for _ in range(5):
        cache.advance(2)

    assert cache._lengths_pending == 10
    assert cache.lengths == 0
    assert cache.left_padding == -6
    # Folding is a one-off: reading again must not subtract twice.
    assert (cache.lengths, cache.left_padding) == (0, -6)


def test_writing_a_field_supersedes_its_pending_count(fresh):
    cls = fresh(stock_bodied_cache())
    engine._install_arrays_cache_advance_fix()

    cache = cls()
    cache.lengths = 10
    cache.advance(3)
    cache.lengths = 7
    assert cache.lengths == 7


def test_advance_with_a_non_integer_step_keeps_stock_arithmetic(fresh):
    # A float cannot go in the counter, and deferring moves the error to a later reader.
    cls = fresh(stock_bodied_cache())
    engine._install_arrays_cache_advance_fix()

    cache = cls()
    cache.lengths, cache.left_padding = 10.0, 4.0
    cache.advance(2.5)

    assert cache._lengths_pending == 0
    assert cache.lengths == 7.5
    assert cache.left_padding == 1.5


def test_a_cache_built_before_installation_keeps_its_metadata(fresh):
    cls = fresh(stock_bodied_cache())
    legacy = cls()
    legacy.__dict__.update(lengths=9, left_padding=2)

    engine._install_arrays_cache_advance_fix()
    legacy.advance(1)

    assert (legacy.lengths, legacy.left_padding) == (8, 1)


def test_a_partly_installed_class_is_still_correct(fresh):
    # The class assignments are not atomic and readers do not take the lock.
    steps = [
        lambda c: setattr(c, "_lengths", None),
        lambda c: setattr(c, "_lengths_pending", 0),
        lambda c: setattr(c, "_left_padding", None),
        lambda c: setattr(c, "_left_padding_pending", 0),
        lambda c: setattr(c, "lengths", engine._deferred_metadata("lengths")),
        lambda c: setattr(c, "left_padding", engine._deferred_metadata("left_padding")),
        lambda c: setattr(c, "advance", engine._deferred_advance),
    ]
    for prefix in range(len(steps) + 1):
        cls = stock_bodied_cache()
        for apply in steps[:prefix]:
            apply(cls)
        cache = cls()
        cache.lengths, cache.left_padding = 8, 3
        cache.advance(1)
        assert (cache.lengths, cache.left_padding) == (7, 2), f"broken after {prefix} steps"




def test_generate_batch_installs_before_it_generates(monkeypatch):
    order = []
    real_install = engine._install_arrays_cache_advance_fix

    def record_install():
        order.append("install")
        return real_install()

    class Adapter:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, requests):
            order.append("generate")
            return []

    import contextlib

    monkeypatch.setattr(engine, "_install_arrays_cache_advance_fix", record_install)
    monkeypatch.setattr(engine, "_TextBatchAdapter", Adapter)
    monkeypatch.setattr(engine, "generation_mode", lambda model: contextlib.nullcontext())
    monkeypatch.setattr(
        engine, "_generation_cache_hygiene", lambda: contextlib.nullcontext()
    )

    engine.generate_batch(
        object(),
        object(),
        [engine.GenerationRequest(prompt="hi", max_tokens=1)],
        defaults=engine.GenerationDefaults(),
    )

    assert order == ["install", "generate"]
