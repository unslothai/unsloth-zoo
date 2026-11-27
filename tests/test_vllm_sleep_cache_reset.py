"""Resets precede gc.collect/sleep; missing methods no-op, errors swallowed, patch idempotent."""

from __future__ import annotations

import types

import pytest


@pytest.fixture(scope="module")
def vllm_utils():
    try:
        import unsloth_zoo.vllm_utils as m
    except Exception as e:  # no GPU / accelerator on this host
        pytest.skip(f"unsloth_zoo.vllm_utils unavailable: {e}")
    return m


def _make_fake_vllm(calls, *, has_mm=True, has_encoder=True,
                    has_engine=True, mm_raises=False):
    """Builds a fake vllm module exposing an LLM class that logs every call to calls; sleep stands in for vLLM's real LLM.sleep (the wrap target)."""

    class Engine:
        pass

    if has_encoder:
        def reset_encoder_cache(self):
            calls.append("reset_encoder_cache")
        Engine.reset_encoder_cache = reset_encoder_cache

    class LLM:
        def __init__(self):
            if has_engine:
                self.llm_engine = Engine()

        def sleep(self, *args, **kwargs):
            calls.append("orig_sleep")

    if has_mm:
        def reset_mm_cache(self):
            calls.append("reset_mm_cache")
            if mm_raises:
                raise RuntimeError("boom from reset_mm_cache")
        LLM.reset_mm_cache = reset_mm_cache

    return types.SimpleNamespace(LLM=LLM)


def _install(vllm_utils, monkeypatch, fake, calls):
    """Point vllm_utils at the fake vllm and spy gc.collect, then patch."""
    monkeypatch.setattr(vllm_utils, "vllm", fake, raising=False)
    monkeypatch.setattr(vllm_utils.gc, "collect",
                        lambda *a, **k: calls.append("gc_collect"))
    vllm_utils.patch_vllm_reset_caches_on_sleep()


def test_resets_fire_before_sleep_in_order(vllm_utils, monkeypatch):
    calls = []
    fake = _make_fake_vllm(calls)
    _install(vllm_utils, monkeypatch, fake, calls)

    fake.LLM().sleep()

    assert calls == ["reset_mm_cache", "reset_encoder_cache",
                     "gc_collect", "orig_sleep"]


def test_encoder_cache_absent_is_noop(vllm_utils, monkeypatch):
    # Mirrors vLLM 0.15.1: LLM.reset_mm_cache exists, reset_encoder_cache does not.
    calls = []
    fake = _make_fake_vllm(calls, has_encoder=False)
    _install(vllm_utils, monkeypatch, fake, calls)

    fake.LLM().sleep()

    assert calls == ["reset_mm_cache", "gc_collect", "orig_sleep"]


def test_text_model_no_caches_is_clean_noop(vllm_utils, monkeypatch):
    calls = []
    fake = _make_fake_vllm(calls, has_mm=False, has_encoder=False,
                           has_engine=False)
    _install(vllm_utils, monkeypatch, fake, calls)

    fake.LLM().sleep()  # must not raise

    assert calls == ["gc_collect", "orig_sleep"]


def test_raising_reset_is_swallowed_and_sleep_still_runs(vllm_utils, monkeypatch):
    calls = []
    fake = _make_fake_vllm(calls, mm_raises=True)
    _install(vllm_utils, monkeypatch, fake, calls)

    fake.LLM().sleep()  # the RuntimeError from reset_mm_cache must be swallowed

    assert calls == ["reset_mm_cache", "reset_encoder_cache",
                     "gc_collect", "orig_sleep"]


def test_double_patch_is_idempotent(vllm_utils, monkeypatch):
    calls = []
    fake = _make_fake_vllm(calls)
    _install(vllm_utils, monkeypatch, fake, calls)

    wrapped_once = fake.LLM.sleep
    assert getattr(fake.LLM, "_unsloth_reset_caches_on_sleep", False) is True

    vllm_utils.patch_vllm_reset_caches_on_sleep()
    assert fake.LLM.sleep is wrapped_once

    # And a single sleep still triggers exactly one reset cycle (no double-wrap).
    fake.LLM().sleep()
    assert calls == ["reset_mm_cache", "reset_encoder_cache",
                     "gc_collect", "orig_sleep"]


def test_nested_engine_sleep_collects_once(vllm_utils, monkeypatch):
    # LLM.sleep -> llm_engine.sleep, both wrapped: caches still reset at each level, but one full gc per sleep.
    calls = []

    class LLMEngine:
        def reset_encoder_cache(self):
            calls.append("reset_encoder_cache")

        def sleep(self, *args, **kwargs):
            calls.append("engine_sleep")

    class LLM:
        def __init__(self):
            self.llm_engine = LLMEngine()

        def reset_mm_cache(self):
            calls.append("reset_mm_cache")

        def sleep(self, *args, **kwargs):
            calls.append("orig_sleep")
            self.llm_engine.sleep(*args, **kwargs)

    fake = types.SimpleNamespace(LLM=LLM, LLMEngine=LLMEngine)
    _install(vllm_utils, monkeypatch, fake, calls)

    llm = LLM()
    llm.sleep(1)
    assert calls.count("gc_collect") == 1
    assert calls[:4] == ["reset_mm_cache", "reset_encoder_cache", "gc_collect", "orig_sleep"]
    assert calls[-2:] == ["reset_encoder_cache", "engine_sleep"]

    # The guard resets: a second sleep collects again, and a raising inner sleep does not leave it stuck.
    calls.clear()
    llm.sleep(1)
    assert calls.count("gc_collect") == 1

    def boom(self, *a, **k):
        raise RuntimeError("inner sleep failed")
    monkeypatch.setattr(LLMEngine, "sleep", boom)
    monkeypatch.setattr(LLMEngine, "_unsloth_reset_caches_on_sleep", False, raising=False)
    vllm_utils.patch_vllm_reset_caches_on_sleep()
    with pytest.raises(RuntimeError):
        llm.sleep(1)
    calls.clear()
    llm.llm_engine.sleep = lambda *a, **k: calls.append("engine_sleep")
    llm.sleep(1)
    assert calls.count("gc_collect") == 1
