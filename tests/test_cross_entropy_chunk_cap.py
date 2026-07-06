"""Fused cross entropy chunk sizing must stay active on large-memory GPUs.

On a GPU with a very large free pool, half of that pool used to become the
per-chunk target, which rounds the chunk count down to a single chunk. That
materializes the full logits and their float32 upcast at once, so at long
sequence lengths the transient dominates peak memory. The target is capped so
chunking keeps working regardless of how much memory is free.
"""
import os
# Let the module import on CPU-only CI (device detection otherwise raises).
os.environ.setdefault("UNSLOTH_ALLOW_CPU", "1")

import importlib
import types

import pytest


def _load_module(monkeypatch, free_bytes):
    try:
        ce = importlib.import_module("unsloth_zoo.fused_losses.cross_entropy_loss")
    except ImportError as e:
        # A zoo-only checkout (no separate `unsloth` package installed) makes
        # `unsloth_zoo/__init__` raise before this module loads. Skip rather
        # than fail; where `unsloth` is present the full assertions still run.
        pytest.skip(f"unsloth_zoo import unavailable: {e}")
    # Force the CUDA path and a fixed, very large free pool.
    monkeypatch.setattr(ce, "DEVICE_TYPE", "cuda", raising=False)

    fake_cuda = types.SimpleNamespace(mem_get_info=lambda index=0: (free_bytes, free_bytes))
    monkeypatch.setattr(ce.torch, "cuda", fake_cuda, raising=False)
    # _get_chunk_multiplier is functools.cache'd; clear so the mock is honored.
    ce._get_chunk_multiplier.cache_clear()
    return ce


def test_chunk_count_stays_above_one_on_huge_gpu(monkeypatch):
    huge_free = 180 * 1024 ** 3  # 180 GiB, e.g. a B200
    ce = _load_module(monkeypatch, huge_free)

    # A realistic large-vocab, long-context step.
    vocab_size = 256_000
    bsz, qlen = 1, 32_768
    n_splits = ce.get_chunk_size(bsz, qlen, vocab_size)
    assert n_splits > 1, f"expected chunking to stay active, got {n_splits} chunks"


def test_cap_effective_in_4_to_8_gib_band(monkeypatch):
    # Full float32 logits between 4 and 8 GiB used to round down to a single
    # uncapped chunk (round(0.5) * 4 == 0 -> max(.., 1) == 1). vocab=65536,
    # qlen=32768 is exactly 8 GiB; the cap must split it.
    huge_free = 180 * 1024 ** 3
    ce = _load_module(monkeypatch, huge_free)
    vocab_size = 65_536
    bsz, qlen = 1, 32_768
    n_splits = ce.get_chunk_size(bsz, qlen, vocab_size)
    assert n_splits > 1, f"expected the 4-8 GiB band to chunk, got {n_splits}"
    # Every chunk must stay within the 4 GiB target.
    total_gib = bsz * qlen * vocab_size * 4 / 1024 ** 3
    assert total_gib / n_splits <= 4.0 + 1e-6, (total_gib, n_splits)


def test_small_logits_stay_single_chunk(monkeypatch):
    # Configs whose full logits already fit the target keep a single chunk.
    ce = _load_module(monkeypatch, 180 * 1024 ** 3)
    # 2 GiB of float32 logits (< 4 GiB cap).
    assert ce.get_chunk_size(1, 8_192, 65_536) == 1


def test_cap_bounds_target_independent_of_free(monkeypatch):
    # The multiplier for the auto (target_gb=None) path must not shrink as the
    # free pool grows past the cap: two very different huge pools give the same
    # multiplier once the cap dominates.
    ce = _load_module(monkeypatch, 120 * 1024 ** 3)
    m_120 = ce._get_chunk_multiplier(256_000)
    ce = _load_module(monkeypatch, 320 * 1024 ** 3)
    m_320 = ce._get_chunk_multiplier(256_000)
    assert m_120 == pytest.approx(m_320), (m_120, m_320)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
