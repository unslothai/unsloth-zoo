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
    ce = importlib.import_module("unsloth_zoo.fused_losses.cross_entropy_loss")
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
