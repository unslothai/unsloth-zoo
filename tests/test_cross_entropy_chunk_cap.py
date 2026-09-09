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
        # A zoo-only checkout (no `unsloth` installed) makes `unsloth_zoo/__init__` raise first.
        pytest.skip(f"unsloth_zoo import unavailable: {e}")
    monkeypatch.setattr(ce, "DEVICE_TYPE", "cuda", raising=False)

    fake_cuda = types.SimpleNamespace(mem_get_info=lambda index=0: (free_bytes, free_bytes))
    monkeypatch.setattr(ce.torch, "cuda", fake_cuda, raising=False)
    # _get_chunk_multiplier is functools.cache'd; clear so the mock is honored.
    ce._get_chunk_multiplier.cache_clear()
    return ce


def test_chunk_count_stays_above_one_on_huge_gpu(monkeypatch):
    huge_free = 180 * 1024 ** 3  # 180 GiB, e.g. a B200
    ce = _load_module(monkeypatch, huge_free)

    vocab_size = 256_000
    bsz, qlen = 1, 32_768
    n_splits = ce.get_chunk_size(bsz, qlen, vocab_size)
    assert n_splits > 1, f"expected chunking to stay active, got {n_splits} chunks"


def test_cap_effective_in_4_to_8_gib_band(monkeypatch):
    # 4-8 GiB of float32 logits used to round down to one uncapped chunk
    # (round(0.5) * 4 == 0 -> max(.., 1) == 1); 65536 x 32768 is exactly 8 GiB.
    huge_free = 180 * 1024 ** 3
    ce = _load_module(monkeypatch, huge_free)
    vocab_size = 65_536
    bsz, qlen = 1, 32_768
    n_splits = ce.get_chunk_size(bsz, qlen, vocab_size)
    assert n_splits > 1, f"expected the 4-8 GiB band to chunk, got {n_splits}"
    total_gib = bsz * qlen * vocab_size * 4 / 1024 ** 3
    assert total_gib / n_splits <= 4.0 + 1e-6, (total_gib, n_splits)


def test_small_logits_stay_single_chunk(monkeypatch):
    ce = _load_module(monkeypatch, 180 * 1024 ** 3)
    # 2 GiB footprint at 16 bytes/element (< 4 GiB cap).
    assert ce.get_chunk_size(1, 2_048, 65_536) == 1


def test_bytes_per_logit_covers_the_whole_chain(monkeypatch):
    # Below the eager cost of 14 bytes/element chunks overrun target_gb (#946).
    ce = _load_module(monkeypatch, 180 * 1024 ** 3)
    assert ce._CE_BYTES_PER_LOGIT >= 14.0


def test_unchunkable_memory_is_charged_to_the_target(monkeypatch):
    ce = _load_module(monkeypatch, 180 * 1024 ** 3)
    none = ce.get_chunk_size(1, 16_384, 151_936, target_gb=4.0, fixed_gb=0.0)
    some = ce.get_chunk_size(1, 16_384, 151_936, target_gb=4.0, fixed_gb=2.0)
    assert some > none, (none, some)


def test_fixed_larger_than_target_does_not_explode_chunks(monkeypatch):
    # No chunk count can satisfy it, so sizing falls back to the plain budget.
    ce = _load_module(monkeypatch, 180 * 1024 ** 3)
    plain = ce.get_chunk_size(1, 16_384, 151_936, target_gb=1.0, fixed_gb=0.0)
    swamped = ce.get_chunk_size(1, 16_384, 151_936, target_gb=1.0, fixed_gb=8.0)
    assert swamped == plain, (plain, swamped)
    assert swamped <= 16_384


def test_overwrite_does_not_charge_the_aliased_grad_inputs(monkeypatch):
    # Under overwrite grad_inputs IS hidden_states, so it costs no new memory.
    ce = _load_module(monkeypatch, 180 * 1024 ** 3)
    aliased = ce.get_chunk_size(1, 8_192, 151_936, target_gb=4.0, fixed_gb=0.0)
    charged = ce.get_chunk_size(1, 8_192, 151_936, target_gb=4.0, fixed_gb=2.0)
    assert charged > aliased, (aliased, charged)


def test_chunks_never_exceed_token_count(monkeypatch):
    ce = _load_module(monkeypatch, 180 * 1024 ** 3)
    n = ce.get_chunk_size(1, 64, 262_144, target_gb=0.001)
    assert 1 <= n <= 64, n


def test_cap_bounds_target_independent_of_free(monkeypatch):
    # The auto (target_gb=None) multiplier must not shrink as the free pool grows past the cap.
    ce = _load_module(monkeypatch, 120 * 1024 ** 3)
    m_120 = ce._get_chunk_multiplier(256_000)
    ce = _load_module(monkeypatch, 320 * 1024 ** 3)
    m_320 = ce._get_chunk_multiplier(256_000)
    assert m_120 == pytest.approx(m_320), (m_120, m_320)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
