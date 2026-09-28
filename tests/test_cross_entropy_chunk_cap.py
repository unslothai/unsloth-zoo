"""Fused cross entropy chunk sizing must stay active on large-memory GPUs.

On a GPU with a very large free pool, half of that pool used to become the
per-chunk target, which rounds the chunk count down to a single chunk. That
materializes the full logits and their float32 upcast at once, so at long
sequence lengths the transient dominates peak memory. The target is capped so
chunking keeps working regardless of how much memory is free.
"""
import importlib
import types

import pytest


@pytest.fixture(autouse = True)
def _allow_cpu_import(monkeypatch):
    """Let `_load_module` import on CPU-only CI, where device detection otherwise raises.

    A fixture rather than the module-scope `os.environ.setdefault` this used to be. That
    ran at COLLECTION, and pytest imports every selected module before running anything, so
    the flag was not scoped to this file: it stayed on for every test the xdist worker took
    afterwards. test_allow_cpu_import_driverless.py is about what the import does WITHOUT
    it, and was deciding that question against a flag this file turned on.
    """
    monkeypatch.setenv("UNSLOTH_ALLOW_CPU", "1")


def _modules():
    """The CE module and the one that owns the memory budget it defers to."""
    try:
        ce = importlib.import_module("unsloth_zoo.fused_losses.cross_entropy_loss")
        tiled = importlib.import_module("unsloth_zoo.tiled_mlp")
    except ImportError as e:
        # A zoo-only checkout (no `unsloth` installed) makes `unsloth_zoo/__init__` raise first.
        pytest.skip(f"unsloth_zoo import unavailable: {e}")
    return ce, tiled


def _load_module(monkeypatch, free_bytes):
    ce, tiled = _modules()
    # `_free_target_gb` calls `tiled_mlp._default_target_gb`, so the backend is mocked THERE.
    # One implementation of "how much memory may this chunk use" is the point; a copy in the
    # CE module that measured something else would be the bug this fixes, one module over.
    monkeypatch.setattr(tiled, "DEVICE_TYPE", "cuda", raising=False)
    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        mem_get_info=lambda index=0: (free_bytes, free_bytes),
    )
    monkeypatch.setattr(tiled.torch, "cuda", fake_cuda, raising=False)
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


def _no_device_pool(monkeypatch, device_type, host_gb = 64.0):
    """No device pool to ask, as on CPU, MLX or a torch built without CUDA.

    Only `mem_get_info` and `is_available` are replaced, not the whole `torch.cuda` namespace:
    `torch.manual_seed` reaches `torch.cuda._is_in_bad_fork`, so a stand-in namespace breaks
    seeding rather than the thing under test.
    """
    ce, tiled = _modules()
    monkeypatch.setattr(tiled, "DEVICE_TYPE", device_type, raising = False)

    def refuse(index = 0):
        raise AssertionError("Torch not compiled with CUDA enabled")

    for namespace in ("cuda", "xpu"):
        target = getattr(tiled.torch, namespace, None)
        if target is not None:
            monkeypatch.setattr(target, "mem_get_info", refuse, raising = False)
            monkeypatch.setattr(target, "is_available", lambda: False, raising = False)
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: types.SimpleNamespace(available = int(host_gb * 1024 ** 3)),
    )
    ce._get_chunk_multiplier.cache_clear()
    return ce


@pytest.mark.parametrize("device_type", ["cpu", "mlx", "cuda"])
def test_a_machine_with_no_gpu_still_gets_a_chunk_size(monkeypatch, device_type):
    """`mem_get_info` does not return a number on a torch built without CUDA, it raises.

    Reached through the ordinary forward, so a CPU or Apple Silicon run died inside a memory
    query rather than on anything it was computing. "cuda" is in the list because a device
    type naming a GPU on a torch that has none is the same situation.
    """
    ce = _no_device_pool(monkeypatch, device_type)
    assert ce._free_target_gb() == ce._CE_TARGET_GB_CAP
    assert ce.get_chunk_size(1, 8, 32_000) >= 1


def test_the_no_gpu_budget_matches_what_a_large_gpu_lands_on(monkeypatch):
    """Not a number invented for CPU: the same cap any device with 8 GiB free reaches."""
    ce = _load_module(monkeypatch, 180 * 1024 ** 3)
    with_gpu = ce._get_chunk_multiplier(256_000)
    ce = _no_device_pool(monkeypatch, "cpu", host_gb = 64.0)
    without = ce._get_chunk_multiplier(256_000)
    assert with_gpu == pytest.approx(without), (with_gpu, without)


@pytest.mark.parametrize("device_type", ["cpu", "mlx"])
def test_a_small_host_budgets_smaller_chunks_rather_than_assuming_the_cap(
    monkeypatch, device_type
):
    """The activations live in host RAM here, so a machine with little of it must chunk finer.

    Returning the GPU-oriented cap unconditionally lets a chunk's transient approach 4 GiB on
    a box that does not have it, on top of the model already resident, which is an OOM kill
    where smaller chunks would have fit.
    """
    ce = _no_device_pool(monkeypatch, device_type, host_gb = 2.0)
    assert ce._free_target_gb() == pytest.approx(1.0)  # half of 2 GiB available
    small_host = ce.get_chunk_size(1, 8_192, 256_000)

    ce = _no_device_pool(monkeypatch, device_type, host_gb = 64.0)
    big_host = ce.get_chunk_size(1, 8_192, 256_000)
    assert small_host > big_host, (small_host, big_host)


def test_a_host_with_plenty_still_respects_the_cap(monkeypatch):
    """Half of a large host RAM would overshoot the cap the GPU path is held to."""
    ce = _no_device_pool(monkeypatch, "cpu", host_gb = 512.0)
    assert ce._free_target_gb() == ce._CE_TARGET_GB_CAP


def test_a_gpu_that_answers_is_still_measured_not_capped_blindly(monkeypatch):
    """The fallback must not swallow the real query: a small pool still sizes smaller chunks."""
    ce = _load_module(monkeypatch, 2 * 1024 ** 3)  # 2 GiB free -> 1 GiB target, under the cap
    assert ce._free_target_gb() == pytest.approx(1.0)
    assert ce._free_target_gb() < ce._CE_TARGET_GB_CAP


def test_the_fused_loss_runs_on_cpu_and_matches_plain_cross_entropy(monkeypatch):
    """End to end past the memory query, because a number is not the point: the loss is.

    The kernel is ordinary torch, so it computes on CPU once nothing asks a GPU how much
    memory it has. Checked against `F.cross_entropy` over the same shift, so a fallback that
    let it run but produced a different loss would fail here rather than pass quietly.
    """
    torch = pytest.importorskip("torch")
    ce = _no_device_pool(monkeypatch, "cpu")

    torch.manual_seed(0)
    hidden_size, vocab_size, bsz, qlen = 16, 64, 2, 6
    hidden = torch.randn(bsz, qlen, hidden_size, dtype = torch.float32, requires_grad = True)
    weight = torch.randn(vocab_size, hidden_size, dtype = torch.float32, requires_grad = True)
    labels = torch.randint(0, vocab_size, (bsz, qlen))

    loss = ce.unsloth_fused_ce_loss(
        trainer = None,
        hidden_states = hidden,
        lm_head_weight = weight,
        lm_head_bias = None,
        labels = labels,
        n_items = None,
        shift_labels = True,
    )
    assert torch.isfinite(loss), loss
    loss.backward()
    assert torch.isfinite(hidden.grad).all() and torch.isfinite(weight.grad).all()

    logits = (hidden.detach() @ weight.detach().T).float()
    reference = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, vocab_size), labels[:, 1:].reshape(-1)
    )
    assert torch.allclose(loss.detach(), reference, atol = 1e-4), (loss, reference)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
