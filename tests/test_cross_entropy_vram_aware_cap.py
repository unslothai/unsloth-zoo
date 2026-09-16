"""The fused-CE per-chunk budget scales with the card, without regressing small ones.

The cap used to be the constant 4.0 GiB. That is right on a 16 GiB card but does
not scale: on a 183 GiB B200 at vocab 248320 / 2048 tokens the whole transient is
~7.6 GiB, and the constant still forced 4 chunks (13.46 ms vs 8.84 ms at 1 chunk,
for 0.71 GiB saved on a card with ~175 GiB spare).

Setting UNSLOTH_CE_VRAM_AWARE_CAP=1 opts in to a cap that scales with TOTAL VRAM,
clamped to [4, 16] GiB. It is opt-in because it is not a universal win: the gain
comes mostly from the lm_head weight gradient, so an ordinary LoRA run with a
frozen lm_head pays more memory for much less speed, and on a 40 GiB A100 the
single-chunk landing measured slightly slower than the stock 4 chunks.

These tests pin the four properties that make it safe:
  1. the default is off, and off is the old budget bit-for-bit on every card,
  2. every card <= 24 GiB keeps the old budget bit-for-bit even when opted in,
  3. the cap stays bounded, so a huge card never stops chunking a huge transient,
  4. the live free-VRAM reading is not frozen by the multiplier memo.
"""
import importlib
import types

import pytest


@pytest.fixture(autouse = True)
def _allow_cpu_import(monkeypatch):
    monkeypatch.setenv("UNSLOTH_ALLOW_CPU", "1")


@pytest.fixture(autouse = True)
def _opt_in(monkeypatch):
    """The VRAM aware cap is opt-in, so every test here switches it on.

    test_default_is_off_and_matches_the_historical_constant below deliberately
    turns it back off to pin the default.
    """
    monkeypatch.setenv("UNSLOTH_CE_VRAM_AWARE_CAP", "1")


GIB = 1024 ** 3


@pytest.fixture
def ce():
    try:
        mod = importlib.import_module("unsloth_zoo.fused_losses.cross_entropy_loss")
    except ImportError as e:
        pytest.skip(f"unsloth_zoo import unavailable: {e}")
    mod._get_chunk_multiplier.cache_clear()
    yield mod
    mod._get_chunk_multiplier.cache_clear()


def _fake_device(monkeypatch, ce, free_bytes, total_bytes):
    monkeypatch.setattr(ce, "DEVICE_TYPE", "cuda", raising = False)
    monkeypatch.setattr(
        ce.torch, "cuda",
        types.SimpleNamespace(mem_get_info = lambda index = 0: (free_bytes, total_bytes)),
        raising = False,
    )
    ce._get_chunk_multiplier.cache_clear()


# --------------------------------------------------------------- the policy itself
@pytest.mark.parametrize("total_gib", [4, 8, 11, 15, 16, 22.5, 24])
def test_small_cards_keep_the_historical_constant_exactly(ce, total_gib):
    """<= 24 GiB must reproduce min(free * 0.5, 4.0) bit-for-bit."""
    total = int(total_gib * GIB)
    for frac in (0.1, 0.3, 0.5, 0.9, 1.0):
        free = int(total * frac)
        assert ce._auto_target_gb_from(free, total) == min(free / GIB * 0.5, 4.0)


@pytest.mark.parametrize(
    "total_gib, expected_cap",
    [(40, 40 / 6), (80, 80 / 6), (96, 16.0), (141, 16.0), (183, 16.0), (320, 16.0)],
)
def test_large_cards_scale_the_cap_and_then_saturate(ce, total_gib, expected_cap):
    total = int(total_gib * GIB)
    # Fully free, so the cap is what binds.
    got = ce._auto_target_gb_from(total, total)
    assert got == pytest.approx(expected_cap, rel = 1e-6)


def test_cap_is_bounded_so_huge_gpus_still_chunk(ce):
    """Property (2): the ceiling means an arbitrarily large transient still splits."""
    for total_gib in (183, 640, 4096):
        total = int(total_gib * GIB)
        assert ce._auto_target_gb_from(total, total) <= ce._CE_CAP_MAX_GB


def test_busy_large_card_still_chunks_aggressively(ce):
    """free * 0.5 is still the other half of the min()."""
    total = int(183 * GIB)
    assert ce._auto_target_gb_from(int(2 * GIB), total) == pytest.approx(1.0)


def test_cap_never_drops_below_the_historical_floor(ce):
    tiny = int(2 * GIB)
    assert ce._auto_target_gb_from(tiny, tiny) == pytest.approx(1.0)  # free half binds
    assert ce._auto_target_gb_from(int(100 * GIB), tiny) == pytest.approx(4.0)  # floor


# ------------------------------------------------------------------- chunk counts
def test_b200_picks_one_chunk_for_a_qwen_sized_head(ce, monkeypatch):
    _fake_device(monkeypatch, ce, int(175 * GIB), int(183 * GIB))
    assert ce.get_chunk_size(1, 2048, 248320, fixed_gb = 2.04) == 1


def test_t4_chunk_count_is_unchanged(ce, monkeypatch):
    free, total = int(6.75 * GIB), int(15 * GIB)
    _fake_device(monkeypatch, ce, free, total)
    new = ce.get_chunk_size(1, 2048, 248320, fixed_gb = 2.04)
    ce._get_chunk_multiplier.cache_clear()
    old = ce.get_chunk_size(1, 2048, 248320, target_gb = min(free / GIB * 0.5, 4.0),
                            fixed_gb = 2.04)
    assert new == old


def test_huge_transient_still_chunks_on_a_b200(ce, monkeypatch):
    _fake_device(monkeypatch, ce, int(175 * GIB), int(183 * GIB))
    n = ce.get_chunk_size(1, 32_768, 256_000)
    assert n > 1
    transient_gib = 32_768 * 256_000 * ce._CE_BYTES_PER_LOGIT / GIB
    assert transient_gib / n <= ce._CE_CAP_MAX_GB + 1e-6


# ------------------------------------------------------- the memo must not freeze
def test_live_free_vram_is_not_frozen_by_the_multiplier_memo(ce, monkeypatch):
    """Property (3): the reason target_gb is resolved OUTSIDE the cached function.

    Passing the None sentinel into the memo would key every later call on the
    first observation, so a card that filled up afterwards would keep the roomy
    chunk count and OOM.
    """
    total = int(16 * GIB)
    _fake_device(monkeypatch, ce, int(14 * GIB), total)
    roomy = ce.get_chunk_size(1, 8192, 248320)
    # Same vocab / shape / fixed_gb: identical key EXCEPT the live reading.
    # Deliberately do NOT cache_clear here - that is the whole point.
    monkeypatch.setattr(
        ce.torch, "cuda",
        types.SimpleNamespace(mem_get_info = lambda index = 0: (int(1.5 * GIB), total)),
        raising = False,
    )
    cramped = ce.get_chunk_size(1, 8192, 248320)
    assert cramped > roomy, (roomy, cramped)


def test_multiplier_memo_is_bounded(ce):
    """A live float key must not grow an unbounded cache."""
    assert ce._get_chunk_multiplier.cache_info().maxsize is not None
    for i in range(3000):
        ce._get_chunk_multiplier(248320, 1.0 + i * 1e-4, 0.0)
    info = ce._get_chunk_multiplier.cache_info()
    assert info.currsize <= info.maxsize


def test_cache_clear_still_exists_for_existing_callers(ce):
    assert callable(ce._get_chunk_multiplier.cache_clear)
    assert callable(ce._get_chunk_multiplier.cache_info)


# ------------------------------------------------------------------- degradation
def test_missing_mem_get_info_falls_back_to_the_old_constant(ce, monkeypatch):
    monkeypatch.setattr(ce, "DEVICE_TYPE", "cpu", raising = False)

    def _boom(index = 0):
        raise RuntimeError("no CUDA driver")

    monkeypatch.setattr(ce.torch, "cuda", types.SimpleNamespace(mem_get_info = _boom),
                        raising = False)
    assert ce._device_mem_info() is None
    assert ce._auto_target_gb() == ce._CE_CAP_MIN_GB


def test_xpu_uses_the_xpu_mem_get_info(ce, monkeypatch):
    monkeypatch.setattr(ce, "DEVICE_TYPE", "xpu", raising = False)
    seen = []

    def _xpu_info(index = 0):
        seen.append(index)
        return (int(40 * GIB), int(48 * GIB))

    monkeypatch.setattr(ce.torch, "xpu", types.SimpleNamespace(mem_get_info = _xpu_info),
                        raising = False)
    assert ce._auto_target_gb() == pytest.approx(48 / 6)
    assert seen == [0]


# ------------------------------------------------------------------ env overrides
def test_env_overrides_still_bypass_the_policy(ce, monkeypatch):
    """target_gb / n_chunks overrides must be untouched by the new policy."""
    _fake_device(monkeypatch, ce, int(175 * GIB), int(183 * GIB))
    # explicit target_gb wins over the VRAM-derived cap
    assert ce.get_chunk_size(1, 2048, 248320, target_gb = 4.0) == \
        ce.get_chunk_size(1, 2048, 248320, target_gb = 4.0)
    ce._get_chunk_multiplier.cache_clear()
    forced = ce.get_chunk_size(1, 2048, 248320, target_gb = 0.5)
    ce._get_chunk_multiplier.cache_clear()
    auto = ce.get_chunk_size(1, 2048, 248320)
    assert forced > auto
    # the module-level env knobs are still read the same way
    import inspect
    src = inspect.getsource(ce.unsloth_fused_ce_loss)
    assert "TARGET_GB" in src and "N_CHUNKS" in src


# --------------------------------------------------------------- device index
def test_budget_is_read_from_the_lm_head_device(ce, monkeypatch):
    """device_map='balanced' can put the lm_head on cuda:3; size against THAT card."""
    sizes = {0: (int(2 * GIB), int(16 * GIB)), 3: (int(160 * GIB), int(183 * GIB))}
    monkeypatch.setattr(ce, "DEVICE_TYPE", "cuda", raising = False)
    monkeypatch.setattr(
        ce.torch, "cuda",
        types.SimpleNamespace(mem_get_info = lambda index = 0: sizes[index]),
        raising = False,
    )
    ce._get_chunk_multiplier.cache_clear()
    assert ce._auto_target_gb(0) == pytest.approx(1.0)      # tiny, busy card
    assert ce._auto_target_gb(3) == pytest.approx(16.0)     # big, free card
    ce._get_chunk_multiplier.cache_clear()
    on0 = ce.get_chunk_size(1, 2048, 248320, device_index = 0)
    ce._get_chunk_multiplier.cache_clear()
    on3 = ce.get_chunk_size(1, 2048, 248320, device_index = 3)
    assert on0 > on3, (on0, on3)


def test_device_index_defaults_to_zero_when_omitted(ce, monkeypatch):
    """Trailing optional arg: existing call sites keep the old device-0 read."""
    seen = []
    monkeypatch.setattr(ce, "DEVICE_TYPE", "cuda", raising = False)

    def _info(index = 0):
        seen.append(index)
        return (int(8 * GIB), int(16 * GIB))

    monkeypatch.setattr(ce.torch, "cuda", types.SimpleNamespace(mem_get_info = _info),
                        raising = False)
    ce._get_chunk_multiplier.cache_clear()
    ce.get_chunk_size(1, 2048, 248320)
    ce._get_chunk_multiplier.cache_clear()
    ce.get_chunk_size(1, 2048, 248320, device_index = None)
    assert seen == [0, 0]


def test_get_chunk_size_signature_is_backwards_compatible(ce):
    import inspect
    params = list(inspect.signature(ce.get_chunk_size).parameters)
    assert params[:5] == ["bsz", "qlen", "vocab_size", "target_gb", "fixed_gb"]
    assert inspect.signature(ce.get_chunk_size).parameters["device_index"].default is None


@pytest.mark.parametrize("dev, expected", [
    ("cpu", None), ("cuda", 0), ("cuda:0", 0), ("cuda:3", 3),
    ("xpu:1", 1), ("meta", None),
])
def test_device_index_of(ce, dev, expected):
    assert ce._device_index_of(dev) == expected


def test_device_index_of_tolerates_garbage(ce):
    assert ce._device_index_of(object()) is None
    assert ce._device_index_of(None) is None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))


# ------------------------------------------------------------ the default is off
@pytest.mark.parametrize("total_gib", [15, 22.5, 24, 40, 80, 183, 320])
def test_default_is_off_and_matches_the_historical_constant(ce, monkeypatch, total_gib):
    """With the flag unset, the policy must be the pre-change formula on every card."""
    monkeypatch.delenv("UNSLOTH_CE_VRAM_AWARE_CAP", raising = False)
    total = int(total_gib * GIB)
    free  = int(total_gib * 0.98 * GIB)
    assert ce._auto_target_gb_from(free, total) == min(free / GIB * 0.5, 4.0)


def test_default_chunk_count_is_unchanged_on_a_large_card(ce, monkeypatch):
    """A B200 sized card must still pick the same chunk count it always did."""
    _fake_device(monkeypatch, ce, int(175 * GIB), int(183 * GIB))
    monkeypatch.setenv("UNSLOTH_CE_VRAM_AWARE_CAP", "1")
    ce._get_chunk_multiplier.cache_clear()
    opted_in = ce.get_chunk_size(1, 2048, 248320, fixed_gb = 2.04)

    monkeypatch.delenv("UNSLOTH_CE_VRAM_AWARE_CAP", raising = False)
    ce._get_chunk_multiplier.cache_clear()
    default = ce.get_chunk_size(1, 2048, 248320, fixed_gb = 2.04)

    stock = ce.get_chunk_size(1, 2048, 248320, fixed_gb = 2.04, target_gb = 4.0)
    assert default == stock, "default behaviour changed on a large card"
    assert opted_in != default, "the flag did not change anything"


def test_flag_is_read_live_not_captured_at_import(ce, monkeypatch):
    """Toggling the env var inside a process must take effect immediately."""
    total = int(183 * GIB)
    monkeypatch.delenv("UNSLOTH_CE_VRAM_AWARE_CAP", raising = False)
    assert ce._auto_target_gb_from(total, total) == pytest.approx(4.0)
    monkeypatch.setenv("UNSLOTH_CE_VRAM_AWARE_CAP", "1")
    assert ce._auto_target_gb_from(total, total) == pytest.approx(16.0)
    monkeypatch.setenv("UNSLOTH_CE_VRAM_AWARE_CAP", "0")
    assert ce._auto_target_gb_from(total, total) == pytest.approx(4.0)
