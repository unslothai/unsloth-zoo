# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""Unified-memory GPUs (AMD APUs gfx1150/1151, NVIDIA GB10): "unsloth" gradient checkpointing keeps the checkpointed
inputs on the device instead of copying them into host RAM, which is the same DRAM there. Detection runs on stub
device properties; the checkpoint paths run on CPU with streams stubbed (as test_gc_retained_graph_slots.py does)."""
import contextlib

import pytest
import torch

from unsloth_zoo import integrated_device as D


class _Props:
    def __init__(self, name = "stub", arch = None, **fields):
        self.name = name
        if arch is not None:
            self.gcnArchName = arch
        for k, v in fields.items():
            setattr(self, k, v)


GIB = 1024 ** 3


def _install(monkeypatch, props_list, hip = None, carveout = 512 * 1024 ** 2, platform = "linux"):
    """carveout: what the sysfs / Windows probe reports for the integrated parts (None = unreadable)."""
    monkeypatch.delenv(D.KEEP_ON_DEVICE_ENV, raising = False)
    monkeypatch.setattr(torch.version, "hip", hip, raising = False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(props_list))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: props_list[i])
    monkeypatch.setattr(D.sys, "platform", platform)
    monkeypatch.setattr(D, "_amdgpu_sysfs_vram_total", lambda props: carveout)
    monkeypatch.setattr(D, "_windows_hidden_memory", lambda: carveout)


@pytest.mark.parametrize("props, hip, expect", [
    # Strix Halo with the flag left unset (ROCm < 6.2 left it unassigned): the arch name decides.
    (_Props("Radeon 8060S Graphics", "gfx1151:xnack-", is_integrated = 0), "6.1.40091", True),
    (_Props("Radeon 890M", "gfx1150", is_integrated = 1), "7.2.0", True),
    # A discrete card with a garbage flag on old ROCm is not trusted.
    (_Props("Radeon RX 7900 XTX", "gfx1100", is_integrated = 1), "6.1.40091", False),
    (_Props("Radeon RX 7900 XTX", "gfx1100", is_integrated = 0), "7.2.0", False),
    # MI300A-style: discrete arch name, flag set on a ROCm that fills it in.
    (_Props("AMD Instinct MI300A", "gfx942:sramecc+:xnack-", is_integrated = 1), "7.0.1", True),
    (_Props("AMD Instinct MI300X", "gfx942:sramecc+:xnack-", is_integrated = 0), "7.0.1", False),
    (_Props("NVIDIA GB10", is_integrated = 1), None, True),
    (_Props("NVIDIA GB10"), None, True),  # wheel without the field
    (_Props("Orin", integrated = 1), None, True),
    (_Props("NVIDIA H100 80GB HBM3", is_integrated = 0), None, False),
    (_Props("Tesla T4"), None, False),
])
def test_integrated_part_detection(monkeypatch, props, hip, expect):
    if getattr(props, "total_memory", None) is None:
        props.total_memory = 100 * GIB
    _install(monkeypatch, [props], hip)
    assert D.device_is_integrated_part(0) is expect
    # Small (512 MiB) carve-out as on the gfx1151 CI boxes: every integrated part shares host RAM.
    assert D.device_shares_host_memory(0) is expect
    assert D.unified_memory_device_indices() == (frozenset({0}) if expect else frozenset())


# Measured on the gfx1151 CI boxes (Radeon 8060S, 128 GiB): Linux carve-out 512 MiB, torch total_memory 100 GiB
# (allocations land in GTT, MemAvailable drops 1:1); Windows 11 hidden RAM 0.92e9 B, total_memory 82.45e9 B.
@pytest.mark.parametrize("platform, carveout, total, pool", [
    ("linux", 512 * 1024 ** 2, 100 * GIB, "shared"),         # CI box, Linux
    ("win32", 920_000_000, 82_450_841_600, "shared"),         # CI box, Windows 11
    ("linux", 16 * GIB, 100 * GIB, "shared"),                # 16 GiB carve-out: most of the budget is GTT
    ("linux", 64 * GIB, 64 * GIB, "carveout"),               # BIOS 64 GiB: allocations come from the carve-out
    ("linux", 96 * GIB, 112 * GIB, "carveout"),              # 96 GiB carve-out even if GTT is counted on top
    ("win32", 96 * GIB, 110 * GIB, "carveout"),
    ("linux", None, 100 * GIB, "unknown"),                   # no sysfs (container): keep the offload
    ("win32", None, 80 * GIB, "unknown"),
])
def test_carveout_decides_the_pool(monkeypatch, platform, carveout, total, pool):
    props = _Props("Radeon 8060S Graphics", "gfx1151", is_integrated = 1, total_memory = total)
    _install(monkeypatch, [props], "7.2.0", carveout = carveout, platform = platform)
    info = D.device_memory_pool(0)
    assert info["pool"] == pool and info["carveout"] == carveout and info["total_memory"] == total
    assert D.device_shares_host_memory(0) is (pool == "shared")
    assert D.unified_memory_device_indices() == (frozenset({0}) if pool == "shared" else frozenset())
    # The env override still forces either way.
    monkeypatch.setenv(D.KEEP_ON_DEVICE_ENV, "1")
    assert D.unified_memory_device_indices() == frozenset({0})


def test_nvidia_integrated_needs_no_carveout(monkeypatch):
    _install(monkeypatch, [_Props("NVIDIA GB10", is_integrated = 1, total_memory = 119 * GIB)], None, carveout = None)
    assert D.device_memory_pool(0)["pool"] == "shared"


def test_discrete_pool(monkeypatch):
    _install(monkeypatch, [_Props("Radeon RX 7900 XTX", "gfx1100", is_integrated = 0, total_memory = 24 * GIB)], "7.2.0")
    assert D.device_memory_pool(0)["pool"] == "discrete"


def test_sysfs_carveout_matched_by_pci_bus(tmp_path):
    # Two amdgpu cards: the iGPU at 0000:c3:00.0 (bus 195) with 512 MiB, a dGPU at 0000:03:00.0 with 24 GiB.
    for card, address, vram in (("card0", "0000:03:00.0", 24 * GIB), ("card1", "0000:c3:00.0", 512 * 1024 ** 2)):
        real = tmp_path / "devices" / address
        real.mkdir(parents = True)
        (real / "mem_info_vram_total").write_text(f"{vram}\n")
        (tmp_path / card).mkdir()
        (tmp_path / card / "device").symlink_to(real)
    assert D._amdgpu_sysfs_vram_total(_Props(pci_bus_id = 195), root = str(tmp_path)) == 512 * 1024 ** 2
    assert D._amdgpu_sysfs_vram_total(_Props(pci_bus_id = 3), root = str(tmp_path)) == 24 * GIB
    assert D._amdgpu_sysfs_vram_total(_Props(pci_bus_id = 7), root = str(tmp_path)) is None
    assert D._amdgpu_sysfs_vram_total(_Props(), root = str(tmp_path / "nothing")) is None


def test_mixed_box_only_selects_the_integrated_device(monkeypatch):
    # A Ryzen desktop with its iGPU visible next to a Radeon card: the discrete card keeps offloading.
    _install(monkeypatch, [_Props("Radeon RX 7900 XTX", "gfx1100", is_integrated = 0, total_memory = 24 * GIB),
                           _Props("AMD Radeon Graphics", "gfx1036", is_integrated = 1, total_memory = 32 * GIB)],
             "7.2.0")
    assert D.unified_memory_device_indices() == frozenset({1})


def test_env_override(monkeypatch):
    _install(monkeypatch, [_Props("Tesla T4", total_memory = 15 * GIB),
                           _Props("NVIDIA GB10", is_integrated = 1, total_memory = 119 * GIB)])
    monkeypatch.setenv(D.KEEP_ON_DEVICE_ENV, "1")
    assert D.unified_memory_device_indices() == frozenset({0, 1})
    monkeypatch.setenv(D.KEEP_ON_DEVICE_ENV, "0")
    assert D.unified_memory_device_indices() == frozenset()


def test_properties_error_is_not_unified(monkeypatch):
    def boom(i): raise RuntimeError("no device")
    monkeypatch.setattr(torch.cuda, "get_device_properties", boom)
    assert D.device_shares_host_memory(0) is False


class _NoStream:
    def wait_stream(self, other): pass
    def wait_event(self, event): pass
    def record_event(self): return None


@pytest.fixture
def G(monkeypatch):
    monkeypatch.setenv("UNSLOTH_ALLOW_CPU", "1")
    monkeypatch.setenv("UNSLOTH_DISABLE_PINNED_MEMORY", "1")
    monkeypatch.setenv("UNSLOTH_DISABLE_DOUBLE_BUFFER", "1")
    try:
        from unsloth_zoo import gradient_checkpointing as G
    except ImportError as e:
        pytest.skip(f"unsloth_zoo import unavailable: {e}")
    G._double_buffer_disabled.cache_clear()
    getattr(G._keep_on_device_indices, "cache_clear", lambda: None)()
    names = ("CPU_BUFFERS", "CPU_INDEX", "GPU_BUFFERS", "GPU_BUFFERS_B", "USE_DOUBLE_BUFFER", "EXTRA_STREAMS",
             "MAIN_STREAMS", "NEXT_BUFFER_SLOT", "MINIMUM_SIZE", "FP32_OFFLOAD_EXACT", "BACKWARD_PASS", "FIRST_PASS",
             "LAST_GC_INDEX", "CURRENT_GC_INDEX", "USE_UNSLOTH_GC", "BUFFER_EVENTS_A", "BUFFER_EVENTS_B",
             "GPU_BUFFER_DTYPE")
    saved = {n: getattr(G, n, None) for n in names}
    G.initialize_unsloth_gradient_checkpointing(torch.float32)
    # Plenty of device memory unless a test says otherwise; counts the (driver) reads.
    G._test_free_reads = []

    def _free(i):
        G._test_free_reads.append(i)
        return G._test_free(len(G._test_free_reads)) if callable(G._test_free) else G._test_free
    G._test_free = (64 << 30, 128 << 30)
    monkeypatch.setattr(G, "_device_free_bytes", _free)
    G._KEEP_BUDGET.clear()
    monkeypatch.setattr(G, "_WARNED_KEEP_LOW_MEMORY", False)
    stream = _NoStream()
    monkeypatch.setattr(G, "_ensure_device_buffers", lambda i: None)
    monkeypatch.setattr(G, "_compute_stream", lambda i: stream)
    monkeypatch.setattr(G, "torch_gpu_stream", lambda s: contextlib.nullcontext(), raising = False)
    G.GPU_BUFFERS = {None: torch.empty(G.INITIAL_GPU_BUFFER_SIZE)}
    G.GPU_BUFFERS_B = None
    G.USE_DOUBLE_BUFFER = False
    G.EXTRA_STREAMS = {None: stream}
    G.NEXT_BUFFER_SLOT = {None: 0}
    G.MINIMUM_SIZE = 0
    G.FP32_OFFLOAD_EXACT = True
    leases = []
    real = G._CpuSlotLease

    class _Counting(real):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            leases.append(1)
    monkeypatch.setattr(G, "_CpuSlotLease", _Counting)
    G._test_leases = leases
    yield G
    for n, v in saved.items(): setattr(G, n, v)
    G._reset_cpu_slots()
    G._double_buffer_disabled.cache_clear()
    getattr(G._keep_on_device_indices, "cache_clear", lambda: None)()


H, S, L = 32, 16, 5


def _grads(G, ckpt):
    g = torch.Generator().manual_seed(0)
    ws = [(torch.randn(H, H, generator = g) / H ** 0.5).requires_grad_() for _ in range(L)]
    x = torch.randn(2, S, H, generator = g).requires_grad_()
    for step in range(2):  # FIRST_PASS, then the steady state that skips the last layer
        h = x * 1.0
        for w in ws:
            f = (lambda w: lambda h: h + torch.tanh(h @ w))(w)
            h = G.UnslothCheckpointFunction.apply(f, True, h) if ckpt else f(h)
        h.square().mean().backward()
        G.BACKWARD_PASS = True
        G.FIRST_PASS = False
    return [w.grad.clone() for w in ws] + [x.grad.clone()]


def test_discrete_device_still_offloads(G, monkeypatch):
    # CPU tensors have device.index None; an empty set means nothing keeps on device.
    monkeypatch.setattr(G, "_keep_on_device_indices", lambda: frozenset())
    ref = _grads(G, ckpt = False)
    got = _grads(G, ckpt = True)
    assert len(G._test_leases) > 0
    for a, b in zip(ref, got): torch.testing.assert_close(a, b, rtol = 0, atol = 0)


def test_unified_device_keeps_inputs_on_device(G, monkeypatch):
    monkeypatch.setattr(G, "_keep_on_device_indices", lambda: frozenset({None}))
    ref = _grads(G, ckpt = False)
    got = _grads(G, ckpt = True)
    assert G._test_leases == []
    for a, b in zip(ref, got): torch.testing.assert_close(a, b, rtol = 0, atol = 0)


def test_unified_skips_initial_pinned_pool(G, monkeypatch):
    monkeypatch.setattr(G, "_keep_on_device_indices", lambda: frozenset({0}))
    monkeypatch.setattr(G, "_visible_device_count", lambda: 1)
    G.initialize_unsloth_gradient_checkpointing(torch.float32)
    assert len(G.CPU_BUFFERS) == 0
    monkeypatch.setattr(G, "_keep_on_device_indices", lambda: frozenset())
    G.initialize_unsloth_gradient_checkpointing(torch.float32)
    assert len(G.CPU_BUFFERS) == G.INITIAL_CPU_BUFFER_COUNT


def test_non_reentrant_offloader_keeps_unified(G, monkeypatch):
    monkeypatch.setattr(G, "_keep_on_device_indices", lambda: frozenset({None}))
    off = G.NonReentrantActivationOffloader()
    monkeypatch.setattr(off, "_on_host", lambda x: False)  # pretend the CPU tensor is a device tensor
    x = torch.randn(4, 8, requires_grad = True)
    packed = off.pack(x)
    assert packed[0] is None and packed[1].data_ptr() == x.data_ptr()
    assert off.stats["kept_unified"] == 1 and off.stats["kept_unified_bytes"] == x.numel() * x.element_size()
    assert off.unpack(packed) is packed[1]


def _layer_input_bytes():
    return 2 * S * H * 4  # fp32 [2, S, H]


def test_free_memory_is_read_rarely_when_there_is_room(G, monkeypatch):
    monkeypatch.setattr(G, "_keep_on_device_indices", lambda: frozenset({None}))
    _grads(G, ckpt = True)
    assert G._test_leases == []
    assert len(G._test_free_reads) == 1  # one read for 2 steps x 5 layers


def test_low_device_memory_falls_back_to_offload(G, monkeypatch):
    # Headroom is max(2 GiB, 5% of total): with free = headroom + 2.5 inputs, the first inputs stay, the rest offload.
    monkeypatch.setattr(G, "_keep_on_device_indices", lambda: frozenset({None}))
    total = 16 << 30
    headroom = max(G.KEEP_ON_DEVICE_HEADROOM_MIN, int(total * G.KEEP_ON_DEVICE_HEADROOM_FRAC))
    start = headroom + int(2.5 * _layer_input_bytes())
    # The driver sees every input kept so far still allocated: each read reports one more layer used.
    G._test_free = lambda n: (start - (n - 1) * _layer_input_bytes(), total)
    ref = _grads(G, ckpt = False)
    got = _grads(G, ckpt = True)
    assert len(G._test_leases) > 0
    assert G._WARNED_KEEP_LOW_MEMORY is True
    for a, b in zip(ref, got): torch.testing.assert_close(a, b, rtol = 0, atol = 0)


def test_no_device_memory_offloads_like_a_discrete_card(G, monkeypatch):
    monkeypatch.setattr(G, "_keep_on_device_indices", lambda: frozenset())
    _grads(G, ckpt = True)
    discrete = len(G._test_leases)
    del G._test_leases[:]
    G._reset_cpu_slots()
    G.FIRST_PASS, G.LAST_GC_INDEX, G.CURRENT_GC_INDEX, G.BACKWARD_PASS = True, 0, 0, True
    monkeypatch.setattr(G, "_keep_on_device_indices", lambda: frozenset({None}))
    G._test_free = (0, 16 << 30)
    _grads(G, ckpt = True)
    assert len(G._test_leases) == discrete > 0


def test_unreadable_free_memory_offloads(G, monkeypatch):
    monkeypatch.setattr(G, "_keep_on_device_indices", lambda: frozenset({None}))

    def boom(i): raise RuntimeError("mem_get_info unavailable")
    monkeypatch.setattr(G, "_device_free_bytes", boom)
    assert G._keeps_on_device(None, 1024) is False


def test_non_reentrant_offloader_low_memory_offloads(G, monkeypatch):
    monkeypatch.setattr(G, "_keep_on_device_indices", lambda: frozenset({None}))
    G._test_free = (0, 16 << 30)
    off = G.NonReentrantActivationOffloader()
    monkeypatch.setattr(off, "_on_host", lambda x: False)
    monkeypatch.setattr(off, "_streamable", lambda x: False)
    x = torch.randn(4, 8, requires_grad = True)
    packed = off.pack(x)
    assert packed[0] is not None  # offloaded (eager copy), not kept
    assert off.stats["kept_unified"] == 0
