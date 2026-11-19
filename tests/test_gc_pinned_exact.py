# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""Exact-size pinned blocks for gradient checkpointing offload: the gate that sets
pinned_max_round_threshold_mb (CUDA, torch 2.14+, no user pinned knob, kill switch), the growth
granularity, and the counted accounting. CPU only: torch's setter and allocator are stubbed."""
import os
import subprocess
import sys

import pytest
import torch

from test_gc_pinned_counted import allocator, gc_module, pin, unpin  # noqa: F401  (fixtures)

MIB = 1 << 20
ENV_VARS = ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_ALLOC_CONF")


class _Calls(list):
    state = None


@pytest.fixture
def gate(monkeypatch, gc_module):
    """Stub torch's allocator settings; returns the list of strings passed to the setter."""
    state = {"settings": ""}
    calls = _Calls()

    def setter(text):
        calls.append(text)
        state["settings"] = text
    monkeypatch.setattr(torch._C, "_accelerator_setAllocatorSettings", setter, raising = False)
    monkeypatch.setattr(torch._C, "_accelerator_getAllocatorSettings", lambda: state["settings"], raising = False)
    monkeypatch.setattr(gc_module, "_GC_PINNED_EXACT", False, raising = False)
    monkeypatch.setattr(gc_module, "_TORCH_MINOR", (2, 14), raising = False)
    monkeypatch.setattr(gc_module, "_HOST_ALLOC_KNOBS", True, raising = False)
    monkeypatch.setattr(gc_module, "_HOST_ALLOC_POLICY", (None, None, None), raising = False)
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.delenv("UNSLOTH_GC_PINNED_EXACT", raising = False)
    for var in ENV_VARS: monkeypatch.delenv(var, raising = False)
    calls.state = state
    return calls


def test_enabled_on_cuda_torch_214_keeps_the_settings_in_effect(gc_module, gate):
    gate.state["settings"] = "expandable_segments:True,max_split_size_mb:256"
    assert gc_module._maybe_enable_exact_pinned()
    assert gate == ["expandable_segments:True,max_split_size_mb:256,pinned_max_round_threshold_mb:4"]
    assert gc_module._maybe_enable_exact_pinned() and len(gate) == 1  # set once


def test_keeps_a_pinned_threshold_from_an_earlier_runtime_update(gc_module, gate):
    gate.state["settings"] = "pinned_max_round_threshold_mb:64"
    gc_module._host_alloc_policy()  # seen once, as the GC cap accounting does
    gate.state["settings"] = "max_split_size_mb:256"  # an unrelated later update
    assert not gc_module._maybe_enable_exact_pinned() and gate == []


def test_flip_160mb_slot_pins_160mb_not_256mb(gc_module, gate):
    assert gc_module._pinned_block_bytes(160 * MIB) == 256 * MIB  # torch's power-of-two rounding
    assert gc_module._maybe_enable_exact_pinned()
    assert gc_module._pinned_block_bytes(160 * MIB) == 160 * MIB
    assert gc_module._pinned_block_bytes(3 * MIB) == 4 * MIB      # small blocks keep rounding
    assert gc_module._host_alloc_settings_active()                # the allocator's own count decides


@pytest.mark.parametrize("case", ["torch_213", "hip_version", "device_hip", "device_xpu", "kill_switch", "no_setter"])
def test_gate_stays_off(monkeypatch, gc_module, gate, case):
    if case == "torch_213": monkeypatch.setattr(gc_module, "_TORCH_MINOR", (2, 13), raising = False)
    if case == "hip_version": monkeypatch.setattr(torch.version, "hip", "6.4", raising = False)
    if case == "device_hip": monkeypatch.setattr(gc_module, "DEVICE_TYPE", "hip", raising = False)
    if case == "device_xpu": monkeypatch.setattr(gc_module, "DEVICE_TYPE", "xpu", raising = False)
    if case == "kill_switch": monkeypatch.setenv("UNSLOTH_GC_PINNED_EXACT", "0")
    if case == "no_setter": monkeypatch.delattr(torch._C, "_accelerator_setAllocatorSettings")
    assert not gc_module._maybe_enable_exact_pinned()
    assert gate == [] and not gc_module._GC_PINNED_EXACT
    assert gc_module._pinned_block_bytes(160 * MIB) == 256 * MIB
    assert gc_module._exact_growth_numel(80 * MIB + 3, 2) == 80 * MIB + 3


@pytest.mark.parametrize("var", ENV_VARS)
@pytest.mark.parametrize("knob", ["pinned_max_round_threshold_mb:64", "pinned_max_cached_size_mb:512"])
def test_user_pinned_knob_is_never_overridden(monkeypatch, gc_module, gate, var, knob):
    monkeypatch.setenv(var, f"expandable_segments:True,{knob}")
    assert not gc_module._maybe_enable_exact_pinned() and gate == []


def test_user_pinned_knob_set_at_runtime_is_never_overridden(gc_module, gate):
    gate.state["settings"] = "pinned_max_cached_size_mb:1024"
    assert not gc_module._maybe_enable_exact_pinned() and gate == []


def test_setter_error_leaves_it_off(monkeypatch, gc_module, gate):
    def boom(text): raise RuntimeError("Unrecognized key")
    monkeypatch.setattr(torch._C, "_accelerator_setAllocatorSettings", boom, raising = False)
    assert not gc_module._maybe_enable_exact_pinned() and not gc_module._GC_PINNED_EXACT


def test_growth_granularity(monkeypatch, gc_module):
    monkeypatch.setattr(gc_module, "_GC_PINNED_EXACT", True, raising = False)
    grow = lambda nbytes: gc_module._exact_growth_numel(nbytes // 2, 2) * 2
    assert grow(160 * MIB) == 160 * MIB          # Qwen3-4B 32k slot: already on a step
    assert grow(40 * MIB) == 40 * MIB            # 8k
    assert grow(130 * MIB) == 144 * MIB          # 1/8 of 128 MiB steps
    assert grow(4 * MIB) == 4 * MIB              # at or under the threshold: untouched
    for nbytes in range(5 * MIB, 600 * MIB, 7_777_777):
        nbytes -= nbytes % 2
        assert nbytes <= grow(nbytes) < nbytes * 1.125 + 2


def test_grow_host_buffer_uses_the_granularity(monkeypatch, gc_module):
    monkeypatch.setattr(gc_module, "_GC_PINNED_EXACT", True, raising = False)
    monkeypatch.setenv("UNSLOTH_DISABLE_PINNED_MEMORY", "1")  # no GPU: the pageable path
    buffer = gc_module._mark_host_buffer(torch.empty(16, dtype = torch.uint8))
    grown = gc_module._grow_host_buffer(buffer, 130 * MIB)
    assert grown.numel() == 144 * MIB
    monkeypatch.setattr(gc_module, "_GC_PINNED_EXACT", False, raising = False)
    grown = gc_module._grow_host_buffer(gc_module._mark_host_buffer(torch.empty(16, dtype = torch.uint8)), 130 * MIB)
    assert grown.numel() == 130 * MIB


def test_counted_accounting_matches_the_allocator(monkeypatch, gc_module, gate, allocator):
    assert gc_module._maybe_enable_exact_pinned()
    fake = allocator(gate.state["settings"])
    monkeypatch.setattr(gc_module, "_block_fits_host_ram", lambda block: True, raising = False)
    blocks = [pin(gc_module, fake, 160 * MIB) for _ in range(3)]
    assert all(blocks) and fake.held() == gc_module.GC_PINNED_BYTES == 3 * 160 * MIB
    unpin(gc_module, fake, 160 * MIB, blocks.pop())
    again = pin(gc_module, fake, 150 * MIB)  # torch hands back the cached 160 MiB block
    assert again and fake.held() == 3 * 160 * MIB


# pinned_max_round_threshold_mb is not a recognized allocator setting before torch 2.13; match the production gate (2.14).
_TORCH_MINOR = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])


@pytest.mark.skipif(
    _TORCH_MINOR < (2, 14) or not hasattr(torch._C, "_accelerator_setAllocatorSettings"),
    reason = "torch < 2.14",
)
def test_real_setter_keeps_user_settings_string():
    # Real torch, no GPU needed: the settings string in effect after the call still carries the user's keys.
    code = (
        "import torch\n"
        "s = torch._C._accelerator_getAllocatorSettings()\n"
        "torch._C._accelerator_setAllocatorSettings(','.join(filter(None, (s, 'pinned_max_round_threshold_mb:4'))))\n"
        "print(torch._C._accelerator_getAllocatorSettings())\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in ENV_VARS}
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,expandable_segments:True"
    out = subprocess.run([sys.executable, "-c", code], env = env, capture_output = True, text = True, check = True)
    assert out.stdout.strip() == "max_split_size_mb:256,expandable_segments:True,pinned_max_round_threshold_mb:4"


def test_outgrown_exact_block_is_flushed_once_at_the_next_step(monkeypatch, gc_module):
    flushes, raw = [], []
    monkeypatch.setattr(gc_module, "_empty_host_cache", lambda: flushes.append(1) or True, raising = False)
    monkeypatch.setattr(torch._C, "_host_emptyCache", lambda: raw.append(1), raising = False)
    monkeypatch.setattr(gc_module, "_EXACT_OUTGROWN", False, raising = False)
    monkeypatch.setattr(gc_module, "_GC_PINNED_EXACT", True, raising = False)
    gc_module._note_outgrown_block(256 * 1024)      # the initial small slots: torch reuses those
    gc_module._flush_outgrown_blocks()
    assert flushes == []
    gc_module._note_outgrown_block(6 * MIB)
    gc_module._note_outgrown_block(7 * MIB)
    assert len(raw) == 2                          # freed at once when its copies are done, no sync
    gc_module._flush_outgrown_blocks()
    gc_module._flush_outgrown_blocks()
    assert flushes == [1] and len(raw) == 2
    monkeypatch.setattr(gc_module, "_GC_PINNED_EXACT", False, raising = False)  # kill switch / base: never
    gc_module._note_outgrown_block(6 * MIB)
    gc_module._flush_outgrown_blocks()
    assert flushes == [1] and len(raw) == 2
