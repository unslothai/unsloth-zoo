# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Comprehensive _active_merge_device() dispatch matrix.

Drives every supported accelerator combination from a single test host
by spoofing torch.cuda.is_available, torch.xpu.is_available,
torch.backends.mps.is_available, and torch.version.hip so we can
exercise the cascade

    cuda (covers ROCm)  ->  xpu  ->  mps  ->  cpu

deterministically without owning the actual hardware.

This is the regression net for the LoRA merge fix landed in PR #620.
A future change that hardcodes "cuda", reorders the cascade, or drops
a backend will fail loudly here.

Add a row to ``PROFILES`` to extend coverage; tests parametrize over
it automatically. No real hardware required.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Optional
from unittest import mock

import pytest
import torch


# ---------------------------------------------------------------------------
# Profile definition
# ---------------------------------------------------------------------------

@dataclass
class AcceleratorProfile:
    name: str
    cuda_available: bool          # torch.cuda.is_available()
    hip_version: Optional[str]    # torch.version.hip; None on NVIDIA
    xpu_available: bool           # torch.xpu.is_available()
    mps_available: bool           # torch.backends.mps.is_available()
    expect_device: str            # _active_merge_device() return value
    notes: str = ""


PROFILES = [
    AcceleratorProfile(
        name="nvidia_cuda",
        cuda_available=True, hip_version=None,
        xpu_available=False, mps_available=False,
        expect_device="cuda",
    ),
    AcceleratorProfile(
        name="amd_rocm",
        cuda_available=True, hip_version="6.1",
        xpu_available=False, mps_available=False,
        expect_device="cuda",
        notes="PyTorch ROCm aliases torch.cuda.* over HIP, so the cuda "
              "branch correctly covers AMD without a separate code path.",
    ),
    AcceleratorProfile(
        name="intel_xpu",
        cuda_available=False, hip_version=None,
        xpu_available=True, mps_available=False,
        expect_device="xpu",
    ),
    AcceleratorProfile(
        name="apple_silicon_mps",
        cuda_available=False, hip_version=None,
        xpu_available=False, mps_available=True,
        expect_device="mps",
        notes="MPS is the on-host LoRA merge device for the MLX backend; "
              "previously dropped when the helper routed through "
              "DEVICE_TYPE_TORCH (cuda/xpu/hip only).",
    ),
    AcceleratorProfile(
        name="cpu_only",
        cuda_available=False, hip_version=None,
        xpu_available=False, mps_available=False,
        expect_device="cpu",
    ),
]

PROFILE_IDS = [p.name for p in PROFILES]


# ---------------------------------------------------------------------------
# Spoofing fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def spoof_accelerator(monkeypatch):
    """Apply an AcceleratorProfile to torch in-process and reset the
    @lru_cache on _active_merge_device so the next call re-probes.

    Cache hygiene: ``_active_merge_device`` is decorated with
    ``functools.lru_cache(maxsize=1)`` so its first return is sticky for
    the rest of the process. The fixture clears the cache before AND
    after each test so neither this test nor any subsequent test in the
    session sees a stale "cpu" / "mps" answer baked in by an earlier
    spoof.
    """
    from unsloth_zoo.saving_utils import _active_merge_device

    _active_merge_device.cache_clear()

    def _apply(profile: AcceleratorProfile):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: profile.cuda_available)

        torch_version = torch.version
        monkeypatch.setattr(torch_version, "hip", profile.hip_version, raising=False)

        if hasattr(torch, "xpu"):
            monkeypatch.setattr(torch.xpu, "is_available", lambda: profile.xpu_available)
        elif profile.xpu_available:
            xpu_stub = types.SimpleNamespace(is_available=lambda: True)
            monkeypatch.setattr(torch, "xpu", xpu_stub, raising=False)

        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available",
                                lambda: profile.mps_available)
        elif profile.mps_available:
            mps_stub = types.SimpleNamespace(is_available=lambda: True)
            monkeypatch.setattr(torch.backends, "mps", mps_stub, raising=False)

        _active_merge_device.cache_clear()

    yield _apply

    _active_merge_device.cache_clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("profile", PROFILES, ids=PROFILE_IDS)
def test_active_merge_device_cascade(profile, spoof_accelerator):
    """_active_merge_device() returns the right backend per profile."""
    from unsloth_zoo.saving_utils import _active_merge_device
    spoof_accelerator(profile)
    actual = _active_merge_device()
    assert actual == profile.expect_device, (
        f"profile {profile.name}: expected {profile.expect_device!r}, "
        f"got {actual!r}. {profile.notes}"
    )


def test_active_merge_device_takes_no_args(spoof_accelerator):
    """The helper has no required positional args. The pre-fix signature was
    _active_merge_device(W) which silently dropped MPS and propagated the
    foreign device.index across families.
    """
    from unsloth_zoo.saving_utils import _active_merge_device
    spoof_accelerator(PROFILES[0])
    _active_merge_device()  # must not raise


def test_cuda_takes_priority_over_xpu_mps(spoof_accelerator):
    """If CUDA + XPU + MPS are all available, cuda wins."""
    from unsloth_zoo.saving_utils import _active_merge_device
    profile = AcceleratorProfile(
        name="cuda_plus_xpu_plus_mps",
        cuda_available=True, hip_version=None,
        xpu_available=True, mps_available=True,
        expect_device="cuda",
    )
    spoof_accelerator(profile)
    assert _active_merge_device() == "cuda"


def test_xpu_takes_priority_over_mps(spoof_accelerator):
    """If XPU + MPS are both available (no CUDA), xpu wins."""
    from unsloth_zoo.saving_utils import _active_merge_device
    profile = AcceleratorProfile(
        name="xpu_plus_mps",
        cuda_available=False, hip_version=None,
        xpu_available=True, mps_available=True,
        expect_device="xpu",
    )
    spoof_accelerator(profile)
    assert _active_merge_device() == "xpu"


def test_lru_cache_freezes_first_result(spoof_accelerator):
    """_active_merge_device() is @functools.lru_cache(maxsize=1). After the
    first call returns, subsequent calls under different spoofs return
    the cached value until cache_clear() is invoked. The fixture's
    cache_clear discipline is what makes the parametrized tests reliable.
    """
    from unsloth_zoo.saving_utils import _active_merge_device

    spoof_accelerator(PROFILES[0])  # nvidia_cuda
    first = _active_merge_device()
    assert first == "cuda"

    # Simulate a transient spoof change WITHOUT clearing the cache.
    with mock.patch.object(torch.cuda, "is_available", return_value=False):
        cached = _active_merge_device()
        assert cached == "cuda", "lru_cache should freeze the first result"

    # After cache_clear, re-probing picks up the new state.
    _active_merge_device.cache_clear()
    with mock.patch.object(torch.cuda, "is_available", return_value=False), \
         mock.patch.object(torch.backends.mps, "is_available", return_value=True):
        if hasattr(torch, "xpu"):
            with mock.patch.object(torch.xpu, "is_available", return_value=False):
                fresh = _active_merge_device()
        else:
            fresh = _active_merge_device()
        assert fresh == "mps"
