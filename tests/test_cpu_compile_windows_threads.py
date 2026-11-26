# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Windows CPU compile builds its C++ kernels in parallel (torch defaults compile_threads to 1 on win32)."""

import os
import sys

import pytest
import torch
import torch._inductor.config as inductor_config


@pytest.fixture
def windows(monkeypatch):
    from unsloth_zoo.cpu import loader
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(os, "process_cpu_count", lambda: 4, raising = False)
    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    for name in ("TORCHINDUCTOR_COMPILE_THREADS", "UNSLOTH_FORCE_SINGLE_COMPILE_WORKER", "TORCH_WARM_POOL"):
        monkeypatch.delenv(name, raising = False)
    seen = {}

    def fake_compile_layers(model):
        seen["threads"] = inductor_config.compile_threads
        seen["warm_pool"] = os.environ.get("TORCH_WARM_POOL")
        return 2
    monkeypatch.setattr(loader, "_cpu_compile_unavailable", lambda: None)
    monkeypatch.setattr(loader, "_compile_layers", fake_compile_layers)
    # config.patch, not monkeypatch: pytest's undo deletes the entry from torch 2.15's config module.
    with inductor_config.patch(compile_threads = 1):
        yield loader, seen


def _model():
    return torch.nn.Linear(2, 2)


def test_windows_compile_builds_kernels_in_parallel(windows):
    loader, seen = windows
    assert loader.compile_cpu_layers(_model(), explicit = True) == 2
    assert seen["threads"] == 4
    assert seen["warm_pool"] == "0"  # the Triton worker process pool stays off


@pytest.mark.parametrize("setup", ["env", "sentinel", "raised"])
def test_windows_user_choice_is_kept(windows, monkeypatch, setup):
    loader, seen = windows
    if setup == "env":
        monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "1")
    elif setup == "sentinel":
        monkeypatch.setenv("UNSLOTH_FORCE_SINGLE_COMPILE_WORKER", "1")
    else:
        inductor_config.compile_threads = 3
    loader.compile_cpu_layers(_model(), explicit = True)
    assert seen["threads"] == (3 if setup == "raised" else 1)
    assert seen["warm_pool"] is None


def test_windows_single_cpu_stays_serial(windows, monkeypatch):
    loader, seen = windows
    monkeypatch.setattr(os, "process_cpu_count", lambda: 1, raising = False)
    monkeypatch.setattr(os, "cpu_count", lambda: 1)
    loader.compile_cpu_layers(_model(), explicit = True)
    assert seen["threads"] == 1


def test_windows_thread_count_is_capped(windows, monkeypatch):
    loader, seen = windows
    monkeypatch.setattr(os, "process_cpu_count", lambda: 128, raising = False)
    monkeypatch.setattr(os, "cpu_count", lambda: 128)
    loader.compile_cpu_layers(_model(), explicit = True)
    assert seen["threads"] == 32


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_other_platforms_are_unchanged(windows, monkeypatch, platform):
    loader, seen = windows
    monkeypatch.setattr(sys, "platform", platform)
    loader.compile_cpu_layers(_model(), explicit = True)
    assert seen["threads"] == 1
    assert seen["warm_pool"] is None


def test_eager_fallback_leaves_compile_threads_alone(windows, monkeypatch):
    loader, seen = windows
    monkeypatch.setattr(loader, "_cpu_compile_unavailable", lambda: "no MSVC")
    assert loader.compile_cpu_layers(_model(), explicit = True) == 0
    assert inductor_config.compile_threads == 1
    assert "TORCH_WARM_POOL" not in os.environ
