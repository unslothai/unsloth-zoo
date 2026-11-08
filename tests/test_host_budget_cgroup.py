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

"""Host activation / pin budgets inside an enforcing cgroup: a fixture cgroup tree, psutil reporting the machine."""

from pathlib import Path
from types import SimpleNamespace

import pytest

GiB = 1 << 30
ENV = "UNSLOTH_HOST_BUDGET_CGROUP"


def _write(directory: Path, **files):
    directory.mkdir(parents = True, exist_ok = True)
    for name, value in files.items():
        (directory / name.replace("__", ".")).write_text(f"{value}\n")


@pytest.fixture
def host(monkeypatch, tmp_path):
    """psutil sees a 64 GiB machine with 48 GiB available; the cgroup tree under tmp_path is what binds."""
    import psutil
    from unsloth_zoo import hf_xet_tuning

    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: SimpleNamespace(total = 64 * GiB, available = 48 * GiB)
    )
    monkeypatch.setattr(hf_xet_tuning, "CGROUP_ROOT", tmp_path)
    monkeypatch.setattr(hf_xet_tuning, "_proc_self_cgroup", lambda: ["0::/pod/job"])
    monkeypatch.delenv(ENV, raising = False)
    return tmp_path


def _v2(root: Path, limit, current, file_lru = 0, high = "max"):
    _write(root / "pod" / "job", memory__max = "max", memory__high = "max", memory__current = current)
    _write(root / "pod", memory__max = limit, memory__high = high, memory__current = current)
    (root / "pod" / "memory.stat").write_text(
        f"anon {current - file_lru}\nfile {file_lru}\nactive_file {file_lru // 2}\n"
        f"inactive_file {file_lru - file_lru // 2}\nshmem 0\n"
    )


def test_activation_target_follows_cgroup_headroom(host):
    from unsloth_zoo import memory_budget as mb

    _v2(host, 16 * GiB, 12 * GiB)
    # 16 GiB limit, 12 GiB charged: 4 GiB left, half of it is the target (was half of the machine's 48 GiB)
    assert mb._host_target_gb() == 2.0
    assert mb.host_memory_bytes() == (16 * GiB, 4 * GiB)


def test_reclaimable_page_cache_is_not_counted_as_used(host):
    from unsloth_zoo import memory_budget as mb

    # 12 GiB charged of which 6 GiB is file LRU (a load's mmap cache): 10 GiB is allocatable, as MemAvailable would say
    _v2(host, 16 * GiB, 12 * GiB, file_lru = 6 * GiB)
    assert mb._host_target_gb() == 5.0
    assert mb.host_memory_bytes() == (16 * GiB, 10 * GiB)


def test_memory_high_binds_when_lower(host):
    from unsloth_zoo import memory_budget as mb

    _v2(host, 16 * GiB, 4 * GiB, high = 8 * GiB)
    assert mb._host_target_gb() == 2.0
    assert mb.host_memory_bytes() == (8 * GiB, 4 * GiB)


def test_cgroup_v1_limit(host, monkeypatch):
    from unsloth_zoo import hf_xet_tuning
    from unsloth_zoo import memory_budget as mb

    monkeypatch.setattr(hf_xet_tuning, "_proc_self_cgroup", lambda: ["5:memory:/docker/abc"])
    _write(
        host / "memory" / "docker" / "abc",
        memory__limit_in_bytes = 16 * GiB,
        memory__usage_in_bytes = 13 * GiB,
    )
    (host / "memory" / "docker" / "abc" / "memory.stat").write_text(
        f"total_active_file {GiB}\ntotal_inactive_file 0\n"
    )
    _write(host / "memory", memory__limit_in_bytes = 9223372036854771712)
    assert mb._host_target_gb() == 2.0
    assert mb.host_memory_bytes() == (16 * GiB, 4 * GiB)


def test_no_limit_and_kill_switch_keep_the_psutil_reading(host, monkeypatch):
    from unsloth_zoo import memory_budget as mb

    _write(host / "pod" / "job", memory__max = "max", memory__current = 60 * GiB)
    _write(host / "pod", memory__max = "max", memory__current = 60 * GiB)
    assert mb._host_target_gb() == 24.0
    _v2(host, 16 * GiB, 12 * GiB)
    monkeypatch.setenv(ENV, "0")
    assert mb._host_target_gb() == 24.0


def test_checkpoint_pin_admission_respects_the_cgroup(host):
    from unsloth_zoo import gradient_checkpointing as gc

    _v2(host, 16 * GiB, 12 * GiB)
    # 4 GiB headroom: an 8 GiB pinned block no longer fits (the machine-wide reading admitted it)
    assert not gc._block_fits_host_ram(8 * GiB)
    assert gc._block_fits_host_ram(2 * GiB)


def test_multi_rank_pin_cap_uses_the_cgroup(host, monkeypatch):
    from unsloth_zoo import gradient_checkpointing as gc

    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.delenv("UNSLOTH_GC_PINNED_MAX_GB", raising = False)
    _v2(host, 32 * GiB, 8 * GiB)
    # 24 GiB headroom less max(4 GiB, 15% of the 32 GiB limit) reserve, split over 2 ranks
    reserve = max(4 * GiB, int(32 * GiB * 0.15))
    assert gc._compute_pinned_cap(held = 0) == (24 * GiB - reserve) // 2
    monkeypatch.setenv("UNSLOTH_GC_PINNED_MAX_GB", "1")
    assert gc._compute_pinned_cap(held = 0) == GiB


def test_block_swap_pin_budget_uses_the_cgroup(host, monkeypatch):
    from unsloth_zoo import block_swap as bs

    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising = False)
    _v2(host, 16 * GiB, 4 * GiB)
    assert bs._pin_budget() == 12 * GiB - 4 * GiB
