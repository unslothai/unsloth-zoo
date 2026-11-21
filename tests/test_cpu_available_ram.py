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

"""available_ram(): MemAvailable capped by the cgroup v2 headroom of the process's cgroup and every ancestor."""

import builtins

import pytest


def _fake_tree(tmp_path, monkeypatch, cgroup_line, files):
    (tmp_path / "proc").mkdir()
    (tmp_path / "proc" / "meminfo").write_text("MemTotal: 67108864 kB\nMemAvailable: 50331648 kB\n")  # 48 GiB free
    (tmp_path / "proc" / "cgroup").write_text(cgroup_line)
    for rel, text in files.items():
        path = tmp_path / "cg" / rel
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_text(text)
    real_open = builtins.open

    def fake_open(file, *args, **kwargs):
        name = str(file)
        if name == "/proc/meminfo":
            name = str(tmp_path / "proc" / "meminfo")
        elif name == "/proc/self/cgroup":
            name = str(tmp_path / "proc" / "cgroup")
        elif name.startswith("/sys/fs/cgroup"):
            name = str(tmp_path / "cg") + name[len("/sys/fs/cgroup"):]
        return real_open(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", fake_open)


GiB = 1 << 30


@pytest.mark.parametrize("files, expected", [
    # Leaf unlimited, the enclosing job capped at 8 GiB with 6 GiB used: 2 GiB left.
    ({"jobs/memory.max": str(8 * GiB), "jobs/memory.current": str(6 * GiB),
      "jobs/train/memory.max": "max", "jobs/train/memory.current": str(GiB)}, 2 * GiB),
    # Leaf limit tighter than the ancestor's: the leaf wins.
    ({"jobs/memory.max": str(8 * GiB), "jobs/memory.current": str(GiB),
      "jobs/train/memory.max": str(3 * GiB), "jobs/train/memory.current": str(GiB)}, 2 * GiB),
    # No finite limit anywhere: MemAvailable.
    ({"jobs/memory.max": "max", "jobs/memory.current": "0", "jobs/train/memory.max": "max",
      "jobs/train/memory.current": "0"}, 48 * GiB),
    # cgroup path not mounted here (container without a cgroup namespace): MemAvailable.
    ({}, 48 * GiB),
])
def test_available_ram_caps_by_every_cgroup_ancestor(tmp_path, monkeypatch, files, expected):
    from unsloth_zoo.cpu import runtime
    _fake_tree(tmp_path, monkeypatch, "0::/jobs/train\n", files)
    assert runtime.available_ram() == expected
