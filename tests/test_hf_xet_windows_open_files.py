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

"""psutil < 8 open_files() can hang a Windows process (giampaolo/psutil#1967), so the stall watchdog and the
live-writer walk must not call it there."""

import sys
import time
import types

import pytest

from unsloth_zoo_downloads import hf_xet_fallback as xf


def _fake_psutil(version, calls):
    class _Proc:
        def __init__(self, pid = 0):
            self.pid = pid

        def open_files(self):
            calls.append(self.pid)
            return [types.SimpleNamespace(path = "/c/blobs/abc" + xf.INCOMPLETE_SUFFIX)]

    mod = types.ModuleType("psutil")
    mod.version_info = version
    mod.Process = _Proc
    mod.process_iter = lambda: iter([_Proc(7)])
    mod.NoSuchProcess = mod.ZombieProcess = LookupError
    return mod


@pytest.mark.parametrize(
    "platform, version, probed",
    [("win32", (7, 2, 2), False), ("win32", (8, 0, 0), True), ("linux", (7, 2, 2), True), ("darwin", (7, 2, 2), True)],
)
def test_open_files_is_only_used_where_it_cannot_hang(monkeypatch, platform, version, probed):
    calls = []
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(version, calls))
    monkeypatch.setattr(xf.sys, "platform", platform)

    owned = xf._child_open_incomplete_blobs(42)
    writers = xf._partial_paths_with_a_live_writer()

    if probed:
        assert owned == {"abc" + xf.INCOMPLETE_SUFFIX}
        assert writers is not None and len(writers) == 1
        assert calls == [42, 7]
    else:
        assert owned is None and writers is None
        assert calls == []


def test_watchdog_still_stalls_a_frozen_partial_without_open_files(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil((7, 2, 2), calls))
    monkeypatch.setattr(xf.sys, "platform", "win32")
    blobs = tmp_path / "models--ztest--win" / "blobs"
    blobs.mkdir(parents = True)
    (blobs / "frozen.incomplete").write_bytes(b"\0" * 4096)

    stalls = []
    stop = xf.start_watchdog(
        repo_ids = ["ztest/win"], cache_dir = str(tmp_path), on_stall = stalls.append, interval = 0.05,
        stall_timeout = 0.3, watch_new_partials_only = True, child_pid = 4242,
    )
    try:
        deadline = time.monotonic() + 5
        while not stalls and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        stop.set()
    assert stalls, "the repo-wide fallback never fired on a frozen partial"
    assert calls == []
