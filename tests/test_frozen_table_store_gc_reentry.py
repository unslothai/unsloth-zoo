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

"""Finalizers can run in any allocation made under _FILES_LOCK; blocking there self-deadlocks."""

import gc
import threading

import torch

import unsloth_zoo.frozen_table_store as fts
from unsloth_zoo.frozen_table_store import FrozenTableStore

from test_frozen_table_store import _write_sharded  # sibling test module, on sys.path under pytest


def test_a_table_collected_while_another_maps_its_files_does_not_deadlock(tmp_path, monkeypatch):
    torch.manual_seed(0)
    index, names = _write_sharded(tmp_path, [torch.randn(n, 8).to(torch.bfloat16) for n in (5, 3)])
    real_identity = fts._file_identity

    def collecting_identity(path):
        # _acquire_file calls this with _FILES_LOCK held: run the cyclic collector exactly there.
        gc.collect()
        return real_identity(path)

    built = []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        dead = FrozenTableStore.from_checkpoint(index, {"weight": names}, mode = "disk")
        dead._cycle = dead  # only the cyclic collector frees it, so its finalizer runs inside gc.collect()
        del dead
        monkeypatch.setattr(fts, "_file_identity", collecting_identity)
        worker = threading.Thread(
            target = lambda: built.append(FrozenTableStore.from_checkpoint(index, {"weight": names}, mode = "disk")),
            daemon = True,
        )
        worker.start()
        worker.join(30)
    finally:
        if was_enabled:
            gc.enable()
    assert built, "a table finalizer run inside _acquire_file blocked on _FILES_LOCK held by its own thread"
    store = built[0]
    monkeypatch.setattr(fts, "_file_identity", real_identity)
    # The dead table's references were dropped once, after the live one took its own: every mapping stays open.
    assert all(f.mm is not None and f.refs >= 1 for f in fts._FILES.values())
    store.close()
    assert not any(f.path.startswith(str(tmp_path)) for f in fts._FILES.values())


class _FakeMapping:
    def __init__(self, path):
        self.path, self.refs, self.closed = path, 1, False

    def close(self):
        self.closed = True


def test_a_release_queued_during_another_drain_is_not_stranded(monkeypatch):
    """Thread A must recheck the queue after unlocking, else B's queued release stays stranded."""
    early, late = _FakeMapping("early"), _FakeMapping("late")
    real_drain = fts._drain_releases
    arrived = []

    def drain_then_a_late_release():
        real_drain()
        if not arrived:
            arrived.append(True)
            fts._PENDING_RELEASES.append([late])  # what B's _release_files does while A still holds the lock

    monkeypatch.setattr(fts, "_drain_releases", drain_then_a_late_release)
    fts._release_files([early])
    assert early.closed and late.closed
    assert not fts._PENDING_RELEASES
