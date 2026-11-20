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

"""The Neural Engine split's host-side fallbacks, without the Neural Engine."""

import os

import pytest

pytest.importorskip("mlx.core")

from unsloth_zoo.mlx import ane  # noqa: E402


def test_an_unwritable_calibration_cache_is_skipped(tmp_path, monkeypatch):
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setattr(ane, "_CACHE_PATH", str(blocker / "cache.json"))
    ane._store_cache("k", {"share": 0.5})
    assert ane._load_cache() == {}


def test_a_prepare_that_raises_ane_error_reports_failed_and_attaches_nothing(monkeypatch):
    def broken(*_a):
        raise ane.AneError("IOSurfaceCreate failed")

    monkeypatch.setattr(ane, "_prepare", broken)
    model = type("M", (), {})()
    status = ane.prepare_ane_prefill(model)
    assert (status.layers, status.code) == (0, "failed") and model.__dict__[ane._PREPARED] is status


def test_no_prepare_runs_after_an_evaluation_timed_out(monkeypatch):
    monkeypatch.setattr(ane, "available", lambda: True)
    monkeypatch.setattr(ane.mx, "default_device", lambda: ane.mx.gpu)
    monkeypatch.setattr(ane, "_ABANDONED", [object()])
    assert ane._prepare(object(), 64, None).code == "timed_out"


def test_the_directory_lock_does_not_follow_a_planted_symlink(tmp_path):
    target = tmp_path / "victim"
    target.write_text("keep")
    os.symlink(target, tmp_path / "dir.lock")
    with pytest.raises(OSError):
        with ane._directory_lock(str(tmp_path / "dir")):
            pass
    assert target.read_text() == "keep"
    with ane._directory_lock(str(tmp_path / "other")):
        pass


def test_every_owner_of_a_program_directory_keeps_it_until_the_last_releases(tmp_path):
    directory = tmp_path / "program"
    owners = []
    for _ in range(2):
        program = ane.Program.__new__(ane.Program)
        program.directory = str(directory)
        owners.append(program._mark())
    ane._release_directory(str(directory), owners[0])
    assert directory.is_dir()
    ane._release_directory(str(directory), owners[1])
    assert not directory.exists()
