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

"""Hub cache fallback must not sit in a group-writable, non-sticky dir owned by another account,
where members could swap our folder; Path.stat is patched since a real owner needs root."""

import os
import pathlib
import stat

import pytest

from unsloth_zoo_downloads import hf_cache

pytestmark = pytest.mark.skipif(not hasattr(os, "getuid"), reason = "POSIX ownership check")


def _foreign(monkeypatch, *dirs):
    foreign = {str(d) for d in dirs}
    real_stat = pathlib.Path.stat

    def fake_stat(self, *args, **kwargs):
        st = real_stat(self, *args, **kwargs)
        if str(self) in foreign:
            fields = list(st)
            fields[stat.ST_UID] = os.getuid() + 1
            return os.stat_result(fields)
        return st

    monkeypatch.setattr(pathlib.Path, "stat", fake_stat)


def _scratch(tmp_path, mode):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    os.chmod(scratch, mode)
    return scratch


def test_group_writable_foreign_parent_refused(tmp_path, monkeypatch):
    scratch = _scratch(tmp_path, 0o2770)
    _foreign(monkeypatch, scratch)
    assert not hf_cache._is_safe_private_dir(scratch / "huggingface_user")


def test_existing_leaf_under_group_writable_foreign_parent_refused(tmp_path, monkeypatch):
    scratch = _scratch(tmp_path, 0o2770)
    (scratch / "huggingface_user").mkdir(mode = 0o700)
    _foreign(monkeypatch, scratch)
    assert not hf_cache._is_safe_private_dir(scratch / "huggingface_user")


def test_world_writable_existing_leaf_parent_refused(tmp_path, monkeypatch):
    scratch = _scratch(tmp_path, 0o777)
    (scratch / "huggingface_user").mkdir(mode = 0o700)
    _foreign(monkeypatch, scratch)
    assert not hf_cache._is_safe_private_dir(scratch / "huggingface_user")


def test_sticky_shared_tmp_still_accepted(tmp_path, monkeypatch):
    # /tmp (1777, root) and a sticky group scratch keep working: the default fallback is unchanged.
    scratch = _scratch(tmp_path, 0o1777)
    _foreign(monkeypatch, scratch)
    leaf = scratch / "huggingface_user"
    assert hf_cache._is_safe_private_dir(leaf)
    assert stat.S_IMODE(os.stat(leaf).st_mode) == 0o700
    assert hf_cache._is_safe_private_dir(leaf)  # idempotent on reuse


def test_own_group_writable_parent_accepted(tmp_path):
    scratch = _scratch(tmp_path, 0o2775)
    assert hf_cache._is_safe_private_dir(scratch / "huggingface_user")
