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

"""Compile cache temp fallback must not reuse a folder another user made:
<tmp>/unsloth_compiled_cache has a fixed name, so another account could pre-create it and serve
rewritable Python; foreign ownership is faked by patching os.geteuid, since a real one needs
root."""
import os
import stat
import sys

import pytest

pytestmark = pytest.mark.skipif(os.name != "posix", reason = "POSIX shared temp dir")

compiler = pytest.importorskip("unsloth_zoo.compiler")


@pytest.fixture
def temp_base(tmp_path, monkeypatch):
    base = tmp_path / "shared_tmp"
    base.mkdir()
    monkeypatch.setattr(compiler.tempfile, "gettempdir", lambda: str(base))
    monkeypatch.setattr(compiler, "UNSLOTH_COMPILE_LOCATION", "unsloth_compiled_cache")
    monkeypatch.setattr(compiler, "UNSLOTH_COMPILE_USE_TEMP", False)
    return base


def _assert_private(location):
    st = os.lstat(location)
    assert stat.S_ISDIR(st.st_mode)
    assert st.st_uid == os.geteuid()
    assert not (st.st_mode & 0o022)


def test_fresh_temp_folder_is_default_name_and_private(temp_base):
    location, use_temp = compiler._get_compile_folder(use_tempfile = True)
    assert use_temp is True
    assert location == str(temp_base / "unsloth_compiled_cache")
    _assert_private(location)
    # Idempotent: a second call reuses it.
    again, _ = compiler._get_compile_folder(use_tempfile = True)
    assert again == location


def test_world_writable_planted_folder_is_not_used(temp_base):
    planted = temp_base / "unsloth_compiled_cache"
    planted.mkdir()
    os.chmod(planted, 0o777)
    location, _ = compiler._get_compile_folder(use_tempfile = True)
    assert location != str(planted)
    _assert_private(location)


def test_symlinked_planted_folder_is_not_used(temp_base, tmp_path):
    attacker = tmp_path / "attacker_dir"
    attacker.mkdir()
    (temp_base / "unsloth_compiled_cache").symlink_to(attacker)
    location, _ = compiler._get_compile_folder(use_tempfile = True)
    assert os.path.realpath(location) != str(attacker)
    _assert_private(location)


def test_folder_owned_by_another_user_is_not_used(temp_base, monkeypatch):
    planted = temp_base / "unsloth_compiled_cache"
    planted.mkdir(mode = 0o755)
    real_uid = os.geteuid()
    # Pretend we are someone else: the existing folder now belongs to "another user".
    monkeypatch.setattr(compiler.os, "geteuid", lambda: real_uid + 1)
    real_lstat = os.lstat

    def lstat(path, *a, **k):
        result = real_lstat(path, *a, **k)
        if os.path.basename(str(path)) == f"unsloth_compiled_cache_{real_uid + 1}":
            # The per-user folder we create is ours under the patched uid.
            return os.stat_result((result.st_mode, result.st_ino, result.st_dev,
                                   result.st_nlink, real_uid + 1, result.st_gid,
                                   result.st_size, int(result.st_atime),
                                   int(result.st_mtime), int(result.st_ctime)))
        return result

    monkeypatch.setattr(compiler.os, "lstat", lstat)
    location, _ = compiler._get_compile_folder(use_tempfile = True)
    assert location == str(temp_base / f"unsloth_compiled_cache_{real_uid + 1}")
