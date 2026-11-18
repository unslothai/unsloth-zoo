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

"""The multi-rank RAM ledger in the shared temp dir is only used when this user owns it privately."""

import os
import tempfile

import pytest

pytestmark = pytest.mark.skipif(os.name == "nt", reason = "the temp dir is per user on Windows")


def _bs(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_IS_PRESENT", "1")
    import unsloth_zoo.block_swap as bs

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "job")
    monkeypatch.setattr(bs, "_local_world_size", lambda: 2)
    return bs, os.path.join(str(tmp_path), f"unsloth_block_swap_{os.getuid()}_job_{os.getppid()}")


def test_a_planted_writable_dir_is_not_used(tmp_path, monkeypatch):
    bs, path = _bs(tmp_path, monkeypatch)
    os.mkdir(path)
    os.chmod(path, 0o777)  # stands in for a directory another OS user created first
    victim = tmp_path / "victim"
    victim.write_text("precious")
    os.symlink(victim, os.path.join(path, f"0.{os.getpid()}.tmp"))
    bs._publish_host_pending(123)
    assert victim.read_text() == "precious"
    assert bs._rank_ledger() is None


def test_a_symlinked_dir_is_not_used(tmp_path, monkeypatch):
    bs, path = _bs(tmp_path, monkeypatch)
    real = tmp_path / "elsewhere"
    real.mkdir(mode = 0o700)
    os.symlink(real, path)
    assert bs._rank_ledger() is None


def test_own_private_dir_still_works_and_peer_bytes_are_clamped(tmp_path, monkeypatch):
    bs, path = _bs(tmp_path, monkeypatch)
    led = bs._rank_ledger()
    assert led is not None and led[0] == path
    assert (os.stat(path).st_mode & 0o077) == 0
    bs._publish_host_pending(5)
    with open(os.path.join(path, "0")) as f:
        assert f.read().split()[1] == "5"
    with open(os.path.join(path, "1"), "w") as f:
        f.write(f"{bs._HOST_SESSION} -999999999999")
    assert bs._others_host_pending(10) >= 0
    assert bs._rank_ledger() is not None
