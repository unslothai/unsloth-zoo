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

import ast, os, stat, tempfile
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "unsloth_zoo" / "temporary_patches" / "gpt_oss.py"


def _load_replace_marker():
    tree = ast.parse(SRC.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_gpt_oss_replace_marker")
    ns = {"os": os}
    exec(compile(ast.Module(body = [fn], type_ignores = []), str(SRC), "exec"), ns)
    return ns["_gpt_oss_replace_marker"]


@pytest.mark.skipif(os.name != "posix", reason = "POSIX modes")
def test_marker_chmod_does_not_follow_a_swapped_symlink(tmp_path, monkeypatch):
    replace_marker = _load_replace_marker()
    cache = tmp_path / "cache"; cache.mkdir()
    victim = tmp_path / "victim"; victim.write_text("secret"); os.chmod(victim, 0o600)
    real_mkstemp = tempfile.mkstemp

    def racing_mkstemp(*a, **k):
        # A group member of the shared cache swaps the fresh temp name for a symlink to another of our files.
        fd, path = real_mkstemp(*a, **k)
        os.rename(path, path + ".moved")
        os.symlink(victim, path)
        return fd, path

    monkeypatch.setattr(tempfile, "mkstemp", racing_mkstemp)
    replace_marker(str(cache / "flavor"), "gpt_oss_mxfp4", mode = 0o664)
    assert stat.S_IMODE(os.stat(victim).st_mode) == 0o600
    assert victim.read_text() == "secret"


def test_marker_written_with_mode(tmp_path):
    replace_marker = _load_replace_marker()
    replace_marker(str(tmp_path / "flavor"), "x", mode = 0o664)
    assert (tmp_path / "flavor").read_text() == "x"
    if os.name == "posix":
        assert stat.S_IMODE(os.stat(tmp_path / "flavor").st_mode) == 0o664
    replace_marker(str(tmp_path / "flavor"), "y", mode = 0o600)
    assert (tmp_path / "flavor").read_text() == "y"
