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

import ast, os
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "unsloth_zoo" / "__init__.py"


def _load():
    tree = ast.parse(SRC.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "has_429_exact_full_read")
    ns = {"os": os, "Path": Path}
    exec(compile(ast.Module(body = [fn], type_ignores = []), str(SRC), "exec"), ns)
    return ns["has_429_exact_full_read"]


@pytest.mark.skipif(os.name != "posix", reason = "symlinks")
def test_manifest_write_does_not_follow_planted_symlink(tmp_path):
    fn = _load()
    logs = tmp_path / "xet" / "logs"; logs.mkdir(parents = True)
    (logs / "a.log").write_bytes(b"hello\n")
    victim = tmp_path / "victim"; victim.write_text("precious")
    # Another account in a shared HF cache plants the predictable pid temp name.
    os.symlink(victim, logs / f".unsloth_429_scan.json.{os.getpid()}.tmp")
    assert fn(logs) is False
    assert victim.read_text() == "precious"


def test_manifest_cache_still_works(tmp_path):
    fn = _load()
    logs = tmp_path / "logs"; logs.mkdir()
    (logs / "a.log").write_bytes(b"x 429 Too Many Requests y")
    assert fn(logs) is True
    assert fn(logs) is True
    (logs / "a.log").write_bytes(b"ok")
    assert fn(logs) is False
    assert (logs / ".unsloth_429_scan.json").is_file()
    assert not [p for p in logs.iterdir() if p.name.endswith(".tmp")]
