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

"""MLX Laya (decision) saves copy the source encoder / tokenizer folders with copytree. A re-save must not reopen a
folder the user restricted (copytree copies the source folder's mode onto it last), and a first save of restricted
source files must not leave them readable at the umask default while they are written."""

import ast, json, os, shutil, stat
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "unsloth_zoo"
pytestmark = pytest.mark.skipif(os.name != "posix", reason = "POSIX modes")


def _mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


def _staging():
    from unsloth_zoo import private_staging
    assert Path(private_staging.__file__).resolve().parent == ROOT  # this arm's own tree
    return private_staging


def _save_decision_model():
    staging = _staging()
    ns = {name: getattr(staging, name) for name in staging.__all__}

    class _MX:
        def load(self, path): return {}
        def save_safetensors(self, path, weights): Path(path).write_bytes(b"w")

    ns.update(mx = _MX(), json = json, os = os, shutil = shutil, Path = Path,
              _merged_parameters = lambda model: {}, _state_name = lambda n: n)
    path = ROOT / "mlx" / "decision.py"
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "save_decision_model"]
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(path), "exec"), ns)
    return ns["save_decision_model"]


@pytest.fixture
def umask022():
    old = os.umask(0o022); yield; os.umask(old)


def _source(tmp_path, file_mode = 0o644):
    source = tmp_path / "src"
    for name in ("encoder", "tokenizer"):
        (source / name / "sub").mkdir(parents = True)
        for f in (source / name / "w.bin", source / name / "sub" / "v.json"):
            f.write_bytes(b"e"); os.chmod(f, file_mode)
    (source / "rl_agent_config.json").write_text("{}"); (source / "model.safetensors").write_bytes(b"s")
    return source


def test_resave_keeps_restricted_folders(tmp_path, umask022):
    save = _save_decision_model()
    source, out = _source(tmp_path), tmp_path / "out"
    save(None, out, source)
    for name in ("encoder", "tokenizer"):
        os.chmod(out / name / "sub", 0o700); os.chmod(out / name, 0o700)
    save(None, out, source)
    for name in ("encoder", "tokenizer"):
        assert _mode(out / name) == 0o700, f"Laya re-save reopened the restricted {name} folder"
        assert _mode(out / name / "sub") == 0o700, f"Laya re-save reopened a restricted folder inside {name}"


def test_first_save_folders_follow_the_source_and_umask(tmp_path, umask022):
    save = _save_decision_model()
    source, out = _source(tmp_path), tmp_path / "out"
    os.chmod(source / "encoder", 0o700)
    save(None, out, source)
    assert _mode(out / "encoder") == 0o700 and _mode(out / "tokenizer") == 0o755
    assert _mode(out / "encoder" / "w.bin") == 0o644


def test_first_copy_of_restricted_file_private_while_written(tmp_path, umask022, monkeypatch):
    staging = _staging()
    src, dst = tmp_path / "secret.bin", tmp_path / "out.bin"
    src.write_bytes(b"x" * 4096); os.chmod(src, 0o600)
    seen = []
    for name in ("_fastcopy_sendfile", "_fastcopy_fcopyfile", "copyfileobj"):  # Linux, macOS, fallback
        original = getattr(shutil, name, None)
        if original is None:
            continue
        def spy(fsrc, fdst, *args, _original = original, **kwargs):
            seen.append(stat.S_IMODE(os.fstat(fdst.fileno()).st_mode))
            return _original(fsrc, fdst, *args, **kwargs)
        monkeypatch.setattr(shutil, name, spy)
    staging.copy_keep_mode(str(src), str(dst))
    assert seen and all(mode == 0o600 for mode in seen), f"restricted file written at {[oct(m) for m in seen]}"
    assert _mode(dst) == 0o600 and dst.read_bytes() == src.read_bytes()


def test_first_copy_of_ordinary_file_unchanged(tmp_path, umask022):
    staging = _staging()
    src, dst = tmp_path / "a.json", tmp_path / "b.json"
    src.write_text("{}"); os.chmod(src, 0o644)
    staging.copy_keep_mode(str(src), str(dst))
    assert _mode(dst) == 0o644 and dst.read_text() == "{}"


def test_failed_first_copy_leaves_nothing(tmp_path, umask022):
    staging = _staging()
    with pytest.raises(OSError):
        staging.copy_keep_mode(str(tmp_path / "missing.bin"), str(tmp_path / "out.bin"))
    assert not os.path.lexists(tmp_path / "out.bin")


def test_symlink_copied_as_link(tmp_path, umask022):
    staging = _staging()
    (tmp_path / "target.txt").write_text("t")
    os.symlink("target.txt", tmp_path / "link.txt")
    staging.copy_keep_mode(str(tmp_path / "link.txt"), str(tmp_path / "copy.txt"), follow_symlinks = False)
    assert os.path.islink(tmp_path / "copy.txt")


@pytest.mark.skipif(os.name != "posix" or (hasattr(os, "geteuid") and os.geteuid() == 0), reason = "POSIX modes, non-root")
def test_first_copy_works_under_a_umask_without_owner_write(tmp_path):
    from unsloth_zoo.private_staging import copy_keep_mode
    src = tmp_path / "src.bin"
    src.write_bytes(b"weights")
    dst = tmp_path / "dst.bin"
    old = os.umask(0o222)
    try:
        copy_keep_mode(str(src), str(dst))
    finally:
        os.umask(old)
    assert dst.read_bytes() == b"weights"
    assert stat.S_IMODE(os.stat(dst).st_mode) == 0o444
