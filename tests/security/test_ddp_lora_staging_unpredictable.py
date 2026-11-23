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

"""A distributed LoRA save stages each rank's adapters in a sibling folder. A predictable sibling name let anyone who
can write the parent (a shared output dir, sticky /tmp) pre-create it (or a symlink) so exist_ok accepted it: the
private adapters were written into a folder they own, and files they swapped in were then published as ours."""

import ast, os, shutil, stat
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "unsloth_zoo"
pytestmark = pytest.mark.skipif(os.name != "posix", reason = "POSIX modes")


def _save_dir_atomically():
    tree = ast.parse((ROOT / "vllm_utils.py").read_text(encoding = "utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_save_dir_atomically"]
    from unsloth_zoo import private_staging
    assert Path(private_staging.__file__).resolve().parent == ROOT
    ns = dict(os = os, shutil = shutil, keep_mode = private_staging.keep_mode)
    exec(compile(ast.Module(body = nodes, type_ignores = []), "vllm_utils.py", "exec"), ns)
    return ns["_save_dir_atomically"]


def test_preplanted_staging_folder_is_not_used(tmp_path):
    save = _save_dir_atomically()
    out = tmp_path / "lora"
    planted = tmp_path / "planted"
    planted.mkdir(mode = 0o777); os.chmod(planted, 0o777)
    # Every predictable name a peer could guess for this process.
    for rank in (0, 1):
        os.symlink(planted, f"{out}.unsloth_tmp_rank{rank}_{os.getpid()}")
    used = []

    def save_fn(d):
        used.append(d)
        Path(d, "adapter_model.safetensors").write_bytes(b"w")

    save(save_fn, str(out), 1)
    assert list(planted.iterdir()) == [], "adapters were staged in a folder another account controls"
    assert not os.path.islink(used[0])
    assert (out / "adapter_model.safetensors").read_bytes() == b"w"
    assert sorted(p.name for p in tmp_path.iterdir() if not p.is_symlink()) == ["lora", "planted"]


def test_staging_is_owner_only_and_cleaned(tmp_path):
    save = _save_dir_atomically()
    out, seen = tmp_path / "lora", []

    def save_fn(d):
        os.makedirs(d, exist_ok = True)  # as PEFT save_pretrained does
        seen.append(stat.S_IMODE(os.stat(d).st_mode))
        Path(d, "adapter_config.json").write_text("{}")

    save(save_fn, str(out), 0)
    save(save_fn, str(out) + os.sep, 0)  # idempotent, trailing separator
    assert seen == [0o700, 0o700]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["lora"]
