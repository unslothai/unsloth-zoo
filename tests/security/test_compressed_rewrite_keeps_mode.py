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

"""Dequantizing a compressed-tensors (int-packed / MXFP4-packed) shard rewrites it through `<shard>.unsloth_tmp`; that
temp was created at the umask default and os.replace'd over the shard, so a re-export into a folder whose weights the
user restricted (chmod 600) published them 0644. The T4 split pieces are likewise private while written."""

import ast, os, stat
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from safetensors import safe_open
from safetensors.torch import save, save_file

ROOT = Path(__file__).resolve().parents[2] / "unsloth_zoo"
pytestmark = pytest.mark.skipif(os.name != "posix", reason = "POSIX modes")


def _mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


@pytest.fixture
def umask022():
    old = os.umask(0o022); yield; os.umask(old)


def _extract(names, **namespace):
    from unsloth_zoo import private_staging
    assert Path(private_staging.__file__).resolve().parent == ROOT  # this arm's own tree
    tree = ast.parse((ROOT / "saving_utils.py").read_text(encoding = "utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    ns = dict(os = os, stat = stat, torch = torch, safe_open = safe_open,
              keep_mode = private_staging.keep_mode, precreate_private_temp = private_staging.precreate_private_temp)
    ns.update(namespace)
    exec(compile(ast.Module(body = nodes, type_ignores = []), "saving_utils.py", "exec"), ns)
    return ns


def _save_file_at_umask(seen):
    def save_file_(tensors, path, metadata = None):  # write through the existing temp (truncate), as safetensors does
        with open(path, "wb") as f:
            f.write(save(tensors, metadata = metadata))
        seen.append(_mode(path))
    return save_file_


def test_int_packed_rewrite_keeps_restricted_mode(tmp_path, umask022):
    seen = []
    ns = _extract(
        ["_rewrite_compressed_int_shard"],
        save_file = _save_file_at_umask(seen),
        _active_merge_device = lambda: "cpu",
        _INT_PACKED_SUFFIXES = ("weight_packed", "weight_scale"),
        _decompress_int_packed = lambda state, scheme: torch.zeros(4),
    )
    shard = tmp_path / "model.safetensors"
    save_file({"l.weight_packed": torch.zeros(4, dtype = torch.int32), "l.weight_scale": torch.ones(1)}, str(shard))
    os.chmod(shard, 0o600)
    plan = {"bases": {"l": None}, "elsewhere": {}, "logical": {"l": [4]}}
    ns["_rewrite_compressed_int_shard"](str(tmp_path), "model.safetensors", plan)
    with safe_open(str(shard), "pt") as f:
        assert "l.weight" in f.keys()  # the rewrite ran
    assert seen == [0o600], "dequantized shard was written readable to others"
    assert _mode(shard) == 0o600, "compressed re-export published a 0600 shard at 0644"
    assert not (tmp_path / "model.safetensors.unsloth_tmp").exists()


def test_t4_split_pieces_private_while_written(tmp_path, umask022):
    from collections import OrderedDict
    seen = []
    ns = _extract(["split_safetensor_file", "split_safetensors_to_shards"], save_file = _save_file_at_umask(seen),
                  OrderedDict = OrderedDict, UNSLOTH_ENABLE_LOGGING = False)
    path = tmp_path / "model.safetensors"
    save_file({"a": torch.zeros(64), "b": torch.zeros(64)}, str(path)); os.chmod(path, 0o600)
    pieces = ns["split_safetensor_file"]("model.safetensors", str(tmp_path), max_shard_size_gb = 300 / 2**30)
    assert len(pieces) == 2 and seen == [0o600, 0o600], f"split pieces written at {list(map(oct, seen))}"
