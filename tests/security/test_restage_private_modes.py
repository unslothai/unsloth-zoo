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

"""Re-exports over a user-restricted file keep its mode: the staging temp is private while written,
GPU shards never publish the base's wider mode; first exports keep the umask."""

import ast, json, os, shutil, stat, struct, tempfile, time, types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "unsloth_zoo"
pytestmark = pytest.mark.skipif(os.name != "posix", reason = "POSIX modes")


def _umask_default():
    old = os.umask(0o022); os.umask(old)
    return 0o666 & ~old


def _extract(path, names, namespace):
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(path), "exec"), namespace)
    return namespace


def _with_keep_mode(ns):
    try:
        from unsloth_zoo.private_staging import keep_mode
        ns["keep_mode"] = keep_mode
    except ImportError:
        pass
    return ns


@pytest.fixture
def umask022():
    old = os.umask(0o022)
    yield
    os.umask(old)


def _tiny_peft():
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    from peft import LoraConfig, get_peft_model
    config = transformers.LlamaConfig(vocab_size = 64, hidden_size = 32, intermediate_size = 64, num_hidden_layers = 1,
                                      num_attention_heads = 2, num_key_value_heads = 2, tie_word_embeddings = False)
    torch.manual_seed(0)
    return get_peft_model(transformers.LlamaForCausalLM(config), LoraConfig(r = 4, target_modules = ["q_proj"]))


def test_cpu_streamed_resave_temp_is_private_while_written(tmp_path, umask022, monkeypatch):
    from unsloth_zoo.cpu import loader
    assert Path(loader.__file__).resolve() == ROOT / "cpu" / "loader.py"  # this arm's own tree
    model = _tiny_peft()
    out = tmp_path / "merged"
    seen = []
    real = loader._write_safetensors
    def spy(path, *a, **k):
        result = real(path, *a, **k)
        seen.append(stat.S_IMODE(os.stat(path).st_mode))
        return result
    monkeypatch.setattr(loader, "_write_safetensors", spy)

    assert loader._stream_merged_save(model, str(out), "merged_16bit")
    weights = out / "model.safetensors"
    assert seen == [0o644] and stat.S_IMODE(os.stat(weights).st_mode) == 0o644  # first export: umask default

    os.chmod(weights, 0o600)
    seen.clear()
    assert loader._stream_merged_save(model, str(out), "merged_16bit")
    assert seen == [0o600], "staging temp was readable by others while the restricted weights were written"
    assert stat.S_IMODE(os.stat(weights).st_mode) == 0o600


def _fake_shard(path):
    header = json.dumps({"w": {"dtype": "U8", "shape": [8], "data_offsets": [0, 8]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(8))


def test_single_pass_writer_keeps_restricted_output_mode(tmp_path, umask022):
    ns = _extract(ROOT / "saving_utils.py", ["_SinglePassShardWriter"],
                  _with_keep_mode(dict(os = os, stat = stat, shutil = shutil, time = time, json = json, tempfile = tempfile)))
    Writer = ns["_SinglePassShardWriter"]
    source = tmp_path / "base.safetensors"; _fake_shard(source); os.chmod(source, 0o644)  # HF cache blob mode
    out = tmp_path / "out"; out.mkdir()
    dest = out / "model.safetensors"

    w = Writer(source, str(dest)); w.commit()
    assert stat.S_IMODE(os.stat(dest).st_mode) == 0o644  # first export: same as copy2, unchanged

    os.chmod(dest, 0o600)
    w = Writer(source, str(dest)); w.commit()
    assert stat.S_IMODE(os.stat(dest).st_mode) == 0o600, "re-export published the restricted shard world-readable"


def test_mlx_optimizer_resave_temp_is_private_while_written(tmp_path, umask022):
    # MLX cannot run on Linux: extract save_optimizer_state and drive it with an mx stub that, like mlx's FileWriter,
    # opens O_CREAT | O_TRUNC (an existing file keeps its mode).
    seen = []
    def save_safetensors(path, flat, metadata = None):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.write(fd, b"x"); seen.append(stat.S_IMODE(os.fstat(fd).st_mode)); os.close(fd)
    mx = types.SimpleNamespace(save_safetensors = save_safetensors, eval = lambda *a: None)
    mlx = types.SimpleNamespace(utils = types.SimpleNamespace(tree_flatten = lambda state: list(state.items())))
    ns = dict(mx = mx, mlx = mlx, os = os)
    src = (ROOT / "mlx" / "utils.py").read_text(encoding = "utf-8")
    if "precreate_private_temp" in src:
        from unsloth_zoo.private_staging import precreate_private_temp
        ns["precreate_private_temp"] = precreate_private_temp
    _extract(ROOT / "mlx" / "utils.py", ["_inherit_target_permissions", "save_optimizer_state"], ns)
    opt = types.SimpleNamespace(state = {"m": 1})
    target = tmp_path / "optimizer_state.safetensors"

    ns["save_optimizer_state"](opt, str(tmp_path))
    assert seen == [0o644] and stat.S_IMODE(os.stat(target).st_mode) == 0o644  # first save: umask default

    os.chmod(target, 0o600); seen.clear()
    ns["save_optimizer_state"](opt, str(tmp_path))
    assert seen == [0o600], "optimizer staging temp was readable by others while written"
    assert stat.S_IMODE(os.stat(target).st_mode) == 0o600


def test_precreate_is_idempotent_and_noop_on_first_export(tmp_path, umask022):
    from unsloth_zoo.private_staging import precreate_private_temp
    tmp, final = tmp_path / ".x.tmp", tmp_path / "x"
    precreate_private_temp(tmp, final)
    assert not tmp.exists()  # no final: nothing staged, umask default stands
    final.write_bytes(b"old")
    tmp.write_bytes(b"stale"); os.chmod(tmp, 0o644)
    precreate_private_temp(tmp, final); precreate_private_temp(tmp, final)
    assert stat.S_IMODE(os.stat(tmp).st_mode) == 0o600 and tmp.read_bytes() == b""
