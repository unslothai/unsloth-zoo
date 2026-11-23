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

"""Every re-export / re-save keeps the mode of a weights file the user restricted (chmod 600) instead of publishing
the base checkpoint's, the umask default's or a fixed 0644 over it, and a first save never ignores the user's umask.
First exports under the usual 022 umask are unchanged (0644)."""

import ast, json, os, shutil, stat, struct, tempfile, time, types
from pathlib import Path
from typing import Union

import pytest

ROOT = Path(__file__).resolve().parents[2] / "unsloth_zoo"
pytestmark = pytest.mark.skipif(os.name != "posix", reason = "POSIX modes")


def _mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


def _helpers(ns):
    try:
        from unsloth_zoo import private_staging
        assert Path(private_staging.__file__).resolve().parent == ROOT  # this arm's own tree
        ns.update(keep_mode = private_staging.keep_mode, copy_keep_mode = private_staging.copy_keep_mode,
                  current_umask = private_staging.current_umask)
    except (ImportError, AttributeError):
        pass
    return ns


def _extract(path, names, namespace):
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(path), "exec"), _helpers(namespace))
    return namespace


@pytest.fixture
def umask022():
    old = os.umask(0o022); yield; os.umask(old)


@pytest.fixture
def umask077():
    old = os.umask(0o077); yield; os.umask(old)


def _fake_shard(path):
    header = json.dumps({"w": {"dtype": "U8", "shape": [8], "data_offsets": [0, 8]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(8))


def _copy_fn():
    ns = _extract(ROOT / "saving_utils.py", ["_copy_file_from_source"],
                  dict(os = os, stat = stat, shutil = shutil, tempfile = tempfile, Path = Path, Union = Union))
    return ns["_copy_file_from_source"]


def test_cache_copy_keeps_restricted_output_mode(tmp_path, umask022):
    copy = _copy_fn()
    src = tmp_path / "base.safetensors"; _fake_shard(src); os.chmod(src, 0o644)
    out = tmp_path / "out"; out.mkdir()
    copy(src, str(out), "model.safetensors")
    assert _mode(out / "model.safetensors") == 0o644  # first export unchanged
    os.chmod(out / "model.safetensors", 0o600)
    copy(src, str(out), "model.safetensors"); copy(src, str(out), "model.safetensors")
    assert _mode(out / "model.safetensors") == 0o600, "re-export copied the base shard's 0644 over a 0600 output"


def test_cache_copy_first_export_honours_umask(tmp_path, umask077):
    copy = _copy_fn()
    src = tmp_path / "base.safetensors"; _fake_shard(src); os.chmod(src, 0o644)
    copy(src, str(tmp_path), "model.safetensors")
    assert _mode(tmp_path / "model.safetensors") == 0o600, "first export ignored a 077 umask"


def test_single_pass_writer_first_export_honours_umask(tmp_path, umask077):
    ns = _extract(ROOT / "saving_utils.py", ["_SinglePassShardWriter"],
                  dict(os = os, stat = stat, shutil = shutil, time = time, json = json, tempfile = tempfile))
    source = tmp_path / "base.safetensors"; _fake_shard(source); os.chmod(source, 0o644)
    dest = tmp_path / "model.safetensors"
    ns["_SinglePassShardWriter"](source, str(dest)).commit()
    assert _mode(dest) == 0o600, "first single-pass export took the base checkpoint's 0644 under a 077 umask"


def test_t4_split_keeps_restricted_mode(tmp_path, umask022):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file, safe_open, save
    from collections import OrderedDict
    def save_file_at_umask(tensors, path, metadata = None):  # safetensors < 0.8 creates the file at the umask default
        Path(path).write_bytes(save(tensors, metadata = metadata))
    ns = _extract(ROOT / "saving_utils.py", ["split_safetensor_file", "split_safetensors_to_shards"],
                  dict(os = os, save_file = save_file_at_umask, safe_open = safe_open, OrderedDict = OrderedDict,
                       UNSLOTH_ENABLE_LOGGING = False))
    path = tmp_path / "model.safetensors"
    save_file({"a": torch.zeros(64), "b": torch.zeros(64)}, str(path)); os.chmod(path, 0o600)
    pieces = ns["split_safetensor_file"]("model.safetensors", str(tmp_path), max_shard_size_gb = 300 / 2**30)
    assert len(pieces) == 2
    assert [_mode(tmp_path / p) for p in pieces] == [0o600, 0o600], "T4 split published a 0600 shard at 0644"


def test_cpu_streamed_shard_count_change_keeps_restricted_mode(tmp_path, umask022):
    pytest.importorskip("torch"); pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    import torch
    from peft import LoraConfig, get_peft_model
    from unsloth_zoo.cpu import loader
    assert Path(loader.__file__).resolve() == ROOT / "cpu" / "loader.py"
    config = transformers.LlamaConfig(vocab_size = 64, hidden_size = 32, intermediate_size = 64, num_hidden_layers = 2,
                                      num_attention_heads = 2, num_key_value_heads = 2, tie_word_embeddings = False)
    torch.manual_seed(0)
    model = get_peft_model(transformers.LlamaForCausalLM(config), LoraConfig(r = 4, target_modules = ["q_proj"]))
    out = tmp_path / "merged"
    assert loader._stream_merged_save(model, str(out), "merged_16bit")
    assert _mode(out / "model.safetensors") == 0o644  # first export: umask default
    os.chmod(out / "model.safetensors", 0o600)
    assert loader._stream_merged_save(model, str(out), "merged_16bit", max_shard_size = "20KB")
    shards = sorted(out.glob("model-*.safetensors"))
    assert len(shards) > 1 and not (out / "model.safetensors").exists()
    assert {_mode(p) for p in shards} == {0o600}, "re-export with more shards published restricted weights at 0644"


def test_vllm_atomic_dir_save_keeps_restricted_mode(tmp_path, umask022):
    ns = _extract(ROOT / "vllm_utils.py", ["_save_dir_atomically"], dict(os = os, shutil = shutil))
    out = tmp_path / "lora"
    def save_fn(d):
        os.makedirs(d, exist_ok = True)
        Path(d, "adapter_model.safetensors").write_bytes(b"w")
    ns["_save_dir_atomically"](save_fn, str(out), 0)
    assert _mode(out / "adapter_model.safetensors") == 0o644
    os.chmod(out / "adapter_model.safetensors", 0o600)
    ns["_save_dir_atomically"](save_fn, str(out), 0)
    assert _mode(out / "adapter_model.safetensors") == 0o600, "distributed re-save widened restricted adapters"


def test_mtp_head_first_save_honours_umask(tmp_path, umask077):
    torch = pytest.importorskip("torch")
    from unsloth_zoo.speculative import stitch
    assert Path(stitch.__file__).resolve() == ROOT / "speculative" / "stitch.py"
    stitch.save_mtp_head(tmp_path, {"mtp.layers.0.weight": torch.zeros(4, 4)}, {})
    assert _mode(tmp_path / stitch.MTP_HEAD_WEIGHTS_NAME) == 0o600, "first MTP head save chmod-ed 0644 past a 077 umask"


def _mx_stub():
    def save_safetensors(path, weights, metadata = None):
        Path(path).write_bytes(b"w")
    return types.SimpleNamespace(save_safetensors = save_safetensors, load = lambda path: {},
                                 float16 = "f16", isfinite = None)


def test_mlx_decision_resave_keeps_restricted_mode(tmp_path, umask022):
    # MLX cannot run on Linux: extract save_decision_model and drive it with an mx stub.
    ns = _extract(ROOT / "mlx" / "decision.py", ["save_decision_model"],
                  dict(mx = _mx_stub(), json = json, os = os, shutil = shutil, Path = Path,
                       _merged_parameters = lambda model: {}, _state_name = lambda n: n))
    source = tmp_path / "src"
    for name in ("encoder", "tokenizer"):
        (source / name).mkdir(parents = True); (source / name / "w.bin").write_bytes(b"e"); os.chmod(source / name / "w.bin", 0o644)
    (source / "rl_agent_config.json").write_text("{}"); (source / "model.safetensors").write_bytes(b"s")
    out = tmp_path / "out"
    ns["save_decision_model"](None, out, source)
    assert _mode(out / "model.safetensors") == 0o644 and _mode(out / "encoder" / "w.bin") == 0o644
    os.chmod(out / "model.safetensors", 0o600); os.chmod(out / "encoder" / "w.bin", 0o600)
    ns["save_decision_model"](None, out, source)
    assert _mode(out / "model.safetensors") == 0o600, "MLX decision re-save widened restricted weights"
    assert _mode(out / "encoder" / "w.bin") == 0o600, "MLX decision re-save copied the source encoder's mode over"


def test_mlx_clef_resave_keeps_restricted_mode(tmp_path, umask022):
    ns = _extract(ROOT / "mlx" / "decision.py", ["_commit_clef"],
                  dict(mx = _mx_stub(), json = json, os = os, shutil = shutil, Path = Path,
                       _CLEF_CONFIG = "unsloth_decision_config.json", _CLEF_HEAD_CONFIG = "joint_head_config.json"))
    pipeline = types.SimpleNamespace(head_config = {})
    source, out = tmp_path / "src", tmp_path / "out"; source.mkdir(); out.mkdir()
    def commit():
        staging = Path(tempfile.mkdtemp(dir = out, prefix = ".saving-"))
        (staging / "model.safetensors").write_bytes(b"m")
        ns["_commit_clef"](pipeline, staging, out, source, {}, {})
        shutil.rmtree(staging, ignore_errors = True)
    commit()
    assert _mode(out / "model.safetensors") == 0o644 and _mode(out / "joint_head.safetensors") == 0o644
    os.chmod(out / "model.safetensors", 0o600); os.chmod(out / "joint_head.safetensors", 0o600)
    commit(); commit()
    assert _mode(out / "model.safetensors") == 0o600, "Clef re-save widened restricted weights"
    assert _mode(out / "joint_head.safetensors") == 0o600, "Clef re-save widened the restricted head"


def test_keep_mode_is_noop_on_first_save_and_idempotent(tmp_path, umask022):
    from unsloth_zoo.private_staging import keep_mode
    tmp = tmp_path / "t"; tmp.write_bytes(b"x"); os.chmod(tmp, 0o640)
    keep_mode(tmp, tmp_path / "missing")
    assert _mode(tmp) == 0o640
    final = tmp_path / "f"; final.write_bytes(b"y"); os.chmod(final, 0o600)
    keep_mode(tmp, final); keep_mode(tmp, final)
    assert _mode(tmp) == 0o600
