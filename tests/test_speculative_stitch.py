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

import hashlib
import json
import os

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from unsloth_zoo.speculative.stitch import (
    MTP_APPENDED_SHARD_NAME,
    load_saved_mtp_head,
    save_mtp_head,
    write_mtp_tensors,
)

HEAD_NAMES = {
    "fc.weight": (8, 16),
    "pre_fc_norm_embedding.weight": (8,),
    "pre_fc_norm_hidden.weight": (8,),
    "layers.0.self_attn.q_proj.weight": (8, 8),
    "layers.0.mlp.up_proj.weight": (12, 8),
    "norm.weight": (8,),
}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path):
    with safe_open(str(path), framework = "pt") as f:
        return {k: f.get_tensor(k) for k in f.keys()}, f.metadata()


def _checkpoint(tmp_path, prefix = "mtp.", backbone_prefix = "model.", with_mtp = True, sharded = True,
                text_config = False):
    torch.manual_seed(0)
    shard1 = {
        f"{backbone_prefix}embed_tokens.weight": torch.randn(32, 8, dtype = torch.bfloat16),
        f"{backbone_prefix}layers.0.mlp.up_proj.weight": torch.randn(12, 8, dtype = torch.bfloat16),
    }
    shard2 = {f"{backbone_prefix}norm.weight": torch.randn(8, dtype = torch.bfloat16)}
    if with_mtp:
        shard2.update({prefix + n: torch.randn(*s, dtype = torch.bfloat16) for n, s in HEAD_NAMES.items()})
    if sharded:
        files = {"model-00001-of-00002.safetensors": shard1, "model-00002-of-00002.safetensors": shard2}
        for name, tensors in files.items():
            save_file(tensors, tmp_path / name, metadata = {"format": "pt", "keep": "me"})
        weight_map = {k: name for name, tensors in files.items() for k in tensors}
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map})
        )
    else:
        save_file({**shard1, **shard2}, tmp_path / "model.safetensors", metadata = {"format": "pt"})
    config = {"num_hidden_layers": 1, "hidden_size": 8}
    if text_config:
        config = {"architectures": ["Qwen3_5ForConditionalGeneration"], "text_config": config}
    if with_mtp:
        (config["text_config"] if text_config else config)["mtp_num_hidden_layers"] = 1
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


def _trained_head(dtype = torch.float32):
    torch.manual_seed(1)
    return {n: torch.randn(*s, dtype = dtype) for n, s in HEAD_NAMES.items()}


@pytest.mark.parametrize("prefix", ["mtp.", "model.language_model.mtp."])
def test_replace_touches_only_mtp_shard(tmp_path, prefix):
    ckpt = _checkpoint(tmp_path, prefix = prefix)
    other = ckpt / "model-00001-of-00002.safetensors"
    mtp_shard = ckpt / "model-00002-of-00002.safetensors"
    other_sha, index_sha, config_sha = _sha(other), _sha(ckpt / "model.safetensors.index.json"), _sha(ckpt / "config.json")
    old, old_meta = _read(mtp_shard)
    head = _trained_head()

    report = write_mtp_tensors(ckpt, head, verify = True)

    assert report["mode"] == "replaced" and report["prefix"] == prefix
    assert report["files_touched"] == ["model-00002-of-00002.safetensors"]
    assert _sha(other) == other_sha
    assert report["untouched"][other.name]["before"] == other_sha == report["untouched"][other.name]["after"]
    assert _sha(ckpt / "model.safetensors.index.json") == index_sha
    assert _sha(ckpt / "config.json") == config_sha
    new, new_meta = _read(mtp_shard)
    assert new_meta == old_meta
    assert torch.equal(new["model.norm.weight"], old["model.norm.weight"])
    for name, tensor in head.items():
        assert new[prefix + name].dtype == torch.bfloat16
        assert torch.equal(new[prefix + name], tensor.to(torch.bfloat16))


def test_prefixed_head_names_are_accepted(tmp_path):
    ckpt = _checkpoint(tmp_path)
    head = {"mtp." + k: v for k, v in _trained_head().items()}
    report = write_mtp_tensors(ckpt, head)
    assert report["mode"] == "replaced" and len(report["tensors_written"]) == len(HEAD_NAMES)


def test_strict_mismatch_raises_and_leaves_files(tmp_path):
    ckpt = _checkpoint(tmp_path)
    shard = ckpt / "model-00002-of-00002.safetensors"
    before = _sha(shard)
    head = _trained_head()
    head.pop("norm.weight")
    with pytest.raises(ValueError, match = "does not match"):
        write_mtp_tensors(ckpt, head)
    bad_shape = _trained_head()
    bad_shape["fc.weight"] = torch.zeros(8, 15)
    with pytest.raises(ValueError, match = "fc.weight"):
        write_mtp_tensors(ckpt, bad_shape)
    assert _sha(shard) == before
    assert not [p for p in ckpt.iterdir() if p.name.endswith(".tmp")]
    report = write_mtp_tensors(ckpt, head, strict = False)
    assert "mtp.norm.weight" not in report["tensors_written"]


@pytest.mark.parametrize("sharded", [True, False])
@pytest.mark.parametrize("text_config", [True, False])
def test_append_when_absent(tmp_path, sharded, text_config):
    ckpt = _checkpoint(tmp_path, with_mtp = False, sharded = sharded, text_config = text_config)
    head = _trained_head()
    report = write_mtp_tensors(ckpt, head)
    assert report["mode"] == "appended" and report["config_declared"]
    appended, _ = _read(ckpt / MTP_APPENDED_SHARD_NAME)
    assert set(appended) == {"mtp." + n for n in HEAD_NAMES}
    assert all(t.dtype == torch.bfloat16 for t in appended.values())
    index = json.loads((ckpt / "model.safetensors.index.json").read_text())
    assert all(index["weight_map"]["mtp." + n] == MTP_APPENDED_SHARD_NAME for n in HEAD_NAMES)
    assert index["weight_map"]["model.norm.weight"].startswith("model")
    config = json.loads((ckpt / "config.json").read_text())
    holder = config["text_config"] if text_config else config
    assert holder["mtp_num_hidden_layers"] == 1
    if text_config:
        assert "mtp_num_hidden_layers" not in config

    # Second call finds the appended head and only patches it.
    again = write_mtp_tensors(ckpt, head)
    assert again["mode"] == "replaced" and again["files_touched"] == [MTP_APPENDED_SHARD_NAME]


def test_idempotent(tmp_path):
    ckpt = _checkpoint(tmp_path)
    head = _trained_head()
    write_mtp_tensors(ckpt, head)
    first = {p.name: _sha(p) for p in ckpt.iterdir()}
    write_mtp_tensors(ckpt, head)
    assert {p.name: _sha(p) for p in ckpt.iterdir()} == first


def test_reconcile_keeps_declaration(tmp_path):
    from unsloth_zoo.saving_utils import reconcile_mtp_config

    ckpt = _checkpoint(tmp_path, with_mtp = False, text_config = True)
    write_mtp_tensors(ckpt, _trained_head())
    assert reconcile_mtp_config(ckpt) == "agrees"


def test_standalone_head_round_trip(tmp_path):
    head = _trained_head(torch.bfloat16)
    save_mtp_head(tmp_path, {"mtp." + k: v for k, v in head.items()},
                  {"source": "unsloth/Qwen3.5-0.8B", "hidden_size": 8, "not_json": object()})
    loaded, info = load_saved_mtp_head(tmp_path)
    assert set(loaded) == set(head)
    assert all(torch.equal(loaded[k], head[k]) for k in head)
    assert info["num_layers"] == 1 and info["source"] == "unsloth/Qwen3.5-0.8B" and "not_json" not in info
    assert load_saved_mtp_head(tmp_path / "missing") is None


def test_index_pointing_outside_the_folder_is_refused(tmp_path):
    _checkpoint(tmp_path)
    index_path = tmp_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"] = {k: "../elsewhere.safetensors" for k in index["weight_map"]}
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match = "outside the folder"):
        write_mtp_tensors(tmp_path, _trained_head())


def test_append_refuses_to_overwrite_a_backbone_shard(tmp_path):
    _checkpoint(tmp_path, with_mtp = False)
    os.rename(tmp_path / "model-00002-of-00002.safetensors", tmp_path / "model-mtp.safetensors")
    index_path = tmp_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"] = {k: ("model-mtp.safetensors" if v.endswith("00002.safetensors") else v) for k, v in index["weight_map"].items()}
    index_path.write_text(json.dumps(index))
    before = _sha(tmp_path / "model-mtp.safetensors")
    with pytest.raises(ValueError, match = "already holds model tensors"):
        write_mtp_tensors(tmp_path, _trained_head())
    assert _sha(tmp_path / "model-mtp.safetensors") == before
