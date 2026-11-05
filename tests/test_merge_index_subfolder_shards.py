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

"""A base with no root shard, whose root index maps into subfolders (nvidia/Cosmos3-Edge keeps its
weights under `transformer/` and `vision_encoder/`), must be found and merged.

Only root `.safetensors` counted, so the repo read as absent and merged_16bit / GGUF export raised
"could not be read locally or on Hugging Face".
"""

from __future__ import annotations

import json
import os
import shutil

import pytest
import torch
from safetensors import safe_open

import _merge_e2e_helpers as H
from unsloth_zoo import saving_utils as S

REPO = "ns/cosmos3-shaped"
TARGETS = ["q_proj", "v_proj", "down_proj"]


def _write_tree(root, files):
    for name, payload in files.items():
        path = os.path.join(root, name)
        os.makedirs(os.path.dirname(path), exist_ok = True)
        with open(path, "wb") as f:
            f.write(payload)


def _cosmos3_index(shards):
    return json.dumps({
        "metadata": {"total_size": 1},
        "weight_map": {f"layers.{i}.weight": s for i, s in enumerate(shards)},
    }).encode()


_SHARDS = [
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
    "vision_encoder/model.safetensors",
]


def _fake_hub(monkeypatch, root):
    """Serve the directory `root` as REPO: listings per folder, downloads by relative name."""
    class _FS:
        def __init__(self, *a, **k):
            pass

        def ls(self, name, detail = True, **k):
            rel = name[len(REPO):].strip("/")
            folder = os.path.join(root, rel)
            if not os.path.isdir(folder):
                raise FileNotFoundError(name)
            return [
                {"name": f"{name}/{f}", "size": os.path.getsize(os.path.join(folder, f)),
                 "type": "directory" if os.path.isdir(os.path.join(folder, f)) else "file"}
                for f in sorted(os.listdir(folder))
            ]

    def _copy(names, local_dir):
        for f in names:
            src = os.path.join(root, f)
            if os.path.isfile(src):
                os.makedirs(os.path.dirname(os.path.join(local_dir, f)), exist_ok = True)
                shutil.copyfile(src, os.path.join(local_dir, f))

    def _hf_hub(repo_id = None, filename = None, local_dir = None, **k):
        if not os.path.isfile(os.path.join(root, filename)):
            raise FileNotFoundError(filename)
        if local_dir is None:
            return os.path.join(root, filename)
        _copy([filename], local_dir)
        return os.path.join(local_dir, filename)

    def _snapshot(repo_id = None, local_dir = None, allow_patterns = (), **k):
        _copy(list(allow_patterns), local_dir)
        return local_dir

    monkeypatch.setattr(S, "HfFileSystem", _FS)
    monkeypatch.setattr(S, "hf_hub_download", _hf_hub)
    monkeypatch.setattr(S, "snapshot_download", _snapshot)
    monkeypatch.setattr(S, "_resolve_hub_commit", lambda *a, **k: "0" * 40)
    monkeypatch.setattr(S, "_try_copy_all_from_cache", lambda *a, **k: False)


# Existence checks.

def test_hub_repo_with_only_subfolder_shards_exists(monkeypatch, tmp_path):
    _write_tree(str(tmp_path), {"config.json": b"{}", "model.safetensors.index.json": _cosmos3_index(_SHARDS),
                                **{s: b"x" for s in _SHARDS}})
    _fake_hub(monkeypatch, str(tmp_path))
    assert S.check_hf_model_exists(REPO) is True


def test_hub_index_naming_a_missing_subfolder_shard_is_not_a_base(monkeypatch, tmp_path):
    _write_tree(str(tmp_path), {"config.json": b"{}", "model.safetensors.index.json": _cosmos3_index(_SHARDS),
                                **{s: b"x" for s in _SHARDS[:2]}})
    _fake_hub(monkeypatch, str(tmp_path))
    assert S.check_hf_model_exists(REPO) is False


def test_hub_repo_with_no_weights_still_absent(monkeypatch, tmp_path):
    _write_tree(str(tmp_path), {"config.json": b"{}", "transformer/x.safetensors": b"x"})
    _fake_hub(monkeypatch, str(tmp_path))
    assert S.check_hf_model_exists(REPO) is False


def test_hub_index_escaping_the_repo_is_not_a_base(monkeypatch, tmp_path):
    _write_tree(str(tmp_path), {"model.safetensors.index.json": _cosmos3_index(["../x/model.safetensors"])})
    _fake_hub(monkeypatch, str(tmp_path))
    assert S.check_hf_model_exists(REPO) is False


def test_local_dir_with_only_subfolder_shards_exists(tmp_path):
    base = tmp_path / "base"
    _write_tree(str(base), {"config.json": b"{}", "model.safetensors.index.json": _cosmos3_index(_SHARDS),
                            **{s: b"x" for s in _SHARDS}})
    assert S.check_local_model_exists(str(base)) == os.path.abspath(str(base))
    assert S._local_snapshot_is_complete(str(base)) is True
    os.remove(base / _SHARDS[-1])
    assert S.check_local_model_exists(str(base)) is None
    assert S._local_snapshot_is_complete(str(base)) is False


# The merge itself: tensors read from and written back to the subfolders.

def _build_cosmos3_shaped_base(tmp_path):
    """A tiny llama saved sharded, then moved to Cosmos3's layout: no root shard, root index into subfolders."""
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    H.set_offline_cpu_env()
    if not H.family_available("llama"):
        pytest.skip("llama unavailable")
    base = str(tmp_path / "base")
    torch.manual_seed(H.SEED)
    AutoModelForCausalLM.from_config(H.make_spec("llama").config).to(torch.float32).save_pretrained(
        base, safe_serialization = True, max_shard_size = "20KB")
    model = AutoModelForCausalLM.from_pretrained(base, dtype = torch.float32)
    model.config._name_or_path = base
    peft_model = get_peft_model(model, LoraConfig(
        r = 4, lora_alpha = 8, lora_dropout = 0.0, bias = "none", target_modules = TARGETS))
    H.seed_lora(peft_model)

    index_path = os.path.join(base, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    shards = sorted(set(index["weight_map"].values()))
    assert len(shards) >= 2
    moved = {}
    for i, s in enumerate(shards):
        folder = "vision_encoder" if i == len(shards) - 1 else "transformer"
        os.makedirs(os.path.join(base, folder), exist_ok = True)
        os.replace(os.path.join(base, s), os.path.join(base, folder, s))
        moved[s] = f"{folder}/{s}"
    index["weight_map"] = {k: moved[v] for k, v in index["weight_map"].items()}
    with open(index_path, "w") as f:
        json.dump(index, f)
    assert not [f for f in os.listdir(base) if f.endswith(".safetensors")]
    return base, peft_model


def _read_nested(directory):
    out = {}
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith(".safetensors"):
                with safe_open(os.path.join(root, f), framework = "pt", device = "cpu") as g:
                    for k in g.keys():
                        out[k] = g.get_tensor(k)
    return out


def _merge(peft_model, name, out):
    return S.merge_and_overwrite_lora(
        get_model_name = lambda *a, **k: name,
        model = peft_model,
        tokenizer = None,
        save_directory = out,
        save_method = "merged_16bit",
        output_dtype = torch.float32,
        push_to_hub = False,
    )


def _assert_merged_and_reloads(base, out, adapted, peft_model):
    base_tensors, merged = _read_nested(base), _read_nested(out)
    assert sorted(merged) == sorted(base_tensors)
    with open(os.path.join(out, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    for key, shard in weight_map.items():
        assert os.path.isfile(os.path.join(out, shard)), shard
        assert shard.startswith(("transformer/", "vision_encoder/")), shard
    assert not [f for f in os.listdir(out) if f.endswith(".safetensors")]

    n_adapted = 0
    for key, base_t in base_tensors.items():
        hit = [a for ak, a in adapted.items() if key.endswith(ak) or key.endswith(ak + ".weight")]
        if hit:
            ref = H._ref_dense(base_t, hit[0]).to(torch.float32)
            torch.testing.assert_close(merged[key], ref, atol = 1e-5, rtol = 1e-5)
            n_adapted += 1
        else:
            assert torch.equal(merged[key], base_t), key
    assert n_adapted >= 1

    # The export reloads through its root index and matches the LoRA model's output.
    from transformers import AutoModelForCausalLM
    reloaded = AutoModelForCausalLM.from_pretrained(out, dtype = torch.float32).eval()
    ids = torch.arange(1, 9).unsqueeze(0)
    with torch.no_grad():
        want = peft_model.eval()(input_ids = ids).logits
        got = reloaded(input_ids = ids).logits
    torch.testing.assert_close(got, want, atol = 1e-4, rtol = 1e-4)


def test_merged_16bit_from_a_hub_base_with_only_subfolder_shards(monkeypatch, tmp_path):
    base, peft_model = _build_cosmos3_shaped_base(tmp_path)
    adapted = H.extract_adapted(peft_model)
    _fake_hub(monkeypatch, base)
    monkeypatch.setattr(S, "determine_base_model_source", lambda *a, **k: (REPO, False, "HF_unquantized", False, None))
    out = str(tmp_path / "merged")
    _merge(peft_model, REPO, out)
    _assert_merged_and_reloads(base, out, adapted, peft_model)


def test_merged_16bit_from_a_local_base_with_only_subfolder_shards(tmp_path):
    base, peft_model = _build_cosmos3_shaped_base(tmp_path)
    adapted = H.extract_adapted(peft_model)
    out = str(tmp_path / "merged")
    _merge(peft_model, base, out)
    _assert_merged_and_reloads(base, out, adapted, peft_model)
