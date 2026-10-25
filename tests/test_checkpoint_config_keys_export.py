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

"""Ling 3.x GGUF export on transformers 5.20 (unslothai/unsloth#14916).

The built-in bailing_hybrid config saves Ling-3.0-tiny's `norm_topk_prob` as `moe_renormalize` and
rewrites `head_dim`, so the merged export's config.json no longer carried the keys llama.cpp's
BailingMoeV3 converter reads, and the converter itself was handed the built-in class's to_dict().
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from unsloth_zoo.saving_utils import _restore_checkpoint_config_keys


# Ling-3.0-tiny's config.json, trimmed to the keys these tests touch.
_LING_SOURCE = {
    "architectures": ["BailingMoeV3ForCausalLM"],
    "auto_map": {"AutoConfig": "configuration_bailing_moe_v3.BailingMoeV3Config"},
    "model_type": "bailing_hybrid",
    "head_dim": 128,
    "qk_rope_head_dim": 64,
    "kda_lower_bound": -5,
    "max_position_embeddings": 131072,
    "n_group": 8,
    "norm_topk_prob": True,
    "num_experts_per_tok": 8,
    "rope_scaling": None,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.45.0",
    "vocab_size": 157184,
}

# What transformers 5.20's BailingHybridConfig.save_pretrained writes for it (measured).
_LING_NATIVE_SAVE = {
    "architectures": ["BailingMoeV3ForCausalLM"],
    "auto_map": {"AutoConfig": "configuration_bailing_moe_v3.BailingMoeV3Config"},
    "model_type": "bailing_hybrid",
    "head_dim": 64,
    "linear_head_dim": 128,
    "qk_rope_head_dim": 64,
    "kda_lower_bound": -5.0,
    "model_max_length": 131072,
    "num_expert_group": 8,
    "moe_renormalize": True,
    "num_experts_per_token": 8,
    "rope_parameters": {"rope_type": "default", "rope_theta": 6000000},
    "dtype": "bfloat16",
    "transformers_version": "5.20.2",
    "vocab_size": 157200,  # resized by the training run, carried over by the export
}


class _BailingHybridLikeConfig:
    attribute_map = {
        "max_position_embeddings": "model_max_length",
        "norm_topk_prob": "moe_renormalize",
        "n_group": "num_expert_group",
        "num_experts_per_tok": "num_experts_per_token",
    }


def _write(tmp_path, data):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding = "utf-8")
    return path


def test_ling_export_gets_the_checkpoint_keys_back(tmp_path):
    path = _write(tmp_path, _LING_NATIVE_SAVE)
    assert _restore_checkpoint_config_keys(path, json.loads(json.dumps(_LING_SOURCE)), _BailingHybridLikeConfig)
    out = json.loads(path.read_text(encoding = "utf-8"))
    for key in ("norm_topk_prob", "n_group", "num_experts_per_tok", "max_position_embeddings"):
        assert out[key] == _LING_SOURCE[key]
    assert out["head_dim"] == 128
    assert "moe_renormalize" not in out and "linear_head_dim" not in out and "rope_parameters" not in out
    # What the export set itself stays.
    assert out["vocab_size"] == 157200
    assert out["dtype"] == "bfloat16" and "torch_dtype" not in out
    assert out["transformers_version"] == "5.20.2"
    # The saved float: transformers 5.20 refuses the checkpoint's own int -5.
    assert isinstance(out["kda_lower_bound"], float)


def test_nested_managed_keys_come_from_the_export(tmp_path):
    source = {"auto_map": {}, "norm_topk_prob": True, "text_config": {"vocab_size": 10, "head_dim": 128}}
    written = {"auto_map": {}, "moe_renormalize": True, "text_config": {"vocab_size": 12, "head_dim": 64}}
    path = _write(tmp_path, written)
    assert _restore_checkpoint_config_keys(path, source, _BailingHybridLikeConfig)
    out = json.loads(path.read_text(encoding = "utf-8"))
    assert out["text_config"] == {"vocab_size": 12, "head_dim": 128}


@pytest.mark.parametrize("source, written", [
    # Not written for its own code: a built-in checkpoint keeps the built-in save.
    ({"norm_topk_prob": True}, {"moe_renormalize": True}),
    # Repo code, but nothing lost to an alias (DeepSeek-V3, MiniMax-M2 on their built-in classes).
    ({"auto_map": {}, "moe_renormalize": True, "rope_scaling": None}, {"auto_map": {}, "moe_renormalize": True}),
])
def test_other_exports_are_left_alone(tmp_path, source, written):
    path = _write(tmp_path, written)
    before = path.read_bytes()
    assert not _restore_checkpoint_config_keys(path, source, _BailingHybridLikeConfig)
    assert path.read_bytes() == before


def test_unreadable_source_is_a_no_op(tmp_path):
    path = _write(tmp_path, _LING_NATIVE_SAVE)
    before = path.read_bytes()
    assert not _restore_checkpoint_config_keys(path, None, _BailingHybridLikeConfig)
    assert path.read_bytes() == before


def test_real_bailing_hybrid_save_round_trips(tmp_path):
    transformers = pytest.importorskip("transformers")
    config_class = getattr(transformers, "BailingHybridConfig", None)
    if config_class is None:
        pytest.skip("transformers has no built-in bailing_hybrid (added in 5.20)")
    source = dict(_LING_SOURCE)
    config = config_class.from_dict({**source, "kda_lower_bound": float(source["kda_lower_bound"])})
    config.save_pretrained(tmp_path)
    path = tmp_path / "config.json"
    assert "norm_topk_prob" not in json.loads(path.read_text(encoding = "utf-8"))
    assert _restore_checkpoint_config_keys(path, json.loads(json.dumps(source)), config_class)
    out = json.loads(path.read_text(encoding = "utf-8"))
    assert {k: out[k] for k in ("norm_topk_prob", "n_group", "num_experts_per_tok", "head_dim")} == \
        {"norm_topk_prob": True, "n_group": 8, "num_experts_per_tok": 8, "head_dim": 128}
    # The export still loads on the built-in class exactly like the checkpoint does.
    reloaded = config_class.from_pretrained(tmp_path).to_dict()
    for key in ("moe_renormalize", "num_expert_group", "num_experts_per_token", "linear_head_dim", "head_dim"):
        assert reloaded[key] == config.to_dict()[key]


# --- converter side -------------------------------------------------------------------------------

def _load_llama_cpp_module():
    module_path = Path(__file__).resolve().parents[1] / "unsloth_zoo" / "llama_cpp.py"
    spec = importlib.util.spec_from_file_location("llama_cpp_under_test_hparams", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_ENTRYPOINT = b"""\
import os


class ModelBase:
    @staticmethod
    def load_hparams(dir_model, is_mistral_format, guess = True):
        return {"from": "to_dict"}


def main():
    return ModelBase.load_hparams(DIR, False)


if __name__ == '__main__':
    main()
"""


def _run_patched(content, model_dir):
    namespace = {"__name__": "converter_under_test", "DIR": str(model_dir)}
    exec(compile(content, "unsloth_convert_hf_to_gguf.py", "exec"), namespace)
    return namespace


@pytest.fixture(scope = "module")
def llama_cpp():
    return _load_llama_cpp_module()


def test_converter_reads_config_json_for_bailingmoe3(llama_cpp, tmp_path):
    patched, status = llama_cpp._patch_checkpoint_hparams(_ENTRYPOINT)
    assert status == "applied"
    (tmp_path / "config.json").write_text(json.dumps(_LING_SOURCE), encoding = "utf-8")
    ns = _run_patched(patched, tmp_path)
    assert ns["main"]() == _LING_SOURCE
    # Every call site goes through ModelBase.load_hparams, keyword arguments included.
    assert ns["ModelBase"].load_hparams(tmp_path, is_mistral_format = False, guess = False) == _LING_SOURCE


@pytest.mark.parametrize("config", [
    {"architectures": ["Qwen3ForCausalLM"]},
    {"architectures": None},
    "not json",
])
def test_converter_other_architectures_keep_load_hparams(llama_cpp, tmp_path, config):
    patched, _ = llama_cpp._patch_checkpoint_hparams(_ENTRYPOINT)
    text = config if isinstance(config, str) else json.dumps(config)
    (tmp_path / "config.json").write_text(text, encoding = "utf-8")
    ns = _run_patched(patched, tmp_path)
    assert ns["main"]() == {"from": "to_dict"}
    assert ns["ModelBase"].load_hparams(tmp_path, True) == {"from": "to_dict"}


def test_converter_patch_is_idempotent_and_keeps_crlf(llama_cpp):
    once, _ = llama_cpp._patch_checkpoint_hparams(_ENTRYPOINT)
    twice, status = llama_cpp._patch_checkpoint_hparams(once)
    assert status == "already-applied" and twice == once
    crlf, _ = llama_cpp._patch_checkpoint_hparams(_ENTRYPOINT.replace(b"\n", b"\r\n"))
    assert b"\n" not in crlf.replace(b"\r\n", b"")


def test_monolith_converter_is_left_alone(llama_cpp):
    # llama.cpp b4000: `class Model`, `load_hparams(dir_model)`, no ModelBase to wrap.
    legacy = b"class Model:\n    @staticmethod\n    def load_hparams(dir_model):\n        return {}\n\n\nif __name__ == '__main__':\n    pass\n"
    assert llama_cpp._patch_checkpoint_hparams(legacy, "monolith") == (legacy, "not-package")
    exec(compile(legacy, "convert_hf_to_gguf.py", "exec"), {"__name__": "__main__"})


def test_converter_without_main_guard_is_reported(llama_cpp):
    content = b"class ModelBase: pass\n"
    assert llama_cpp._patch_checkpoint_hparams(content) == (content, "pattern-missing")
