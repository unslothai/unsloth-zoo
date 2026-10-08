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

"""`load_lora` must describe the live adapter to vLLM, not a config left on disk (unsloth#2097)."""

import json
import sys
import types

import pytest

pytest.importorskip("peft")
from peft import LoraConfig

import unsloth_zoo.vllm_utils as vu


class _Request:
    def __init__(self, name, int_id, path = None, lora_tensors = None, lora_config = None):
        self.lora_path = path
        self.lora_tensors = lora_tensors
        self.lora_config = lora_config


class _Model:
    def __init__(self, r = 16, lora_alpha = 32, max_lora_rank = 16):
        self.peft_config = {"default": LoraConfig(r = r, lora_alpha = lora_alpha, target_modules = ["q_proj"])}
        lora_config = types.SimpleNamespace(max_lora_rank = max_lora_rank)
        engine = types.SimpleNamespace(vllm_config = types.SimpleNamespace(lora_config = lora_config))
        self.vllm_engine = types.SimpleNamespace(llm_engine = engine)

    def state_dict(self):
        return {}


@pytest.fixture
def load_lora(monkeypatch):
    request_mod = types.ModuleType("vllm.lora.request")
    request_mod.LoRARequest = _Request
    monkeypatch.setitem(sys.modules, "vllm.lora.request", request_mod)
    monkeypatch.setattr(vu, "_remap_moe_expert_lora_keys", lambda model, sd: sd)
    monkeypatch.setattr(vu, "_check_lora_is_servable", lambda *a, **k: None)
    monkeypatch.setattr(vu, "_saved_adapter_lora_keys", lambda d: [])
    monkeypatch.setattr(vu, "_LIVE_LORA_CONFIGS", {}, raising = False)
    # Not the first request of the process: an earlier generate already loaded a LoRA.
    monkeypatch.setattr(vu, "LORA_REQUEST_ID", 5, raising = False)
    return vu.load_lora


def _write_stale(directory, **overrides):
    directory.mkdir()
    config = LoraConfig(r = 16, lora_alpha = 32, target_modules = ["q_proj"]).to_dict()
    config["target_modules"] = sorted(config["target_modules"])
    config["peft_type"] = "LORA"
    config.update(overrides)
    (directory / "adapter_config.json").write_text(json.dumps(config))


@pytest.mark.parametrize("stale", [{"r": 32}, {"lora_alpha": 16}])
def test_load_tensors_ignores_a_stale_config_from_an_earlier_run(tmp_path, load_lora, stale):
    directory = tmp_path / "grpo_trainer_lora_model"
    _write_stale(directory, **stale)
    request = load_lora(_Model(), str(directory), load_tensors = True)
    assert request.lora_config["r"] == 16
    assert request.lora_config["lora_alpha"] == 32
    on_disk = json.loads((directory / "adapter_config.json").read_text())
    assert (on_disk["r"], on_disk["lora_alpha"]) == (16, 32)


def test_load_tensors_writes_the_config_once_per_process(tmp_path, load_lora):
    directory = tmp_path / "trainer"
    model = _Model()
    first = load_lora(model, str(directory), load_tensors = True)
    (directory / "adapter_config.json").write_text("{}")
    second = load_lora(model, str(directory), load_tensors = True)
    assert second.lora_config == first.lora_config
    assert second.lora_config["r"] == 16


def test_config_reads_after_a_resave_see_the_new_file(tmp_path, load_lora):
    directory = tmp_path / "trainer"
    _write_stale(directory, r = 32)
    assert vu.get_peft_config(str(directory))["r"] == 32
    request = load_lora(_Model(), str(directory), load_tensors = True)
    assert request.lora_config["r"] == 16
    assert vu.get_peft_config(str(directory))["r"] == 16


def test_rank_check_reads_a_resaved_adapter_fresh(tmp_path, load_lora):
    directory = tmp_path / "adapter"
    _write_stale(directory, r = 32)
    with pytest.raises(ValueError):
        load_lora(_Model(max_lora_rank = 16), str(directory))
    config = json.loads((directory / "adapter_config.json").read_text())
    config["r"] = 16
    (directory / "adapter_config.json").write_text(json.dumps(config))
    assert load_lora(_Model(max_lora_rank = 16), str(directory)).lora_path == str(directory)


def test_path_load_above_max_lora_rank_is_refused_before_vllm(tmp_path, load_lora):
    directory = tmp_path / "old_adapter"
    _write_stale(directory, r = 32)
    with pytest.raises(ValueError, match = r"r = 32 .* max_lora_rank = 16"):
        load_lora(_Model(max_lora_rank = 16), str(directory))


def test_path_load_with_a_rank_pattern_above_max_lora_rank_is_refused(tmp_path, load_lora):
    directory = tmp_path / "patterned"
    _write_stale(directory, r = 16, rank_pattern = {"q_proj": 32})
    with pytest.raises(ValueError, match = r"r = 32 .* max_lora_rank = 16"):
        load_lora(_Model(max_lora_rank = 16), str(directory))


def test_path_load_within_max_lora_rank_is_passed_through(tmp_path, load_lora):
    directory = tmp_path / "adapter"
    _write_stale(directory, r = 16)
    request = load_lora(_Model(max_lora_rank = 16), str(directory))
    assert request.lora_path == str(directory)


def test_rank_check_fails_open_when_the_engine_is_unreachable(tmp_path, load_lora):
    directory = tmp_path / "adapter"
    _write_stale(directory, r = 32)
    model = _Model()
    model.vllm_engine = object()
    assert load_lora(model, str(directory)).lora_path == str(directory)
