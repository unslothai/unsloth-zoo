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

"""Fused MoE expert adapters saved before unsloth_zoo 2026.9.6 load repacked to PEFT's layout. CPU only."""

from __future__ import annotations

import json

import pytest
import torch

peft = pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")
from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from unsloth_zoo.temporary_patches import moe_utils as MU  # noqa: E402

E, R = 4, 3
EXPERTS = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]


def _base():
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    torch.manual_seed(0)
    config = Qwen3MoeConfig(
        vocab_size = 32, hidden_size = 8, intermediate_size = 16, moe_intermediate_size = 6,
        num_experts = E, num_experts_per_tok = 2, num_hidden_layers = 1,
        num_attention_heads = 2, num_key_value_heads = 1, head_dim = 4,
    )
    model = Qwen3MoeForCausalLM(config).eval()
    if not hasattr(model.model.layers[0].mlp.experts, "gate_up_proj"):
        pytest.skip(reason = "transformers < 5 keeps per-expert Linears: no fused expert LoRA to repack")
    return model


def _expert_lora_B(model):
    return {
        name: p.detach().clone() for name, p in model.named_parameters()
        if "lora_B" in name and "experts" in name
    }


def _delta(model):
    return {
        n: m.get_delta_weight("default").detach().clone() for n, m in model.named_modules()
        if getattr(m, "parameter_name", None) in ("gate_up_proj", "down_proj")
    }


@pytest.fixture
def patched_load(monkeypatch):
    # Unwrap any copy installed at import (the compiled cache's), so this module's own runs.
    original = PeftModel.load_adapter
    while hasattr(original, "__wrapped__"):
        original = original.__wrapped__
    monkeypatch.setattr(PeftModel, "load_adapter", original)
    monkeypatch.setattr(MU, "_original_peft_model_load_adapter", None)
    monkeypatch.delenv("UNSLOTH_MOE_LORA_B_LAYOUT", raising = False)
    assert MU._patch_peft_load_adapter_for_moe_layout()


def _legacy_adapter(tmp_path, config_edit):
    """A trained (rank_major) adapter, plus the same function written the pre-2026.9.6 way."""
    model = get_peft_model(_base(), LoraConfig(r = R, lora_alpha = 2 * R, target_parameters = EXPERTS))
    g = torch.Generator().manual_seed(1)
    for name, p in model.named_parameters():
        if "lora_" in name:
            with torch.no_grad():
                p.copy_(torch.randn(p.shape, generator = g))
    expected_B, expected_delta = _expert_lora_B(model), _delta(model)
    model.save_pretrained(str(tmp_path))
    weights = load_file(str(tmp_path / "adapter_model.safetensors"))
    # Rank-major column j * E + e held expert e's rank j; the old packing put it at e * R + j.
    to_grouped = torch.arange(E * R).view(R, E).t().reshape(-1)
    for key in weights:
        if "lora_B" in key and "experts" in key:
            weights[key] = weights[key][:, to_grouped].contiguous()
    save_file(weights, str(tmp_path / "adapter_model.safetensors"))
    path = tmp_path / "adapter_config.json"
    config = json.loads(path.read_text())
    config.pop(MU.FUSED_EXPERT_LORA_LAYOUT_KEY, None)
    config.pop(MU.FUSED_EXPERT_LORA_DETAIL_KEY, None)
    config["auto_mapping"] = {"base_model_class": "Qwen3MoeForCausalLM", "unsloth_fixed": True}
    config_edit(config)
    path.write_text(json.dumps(config))
    return expected_B, expected_delta


def _keep(config):
    pass


def test_pre_fix_unsloth_adapter_is_repacked_on_load(tmp_path, patched_load):
    expected_B, expected_delta = _legacy_adapter(tmp_path, _keep)
    loaded = PeftModel.from_pretrained(_base(), str(tmp_path))
    for name, value in _expert_lora_B(loaded).items():
        assert torch.equal(value, expected_B[name]), name
    for name, value in _delta(loaded).items():
        assert torch.allclose(value, expected_delta[name], atol = 1e-6), name


def test_repacked_adapter_saves_rank_major_and_reloads_unchanged(tmp_path, patched_load, monkeypatch):
    expected_B, _ = _legacy_adapter(tmp_path / "old", _keep)
    loaded = PeftModel.from_pretrained(_base(), str(tmp_path / "old"))
    monkeypatch.setattr(PeftModel, "save_pretrained", PeftModel.save_pretrained)
    assert MU._patch_peft_save_pretrained_for_moe_layout()
    loaded.save_pretrained(str(tmp_path / "new"))
    config = json.loads((tmp_path / "new" / "adapter_config.json").read_text())
    assert config[MU.FUSED_EXPERT_LORA_LAYOUT_KEY] == MU.LORA_B_LAYOUT_RANK_MAJOR
    again = PeftModel.from_pretrained(_base(), str(tmp_path / "new"))
    for name, value in _expert_lora_B(again).items():
        assert torch.equal(value, expected_B[name]), name


@pytest.mark.parametrize("edit", [
    lambda c: c.update({MU.FUSED_EXPERT_LORA_LAYOUT_KEY: MU.LORA_B_LAYOUT_RANK_MAJOR}),
    lambda c: c.update(auto_mapping = None),
], ids = ["rank_major_marker", "not_saved_by_unsloth"])
def test_rank_major_adapters_are_left_alone(tmp_path, patched_load, edit):
    _legacy_adapter(tmp_path, edit)
    on_disk = {k: v for k, v in load_file(str(tmp_path / "adapter_model.safetensors")).items() if "lora_B" in k}
    loaded = PeftModel.from_pretrained(_base(), str(tmp_path))
    got = sorted(_expert_lora_B(loaded).values(), key = lambda t: (t.shape, t.sum().item()))
    want = sorted(on_disk.values(), key = lambda t: (t.shape, t.sum().item()))
    assert all(torch.equal(a, b) for a, b in zip(got, want))


def test_explicit_layout_env_keeps_the_callers_setting(tmp_path, patched_load, monkeypatch):
    _legacy_adapter(tmp_path, _keep)
    monkeypatch.setenv("UNSLOTH_MOE_LORA_B_LAYOUT", MU.LORA_B_LAYOUT_GROUPED_BY_EXPERT)
    on_disk = sorted(
        (v for k, v in load_file(str(tmp_path / "adapter_model.safetensors")).items() if "lora_B" in k),
        key = lambda t: (t.shape, t.sum().item()),
    )
    loaded = PeftModel.from_pretrained(_base(), str(tmp_path))
    got = sorted(_expert_lora_B(loaded).values(), key = lambda t: (t.shape, t.sum().item()))
    assert all(torch.equal(a, b) for a, b in zip(got, on_disk))


def test_explicit_grouped_marker_is_repacked(tmp_path, patched_load):
    expected_B, _ = _legacy_adapter(
        tmp_path, lambda c: c.update({MU.FUSED_EXPERT_LORA_LAYOUT_KEY: MU.LORA_B_LAYOUT_GROUPED_BY_EXPERT}),
    )
    loaded = PeftModel.from_pretrained(_base(), str(tmp_path))
    for name, value in _expert_lora_B(loaded).items():
        assert torch.equal(value, expected_B[name]), name


def test_inference_is_limited_to_families_the_old_forward_owned():
    assert MU.saved_lora_b_layouts({"target_parameters": ["experts.gate_up_proj"], "auto_mapping": {"unsloth_fixed": True}}) == {
        None: MU.LORA_B_LAYOUT_GROUPED_BY_EXPERT,
    }
    upstream = type("Qwen3MoeExperts", (torch.nn.Module,), {"__module__": "transformers.models.qwen3_moe.modeling_qwen3_moe"})
    assert MU._experts_family(upstream()) == "qwen3_moe"

    compiled = type("Qwen3MoeExperts", (torch.nn.Module,), {"__module__": "unsloth_compiled_module_qwen3_moe"})
    assert MU._experts_family(compiled()) == "qwen3_moe"

    class OlmoeLikeExperts(torch.nn.Module):
        pass
    assert MU._experts_family(OlmoeLikeExperts()) not in MU._LEGACY_GROUPED_LORA_B_FAMILIES


def test_adapter_without_fused_experts_never_reads_its_config(tmp_path, patched_load, monkeypatch):
    model = get_peft_model(_base(), LoraConfig(r = R, target_modules = ["q_proj"]))
    model.save_pretrained(str(tmp_path))

    reads = []
    monkeypatch.setattr(MU, "_read_adapter_config", lambda *args: reads.append(args) or {})
    PeftModel.from_pretrained(_base(), str(tmp_path))
    assert reads == []


def test_current_save_with_a_failed_marker_write_is_not_mistaken_for_legacy(tmp_path, patched_load, monkeypatch):
    model = get_peft_model(_base(), LoraConfig(r = R, lora_alpha = 2 * R, target_parameters = EXPERTS, task_type = "CAUSAL_LM"))
    model.peft_config["default"].auto_mapping = {"base_model_class": "Qwen3MoeForCausalLM", "unsloth_fixed": True}
    g = torch.Generator().manual_seed(2)
    for name, p in model.named_parameters():
        if "lora_B" in name:
            with torch.no_grad():
                p.copy_(torch.randn(p.shape, generator = g))
    expected_B = _expert_lora_B(model)
    monkeypatch.setattr(PeftModel, "save_pretrained", PeftModel.save_pretrained)
    assert MU._patch_peft_save_pretrained_for_moe_layout()

    def fail(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(MU, "write_fused_expert_lora_layout", fail)
    model.save_pretrained(str(tmp_path))
    config = json.loads((tmp_path / "adapter_config.json").read_text())
    assert MU.FUSED_EXPERT_LORA_LAYOUT_KEY not in config
    assert config["auto_mapping"][MU.FUSED_EXPERT_LORA_LAYOUT_KEY] == MU.LORA_B_LAYOUT_RANK_MAJOR
    loaded = PeftModel.from_pretrained(_base(), str(tmp_path))
    for name, value in _expert_lora_B(loaded).items():
        assert torch.equal(value, expected_B[name]), name


def test_hub_reread_uses_the_options_peft_was_given(monkeypatch):
    import huggingface_hub
    calls = []

    def fake_download(repo_id, filename, subfolder = None, revision = None, cache_dir = None,
                      token = None, local_files_only = False):
        calls.append(dict(cache_dir = cache_dir, revision = revision, token = token))
        raise FileNotFoundError
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    with pytest.raises(FileNotFoundError):
        MU._read_adapter_config("org/legacy-adapter", {
            "cache_dir": "/custom/cache", "revision": "abc", "use_auth_token": "hf_x", "is_trainable": True,
        })
    assert calls[-1]["cache_dir"] == "/custom/cache" and calls[-1]["revision"] == "abc"
    assert calls[-1]["token"] == "hf_x" and "is_trainable" not in calls[-1]


def test_offloaded_lora_b_is_refused_not_half_repacked(tmp_path, patched_load, monkeypatch):
    _legacy_adapter(tmp_path, _keep)
    monkeypatch.setenv("UNSLOTH_MOE_LORA_B_LAYOUT", MU.LORA_B_LAYOUT_GROUPED_BY_EXPERT)
    loaded = PeftModel.from_pretrained(_base(), str(tmp_path))
    wrappers = [m for m in loaded.modules() if getattr(m, "parameter_name", None) in ("gate_up_proj", "down_proj")]
    before = {id(m): m.lora_B["default"].weight.detach().clone() for m in wrappers[1:]}
    offloaded = wrappers[0].lora_B["default"]
    offloaded.weight = torch.nn.Parameter(torch.empty_like(offloaded.weight, device = "meta"))
    config = json.loads((tmp_path / "adapter_config.json").read_text())
    with pytest.raises(RuntimeError, match = "offloaded"):
        MU.convert_legacy_lora_b(loaded, "default", config)
    for m in wrappers[1:]:
        assert torch.equal(m.lora_B["default"].weight, before[id(m)])
