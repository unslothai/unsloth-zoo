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

"""Pre-quantized Llama-4 checkpoints storing MoE experts fused (split gate / up / down).

transformers 5 swaps in SequentialLlama4TextExperts and fails with
`"normal_kernel_cuda" not implemented for 'Byte'`. Tiny local fixtures, CPU only.
"""

from __future__ import annotations

import json
import os

import pytest

torch = pytest.importorskip("torch")

_core_model_loading = pytest.importorskip(
    "transformers.core_model_loading",
    reason = "requires transformers v5 core_model_loading",
)
if not hasattr(_core_model_loading.WeightConverter, "target_patterns"):
    pytest.skip("transformers.core_model_loading is a transformers<5 compat stub", allow_module_level = True)

pytest.importorskip("bitsandbytes", reason = "requires bitsandbytes")
from unsloth_zoo.stubs.bitsandbytes_stub import real_bitsandbytes_available

if not real_bitsandbytes_available():
    pytest.skip("bitsandbytes is the unsloth_zoo stub on this host", allow_module_level = True)

import bitsandbytes as bnb
from bitsandbytes.functional import QuantState, dequantize_4bit
from bitsandbytes.nn import Params4bit

try:
    from transformers import BitsAndBytesConfig, Llama4ForCausalLM, Llama4TextConfig
except ImportError:
    pytest.skip("transformers has no Llama-4", allow_module_level = True)
from safetensors import safe_open
from safetensors.torch import save_file

import unsloth_zoo.temporary_patches.moe_utils_bnb4bit as moe_bnb4bit
from unsloth_zoo.temporary_patches.common import TEMPORARY_PATCHES

_REAL_FUSED_FORWARD_AVAILABLE = moe_bnb4bit._fused_forward_available

NUM_EXPERTS, HIDDEN = 4, 64


_PATCH_MODULES = {
    moe_bnb4bit.__name__,
    "unsloth_zoo.temporary_patches.moe_experts_interface",
    "unsloth_zoo.temporary_patches.llama4_moe",
}


def _apply_bnb4bit_patches():
    applied = 0
    for patch in TEMPORARY_PATCHES:
        module = getattr(patch, "__module__", "")
        if module in _PATCH_MODULES:
            patch()
            applied += module == moe_bnb4bit.__name__
    if applied == 0:
        pytest.skip("the transformers v5 MoE bnb-4bit patches are not registered here")


def _quantize_into(out, name, weight):
    """Serialize like transformers' Linear4bit: packed bytes plus the aux tensors."""
    packed, quant_state = bnb.functional.quantize_4bit(
        weight, blocksize = 64, quant_type = "nf4",
        compress_statistics = True, quant_storage = torch.uint8,
    )
    out[name] = packed
    for key, value in quant_state.as_dict(packed = True).items():
        out[f"{name}.{key}"] = value


def _build_checkpoint(path, layout, inter = 128):
    """`layout`: "fused" (unsloth Scout upload) or "per_expert" (transformers)."""
    E, H, I = NUM_EXPERTS, HIDDEN, inter
    torch.manual_seed(0)
    config = Llama4TextConfig(
        vocab_size = 128, hidden_size = H, intermediate_size = I, intermediate_size_mlp = 256,
        num_hidden_layers = 2, num_attention_heads = 4, num_key_value_heads = 2, head_dim = 16,
        num_local_experts = E, num_experts_per_tok = 1, moe_layers = [0, 1],
        interleave_moe_layer_step = 1, max_position_embeddings = 256,
        attention_chunk_size = 128, use_qk_norm = False, no_rope_layers = [1, 1],
    )
    model = Llama4ForCausalLM(config).to(torch.bfloat16)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith(("experts.gate_up_proj", "experts.down_proj")):
                param.normal_(0, 0.02)

    out = {}
    for key, value in model.state_dict().items():
        value = value.contiguous()
        if key.endswith("experts.gate_up_proj"):
            base = key[: -len("gate_up_proj")]
            if layout == "fused":
                _quantize_into(out, base + "gate_proj.weight", value[..., :I].reshape(E * H, I))
                _quantize_into(out, base + "up_proj.weight", value[..., I:].reshape(E * H, I))
            else:
                for e in range(E):
                    _quantize_into(out, f"{base}{e}.gate_proj.weight", value[e, :, :I].T.contiguous())
                    _quantize_into(out, f"{base}{e}.up_proj.weight", value[e, :, I:].T.contiguous())
        elif key.endswith("experts.down_proj"):
            base = key[: -len("down_proj")]
            if layout == "fused":
                _quantize_into(out, base + "down_proj.weight", value.reshape(E * I, H))
            else:
                for e in range(E):
                    _quantize_into(out, f"{base}{e}.down_proj.weight", value[e].T.contiguous())
        elif value.ndim == 2 and key.endswith(".weight") and not any(
            skip in key for skip in ("lm_head", "embed_tokens", "router")
        ):
            _quantize_into(out, key, value)
        else:
            out[key] = value

    os.makedirs(path, exist_ok = True)
    save_file(out, os.path.join(path, "model.safetensors"), metadata = {"format": "pt"})
    config_dict = config.to_dict()
    config_dict["architectures"] = ["Llama4ForCausalLM"]
    config_dict["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit = True, bnb_4bit_quant_type = "nf4", bnb_4bit_use_double_quant = True,
        bnb_4bit_compute_dtype = torch.bfloat16, llm_int8_skip_modules = ["lm_head", "router"],
    ).to_dict()
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config_dict, f)
    return str(path)


def _checkpoint_dequantized(path, name):
    with safe_open(os.path.join(path, "model.safetensors"), "pt") as f:
        aux = {k[len(name) + 1:]: f.get_tensor(k) for k in f.keys() if k.startswith(name + ".")}
        packed = f.get_tensor(name)
    return dequantize_4bit(packed, QuantState.from_dict(aux, device = "cpu"))


def _load(path):
    return Llama4ForCausalLM.from_pretrained(path, device_map = "cpu", output_loading_info = True)


@pytest.fixture(autouse = True)
def _fused_forward_present(monkeypatch):
    monkeypatch.setattr(moe_bnb4bit, "_fused_forward_available", lambda name: True)


@pytest.fixture(scope = "module")
def fused_checkpoint(tmp_path_factory):
    return _build_checkpoint(tmp_path_factory.mktemp("llama4_fused"), "fused")


@pytest.mark.parametrize("inter", [128, 96], ids = ["block_aligned", "block_unaligned"])
def test_fused_checkpoint_loads_into_the_fused_stacks(tmp_path, inter):
    """Aligned widths splice packed bytes (bit-exact); unaligned ones requantize."""
    _apply_bnb4bit_patches()
    path = _build_checkpoint(tmp_path / "ckpt", "fused", inter = inter)
    model, info = _load(path)

    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        assert not info.get(key), f"{key}: {sorted(info[key])[:5]}"
    for index, layer in enumerate(model.model.layers):
        experts = layer.feed_forward.experts
        assert type(experts).__name__ == "Llama4TextExperts"
        prefix = f"model.layers.{index}.feed_forward.experts."
        gate = _checkpoint_dequantized(path, prefix + "gate_proj.weight").reshape(NUM_EXPERTS, HIDDEN, inter)
        up = _checkpoint_dequantized(path, prefix + "up_proj.weight").reshape(NUM_EXPERTS, HIDDEN, inter)
        down = _checkpoint_dequantized(path, prefix + "down_proj.weight").reshape(NUM_EXPERTS, inter, HIDDEN)

        gate_up_proj, down_proj = experts.gate_up_proj, experts.down_proj
        assert isinstance(gate_up_proj, Params4bit) and isinstance(down_proj, Params4bit)
        assert tuple(gate_up_proj._original_shape) == (NUM_EXPERTS, HIDDEN, 2 * inter)
        assert tuple(down_proj._original_shape) == (NUM_EXPERTS, inter, HIDDEN)
        got_gate_up = dequantize_4bit(gate_up_proj.data, gate_up_proj.quant_state)
        assert torch.equal(dequantize_4bit(down_proj.data, down_proj.quant_state), down)
        if inter % 64 == 0:
            assert torch.equal(got_gate_up, torch.cat([gate, up], dim = -1))
        else:
            want = torch.cat([gate, up], dim = -1).float()
            assert (got_gate_up.float() - want).abs().max() <= 0.25 * want.abs().max()


def test_without_the_layout_decision_the_fused_checkpoint_still_fails(fused_checkpoint, monkeypatch):
    """Unknown layout: transformers' swap runs and the load still fails."""
    _apply_bnb4bit_patches()
    monkeypatch.setattr(moe_bnb4bit, "_checkpoint_expert_layout", lambda checkpoint_files: None)
    with pytest.raises(NotImplementedError, match = "Byte"):
        _load(fused_checkpoint)


def test_without_the_fused_forward_the_transformers_swap_still_runs(fused_checkpoint, monkeypatch):
    """No fused-stack forward: keeping fused would only defer the failure, so swap."""
    _apply_bnb4bit_patches()
    monkeypatch.setattr(moe_bnb4bit, "_fused_forward_available", lambda name: False)
    with pytest.raises(NotImplementedError, match = "Byte"):
        _load(fused_checkpoint)


def test_fused_forward_available_follows_the_llama4_patch_module(monkeypatch):
    import importlib.util

    seen = []
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: seen.append(name) or None)
    assert _REAL_FUSED_FORWARD_AVAILABLE("Llama4TextExperts") is False
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: seen.append(name) or object())
    assert _REAL_FUSED_FORWARD_AVAILABLE("Llama4TextExperts") is True
    assert _REAL_FUSED_FORWARD_AVAILABLE("SomeOtherExperts") is True
    assert seen == ["unsloth_zoo.temporary_patches.llama4_moe"] * 2


def test_per_expert_checkpoint_keeps_the_transformers_swap(tmp_path):
    """Per-expert checkpoints still get SequentialLlama4TextExperts."""
    _apply_bnb4bit_patches()
    model, info = _load(_build_checkpoint(tmp_path / "ckpt", "per_expert"))
    assert not info.get("missing_keys") and not info.get("unexpected_keys")
    for layer in model.model.layers:
        assert type(layer.feed_forward.experts).__name__ == "SequentialLlama4TextExperts"


def test_fused_load_never_mutates_the_global_swap_table(fused_checkpoint, monkeypatch):
    """Concurrent loads must still see transformers' full swap table."""
    import transformers.quantizers.base as quantizers_base
    from transformers.quantizers.quantizer_bnb_4bit import Bnb4BitHfQuantizer

    _apply_bnb4bit_patches()
    table = quantizers_base.MODULES_TO_PATCH_FOR_QUANTIZATION
    before = dict(table)
    seen = []
    original = Bnb4BitHfQuantizer._process_model_before_weight_loading

    def spy(self, model, *args, **kwargs):
        seen.append(("Llama4TextExperts" in table, "_convert_model_for_quantization" in vars(self)))
        return original(self, model, *args, **kwargs)

    monkeypatch.setattr(Bnb4BitHfQuantizer, "_process_model_before_weight_loading", spy)
    model, info = _load(fused_checkpoint)
    assert seen == [(True, True)]
    assert quantizers_base.MODULES_TO_PATCH_FOR_QUANTIZATION is table and table == before
    assert type(model.model.layers[0].feed_forward.experts).__name__ == "Llama4TextExperts"
    assert "_convert_model_for_quantization" not in vars(model.hf_quantizer)


def test_without_the_stock_convert_method_the_transformers_swap_runs(fused_checkpoint, monkeypatch):
    """Non-stock convert method: global table untouched, transformers' swap runs."""
    import transformers.quantizers.base as quantizers_base

    _apply_bnb4bit_patches()
    table = quantizers_base.MODULES_TO_PATCH_FOR_QUANTIZATION
    before = dict(table)
    monkeypatch.setattr(moe_bnb4bit, "_convert_without", lambda *args: None)
    with pytest.raises(NotImplementedError, match = "Byte"):
        _load(fused_checkpoint)
    assert table == before


def test_unpacked_expert_slot_gets_dequantized_values(fused_checkpoint, monkeypatch):
    """Float expert slots get dequantized values, never packed bytes."""
    _apply_bnb4bit_patches()
    monkeypatch.setattr(moe_bnb4bit, "replace_expert_params_with_bnb_params", lambda model, **kwargs: model)
    model, info = _load(fused_checkpoint)
    assert not info.get("missing_keys") and not info.get("unexpected_keys")
    experts = model.model.layers[0].feed_forward.experts
    assert not isinstance(experts.gate_up_proj, Params4bit)
    prefix = "model.layers.0.feed_forward.experts."
    gate = _checkpoint_dequantized(fused_checkpoint, prefix + "gate_proj.weight").reshape(NUM_EXPERTS, HIDDEN, -1)
    up = _checkpoint_dequantized(fused_checkpoint, prefix + "up_proj.weight").reshape(NUM_EXPERTS, HIDDEN, -1)
    want = torch.cat([gate, up], dim = -1).to(experts.gate_up_proj.dtype)
    assert torch.equal(experts.gate_up_proj.data, want)


def test_checkpoint_expert_layout(tmp_path):
    def write(name, keys):
        path = str(tmp_path / f"{name}.safetensors")
        save_file({k: torch.zeros(1) for k in keys}, path)
        return path

    fused = write("fused", ["model.layers.0.feed_forward.experts.gate_proj.weight"])
    fused_stack = write("fused_stack", ["model.layers.0.feed_forward.experts.gate_up_proj"])
    per_expert = write("per_expert", ["model.layers.0.feed_forward.experts.3.gate_proj.weight"])
    shared = write("shared", ["model.layers.0.mlp.shared_experts.gate_proj.weight"])
    layout = moe_bnb4bit._checkpoint_expert_layout
    assert layout([fused]) == "fused"
    assert layout([fused_stack]) == "fused"
    assert layout([per_expert]) == "per_expert"
    assert layout([shared]) is None
    assert layout([shared, fused]) == "fused"
    assert layout(None) is None
    assert layout([str(tmp_path / "pytorch_model.bin")]) is None


def test_split_converters_claim_only_fused_expert_keys():
    converters = moe_bnb4bit._bnb4bit_split_fused_expert_conversions()
    renamed = {}
    for key in (
        "model.layers.0.feed_forward.experts.gate_proj.weight",
        "model.layers.0.feed_forward.experts.up_proj.weight.absmax",
        "model.layers.0.feed_forward.experts.down_proj.weight",
        "model.layers.0.feed_forward.experts.3.gate_proj.weight",
        "model.layers.0.feed_forward.shared_expert.down_proj.weight",
        "model.layers.0.mlp.shared_experts.gate_proj.weight",
    ):
        for converter in converters:
            new_key, pattern = converter.rename_source_key(key)
            if pattern is not None:
                renamed[key] = new_key
                break
    assert renamed == {
        "model.layers.0.feed_forward.experts.gate_proj.weight": "model.layers.0.feed_forward.experts.gate_up_proj",
        "model.layers.0.feed_forward.experts.up_proj.weight.absmax": "model.layers.0.feed_forward.experts.gate_up_proj",
        "model.layers.0.feed_forward.experts.down_proj.weight": "model.layers.0.feed_forward.experts.down_proj",
    }
