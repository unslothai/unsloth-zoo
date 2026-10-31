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

"""A dense fused expert stored beside stale bnb quantization stats loads dense.

On transformers v5 the prequantized expert deserializer received a bf16
gate_up_proj plus leftover .absmax / .quant_map keys and called
Params4bit.from_prequantized on it, so the MoE forward read bf16 bytes as nf4.
transformers 4.x ignores the stray keys and keeps the expert exact.
"""
import os
import shutil

import pytest
import torch

STALE_STATS = (
    "absmax",
    "nested_absmax",
    "nested_quant_map",
    "quant_map",
    "quant_state.bitsandbytes__nf4",
)
LAYER = "model.layers.0.mlp.experts.gate_up_proj"


def _require_v5_moe_quantization():
    common = pytest.importorskip("unsloth_zoo.temporary_patches.common")
    if not common.is_transformers_v5_moe_quantization_available():
        pytest.skip("transformers 4.x: the v5 MoE expert deserializer is not registered")


def _deserializer(monkeypatch, recorded):
    _require_v5_moe_quantization()
    M = pytest.importorskip("unsloth_zoo.temporary_patches.moe_utils_bnb4bit")
    import transformers.quantizers.quantizers_utils as qutils

    class _FakeParams4bit:
        @classmethod
        def from_prequantized(cls, data, quantized_stats, **kwargs):
            recorded.append((data.dtype, sorted(quantized_stats)))
            return data

    monkeypatch.setattr(M, "Params4bit", _FakeParams4bit)
    monkeypatch.setattr(qutils, "get_module_from_name", lambda model, name: (torch.nn.Module(), "gate_up_proj"))
    converter = next(
        c for c in M._bnb4bit_expert_weight_conversions(hf_quantizer = None)
        if "gate_up_proj" in str(getattr(c, "target_patterns", ""))
    )
    return converter.operations[0]


def _inputs(weight):
    stats = {f"gate_up_proj.{k}": [torch.zeros(4, dtype = torch.uint8)] for k in STALE_STATS}
    return {"gate_up_proj": [weight], **stats}


def test_a_dense_expert_beside_stale_stats_stays_dense(monkeypatch):
    recorded = []
    op = _deserializer(monkeypatch, recorded)
    dense = torch.randn(4, 8, 64, dtype = torch.bfloat16)
    out = op.convert(_inputs(dense), model = None, full_layer_name = LAYER)
    assert list(out) == ["gate_up_proj"]
    assert out["gate_up_proj"] is dense, "the dense expert was replaced"
    assert recorded == [], "from_prequantized ran on a floating-point expert"


def test_a_packed_expert_still_deserializes(monkeypatch):
    recorded = []
    op = _deserializer(monkeypatch, recorded)
    packed = torch.zeros(1024, 1, dtype = torch.uint8)
    op.convert(_inputs(packed), model = torch.nn.Module(), full_layer_name = LAYER)
    assert recorded and recorded[0][0] == torch.uint8
    assert recorded[0][1] == sorted(f"gate_up_proj.{k}" for k in STALE_STATS)


@pytest.mark.parametrize("storage", [torch.bfloat16, torch.float16, torch.float32])
def test_a_packed_expert_in_float_quant_storage_still_deserializes(monkeypatch, storage):
    # bnb_4bit_quant_storage may be a float dtype: the payload is floating point
    # but flattened to [n, 1], so it is packed, not dense.
    recorded = []
    op = _deserializer(monkeypatch, recorded)
    packed = torch.zeros(512, 1, dtype = storage)
    op.convert(_inputs(packed), model = torch.nn.Module(), full_layer_name = LAYER)
    assert recorded and recorded[0][0] == storage, "a float-storage packed expert was kept as dense"


def test_a_lone_weight_passes_through(monkeypatch):
    recorded = []
    op = _deserializer(monkeypatch, recorded)
    lone = {"gate_up_proj": [torch.randn(4, 8, 64, dtype = torch.bfloat16)]}
    assert op.convert(lone, model = None, full_layer_name = LAYER) is lone
    assert recorded == []


def _has_cuda_bnb():
    if not torch.cuda.is_available():
        return False
    try:
        import bitsandbytes  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _has_cuda_bnb(), reason = "needs a CUDA GPU and bitsandbytes")
def test_tiny_gpt_oss_with_stale_expert_stats_loads_exact_experts(tmp_path):
    _require_v5_moe_quantization()
    import unsloth  # noqa: F401  (patches before transformers)
    from unsloth import FastLanguageModel
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    src = "trl-internal-testing/tiny-GptOssForCausalLM"
    ckpt = str(tmp_path / "bf16")
    ref = AutoModelForCausalLM.from_pretrained(src, dtype = torch.bfloat16)
    torch.manual_seed(0)
    with torch.no_grad():
        for name, param in ref.named_parameters():
            if "experts" in name or "router" in name:
                param.normal_(0.0, 0.05)
    ref.save_pretrained(ckpt)
    AutoTokenizer.from_pretrained(src).save_pretrained(ckpt)
    del ref
    original = load_file(os.path.join(ckpt, "model.safetensors"))

    load = dict(max_seq_length = 64, load_in_4bit = True, dtype = torch.bfloat16)
    model, tokenizer = FastLanguageModel.from_pretrained(ckpt, **load)
    saved = str(tmp_path / "saved_4bit")
    model.save_pretrained(saved)
    tokenizer.save_pretrained(saved)
    del model
    torch.cuda.empty_cache()

    tensors = load_file(os.path.join(saved, "model.safetensors"))
    stale = [n for n in tensors if ".experts.gate_up_proj." in n or ".experts.down_proj." in n]
    if not stale:
        pytest.skip("this save keeps dense experts, so there are no stats to leave stale")
    dense = str(tmp_path / "dense_with_stale_stats")
    os.makedirs(dense)
    for f in os.listdir(saved):
        if f != "model.safetensors":
            shutil.copy(os.path.join(saved, f), dense)
    for name in list(tensors):
        if name.endswith(("experts.gate_up_proj", "experts.down_proj")):
            tensors[name] = original[name]
    save_file(tensors, os.path.join(dense, "model.safetensors"), metadata = {"format": "pt"})

    model, _ = FastLanguageModel.from_pretrained(dense, **load)
    for i, layer in enumerate(model.model.layers):
        for name in ("gate_up_proj", "down_proj"):
            got = getattr(layer.mlp.experts, name)
            assert getattr(got, "quant_state", None) is None, (i, name, "dense expert carries a quant_state")
            want = original[f"model.layers.{i}.mlp.experts.{name}"]
            torch.testing.assert_close(got.detach().cpu(), want, rtol = 0, atol = 0)
