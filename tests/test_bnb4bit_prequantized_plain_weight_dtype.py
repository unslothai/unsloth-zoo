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

"""Unquantized weights of a prequantized bnb-4bit checkpoint must load in the requested dtype.

transformers >= 5.21 registers `Bnb4bitDeserialize` for every `weight` key. An unquantized weight (embeddings,
lm_head, norms, skipped Linears) reaches it alone and is passed through, and the op's `hf_quantizer` makes
`HfQuantizer.get_new_dtype` return None, so the weight keeps its checkpoint dtype. transformers 5.17 cast it to
the requested dtype (or to float32 for `_keep_in_fp32_modules`). A real `from_pretrained` runs on CPU on a tiny
hand-built prequantized Llama checkpoint."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("bitsandbytes", reason = "requires bitsandbytes")
_core_model_loading = pytest.importorskip(
    "transformers.core_model_loading",
    reason = "requires transformers v5 core_model_loading",
)
if not hasattr(_core_model_loading.WeightConverter, "target_patterns"):
    pytest.skip("transformers.core_model_loading is a transformers<5 compat stub", allow_module_level = True)

import bitsandbytes as bnb
from safetensors.torch import load_file, save_file
from transformers import BitsAndBytesConfig, LlamaConfig, LlamaForCausalLM
from transformers.core_model_loading import convert_and_load_state_dict_in_model
from transformers.modeling_utils import LoadStateDictConfig
from transformers.quantizers.quantizer_bnb_4bit import Bnb4BitHfQuantizer

from unsloth_zoo.temporary_patches import moe_utils_bnb4bit as mb

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")


@pytest.fixture(scope = "module", autouse = True)
def _patched():
    mb.patch_bnb4bit_quantizer_weight_conversions()
    # getattr so the test runs (and fails) on a tree without the fix.
    getattr(mb, "patch_bnb4bit_prequantized_plain_weight_dtype", lambda: None)()


class _KeepNormsFp32(LlamaForCausalLM):
    # gpt-oss style: transformers keeps these in float32 on a float16 request.
    _keep_in_fp32_modules = ["input_layernorm", "post_attention_layernorm", "norm"]


def _checkpoint(root, ckpt_dtype):
    """Tiny Llama with nf4 (double quant) attention; mlp / embed / norms / lm_head unquantized in `ckpt_dtype`."""
    cfg = LlamaConfig(
        vocab_size = 128, hidden_size = 64, intermediate_size = 128, num_hidden_layers = 2,
        num_attention_heads = 4, num_key_value_heads = 2, max_position_embeddings = 64,
        tie_word_embeddings = False,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).float()
    dense = root / "dense"
    model.save_pretrained(dense)
    sd = load_file(str(dense / "model.safetensors"))
    out, quantized = {}, set()
    for key, value in sd.items():
        module = key.rsplit(".", 1)[0]
        if key.endswith(".weight") and module.rsplit(".", 1)[-1] in ATTN:
            packed, qs = bnb.functional.quantize_4bit(
                value.to(ckpt_dtype).float().contiguous(), blocksize = 64, quant_type = "nf4",
                compress_statistics = True, quant_storage = torch.uint8,
            )
            out[key] = packed
            out.update({f"{key}.{k}": v for k, v in qs.as_dict(packed = True).items()})
            quantized.add(module)
        else:
            out[key] = value.to(ckpt_dtype).contiguous()
    skip = sorted(
        n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and n not in quantized
    )
    qdir = root / "q"
    qdir.mkdir()
    save_file(out, str(qdir / "model.safetensors"), metadata = {"format": "pt"})
    cfg.dtype = ckpt_dtype
    cfg.quantization_config = BitsAndBytesConfig(
        load_in_4bit = True, bnb_4bit_quant_type = "nf4", bnb_4bit_use_double_quant = True,
        bnb_4bit_compute_dtype = torch.bfloat16, llm_int8_skip_modules = skip,
    ).to_dict()
    cfg.save_pretrained(qdir)
    return qdir, out


def _check(model, raw, expected):
    """Every unquantized floating param / persistent buffer has its expected dtype; quantized storage untouched."""
    n_q = n_plain = 0
    persistent = {n for n, _ in model.named_buffers() if n in model.state_dict()}
    for name, tensor in list(model.named_parameters()) + [
        (n, b) for n, b in model.named_buffers() if n in persistent
    ]:
        if isinstance(tensor, bnb.nn.Params4bit):
            n_q += 1
            assert tensor.dtype == torch.uint8, name
            assert torch.equal(tensor.data.cpu(), raw[name]), name
            qs = tensor.quant_state
            assert qs.code.dtype == torch.float32, name
            assert qs.absmax.dtype == raw[f"{name}.absmax"].dtype, name
            assert torch.equal(qs.absmax.cpu(), raw[f"{name}.absmax"]), name
            assert qs.state2.absmax.dtype == torch.float32, name
            continue
        if not tensor.is_floating_point():
            continue
        n_plain += 1
        want = expected(name)
        assert tensor.dtype == want, f"{name}: {tensor.dtype}, expected {want}"
        assert torch.equal(tensor.detach().cpu(), raw[name].to(want)), f"{name}: values differ from the cast checkpoint"
    assert n_q == 2 * len(ATTN) and n_plain > 0


@pytest.mark.parametrize("req", ["bf16", "fp16", "fp32"])
@pytest.mark.parametrize("ckpt", ["fp32", "bf16", "fp16"])
def test_plain_weights_load_in_requested_dtype(tmp_path, ckpt, req):
    qdir, raw = _checkpoint(tmp_path, DTYPES[ckpt])
    model = LlamaForCausalLM.from_pretrained(qdir, dtype = DTYPES[req], device_map = "cpu")
    _check(model, raw, lambda name: DTYPES[req])
    if req != "fp16":  # CPU fp16 matmul coverage varies across torch builds
        with torch.no_grad():
            logits = model(input_ids = torch.arange(8).view(1, 8)).logits
        assert torch.isfinite(logits).all()


@pytest.mark.parametrize("ckpt", ["fp32", "bf16"])
def test_keep_in_fp32_modules_stay_float32(tmp_path, ckpt):
    qdir, raw = _checkpoint(tmp_path, DTYPES[ckpt])
    model = _KeepNormsFp32.from_pretrained(qdir, dtype = torch.float16, device_map = "cpu")
    _check(model, raw, lambda name: torch.float32 if name.endswith("norm.weight") else torch.float16)


def test_unquantized_fused_expert_passthrough_loads_in_slot_dtype():
    """gpt-oss style fused experts with no model converter go through `_ExpertDeserialize`, which passes a
    lone dense tensor through like `Bnb4bitDeserialize`."""

    class _Model(torch.nn.Module):
        base_model_prefix = ""

        def __init__(self):
            super().__init__()
            self.config = LlamaConfig(hidden_size = 8, num_attention_heads = 2, num_key_value_heads = 2)
            self.experts = torch.nn.Module()
            self.experts.gate_up_proj = torch.nn.Parameter(torch.zeros(2, 8, 4, dtype = torch.bfloat16))

    quantizer = Bnb4BitHfQuantizer(BitsAndBytesConfig(load_in_4bit = True), pre_quantized = True)
    mapping = mb._bnb4bit_expert_weight_conversions(quantizer)
    model = _Model()
    sd = {"experts.gate_up_proj": torch.full((2, 8, 4), 0.5, dtype = torch.float32)}
    config = LoadStateDictConfig(
        device_map = {"": "cpu"}, hf_quantizer = quantizer, dtype = torch.bfloat16, weight_mapping = mapping,
    )
    info, _ = convert_and_load_state_dict_in_model(model, sd, config)
    assert not info.conversion_errors, info.conversion_errors
    loaded = model.experts.gate_up_proj
    assert loaded.dtype == torch.bfloat16, loaded.dtype
    assert torch.equal(loaded.float(), torch.full((2, 8, 4), 0.5))
