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

"""Unquantized weights of a prequantized bnb-8bit checkpoint must load in the requested dtype (transformers >= 5.21)."""

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

from unsloth_zoo.temporary_patches import moe_utils_bnb4bit as mb

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")


@pytest.fixture(scope = "module", autouse = True)
def _patched():
    getattr(mb, "patch_bnb8bit_prequantized_plain_weight_dtype", lambda: None)()


class _KeepNormsFp32(LlamaForCausalLM):
    _keep_in_fp32_modules = ["input_layernorm", "post_attention_layernorm", "norm"]


def _checkpoint(root, ckpt_dtype):
    """Tiny Llama with int8 (weight + SCB) attention; mlp / embed / norms / lm_head unquantized in `ckpt_dtype`."""
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
            cb, scb, _ = bnb.functional.int8_vectorwise_quant(value.to(ckpt_dtype).float().contiguous())
            out[key] = cb.contiguous()
            out[f"{module}.SCB"] = scb.float().contiguous()  # Linear8bitLt state-dict layout
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
    cfg.quantization_config = BitsAndBytesConfig(load_in_8bit = True, llm_int8_skip_modules = skip).to_dict()
    cfg.save_pretrained(qdir)
    return qdir, out


def _check(model, raw, expected):
    """Every unquantized floating param / persistent buffer has its expected dtype; int8 data and SCB untouched."""
    n_q = n_plain = 0
    persistent = {n for n, _ in model.named_buffers() if n in model.state_dict()}
    for name, tensor in list(model.named_parameters()) + [
        (n, b) for n, b in model.named_buffers() if n in persistent
    ]:
        if isinstance(tensor, bnb.nn.Int8Params):
            n_q += 1
            assert tensor.dtype == torch.int8, name
            assert torch.equal(tensor.data.cpu(), raw[name]), name
            scb = raw[f"{name.rsplit('.', 1)[0]}.SCB"]
            assert tensor.SCB.dtype == scb.dtype == torch.float32, name
            assert torch.equal(tensor.SCB.cpu(), scb), name
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


@pytest.mark.parametrize("ckpt", ["fp32", "bf16"])
def test_keep_in_fp32_modules_stay_float32(tmp_path, ckpt):
    qdir, raw = _checkpoint(tmp_path, DTYPES[ckpt])
    model = _KeepNormsFp32.from_pretrained(qdir, dtype = torch.float16, device_map = "cpu")
    _check(model, raw, lambda name: torch.float32 if name.endswith("norm.weight") else torch.float16)
