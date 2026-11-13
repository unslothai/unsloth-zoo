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

"""Granite-4 hybrid: float32 norm weights (high precision layernorm) must not feed float32 into 16-bit Linears.

CPU bf16 stands in for the GPU bf16 / fp16 mismatch ("float != BFloat16" at mamba in_proj without autocast).
"""

import inspect

import pytest
import torch

gm = pytest.importorskip("transformers.models.granitemoehybrid.modeling_granitemoehybrid")
from unsloth_zoo.temporary_patches import granitemoehybrid as patch
from unsloth_zoo.temporary_patches.common import TEMPORARY_PATCHES

NORMS = ("GraniteMoeHybridRMSNorm", "GraniteMoeHybridRMSNormGated")


def _stock(name):
    forward = getattr(gm, name).forward
    return getattr(forward, "_unsloth_original_forward", forward)


def _set_stock(monkeypatch):
    for name in NORMS: monkeypatch.setattr(getattr(gm, name), "forward", _stock(name))


def _apply_granite_patches(monkeypatch):
    # Every patch this module registers, applied through the registry rather than by name.
    _set_stock(monkeypatch)
    monkeypatch.setattr(gm.GraniteMoeHybridDecoderLayer, "forward", gm.GraniteMoeHybridDecoderLayer.forward)
    for fn in TEMPORARY_PATCHES:
        if fn.__module__ == patch.__name__: fn()


def _norm(cls, hidden_size):
    # transformers 5.23 normalises the gated norm per group (`group_size` is required); two groups per row make the
    # rewritten forward match stock only if it keeps the grouping.
    if "group_size" in inspect.signature(cls.__init__).parameters:
        return cls(hidden_size, group_size = hidden_size // 2)
    return cls(hidden_size)


def _tiny_model(dtype, float32_norms):
    config = gm.GraniteMoeHybridConfig(
        vocab_size = 64, hidden_size = 32, intermediate_size = 32, shared_intermediate_size = 48,
        num_hidden_layers = 2, layer_types = ["mamba", "attention"], num_attention_heads = 4,
        num_key_value_heads = 2, num_local_experts = 4, num_experts_per_tok = 2,
        mamba_n_heads = 8, mamba_d_head = 8, mamba_n_groups = 1, mamba_d_state = 16, mamba_chunk_size = 4,
    )
    torch.manual_seed(0)
    model = gm.GraniteMoeHybridForCausalLM(config).to(dtype)
    if float32_norms:
        # What Unsloth does under UNSLOTH_HIGH_PRECISION_LAYERNORM.
        for module in model.modules():
            if type(module).__name__ in NORMS: module.float()
    return model


def _lora_in_proj(model):
    peft = pytest.importorskip("peft")
    config = peft.LoraConfig(r = 4, target_modules = ["in_proj"], init_lora_weights = False)
    return peft.get_peft_model(model, config)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_16bit_model_float32_norms_lora_backward_without_autocast(dtype, monkeypatch):
    _apply_granite_patches(monkeypatch)
    model = _lora_in_proj(_tiny_model(dtype, float32_norms = True)).train()
    assert model.base_model.model.model.layers[0].mamba.norm.weight.dtype == torch.float32
    ids = torch.randint(0, 64, (2, 8))
    out = model(input_ids = ids, labels = ids, use_cache = False)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    grads = [p.grad for n, p in model.named_parameters() if "lora_" in n]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("name", NORMS)
def test_norm_casts_only_float32_weight_16bit_input_no_autocast(name, monkeypatch):
    _apply_granite_patches(monkeypatch)
    cls = getattr(gm, name)
    norm = _norm(cls, 16)
    stock = _stock(name)
    x = torch.randn(3, 16)
    for weight_dtype, input_dtype in [(torch.float32, torch.float32), (torch.bfloat16, torch.bfloat16),
                                      (torch.float32, torch.bfloat16), (torch.float32, torch.float16)]:
        norm.to(weight_dtype)
        xi = x.to(input_dtype)
        got, ref = norm(xi), stock(norm, xi)
        if weight_dtype == torch.float32 and input_dtype != torch.float32:
            assert ref.dtype == torch.float32 and got.dtype == input_dtype
            assert torch.equal(got, ref.to(input_dtype))
        else:
            assert got.dtype == ref.dtype and torch.equal(got, ref)
        with torch.autocast("cpu", dtype = torch.bfloat16):
            got, ref = norm(xi), stock(norm, xi)
        assert got.dtype == ref.dtype and torch.equal(got, ref)


def _logits(model, ids):
    with torch.no_grad(): return model(input_ids = ids, use_cache = False).logits


def test_autocast_outputs_unchanged(monkeypatch):
    model = _tiny_model(torch.bfloat16, float32_norms = True).eval()
    ids = torch.randint(0, 64, (2, 8))
    _set_stock(monkeypatch)
    with torch.autocast("cpu", dtype = torch.bfloat16): expected = _logits(model, ids)
    _apply_granite_patches(monkeypatch)
    assert all(getattr(gm, n).forward is not _stock(n) for n in NORMS)
    with torch.autocast("cpu", dtype = torch.bfloat16): got = _logits(model, ids)
    assert got.dtype == expected.dtype and torch.equal(got, expected)


@pytest.mark.parametrize("float32_norms", [True, False])
def test_float32_model_unchanged(float32_norms, monkeypatch):
    model = _tiny_model(torch.float32, float32_norms).eval()
    ids = torch.randint(0, 64, (2, 8))
    _set_stock(monkeypatch)
    expected = _logits(model, ids)
    _apply_granite_patches(monkeypatch)
    got = _logits(model, ids)
    assert got.dtype == torch.float32 and torch.equal(got, expected)


@pytest.mark.parametrize("name", NORMS)
def test_compiler_reads_the_patched_source(name, monkeypatch):
    # create_standalone_class copies inspect.getsource(_unwrap_undecorated_method(cls.forward)) into the compiled module.
    from unsloth_zoo.compiler import _unwrap_undecorated_method
    _apply_granite_patches(monkeypatch)
    cls = getattr(gm, name)
    source = inspect.getsource(_unwrap_undecorated_method(cls.forward, cls.__qualname__))
    assert source.startswith("def forward")
    assert "torch.is_autocast_enabled(output.device.type)" in source
    assert "output = output.to(input_dtype)" in source


def test_patch_is_idempotent_and_noops_without_the_model(monkeypatch):
    _apply_granite_patches(monkeypatch)
    installed = [getattr(gm, n).forward for n in NORMS]
    patch.patch_GraniteMoeHybridRMSNorm_dtype()
    assert [getattr(gm, n).forward for n in NORMS] == installed
    monkeypatch.delattr(gm, "GraniteMoeHybridRMSNormGated")
    monkeypatch.setattr(gm.GraniteMoeHybridRMSNorm, "forward", _stock("GraniteMoeHybridRMSNorm"))
    patch.patch_GraniteMoeHybridRMSNorm_dtype()
    assert gm.GraniteMoeHybridRMSNorm.forward._unsloth_granite_norm_dtype
    assert patch._rewrite_norm_source("def forward(self, x):\n    return x\n") is None
