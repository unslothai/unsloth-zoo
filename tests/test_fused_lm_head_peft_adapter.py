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

"""Fused CE on an lm_head carrying a PEFT LoRA adapter (or a modules_to_save copy) must match lm_head.forward(): the
kernel used to read only lm_head.weight, so the adapter got no gradient and the loss was the base head's (CPU)."""

import inspect
import os
import textwrap

import pytest
import torch

peft = pytest.importorskip("peft")
gemma3 = pytest.importorskip("transformers.models.gemma3.modeling_gemma3")
from transformers.loss.loss_utils import ForCausalLMLoss  # noqa: E402
from unsloth_zoo import compiler  # noqa: E402
from unsloth_zoo.fused_losses import unsloth_fused_ce_loss, unsloth_fused_lm_head_loss  # noqa: E402
from unsloth_zoo.fused_losses.forward_install import _can_fuse_loss  # noqa: E402

V, H = 96, 32


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(V, H)
        self.proj = torch.nn.Linear(H, H)
        self.lm_head = torch.nn.Linear(H, V, bias = False)

    def forward(self, ids):
        return self.lm_head(self.proj(self.embed(ids)))


def _ids():
    return torch.randint(0, V, (2, 9), generator = torch.Generator().manual_seed(1))


def _wrap(dropout = 0.0, mts = False, **extra):
    torch.manual_seed(0)
    if mts:
        cfg = peft.LoraConfig(r = 4, target_modules = ["proj"], modules_to_save = ["lm_head"])
    else:
        cfg = peft.LoraConfig(r = 4, lora_alpha = 8, target_modules = ["proj", "lm_head"], lora_dropout = dropout,
                              init_lora_weights = False, **extra)
    model = peft.get_peft_model(Toy(), cfg).train()
    inner = model.base_model.model
    if mts:
        with torch.no_grad():
            inner.lm_head.modules_to_save["default"].weight.add_(0.1)
    return model, inner


def _grad(loss, params):
    return torch.autograd.grad(loss, params, allow_unused = True, materialize_grads = True)


def _losses(inner, params, ids):
    hidden = lambda: inner.proj(inner.embed(ids))  # noqa: E731
    torch.manual_seed(3)
    want = ForCausalLMLoss(inner.lm_head(hidden()).float(), ids, vocab_size = V)
    want_grads = _grad(want, params)
    torch.manual_seed(3)
    got = unsloth_fused_lm_head_loss(hidden(), inner.lm_head, ids, vocab_size = V, torch_compile = False)
    got_grads = _grad(got, params)
    return want, want_grads, got, got_grads


def _assert_match(inner, model, ids):
    params = [p for p in model.parameters() if p.requires_grad]
    want, want_grads, got, got_grads = _losses(inner, params, ids)
    torch.testing.assert_close(got, want)
    for g, w in zip(got_grads, want_grads):
        torch.testing.assert_close(g, w, rtol = 1e-4, atol = 1e-6)
    return params, got_grads


@pytest.mark.parametrize("dropout", [0.0, 0.3])
def test_lora_head_matches_its_forward(dropout):
    model, inner = _wrap(dropout = dropout)
    params, grads = _assert_match(inner, model, _ids())
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    lora_b = grads[names.index("base_model.model.lm_head.lora_B.default.weight")]
    assert lora_b.abs().sum() > 0


def test_merged_or_disabled_lora_head_reads_the_base_weight():
    model, inner = _wrap()
    ids = _ids()
    inner.proj.base_layer.weight.requires_grad_(True)  # disabling freezes every adapter: keep one grad to compare
    with model.disable_adapter():
        _assert_match(inner, model, ids)
    inner.lm_head.merge()
    _assert_match(inner, model, ids)


def test_modules_to_save_head_trains_the_copy():
    model, inner = _wrap(mts = True)
    params, grads = _assert_match(inner, model, _ids())
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert "base_model.model.lm_head.original_module.weight" not in names
    assert grads[names.index("base_model.model.lm_head.modules_to_save.default.weight")].abs().sum() > 0


def test_dora_head_is_refused_instead_of_dropped():
    model, inner = _wrap(use_dora = True)
    ids = _ids()
    with pytest.raises(NotImplementedError, match = "LM head"):
        unsloth_fused_lm_head_loss(inner.proj(inner.embed(ids)), inner.lm_head, ids, vocab_size = V,
                                   torch_compile = False)


def _fused_gemma3_forward():
    # Gemma 3 as in test_fused_lm_head_qat.py: the conftest's unsloth import leaves its forward stock.
    source = inspect.getsource(gemma3.Gemma3ForCausalLM.forward)
    source = compiler.fixup_fused_lm_head(source)
    source, _ = compiler.apply_fused_lm_head(source, "Gemma3ForCausalLM")
    assert "unsloth_fused_ce_loss" in source
    from _compiled_module_names import fused_loss_header_names

    scope = dict(vars(gemma3))
    scope.update(fused_loss_header_names())
    scope.update(
        os = os,
        UNSLOTH_ENABLE_CCE = False,
        HAS_CUT_CROSS_ENTROPY = False,
        UNSLOTH_COMPILE_DISABLE = True,
        unsloth_fused_ce_loss = unsloth_fused_ce_loss,
        _can_fuse_loss = _can_fuse_loss,
        unsloth_training_stance = lambda: None,
        EMPTY_LOGITS = torch.empty(0),
    )
    source = textwrap.dedent(source)
    exec(source[source.index("def forward"):], scope)
    return scope["forward"]


@pytest.mark.parametrize("softcap", [None, 30.0])
def test_compiled_forward_trains_the_lora_head(softcap, monkeypatch):
    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "0")
    monkeypatch.setenv("UNSLOTH_FUSED_CE_COMPILE_DISABLE", "1")
    config = gemma3.Gemma3TextConfig(
        vocab_size = V, hidden_size = H, intermediate_size = 64, num_hidden_layers = 1, num_attention_heads = 2,
        num_key_value_heads = 1, head_dim = 16, final_logit_softcapping = softcap, attn_implementation = "eager",
    )
    torch.manual_seed(0)
    model = peft.get_peft_model(
        gemma3.Gemma3ForCausalLM(config).float(),
        peft.LoraConfig(r = 4, lora_alpha = 8, target_modules = ["q_proj", "lm_head"], init_lora_weights = False),
    ).train()
    inner = model.base_model.model
    inner.loss_function = ForCausalLMLoss
    ids = _ids()
    params = [p for p in model.parameters() if p.requires_grad]

    want = model(input_ids = ids, labels = ids).loss
    want_grads = torch.autograd.grad(want, params)
    out = _fused_gemma3_forward()(inner, input_ids = ids, labels = ids)
    assert out.logits.numel() == 0  # the fused kernel ran
    got_grads = torch.autograd.grad(out.loss, params)

    torch.testing.assert_close(out.loss, want)
    for g, w in zip(got_grads, want_grads):
        torch.testing.assert_close(g, w, rtol = 1e-4, atol = 1e-6)
