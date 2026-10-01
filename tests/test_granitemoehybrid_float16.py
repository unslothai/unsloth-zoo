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

"""Granite-4 float16: the decoder folds residual_multiplier into the shared MLP input so output_linear cannot overflow."""

import pytest
import torch

gm = pytest.importorskip("transformers.models.granitemoehybrid.modeling_granitemoehybrid")
from unsloth_zoo.temporary_patches import granitemoehybrid as patch

# transformers 4.56.2 decoder tail (older API: tuple outputs, router_logits); keeps the rewrite honest on the notebook stack.
LEGACY_FORWARD = '''
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(self, hidden_states, attention_mask=None, output_router_logits=False, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(hidden_states=hidden_states, **kwargs)
        hidden_states = residual + hidden_states * self.residual_multiplier

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.has_experts:
            moe_hidden_states, router_logits = self.block_sparse_moe(hidden_states)
            hidden_states = moe_hidden_states + self.shared_mlp(hidden_states)
        else:
            hidden_states = self.shared_mlp(hidden_states)
            router_logits = None

        hidden_states = residual + hidden_states * self.residual_multiplier

        outputs = (hidden_states,)
        return outputs
'''


def _stock_forward():
    forward = gm.GraniteMoeHybridDecoderLayer.forward
    return getattr(forward, "__wrapped__", forward) if getattr(forward, "_unsloth_granite_float16", False) else forward


def test_rewrite_matches_installed_and_legacy_source():
    import inspect
    for source in (inspect.getsource(_stock_forward()), LEGACY_FORWARD):
        out = patch._rewrite_decoder_source(source)
        assert out is not None
        assert out.startswith("def forward")
        assert out.count("self.residual_multiplier") == 2
        assert "_granite_scaled_shared_mlp(self, hidden_states)" in out
        assert "hidden_states = residual + hidden_states\n" in out


def _tiny_model(experts, dtype):
    config = gm.GraniteMoeHybridConfig(
        vocab_size = 64, hidden_size = 32, intermediate_size = 32, shared_intermediate_size = 48,
        num_hidden_layers = 2, layer_types = ["attention", "attention"], num_attention_heads = 4,
        num_key_value_heads = 2, num_local_experts = experts, num_experts_per_tok = 2 if experts else 0,
        residual_multiplier = 0.25, embedding_multiplier = 12.0, logits_scaling = 4.0,
        mamba_n_heads = 8, mamba_d_head = 8, mamba_n_groups = 1,
    )
    torch.manual_seed(0)
    return gm.GraniteMoeHybridForCausalLM(config).to(dtype).eval()


@pytest.mark.parametrize("experts", [0, 4])
def test_float16_forward_is_the_same_function(experts, monkeypatch):
    # Run the rewritten forward in float32: the fold is exact algebra, so logits must match the stock layer.
    model = _tiny_model(experts, torch.float32)
    ids = torch.randint(0, 64, (2, 7))
    stock = _stock_forward()
    monkeypatch.setattr(gm.GraniteMoeHybridDecoderLayer, "forward", stock)
    with torch.no_grad(): expected = model(ids, use_cache = False).logits
    folded = patch._build_float16_decoder_forward(gm, stock)
    monkeypatch.setattr(gm.GraniteMoeHybridDecoderLayer, "forward", folded)
    with torch.no_grad(): got = model(ids, use_cache = False).logits
    torch.testing.assert_close(got, expected, rtol = 1e-5, atol = 1e-5)


def test_float16_dispatch_and_overflow(monkeypatch):
    monkeypatch.setattr(gm.GraniteMoeHybridDecoderLayer, "forward", _stock_forward())
    model = _tiny_model(0, torch.float16)
    with torch.no_grad():
        # Constant activations give a shared-MLP output of ~71k per layer: past float16's 65504, ~18k after residual_multiplier.
        model.model.embed_tokens.weight.fill_(1.0)
        for layer in model.model.layers:
            layer.shared_mlp.input_linear.weight.fill_(0.1)
            layer.shared_mlp.output_linear.weight.fill_(150.0)
    ids = torch.randint(0, 64, (1, 5))
    with torch.no_grad(): stock_logits = model(ids, use_cache = False).logits
    assert not torch.isfinite(stock_logits).all()

    patch.patch_GraniteMoeHybridDecoderLayer_float16()
    assert gm.GraniteMoeHybridDecoderLayer.forward._unsloth_granite_float16
    with torch.no_grad(): fixed = model(ids, use_cache = False).logits
    assert torch.isfinite(fixed).all()

    calls = []
    stock = gm.GraniteMoeHybridDecoderLayer.forward.__wrapped__
    def spy(self, hidden_states, *args, **kwargs):
        calls.append(hidden_states.dtype)
        return stock(self, hidden_states, *args, **kwargs)
    monkeypatch.setattr(gm.GraniteMoeHybridDecoderLayer.forward, "__wrapped__", spy)
    # The dispatcher closed over the stock function, so re-patch against the spy.
    monkeypatch.setattr(gm.GraniteMoeHybridDecoderLayer, "forward", spy)
    patch.patch_GraniteMoeHybridDecoderLayer_float16()
    with torch.no_grad(): model.to(torch.bfloat16)(ids, use_cache = False)
    assert calls and set(calls) == {torch.bfloat16}


def test_float16_fold_keeps_lora_bias_scaled(monkeypatch):
    peft = pytest.importorskip("peft")
    model = _tiny_model(0, torch.float32)
    config = peft.LoraConfig(r = 4, target_modules = ["output_linear"], lora_bias = True, init_lora_weights = False)
    model = peft.get_peft_model(model, config)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name and name.endswith("bias"): param.fill_(0.5)
    ids = torch.randint(0, 64, (2, 7))
    stock = _stock_forward()
    monkeypatch.setattr(gm.GraniteMoeHybridDecoderLayer, "forward", stock)
    with torch.no_grad(): expected = model(input_ids = ids, use_cache = False).logits
    monkeypatch.setattr(gm.GraniteMoeHybridDecoderLayer, "forward", patch._build_float16_decoder_forward(gm, stock))
    with torch.no_grad(): got = model(input_ids = ids, use_cache = False).logits
    torch.testing.assert_close(got, expected, rtol = 1e-5, atol = 1e-5)


def test_float16_forward_helper_is_a_module_global():
    # torch.compile guards resolve globals via sys.modules[__name__].
    import sys
    folded = patch._build_float16_decoder_forward(gm, _stock_forward())
    owner = sys.modules[folded.__globals__["__name__"]]
    assert getattr(owner, "_granite_scaled_shared_mlp", None) is patch._granite_scaled_shared_mlp
