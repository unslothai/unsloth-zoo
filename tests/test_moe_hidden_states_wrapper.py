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

"""The GRPO hidden-states wrapper on Qwen3 MoE heads stays transparent to the compiler and wraps once."""
import inspect

import pytest
import torch

modeling = pytest.importorskip("transformers.models.qwen3_moe.modeling_qwen3_moe")


def _is_wrapper(fn):
    # Not via getsource: it follows __wrapped__ to the real forward.
    return "_original_causal_lm_forward" in fn.__code__.co_freevars


def _stock_forward(fn):
    # Importing unsloth_zoo may already have wrapped the class; peel wrappers via their closure.
    while _is_wrapper(fn):
        cells = dict(zip(fn.__code__.co_freevars, fn.__closure__ or ()))
        fn = cells["_original_causal_lm_forward"].cell_contents
    return fn


@pytest.fixture
def fresh_forward(monkeypatch):
    # Patches mutate the class; restore the stock forward after each test.
    stock = _stock_forward(modeling.Qwen3MoeForCausalLM.forward)
    monkeypatch.setattr(modeling.Qwen3MoeForCausalLM, "forward", stock)
    return stock


def _patch():
    from unsloth_zoo.temporary_patches.qwen3_moe import _patch_causal_lm_forward_for_hidden_states
    _patch_causal_lm_forward_for_hidden_states(
        modeling.Qwen3MoeForCausalLM, modeling.MoeCausalLMOutputWithPast, "Qwen3MoeForCausalLM",
    )


def test_compiler_reads_the_real_forward(fresh_forward):
    from unsloth_zoo.compiler import _unwrap_undecorated_method
    _patch()
    patched = modeling.Qwen3MoeForCausalLM.forward
    assert patched is not fresh_forward
    real = _unwrap_undecorated_method(patched, "Qwen3MoeForCausalLM")
    # The compiler fuses whatever getsource returns here.
    assert "UNSLOTH_RETURN_HIDDEN_STATES" not in inspect.getsource(real)
    assert inspect.getsource(real) == inspect.getsource(fresh_forward)


def test_repeated_phases_wrap_once(fresh_forward):
    for _ in range(3):
        _patch()
    patched = modeling.Qwen3MoeForCausalLM.forward
    assert patched.__wrapped__ is fresh_forward


def _tiny():
    config = modeling.Qwen3MoeConfig(
        vocab_size = 64, hidden_size = 16, intermediate_size = 32, moe_intermediate_size = 8,
        num_hidden_layers = 2, num_attention_heads = 2, num_key_value_heads = 2, head_dim = 8,
        num_experts = 4, num_experts_per_tok = 2,
    )
    torch.manual_seed(0)
    return config, modeling.Qwen3MoeForCausalLM(config).eval()


def test_hidden_states_and_normal_paths(fresh_forward, monkeypatch):
    _patch()
    config, model = _tiny()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        monkeypatch.setenv("UNSLOTH_RETURN_HIDDEN_STATES", "1")
        hidden = model(input_ids = ids).logits
        reference = model.model(input_ids = ids).last_hidden_state
        monkeypatch.setenv("UNSLOTH_RETURN_HIDDEN_STATES", "0")
        out = model(input_ids = ids, labels = ids)
        stock = fresh_forward(model, input_ids = ids, labels = ids)
    assert hidden.shape == (1, 5, config.hidden_size)
    torch.testing.assert_close(hidden, reference)
    torch.testing.assert_close(out.loss, stock.loss)
    torch.testing.assert_close(out.logits, stock.logits)
