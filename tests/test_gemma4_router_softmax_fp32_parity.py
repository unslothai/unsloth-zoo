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

"""higher_precision_softmax must not downcast a softmax the source already runs in float32.

Gemma-4's router computes `softmax(expert_scores, dim=-1, dtype=torch.float32)` and takes top-k, renormalises and
applies per_expert_scale in float32. The compiler used to append `.to(expert_scores.dtype)`, so under bf16 the
routing weights were rounded to bf16 before top-k: FastModel logits moved 0.50 off transformers (transformers' own
bf16 error vs float32 is 0.02). CPU only, tiny random model built from a config (no download).
"""

import inspect
import textwrap

import pytest
import torch

from unsloth_zoo import compiler


@pytest.mark.parametrize("dtype_arg", ["dtype=torch.float32", "dtype = torch.float32", "dtype=torch.float"])
def test_float32_softmax_left_alone(dtype_arg):
    src = f"p = nn.functional.softmax(scores, dim=-1, {dtype_arg})\n"
    assert compiler.higher_precision_softmax(src) == src


@pytest.mark.parametrize("call", ["F.softmax(x, dim=-1)", "F.softmax(x, dim=-1, dtype=x.dtype)",
                                  "nn.functional.softmax(x, dim=1, dtype=torch.bfloat16)"])
def test_low_precision_softmax_still_upcast(call):
    out = compiler.higher_precision_softmax(f"p = {call}\n")
    assert "dtype = torch.float32).to(x.dtype)" in out


def _gemma4():
    mg = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
    if not hasattr(mg, "Gemma4TextRouter"):
        pytest.skip("transformers has no Gemma-4 MoE router")
    return mg


def _tiny_model(mg, dtype):
    from transformers import AutoModelForCausalLM

    config = mg.Gemma4TextConfig(
        vocab_size = 256, hidden_size = 64, intermediate_size = 48, moe_intermediate_size = 40,
        num_hidden_layers = 2, num_attention_heads = 4, num_key_value_heads = 2, head_dim = 16,
        global_head_dim = 16, max_position_embeddings = 256, enable_moe_block = True, num_experts = 8,
        top_k_experts = 2, num_kv_shared_layers = 0, hidden_size_per_layer_input = 0,
        layer_types = ["sliding_attention", "full_attention"], sliding_window = 64, tie_word_embeddings = False,
        pad_token_id = 0, bos_token_id = 1, eos_token_id = 1,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, dtype = torch.float32)
    return model.to(dtype).eval()


def _compiled_router_forward(mg):
    src = textwrap.dedent(inspect.getsource(mg.Gemma4TextRouter.forward))
    namespace = dict(vars(mg))
    exec(compiler.higher_precision_softmax(src), namespace)
    return namespace["forward"]


def _forward(model, ids):
    acts = {}
    handles = [
        m.register_forward_hook(lambda mod, a, o, n = n: acts.__setitem__(n, (o[0] if isinstance(o, tuple) else o).float()))
        for n, m in model.named_modules() if ".layers." in n and n.count(".") <= 4
    ]
    with torch.no_grad():
        logits = model(input_ids = ids).logits.float()
    for h in handles:
        h.remove()
    return logits, acts


def _ids():
    return torch.randint(2, 256, (2, 12), generator = torch.Generator().manual_seed(1234))


def test_compiled_router_keeps_float32_weights():
    mg = _gemma4()
    router = _tiny_model(mg, torch.bfloat16).model.layers[0].router
    hidden = torch.randn(24, 64, generator = torch.Generator().manual_seed(0)).to(torch.bfloat16)
    with torch.no_grad():
        want = mg.Gemma4TextRouter.forward(router, hidden)
        got = _compiled_router_forward(mg)(router, hidden)
    for w, g in zip(want, got):
        assert g.dtype == w.dtype
        assert torch.equal(g, w)


@pytest.mark.parametrize("dtype, tol", [(torch.float32, 1e-5), (torch.bfloat16, 0.0)])
def test_compiled_router_model_parity(monkeypatch, dtype, tol):
    """Logits and every decoder submodule's output match transformers with the compiler-rewritten router."""
    mg = _gemma4()
    ids = _ids()
    ref_logits, ref_acts = _forward(_tiny_model(mg, dtype), ids)
    monkeypatch.setattr(mg.Gemma4TextRouter, "forward", _compiled_router_forward(mg))
    got_logits, got_acts = _forward(_tiny_model(mg, dtype), ids)
    worst = {n: float((ref_acts[n] - got_acts[n]).abs().max()) for n in ref_acts}
    bad = {n: d for n, d in worst.items() if d > tol}
    assert not bad, f"per-layer drift vs transformers ({dtype}): {bad}"
    assert float((ref_logits - got_logits).abs().max()) <= tol
