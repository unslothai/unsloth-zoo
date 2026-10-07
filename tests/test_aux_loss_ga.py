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

"""MoE router aux loss under gradient accumulation: once num_items_in_batch reaches the forward, the aux term is weighted by the micro-batch's token share, so its total weight over the accumulated batch is one instead of the number of micro-batches. CPU only."""

import ast
import inspect
import textwrap

import pytest
import torch

from unsloth_zoo.fused_losses.aux_loss import rewrite_aux_loss_ga, unsloth_ga_scale_aux_loss


SIMPLE = '''
    def forward(self, input_ids=None, attention_mask=None, labels=None, output_router_logits=None, **kwargs):
        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)
        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(outputs.router_logits, self.num_experts)
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device
        return loss
'''

MULTILINE = '''
def forward(self, input_ids=None, labels=None, **lm_kwargs):
    if labels is not None:
        loss += self.config.text_config.router_aux_loss_coef * aux_loss.to(
            loss.device
        )
    return loss
'''


def _aux_calls(source):
    tree = ast.parse(textwrap.dedent(source))
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == "unsloth_ga_scale_aux_loss"]


def test_rewrite_keeps_indentation_and_wraps_the_operand():
    new = rewrite_aux_loss_ga(SIMPLE)
    calls = _aux_calls(new)
    assert len(calls) == 1
    args = [ast.unparse(a) for a in calls[0].args]
    assert args == ["aux_loss.to(loss.device)", "labels", "attention_mask", "kwargs"]
    old_lines, new_lines = SIMPLE.splitlines(), new.splitlines()
    assert len(old_lines) == len(new_lines)
    assert sum(a != b for a, b in zip(old_lines, new_lines)) == 1
    assert rewrite_aux_loss_ga(new) == new  # idempotent


def test_multiline_operand_and_renamed_kwargs():
    new = rewrite_aux_loss_ga(MULTILINE)
    args = [ast.unparse(a) for a in _aux_calls(new)[0].args]
    assert args == ["aux_loss.to(loss.device)", "labels", "None", "lm_kwargs"]


def test_no_kwargs_or_no_labels_is_left_alone():
    no_kw = SIMPLE.replace(", **kwargs):", "):").replace(", **kwargs)", ")")
    assert rewrite_aux_loss_ga(no_kw) == no_kw
    no_labels = MULTILINE.replace("labels=None, ", "")
    assert rewrite_aux_loss_ga(no_labels) == no_labels


def test_weight_is_the_token_share_and_sums_to_one():
    aux = torch.tensor(2.0)
    lab = [torch.tensor([[-100, 5, 6, -100]]), torch.tensor([[-100, 1, 2, 3, 4, 5, 6, 7]])]
    n = sum((t[..., 1:] != -100).sum() for t in lab)
    scaled = [unsloth_ga_scale_aux_loss(aux, t, None, {"num_items_in_batch": n}) for t in lab]
    assert torch.allclose(sum(scaled), aux)
    assert torch.allclose(scaled[0], aux * 2 / 9)


def test_no_count_is_the_stock_aux_loss():
    aux = torch.tensor(3.0, requires_grad = True)
    lab = torch.tensor([[1, 2, 3]])
    assert unsloth_ga_scale_aux_loss(aux, lab, None, {}) is aux
    assert unsloth_ga_scale_aux_loss(aux, lab, None, {"num_items_in_batch": None}) is aux
    assert unsloth_ga_scale_aux_loss(aux, None, None, {"num_items_in_batch": 4}) is aux


def test_attention_mask_matches_the_batch_counter():
    aux = torch.tensor(1.0)
    lab = torch.tensor([[1, 2, 3, 4]])
    mask = torch.tensor([[1, 1, 0, 0]])
    out = unsloth_ga_scale_aux_loss(aux, lab, mask, {"num_items_in_batch": torch.tensor(4)})
    assert torch.allclose(out, torch.tensor(0.25))


def test_compiled_share_drops_the_same_packed_boundaries():
    aux = torch.tensor(1.0)
    lab = torch.tensor([[1, 2, 3, 4, 5, 6]])
    kw = {"num_items_in_batch": torch.tensor(4), "packed_seq_lengths": torch.tensor([2, 0, 2, 2])}
    eager = unsloth_ga_scale_aux_loss(aux, lab, None, kw)
    torch._dynamo.reset()
    compiled = torch.compile(unsloth_ga_scale_aux_loss, backend = "eager", fullgraph = True)(aux, lab, None, kw)
    torch._dynamo.reset()
    assert torch.allclose(eager, torch.tensor(0.75))
    assert torch.allclose(compiled, eager)


def _real(module, name):
    import importlib
    try:
        return getattr(importlib.import_module(module), name)
    except Exception as exc:
        pytest.skip(reason = f"{name} unavailable: {exc}")


@pytest.mark.parametrize("module,name", [
    ("transformers.models.qwen3_moe.modeling_qwen3_moe", "Qwen3MoeForCausalLM"),
    ("transformers.models.mixtral.modeling_mixtral", "MixtralForCausalLM"),
])
def test_hook_route_rewrites_real_moe_heads(module, name):
    cls = _real(module, name)
    src = textwrap.dedent(inspect.getsource(cls.forward))
    if "unsloth_fused_lm_head_loss" not in src:
        pytest.skip(reason = f"{name} not on the hook route here")
    calls = _aux_calls(src)
    assert len(calls) == 1, src
    assert ast.unparse(calls[0].args[1]) == "labels"


@pytest.mark.parametrize("module,name", [
    ("transformers.models.granitemoe.modeling_granitemoe", "GraniteMoeForCausalLM"),
    ("transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe", "Qwen3VLMoeForConditionalGeneration"),
])
def test_compiler_route_rewrites_real_moe_heads(module, name):
    from unsloth_zoo.compiler import fused_lm_head_forward
    cls = _real(module, name)
    src = inspect.getsource(cls.forward)
    if "aux_loss_coef" not in src:
        pytest.skip(reason = f"{name} adds no router aux loss here")
    new, route, _ = fused_lm_head_forward(name, cls, cls.__module__, src)
    if route is None:
        pytest.skip(reason = f"{name} is not fused here")
    assert len(_aux_calls(new)) >= 1
    compile(textwrap.dedent(new), "<fused>", "exec")
