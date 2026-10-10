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

"""Gradient accumulation count plumbing, on the stock forwards that exposed each gap. CPU only."""

import ast
import importlib
import inspect
import os
import sys
import textwrap

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _ga_stock_forwards as stock  # noqa: E402
from test_loss_normalization_contract import _fake_trainer, _loss_utils  # noqa: E402

from unsloth_zoo import compiler  # noqa: E402
from unsloth_zoo.fused_losses import aux_loss as aux_mod  # noqa: E402
from unsloth_zoo.fused_losses import ast_rewriter  # noqa: E402
from unsloth_zoo.fused_losses import cross_entropy_loss as ce_mod  # noqa: E402


def _live(module, name):
    try:
        return getattr(importlib.import_module(f"transformers.models.{module}.modeling_{module}"), name)
    except Exception as exc:
        pytest.skip(reason = f"{name} unavailable: {exc}")


def _isolated(cls):
    sub = type(cls.__name__, (cls,), {})
    sub.__module__ = cls.__module__
    return sub


def _calls(source, name):
    tree = ast.parse(textwrap.dedent(source))
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == name]



def test_loss_count_kwargs_follows_the_loss_signature():
    helper = ce_mod.unsloth_loss_count_kwargs

    def no_count(logits, labels, vocab_size): ...
    def var_kw(logits, labels, vocab_size, **kwargs): ...
    def named(logits, labels, vocab_size, num_items_in_batch = None): ...

    class Head:
        def loss_function(self, logits, labels, **kwargs): ...

    class Unreadable:
        @property
        def __signature__(self): raise RuntimeError("no signature")
        def __call__(self, *a, **k): ...

    n = torch.tensor(7)
    assert helper(no_count, n) == {}
    assert helper(var_kw, n) == {"num_items_in_batch": n}
    assert helper(named, n) == {"num_items_in_batch": n}
    assert helper(Head().loss_function, n) == {"num_items_in_batch": n}
    assert helper(var_kw, None) == {}
    assert helper(Unreadable(), n) == {}
    assert helper(len, n) == {}


def test_regex_route_hands_the_count_only_to_a_loss_that_takes_it():
    cls = _live("qwen3_vl", "Qwen3VLForConditionalGeneration")
    new, route, _ = compiler.fused_lm_head_forward(cls.__name__, cls, cls.__module__, stock.QWEN3_VL_5_16_1)
    assert route == "regex"
    assert "num_items_in_batch = n_items)" not in new
    assert "**unsloth_loss_count_kwargs(self.loss_function, n_items))" in new


def test_ast_route_injection_goes_through_the_helper_and_reads_the_alias():
    new, cap = ast_rewriter.rewrite_forward_source_spliced(stock.XGLM_5_16_1)
    assert new is not None and cap.count_injected
    lf = [c for c in ast.walk(ast.parse(textwrap.dedent(new))) if isinstance(c, ast.Call)
          and ast.unparse(c.func) == "self.loss_function"]
    assert lf
    for call in lf:
        unpacks = [k.value for k in call.keywords if k.arg is None]
        assert len(unpacks) == 1 and ast.unparse(unpacks[0].func) == "unsloth_loss_count_kwargs"
        count = unpacks[0].args[1]
        assert eval(ast.unparse(count), {"kwargs": {"num_items_in_batch": None, "n_items": 5}}) == 5
        assert eval(ast.unparse(count), {"kwargs": {"num_items_in_batch": 3, "n_items": 5}}) == 3


def test_hook_namespace_exports_the_helper():
    from unsloth_zoo.fused_losses import forward_install
    assert "unsloth_loss_count_kwargs" in inspect.getsource(forward_install)
    assert forward_install.unsloth_loss_count_kwargs is ce_mod.unsloth_loss_count_kwargs
    assert "unsloth_loss_count_kwargs" in compiler._disabled_sdpa_code



def test_count_route_falls_back_to_n_items_when_the_count_is_none():
    new = compiler._count_aware_ce_fallback(stock.SWITCH_5_17_0, "SwitchTransformersForConditionalGeneration", None)
    assert new is not None
    (call,) = _calls(new, "unsloth_count_aware_cross_entropy")
    n_items = next(k.value for k in call.keywords if k.arg == "n_items")
    expr = ast.unparse(n_items)
    assert eval(expr, {"kwargs": {"num_items_in_batch": None, "n_items": 7}}) == 7
    assert eval(expr, {"kwargs": {"num_items_in_batch": 2, "n_items": 7}}) == 2
    assert eval(expr, {"kwargs": {}}) is None


def test_count_aware_ce_keeps_a_replica_count_zero_dim():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 11)
    labels = torch.randint(0, 11, (2, 5))
    loss = ce_mod.unsloth_count_aware_cross_entropy(logits, labels, torch.tensor([6]), shift = False)
    assert loss.ndim == 0
    reference = F.cross_entropy(logits.reshape(-1, 11), labels.reshape(-1), reduction = "sum") / 6
    torch.testing.assert_close(loss, reference)



class NoKwargsForCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = nn.Linear(4, 11, bias = False)

    def forward(self, input_ids, labels = None, attention_mask = None):
        return None


class NoKwargsForSequenceClassification(NoKwargsForCausalLM):
    pass


class DistributedDataParallel(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module


class PeftModelForCausalLM(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base_model = base

    def get_base_model(self):
        return self.base_model


def _batches():
    g = torch.Generator().manual_seed(3)
    out = []
    for lengths in ((1, 4), (6, 2), (3, 3)):
        ids = torch.randint(0, 11, (2, 6), generator = g)
        labels = ids.clone()
        mask = torch.ones_like(ids)
        for row, n in enumerate(lengths):
            labels[row, n:] = -100
            mask[row, n:] = 0
        out.append({"input_ids": ids, "labels": labels, "attention_mask": mask})
    return out


SHIFTED, UNSHIFTED = 13, 19


def _count(model, accepts = True, batches = None):
    mod = _loss_utils()
    mod.ALLOWED_NUM_ITEMS_IN_BATCH.clear()
    batches = batches or _batches()
    return mod._unsloth_get_batch_samples(_fake_trainer(model, accepts), iter(batches), len(batches))[1]


def test_counter_reads_the_recorded_convention():
    head = NoKwargsForCausalLM()
    assert _count(head) is None
    head._unsloth_num_items_labels = "shifted"
    assert int(_count(head)) == SHIFTED
    head._unsloth_num_items_labels = "unshifted"
    assert int(_count(head)) == UNSHIFTED


@pytest.mark.parametrize("wrap", ["ddp", "peft", "ddp+peft"])
def test_counter_reads_the_convention_through_wrappers(wrap):
    head = NoKwargsForCausalLM()
    head._unsloth_num_items_labels = "unshifted"
    model = head
    if "peft" in wrap: model = PeftModelForCausalLM(model)
    if "ddp" in wrap: model = DistributedDataParallel(model)
    assert int(_count(model)) == UNSHIFTED


def test_recorded_convention_keeps_the_guards():
    head = NoKwargsForCausalLM()
    head._unsloth_num_items_labels = "shifted"
    assert _count(head, accepts = False) is None
    classifier = NoKwargsForSequenceClassification()
    classifier._unsloth_num_items_labels = "shifted"
    assert _count(classifier) is None
    head._unsloth_num_items_labels = "sideways"
    assert _count(head) is None



def test_aux_helper_reads_the_alias_and_an_explicit_count():
    aux = torch.tensor(2.0)
    lab = torch.tensor([[-100, 5, 6, -100]])
    torch.testing.assert_close(aux_mod.unsloth_ga_scale_aux_loss(aux, lab, None, {"n_items": 4}), torch.tensor(1.0))
    torch.testing.assert_close(
        aux_mod.unsloth_ga_scale_aux_loss(aux, lab, None, None, num_items_in_batch = torch.tensor(4)),
        torch.tensor(1.0),
    )
    torch.testing.assert_close(
        aux_mod.unsloth_ga_scale_aux_loss(aux, lab, None, {"num_items_in_batch": None, "n_items": 8}),
        torch.tensor(0.5),
    )


def test_aux_helper_ratio_is_fp32_and_a_zero_total_is_left_alone():
    aux = torch.tensor(1.0, dtype = torch.float16)
    lab = torch.zeros(1, 70001, dtype = torch.long)
    out = aux_mod.unsloth_ga_scale_aux_loss(aux, lab, None, {"num_items_in_batch": torch.tensor(140000)})
    assert out.dtype == torch.float16
    torch.testing.assert_close(out, torch.tensor(0.5, dtype = torch.float16))
    aux = torch.tensor(3.0, requires_grad = True)
    for zero in (0, torch.tensor(0), torch.tensor([0])):
        out = aux_mod.unsloth_ga_scale_aux_loss(aux, torch.tensor([[1, 2, 3]]), None, {"num_items_in_batch": zero})
        torch.testing.assert_close(out, aux.detach())


def test_aux_helper_counts_the_way_the_head_is_counted():
    aux = torch.tensor(1.0)
    lab = torch.tensor([[1, 2, -100, 4]])
    kw = {"num_items_in_batch": torch.tensor(6)}
    head = NoKwargsForCausalLM()
    torch.testing.assert_close(aux_mod.unsloth_ga_scale_aux_loss(aux, lab, None, kw, model = head), torch.tensor(2 / 6))
    head._unsloth_num_items_labels = "unshifted"
    torch.testing.assert_close(aux_mod.unsloth_ga_scale_aux_loss(aux, lab, None, kw, model = head), torch.tensor(3 / 6))
    marked = NoKwargsForCausalLM()
    marked._unsloth_counts_unshifted_labels = True
    torch.testing.assert_close(aux_mod.unsloth_ga_scale_aux_loss(aux, lab, None, kw, model = marked), torch.tensor(3 / 6))


class KwargsForCausalLM(NoKwargsForCausalLM):
    def forward(self, input_ids, labels = None, attention_mask = None, **kwargs):
        return None


def test_aux_shares_sum_to_one_against_the_counter_with_packing():
    batches = []
    for lengths in ((3, 5), (2, 2, 4), (8,)):
        ids = torch.arange(1, 9).reshape(1, 8)
        batches.append({"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids),
                        "packed_seq_lengths": torch.tensor(lengths, dtype = torch.int32)})
    head = KwargsForCausalLM()
    n = _count(head, batches = batches)
    assert int(n) == (7 - 1) + (7 - 2) + 7  # 7 shifted targets a row, less the internal boundaries
    aux = torch.tensor(1.0)
    shares = [aux_mod.unsloth_ga_scale_aux_loss(
        aux, b["labels"], b["attention_mask"],
        {"num_items_in_batch": n, "packed_seq_lengths": b["packed_seq_lengths"]}, model = head)
        for b in batches]
    torch.testing.assert_close(sum(shares), aux)



def _scaled_operands(source):
    return [ast.unparse(c.args[0]) for c in _calls(source, "unsloth_ga_scale_aux_loss")]


@pytest.mark.parametrize("src", [stock.SWITCH_5_17_0, stock.SWITCH_4_55_4], ids = ["5.17.0", "4.55.4"])
def test_switch_router_terms_are_weighted_on_the_count_route(src):
    new = compiler._count_aware_ce_fallback(src, "SwitchTransformersForConditionalGeneration", None)
    assert new is not None
    assert sorted(_scaled_operands(new)) == ["aux_loss", "z_loss"]
    assert not aux_mod.unscaled_extra_loss_terms(new)
    assert "loss = loss + unsloth_ga_scale_aux_loss(z_loss" in new


def test_nllb_moe_bamba_dbrx_mixtral_spellings():
    nllb = aux_mod.rewrite_aux_loss_ga(stock.NLLB_MOE_5_17_0)
    assert _scaled_operands(nllb) == ["aux_loss"]
    assert "loss = loss + unsloth_ga_scale_aux_loss(aux_loss, labels, attention_mask, kwargs, model = self)" in nllb
    bamba = aux_mod.rewrite_aux_loss_ga(stock.BAMBA_5_17_0)
    assert "loss = loss + self.z_loss_coefficient * unsloth_ga_scale_aux_loss(z_loss," in bamba
    dbrx = aux_mod.rewrite_aux_loss_ga(stock.DBRX_4_55_4)
    assert _scaled_operands(dbrx) == ["aux_loss.to(loss.device)"]
    assert "self.moe_loss_weight * unsloth_ga_scale_aux_loss(" in dbrx
    mixtral = aux_mod.rewrite_aux_loss_ga(stock.MIXTRAL_4_57_6)
    (call,) = _calls(mixtral, "unsloth_ga_scale_aux_loss")
    assert [ast.unparse(a) for a in call.args] == ["aux_loss.to(loss.device)", "labels", "attention_mask", "kwargs"]
    assert {k.arg: ast.unparse(k.value) for k in call.keywords} == {"model": "self"}
    for new in (nllb, bamba, dbrx, mixtral):
        assert aux_mod.rewrite_aux_loss_ga(new) == new
        assert not aux_mod.unscaled_extra_loss_terms(new)


def test_a_statement_with_an_unknown_term_is_left_whole():
    src = textwrap.dedent('''
    def forward(self, labels=None, **kwargs):
        loss = loss + aux_loss + mystery_term
        return loss
    ''')
    assert aux_mod.rewrite_aux_loss_ga(src) == src
    assert aux_mod.unscaled_extra_loss_terms(src)


def test_explicit_count_parameter_is_passed_through():
    src = textwrap.dedent('''
    def forward(self, input_ids=None, labels=None, num_items_in_batch=None):
        loss = self.loss_function(logits, labels, self.vocab_size, num_items_in_batch=num_items_in_batch)
        loss += self.router_aux_loss_coef * aux_loss
        return loss
    ''')
    new = aux_mod.rewrite_aux_loss_ga(src)
    (call,) = _calls(new, "unsloth_ga_scale_aux_loss")
    assert [ast.unparse(a) for a in call.args] == ["aux_loss", "labels", "None", "None"]
    assert {k.arg: ast.unparse(k.value) for k in call.keywords} == {
        "num_items_in_batch": "num_items_in_batch", "model": "self"}


def test_count_route_declines_beside_an_unweighted_term():
    src = stock.SWITCH_5_17_0.replace("loss = loss + z_loss + aux_loss", "loss = loss + z_loss + router_balance")
    assert src != stock.SWITCH_5_17_0
    assert compiler._count_aware_ce_fallback(src, "SwitchTransformersForConditionalGeneration", None) is None


def test_weighted_terms_run_and_stay_stock_without_a_count():
    new = aux_mod.rewrite_aux_loss_ga(stock.NLLB_MOE_5_17_0)
    (stmt,) = [n for n in ast.walk(ast.parse(textwrap.dedent(new)))
               if isinstance(n, ast.Assign) and "unsloth_ga_scale_aux_loss" in ast.unparse(n)]
    code = compile(ast.Module([stmt], []), "<nllb>", "exec")
    labels = torch.tensor([[5, 6, -100, 7]])
    head = NoKwargsForCausalLM()
    head._unsloth_counts_unshifted_labels = True
    env = {"unsloth_ga_scale_aux_loss": aux_mod.unsloth_ga_scale_aux_loss, "labels": labels,
           "attention_mask": None, "self": head, "loss": torch.tensor(1.0), "aux_loss": torch.tensor(0.5)}
    stock_env = dict(env, kwargs = {})
    exec(code, stock_env)
    torch.testing.assert_close(stock_env["loss"], torch.tensor(1.5))
    counted = dict(env, kwargs = {"num_items_in_batch": torch.tensor(6)})
    exec(code, counted)
    torch.testing.assert_close(counted["loss"], torch.tensor(1.0 + 0.5 * 3 / 6))



@pytest.mark.parametrize("module,name", [
    ("moonshine", "MoonshineForConditionalGeneration"),
    ("pp_formulanet", "PPFormulaNetForConditionalGeneration"),
    ("canary", "CanaryForConditionalGeneration"),
    ("cohere_asr", "CohereAsrForConditionalGeneration"),
])
def test_aligned_target_heads_are_marked_unshifted(module, name):
    cls = _isolated(_live(module, name))
    src = inspect.getsource(_live(module, name).forward)
    if "unsloth_fused_lm_head_loss" in src:
        pytest.skip(reason = f"{name} is on the hook route here")
    new, route, _ = compiler.fused_lm_head_forward(name, cls, cls.__module__, src)
    if route != "ast":
        pytest.skip(reason = f"{name} is not on the AST route here ({route})")
    assert cls.__dict__.get("_unsloth_counts_unshifted_labels") is True


def test_moonshine_stock_source_is_marked_and_unshifted_is_its_divisor():
    cls = _isolated(_live("moonshine", "MoonshineForConditionalGeneration"))
    new, route, _ = compiler.fused_lm_head_forward(cls.__name__, cls, cls.__module__, stock.MOONSHINE_5_17_0)
    assert route == "ast"
    assert cls.__dict__.get("_unsloth_counts_unshifted_labels") is True
    from transformers.loss.loss_utils import ForCausalLMLoss
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 13)
    labels = torch.randint(0, 13, (2, 5))
    labels[0, 3:] = -100
    n = int((labels != -100).sum())
    loss = ForCausalLMLoss(logits, None, 13, num_items_in_batch = torch.tensor(n), shift_labels = labels)
    total = F.cross_entropy(logits.reshape(-1, 13).float(), labels.reshape(-1), ignore_index = -100, reduction = "sum")
    torch.testing.assert_close(loss * n, total)



def test_collator_shift_labels_do_not_change_the_count():
    # The fused losses train on `labels`, so the divisor counts `labels[..., 1:]`, not `shift_labels`.
    count_batch_items = _loss_utils().count_batch_items
    labels = torch.randint(0, 13, (2, 8))
    shift_labels = torch.full((2, 8), -100)
    shift_labels[0, :3] = 1
    share = aux_mod.unsloth_ga_scale_aux_loss(
        torch.tensor(1.0), labels, None, {"num_items_in_batch": torch.tensor(28), "shift_labels": shift_labels},
    )
    assert int(count_batch_items(labels)[0]) == 14
    torch.testing.assert_close(share, torch.tensor(0.5))
