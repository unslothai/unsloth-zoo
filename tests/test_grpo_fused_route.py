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

"""grpo_accumulated_loss routed through fused_linear_grpo_loss against the log-prob route (fused_linear_logprobs +
UnslothEfficientGRPO + UnslothGRPOEntropyBonus): loss, metrics and every parameter gradient, on the padded and the
sequence-packed forward, plus the routing decisions."""
import os
import types

import pytest
import torch

import unsloth_zoo.fused_losses.fused_grpo as fg
import unsloth_zoo.fused_losses.fused_linear as fl
import unsloth_zoo.rl_replacements as rr

H, V, LTK = 8, 13, 5
# GRPO_FUSED_ROUTE_DEVICE=cuda runs the same comparisons on a GPU (float32, no autocast).
DEVICE = torch.device(os.environ.get("GRPO_FUSED_ROUTE_DEVICE", "cpu"))
# Prompt 3 + completion 5, pad 0: left pads on rows 1 and 3, completions ending early (pads after EOS) on 0 and 2.
INPUT_IDS = torch.tensor([
    [4, 5, 6, 7, 8, 9, 3, 0],
    [0, 5, 6, 2, 10, 11, 12, 7],
    [4, 9, 6, 7, 1, 0, 0, 0],
    [0, 0, 6, 11, 12, 3, 5, 9],
    [4, 5, 6, 2, 3, 5, 8, 6],
    [8, 1, 6, 9, 3, 2, 7, 10],
])
ROWS = INPUT_IDS.shape[0]


class _Model(torch.nn.Module):
    def __init__(self, head_grad = True, bias = False):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(V, H)
        self.mix = torch.nn.Linear(H, H)
        self.head = torch.nn.Linear(H, V, bias = bias)
        self.head.weight.requires_grad_(head_grad)
        if bias:
            self.head.bias.requires_grad_(head_grad)
        self.device = torch.device("cpu")

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids = None, attention_mask = None, logits_to_keep = None, **kwargs):
        hidden = torch.tanh(self.mix(self.emb(input_ids))) * 2.0
        return types.SimpleNamespace(logits = hidden)


def _trainer(model, beta = 0.0, scaler = None, entropy_coef = 0.0, use_vllm = False, is_correction = False,
             importance_sampling_level = "token", **extra):
    args = types.SimpleNamespace(
        unsloth_grpo_mini_batch = 2, unsloth_logit_chunk_multiplier = 1, steps_per_generation = 2,
        use_bias_correction_kl = extra.pop("use_bias_correction_kl", False), top_entropy_quantile = 1.0,
        entropy_coef = entropy_coef, use_adaptive_entropy = False,
        vllm_importance_sampling_mode = extra.pop("vllm_importance_sampling_mode", "token_truncate"),
        vllm_importance_sampling_cap = 3.0,
        sapo_temperature_pos = 1.0, sapo_temperature_neg = 1.05,
        off_policy_mask_threshold = extra.pop("off_policy_mask_threshold", None),
    )
    return types.SimpleNamespace(
        args = args,
        processing_class = types.SimpleNamespace(pad_token_id = 0),
        model = model,
        accelerator = types.SimpleNamespace(unwrap_model = lambda m, keep_fp32_wrapper = False: m, scaler = scaler),
        use_vllm = use_vllm,
        vllm_importance_sampling_correction = is_correction,
        _autocast_dtype = None,
        beta = beta,
        entropy_coef = entropy_coef,
        **extra,
    )


def _side_inputs(seed = 0):
    """old / ref / vLLM log-probs near the policy's own, on the [rows, W] grid the loss reads."""
    model = _Model()
    pad = rr.calculate_pad_tokens_in_prompt(INPUT_IDS, LTK, 0)
    W = LTK + int(pad.max())
    ids = rr.left_pack_padding(INPUT_IDS, 0)
    with torch.no_grad():
        logp = torch.log_softmax(model.head(model(ids).logits), -1)
        lp = logp[:, :-1].gather(-1, ids[:, 1:, None])[..., 0][:, -W:]
    g = torch.Generator().manual_seed(seed)
    noise = lambda s: lp + s * torch.randn(lp.shape, generator = g)  # noqa: E731
    return noise(0.15), noise(0.3), noise(0.1), torch.randn(ROWS, generator = g)


def _run(fused, monkeypatch, *, loss_type = "grpo", old = None, ref = None, sampling = None, adv = None,
         packed = False, head_grad = True, bias = False, completion_mask = None, loss_kwargs = None, backward_scale = 1.0,
         **trainer_kwargs):
    monkeypatch.setattr(fg, "GRPO_FWD_GRAD", fused)
    monkeypatch.setenv("UNSLOTH_GRPO_SEQ_PACKING", "1" if packed else "0")
    monkeypatch.setenv("UNSLOTH_GRPO_PREFIX_GROUPER", "0")
    model = _Model(head_grad, bias).to(DEVICE)
    model.device = DEVICE
    trainer = _trainer(model, **trainer_kwargs)
    if completion_mask is None:
        completion_mask = (INPUT_IDS[:, -LTK:] != 0).long()
    to = lambda x: x.to(DEVICE) if torch.is_tensor(x) else x  # noqa: E731
    completion_mask, old, ref, sampling, adv = map(to, (completion_mask, old, ref, sampling, adv))
    if loss_kwargs and "sample_weight" in loss_kwargs:
        loss_kwargs = dict(loss_kwargs, sample_weight = to(loss_kwargs["sample_weight"]))
    kwargs = dict(
        loss_type = loss_type, epsilon_low = 0.2, epsilon_high = 0.28 if loss_type != "cispo" else 4.0,
        max_completion_length = LTK, num_items_in_batch = completion_mask.sum().float(),
        current_gradient_accumulation_steps = 2, num_processes = 1,
        importance_sampling_level = trainer_kwargs.get("importance_sampling_level", "token"),
        sampling_per_token_logps = sampling,
    )
    kwargs.update(loss_kwargs or {})
    call = lambda: rr.grpo_accumulated_loss(  # noqa: E731
        trainer, INPUT_IDS.to(DEVICE), (INPUT_IDS != 0).long().to(DEVICE), LTK, completion_mask,
        adv if adv is not None else torch.linspace(-1.0, 1.2, ROWS, device = DEVICE), old, ref, **kwargs,
    )
    if packed:
        call()  # first use verifies the packed forward against per-row forwards
        model.zero_grad(set_to_none = True)
    before = fl.CALLS.get("grpo_fwd_grad", 0)
    out = call()
    engaged = fl.CALLS.get("grpo_fwd_grad", 0) - before
    (out[0] * backward_scale).backward()
    grads = {n: p.grad.cpu() for n, p in model.named_parameters() if p.grad is not None}
    return out, grads, engaged


def _compare(monkeypatch, **kw):
    a, ga, n_fused = _run(True, monkeypatch, **kw)
    b, gb, n_base = _run(False, monkeypatch, **kw)
    assert n_fused == 1 and n_base == 0
    for x, y in zip(a, b):
        if x is None or y is None:
            assert x is None and y is None
        elif x.numel():
            torch.testing.assert_close(x.detach().float().cpu(), y.detach().float().cpu(), rtol = 1e-5, atol = 1e-6)
        else:
            assert y.numel() == 0
    assert ga.keys() == gb.keys() and len(ga) >= 3
    for name in ga:
        torch.testing.assert_close(ga[name], gb[name], rtol = 1e-4, atol = 1e-6, msg = name)
    return a, ga


OLD, REF, SAMPLING, ADV = _side_inputs()


@pytest.mark.parametrize("packed", [False, True], ids = ["padded", "packed"])
@pytest.mark.parametrize("loss_type,beta,old,bias_kl", [
    ("grpo", 0.0, None, False),
    ("grpo", 0.04, OLD, False),
    ("grpo", 0.04, OLD, True),
    ("bnpo", 0.0, OLD, False),
    ("dr_grpo", 0.1, OLD, False),
    ("dapo", 0.0, OLD, False),
    ("luspo", 0.0, OLD, False),
    ("cispo", 0.04, OLD, False),
    ("sapo", 0.02, OLD, True),
])
def test_matches_log_prob_route(monkeypatch, packed, loss_type, beta, old, bias_kl):
    _compare(monkeypatch, packed = packed, loss_type = loss_type, beta = beta, old = old, ref = REF, adv = ADV,
             use_bias_correction_kl = bias_kl)


@pytest.mark.parametrize("packed", [False, True], ids = ["padded", "packed"])
@pytest.mark.parametrize("loss_type", ["grpo", "dapo", "cispo"])
def test_entropy_bonus_folds_into_the_loss(monkeypatch, packed, loss_type):
    _compare(monkeypatch, packed = packed, loss_type = loss_type, beta = 0.04, old = OLD, ref = REF, adv = ADV,
             entropy_coef = 0.05)


@pytest.mark.parametrize("mode", ["token_truncate", "token_mask", "sequence_truncate", "geometric_mask"])
def test_vllm_importance_sampling_ratio(monkeypatch, mode):
    out, _ = _compare(monkeypatch, loss_type = "dapo", beta = 0.0, old = OLD, ref = REF, sampling = SAMPLING,
                      adv = ADV, use_vllm = True, is_correction = True, vllm_importance_sampling_mode = mode,
                      loss_kwargs = dict(use_vllm = True))
    assert out[4] is not None and out[4].numel() > 0  # flat_is_ratio reaches the metrics


def test_sample_weight_truncated_rows_bias_and_frozen_head(monkeypatch):
    mask = (INPUT_IDS[:, -LTK:] != 0).long()
    mask[2] = 0  # a truncated completion TRL zeroed
    _compare(monkeypatch, loss_type = "grpo", beta = 0.04, old = OLD, ref = REF, adv = ADV, completion_mask = mask,
             mask_truncated_completions = True, loss_kwargs = dict(sample_weight = torch.rand(ROWS) / 10))
    _compare(monkeypatch, loss_type = "dapo", old = OLD, adv = ADV, bias = True)
    _, grads = _compare(monkeypatch, loss_type = "dapo", old = OLD, adv = ADV, head_grad = False)
    assert "head.weight" not in grads


def test_mixed_precision_scale_and_deepspeed_upstream(monkeypatch):
    scaler = types.SimpleNamespace(get_scale = lambda: 1024.0)
    # The scaler multiplies the gradient; grad_output (here 0.25) is ignored, as in UnslothEfficientGRPO.
    _compare(monkeypatch, loss_type = "dapo", old = OLD, adv = ADV, scaler = scaler, backward_scale = 0.25)
    # DeepSpeed (no scaler): grad_output times upstream_scale
    _compare(monkeypatch, loss_type = "grpo", old = OLD, adv = ADV, is_deepspeed_enabled = True,
             loss_is_scaled_for_ga = False, compute_loss_func = None, backward_scale = 0.25,
             entropy_coef = 0.05)


def test_trainable_head_over_budget_keeps_the_log_prob_op(monkeypatch):
    monkeypatch.setattr(rr, "grpo_fused_head_fits", lambda lm_head, budget_fraction = None: False)
    before = rr.GRPO_FUSED_ROUTE_STATS["head_budget"]
    a, ga, n = _run(True, monkeypatch, loss_type = "dapo", old = OLD, adv = ADV, entropy_coef = 0.05)
    b, gb, _ = _run(False, monkeypatch, loss_type = "dapo", old = OLD, adv = ADV, entropy_coef = 0.05)
    assert n == 0 and rr.GRPO_FUSED_ROUTE_STATS["head_budget"] == before + 1
    torch.testing.assert_close(a[0], b[0])
    for name in gb:
        torch.testing.assert_close(ga[name], gb[name], msg = name)


@pytest.mark.parametrize("change", [
    dict(importance_sampling_level = "sequence"),
    dict(off_policy_mask_threshold = 0.5),
    dict(loss_type = "vespo"),
])
def test_configs_without_a_closed_form_keep_the_log_prob_route(monkeypatch, change):
    kw = dict(loss_type = "grpo", old = OLD, adv = ADV)
    kw.update(change)
    if "off_policy_mask_threshold" in change:
        kw["get_off_policy_mask"] = lambda **k: torch.ones_like(k["per_token_logps"])
    if kw["loss_type"] == "vespo":
        kw["get_gamma_weights"] = lambda **k: torch.ones(ROWS, 1, device = DEVICE)
    assert _run(True, monkeypatch, **kw)[2] == 0


def test_route_decisions():
    ok = dict(loss_type = "dapo", importance_sampling_level = "token")
    assert rr.grpo_fused_route_ok(ok)
    for bad in (dict(loss_type = "vespo"), dict(importance_sampling_level = "sequence"),
                dict(off_policy_mask_threshold = 0.1), dict(rollout_rs = [("token_k3", None, 1.0)]),
                dict(entropy_mask = torch.ones(1))):
        assert not rr.grpo_fused_route_ok({**ok, **bad}), bad
    assert not rr.grpo_fused_route_ok(ok, entropy_mask_on = True)
    assert not rr.grpo_fused_route_ok(ok, rollout_logps_as_old = True)
    assert not rr.grpo_fused_route_ok(ok, support_kwargs = {"support_top_k": 5})
    assert not rr.grpo_fused_route_ok(ok, pixel_values = torch.ones(1))
    assert not rr.grpo_fused_route_ok(ok, zero3_head = (torch.ones(1), None))
    for lt in ("grpo", "bnpo", "dr_grpo", "dapo", "luspo", "cispo", "sapo"):
        assert rr.grpo_fused_route_ok({**ok, "loss_type": lt})


def test_route_switches(monkeypatch):
    ok = dict(loss_type = "grpo")
    monkeypatch.setattr(fg, "GRPO_FWD_GRAD", False)
    assert not rr.grpo_fused_route_ok(ok)
    monkeypatch.setattr(fg, "GRPO_FWD_GRAD", True)
    monkeypatch.setenv("UNSLOTH_GRPO_LOGPROB_RECOMPUTE", "1")
    assert not rr.grpo_fused_route_ok(ok)
    monkeypatch.setenv("UNSLOTH_GRPO_LOGPROB_RECOMPUTE", "0")
    monkeypatch.setenv("UNSLOTH_VLLM_FP32_LM_HEAD", "1")
    assert not rr.grpo_fused_route_ok(ok)
    monkeypatch.setenv("UNSLOTH_VLLM_FP32_LM_HEAD", "0")
    monkeypatch.setenv("UNSLOTH_GRPO_CHUNKED_LOGPROBS", "0")
    assert not rr.grpo_fused_route_ok(ok)


def test_head_budget():
    frozen = torch.zeros(4, 3)
    assert rr.grpo_fused_head_fits(frozen)
    assert rr.grpo_fused_head_fits(torch.zeros(4, 3, requires_grad = True))  # host memory is not budgeted
    with torch.no_grad():
        assert rr.grpo_fused_head_fits(torch.zeros(4, 3, requires_grad = True))


@pytest.mark.parametrize("loss_type", ["grpo", "bnpo", "dr_grpo", "dapo", "cispo", "luspo", "sapo"])
@pytest.mark.parametrize("sample_weight", [False, True])
def test_token_weights_reproduce_grpo_compute_loss(loss_type, sample_weight):
    g = torch.Generator().manual_seed(1)
    mask = (torch.rand(5, 7, generator = g) > 0.3).long()
    mask[3] = 0
    loss_i = torch.randn(5, 7, generator = g, dtype = torch.float64)
    kw = dict(loss_type = loss_type, max_completion_length = 9, num_items_in_batch = torch.tensor(17.0),
              current_gradient_accumulation_steps = 3, steps_per_generation = 2, num_processes = 2)
    if sample_weight:
        kw["sample_weight"] = torch.rand(5, generator = g)
    # grpo_compute_loss with cispo and a zero advantage-free trick: build loss_i from new = loss_i, old = None
    # would not isolate the normaliser, so compare against its normaliser branch directly.
    w = rr.grpo_fused_token_weights(mask, kw).double()
    m = mask.double()
    ga, spg, npr = 3, 2, 2
    if sample_weight:
        want = (loss_i * m * kw["sample_weight"].double()[:, None]).sum() / (1.0 / npr * ga / spg)
    elif loss_type in ("grpo", "sapo"):
        want = ((loss_i * m).sum(-1) / m.sum(-1).clamp(min = 1.0)).mean() / ga
    elif loss_type == "bnpo":
        want = (loss_i * m).sum() / m.sum().clamp(min = 1.0) / ga
    elif loss_type == "dr_grpo":
        want = (loss_i * m).sum() / (5 * 9) / ga
    elif loss_type in ("cispo", "dapo"):
        want = (loss_i * m).sum() / (17.0 / npr * ga / spg)
    else:
        want = (loss_i * m).sum(-1).mean() / ga
    torch.testing.assert_close((w * loss_i).sum(), want, rtol = 1e-6, atol = 1e-9)


def test_unsloth_kl_clamps_match_grpo_compute_loss():
    """The op's k3 term saturates where grpo_compute_loss clamps: |ref - logp| > 20 and a KL past 10."""
    lp = torch.tensor([-1.0, -2.0, -30.0, -0.5, -3.0], dtype = torch.float64, requires_grad = True)
    ref = torch.tensor([-1.2, -25.0, -1.0, -14.0, -2.9], dtype = torch.float64)
    old = torch.tensor([-1.1, -2.1, -29.0, -0.4, -3.2], dtype = torch.float64)
    adv = torch.tensor([0.5, -1.0, 0.3, 1.0, -0.2], dtype = torch.float64)
    mask = torch.ones(1, 5)
    for bias_kl in (False, True):
        kw = dict(loss_type = "bnpo", use_bias_correction_kl = bias_kl, epsilon_low = 0.2, epsilon_high = 0.2)
        want = rr.grpo_compute_loss(ref[None], lp[None], old[None], None, torch.zeros(1, 5, dtype = torch.long),
                                    mask, 0.1, adv[None], **kw)[0]
        (gw,) = torch.autograd.grad(want, [lp])
        ell, dl = fg._token_terms(lp.detach(), adv, old, ref, None, 0, 0.2, 0.2, 0.0, 0.1, 1.0, 1.05, bias_kl)
        torch.testing.assert_close(ell.mean(), want.detach())
        torch.testing.assert_close(dl / 5, gw)


def test_on_policy_without_old_logps():
    lp = torch.tensor([-1.0, -2.0, -0.3], dtype = torch.float64, requires_grad = True)
    adv = torch.tensor([0.5, -1.0, 0.3], dtype = torch.float64)
    ref = lp.detach() - 0.4
    for kind, lt in ((0, "grpo"), (1, "cispo"), (2, "sapo")):
        want = rr.grpo_compute_loss(ref[None], lp[None], None, None, torch.zeros(1, 3, dtype = torch.long),
                                    torch.ones(1, 3), 0.05, adv[None], loss_type = lt, epsilon_high = 4.0 if kind == 1 else 0.2,
                                    sapo_temperature_pos = 1.0, sapo_temperature_neg = 1.05, num_items_in_batch = 3.0)[0]
        (gw,) = torch.autograd.grad(want, [lp])
        ell, dl = fg._token_terms(lp.detach(), adv, None, ref, None, kind, 0.2, 4.0 if kind == 1 else 0.2, 0.0, 0.05,
                                  1.0, 1.05, False)
        torch.testing.assert_close(ell.mean(), want.detach())
        torch.testing.assert_close(dl / 3, gw)


def test_fused_route_loss_compiles_without_graph_breaks():
    """The routed op call (precomputed weights, on-policy) is one fullgraph frame."""
    g = torch.Generator().manual_seed(2)
    h = torch.randn(1, 9, H, generator = g)
    W = torch.randn(V, H, generator = g)
    t = torch.randint(0, V, (1, 9), generator = g)
    mask = torch.ones(1, 9, dtype = torch.bool)
    adv = torch.randn(1, 9, generator = g)
    w = torch.rand(1, 9, generator = g)
    e = -0.01 * torch.rand(1, 9, generator = g)
    ref = torch.randn(1, 9, generator = g) - 2

    def f(hh, WW):
        return fg.fused_linear_grpo_loss(hh, WW, t, mask = mask, advantages = adv, old_logps = None, ref_logps = ref,
                                         beta = 0.04, token_weights = w, entropy_weights = e, scaling = 2.0)[0]
    hh, WW = h.clone().requires_grad_(), W.clone().requires_grad_()
    eager = torch.autograd.grad(f(hh, WW), [hh, WW])
    torch._dynamo.reset()
    from torch._dynamo.utils import counters
    counters.clear()
    hc, Wc = h.clone().requires_grad_(), W.clone().requires_grad_()
    got = torch.autograd.grad(torch.compile(f, fullgraph = True, backend = "aot_eager")(hc, Wc), [hc, Wc])
    for x, y in zip(got, eager):
        torch.testing.assert_close(x, y)
    assert sum(counters["graph_break"].values()) == 0
