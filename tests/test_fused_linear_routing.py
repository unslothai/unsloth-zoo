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

"""Every fused lm_head entry point (SFT CE, chunked / masked log-probs, the preference API, GRPO) reaches the fused
linear op by default, keeps its contract against an fp64 oracle and the pre-op path, and traces as one graph."""

import pytest
import torch

import unsloth_zoo.fused_losses.chunked_logprob as cl
import unsloth_zoo.fused_losses.cross_entropy_loss as cel
import unsloth_zoo.fused_losses.fused_linear as fl
import unsloth_zoo.fused_losses.preference_loss as pl
from unsloth_zoo import rl_replacements as rr

B, S, H, V = 2, 12, 16, 50


def _calls(key):
    return fl.CALLS[key]


def _ce_oracle(hidden, weight, bias, labels, mask, shift, n_items, smoothing = 0.0, logits_bias = None):
    hd = hidden.detach().double().requires_grad_(True)
    z = hd @ weight.detach().double().t()
    if bias is not None:
        z = z + bias.detach().double()
    if logits_bias is not None:
        z = z + logits_bias.double()
    labels = labels.clone()
    if shift:
        labels = torch.cat((labels[:, 1:], torch.full_like(labels[:, :1], -100)), dim = 1)
        if mask is not None:
            labels[:, :-1][mask[:, 1:] == 0] = -100
    elif mask is not None:
        labels[mask == 0] = -100
    loss = torch.nn.functional.cross_entropy(
        z.reshape(-1, V), labels.reshape(-1), reduction = "sum", label_smoothing = smoothing,
    )
    count = (labels != -100).sum()
    loss = loss / (n_items if n_items is not None else max(int(count), 1))
    loss.backward()
    return loss.detach(), hd.grad


@pytest.mark.parametrize("kwargs", [
    dict(), dict(label_smoothing = 0.1), dict(logits_bias = torch.randn(V)),
])
@pytest.mark.parametrize("shift", [True, False])
def test_fused_ce_reaches_the_op_and_matches_fp64(kwargs, shift):
    torch.manual_seed(0)
    hidden = torch.randn(B, S, H, requires_grad = True)
    weight = torch.randn(V, H) / 2
    labels = torch.randint(0, V, (B, S))
    labels[:, :3] = -100
    mask = torch.ones(B, S, dtype = torch.long)
    mask[1, -4:] = 0
    before = _calls("ce")
    loss = cel.unsloth_fused_ce_loss(None, hidden, weight, None, labels, mask = mask, shift_labels = shift, **kwargs)
    assert _calls("ce") == before + 1
    loss.backward()
    ref, gh = _ce_oracle(
        hidden, weight, None, labels, mask, shift, None, kwargs.get("label_smoothing", 0.0), kwargs.get("logits_bias"),
    )
    torch.testing.assert_close(loss.double(), ref, rtol = 1e-5, atol = 1e-6)
    torch.testing.assert_close(hidden.grad.double(), gh, rtol = 1e-4, atol = 1e-7)


def test_fp32_head_grad_opt_in_reaches_the_op(monkeypatch):
    monkeypatch.setattr(cel, "FP32_HEAD_GRAD", True)
    hidden = torch.randn(B, S, H, requires_grad = True)
    weight = torch.randn(V, H, requires_grad = True)
    before = _calls("ce")
    cel.unsloth_fused_ce_loss(None, hidden, weight, None, torch.randint(0, V, (B, S))).backward()
    assert _calls("ce") == before + 1 and weight.grad.isfinite().all()


def test_fused_ce_does_no_gradient_work_under_no_grad():
    weight = torch.randn(V, H, requires_grad = True)
    with torch.no_grad():
        loss = cel.unsloth_fused_ce_loss(None, torch.randn(B, S, H), weight, None, torch.randint(0, V, (B, S)))
    assert not loss.requires_grad and weight.grad is None


class _Scaler:
    """A GradScaler whose host read must not happen on the fused path."""

    def __init__(self, scale):
        self._scale = torch.full((1,), scale)

    def is_enabled(self):
        return True

    def get_scale(self):
        raise AssertionError("get_scale() syncs the host")


def test_grad_scaler_is_read_as_a_tensor():
    torch.manual_seed(0)
    hidden = torch.randn(B, S, H, requires_grad = True)
    weight = torch.randn(V, H)
    labels = torch.randint(0, V, (B, S))
    loss = cel.unsloth_fused_ce_loss(None, hidden, weight, None, labels, scaling = _Scaler(1024.0))
    loss.backward()
    _, gh = _ce_oracle(hidden, weight, None, labels, None, True, None)
    torch.testing.assert_close(hidden.grad.double(), gh, rtol = 1e-4, atol = 1e-7)


def _graph_breaks(fn, *args):
    # fullgraph raises on any break; backward runs outside (Tensor.backward is not traceable), through AOTAutograd.
    import torch._dynamo
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    counters.clear()
    out = torch.compile(fn, backend = "aot_eager", fullgraph = True)(*args)
    out.backward()
    out = out.detach()
    breaks = sum(counters["graph_break"].values())
    torch._dynamo.reset()
    return out, breaks


@pytest.mark.skipif(type(torch.Tensor.to).__name__ != "method_descriptor",
                    reason = "Tensor.to is patched (Unsloth's Apple Silicon MLX mode), which Dynamo cannot trace")
def test_fused_ce_traces_as_one_graph():
    torch.manual_seed(0)
    weight = torch.randn(V, H)
    labels = torch.randint(0, V, (B, S))
    labels[:, :5] = -100

    def step(hidden, labels):
        return cel.unsloth_fused_ce_loss(None, hidden, weight, None, labels, n_items = torch.tensor(9))

    hidden = torch.randn(B, S, H, requires_grad = True)
    loss, breaks = _graph_breaks(step, hidden, labels)
    assert breaks == 0
    hidden2 = hidden.detach().clone().requires_grad_(True)
    step(hidden2, labels).backward()
    torch.testing.assert_close(loss, cel.unsloth_fused_ce_loss(None, hidden2, weight, None, labels, n_items = torch.tensor(9)).detach())
    torch.testing.assert_close(hidden.grad, hidden2.grad)


_KILL_SWITCH_SCRIPT = """
import torch
from unsloth_zoo.fused_losses.cross_entropy_loss import unsloth_fused_ce_loss
dev = "{device}"
torch.manual_seed(0)
h = torch.randn(2, 16, 32, device = dev, requires_grad = True)
W = torch.randn(64, 32, device = dev) * 0.1
lab = torch.randint(0, 64, (2, 16), device = dev); lab[:, :5] = -100
f = lambda h, W, lab: unsloth_fused_ce_loss(None, h, W, None, lab, n_items = None, logit_softcapping = 0)
out = torch.compile(f)(h, W, lab); out.backward()
h2 = h.detach().clone().requires_grad_(True); ref = f(h2, W, lab); ref.backward()
torch.testing.assert_close(out.detach(), ref.detach()); torch.testing.assert_close(h.grad, h2.grad)
print("KILL_SWITCH_OK")
"""


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_kill_switch_ce_still_compiles(device):
    # UNSLOTH_FUSED_LINEAR=0 (read at import: fresh process): the eager chunked CE must stay opaque to dynamo.
    import os
    import subprocess
    import sys
    env = dict(os.environ, UNSLOTH_FUSED_LINEAR = "0", UNSLOTH_IS_PRESENT = "1")
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""  # no CUDA context (and no VRAM) for the CPU leg
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    p = subprocess.run([sys.executable, "-c", _KILL_SWITCH_SCRIPT.format(device = device)], env = env,
                       capture_output = True, text = True, timeout = 600)
    assert "KILL_SWITCH_OK" in p.stdout, (p.stdout + p.stderr)[-3000:]


def test_masked_logprobs_trace_as_one_graph():
    torch.manual_seed(0)
    weight = torch.randn(V, H)
    labels = torch.randint(0, V, (B, S))
    labels[0, :7] = -100

    def dpo(hidden, labels):
        lp = cl.unsloth_masked_logprobs(hidden, weight, labels).sum(-1)
        with torch.no_grad():
            ref = cl.unsloth_masked_logprobs(hidden.detach() * 1.01, weight, labels).sum(-1)
        return -torch.nn.functional.logsigmoid(0.1 * ((lp[0] - ref[0]) - (lp[1] - ref[1])))

    hidden = torch.randn(B, S, H, requires_grad = True)
    _, breaks = _graph_breaks(dpo, hidden, labels)
    assert breaks == 0 and hidden.grad.isfinite().all()


def _grpo_inputs(seed = 1):
    gen = torch.Generator().manual_seed(seed)
    states = torch.randn(3, 7, H, generator = gen).requires_grad_(True)
    weight = (0.5 * torch.randn(V, H, generator = gen)).requires_grad_(False)
    index = torch.randint(0, V, (3, 7), generator = gen)
    upstream = torch.randn(3, 7, generator = gen)
    return states, weight, index, upstream


@pytest.mark.parametrize("transform", [(0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 30.0, 0.7), (1.5, 2.0, 0.0, 1.2)], ids = str)
def test_grpo_route_reaches_the_op_with_and_without_grad(transform):
    states, weight, index, upstream = _grpo_inputs()
    before = _calls("logprobs")
    got = rr.grpo_chunked_hidden_states_logprobs(states, weight, index, 4, *transform)
    (got * upstream).sum().backward()
    with torch.no_grad():
        scored = rr.grpo_chunked_hidden_states_logprobs(states, weight, index, 4, *transform)
    assert _calls("logprobs") == before + 2
    # Old / reference logprobs and the policy's come from one kernel: the on-policy ratio starts at exactly 1.
    assert torch.equal(scored, got.detach())
    ref_states = states.detach().clone().requires_grad_(True)
    ref = rr.chunked_hidden_states_selective_log_softmax(ref_states, weight, index, 4, *transform)
    (ref * upstream).sum().backward()
    torch.testing.assert_close(got.detach(), ref.detach(), atol = 1e-5, rtol = 0)
    torch.testing.assert_close(states.grad, ref_states.grad, atol = 1e-5, rtol = 0)


def test_grpo_route_passes_max_rows_per_chunk(monkeypatch):
    seen = []
    real = fl.fused_linear_logprobs
    # A cap on the kernel's own chunk size, never a fixed chunk (that would bypass the memory budget).
    monkeypatch.setattr(fl, "fused_linear_logprobs",
                        lambda *a, **k: seen.append((k.get("rows_per_chunk"), k.get("max_rows_per_chunk"))) or real(*a, **k))
    states, weight, index, _ = _grpo_inputs()
    rr.grpo_chunked_hidden_states_logprobs(states, weight, index, 4, max_rows_per_chunk = 3)
    monkeypatch.setenv("UNSLOTH_GRPO_MAX_ROWS_PER_CHUNK", "5")
    rr.grpo_chunked_hidden_states_logprobs(states, weight, index, 4)
    monkeypatch.delenv("UNSLOTH_GRPO_MAX_ROWS_PER_CHUNK")
    rr.grpo_chunked_hidden_states_logprobs(states, weight, index, 4)
    assert seen == [(None, 3), (None, 5), (None, None)]


def test_grpo_truncated_support_keeps_the_old_function():
    states, weight, index, _ = _grpo_inputs()
    before = _calls("logprobs")
    rr.grpo_chunked_hidden_states_logprobs(states, weight, index, 4, support_top_k = 5)
    assert _calls("logprobs") == before


def test_grpo_kill_switch_keeps_one_scorer_for_every_pass(monkeypatch):
    # UNSLOTH_FUSED_LINEAR=0: grad and no-grad passes both take the old function.
    monkeypatch.setattr(cl, "_FUSED_LINEAR", False)
    seen = []
    real = rr.chunked_hidden_states_selective_log_softmax
    monkeypatch.setattr(rr, "chunked_hidden_states_selective_log_softmax",
                        lambda *a, **k: seen.append(torch.is_grad_enabled()) or real(*a, **k))
    states, weight, index, _ = _grpo_inputs()
    got = rr.grpo_chunked_hidden_states_logprobs(states, weight, index, 4)
    with torch.no_grad():
        scored = rr.grpo_chunked_hidden_states_logprobs(states, weight, index, 4)
    assert seen == [True, False]
    assert torch.equal(scored, got.detach())


def test_recompute_entropy_route_matches_the_torch_loop():
    states, weight, index, upstream = _grpo_inputs(seed = 3)
    gen = torch.Generator().manual_seed(4)
    up_entropy = torch.randn(3, 7, generator = gen)
    before = _calls("logprobs")
    logps, ent = rr.chunked_hidden_states_selective_log_softmax_recompute(
        states, weight, index, 4, 0.0, 0.0, 10.0, 0.9, return_entropy = True,
    )
    assert _calls("logprobs") == before + 1
    ((logps * upstream).sum() + (ent * up_entropy).sum()).backward()
    ref_states = states.detach().clone().requires_grad_(True)
    ref_logps, ref_ent = rr.chunked_hidden_states_selective_log_softmax_recompute(
        ref_states, weight, index, 4, 0.0, 0.0, 10.0, 0.9, return_entropy = True, fused_linear = False,
    )
    ((ref_logps * upstream).sum() + (ref_ent * up_entropy).sum()).backward()
    torch.testing.assert_close(logps.detach(), ref_logps.detach(), atol = 1e-5, rtol = 0)
    torch.testing.assert_close(ent.detach(), ref_ent.detach(), atol = 1e-5, rtol = 0)
    torch.testing.assert_close(states.grad, ref_states.grad, atol = 1e-5, rtol = 0)


@pytest.mark.parametrize("dtype,routed", [(torch.float32, True), (torch.float64, False)])
def test_preference_api_routes_by_head_dtype(monkeypatch, dtype, routed):
    torch.manual_seed(0)
    hidden = torch.randn(B, S, H, dtype = dtype, requires_grad = True)
    weight = torch.randn(V, H, dtype = dtype)
    labels = torch.randint(0, V, (B, S))
    mask = labels % 3 != 0
    kwargs = dict(logit_softcapping = 5.0, return_mean_logits = True, return_log_sum_sq_probs = True)
    before = _calls("logprobs")
    out = pl.fused_linear_token_logps(hidden, weight, None, labels, mask, **kwargs)
    assert (_calls("logprobs") > before) == routed
    assert set(out) == {"per_token_logps", "mean_logits", "log_sum_sq_probs"}
    out["per_token_logps"].sum().backward()
    for key in out:
        assert out[key].shape == mask.shape and (out[key][~mask] == 0).all()
    monkeypatch.setattr(pl, "_FUSED_LINEAR", False)
    want = pl.fused_linear_token_logps(hidden.detach(), weight, None, labels, mask, **kwargs)
    for key in out:
        torch.testing.assert_close(out[key].detach(), want[key], atol = 1e-5, rtol = 1e-5)


def test_linear_mean_logits_softcap_matches_the_torch_loop(monkeypatch):
    torch.manual_seed(0)
    hidden = torch.randn(B, S, H)
    weight = torch.randn(V, H)
    bias = torch.randn(V)
    got = pl.linear_mean_logits(hidden, weight, bias, 1.5, 0.0, 4.0)
    monkeypatch.setattr(pl, "_FUSED_LINEAR", False)
    want = pl.linear_mean_logits(hidden, weight, bias, 1.5, 0.0, 4.0)
    torch.testing.assert_close(got, want, atol = 1e-5, rtol = 1e-5)


def test_preference_explicit_chunk_budget_keeps_the_chunked_function(monkeypatch):
    # An explicit UNSLOTH_FUSED_PREFERENCE_CHUNK_MB keeps the chunked Function rather than being ignored.
    import importlib
    torch.manual_seed(0)
    hidden = torch.randn(B, S, H, requires_grad = True)
    weight = torch.randn(V, H)
    labels = torch.randint(0, V, (B, S))
    mask = labels % 3 != 0
    monkeypatch.setenv("UNSLOTH_FUSED_PREFERENCE_CHUNK_MB", "1")
    try:
        importlib.reload(pl)
        before = _calls("logprobs")
        out = pl.fused_linear_token_logps(hidden, weight, None, labels, mask)
        assert _calls("logprobs") == before
        out["per_token_logps"].sum().backward()
    finally:
        monkeypatch.delenv("UNSLOTH_FUSED_PREFERENCE_CHUNK_MB")
        importlib.reload(pl)
    before = _calls("logprobs")
    pl.fused_linear_token_logps(hidden.detach(), weight, None, labels, mask)
    assert _calls("logprobs") > before
