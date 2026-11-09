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

"""Fused preference logps and no-grad fused CE scoring compile with fullgraph=True; traced code must avoid boolean-mask
row compaction (data-dependent size) and GradScaler.get_scale(), which Dynamo skips."""

import os

import pytest
import torch
from torch._dynamo.utils import counters

# UNSLOTH_ALLOW_CPU is read once at import, so set it only around the zoo imports and restore the old value; a module-scope write leaks into later tests.
_PREVIOUS_ALLOW_CPU = os.environ.get("UNSLOTH_ALLOW_CPU")
try:
    os.environ.setdefault("UNSLOTH_ALLOW_CPU", "1")
    import unsloth_zoo.fused_losses.preference_loss as preference
    from unsloth_zoo.fused_losses import forward_adapter
    from unsloth_zoo.fused_losses.preference_loss import fused_linear_token_logps
finally:
    if _PREVIOUS_ALLOW_CPU is None:
        os.environ.pop("UNSLOTH_ALLOW_CPU", None)
    else:
        os.environ["UNSLOTH_ALLOW_CPU"] = _PREVIOUS_ALLOW_CPU


def _inputs(seq_len, valid, dtype = torch.float32, vocab = 40, hidden = 16, head_grad = True):
    generator = torch.Generator().manual_seed(seq_len * 100 + valid)
    states = torch.randn(2, seq_len, hidden, generator = generator).to(dtype).requires_grad_()
    weight = torch.randn(vocab, hidden, generator = generator).to(dtype).requires_grad_(head_grad)
    labels = torch.randint(vocab, (2, seq_len), generator = generator)
    mask = torch.zeros(2, seq_len, dtype = torch.bool)
    mask[0, seq_len - valid:] = True
    mask[1, : max(valid - 1, 0)] = True
    return states, weight, labels, mask


def _score(states, weight, labels, mask):
    out = fused_linear_token_logps(
        states, weight, None, labels, mask, chunk_size = 3,
        return_mean_logits = True, return_log_sum_sq_probs = True,
    )
    return out["per_token_logps"], out["mean_logits"], out["log_sum_sq_probs"]


def _run(fn, args):
    states, weight = args[0], args[1]
    logps, mean_logits, lssp = fn(*args)
    (logps * torch.linspace(0.5, 1.5, logps.shape[-1])).sum().backward()
    grads = (states.grad, weight.grad if weight.requires_grad else None)
    return logps.detach(), mean_logits, lssp, grads


@pytest.mark.parametrize("dynamic", [None, True])
def test_preference_logps_fullgraph_no_count_recompiles(dynamic):
    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(_score, backend = "aot_eager", fullgraph = True, dynamic = dynamic)
    # Same shape, every valid token count, including none and all.
    for valid in (8, 5, 1, 0, 7):
        eager = _run(_score, _inputs(8, valid))
        got = _run(compiled, _inputs(8, valid))
        for a, b in zip(eager[:3], got[:3]):
            assert torch.equal(a, b)
        for a, b in zip(eager[3], got[3]):
            assert torch.equal(a, b)
    assert sum(counters["graph_break"].values()) == 0
    assert counters["stats"]["unique_graphs"] == 1, dict(counters["stats"])


def test_preference_logps_dynamic_lengths_one_graph():
    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(_score, backend = "aot_eager", fullgraph = True, dynamic = True)
    for seq_len, valid in ((8, 8), (8, 5), (12, 9), (16, 12), (16, 3)):
        eager = _run(_score, _inputs(seq_len, valid))
        got = _run(compiled, _inputs(seq_len, valid))
        assert torch.equal(eager[0], got[0])
        assert torch.equal(eager[3][0], got[3][0]) and torch.equal(eager[3][1], got[3][1])
    assert sum(counters["graph_break"].values()) == 0
    assert counters["stats"]["unique_graphs"] == 1, dict(counters["stats"])


def test_preference_logps_explain_zero_breaks():
    torch._dynamo.reset()
    explanation = torch._dynamo.explain(_score)(*_inputs(8, 5))
    assert explanation.graph_break_count == 0, explanation.break_reasons
    assert explanation.graph_count == 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_grad", [True, False])
def test_preference_logps_matches_autograd_function_path(dtype, head_grad, monkeypatch):
    # The custom op wraps the same compaction and chunk loops and must be bitwise equal to the autograd.Function path,
    # including the vocab sliced backward of a trainable half head.
    legacy = getattr(preference, "_fused_linear_token_logps_function", None)
    if legacy is None:
        pytest.skip("no custom op path")
    monkeypatch.setattr(preference, "_FUSED_LINEAR", False)
    def old(states, weight, labels, mask):
        out = legacy(states, weight, None, labels, mask, 0.0, 0.0, 0.0, 3, True, True)
        return out["per_token_logps"], out["mean_logits"], out["log_sum_sq_probs"]
    # (64, 60) selects more than 2 * vocab rows, so a trainable half head takes the per chunk backward.
    for seq_len, valid in ((8, 6), (64, 60), (8, 0)):
        a = _run(old, _inputs(seq_len, valid, dtype, head_grad = head_grad))
        b = _run(_score, _inputs(seq_len, valid, dtype, head_grad = head_grad))
        for x, y in zip(a[:3], b[:3]):
            assert x.dtype == y.dtype and torch.equal(x, y)
        for x, y in zip(a[3], b[3]):
            assert (x is None) == (y is None)
            if x is not None:
                assert x.dtype == y.dtype and torch.equal(x, y)
    # Unselected rows get exact zero gradient.
    assert torch.count_nonzero(b[3][0][~_inputs(8, 0)[3]]) == 0


def test_preference_logps_custom_op_opcheck():
    if not hasattr(torch.library, "opcheck"):
        pytest.skip("torch.library.opcheck unavailable")
    states, weight, labels, mask = _inputs(8, 5)
    torch.library.opcheck(
        preference._token_logps_op,
        (states.reshape(-1, 16), weight, None, labels.reshape(-1), mask.reshape(-1), 3,
         0.0, 0.0, 0.0, True, True, False, True),
        test_utils = ("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
    )


@pytest.mark.parametrize("seq_len, valid", [(8, 5), (64, 60)])
def test_preference_logps_custom_op_opcheck_half_head_hidden_grad(seq_len, valid):
    # Trainable fp16 head with hidden grads: (8, 5) keeps the vocab sliced backward, while (64, 60), selecting more than
    # 2 * vocab rows, switches to the per chunk backward; real op must return lse at the fake shape.
    if not hasattr(torch.library, "opcheck"):
        pytest.skip("torch.library.opcheck unavailable")
    states, weight, labels, mask = _inputs(seq_len, valid, dtype = torch.float16)
    args = (states.reshape(-1, 16), weight, None, labels.reshape(-1), mask.reshape(-1), 3,
            0.0, 0.0, 0.0, True, True, True, True)
    real = preference._token_logps_op(*args)
    with torch._subclasses.fake_tensor.FakeTensorMode() as mode:
        fake = preference._token_logps_op(*(
            mode.from_tensor(a) if isinstance(a, torch.Tensor) else a for a in args
        ))
    assert [tuple(t.shape) for t in real] == [tuple(t.shape) for t in fake]
    assert bool(real[4][1]) == (seq_len == 8)
    torch.library.opcheck(
        preference._token_logps_op, args,
        test_utils = ("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
    )


def _real_scaler():
    scaler = torch.amp.GradScaler("cpu")
    scaler.scale(torch.ones(()))  # initialise _scale, as after the first training step
    return scaler


def test_fused_ce_no_grad_fullgraph_with_real_grad_scaler():
    scaler = _real_scaler()
    head = torch.nn.Linear(8, 16, bias = False).half().requires_grad_(False)
    states = torch.randn(2, 4, 8, dtype = torch.float16)
    labels = torch.randint(16, (2, 4))

    def score(x):
        return forward_adapter.unsloth_fused_lm_head_loss(
            x, head, labels, accelerator_scaler = scaler, n_chunks = 1,
        )

    with torch.no_grad():
        eager = score(states)
        torch._dynamo.reset()
        explanation = torch._dynamo.explain(score)(states)
        assert explanation.graph_break_count == 0, explanation.break_reasons
        torch._dynamo.reset()
        got = torch.compile(score, backend = "aot_eager", fullgraph = True)(states)
    # Eager may run the inner chunk compiled by inductor, so compare both to a float64 oracle.
    logits = states[:, :-1].double() @ head.weight.double().t()
    oracle = torch.nn.functional.cross_entropy(logits.reshape(-1, 16), labels[:, 1:].reshape(-1))
    torch.testing.assert_close(eager.double(), oracle, rtol = 1e-3, atol = 2e-3)  # fp16 logits
    torch.testing.assert_close(got.double(), oracle, rtol = 1e-3, atol = 2e-3)  # fp16 logits


def test_fused_ce_training_still_reads_grad_scaler(monkeypatch):
    calls = []
    scaler = _real_scaler()
    real = scaler.get_scale
    monkeypatch.setattr(scaler, "get_scale", lambda: calls.append(1) or real())
    head = torch.nn.Linear(8, 16, bias = False).half().requires_grad_(False)
    states = torch.randn(2, 4, 8, dtype = torch.float16, requires_grad = True)
    labels = torch.randint(16, (2, 4))
    with torch.no_grad():
        forward_adapter.unsloth_fused_lm_head_loss(states, head, labels, accelerator_scaler = scaler, n_chunks = 1)
    assert calls == []
    loss = forward_adapter.unsloth_fused_lm_head_loss(states, head, labels, accelerator_scaler = scaler, n_chunks = 1)
    assert calls == [1]
    loss.backward()
    assert states.grad is not None and torch.isfinite(states.grad).all()
