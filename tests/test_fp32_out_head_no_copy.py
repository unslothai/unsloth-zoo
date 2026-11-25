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

"""An fp32 lm_head on a 16-bit CUDA head projects with a float32-out GEMM (aten::mm.dtype), never a float32 copy of
the head: the non-fused scorer and the kept-set scorer route through fp32_out_linear, and its backward matches the
float32-copy path. The CPU tests stand in for mm.dtype (CUDA / ROCm only); the CUDA test runs the real op."""

import pytest
import torch
from torch.overrides import TorchFunctionMode

from unsloth_zoo import rl_replacements as rr
from unsloth_zoo.fused_losses import chunked_logprob as cl


def _inputs(seed = 0, dtype = torch.bfloat16, device = "cpu"):
    gen = torch.Generator().manual_seed(seed)
    weight = (torch.randn(300, 32, generator = gen) * 0.2).to(device = device, dtype = dtype)
    hidden = (torch.randn(2, 9, 32, generator = gen) * 2).to(device = device, dtype = dtype)
    index = torch.randint(0, 300, (2, 9), generator = gen).to(device)
    return hidden, weight, index


class _CountHeadCopies(TorchFunctionMode):
    def __init__(self, shape):
        super().__init__()
        self.shape, self.copies = shape, 0

    def __torch_function__(self, func, types, args = (), kwargs = None):
        out = func(*args, **(kwargs or {}))
        if isinstance(out, torch.Tensor) and out.shape == self.shape and out.dtype == torch.float32 \
                and args and isinstance(args[0], torch.Tensor) and args[0].shape == self.shape \
                and args[0].dtype != torch.float32:
            self.copies += 1
        return out


def _cpu_stand_in(monkeypatch):
    calls = []

    def fake_linear(x, w, b = None):
        calls.append(w.dtype)
        out = torch.nn.functional.linear(x.to(w.dtype).double(), w.double(), None if b is None else b.double())
        return out.float()

    monkeypatch.setattr(cl, "fp32_out_linear_ok", lambda w: True)
    monkeypatch.setattr(cl, "fp32_out_linear", fake_linear)
    return calls


def _reference(hidden, weight, index):
    logits = hidden.double() @ weight.double().t()
    return logits.gather(-1, index.unsqueeze(-1)).squeeze(-1) - logits.logsumexp(-1)


def test_non_fused_scorer_uses_fp32_out_without_a_head_copy(monkeypatch):
    calls = _cpu_stand_in(monkeypatch)
    hidden, weight, index = _inputs()
    with torch.no_grad(), _CountHeadCopies(weight.shape) as mode:
        out = rr.chunked_hidden_states_selective_log_softmax(
            hidden, weight, index, 3, 0.0, 0.0, 0.0, 1.0, fp32_lm_head = True, fused_linear = False,
        )
    assert mode.copies == 0 and calls == [torch.bfloat16] * 3
    assert out.dtype == torch.float32
    torch.testing.assert_close(out.double(), _reference(hidden, weight, index), atol = 1e-5, rtol = 0)


def test_non_fused_scorer_copies_the_head_where_fp32_out_is_missing():
    # CPU has no mm.dtype: one fp32 head copy per call, as before.
    hidden, weight, index = _inputs(1)
    with torch.no_grad(), _CountHeadCopies(weight.shape) as mode:
        out = rr.chunked_hidden_states_selective_log_softmax(
            hidden, weight, index, 3, 0.0, 0.0, 0.0, 1.0, fp32_lm_head = True, fused_linear = False,
        )
    assert mode.copies == 1
    torch.testing.assert_close(out.double(), _reference(hidden, weight, index), atol = 1e-4, rtol = 0)


def test_kept_set_scorer_uses_fp32_out_without_a_head_copy(monkeypatch):
    calls = _cpu_stand_in(monkeypatch)
    hidden, weight, index = _inputs(2)
    hidden = hidden.requires_grad_(True)
    with _CountHeadCopies(weight.shape) as mode:
        out = rr.grpo_kept_set_logprobs(
            hidden, weight, index, 3, 0.0, 0.0, 0.0, 1.0, support_top_k = 40, fp32_lm_head = True,
        )
        out[torch.isfinite(out)].sum().backward()
    assert mode.copies == 0 and calls and set(calls) == {torch.bfloat16}
    assert out.dtype == torch.float32
    # The hidden-state gradient comes back in its 16-bit dtype, from 16-bit GEMMs (no fp32 head in backward either).
    assert hidden.grad.dtype == torch.bfloat16 and torch.isfinite(hidden.grad.float()).all()


def test_trainable_and_fp32_heads_keep_their_previous_paths():
    # A trainable head keeps the fp32 copy (its gradient accumulates over chunks in float32), and a float32 head
    # never takes the fused shortcut (autocast would round it) nor the unset default.
    weight = torch.zeros(300, 32, dtype = torch.bfloat16)
    assert not cl.fp32_out_linear_ok(weight.requires_grad_(True))
    fp32_head = torch.zeros(300, 32)
    assert not cl.fused_fp32_logits_ok(fp32_head)
    assert not cl.fp32_lm_head_auto_ok(fp32_head)
    if torch.cuda.is_available():
        assert not cl.fused_fp32_logits_ok(fp32_head.cuda())
        assert not cl.fp32_lm_head_auto_ok(fp32_head.cuda())
        assert cl.fp32_lm_head_auto_ok(fp32_head.cuda().bfloat16())


@pytest.mark.skipif(not (torch.cuda.is_available() and cl._MM_OUT_DTYPE), reason = "needs CUDA and aten::mm.dtype")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fp32_out_linear_matches_the_fp32_copy_on_cuda(dtype):
    hidden, weight, index = _inputs(3, dtype, "cuda")
    x = hidden.reshape(-1, hidden.shape[-1])
    assert cl.fp32_out_linear_ok(weight)
    a_x, a_w = x.clone().requires_grad_(True), weight.clone().requires_grad_(True)
    b_x, b_w = x.clone().requires_grad_(True), weight.clone().requires_grad_(True)
    out = cl.fp32_out_linear(a_x, a_w)
    ref = torch.nn.functional.linear(b_x.float(), b_w.float())
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref, atol = 1e-4, rtol = 1e-5)
    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad)
    assert a_x.grad.dtype == dtype and a_w.grad.dtype == dtype
    for got, want in ((a_x.grad, b_x.grad), (a_w.grad, b_w.grad)):
        assert ((got.float() - want.float()).norm() / want.float().norm()).item() < 1e-2


def test_unset_default_covers_cuda_heads_only(monkeypatch):
    # fp32_lm_head_cuda (the unset default on CUDA hosts) leaves a CPU head on its 16-bit path, fused route included.
    hidden, weight, index = _inputs(4)
    with torch.no_grad():
        for fused in (True, False):
            auto = rr.chunked_hidden_states_selective_log_softmax(
                hidden, weight, index, 3, 0.0, 0.0, 0.0, 1.0, fp32_lm_head = False, fp32_lm_head_cuda = True,
                fused_linear = fused,
            )
            off = rr.chunked_hidden_states_selective_log_softmax(
                hidden, weight, index, 3, 0.0, 0.0, 0.0, 1.0, fp32_lm_head = False, fp32_lm_head_cuda = False,
                fused_linear = fused,
            )
            assert torch.equal(auto, off)
    monkeypatch.delenv("UNSLOTH_VLLM_FP32_LM_HEAD", raising = False)
    assert rr.grpo_fp32_lm_head_default(weight) is False
    monkeypatch.setenv("UNSLOTH_VLLM_FP32_LM_HEAD", "1")
    assert rr.grpo_fp32_lm_head_default(weight) is True


def _old_half_logit_grad(grad, dtype):
    return grad.to(dtype), None


@pytest.mark.skipif(not (torch.cuda.is_available() and cl._MM_OUT_DTYPE), reason = "needs CUDA and aten::mm.dtype")
@pytest.mark.parametrize("rescale", [True, False])
def test_fp16_small_logit_gradients_survive(monkeypatch, rescale):
    # A wide vocabulary under a small upstream weight (GRPO's 1 / tokens) puts each non-target logit gradient below
    # float16's subnormal step while their sum on the hidden state stays representable. The plain cast
    # (rescale=False) flushes them; the test must catch that against the float32-copy path.
    if not rescale:
        monkeypatch.setattr(cl, "half_logit_grad", _old_half_logit_grad)
    gen = torch.Generator().manual_seed(5)
    # Large weights and small hidden states (logits ~ N(0, 2.3)) keep the hidden-state gradient in float16's normal
    # range while upstream * p sits far below its subnormal step.
    weight = (torch.randn(30000, 32, generator = gen) * 40).to("cuda", torch.float16)
    hidden = (torch.randn(2, 9, 32, generator = gen) * 0.01).to("cuda", torch.float16)
    index = torch.randint(0, 30000, (2, 9), generator = gen).cuda()
    upstream = torch.full(index.shape, 1e-5, device = "cuda")
    grads = {}
    for name in ("fp32_out", "copy"):
        if name == "copy":
            monkeypatch.setattr(cl, "fp32_out_linear_ok", lambda w: False)
        for scorer, kw in (("plain", dict(fused_linear = False)), ("kept_set", dict(support_top_k = 29999))):
            h = hidden.clone().requires_grad_(True)
            fn = rr.grpo_kept_set_logprobs if scorer == "kept_set" else rr.chunked_hidden_states_selective_log_softmax
            out = fn(h, weight, index, 3, 0.0, 0.0, 0.0, 1.0, fp32_lm_head = True, **kw)
            (out * upstream).sum().backward()
            grads[(name, scorer)] = h.grad.float()
    for scorer in ("plain", "kept_set"):
        new, ref = grads[("fp32_out", scorer)], grads[("copy", scorer)]
        assert ref.norm() > 0
        err = ((new - ref).norm() / ref.norm()).item()
        if rescale:
            assert err < 2e-2, (scorer, err)
        else:
            assert err > 0.1, (scorer, err)
