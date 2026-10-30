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

"""TRL 1.17 appends support_ids / score_ids to ChunkedLogProbFunction and unpacks six outputs. The CPU
dispatch must accept the longer call, pad the torch path to six outputs, and leave calls that use either
id set on TRL's kernel; the five-output TRL 1.x contract must keep working."""

import pytest
import torch

pytest.importorskip("trl")


def _install(monkeypatch, kernel):
    from trl.trainer import utils as trl_utils
    from unsloth_zoo.cpu import ops
    monkeypatch.setattr(trl_utils, "_ChunkedLogProbFunction", kernel, raising = False)
    assert ops.patch_trl_fused_lm_head() is True
    return trl_utils._ChunkedLogProbFunction


def _inputs():
    torch.manual_seed(0)
    hidden = torch.randn(6, 16, requires_grad = True)
    weight = torch.randn(50, 16, requires_grad = True)
    return hidden, weight, torch.randint(0, 50, (6,))


class _Kernel117(torch.autograd.Function):
    """Signature of TRL 1.17's kernel; records the calls that reach it."""
    calls = []

    @staticmethod
    def forward(ctx, hidden, weight, bias, targets, temperature, chunk_size, final_logit_softcapping = None,
                logit_scale = 1.0, outputs = (), support_ids = None, score_ids = None):
        _Kernel117.calls.append(score_ids is not None or support_ids is not None)
        z = hidden.new_zeros(hidden.shape[0])
        return z, None, None, None, None, z

    @staticmethod
    def backward(ctx, *grads):
        return (None,) * 11


class _Kernel115(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, bias, targets, temperature, chunk_size, final_logit_softcapping = None,
                logit_scale = 1.0, outputs = ()):
        raise AssertionError("the CPU call must not reach the Triton kernel")

    @staticmethod
    def backward(ctx, *grads):
        return (None,) * 9


def test_trl_117_call_on_cpu_takes_the_torch_path_with_six_outputs(monkeypatch):
    _Kernel117.calls.clear()
    dispatch = _install(monkeypatch, _Kernel117)
    hidden, weight, targets = _inputs()
    from unsloth_zoo.cpu.ops import cpu_chunked_log_probs
    out = dispatch.apply(hidden, weight, None, targets, 1.0, 4096, None, 1.0, ("log_probs", "entropy"), None, None)
    assert len(out) == 6 and out[5] is None and _Kernel117.calls == []
    ref = cpu_chunked_log_probs(hidden, weight, None, targets, 1.0, 4096, None, 1.0, ("log_probs", "entropy"))
    torch.testing.assert_close(out[0], ref[0])
    torch.testing.assert_close(out[1], ref[1])
    out[0].sum().backward()
    assert weight.grad is not None and torch.count_nonzero(weight.grad) > 0


def test_trl_117_kept_set_call_stays_on_trl_kernel(monkeypatch):
    _Kernel117.calls.clear()
    dispatch = _install(monkeypatch, _Kernel117)
    hidden, weight, targets = _inputs()
    score_ids = torch.full((6, 2), -1, dtype = torch.long)
    out = dispatch.apply(hidden, weight, None, targets, 1.0, 4096, None, 1.0, ("log_probs",), None, score_ids)
    assert len(out) == 6 and _Kernel117.calls == [True]
    dispatch.apply(hidden, weight, None, targets, 1.0, 4096, None, 1.0, ("log_probs",), score_ids, None)
    assert _Kernel117.calls == [True, True]


def test_trl_1x_five_output_contract_unchanged(monkeypatch):
    dispatch = _install(monkeypatch, _Kernel115)
    hidden, weight, targets = _inputs()
    out = dispatch.apply(hidden, weight, None, targets, 1.0, 4096, None, 1.0, ("log_probs",))
    assert len(out) == 5 and out[0].shape == (6,)


def test_unknown_future_contract_falls_back_to_trl_kernel(monkeypatch):
    class _KernelNext(torch.autograd.Function):
        seen = []

        @staticmethod
        def forward(ctx, hidden, weight, bias, targets, temperature, chunk_size, final_logit_softcapping = None,
                    logit_scale = 1.0, outputs = (), support_ids = None, score_ids = None, new_arg = None):
            _KernelNext.seen.append(new_arg)
            return (hidden.new_zeros(hidden.shape[0]),) * 7

        @staticmethod
        def backward(ctx, *grads):
            return (None,) * 12

    dispatch = _install(monkeypatch, _KernelNext)
    hidden, weight, targets = _inputs()
    out = dispatch.apply(hidden, weight, None, targets, 1.0, 4096, None, 1.0, (), None, None, None)
    assert len(out) == 7 and _KernelNext.seen == [None]


def test_installed_trl_fused_head_runs_on_cpu(monkeypatch):
    """End to end on the installed TRL: the patched fused head gives TRL's own CPU loss where TRL has one."""
    trl_utils = pytest.importorskip("trl.trainer.utils")
    if not hasattr(trl_utils, "add_fused_lm_head"):
        pytest.skip("TRL without the fused LM head")
    from transformers import LlamaConfig, LlamaForCausalLM
    from unsloth_zoo.cpu import ops
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size = 64, hidden_size = 16, intermediate_size = 32, num_hidden_layers = 1,
                         num_attention_heads = 2, num_key_value_heads = 1)
    model = LlamaForCausalLM(config)
    trl_utils.add_fused_lm_head(model, outputs = ("log_probs", "entropy"))
    ids = torch.randint(0, 64, (2, 8))
    labels = ids.clone()
    labels[0, :3] = -100
    try:
        reference = float(model(input_ids = ids, labels = labels, fused_lm_head = True).loss)
    except Exception:
        reference = None  # TRL before 1.17 has no CPU path of its own.
    monkeypatch.setattr(trl_utils, "_ChunkedLogProbFunction", trl_utils._ChunkedLogProbFunction)
    assert ops.patch_trl_fused_lm_head() is True
    out = model(input_ids = ids, labels = labels, fused_lm_head = True)
    out.loss.backward()
    assert torch.isfinite(out.loss)
    if reference is not None:
        assert abs(float(out.loss) - reference) < 1e-5
