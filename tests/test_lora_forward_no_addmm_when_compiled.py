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

# unsloth#3271: traced inside a compiled decoder MLP, torch 2.8 / 2.9 inductor miscompiled the
# addmm(alpha = scaling, beta = 1) LoRA epilogue (Qwen2.5-VL text-only LoRA diverged).
import pytest
import torch

from unsloth_zoo import compiler

TEMPLATES = {
    "default": compiler.COMPILED_LORA_FORWARD,
    "forced_float32": compiler.COMPILED_LORA_FORWARD_forced_float32,
}


def _load(source, no_compiled_addmm = None):
    namespace = {"torch": torch}
    exec(source, namespace)
    if no_compiled_addmm is not None:
        namespace["torch_lora_no_compiled_addmm"] = no_compiled_addmm
    return namespace["lora_forward"]


def _adapter(bias):
    torch.manual_seed(3407)
    lora_A = torch.nn.Linear(64, 8, bias = False)
    lora_B = torch.nn.Linear(8, 32, bias = bias)
    with torch.no_grad():
        lora_B.weight.normal_(0, 0.05)
        if bias:
            lora_B.bias.normal_(0, 0.05)
    return lora_A, lora_B


def _traced_ops(fn, *args):
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    torch._dynamo.reset()
    out = torch.compile(fn, backend = backend, dynamic = True)(*args)
    targets = {node.target for gm in graphs for node in gm.graph.nodes if node.op == "call_function"}
    return out, targets


def _half_matmul_supported():
    try:
        torch.ones(2, 2, dtype = torch.float16) @ torch.ones(2, 2, dtype = torch.float16)
        return True
    except RuntimeError:
        return False


def _args(bias):
    lora_A, lora_B = _adapter(bias)
    return torch.randn(2, 5, 32), lora_A, lora_B, torch.nn.Identity(), torch.randn(2, 5, 64), 16.0


@pytest.mark.parametrize("template", list(TEMPLATES))
def test_workaround_only_on_affected_torch(template):
    namespace = {"torch": torch}
    exec(TEMPLATES[template], namespace)
    major_minor = tuple(int(v) for v in torch.__version__.split(".")[:2])
    assert namespace["torch_lora_no_compiled_addmm"] == (major_minor < (2, 10))


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("template", list(TEMPLATES))
def test_affected_torch_traces_no_addmm_and_matches_eager(template, bias):
    if template == "forced_float32" and not _half_matmul_supported():
        pytest.skip("float16 matmul unsupported on this CPU build")
    lora_forward = _load(TEMPLATES[template], no_compiled_addmm = True)
    args = _args(bias)
    eager = lora_forward(*args)
    traced, targets = _traced_ops(lora_forward, *args)
    assert torch.addmm not in targets
    assert traced.shape == eager.shape and traced.dtype == eager.dtype
    torch.testing.assert_close(traced, eager, rtol = 2e-3, atol = 2e-3)


@pytest.mark.parametrize("template", list(TEMPLATES))
def test_fixed_torch_keeps_addmm_when_traced(template):
    if template == "forced_float32" and not _half_matmul_supported():
        pytest.skip("float16 matmul unsupported on this CPU build")
    lora_forward = _load(TEMPLATES[template], no_compiled_addmm = False)
    _, targets = _traced_ops(lora_forward, *_args(False))
    assert torch.addmm in targets


def test_eager_lora_forward_keeps_fused_addmm():
    lora_forward = _load(compiler.COMPILED_LORA_FORWARD, no_compiled_addmm = True)
    calls = []
    real_addmm = torch.addmm
    lora_forward.__globals__["torch_addmm"] = lambda *a, **k: calls.append(k) or real_addmm(*a, **k)
    lora_A, lora_B = _adapter(False)
    lora_forward(torch.randn(3, 32), lora_A, lora_B, torch.nn.Identity(), torch.randn(3, 64), 2.0)
    assert calls and calls[0]["alpha"] == 2.0 and calls[0]["beta"] == 1
