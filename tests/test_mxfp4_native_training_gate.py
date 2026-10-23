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

"""sm_12x + release/3.6.x trips ``assert num_stages >= 1`` in opt_flags, so training falls back to bf16
decodes there. CPU only."""

import types

import pytest
import torch

from unsloth_zoo import triton_kernels_compat as compat
from unsloth_zoo.temporary_patches import gpt_oss as go

E, H, I, T, K = 4, 64, 32, 16, 2
ALPHA, LIMIT = 1.702, 7.0


def _backend(api):
    return types.SimpleNamespace(api = api, name = "fake_triton_kernels")


@pytest.fixture(autouse = True)
def _fresh_capability_cache(monkeypatch):
    monkeypatch.setattr(compat, "_CAPABILITY", {})


@pytest.mark.parametrize("capability, api, native", [
    ((12, 0), compat.API_OGS_RAGGED, False),
    ((12, 1), compat.API_OGS_RAGGED, False),
    ((12, 0), compat.API_MATMUL_RAGGED, True),
    ((12, 0), compat.API_OGS_ROUTING, True),
    ((10, 0), compat.API_OGS_RAGGED, True),
    ((10, 0), compat.API_MATMUL_RAGGED, True),
    ((9, 0), compat.API_OGS_RAGGED, True),
    ((8, 0), compat.API_OGS_RAGGED, True),
    ((7, 5), compat.API_OGS_RAGGED, True),
])
def test_only_sm12x_with_the_matmul_ogs_ragged_generation_is_rejected(monkeypatch, capability, api, native):
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device = None: capability)
    assert compat.native_mxfp4_matmul_ok("cuda:0", _backend(api)) is native


def test_hip_cpu_and_no_backend_keep_the_current_route(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device = None: (12, 0))
    ragged = _backend(compat.API_OGS_RAGGED)
    assert compat.native_mxfp4_matmul_ok("cpu", ragged)
    assert compat.native_mxfp4_matmul_ok("cuda:0", None)
    monkeypatch.setattr(torch.version, "hip", "6.4")
    assert compat.native_mxfp4_matmul_ok("cuda:0", ragged)


def test_capability_is_read_once_per_device(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    reads = []
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device = None: reads.append(device) or (12, 0))
    ragged = _backend(compat.API_OGS_RAGGED)
    for _ in range(3):
        assert not compat.native_mxfp4_matmul_ok(torch.device("cuda", 0), ragged)
    assert not compat.native_mxfp4_matmul_ok("cuda:1", ragged)
    assert len(reads) == 2


class _Recorder:
    """Backend stand-in: counts native launches and answers them with the dequant math."""

    def __init__(self, api):
        self.api, self.name, self.calls = api, "fake_triton_kernels", 0

    def gate_up(self, x, weight, bias, routing_data, gather_idx, precision_config, fused_activation = None):
        self.calls += 1
        return go._ogs_dequant_gate_up(x, weight, bias, routing_data, gather_idx, precision_config)

    def down(self, h, weight, bias, routing_data, scatter_idx, precision_config, gammas = None):
        self.calls += 1
        return go._ogs_dequant_down(h, gammas, weight, bias, routing_data, scatter_idx, precision_config)


def _problem(seed = 0):
    gen = torch.Generator().manual_seed(seed)
    module = types.SimpleNamespace(alpha = ALPHA, limit = LIMIT, num_experts = E)
    # Dense (E, in, out) bf16 stacks: the still-packed transformers 5 path hands _OgsGateUp / _OgsDown these.
    module.gate_up_proj = (torch.randn(E, H, 2 * I, generator = gen) * 0.1).to(torch.bfloat16)
    module.down_proj = (torch.randn(E, I, H, generator = gen) * 0.1).to(torch.bfloat16)
    module.gate_up_proj_precision_config = module.down_proj_precision_config = None
    module.gate_up_proj_bias = torch.nn.Parameter(torch.randn(E, 2 * I, generator = gen) * 0.1)
    module.down_proj_bias = torch.nn.Parameter(torch.randn(E, H, generator = gen) * 0.1)
    x = torch.randn(T, H, generator = gen).to(torch.bfloat16)
    logits = torch.randn(T, E, generator = gen)
    return module, x, logits


def _routing(logits):
    values, experts = torch.topk(logits, K, dim = -1)
    weights = torch.softmax(values, -1)
    gate_scal, hist, combine, dispatch = go.topk_to_routing_tensors(experts, weights, E)
    routing = compat.RoutingData(gate_scal, hist, E, K)
    gather = compat.GatherIndx(src_indx = combine, dst_indx = dispatch)
    scatter = compat.ScatterIndx(src_indx = dispatch, dst_indx = combine)
    return routing, gather, scatter, experts, weights


def _swiglu(pre):
    gate, linear = pre[..., ::2].clamp(max = LIMIT), pre[..., 1::2].clamp(-LIMIT, LIMIT)
    return gate * torch.sigmoid(ALPHA * gate) * (linear + 1)


def _reference(module, x, experts, weights):
    gu, dn = module.gate_up_proj.float(), module.down_proj.float()
    out = 0
    for slot in range(K):
        e = experts[:, slot]
        pre = torch.bmm(x.unsqueeze(1), gu[e]).squeeze(1) + module.gate_up_proj_bias[e]
        y = torch.bmm(_swiglu(pre).unsqueeze(1), dn[e]).squeeze(1) + module.down_proj_bias[e]
        out = out + weights[:, slot : slot + 1] * y
    return out


def _train_step(monkeypatch, capability, api):
    monkeypatch.setattr(compat, "_device_capability", lambda device: capability)
    monkeypatch.setattr(go, "_OGS_NATIVE_REJECT_WARNED", False)
    backend = _Recorder(api)
    monkeypatch.setattr(go, "_ogs_backend_for", lambda *args, **kwargs: backend)
    module, x0, logits0 = _problem()
    grad = torch.randn(T, H, generator = torch.Generator().manual_seed(1))

    x, logits = x0.clone().requires_grad_(), logits0.clone().requires_grad_()
    routing, gather, scatter, _, _ = _routing(logits)
    out = go.mxfp4_ogs_experts_forward(module, x, routing, gather, scatter)
    (out.float() * grad).sum().backward()
    ours = {"out": out.detach().float(), "x": x.grad.float(), "logits": logits.grad,
            "gate_up_bias": module.gate_up_proj_bias.grad, "down_bias": module.down_proj_bias.grad}
    module.gate_up_proj_bias.grad = module.down_proj_bias.grad = None

    x_ref, logits_ref = x0.float().requires_grad_(), logits0.clone().requires_grad_()
    _, experts, weights = _routing(logits_ref)[2:]
    ref = _reference(module, x_ref, experts, weights)
    (ref * grad).sum().backward()
    theirs = {"out": ref.detach(), "x": x_ref.grad, "logits": logits_ref.grad,
              "gate_up_bias": module.gate_up_proj_bias.grad, "down_bias": module.down_proj_bias.grad}
    errors = {key: ((ours[key] - theirs[key]).norm() / theirs[key].norm()).item() for key in theirs}
    return backend.calls, errors


@pytest.mark.parametrize("capability, api, native", [
    ((12, 0), compat.API_OGS_RAGGED, False),  # RTX 50xx / PRO 6000 + Studio's release/3.6.x pin
    ((10, 0), compat.API_OGS_RAGGED, True),   # B200
    ((12, 0), compat.API_MATMUL_RAGGED, True),  # sm_120 + release/3.8.x
])
def test_training_step_route_and_parity(monkeypatch, capability, api, native):
    warnings = []
    monkeypatch.setattr(go.logger, "warning", lambda message, *args, **kwargs: warnings.append(message))
    calls, errors = _train_step(monkeypatch, capability, api)
    assert (calls == 2) is native and (calls == 0) is (not native), calls
    assert all(error < 2e-2 for error in errors.values()), errors
    assert len([w for w in warnings if "dequantized to bf16" in w]) == (0 if native else 1)


def test_the_fallback_warns_once(monkeypatch):
    warnings = []
    monkeypatch.setattr(go.logger, "warning", lambda message, *args, **kwargs: warnings.append(message))
    monkeypatch.setattr(compat, "_device_capability", lambda device: (12, 0))
    monkeypatch.setattr(go, "_OGS_NATIVE_REJECT_WARNED", False)
    ragged = _Recorder(compat.API_OGS_RAGGED)
    for _ in range(3):
        assert go._ogs_native_training_ok(torch.device("cuda", 0), ragged) is False
    assert len(warnings) == 1 and "sm_120" in warnings[0]


def test_the_fallback_returns_the_native_shapes_and_dtypes(monkeypatch):
    module, x, logits = _problem(seed = 4)
    routing, gather, scatter, _, _ = _routing(logits)
    whole = go._ogs_dequant_gate_up(x, module.gate_up_proj, module.gate_up_proj_bias, routing, gather, None)
    h = go._OgsSwiglu.apply(whole, ALPHA, LIMIT)
    down = go._ogs_dequant_down(h, routing.gate_scal, module.down_proj, module.down_proj_bias, routing, scatter, None)
    assert whole.dtype == torch.bfloat16 and whole.shape == (T * K, 2 * I)
    assert down.dtype == torch.bfloat16 and down.shape == (T, H)


def _cancelling_problem(device = "cpu", terms = 2, value = 1.0):
    """Every expert's product is 256 + 1 = 257 (not a bf16 value) and its bias -256: the exact answer is 1."""
    module, x, logits = _problem(seed = 8)
    x = torch.zeros(T, H, dtype = torch.bfloat16)
    x[:, 0], x[:, 1] = 256, 1
    module.gate_up_proj = torch.zeros(E, H, 2 * I, dtype = torch.bfloat16)
    module.gate_up_proj[:, :2] = 1
    module.down_proj = torch.zeros(E, I, H, dtype = torch.bfloat16)
    module.down_proj[:, :2] = 1
    module.gate_up_proj_bias = torch.full((E, 2 * I), -256.0)
    module.down_proj_bias = torch.full((E, H), -256.0)
    h = torch.zeros(T * K, I, dtype = torch.bfloat16)
    h[:, 0], h[:, 1] = 256, 1
    return module, x.to(device), h.to(device), logits


def test_the_fallback_adds_the_bias_before_rounding():
    # The native epilogue adds the fp32 bias to the fp32 accumulator; rounding 257 to bf16 (256) first gave 0.
    module, x, h, logits = _cancelling_problem()
    routing, gather, scatter, _, _ = _routing(logits)
    gate_up = go._ogs_dequant_gate_up(x, module.gate_up_proj, module.gate_up_proj_bias, routing, gather, None)
    assert torch.equal(gate_up.float(), torch.ones_like(gate_up, dtype = torch.float32))
    down = go._ogs_dequant_down(h, routing.gate_scal, module.down_proj, module.down_proj_bias, routing, scatter, None)
    torch.testing.assert_close(down.float(), torch.ones_like(down, dtype = torch.float32), rtol = 0, atol = 1e-2)


def test_the_fallback_routing_weight_grad_uses_the_same_epilogue(monkeypatch):
    module, x, h, logits = _cancelling_problem()
    routing, gather, scatter, _, _ = _routing(logits)
    module.down_proj_precision_config = None
    monkeypatch.setattr(go, "_ogs_backend_for", lambda *args, **kwargs: None)
    monkeypatch.setattr(go, "_ogs_native_training_ok", lambda *args, **kwargs: False)
    gammas = routing.gate_scal.detach().clone().requires_grad_(True)
    out = go._OgsDown.apply(h, gammas, module.down_proj_bias, module, routing, scatter)
    out.float().sum().backward()
    torch.testing.assert_close(gammas.grad.float(), torch.full_like(gammas.grad, H, dtype = torch.float32))


class _NoGradExperts:
    """The LoRA-patched experts forward's view of a module: dense stacks, no adapters, a recording native forward."""

    def __init__(self, module):
        self.__dict__.update(module.__dict__)
        self.original_calls = 0

    def _original_forward(self, *args, **kwargs):
        self.original_calls += 1
        return torch.zeros(T, H)


@pytest.mark.parametrize("capability, api, native", [
    ((12, 0), compat.API_OGS_RAGGED, False),
    ((12, 0), compat.API_MATMUL_RAGGED, True),
    ((10, 0), compat.API_OGS_RAGGED, True),
])
def test_no_grad_forward_without_lora_fails_closed_on_the_rejected_cell(monkeypatch, capability, api, native):
    import contextlib
    monkeypatch.setattr(compat, "_device_capability", lambda device: capability)
    monkeypatch.setattr(go, "_OGS_NATIVE_REJECT_WARNED", False)
    monkeypatch.setattr(go.logger, "warning", lambda *args, **kwargs: None)
    backend = _Recorder(api)
    monkeypatch.setattr(go, "_ogs_backend_for", lambda *args, **kwargs: backend)
    monkeypatch.setattr(go, "_ogs_module_backend", lambda module: backend)
    monkeypatch.setattr(go, "is_triton_kernels_available", lambda: True)
    monkeypatch.setattr(go, "_get_lora_wrapper_for_param", lambda module, name: None)
    monkeypatch.setattr(go, "torch_cuda_device", lambda device: contextlib.nullcontext())
    monkeypatch.setattr(
        go, "_mxfp4_experts_routing",
        lambda module, hs, rd, gather, scatter, ri, rw: (hs, rd, gather, scatter, None),
    )
    module, x, logits = _problem(seed = 6)
    experts = _NoGradExperts(module)
    routing, gather, scatter, top, weights = _routing(logits)
    with torch.no_grad():
        out = go.forward_mxfp4_gpt_oss_with_lora(experts, x, routing, gather, scatter)
    assert experts.original_calls == (1 if native else 0)
    if not native:
        assert backend.calls == 0
        ref = _reference(module, x.float(), top, weights)
        assert ((out.float() - ref).norm() / ref.norm()).item() < 2e-2


@pytest.mark.parametrize("grad", [False, True])
def test_lora_patched_forward_returns_the_fp16_input_dtype(monkeypatch, grad):
    # T4 fp16: a bf16 expert output promoted the residual to fp32 and broke the next fp16 Linear.
    import contextlib
    monkeypatch.setattr(compat, "_device_capability", lambda device: (12, 0))
    monkeypatch.setattr(go, "_OGS_NATIVE_REJECT_WARNED", False)
    monkeypatch.setattr(go.logger, "warning", lambda *args, **kwargs: None)
    backend = _Recorder(compat.API_OGS_RAGGED)
    monkeypatch.setattr(go, "_ogs_backend_for", lambda *args, **kwargs: backend)
    monkeypatch.setattr(go, "_ogs_module_backend", lambda module: backend)
    monkeypatch.setattr(go, "is_triton_kernels_available", lambda: True)
    monkeypatch.setattr(go, "_get_lora_wrapper_for_param", lambda module, name: None)
    monkeypatch.setattr(go, "torch_cuda_device", lambda device: contextlib.nullcontext())
    monkeypatch.setattr(
        go, "_mxfp4_experts_routing",
        lambda module, hs, rd, gather, scatter, ri, rw: (hs, rd, gather, scatter, None),
    )
    module, x, logits = _problem(seed = 7)
    experts = _NoGradExperts(module)
    routing, gather, scatter, top, weights = _routing(logits)
    x = x.to(torch.float16).requires_grad_(grad)
    with torch.set_grad_enabled(grad):
        out = go.forward_mxfp4_gpt_oss_with_lora(experts, x, routing, gather, scatter)
        residual = x + out
        assert out.dtype == torch.float16 and residual.dtype == torch.float16
        torch.nn.Linear(H, H, dtype = torch.float16)(residual)  # the next layer, no autocast
        if grad:
            out.float().sum().backward()
            assert x.grad is not None and x.grad.dtype == torch.float16
    assert experts.original_calls == 0
    ref = _reference(module, x.detach().float(), top, weights)
    assert ((out.detach().float() - ref).norm() / ref.norm()).item() < 2e-2


def test_matmul_ogs_available_rejects_the_cell_without_probing(monkeypatch):
    probes = []
    monkeypatch.setattr(compat, "_DEVICE_OK", {})
    monkeypatch.setattr(compat, "_self_check", lambda device: probes.append(device) or True)
    monkeypatch.setattr(compat, "get_triton_kernels_backend", lambda: _backend(compat.API_OGS_RAGGED))
    monkeypatch.setattr(compat, "_device_capability", lambda device: (12, 0))
    assert compat.matmul_ogs_available("cuda:0") is False and probes == []
    import contextlib
    monkeypatch.setattr(compat, "_DEVICE_OK", {})
    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())
    monkeypatch.setattr(compat, "_device_capability", lambda device: (10, 0))
    assert compat.matmul_ogs_available("cuda:0") is True and len(probes) == 1


def _overflowing_problem():
    # x @ W_gate_up = 64 * 2048 = 131072 exactly (bf16 and MXFP4 exact), past fp16's 65504; the bias
    # cancels it back to small pre-activations, so the right answer is finite and fp16-representable.
    module, _, logits = _problem(seed = 7)
    gen = torch.Generator().manual_seed(8)
    module.gate_up_proj = torch.ones(E, H, 2 * I, dtype = torch.bfloat16)
    module.gate_up_proj_bias = torch.nn.Parameter(torch.randn(E, 2 * I, generator = gen) * 2 - 64 * 2048)
    x = torch.full((T, H), 2048.0, dtype = torch.bfloat16)
    return module, x, logits


def test_fallback_ignores_fp16_autocast_and_matches_the_fp32_reference(monkeypatch):
    # The fallback's explicit bf16 matmuls must not be recast to fp16 by the caller's autocast (T4 / fp16 runs).
    monkeypatch.setattr(compat, "_device_capability", lambda device: (12, 0))
    monkeypatch.setattr(go, "_OGS_NATIVE_REJECT_WARNED", False)
    monkeypatch.setattr(go.logger, "warning", lambda *args, **kwargs: None)
    backend = _Recorder(compat.API_OGS_RAGGED)
    monkeypatch.setattr(go, "_ogs_backend_for", lambda *args, **kwargs: backend)
    module, x0, logits0 = _overflowing_problem()
    grad = torch.randn(T, H, generator = torch.Generator().manual_seed(9))

    x, logits = x0.clone().requires_grad_(), logits0.clone().requires_grad_()
    routing, gather, scatter, _, _ = _routing(logits)
    with torch.autocast(device_type = "cpu", dtype = torch.float16):
        out = go.mxfp4_ogs_experts_forward(module, x, routing, gather, scatter)
    (out.float() * grad).sum().backward()
    assert backend.calls == 0
    ours = {"out": out.detach().float(), "x": x.grad.float(), "logits": logits.grad,
            "gate_up_bias": module.gate_up_proj_bias.grad, "down_bias": module.down_proj_bias.grad}
    module.gate_up_proj_bias.grad = module.down_proj_bias.grad = None

    x_ref, logits_ref = x0.float().requires_grad_(), logits0.clone().requires_grad_()
    _, experts, weights = _routing(logits_ref)[2:]
    ref = _reference(module, x_ref, experts, weights)
    (ref * grad).sum().backward()
    theirs = {"out": ref.detach(), "x": x_ref.grad, "logits": logits_ref.grad,
              "gate_up_bias": module.gate_up_proj_bias.grad, "down_bias": module.down_proj_bias.grad}
    assert ref.abs().max() < 65504
    for key in theirs:
        assert torch.isfinite(ours[key]).all(), key
        error = ((ours[key] - theirs[key]).norm() / theirs[key].norm()).item()
        assert error < 2e-2, (key, error)


def test_fallback_down_does_not_overflow_before_the_routing_weight():
    # h @ W_down = 32 * 32 * 64 = 65536 overflows fp16; the 0.25 routing weight brings it back to 16384.
    module, _, _ = _problem()
    routing = compat.RoutingData(torch.full((1,), 0.25), torch.tensor([1, 0, 0, 0]), E, 1)
    scatter = compat.ScatterIndx(src_indx = torch.tensor([0]), dst_indx = torch.tensor([0]))
    h = torch.full((1, I), 32.0, dtype = torch.bfloat16)
    weight = torch.full((E, I, H), 64.0, dtype = torch.bfloat16)
    eager = go._ogs_dequant_down(h, routing.gate_scal, weight, None, routing, scatter, None)
    with torch.autocast(device_type = "cpu", dtype = torch.float16):
        amp = go._ogs_dequant_down(h, routing.gate_scal, weight, None, routing, scatter, None)
    assert eager.dtype == amp.dtype == torch.bfloat16
    assert torch.equal(eager, torch.full((1, H), 16384.0, dtype = torch.bfloat16))
    assert torch.equal(amp, eager)
