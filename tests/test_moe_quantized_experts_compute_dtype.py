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

"""FP8 / bnb 4-bit experts train as W8A16 / W4A16: never dequantized to a float32 activation's dtype.

Qwen4Exp's PLE `sum` runs in float32 under autocast, which made its residual stream (and every later MoE
input) float32. torch._grouped_mm is not autocast-cast, so dequantizing to the activation dtype sent the
experts to its per-group float32 fallback (72% of the Qwen3.8-Flash-Next-FP8 step's device time).
"""

import pytest
import torch

from unsloth_zoo.temporary_patches import moe_utils, moe_utils_bnb4bit, moe_utils_fp8

E, K, FB, T = 4, 2, 128, 32
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")


def test_compute_dtype_is_half():
    x32 = torch.randn(4, 8)
    assert moe_utils.moe_compute_dtype(x32.bfloat16()) is torch.bfloat16
    assert moe_utils.moe_compute_dtype(x32.half()) is torch.float16
    assert moe_utils.moe_compute_dtype(x32) is torch.bfloat16
    assert moe_utils.moe_compute_dtype(torch.empty(4, 8, device = "meta")) is torch.bfloat16
    with torch.autocast("cpu", dtype = torch.bfloat16):
        assert moe_utils.moe_compute_dtype(x32) is torch.bfloat16
        # An activation already in half keeps its dtype: nothing moves for bf16 / fp16 models.
        assert moe_utils.moe_compute_dtype(x32.half()) is torch.float16


def test_fp8_dequant_target_is_never_float32():
    x32 = torch.randn(4, 8)
    assert moe_utils_fp8._get_fp8_dequant_target_dtype(x32) is torch.bfloat16
    assert moe_utils_fp8._get_fp8_dequant_target_dtype(x32.bfloat16()) is torch.bfloat16
    assert moe_utils_fp8._get_fp8_dequant_target_dtype(x32.half()) is torch.float16
    with torch.autocast("cpu", dtype = torch.bfloat16):
        assert moe_utils_fp8._get_fp8_dequant_target_dtype(x32) is torch.bfloat16


def _fp8_experts(device):
    finegrained_fp8 = pytest.importorskip("transformers.integrations.finegrained_fp8")
    if not hasattr(finegrained_fp8, "FP8Experts"):
        pytest.skip(reason = "FP8Experts only exists in transformers 5")
    from transformers import PretrainedConfig

    config = PretrainedConfig()
    config.hidden_size, config.moe_intermediate_size, config.num_experts = FB, FB, E
    config.intermediate_size = FB
    config.hidden_act = "silu"
    experts = finegrained_fp8.FP8Experts(config, block_size = (FB, FB)).to(device)
    fmax = torch.finfo(torch.float8_e4m3fn).max
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, shape, scale in (("gate_up_proj", (E, 2 * FB, FB), 0.3), ("down_proj", (E, FB, FB), 0.05)):
            w = torch.randn(*shape, generator = g) * scale
            blk = w.reshape(E, shape[1] // FB, FB, 1, FB)
            s = blk.abs().amax(dim = (2, 4)).clamp(min = 1e-12) / fmax
            q = (blk / s[:, :, None, :, None]).clamp(-fmax, fmax).to(torch.float8_e4m3fn).reshape(shape)
            getattr(experts, name).copy_(q.to(device))
            getattr(experts, name + "_scale_inv").copy_(s.to(device))
    g = torch.Generator().manual_seed(1)
    hidden = (torch.randn(T, FB, generator = g) * 2).to(device)
    top_k_index = torch.stack([torch.randperm(E, generator = g)[:K] for _ in range(T)]).to(device)
    top_k_weights = torch.rand(T, K, generator = g).to(device, torch.bfloat16)
    return experts, hidden, top_k_index, top_k_weights


def _run_fp8(device, monkeypatch, backend, autocast):
    monkeypatch.setenv("UNSLOTH_MOE_BACKEND", backend)
    # select_moe_backend is lru_cached: read it under this env, and do not leave this backend for later tests.
    moe_utils.select_moe_backend.cache_clear()
    try:
        _run_fp8_checks(device, monkeypatch, autocast)
    finally:
        moe_utils.select_moe_backend.cache_clear()


def _run_fp8_checks(device, monkeypatch, autocast):
    experts, hidden, top_k_index, top_k_weights = _fp8_experts(device)
    seen = []
    real = moe_utils_fp8._dequantize_full_expert_weights

    def spy(weight, quant_state, target_dtype, quant_kind = None):
        seen.append(target_dtype)
        return real(weight, quant_state, target_dtype, quant_kind = quant_kind)

    monkeypatch.setattr(moe_utils_fp8, "_dequantize_full_expert_weights", spy)
    with torch.no_grad():
        reference = moe_utils_fp8.forward_moe_backend_fp8(experts, hidden.bfloat16(), top_k_index, top_k_weights)
        assert set(seen) == {torch.bfloat16}
        seen.clear()
        with torch.autocast(device, dtype = torch.bfloat16, enabled = autocast):
            out = moe_utils_fp8.forward_moe_backend_fp8(experts, hidden, top_k_index, top_k_weights)
    # Experts dequantized to bf16 (not float32), output handed back in the caller's float32.
    assert set(seen) == {torch.bfloat16}, seen
    assert out.dtype == torch.float32
    # Same bf16 math as a bf16 activation: only the final cast differs.
    torch.testing.assert_close(out, reference.float(), rtol = 0, atol = 0)


@pytest.mark.parametrize("autocast", [True, False])
def test_fp8_experts_dequantize_to_bf16_for_float32_activations_cpu(monkeypatch, autocast):
    # The Triton dequant needs a GPU driver; the vectorized torch dequant is the CPU path.
    monkeypatch.setattr(moe_utils_fp8, "_dequantize_full_expert_weights_unsloth", lambda *a, **k: None)
    _run_fp8("cpu", monkeypatch, "native_torch", autocast)


@pytest.mark.gpu
@needs_cuda
@pytest.mark.parametrize("autocast", [True, False])
def test_fp8_experts_use_bf16_grouped_mm_for_float32_activations(monkeypatch, autocast):
    if not moe_utils._check_torch_grouped_mm_supported():
        pytest.skip(reason = "torch._grouped_mm needs sm >= 8.0 and torch >= 2.8")
    calls = []
    real = torch._grouped_mm

    def spy(a, b, *args, **kwargs):
        calls.append((a.dtype, b.dtype))
        return real(a, b, *args, **kwargs)

    monkeypatch.setattr(torch, "_grouped_mm", spy)
    _run_fp8("cuda", monkeypatch, "grouped_mm", autocast)
    assert calls and all(c == (torch.bfloat16, torch.bfloat16) for c in calls), calls


@pytest.mark.parametrize("autocast", [True, False])
def test_bnb4bit_dispatcher_runs_float32_activations_in_bf16(monkeypatch, autocast):
    seen = []

    def fake_grouped_mm(self, hidden_states, top_k_index, top_k_weights):
        seen.append(hidden_states.dtype)
        return hidden_states * 2

    monkeypatch.setattr(moe_utils, "select_moe_backend", lambda: "grouped_mm")
    monkeypatch.setattr(moe_utils, "_moe_recompute_enabled", lambda *a, **k: True)
    monkeypatch.setattr(moe_utils, "forward_native_grouped_mm", fake_grouped_mm)
    monkeypatch.setattr(moe_utils_bnb4bit, "_is_bnb4bit_param", lambda p: True)
    experts = torch.nn.Module()
    experts.gate_up_proj = experts.down_proj = None
    hidden = torch.randn(T, FB)
    with torch.autocast("cpu", dtype = torch.bfloat16, enabled = autocast):
        out = moe_utils_bnb4bit.forward_moe_backend_bnb4bit(experts, hidden, None, None)
    assert seen == [torch.bfloat16] and out.dtype == torch.float32
    seen.clear()
    out = moe_utils_bnb4bit.forward_moe_backend_bnb4bit(experts, hidden.bfloat16(), None, None)
    assert seen == [torch.bfloat16] and out.dtype == torch.bfloat16


qwen4_exp_modeling = None
try:
    import transformers.models.qwen4_exp.modeling_qwen4_exp as qwen4_exp_modeling
except Exception:
    pass
needs_qwen4_exp = pytest.mark.skipif(qwen4_exp_modeling is None, reason = "transformers has no qwen4_exp")


def _ple_layer(device):
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from unsloth_zoo.temporary_patches.qwen4_exp import patch_qwen4_exp

    patch_qwen4_exp()
    cfg = Qwen4ExpTextConfig(
        hidden_size = 64, num_hidden_layers = 4, num_attention_heads = 4, num_key_value_heads = 2, head_dim = 16,
        indexer_n_heads = 2, indexer_kv_heads = 1, indexer_head_dim = 16, indexer_budget = 16,
        indexer_compress_ratio = 4, linear_num_key_heads = 2,
        linear_num_value_heads = 2, linear_key_head_dim = 16, linear_value_head_dim = 16, num_experts = 4,
        num_experts_per_tok = 2, moe_intermediate_size = 16, shared_expert_intermediate_size = 16, vocab_size = 128,
        hc_count = 2, hc_lowrank = 8, ple_layer_ids = [2], ple_embed_dim = 64, ngram_vocab_size_base = 100,
        eos_token_id = 1,
    )
    torch.manual_seed(0)
    ple = qwen4_exp_modeling.Qwen4ExpTextPLELayer(cfg, layer_idx = 1, ple_layer_index = 0).to(device, torch.bfloat16)
    hidden = torch.randn(2, 8, cfg.hc_count * cfg.hidden_size, device = device, dtype = torch.bfloat16)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8), device = device)
    return ple, hidden, input_ids


@needs_qwen4_exp
def test_qwen4_exp_ple_returns_the_residual_dtype(monkeypatch):
    from unsloth_zoo.temporary_patches import qwen4_exp

    ple, hidden, input_ids = _ple_layer("cpu")
    assert type(ple).forward is qwen4_exp.qwen4_exp_ple_layer_forward
    reference = qwen4_exp._reference_ple_forward
    # What CUDA autocast does to the gate `sum`: the PLE output comes back float32.
    monkeypatch.setattr(qwen4_exp, "_reference_ple_forward", lambda *a, **k: reference(*a, **k).float())
    assert ple(hidden, input_ids, None).dtype == torch.bfloat16


@pytest.mark.gpu
@needs_cuda
@needs_qwen4_exp
def test_qwen4_exp_ple_keeps_bf16_under_cuda_autocast():
    ple, hidden, input_ids = _ple_layer("cuda")
    with torch.no_grad():
        plain = ple(hidden, input_ids, None)
        with torch.autocast("cuda", dtype = torch.bfloat16):
            out = ple(hidden, input_ids, None)
    assert plain.dtype == torch.bfloat16
    # Training (autocast) computes PLE exactly as inference does: bf16 throughout.
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, plain)


@pytest.mark.gpu
@needs_cuda
def test_half_stack_float32_activation_keeps_the_stack_dtype(monkeypatch):
    # Plain half stacks are not cast in the grouped_mm provider: a float32 activation must take the stack's
    # dtype, even when autocast asks for the other half dtype (fp16 weights under bf16 autocast).
    if not moe_utils._check_torch_grouped_mm_supported():
        pytest.skip(reason = "torch._grouped_mm needs sm >= 8.0 and torch >= 2.8")
    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

    config = Qwen3MoeConfig(hidden_size = FB, moe_intermediate_size = FB, num_experts = E, num_experts_per_tok = K)
    torch.manual_seed(0)
    experts = Qwen3MoeExperts(config).to("cuda", torch.float16)
    with torch.no_grad():
        for p in experts.parameters():
            p.normal_(0, 0.05)
    g = torch.Generator().manual_seed(1)
    hidden = torch.randn(T, FB, generator = g).cuda()
    top_k_index = torch.stack([torch.randperm(E, generator = g)[:K] for _ in range(T)]).cuda()
    top_k_weights = torch.rand(T, K, generator = g).cuda()
    calls = []
    real = torch._grouped_mm

    def spy(a, b, *args, **kwargs):
        calls.append((a.dtype, b.dtype))
        return real(a, b, *args, **kwargs)

    monkeypatch.setattr(torch, "_grouped_mm", spy)
    with torch.no_grad(), torch.autocast("cuda", dtype = torch.bfloat16):
        out = moe_utils.forward_native_grouped_mm(experts, hidden, top_k_index, top_k_weights)
    assert calls and all(c == (torch.float16, torch.float16) for c in calls), calls
    assert torch.isfinite(out).all()
