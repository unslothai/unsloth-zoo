# SPDX-License-Identifier: AGPL-3.0-only
"""bnb 4-bit experts of a family Unsloth has no model-specific MoE patch for.

replace_expert_params_with_bnb_params quantizes the gate_up_proj / down_proj of every
transformers v5 experts module, but only the patched families (qwen3_moe, glm4_moe,
lfm2_moe, ...) had their forward routed through forward_moe_backend, which dequantizes.
Any other family, tencent/Hy3's HYV3Experts among them, kept transformers' generic
experts forward and matmul'd the packed uint8 storage:
"Expected mat_a to be Float32, BFloat16 or Float16 matrix, got Byte".

The class is now routed when its experts are prepared for 4-bit, only when its layout is
the one forward_moe_backend computes, and 16-bit instances keep the original forward.
"""
import copy
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

moe_integration = pytest.importorskip("transformers.integrations.moe")
use_experts_implementation = getattr(moe_integration, "use_experts_implementation", None)
if use_experts_implementation is None:
    pytest.skip("transformers has no use_experts_implementation", allow_module_level = True)

from unsloth_zoo.temporary_patches import moe_utils_bnb4bit as mb


E, H, I = 4, 64, 32


def _make_experts_class(**decorator_kwargs):
    # A fresh class per test: routing is recorded on the class.
    class ToyExperts(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.num_experts = config.num_experts
            self.hidden_dim = config.hidden_size
            self.intermediate_dim = config.moe_intermediate_size
            self.gate_up_proj = nn.Parameter(torch.empty(E, 2 * I, H))
            self.down_proj = nn.Parameter(torch.empty(E, H, I))
            self.act_fn = nn.SiLU()

        def forward(self, hidden_states, top_k_index, top_k_weights):
            final = torch.zeros_like(hidden_states)
            for e in range(self.num_experts):
                token_idx, k = torch.where(top_k_index == e)
                if token_idx.numel() == 0:
                    continue
                gate, up = nn.functional.linear(hidden_states[token_idx], self.gate_up_proj[e]).chunk(2, dim = -1)
                out = nn.functional.linear(self.act_fn(gate) * up, self.down_proj[e])
                final.index_add_(0, token_idx, (out * top_k_weights[token_idx, k, None]).to(final.dtype))
            return final

    if decorator_kwargs:
        return use_experts_implementation(ToyExperts, **decorator_kwargs)
    return use_experts_implementation(ToyExperts)


def _config(impl = "eager"):
    return SimpleNamespace(num_experts = E, hidden_size = H, moe_intermediate_size = I, _experts_implementation = impl)


def _init(module, seed = 0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        module.gate_up_proj.copy_(torch.randn(E, 2 * I, H, generator = g) * 0.05)
        module.down_proj.copy_(torch.randn(E, H, I, generator = g) * 0.05)
    return module


def _routing(n_tokens, seed = 1, device = "cpu"):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(n_tokens, H, generator = g)
    top_k_index = torch.stack([torch.randperm(E, generator = g)[:2] for _ in range(n_tokens)])
    top_k_weights = torch.rand(n_tokens, 2, generator = g)
    return hidden.to(device), top_k_index.to(device), top_k_weights.to(device)


# ----------------------------------------------------------------------------- classifier


def test_generic_forward_is_recognised_and_others_are_not():
    klass = _make_experts_class()
    assert mb._is_generic_transformers_experts_forward(klass.__dict__["forward"])
    assert not mb._is_generic_transformers_experts_forward(_routing)
    assert not mb._is_generic_transformers_experts_forward(None)


def test_only_the_standard_layout_is_routable():
    assert mb._experts_layout_is_standard(_make_experts_class()(_config()))
    for kwargs in ({"has_bias": True}, {"is_transposed": True}):
        try:
            klass = _make_experts_class(**kwargs)
        except TypeError:
            continue  # this transformers has no such flag
        assert not mb._experts_layout_is_standard(klass(_config())), kwargs

    custom_gate = _make_experts_class()
    custom_gate._apply_gate = lambda self, x: x[..., : x.shape[-1] // 2]
    assert not mb._experts_layout_is_standard(custom_gate(_config()))

    no_act = _make_experts_class()(_config())
    no_act.act_fn = None
    assert not mb._experts_layout_is_standard(no_act)

    wrong_shape = _make_experts_class()(_config())
    wrong_shape.down_proj = nn.Parameter(torch.empty(E, H, I + 1))
    assert not mb._experts_layout_is_standard(wrong_shape)


def test_routing_leaves_16bit_instances_on_the_original_forward():
    klass = _make_experts_class()
    reference = _init(klass(_config()))
    routed = copy.deepcopy(reference)
    args = _routing(16)
    expected = reference(*args)
    assert mb._route_generic_bnb4bit_experts_class(routed)
    assert klass._unsloth_bnb4bit_routed
    torch.testing.assert_close(routed(*args), expected, rtol = 0, atol = 0)
    # Idempotent: a second module of the same class does not wrap the wrapper.
    forward = klass.forward
    assert mb._route_generic_bnb4bit_experts_class(klass(_config()))
    assert klass.forward is forward


def test_routed_forward_is_seen_to_apply_the_expert_lora_stash():
    # The ParamWrapper patch answers this statically on a compiled first call; "not read"
    # there makes the forward and the checkpoint recompute trace different graphs.
    from unsloth_zoo.temporary_patches import moe_utils as mu

    klass = _make_experts_class()
    module = klass(_config())
    assert not mu._forward_statically_reads_stash(module)
    assert mb._route_generic_bnb4bit_experts_class(module)
    assert mu._forward_statically_reads_stash(module) is True


def test_the_backend_call_is_kept_out_of_compiled_regions():
    fn = mb._forward_generic_experts_eagerly
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "disable"):
        assert getattr(fn, "__wrapped__", None) is not None  # torch.compiler.disable wrapper


def test_a_class_with_its_own_forward_is_not_touched():
    class Patched(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.empty(E, 2 * I, H))
            self.down_proj = nn.Parameter(torch.empty(E, H, I))
            self.act_fn = nn.SiLU()

        def forward(self, hidden_states, top_k_index, top_k_weights):
            return hidden_states

    forward = Patched.forward
    assert not mb._route_generic_bnb4bit_experts_class(Patched())
    assert Patched.forward is forward
    assert not getattr(Patched, "_unsloth_bnb4bit_routed", False)


# ----------------------------------------------------------------------------- 4-bit forward


def _quantize_like_the_loader(model, device):
    """What from_pretrained does on a 4-bit load: meta Params4bit placeholders from
    replace_expert_params_with_bnb_params, then the checkpoint weight quantized into them."""
    from bitsandbytes.nn import Params4bit
    from transformers import BitsAndBytesConfig

    experts = model.experts
    dense = {name: getattr(experts, name).detach().clone() for name in ("gate_up_proj", "down_proj")}
    quantization_config = BitsAndBytesConfig(
        load_in_4bit = True, bnb_4bit_quant_type = "nf4", bnb_4bit_compute_dtype = torch.bfloat16
    )
    mb.replace_expert_params_with_bnb_params(model, quantization_config = quantization_config)
    for name, value in dense.items():
        param = Params4bit(value.to(torch.bfloat16), requires_grad = False, quant_type = "nf4").to(device)
        param._original_shape = value.shape
        setattr(experts, name, param)
    return model


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "bnb 4-bit needs CUDA")
@pytest.mark.parametrize("impl", ["eager", "grouped_mm"])
def test_bnb4bit_generic_experts_forward_matches_dequantized_reference(impl):
    pytest.importorskip("bitsandbytes")
    if impl not in ("eager",) and impl not in getattr(moe_integration, "ALL_EXPERTS_FUNCTIONS", {}):
        pytest.skip(f"{impl} experts implementation unavailable")
    klass = _make_experts_class()
    model = nn.Module()
    model.experts = _init(klass(_config(impl)))
    _quantize_like_the_loader(model, "cuda")
    experts = model.experts
    assert mb._moe_uses_bnb4bit_expert_weights(experts)

    # Reference: the original forward on the dequantized weights.
    reference = klass(_config("eager")).to("cuda", torch.bfloat16)
    with torch.no_grad():
        reference.gate_up_proj.copy_(mb._dequantize_bnb4bit_expert_weights(experts.gate_up_proj, torch.bfloat16))
        reference.down_proj.copy_(mb._dequantize_bnb4bit_expert_weights(experts.down_proj, torch.bfloat16))
    hidden, top_k_index, top_k_weights = _routing(64, device = "cuda")
    hidden = hidden.to(torch.bfloat16)
    expected = reference(hidden, top_k_index, top_k_weights.to(torch.bfloat16))

    out = experts(hidden, top_k_index, top_k_weights.to(torch.bfloat16))
    assert out.dtype == torch.bfloat16 and out.shape == hidden.shape
    torch.testing.assert_close(out.float(), expected.float(), rtol = 2e-2, atol = 2e-3)

    hidden = hidden.clone().requires_grad_(True)
    experts(hidden, top_k_index, top_k_weights.to(torch.bfloat16)).float().pow(2).sum().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()


def test_expert_parallel_sentinel_slots_are_dropped():
    """RouterParallel marks non-local routes with index num_experts and weight 0."""
    module = SimpleNamespace(num_experts = 4)
    index = torch.tensor([[0, 4], [3, 4], [4, 4]])
    weights = torch.tensor([[0.5, 0.0], [1.0, 0.0], [0.0, 0.0]])
    new_index, new_weights = mb._drop_expert_parallel_sentinel(module, index, weights)
    assert int(new_index.max()) < 4
    assert torch.equal(new_weights, torch.tensor([[0.5, 0.0], [1.0, 0.0], [0.0, 0.0]]))
    assert torch.equal(new_index, torch.tensor([[0, 0], [3, 0], [0, 0]]))
    # Without sentinels nothing changes.
    same_index, same_weights = mb._drop_expert_parallel_sentinel(module, index.clamp(max = 3), weights)
    assert torch.equal(same_index, index.clamp(max = 3)) and torch.equal(same_weights, weights)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "bnb 4-bit needs CUDA")
def test_bnb4bit_forward_accepts_expert_parallel_sentinel_routes():
    pytest.importorskip("bitsandbytes")
    klass = _make_experts_class()
    model = nn.Module()
    model.experts = _init(klass(_config("grouped_mm")))
    _quantize_like_the_loader(model, "cuda")
    experts = model.experts
    num_experts = experts.gate_up_proj._original_shape[0] if hasattr(experts.gate_up_proj, "_original_shape") else experts.gate_up_proj.shape[0]
    hidden, top_k_index, top_k_weights = _routing(64, device = "cuda")
    hidden = hidden.to(torch.bfloat16)
    top_k_weights = top_k_weights.to(torch.bfloat16)
    local = experts(hidden, top_k_index, top_k_weights)
    # Append a non-local slot per token: sentinel index, zero weight.
    sentinel_index = torch.cat([top_k_index, torch.full_like(top_k_index[:, :1], num_experts)], dim = 1)
    sentinel_weights = torch.cat([top_k_weights, torch.zeros_like(top_k_weights[:, :1])], dim = 1)
    out = experts(hidden, sentinel_index, sentinel_weights)
    torch.testing.assert_close(out.float(), local.float(), rtol = 2e-2, atol = 2e-3)
