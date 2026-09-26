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

import pytest
import torch

pytest.importorskip("transformers")
try:
    from transformers.models.llama4.configuration_llama4 import Llama4TextConfig
    from transformers.models.llama4.modeling_llama4 import Llama4TextMoe, Llama4TextExperts
except Exception as e:  # pragma: no cover
    pytest.skip(f"Llama-4 not in this transformers: {e}", allow_module_level = True)

from unsloth_zoo.temporary_patches.llama4_moe import patch_llama4_moe, Llama4TextMoe_forward
from unsloth_zoo.temporary_patches.moe_experts_interface import expert_forward_is_handled

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _fp32_tolerance(default):
    # Triton grouped GEMM accumulates at ~1e-3 in fp32; grouped_mm and the native loop match to 1e-6.
    import sys
    module = sys.modules.get(getattr(Llama4TextExperts.forward, "__module__", ""), None)
    select = getattr(module, "select_moe_backend", None)
    if select is None:
        from unsloth_zoo.temporary_patches.moe_utils import select_moe_backend as select
    return 3e-3 if select() == "unsloth_triton" else default


@pytest.fixture(autouse = True, scope = "module")
def _restore_llama4_classes():
    saved = {cls: dict(vars(cls)) for cls in (Llama4TextExperts, Llama4TextMoe)}
    yield
    for cls, attrs in saved.items():
        for name in list(vars(cls)):
            if name not in attrs:
                delattr(cls, name)
        for name, value in attrs.items():
            if name not in ("__dict__", "__weakref__"):
                setattr(cls, name, value)


def reference_forward(moe, hidden_states):
    hidden_states = hidden_states.reshape(-1, moe.hidden_dim)
    router_logits = torch.nn.functional.linear(hidden_states, moe.router.weight)
    top_value, top_index = torch.topk(router_logits, moe.top_k, dim = 1)
    router_scores = torch.full_like(router_logits, float("-inf")).scatter_(1, top_index, top_value)
    router_scores = torch.sigmoid(router_scores.float()).to(router_scores.dtype)
    routed_in = hidden_states.repeat(router_scores.shape[1], 1)
    routed_in = routed_in * router_scores.transpose(0, 1).reshape(-1, 1)
    experts = moe.experts
    x = routed_in.view(experts.gate_up_proj.shape[0], -1, experts.hidden_size)
    gate_up = torch.bmm(x, experts.gate_up_proj)
    gate, up = gate_up.chunk(2, dim = -1)
    routed_out = torch.bmm(up * experts.act_fn(gate), experts.down_proj).view(-1, experts.hidden_size)
    out = moe.shared_expert(hidden_states)
    out = out + routed_out.reshape(router_scores.shape[1], -1, routed_out.shape[-1]).sum(dim = 0)
    return out, router_logits


# Multiples of the Triton tiles (K % BLOCK_SIZE_K == 0); 2 * 128 == 256 makes gate_up square on purpose.
HIDDEN = 256


def make_moe(top_k, dtype, seed = 0):
    torch.manual_seed(seed)
    config = Llama4TextConfig(
        hidden_size = HIDDEN, intermediate_size = 128, intermediate_size_mlp = 256,
        num_local_experts = 4, num_experts_per_tok = top_k, num_hidden_layers = 1,
        num_attention_heads = 2, num_key_value_heads = 1, head_dim = 32, vocab_size = 256,
    )
    moe = Llama4TextMoe(config)
    with torch.no_grad():
        for p in moe.parameters():
            p.normal_(0, 0.05)
    return moe.to(DEVICE, dtype)


@pytest.mark.parametrize("top_k", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_patched_forward_matches_reference(top_k, dtype):
    patch_llama4_moe()
    assert Llama4TextMoe.forward is Llama4TextMoe_forward
    assert Llama4TextExperts.is_transposed is True
    moe = make_moe(top_k, dtype)
    assert expert_forward_is_handled(moe.experts)
    x = torch.randn(3, 5, HIDDEN, device = DEVICE, dtype = dtype)

    ref_out, ref_logits = reference_forward(moe, x.clone())
    out, logits = moe(x)
    assert out.shape == ref_out.shape == (15, HIDDEN)
    assert torch.equal(logits, ref_logits)
    tol = _fp32_tolerance(1e-5) if dtype is torch.float32 else 2e-2
    assert torch.allclose(out.float(), ref_out.float(), atol = tol, rtol = tol), \
        (out.float() - ref_out.float()).abs().max()


def test_backward_matches_reference_fp32():
    patch_llama4_moe()
    moe = make_moe(1, torch.float32)
    x = torch.randn(2, 6, HIDDEN, device = DEVICE)

    ref_out, _ = reference_forward(moe, x)
    ref_out.square().sum().backward()
    ref = {n: p.grad.clone() for n, p in moe.named_parameters() if p.grad is not None}
    moe.zero_grad(set_to_none = True)

    out, _ = moe(x)
    out.square().sum().backward()
    got = {n: p.grad.clone() for n, p in moe.named_parameters() if p.grad is not None}

    for name in ("experts.gate_up_proj", "experts.down_proj", "router.weight", "shared_expert.gate_proj.weight"):
        assert name in ref and name in got, name
        if _fp32_tolerance(None) is None:
            assert torch.allclose(got[name], ref[name], atol = 1e-4, rtol = 1e-4), \
                (name, (got[name] - ref[name]).abs().max())
            continue
        assert (got[name] - ref[name]).abs().max() <= 1e-2 * ref[name].abs().max(), \
            (name, (got[name] - ref[name]).abs().max())


def test_patch_is_idempotent():
    patch_llama4_moe()
    before = Llama4TextExperts.forward
    patch_llama4_moe()
    assert Llama4TextExperts.forward is before


def test_a_failed_moe_patch_leaves_both_forwards_alone(monkeypatch):
    import unsloth_zoo.temporary_patches.llama4_moe as llama4_moe

    def experts_forward(self, hidden_states):
        return hidden_states

    def moe_forward(self, hidden_states):
        return hidden_states

    monkeypatch.setattr(Llama4TextExperts, "forward", experts_forward)
    monkeypatch.setattr(Llama4TextMoe, "forward", moe_forward)
    monkeypatch.setattr(Llama4TextExperts, "_unsloth_already_patched", False)
    real_patch_function = llama4_moe.patch_function

    def refuse_moe(target, name, *args, **kwargs):
        if target is Llama4TextMoe:
            return False
        return real_patch_function(target, name, *args, **kwargs)

    monkeypatch.setattr(llama4_moe, "patch_function", refuse_moe)
    llama4_moe.patch_llama4_moe()
    assert Llama4TextExperts.forward is experts_forward
    assert Llama4TextMoe.forward is moe_forward
    assert not Llama4TextExperts._unsloth_already_patched


def test_no_patch_before_the_tuple_returning_router(monkeypatch):
    import transformers.models.llama4.modeling_llama4 as modeling_llama4

    def experts_forward(self, hidden_states):
        return hidden_states

    def moe_forward(self, hidden_states):
        return hidden_states

    monkeypatch.setattr(Llama4TextExperts, "forward", experts_forward)
    monkeypatch.setattr(Llama4TextMoe, "forward", moe_forward)
    monkeypatch.setattr(Llama4TextExperts, "_unsloth_already_patched", False)
    monkeypatch.delattr(modeling_llama4, "Llama4Router", raising = False)
    patch_llama4_moe()
    assert Llama4TextExperts.forward is experts_forward
    assert Llama4TextMoe.forward is moe_forward


def test_per_expert_experts_keep_the_models_own_forward():
    try:
        from transformers.quantizers.base import SequentialLlama4TextExperts
    except Exception as e:
        pytest.skip(f"no SequentialLlama4TextExperts in this transformers: {e}")
    patch_llama4_moe()
    assert Llama4TextMoe.forward is Llama4TextMoe_forward
    moe = make_moe(1, torch.float32)
    config = Llama4TextConfig(
        hidden_size = HIDDEN, intermediate_size = 128, intermediate_size_mlp = 256,
        num_local_experts = 4, num_experts_per_tok = 1, num_hidden_layers = 1,
        num_attention_heads = 2, num_key_value_heads = 1, head_dim = 32, vocab_size = 256,
    )
    moe.experts = SequentialLlama4TextExperts(config).to(DEVICE)
    with torch.no_grad():
        for p in moe.experts.parameters():
            p.normal_(0, 0.05)
    x = torch.randn(3, 5, HIDDEN, device = DEVICE)
    out, logits = moe(x)
    flat = x.reshape(-1, HIDDEN)
    ref_logits = torch.nn.functional.linear(flat, moe.router.weight)
    top_value, top_index = torch.topk(ref_logits, 1, dim = 1)
    ref_out = moe.shared_expert(flat)
    for t in range(flat.shape[0]):
        e = int(top_index[t, 0])
        ref_out[t] += moe.experts[e](flat[t : t + 1] * torch.sigmoid(top_value[t, 0]))[0]
    assert torch.equal(logits, ref_logits)
    assert torch.allclose(out, ref_out, atol = 1e-5, rtol = 1e-5), (out - ref_out).abs().max()


@pytest.mark.parametrize("top_k", [1, 2])
def test_lora_wrapped_experts_take_the_routed_path(top_k):
    peft = pytest.importorskip("peft")
    patch_llama4_moe()
    moe = make_moe(top_k, torch.float32)
    x = torch.randn(3, 5, HIDDEN, device = DEVICE)
    ref_out, ref_logits = reference_forward(moe, x.clone())
    config = peft.LoraConfig(r = 4, lora_alpha = 8, target_modules = [],
                             target_parameters = ["experts.gate_up_proj", "experts.down_proj"])
    moe = peft.inject_adapter_in_model(config, moe)
    assert not isinstance(moe.experts, Llama4TextExperts)
    out, logits = moe(x)
    assert torch.equal(logits, ref_logits)
    tol = _fp32_tolerance(1e-5)
    assert torch.allclose(out, ref_out, atol = tol, rtol = tol), (out - ref_out).abs().max()
