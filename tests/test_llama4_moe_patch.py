# Llama-4 MoE on Unsloth's grouped expert path must reproduce the model's own
# dense forward: the sigmoid router score scales the expert INPUT, top_k pairs
# only, shared expert added, router logits returned.
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
    """transformers' Llama4TextMoe.forward as shipped (dense over all experts,
    router score multiplied into the input), written against the raw params."""
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


def make_moe(top_k, dtype, seed = 0):
    torch.manual_seed(seed)
    config = Llama4TextConfig(
        hidden_size = 64, intermediate_size = 96, intermediate_size_mlp = 128,
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
    x = torch.randn(3, 5, 64, device = DEVICE, dtype = dtype)

    ref_out, ref_logits = reference_forward(moe, x.clone())
    out, logits = moe(x)
    assert out.shape == ref_out.shape == (15, 64)
    assert torch.equal(logits, ref_logits)
    tol = 1e-5 if dtype is torch.float32 else 2e-2
    assert torch.allclose(out.float(), ref_out.float(), atol = tol, rtol = tol), \
        (out.float() - ref_out.float()).abs().max()


def test_backward_matches_reference_fp32():
    patch_llama4_moe()
    moe = make_moe(1, torch.float32)
    x = torch.randn(2, 6, 64, device = DEVICE)

    ref_out, _ = reference_forward(moe, x)
    ref_out.square().sum().backward()
    ref = {n: p.grad.clone() for n, p in moe.named_parameters() if p.grad is not None}
    moe.zero_grad(set_to_none = True)

    out, _ = moe(x)
    out.square().sum().backward()
    got = {n: p.grad.clone() for n, p in moe.named_parameters() if p.grad is not None}

    for name in ("experts.gate_up_proj", "experts.down_proj", "router.weight", "shared_expert.gate_proj.weight"):
        assert name in ref and name in got, name
        assert torch.allclose(got[name], ref[name], atol = 1e-4, rtol = 1e-4), \
            (name, (got[name] - ref[name]).abs().max())


def test_patch_is_idempotent():
    patch_llama4_moe()
    before = Llama4TextExperts.forward
    patch_llama4_moe()
    assert Llama4TextExperts.forward is before


def test_a_failed_moe_patch_leaves_both_forwards_alone(monkeypatch):
    # The patched experts forward needs routing indices and weights that only the patched MoE
    # forward passes, so a refused MoE patch must not leave the experts one installed.
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
