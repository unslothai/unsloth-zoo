"""topk_to_routing_tensors must order picks exactly as triton_kernels.routing.routing_torch (CPU, no triton)."""

import pytest
import torch

from unsloth_zoo.temporary_patches import gpt_oss
from unsloth_zoo.temporary_patches.gpt_oss import topk_to_routing_tensors


def _routing_torch_reference(logits, top_k):
    # triton_kernels/routing.py::routing_torch, sm_first=False, no user indices
    n_expts_tot = logits.shape[1]
    expt_scal, expt_indx = torch.topk(logits, top_k, dim=1)
    expt_scal = torch.softmax(expt_scal, dim=-1)
    expt_indx, order = torch.sort(expt_indx, dim=1)
    expt_scal = torch.gather(expt_scal, 1, order).reshape(-1)
    expt_indx = expt_indx.reshape(-1).to(torch.int32)
    combine_indx = torch.argsort(expt_indx, stable=True)
    dispatch_indx = torch.argsort(combine_indx, stable=True)
    hist = torch.histc(expt_indx.float(), bins=n_expts_tot, min=0, max=n_expts_tot - 1).int()
    return expt_scal[combine_indx], hist, combine_indx.int(), dispatch_indx.int()


def _hf_router(logits, top_k, dense):
    # GptOssTopKRouter: transformers 4.x scatters to dense scores, 5.x returns them top-k aligned
    top_val, idx = torch.topk(logits, top_k, dim=-1)
    top_val = torch.softmax(top_val, dim=1, dtype=top_val.dtype)
    if dense:
        return torch.zeros_like(logits).scatter_(1, idx, top_val), idx
    return top_val, idx


@pytest.mark.parametrize("dense", [True, False])
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("n_tokens, n_experts, top_k", [(37, 32, 4), (1, 32, 4), (64, 128, 4), (5, 8, 8)])
def test_matches_routing_torch(dense, seed, n_tokens, n_experts, top_k):
    if not dense and top_k == n_experts:
        pytest.skip("top-k aligned weights are indistinguishable from dense when top_k == n_experts")
    logits = torch.randn(n_tokens, n_experts, generator=torch.Generator().manual_seed(seed))
    scores, idx = _hf_router(logits, top_k, dense)
    got = topk_to_routing_tensors(idx, scores, n_experts)
    for g, w in zip(got, _routing_torch_reference(logits, top_k)):
        assert g.dtype == w.dtype
        torch.testing.assert_close(g, w, rtol=0, atol=0)


def test_rejects_mismatched_weights():
    with pytest.raises(ValueError):
        topk_to_routing_tensors(torch.zeros(4, 2, dtype=torch.int64), torch.ones(4, 3), n_expts_tot=8)


@pytest.mark.parametrize("positional", [True, False])
def test_experts_routing_call_forms(monkeypatch, positional):
    seen = {}

    def fake_routing(module, router_indices, routing_weights):
        seen["shapes"] = (tuple(router_indices.shape), tuple(routing_weights.shape))
        return "rd", "gi", "si"

    monkeypatch.setattr(gpt_oss, "_mxfp4_routing_from_topk", fake_routing)
    hidden = torch.randn(2, 3, 16)
    idx = torch.zeros(6, 4, dtype=torch.int64)
    weights = torch.full((6, 4), 0.25)
    if positional:
        args = (hidden, idx, weights, None, None, None)
    else:
        args = (hidden, None, None, None, idx, weights)
    flat, rd, gi, si, leading = gpt_oss._mxfp4_experts_routing(None, *args)
    assert flat.shape == (6, 16) and (rd, gi, si) == ("rd", "gi", "si") and tuple(leading) == (2, 3)
    assert seen["shapes"] == ((6, 4), (6, 4))

    # mlp_forward's triton routing objects pass straight through
    out = gpt_oss._mxfp4_experts_routing(None, hidden, "rd", "gi", "si", None, None)
    assert out == (hidden, "rd", "gi", "si", None)

    with pytest.raises(TypeError):
        gpt_oss._mxfp4_experts_routing(None, hidden, None, None, None, idx, None)
