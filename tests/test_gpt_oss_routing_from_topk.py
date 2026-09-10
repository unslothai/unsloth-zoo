"""topk_to_routing_tensors must order picks exactly as triton_kernels.routing does.

The reference below is triton_kernels.routing.routing_torch with the dataclasses
stripped, so the check is against the ordering the MXFP4 matmul_ogs kernels are
tested against upstream. It runs on CPU; no triton needed.
"""

import pytest
import torch

from unsloth_zoo.temporary_patches.gpt_oss import topk_to_routing_tensors


def _routing_torch_reference(logits, top_k):
    # triton_kernels/routing.py::routing_torch, sm_first=False, no user indices
    n_expts_tot = logits.shape[1]
    expt_scal, expt_indx = torch.topk(logits, top_k, dim=1)
    expt_scal = torch.softmax(expt_scal, dim=-1)
    expt_indx, order = torch.sort(expt_indx, dim=1)
    expt_scal = torch.gather(expt_scal, 1, order)
    expt_scal = expt_scal.reshape(-1)
    expt_indx = expt_indx.reshape(-1).to(torch.int32)
    combine_indx = torch.argsort(expt_indx, stable=True)
    dispatch_indx = torch.argsort(combine_indx, stable=True)
    gate_scal = expt_scal[combine_indx]
    hist = torch.histc(expt_indx.float(), bins=n_expts_tot, min=0, max=n_expts_tot - 1).int()
    return gate_scal, hist, combine_indx.int(), dispatch_indx.int()


def _hf_router(logits, top_k):
    # tail of GptOssTopKRouter.forward as patched in gpt_oss.py
    top_val, idx = torch.topk(logits, top_k, dim=-1)
    top_val = torch.softmax(top_val, dim=1, dtype=top_val.dtype)
    scores = torch.zeros_like(logits).scatter_(1, idx, top_val)
    return scores, idx


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("n_tokens, n_experts, top_k", [(37, 32, 4), (1, 32, 4), (64, 128, 4), (5, 8, 8)])
def test_matches_routing_torch(seed, n_tokens, n_experts, top_k):
    logits = torch.randn(n_tokens, n_experts, generator=torch.Generator().manual_seed(seed))
    scores, idx = _hf_router(logits, top_k)
    got = topk_to_routing_tensors(idx, scores, n_experts)
    want = _routing_torch_reference(logits, top_k)
    for g, w in zip(got, want):
        assert g.dtype == w.dtype
        torch.testing.assert_close(g, w, rtol=0, atol=0)


def test_weights_are_the_routers_not_renormalised():
    # The review's point: feeding the scores back through routing() as logits would
    # softmax an already-normalised distribution. gate_scal must be the router's
    # weights themselves, only permuted.
    n_tokens, n_experts, top_k = 16, 32, 4
    logits = torch.randn(n_tokens, n_experts, generator=torch.Generator().manual_seed(0))
    scores, idx = _hf_router(logits, top_k)
    gate_scal, hist, combine_indx, dispatch_indx = topk_to_routing_tensors(idx, scores, n_experts)

    # undo the expert-major permutation: slot s = token * top_k + j
    picks = torch.empty(n_tokens * top_k)
    picks[combine_indx.long()] = gate_scal
    picks = picks.view(n_tokens, top_k)
    assert torch.equal(picks.sort(dim=1).values, torch.gather(scores, 1, idx).sort(dim=1).values)
    torch.testing.assert_close(picks.sum(dim=1), torch.ones(n_tokens))
    assert int(hist.sum()) == n_tokens * top_k
    assert torch.equal(dispatch_indx[combine_indx.long()], torch.arange(n_tokens * top_k, dtype=torch.int32))


def test_rejects_topk_shaped_weights():
    idx = torch.zeros(4, 2, dtype=torch.int64)
    with pytest.raises(ValueError):
        topk_to_routing_tensors(idx, torch.ones(4, 2), n_expts_tot=8)
