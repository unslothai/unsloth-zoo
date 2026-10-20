"""forward_native_moe_loop routes with one sort and one host read of the expert counts, and keeps the
per-expert (top-k slot, token) row order of the one_hot / torch.where routing it replaced, so the
output and every gradient are bit-identical."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import unsloth_zoo.temporary_patches.moe_utils as mu

E, H, I, R, SCALING = 16, 32, 24, 4, 0.5


def _experts():
    m = nn.Module()
    m.num_experts = E
    m.gate_up_proj = nn.Parameter(torch.randn(E, 2 * I, H) * 0.1)
    m.down_proj = nn.Parameter(torch.randn(E, H, I) * 0.1)
    m.act_fn = F.silu
    return m


def _reference(m, x, top_k_index, top_k_weights, lora):
    # The routing the loop used before: one_hot mask, nonzero, then torch.where per expert.
    (gu_a, gu_b), (dn_a, dn_b) = lora["gate_up_proj"], lora["down_proj"]
    out = torch.zeros_like(x)
    mask = F.one_hot(top_k_index, num_classes=E).permute(2, 1, 0)
    for e in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero():
        e = e.item()
        top_k_pos, token_idx = torch.where(mask[e])
        state = x[token_idx]
        gate_up = F.linear(state, m.gate_up_proj[e]) + ((state @ gu_a[e]) @ gu_b[e]) * SCALING
        gate, up = gate_up.chunk(2, dim=-1)
        h = F.silu(gate) * up
        h = F.linear(h, m.down_proj[e]) + ((h @ dn_a[e]) @ dn_b[e]) * SCALING
        out.index_add_(0, token_idx, h * top_k_weights[token_idx, top_k_pos, None])
    return out


def _grads(fn, m, x, top_k_weights, lora):
    for p in m.parameters():
        p.grad = None
    x = x.detach().clone().requires_grad_()
    w = top_k_weights.detach().clone().requires_grad_()
    out = fn(x, w)
    out.square().sum().backward()
    return [out.detach(), x.grad, w.grad, m.gate_up_proj.grad, m.down_proj.grad] + [t.grad for k in sorted(lora) for t in lora[k]]


@pytest.mark.parametrize("T,top_k", [(1, 2), (9, 2), (64, 4)])
def test_loop_matches_where_routing_bitwise_with_one_host_read(T, top_k, monkeypatch):
    torch.manual_seed(T)
    m = _experts()
    x = torch.randn(T, H)
    pool = torch.tensor([1, 2, 5, 6, 9, 13, 14])   # experts 0, 3, 4, ... never routed
    top_k_index = pool[torch.stack([torch.randperm(pool.numel())[:top_k] for _ in range(T)])]
    top_k_weights = torch.rand(T, top_k)
    lora = {
        "gate_up_proj": [torch.randn(E, H, R) * 0.1, torch.randn(E, R, 2 * I) * 0.1],
        "down_proj": [torch.randn(E, I, R) * 0.1, torch.randn(E, R, H) * 0.1],
    }
    ref_lora = {k: [t.clone().requires_grad_() for t in v] for k, v in lora.items()}
    new_lora = {k: [t.clone().requires_grad_() for t in v] for k, v in lora.items()}
    expected = _grads(lambda x, w: _reference(m, x, top_k_index, w, ref_lora), m, x, top_k_weights, ref_lora)

    monkeypatch.setattr(mu, "take_moe_lora_stash", lambda module, name: (*new_lora[name], SCALING))
    host_reads = {"item": 0, "tolist": 0}
    item, tolist = torch.Tensor.item, torch.Tensor.tolist

    def counted(name, fn):
        def wrapper(self, *args, **kwargs):
            host_reads[name] += 1
            return fn(self, *args, **kwargs)
        return wrapper

    monkeypatch.setattr(torch.Tensor, "item", counted("item", item))
    monkeypatch.setattr(torch.Tensor, "tolist", counted("tolist", tolist))
    got = _grads(lambda x, w: mu.forward_native_moe_loop(m, x, top_k_index, w), m, x, top_k_weights, new_lora)
    monkeypatch.setattr(torch.Tensor, "item", item)
    monkeypatch.setattr(torch.Tensor, "tolist", tolist)

    assert host_reads == {"item": 0, "tolist": 1}
    for a, b in zip(got, expected):
        assert torch.equal(a, b)
    unused = torch.ones(E, dtype=torch.bool)
    unused[pool] = False
    for g in got[-4:]:
        assert g[unused].abs().sum().item() == 0
