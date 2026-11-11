"""Unsloth gradient checkpointing: the backward replay of an MoE block must route tokens to the experts the first
(no-grad) pass picked. Under block compile the two passes run different compiled graphs whose rounding can move a
router input across a top-k boundary; the toy router below flips its choice in grad mode to model that."""

import os

import pytest
import torch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

import unsloth_zoo.gradient_checkpointing as gcm
from unsloth_zoo.gradient_checkpointing import (
    MOE_GC_ROUTING_STATS,
    Unsloth_Gradient_Checkpointer,
    Unsloth_Offloaded_Gradient_Checkpointer,
    moe_checkpoint_routing,
)

T, H, E, K = 24, 8, 6, 2


class ToyMoE(torch.nn.Module):
    def __init__(self, seed = 0, flip_in_grad_mode = True):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.router = torch.nn.Parameter(torch.randn(H, E, generator = g))
        self.experts = torch.nn.Parameter(torch.randn(E, H, H, generator = g) * 0.3)
        # Added to the logits only in grad mode: a large swing, so many tokens change experts in the replay.
        self.register_buffer("swing", torch.randn(E, generator = g) * 3.0)
        self.flip_in_grad_mode = flip_in_grad_mode

    def forward(self, x):
        logits = x @ self.router
        if self.flip_in_grad_mode and torch.is_grad_enabled():
            logits = logits + self.swing
        probs = torch.softmax(logits.float(), dim = -1)
        weights, index = torch.topk(probs, K, dim = -1)
        weights = (weights / weights.sum(-1, keepdim = True)).to(x.dtype)
        index, weights = moe_checkpoint_routing(self, index, weights)
        per_expert = torch.einsum("th,ehd->ted", x, self.experts)                 # [T, E, H]
        picked = per_expert.gather(1, index.unsqueeze(-1).expand(-1, -1, H))     # [T, K, H]
        return x + (picked * weights.unsqueeze(-1)).sum(1)


def _no_grad_routing(layer, x):
    with torch.no_grad():
        logits = x @ layer.router
        return torch.topk(torch.softmax(logits.float(), -1), K, -1).indices


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv("UNSLOTH_MOE_GC_SAME_ROUTING", raising = False)
    MOE_GC_ROUTING_STATS.clear()
    yield
    assert getattr(gcm._GC_ROUTING_TLS, "frame", None) is None


def _run(layer, x, checkpointer, fn = None):
    fn = fn or layer
    x = x.detach().clone().requires_grad_(True)
    layer.zero_grad(set_to_none = True)
    out = checkpointer.apply(fn, x) if checkpointer is not None else fn(x)
    loss = (out.float() ** 2).sum()
    loss.backward()
    return loss.detach(), layer.experts.grad.clone(), x.grad.clone()


def _reference(seed, x):
    """No checkpoint, the first pass's routing (no swing) differentiated directly."""
    ref = ToyMoE(seed, flip_in_grad_mode = False)
    return _run(ref, x, None)


@pytest.mark.parametrize("checkpointer", [Unsloth_Gradient_Checkpointer, Unsloth_Offloaded_Gradient_Checkpointer])
def test_replay_reuses_first_pass_routing_eager(checkpointer):
    torch.manual_seed(0)
    x = torch.randn(T, H)
    layer = ToyMoE(0)
    flipped = int((_no_grad_routing(layer, x).sort(-1).values
                   != torch.topk(torch.softmax((x @ layer.router + layer.swing).float(), -1), K, -1).indices.sort(-1).values)
                  .any(-1).sum())
    assert flipped > 0   # the toy really routes differently in grad mode
    loss, g_exp, g_x = _run(layer, x, checkpointer)
    ref_loss, ref_exp, ref_x = _reference(0, x)
    assert torch.equal(loss, ref_loss)
    torch.testing.assert_close(g_exp, ref_exp)        # expert grads: the routing the loss used
    assert MOE_GC_ROUTING_STATS["stored"] == 1 and MOE_GC_ROUTING_STATS["reused"] == 1


def test_kill_switch_keeps_replay_routing(monkeypatch):
    monkeypatch.setenv("UNSLOTH_MOE_GC_SAME_ROUTING", "0")
    torch.manual_seed(0)
    x = torch.randn(T, H)
    _, g_exp, _ = _run(ToyMoE(0), x, Unsloth_Gradient_Checkpointer)
    _, ref_exp, _ = _reference(0, x)
    assert not torch.allclose(g_exp, ref_exp)
    assert MOE_GC_ROUTING_STATS["reused"] == 0


def test_unchanged_routing_is_bitwise_identical():
    """No flip: checkpointed grads equal the plain (no-checkpoint) grads bit for bit."""
    torch.manual_seed(1)
    x = torch.randn(T, H)
    a = _run(ToyMoE(1, flip_in_grad_mode = False), x, Unsloth_Gradient_Checkpointer)
    b = _reference(1, x)
    for u, v in zip(a, b):
        assert torch.equal(u, v)
    assert MOE_GC_ROUTING_STATS["reused"] == 1


def test_no_frame_returns_same_objects():
    layer = ToyMoE(0)
    idx, w = torch.zeros(T, K, dtype = torch.long), torch.ones(T, K)
    with torch.no_grad():
        a, b = moe_checkpoint_routing(layer, idx, w)
    assert a is idx and b is w
    layer.eval()
    with gcm._checkpoint_routing_scope(gcm.CheckpointRoutingFrame(), False):
        a, b = moe_checkpoint_routing(layer, idx, w)
    assert a is idx and b is w and MOE_GC_ROUTING_STATS["stored"] == 0


def test_grad_mode_trial_in_first_pass_does_not_store():
    """Block compile runs a grad-mode trial inside the checkpoint's no-grad first pass: not the loss's forward."""
    torch.manual_seed(0)
    x = torch.randn(T, H)
    layer = ToyMoE(0)

    def with_trial(h):
        if not torch.is_grad_enabled():
            with torch.enable_grad():
                layer(h.detach().requires_grad_())
        return layer(h)
    _, g_exp, _ = _run(layer, x, Unsloth_Gradient_Checkpointer, with_trial)
    _, ref_exp, _ = _reference(0, x)
    torch.testing.assert_close(g_exp, ref_exp)
    assert MOE_GC_ROUTING_STATS["stored"] == 1 and MOE_GC_ROUTING_STATS["reused"] == 1


def test_mismatched_entry_falls_back():
    layer = ToyMoE(0)
    frame = gcm.CheckpointRoutingFrame()
    frame.entries.append((torch.zeros(3, K, dtype = torch.long), torch.ones(3, K)))
    frame.replaying = True
    idx, w = torch.zeros(T, K, dtype = torch.long), torch.ones(T, K)
    with gcm._checkpoint_routing_scope(frame, True):
        a, b = moe_checkpoint_routing(layer, idx, w)
    assert a is idx and b is w and MOE_GC_ROUTING_STATS["mismatch"] == 1


def test_dense_weights_straight_through():
    """GPT-OSS hands dense [T, E] scores: values from the first pass, gradient through the replay's scores."""
    idx0 = torch.tensor([[0, 1], [2, 3]])
    w0 = torch.tensor([[0.6, 0.4, 0.0, 0.0], [0.0, 0.0, 0.7, 0.3]])
    idx1 = torch.tensor([[0, 2], [2, 3]])
    w1 = torch.tensor([[0.5, 0.0, 0.5, 0.0], [0.0, 0.0, 0.6, 0.4]], requires_grad = True)
    out = gcm._blend_checkpoint_routing(idx1, w1, idx0, w0)
    assert torch.equal(out.detach(), w0)
    out.sum().backward()
    assert torch.equal(w1.grad, torch.ones_like(w1))


def test_aligned_weights_slot_matching():
    idx0 = torch.tensor([[3, 1]])
    w0 = torch.tensor([[0.7, 0.3]])
    idx1 = torch.tensor([[1, 5]])         # expert 1 kept (other slot), expert 3 dropped
    w1 = torch.tensor([[0.4, 0.6]], requires_grad = True)
    out = gcm._blend_checkpoint_routing(idx1, w1, idx0, w0)
    assert torch.equal(out.detach(), w0)
    (out * torch.tensor([[10.0, 1.0]])).sum().backward()
    # Slot of expert 1 (stored slot 1, multiplier 1) maps to replay slot 0; expert 5 takes nothing.
    assert torch.equal(w1.grad, torch.tensor([[1.0, 0.0]]))


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_compiled_replay_reuses_first_pass_routing(backend):
    """The routing read sits in an opaque op: fullgraph, one graph per grad mode, no recompile per call."""
    import torch._dynamo
    torch._dynamo.reset()
    from torch._dynamo.utils import counters
    counters.clear()
    torch.manual_seed(0)
    x = torch.randn(T, H)
    layer = ToyMoE(0)
    compiled = torch.compile(layer.forward, backend = backend, fullgraph = True, dynamic = False)
    for _ in range(3):
        MOE_GC_ROUTING_STATS.clear()
        loss, g_exp, _ = _run(layer, x, Unsloth_Gradient_Checkpointer, compiled)
        ref_loss, ref_exp, _ = _reference(0, x)
        torch.testing.assert_close(loss, ref_loss)
        torch.testing.assert_close(g_exp, ref_exp)
        assert MOE_GC_ROUTING_STATS["stored"] == 1 and MOE_GC_ROUTING_STATS["reused"] == 1
    assert sum(counters["graph_break"].values()) == 0
    assert counters["stats"]["unique_graphs"] == 2      # no-grad first pass + grad replay, nothing more
    # No frame (no checkpoint): the compiled grad-mode graph is reused and routes as the replay would.
    MOE_GC_ROUTING_STATS.clear()
    layer.flip_in_grad_mode = True
    _run(layer, x, None, compiled)
    assert counters["stats"]["unique_graphs"] == 2 and MOE_GC_ROUTING_STATS["reused"] == 0


def test_probe_forward_inside_trial_does_not_store():
    """Block compile's grad-mode trial runs inside the first pass, and the LoRA stash probe inside it runs a no_grad
    throwaway forward: a suspended frame keeps that routing (the grad graph's) out of the replay."""
    torch.manual_seed(0)
    x = torch.randn(T, H)
    layer = ToyMoE(0)

    def with_trial_and_probe(h):
        if not torch.is_grad_enabled():
            with torch.enable_grad():
                with gcm.suspended_checkpoint_routing(), torch.no_grad():
                    layer.flip_in_grad_mode, keep = False, layer.flip_in_grad_mode
                    layer(h + layer.swing[:1])   # some other routing, as a probe of another graph would see
                    layer.flip_in_grad_mode = keep
                layer(h.detach().requires_grad_())
        return layer(h)
    _, g_exp, _ = _run(layer, x, Unsloth_Gradient_Checkpointer, with_trial_and_probe)
    _, ref_exp, _ = _reference(0, x)
    torch.testing.assert_close(g_exp, ref_exp)
    assert MOE_GC_ROUTING_STATS["stored"] == 1 and MOE_GC_ROUTING_STATS["reused"] == 1
    assert MOE_GC_ROUTING_STATS["unmatched"] == 0
