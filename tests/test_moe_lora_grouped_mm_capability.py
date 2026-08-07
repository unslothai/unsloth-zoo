# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""A device without torch._grouped_mm must not reach torch._grouped_mm.

`select_moe_backend` already falls to `unsloth_triton` when the probe says the
kernel is unsupported, but `forward_triton_grouped_gemm` applied its separated
LoRA delta through `native_moe_grouped_mm`, which calls the primitive the
backend was chosen to avoid. Training a LoRA MoE on anything that is not an
H100 therefore died at the first step:

    RuntimeError: torch._grouped_mm is only supported on CUDA devices with
    compute capability = 9.0

Found on a B200 running `Qwen3_5_MoE.ipynb` under torch 2.8.0, whose check is
`dprops->major == 9` exactly (Blas.cpp, `sm90_only`), so an A100 is refused the
same way. torch 2.6 and 2.7 have no `_grouped_mm` at all; 2.9 onwards falls back
inside torch, which is why only the older pins show this.

The guard goes in `_grouped_mm_with_backward_fix`, the one choke point every
LoRA and base path shares.
"""

import pytest
import torch

from unsloth_zoo.temporary_patches import moe_utils as M


@pytest.fixture
def unsupported(monkeypatch):
    """Report the kernel as unsupported, and make reaching it an error."""
    monkeypatch.setattr(M, "_check_torch_grouped_mm_supported", lambda: False)

    def _explode(*args, **kwargs):
        raise RuntimeError(
            "torch._grouped_mm is only supported on CUDA devices with "
            "compute capability = 9.0")
    monkeypatch.setattr(torch, "_grouped_mm", _explode, raising = False)


def _case(n_experts = 3, rows_each = 4, k = 8, n = 6, dtype = torch.float32):
    inputs = torch.randn(n_experts * rows_each, k, dtype = dtype)
    weight = torch.randn(n_experts, k, n, dtype = dtype)
    offsets = torch.tensor(
        [(i + 1) * rows_each for i in range(n_experts)], dtype = torch.int32)
    return inputs, weight, offsets


def test_it_does_not_call_the_kernel_at_all(unsupported):
    inputs, weight, offsets = _case()
    out = M._grouped_mm_with_backward_fix(inputs, weight, offsets)
    assert out.shape == (inputs.shape[0], weight.shape[-1])


def test_the_result_matches_a_per_expert_matmul(unsupported):
    inputs, weight, offsets = _case()
    out = M._grouped_mm_with_backward_fix(inputs, weight, offsets)
    expected = torch.cat([
        inputs[i * 4:(i + 1) * 4] @ weight[i] for i in range(3)], dim = 0)
    torch.testing.assert_close(out, expected)


def test_it_is_still_differentiable(unsupported):
    """The LoRA delta is trained, so the fallback has to carry gradients."""
    inputs, weight, offsets = _case()
    weight = weight.requires_grad_(True)
    inputs = inputs.requires_grad_(True)
    M._grouped_mm_with_backward_fix(inputs, weight, offsets).sum().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()


def test_an_empty_group_is_skipped(unsupported):
    """A router can leave an expert with no tokens; offsets then repeat."""
    inputs = torch.randn(4, 8)
    weight = torch.randn(3, 8, 6)
    offsets = torch.tensor([0, 4, 4], dtype = torch.int32)
    out = M._grouped_mm_with_backward_fix(inputs, weight, offsets)
    torch.testing.assert_close(out, inputs @ weight[1])


def test_the_lora_helper_goes_through_the_same_guard(unsupported):
    """`_apply_lora_grouped_mm` is the call site that actually failed."""
    inputs = torch.randn(12, 8)
    lora_B = torch.randn(3, 8, 2)
    lora_A = torch.randn(3, 2, 6)
    offsets = torch.tensor([4, 8, 12], dtype = torch.int32)
    out = M._apply_lora_grouped_mm(inputs, lora_B, lora_A, offsets, 0.5)
    expected = torch.cat([
        ((inputs[i * 4:(i + 1) * 4] @ lora_B[i]) @ lora_A[i]) * 0.5
        for i in range(3)], dim = 0)
    torch.testing.assert_close(out, expected)


def test_a_supported_device_still_uses_the_kernel(monkeypatch):
    """The guard must not cost H100 users the kernel they do have."""
    monkeypatch.setattr(M, "_check_torch_grouped_mm_supported", lambda: True)
    monkeypatch.setattr(M, "_transposed_view_grouped_mm_is_safe", lambda: True)
    seen = {}

    def _record(x, w, offs = None):
        seen["called"] = True
        return torch.cat([
            x[i * 4:(i + 1) * 4] @ w[i] for i in range(w.shape[0])], dim = 0)
    monkeypatch.setattr(torch, "_grouped_mm", _record, raising = False)

    inputs, weight, offsets = _case()
    M._grouped_mm_with_backward_fix(inputs, weight, offsets)
    assert seen.get("called"), "the guard swallowed a device that is supported"


# --- what the first live B200 run of Qwen3_5_MoE found ----------------------

def _reference(inputs, weight, offsets):
    """Plain autograd over per-group matmuls, gradients included."""
    outs, start = [], 0
    for g, end in enumerate(offsets.tolist()):
        if start < end:
            outs.append(inputs[start:end] @ weight[g])
        start = end
    return torch.cat(outs, dim = 0)


def test_the_gradients_match_plain_autograd(unsupported):
    inputs, weight, offsets = _case(dtype = torch.float64)
    a_in = inputs.clone().requires_grad_(True)
    a_w = weight.clone().requires_grad_(True)
    b_in = inputs.clone().requires_grad_(True)
    b_w = weight.clone().requires_grad_(True)
    seed = torch.randn(inputs.shape[0], weight.shape[-1], dtype = torch.float64)

    M._grouped_mm_with_backward_fix(a_in, a_w, offsets).backward(seed)
    _reference(b_in, b_w, offsets).backward(seed)

    torch.testing.assert_close(a_in.grad, b_in.grad)
    torch.testing.assert_close(a_w.grad, b_w.grad)


def test_a_gradient_is_only_produced_where_it_is_wanted(unsupported):
    """A frozen base stack asks for dX only, and building dW anyway is a
    full-size allocation per call for nothing."""
    inputs, weight, offsets = _case()
    inputs = inputs.requires_grad_(True)
    M._grouped_mm_with_backward_fix(inputs, weight, offsets).sum().backward()
    assert inputs.grad is not None
    assert weight.grad is None


def test_an_empty_group_contributes_no_gradient(unsupported):
    inputs = torch.randn(4, 8, requires_grad = True)
    weight = torch.randn(3, 8, 6, requires_grad = True)
    offsets = torch.tensor([0, 4, 4], dtype = torch.int32)
    M._grouped_mm_with_backward_fix(inputs, weight, offsets).sum().backward()
    assert torch.count_nonzero(weight.grad[0]) == 0
    assert torch.count_nonzero(weight.grad[2]) == 0
    assert torch.count_nonzero(weight.grad[1]) > 0


def test_it_saves_only_shape_stable_tensors(unsupported):
    """The whole point. A per-group slice's shape is the group size, which the
    router decides, so saving slices made non-reentrant checkpointing compare
    metadata across two different routings and abort the backward:

        CheckpointError: saved torch.Size([38, 8]) recomputed torch.Size([39, 8])

    Qwen3_5_MoE on a B200 died exactly there once the capability guard let it
    reach training. Saved shapes must depend on nothing but the inputs.
    """
    inputs, weight, offsets = _case()
    inputs = inputs.requires_grad_(True)
    out = M._grouped_mm_with_backward_fix(inputs, weight, offsets)
    saved = out.grad_fn.saved_tensors
    assert {tuple(t.shape) for t in saved} == {
        tuple(inputs.shape), tuple(weight.shape), tuple(offsets.shape)}, \
        [tuple(t.shape) for t in saved]


def test_it_survives_non_reentrant_checkpointing(unsupported):
    """End to end: the same call inside a checkpointed region, backward and all."""
    from torch.utils.checkpoint import checkpoint
    inputs, weight, offsets = _case()
    inputs = inputs.requires_grad_(True)
    weight = weight.requires_grad_(True)
    out = checkpoint(lambda x, w: M._grouped_mm_with_backward_fix(x, w, offsets),
                     inputs, weight, use_reentrant = False)
    out.sum().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()


def test_the_modulelist_stride_fallback_shares_the_same_helper():
    """`moe_grouped_modulelist._grouped_mm_fix` had its own copy of the loop.

    Two of its callers put the result on the tape, so the same slice-shaped
    saved tensors were reachable there through the 16-byte stride error.
    """
    from unsloth_zoo.temporary_patches import moe_grouped_modulelist as G

    def _stride_error(*args, **kwargs):
        raise RuntimeError("strides should be multiple of 16 bytes")

    inputs = torch.randn(12, 8, requires_grad = True)
    weight = torch.randn(3, 8, 6)
    offsets = torch.tensor([4, 8, 12], dtype = torch.int32)

    real = getattr(torch, "_grouped_mm", None)
    torch._grouped_mm = _stride_error
    try:
        out = G._grouped_mm_fix(inputs, weight, offsets)
    finally:
        if real is None: delattr(torch, "_grouped_mm")
        else: torch._grouped_mm = real

    saved = out.grad_fn.saved_tensors
    assert {tuple(t.shape) for t in saved} == {
        tuple(inputs.shape), tuple(weight.shape), tuple(offsets.shape)}, \
        [tuple(t.shape) for t in saved]


# --- what the first review round found --------------------------------------

def test_a_routing_change_in_the_replay_is_named_not_swallowed(unsupported):
    """Shape-stable saves are parity with the fused op, not permission to lie.

    Non-reentrant checkpointing replaces the SAVED tensors with the replay's,
    so a backward that just uses them pairs the original `grad_output` with the
    replay's partition: a gradient for a routing that never produced the loss.
    That is worse than the CheckpointError it replaced, so the forward's own
    boundaries are kept off the tape and compared.
    """
    class _Ctx:
        """What checkpointing hands backward: the REPLAY's saved tensors."""
        needs_input_grad = (True, True, False)
        def __init__(self, saved, forward_offsets):
            self.saved_tensors = saved
            self.forward_offsets = forward_offsets

    inputs = torch.randn(12, 8)
    weight = torch.randn(3, 8, 6)
    replayed = torch.tensor([4, 9, 12], dtype = torch.int32)   # one token over
    ctx = _Ctx((inputs, weight, replayed), [4, 8, 12])

    with pytest.raises(RuntimeError, match = "assigned tokens differently"):
        M._ManualGroupedMM.backward(ctx, torch.ones(12, 6))


def test_the_real_checkpoint_replay_shape_is_caught(unsupported):
    """End to end, with a router that really does route differently on replay."""
    from torch.utils.checkpoint import checkpoint
    seen = {"n": 0}

    def region(x, w):
        seen["n"] += 1
        ends = [4, 8, 12] if seen["n"] == 1 else [4, 9, 12]
        offsets = torch.tensor(ends, dtype = torch.int32)
        return M._grouped_mm_with_backward_fix(x, w, offsets)

    inputs = torch.randn(12, 8, requires_grad = True)
    weight = torch.randn(3, 8, 6, requires_grad = True)
    out = checkpoint(region, inputs, weight, use_reentrant = False)
    with pytest.raises(RuntimeError, match = "assigned tokens differently"):
        out.sum().backward()


def test_the_same_routing_twice_is_not_flagged(unsupported):
    """The check must not fire on the overwhelmingly common case."""
    inputs, weight, offsets = _case()
    inputs = inputs.requires_grad_(True)
    M._grouped_mm_with_backward_fix(inputs, weight, offsets).sum().backward()
    assert inputs.grad is not None


def test_backward_survives_a_reduced_precision_grad_output(unsupported):
    """`Function.backward` runs OUTSIDE the forward's autocast region, so a
    forward that autocast fp32 weights to bf16 hands back a bf16 grad_output
    while the saved tensors are still fp32. Unaligned, the first matmul raises."""
    inputs = torch.randn(8, 4, dtype = torch.float32, requires_grad = True)
    weight = torch.randn(2, 4, 6, dtype = torch.float32, requires_grad = True)
    offsets = torch.tensor([4, 8], dtype = torch.int32)

    out = M._grouped_mm_with_backward_fix(inputs, weight, offsets)
    out.backward(torch.ones_like(out, dtype = torch.bfloat16))

    assert inputs.grad.dtype == torch.float32
    assert weight.grad.dtype == torch.float32
    assert torch.isfinite(inputs.grad).all() and torch.isfinite(weight.grad).all()


def test_it_is_marked_as_not_compilable():
    """A data-dependent group loop cannot be traced; breaking cleanly beats
    aborting mid-trace, and the tensorized rewrites all cost more than they save."""
    assert getattr(M._manual_grouped_mm, "_torchdynamo_disable", False) or \
        "disable" in type(M._manual_grouped_mm).__name__.lower() or \
        hasattr(M._manual_grouped_mm, "__wrapped__"), M._manual_grouped_mm
