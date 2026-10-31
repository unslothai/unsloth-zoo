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

"""A device without torch._grouped_mm support must not reach it: torch 2.8 checks dprops->major == 9 exactly (B200 and
A100 refused); the guard sits in _grouped_mm_with_backward_fix, the shared choke point; 2.6/2.7 lack it, 2.9 falls
back internally."""

from unittest import mock

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
    """A frozen base stack asks for dX only; building dW anyway is a full-size allocation per call for nothing."""
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
    """Saved shapes must depend only on inputs: a slice's shape is the router-decided group size, so saving slices made
    non-reentrant checkpointing abort with CheckpointError (Qwen3_5_MoE on B200)."""
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
    """moe_grouped_modulelist._grouped_mm_fix had its own copy of the loop, and two callers tape its result, so the same
    slice-shaped saved tensors were reachable via the 16-byte stride error."""
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
    """A replayed routing change must be named, not swallowed: non-reentrant checkpointing would pair the original
    grad_output with the replay's partition, so boundaries are compared off-tape."""
    inputs = torch.randn(12, 8)
    weight = torch.randn(3, 8, 6)
    replayed = torch.tensor([4, 9, 12], dtype = torch.int32)   # one token over
    ctx = _ReplayCtx(
        (inputs, weight, replayed),
        M._routing_signature(inputs, torch.tensor([4, 8, 12], dtype = torch.int32)),
    )

    with pytest.raises(RuntimeError, match = "assigned tokens differently"):
        M._ManualGroupedMM.backward(ctx, torch.ones(12, 6))


class _ReplayCtx:
    """What checkpointing hands backward: the REPLAY's saved tensors."""
    needs_input_grad = (True, True, False)
    def __init__(self, saved, forward_routing):
        self.saved_tensors = saved
        self.forward_routing = forward_routing


def test_a_count_preserving_reshuffle_is_caught_too(unsupported):
    """Offsets only record the routing histogram; swapping tokens between equal-sized experts keeps every boundary, so offsets-only checks miss the reorder."""
    inputs = torch.randn(12, 8)
    weight = torch.randn(3, 8, 6)
    offsets = torch.tensor([4, 8, 12], dtype = torch.int32)
    forward = M._routing_signature(inputs, offsets)

    reshuffled = inputs.clone()
    reshuffled[3], reshuffled[4] = inputs[4].clone(), inputs[3].clone()
    assert M._routing_signature(reshuffled, offsets) != forward

    ctx = _ReplayCtx((reshuffled, weight, offsets), forward)
    with pytest.raises(RuntimeError, match = "the same experts in a different order"):
        M._ManualGroupedMM.backward(ctx, torch.ones(12, 6))


def test_the_signature_is_one_device_transfer(unsupported):
    """The group loop already pays a sync; the checksum must not add another."""
    offsets = torch.tensor([4, 8, 12], dtype = torch.int32)
    calls = []
    real = torch.Tensor.cpu
    def counting_cpu(self, *a, **k):
        calls.append(tuple(self.shape))
        return real(self, *a, **k)
    with mock.patch.object(torch.Tensor, "cpu", counting_cpu):
        M._routing_signature(torch.randn(12, 8), offsets)
    assert len(calls) == 1, calls
    # 3 offsets + one checksum int per projection, packed into one transfer.
    assert calls[0] == (3 + M._SIGNATURE_WIDTH,)


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
    """Function.backward runs outside the forward's autocast, so grad_output can be bf16 while saved tensors stay fp32; unaligned, the first matmul raises."""
    inputs = torch.randn(8, 4, dtype = torch.float32, requires_grad = True)
    weight = torch.randn(2, 4, 6, dtype = torch.float32, requires_grad = True)
    offsets = torch.tensor([4, 8], dtype = torch.int32)

    out = M._grouped_mm_with_backward_fix(inputs, weight, offsets)
    out.backward(torch.ones_like(out, dtype = torch.bfloat16))

    assert inputs.grad.dtype == torch.float32
    assert weight.grad.dtype == torch.float32
    assert torch.isfinite(inputs.grad).all() and torch.isfinite(weight.grad).all()


def test_it_is_marked_as_not_compilable():
    """A data-dependent group loop cannot be traced, so it is marked not compilable: breaking cleanly beats aborting
    mid-trace, and tensorized rewrites cost more than they save."""
    assert getattr(M._manual_grouped_mm, "_torchdynamo_disable", False) or \
        "disable" in type(M._manual_grouped_mm).__name__.lower() or \
        hasattr(M._manual_grouped_mm, "__wrapped__"), M._manual_grouped_mm


def test_the_signature_ignores_autocast(unsupported):
    """mv/mm are on the autocast lower-precision list and only the forward sees it, so FP32 inputs hashed bf16 on one
    side and fp32 on the other, and every backward raised a routing error."""
    inputs = torch.randn(12, 8, dtype = torch.float32)
    offsets = torch.tensor([4, 8, 12], dtype = torch.int32)

    plain = M._routing_signature(inputs, offsets)
    with torch.autocast(device_type = "cpu", dtype = torch.bfloat16):
        under = M._routing_signature(inputs, offsets)
    assert under == plain


def test_a_row_orthogonal_to_the_projection_does_not_collide(unsupported):
    """A dot product per row maps it to a scalar, so rows orthogonal to the projection hash like the zero row; swapping
    such a pair across an expert boundary must still be seen."""
    hidden = 8
    ramp = torch.linspace(1.0, 2.0, hidden, dtype = torch.float32)
    p = torch.sin(ramp * M._SIGNATURE_STRIDES[0])
    orthogonal = torch.zeros(hidden)
    orthogonal[0], orthogonal[1] = p[1], -p[0]
    assert torch.isclose(orthogonal.dot(p), torch.tensor(0.0), atol = 1e-6)

    offsets = torch.tensor([1, 2], dtype = torch.int32)
    before = torch.stack([orthogonal, torch.zeros(hidden)])
    after = torch.stack([torch.zeros(hidden), orthogonal])
    assert M._routing_signature(before, offsets) != \
        M._routing_signature(after, offsets)


def test_no_signature_is_taken_when_no_backward_can_run(unsupported):
    """With grad off no node is built, so no signature is taken (the [T, hidden] projection is pure cost); a stash keyed
    on the weight missed replays and pinned GPU tensors per decode step."""
    inputs, weight, offsets = _case()
    taken = []
    real = M._routing_signature
    M._routing_signature = lambda *a, **k: (taken.append(1), real(*a, **k))[1]
    try:
        for ctx in (torch.inference_mode(), torch.no_grad()):
            taken.clear()
            with ctx:
                out = M._grouped_mm_with_backward_fix(inputs, weight, offsets)
            assert not taken, f"signature taken under {type(ctx).__name__}"
            assert out.shape == (inputs.shape[0], weight.shape[-1])

        M._grouped_mm_with_backward_fix(
            inputs.clone().requires_grad_(True), weight, offsets)
        assert taken, "signature skipped with grad ON"
    finally:
        M._routing_signature = real


def _live_tensor_count():
    """Counts live tensors via gc.get_objects, skipping dead weakref.proxy objects: isinstance on one raises
    ReferenceError, and any test in a shared xdist worker may leave one (PEFT parametrization makes them)."""
    import gc

    total = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                total += 1
        except ReferenceError:
            continue
    return total


def test_no_grad_decoding_retains_nothing(unsupported):
    """Nothing module-level may grow across grad-off calls: an earlier stash held a strong ref to each weight, and PEFT
    passes a fresh tensor per call, so long generations accumulated live GPU tensors."""
    import gc

    inputs, _, offsets = _case()
    before = _live_tensor_count()
    with torch.no_grad():
        for _ in range(50):
            M._grouped_mm_with_backward_fix(inputs, torch.randn(3, 8, 6), offsets)
    gc.collect()
    after = _live_tensor_count()
    assert after - before < 20, f"grad-off calls retained {after - before} tensors"


def test_the_live_tensor_walk_survives_a_dead_weakproxy():
    """Pins the dead-weakproxy guard in _live_tensor_count: without it, counting raises ReferenceError once another module in the worker left a dead proxy behind."""
    import gc
    import weakref

    class _Referent:
        pass

    referent = _Referent()
    proxy = weakref.proxy(referent)
    del referent
    gc.collect()

    # The proxy is dead but still reachable, exactly as it is in a shared xdist worker.
    with pytest.raises(ReferenceError):
        isinstance(proxy, torch.Tensor)

    count = _live_tensor_count()
    assert isinstance(count, int) and count >= 0


def test_the_signature_carries_a_term_the_projections_cannot_reach(unsupported):
    """Four projections leave a hidden - 4 null space; the norm term is not linear in the row so it escapes that space,
    and must reach the packed output. Checked structurally, since float32 cannot represent an exact null-space
    collision."""
    assert M._SIGNATURE_EXTRA == 1
    assert M._SIGNATURE_WIDTH == len(M._SIGNATURE_STRIDES) + M._SIGNATURE_EXTRA

    offsets = torch.tensor([2], dtype = torch.int32)
    hidden = 8
    # Same direction, different length: the norm separates them by magnitude, which no linear projection can be relied on to do inside the shared null space.
    unit = torch.zeros(2, hidden); unit[0, 0] = 1.0; unit[1, 0] = 1.0
    scaled = unit.clone(); scaled[1, 0] = 4.0
    assert M._routing_signature(unit, offsets) != \
        M._routing_signature(scaled, offsets)

    packed = M._routing_signature(unit, offsets)
    assert len(packed) == 1 + M._SIGNATURE_WIDTH


def test_the_forward_does_not_sync_the_offsets_twice(unsupported):
    """_routing_signature packs the offsets into its own transfer, so re-reading them in the group loop would add a second stream sync per grouped matmul."""
    inputs, weight, offsets = _case()
    calls = []
    real = torch.Tensor.cpu
    def counting_cpu(self, *a, **k):
        calls.append(tuple(self.shape))
        return real(self, *a, **k)
    with mock.patch.object(torch.Tensor, "cpu", counting_cpu):
        M._ManualGroupedMM.apply(inputs.requires_grad_(True), weight, offsets)
    assert len(calls) == 1, calls


def test_the_norm_does_not_materialize_fp32_copies(unsupported):
    """The norm uses vector_norm: (x.float() * x.float()).sum() held two fp32 copies and their product at once; asserted
    on source since CPU tests cannot snapshot the allocator."""
    import inspect

    body = inspect.getsource(M._routing_signature)
    assert "vector_norm" in body
    assert "inputs.float() * inputs.float()" not in body


def _backward_matmul_out_usage(out, seed):
    """Records which torch.matmul calls during out.backward(seed) passed out=; only the in-place branch calls
    torch.matmul by name, so an empty record means the casting branch ran."""
    from unittest import mock

    seen = []
    real = torch.matmul

    def recording_matmul(a, b, *, out = None):
        seen.append(out is not None)
        return real(a, b, out = out) if out is not None else real(a, b)

    with mock.patch.object(torch, "matmul", recording_matmul):
        out.backward(seed)
    return seen


def test_the_backward_writes_each_group_straight_into_its_slice(unsupported):
    """Backward writes each group with out= to avoid a temporary and a copy per group; pinned on the call, since gradients match either way."""
    inputs, weight, offsets = _case()
    out = M._ManualGroupedMM.apply(
        inputs.requires_grad_(True), weight.requires_grad_(True), offsets)

    seen = _backward_matmul_out_usage(out, torch.ones_like(out))
    assert seen and all(seen), seen


def test_a_cast_still_takes_the_temporary(unsupported):
    """out= cannot cast, so an autocast forward over fp32 saves must keep the plain path; left ungated it raises rather than writing."""
    inputs, weight, offsets = _case(dtype = torch.float32)
    inputs.requires_grad_(True)
    weight.requires_grad_(True)
    with torch.autocast(device_type = "cpu", dtype = torch.bfloat16):
        out = M._ManualGroupedMM.apply(inputs, weight, offsets)
    assert out.dtype == torch.bfloat16

    assert _backward_matmul_out_usage(out, torch.ones_like(out)) == []
    assert inputs.grad.dtype == torch.float32
    assert weight.grad.dtype == torch.float32


def test_writing_in_place_changes_no_gradient(unsupported):
    """Each group's gradient must land in its own slice; compared by closeness, not torch.equal, since out= rounding
    varies by BLAS build (a misplaced slice moves values by whole magnitudes)."""
    inputs, weight, offsets = _case(n_experts = 4, rows_each = 3, k = 6, n = 5)
    x = inputs.detach().clone().requires_grad_(True)
    w = weight.detach().clone().requires_grad_(True)
    out = M._ManualGroupedMM.apply(x, w, offsets)
    out.backward(torch.ones_like(out))

    same_form_x, same_form_w = torch.zeros_like(inputs), torch.zeros_like(weight)
    assign_x, assign_w = torch.zeros_like(inputs), torch.zeros_like(weight)
    start = 0
    for expert_idx, end in enumerate(offsets.tolist()):
        g = torch.ones(end - start, weight.shape[-1], dtype = inputs.dtype)
        torch.matmul(g, weight[expert_idx].transpose(-2, -1),
                     out = same_form_x[start:end])
        torch.matmul(inputs[start:end].transpose(-2, -1), g,
                     out = same_form_w[expert_idx])
        assign_x[start:end] = g @ weight[expert_idx].transpose(-2, -1)
        assign_w[expert_idx] = inputs[start:end].transpose(-2, -1) @ g
        start = end

    torch.testing.assert_close(x.grad, same_form_x)
    torch.testing.assert_close(w.grad, same_form_w)
    torch.testing.assert_close(x.grad, assign_x)
    torch.testing.assert_close(w.grad, assign_w)


def test_higher_order_gradients_still_work(unsupported):
    """out= refuses autograd, so the in-place write must stand down when the backward is itself differentiated;
    torch._grouped_mm double-backwards on a supported device (B200, torch 2.9.1), so a raising fallback would make
    results depend on the card."""
    inputs, weight, offsets = _case()
    x = inputs.detach().clone().requires_grad_(True)
    w = weight.detach().clone().requires_grad_(True)
    out = M._ManualGroupedMM.apply(x, w, offsets)
    (grad_x,) = torch.autograd.grad(out.sum(), x, create_graph = True)
    (second,) = torch.autograd.grad(grad_x.sum(), w)

    # Same quantity off plain autograd over the per-group loop.
    rx = inputs.detach().clone().requires_grad_(True)
    rw = weight.detach().clone().requires_grad_(True)
    parts, start = [], 0
    for expert_idx, end in enumerate(offsets.tolist()):
        parts.append(rx[start:end] @ rw[expert_idx])
        start = end
    (ref_grad_x,) = torch.autograd.grad(
        torch.cat(parts, 0).sum(), rx, create_graph = True)
    (ref_second,) = torch.autograd.grad(ref_grad_x.sum(), rw)
    torch.testing.assert_close(second, ref_second)


def test_an_ordinary_backward_still_writes_in_place(unsupported):
    """The autograd guard must not cost the common case: the engine runs a plain backward with grad mode off, so out= still applies."""
    inputs, weight, offsets = _case()
    out = M._ManualGroupedMM.apply(
        inputs.requires_grad_(True), weight.requires_grad_(True), offsets)
    assert torch.is_grad_enabled()          # the caller's mode is irrelevant
    seen = _backward_matmul_out_usage(out, torch.ones_like(out))
    assert seen and all(seen), seen


def _stub_triton_language(monkeypatch, has_make_tensor_descriptor):
    import sys, types
    triton = types.ModuleType("triton")
    language = types.ModuleType("triton.language")
    if has_make_tensor_descriptor:
        language.make_tensor_descriptor = lambda *a, **kw: None
    else:
        language._experimental_make_tensor_descriptor = lambda *a, **kw: None
    triton.language = language
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", language)


def test_the_triton_backend_is_not_offered_without_make_tensor_descriptor(monkeypatch):
    import sys, types
    for name in (
        "unsloth", "unsloth.kernels", "unsloth.kernels.moe",
        "unsloth.kernels.moe.grouped_gemm", "unsloth.kernels.moe.grouped_gemm.interface",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    interface = sys.modules["unsloth.kernels.moe.grouped_gemm.interface"]
    interface.grouped_gemm = lambda *a, **kw: None
    interface.supports_tma = lambda *a, **kw: False
    monkeypatch.setattr(M, "_init_triton_allocator", lambda *a, **kw: None, raising = False)
    monkeypatch.setattr(M.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(M, "is_mlx_available", lambda: False)
    monkeypatch.delenv("UNSLOTH_DISABLE_MOE_TRITON", raising = False)

    _stub_triton_language(monkeypatch, has_make_tensor_descriptor = False)
    monkeypatch.setattr(M, "_GROUPED_GEMM_AVAILABLE", None, raising = False)
    assert M._check_grouped_gemm_available() is False

    _stub_triton_language(monkeypatch, has_make_tensor_descriptor = True)
    monkeypatch.setattr(M, "_GROUPED_GEMM_AVAILABLE", None, raising = False)
    assert M._check_grouped_gemm_available() is True
    monkeypatch.setattr(M, "_GROUPED_GEMM_AVAILABLE", None, raising = False)


def test_the_triton_backend_is_not_offered_without_a_cuda_device(monkeypatch):
    """_check_grouped_gemm_available must refuse without CUDA even when the kernel imports, since the kernel asserts
    CUDA; the import is stubbed so the device check runs."""
    import sys, types

    for name in (
        "unsloth", "unsloth.kernels", "unsloth.kernels.moe",
        "unsloth.kernels.moe.grouped_gemm", "unsloth.kernels.moe.grouped_gemm.interface",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    interface = sys.modules["unsloth.kernels.moe.grouped_gemm.interface"]
    interface.grouped_gemm = lambda *a, **kw: None
    interface.supports_tma = lambda *a, **kw: False
    monkeypatch.setattr(M, "_init_triton_allocator", lambda *a, **kw: None, raising = False)
    _stub_triton_language(monkeypatch, has_make_tensor_descriptor = True)
    monkeypatch.setattr(M, "is_mlx_available", lambda: False)

    monkeypatch.setattr(M, "_GROUPED_GEMM_AVAILABLE", None, raising = False)
    monkeypatch.delenv("UNSLOTH_DISABLE_MOE_TRITON", raising = False)

    # The stub really is importable, so a False below is about the device and nothing else.
    monkeypatch.setattr(M.torch.cuda, "is_available", lambda: True)
    assert M._check_grouped_gemm_available() is True

    monkeypatch.setattr(M.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(M, "_GROUPED_GEMM_AVAILABLE", None, raising = False)
    assert M._check_grouped_gemm_available() is False

    # The selector must pick a backend that runs on this device; select_moe_backend is lru_cached, so clear it before
    # and after, or a native_torch pick made under these patches outlives them and later tests skip GPU backends.
    monkeypatch.setattr(M, "_TORCH_GROUPED_MM_SUPPORTED", False, raising = False)
    monkeypatch.setattr(M, "_GROUPED_GEMM_AVAILABLE", None, raising = False)
    monkeypatch.delenv("UNSLOTH_MOE_BACKEND", raising = False)
    M.select_moe_backend.cache_clear()
    try:
        assert M.select_moe_backend() == "native_torch"
    finally:
        M.select_moe_backend.cache_clear()
