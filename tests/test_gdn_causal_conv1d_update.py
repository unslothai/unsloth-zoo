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

"""Triton GatedDeltaNet `causal_conv1d_update` (cached decode) vs the transformers torch fallback."""

import inspect
import os
import types

import pytest
import torch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

from unsloth_zoo.temporary_patches import gdn_causal_conv1d as gcc  # noqa: E402

_HAS_CUDA = torch.cuda.is_available() and getattr(torch.version, "hip", None) is None
requires_cuda = pytest.mark.skipif(not _HAS_CUDA, reason = "needs a CUDA GPU")


def _inputs(B, D, T, W, L, dtype, bias, seed = 0):
    g = torch.Generator(device = "cuda").manual_seed(seed)
    # Decode feeds a transposed (B, T, D) projection: channel-last, like the modeling code.
    x = torch.randn(B, T, D, device = "cuda", generator = g).to(dtype).transpose(1, 2)
    state = torch.randn(B, D, L, device = "cuda", generator = g).to(dtype)
    w = (torch.randn(D, W, device = "cuda", generator = g) * 0.5).to(dtype)
    b = (torch.randn(D, device = "cuda", generator = g) * 0.1).to(dtype) if bias else None
    return x, state, w, b


def _to64(t):
    return t.double() if t is not None else None


def _rel(a, ref):
    a, ref = a.double(), ref.double()
    return ((a - ref).norm() / ref.norm().clamp_min(1e-30)).item()


@requires_cuda
@pytest.mark.parametrize("B,D,T,W,L", [
    (1, 64, 1, 4, 4), (2, 6144, 1, 4, 4), (3, 100, 1, 4, 3), (2, 130, 3, 4, 4),
    (1, 96, 7, 3, 2), (2, 70, 5, 2, 6), (4, 2048, 1, 4, 4), (1, 64, 64, 4, 4),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("act", ["silu", None])
def test_matches_reference(B, D, T, W, L, dtype, bias, act):
    x, state, w, b = _inputs(B, D, T, W, L, dtype, bias)
    ref_state = state.clone()
    ref = gcc.causal_conv1d_update_reference(x, ref_state, w, b, act)
    ours = gcc.triton_causal_conv1d_update(x, state, w, b, act)
    # The new state is a copy of inputs: exact.
    assert torch.equal(state, ref_state)
    assert ours.shape == ref.shape and ours.dtype == ref.dtype
    if dtype == torch.float32:
        torch.testing.assert_close(ours, ref, rtol = 1e-5, atol = 1e-5)
    else:
        # Both accumulate in fp32 and round the conv output once; summation order and the SiLU
        # formula differ by an ulp here and there, so judge both against an fp64 oracle.
        oracle = gcc.causal_conv1d_update_reference(_to64(x), _to64(state.clone()), _to64(w), _to64(b), act)
        assert _rel(ours, ref) < 5e-3
        assert _rel(ours, oracle) <= max(1.25 * _rel(ref, oracle), 1e-3)
        assert (ours == ref).float().mean().item() > 0.7


@requires_cuda
def test_non_contiguous_state_view():
    # The cache hands a view of a larger buffer; strides must be honoured.
    x, _, w, b = _inputs(2, 256, 1, 4, 4, torch.bfloat16, True)
    big = torch.randn(2, 256, 9, device = "cuda").to(torch.bfloat16)
    ref_big = big.clone()
    ref = gcc.causal_conv1d_update_reference(x, ref_big[:, :, 2:6], w, b, "silu")
    ours = gcc.triton_causal_conv1d_update(x, big[:, :, 2:6], w, b, "silu")
    assert torch.equal(big, ref_big)
    assert _rel(ours, ref) < 5e-3


@requires_cuda
def test_repeated_decode_matches_full_conv():
    # Token by token through the update == one causal conv over the whole sequence.
    B, D, T, W = 2, 192, 17, 4
    g = torch.Generator(device = "cuda").manual_seed(1)
    x = torch.randn(B, D, T, device = "cuda", generator = g, dtype = torch.float32)
    w = torch.randn(D, W, device = "cuda", generator = g, dtype = torch.float32)
    b = torch.randn(D, device = "cuda", generator = g, dtype = torch.float32)
    full = gcc.causal_conv1d_reference(x, w, b, "silu")
    state = torch.zeros(B, D, W, device = "cuda")
    steps = [gcc.triton_causal_conv1d_update(x[:, :, t:t + 1], state, w, b, "silu") for t in range(T)]
    torch.testing.assert_close(torch.cat(steps, dim = -1), full, rtol = 1e-5, atol = 1e-5)


@requires_cuda
def test_dispatch_fallbacks_and_kill_switch(monkeypatch):
    monkeypatch.delenv(gcc._KILL_SWITCH, raising = False)
    monkeypatch.setattr(gcc, "_update_broken", False)
    x, state, w, b = _inputs(1, 64, 1, 4, 4, torch.bfloat16, True)
    calls = []

    def torch_fn(hidden_states, conv_state, weight, bias = None, activation = None):
        calls.append(1)
        return gcc.causal_conv1d_update_reference(hidden_states, conv_state, weight, bias, activation)

    torch_fn.__module__ = "transformers.models.qwen3_5.modeling_qwen3_5"
    fn = gcc._make_update_dispatch(torch_fn)
    assert list(inspect.signature(fn).parameters) == ["hidden_states", "conv_state", "weight", "bias", "activation"]
    before = dict(gcc.GDN_CAUSAL_CONV1D_STATS)
    fn(x, state, w, b, "silu")
    assert gcc.GDN_CAUSAL_CONV1D_STATS["triton_update"] == before["triton_update"] + 1 and not calls
    fn(x.float().cpu(), state.float().cpu(), w.float().cpu(), b.float().cpu(), "silu")  # CPU
    fn(x, state, w.float(), b.float(), "silu")                                          # dtype mix
    fn(x, state, w, b, "gelu")                                                          # activation
    long_x = torch.randn(1, 64, gcc._MAX_UPDATE_T + 1, device = "cuda", dtype = torch.bfloat16)
    fn(long_x, state, w, b, "silu")                                                     # long T
    assert len(calls) == 4
    monkeypatch.setenv(gcc._KILL_SWITCH, "1")
    gcc._refresh_kill_switch()  # read once, not per call
    fn(x, state, w, b, "silu")
    monkeypatch.delenv(gcc._KILL_SWITCH)
    gcc._refresh_kill_switch()
    assert len(calls) == 5
    assert gcc.GDN_CAUSAL_CONV1D_STATS["triton_update"] == before["triton_update"] + 1
    assert gcc.GDN_CAUSAL_CONV1D_STATS["update_fallback"] == before["update_fallback"] + 5


@requires_cuda
@pytest.mark.parametrize("which", ["x", "weight", "bias"])
def test_grad_tracked_inputs_take_the_torch_path(monkeypatch, which):
    """The raw launch has no backward, so a grad-tracked decode step must keep autograd's graph."""
    monkeypatch.setattr(gcc, "_update_broken", False)
    monkeypatch.setattr(gcc, "_update_disabled", False)
    x, state, w, b = _inputs(1, 64, 1, 4, 4, torch.float32, True)
    leaf = {"x": x, "weight": w, "bias": b}[which].detach().requires_grad_(True)
    args = {"x": x, "weight": w, "bias": b, which: leaf}
    fn = gcc._make_update_dispatch(gcc.causal_conv1d_update_reference)
    before = dict(gcc.GDN_CAUSAL_CONV1D_STATS)
    out = fn(args["x"], state.clone(), args["weight"], args["bias"], "silu")
    assert out.requires_grad
    out.sum().backward()
    assert leaf.grad is not None and leaf.grad.abs().sum() > 0
    assert gcc.GDN_CAUSAL_CONV1D_STATS["triton_update"] == before["triton_update"]
    # Same tensors under no_grad: the Triton path is back.
    with torch.no_grad():
        fn(args["x"], state.clone(), args["weight"], args["bias"], "silu")
    assert gcc.GDN_CAUSAL_CONV1D_STATS["triton_update"] == before["triton_update"] + 1


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cached_launch_matches_reference_over_many_steps(monkeypatch, dtype):
    # Decode steps after the first go through the cached launcher; every one must match torch.
    monkeypatch.setattr(gcc, "_update_plans", {})
    monkeypatch.setattr(gcc, "_fast_launch_mode", None)
    monkeypatch.setattr(gcc, "_update_broken", False)
    monkeypatch.setattr(gcc, "_update_disabled", False)
    B, D, W, L = 2, 640, 4, 4
    g = torch.Generator(device = "cuda").manual_seed(3)
    w = (torch.randn(D, W, device = "cuda", generator = g) * 0.5).to(dtype)
    b = (torch.randn(D, device = "cuda", generator = g) * 0.1).to(dtype)
    state = torch.randn(B, D, L, device = "cuda", generator = g).to(dtype)
    ref_state = state.clone()
    for step in range(6):
        x = torch.randn(B, 1, D, device = "cuda", generator = g).to(dtype).transpose(1, 2)
        full = gcc.triton_causal_conv1d_update(x, state.clone(), w, b, "silu")
        ref = gcc.causal_conv1d_update_reference(x, ref_state, w, b, "silu")
        ours = gcc._try_fast_update(x, state, w, b, "silu")
        assert ours is not None and ours.stride() == x.stride()
        assert torch.equal(state, ref_state), step
        # Same kernel either way: the cached launch is bitwise the full launch.
        assert torch.equal(ours, full), step
        assert _rel(ours, ref) < (1e-5 if dtype == torch.float32 else 5e-3), step
    # One plan for the repeated shape, and the cached launcher is used only where it was verified.
    assert len(gcc._update_plans) == 1
    assert gcc._fast_launch_mode == ("all" if gcc._triton_launcher_verified() else False)


@requires_cuda
def test_unaligned_and_uncached_paths(monkeypatch):
    monkeypatch.setattr(gcc, "_update_plans", {})
    monkeypatch.setattr(gcc, "_fast_launch_mode", False)  # a Triton whose launcher did not check out
    monkeypatch.setattr(gcc, "_update_broken", False)
    monkeypatch.setattr(gcc, "_update_disabled", False)
    x, state, w, b = _inputs(1, 256, 1, 4, 4, torch.bfloat16, True)
    for _ in range(3):
        ref_state = state.clone()
        ref = gcc.causal_conv1d_update_reference(x, ref_state, w, b, "silu")
        assert _rel(gcc._try_fast_update(x, state, w, b, "silu"), ref) < 5e-3
        assert torch.equal(state, ref_state)
    # A 2-byte offset view: not 16-byte aligned, so never served from the cached plan.
    buf = torch.randn(1, 256 * 2 + 1, device = "cuda").to(torch.bfloat16)
    xu = buf[:, 1:257].unsqueeze(-1)
    monkeypatch.setattr(gcc, "_fast_launch_mode", None)
    for _ in range(3):
        ref_state = state.clone()
        ref = gcc.causal_conv1d_update_reference(xu, ref_state, w, b, "silu")
        assert _rel(gcc._try_fast_update(xu, state, w, b, "silu"), ref) < 5e-3
        assert torch.equal(state, ref_state)
    assert gcc._fast_launch_mode is None  # the unaligned input never reached the cached launcher


def test_non_tensor_arguments_fall_back_without_disabling(monkeypatch):
    monkeypatch.setattr(gcc, "_update_broken", False)
    monkeypatch.setattr(gcc, "_update_disabled", False)
    assert gcc._try_fast_update(torch.zeros(1, 4, 1), None, torch.zeros(4, 4), None, "silu") is None
    assert gcc._update_broken is False


def test_failure_disables_update_but_oom_propagates(monkeypatch):
    monkeypatch.delenv(gcc._KILL_SWITCH, raising = False)
    monkeypatch.setattr(gcc, "_update_broken", False)
    monkeypatch.setattr(gcc, "_update_disabled", False)
    x = torch.zeros(1, 4, 1)

    def oom(*a):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(gcc, "_run_update", oom)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        gcc._try_fast_update(x, x, x[0], None, "silu")
    assert gcc._update_broken is False

    def fail(*a):
        raise RuntimeError("triton launch failed")

    monkeypatch.setattr(gcc, "_run_update", fail)
    assert gcc._try_fast_update(x, x, x[0], None, "silu") is None
    assert gcc._update_broken is True


def test_counters_untouched_while_tracing(monkeypatch):
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    x, state, w = torch.randn(1, 8, 1), torch.randn(1, 8, 4), torch.randn(8, 4)
    before = dict(gcc.GDN_CAUSAL_CONV1D_STATS)
    gcc._make_update_dispatch(gcc.causal_conv1d_update_reference)(x, state, w, None, "silu")
    assert gcc.GDN_CAUSAL_CONV1D_STATS == before


def test_legacy_module_globals_rebound():
    # transformers 4.57.6 qwen3_next: module global None unless the package imported, and the
    # layer copies `causal_conv1d_update or torch_causal_conv1d_update` at __init__.
    legacy = types.ModuleType("transformers.models.qwen3_next.modeling_qwen3_next")
    legacy.causal_conv1d_fn = None
    legacy.causal_conv1d_update = None
    seen = []

    def torch_causal_conv1d_update(hidden_states, conv_state, weight, bias = None, activation = None):
        seen.append(1)
        return gcc.causal_conv1d_update_reference(hidden_states, conv_state, weight, bias, "silu")

    legacy.torch_causal_conv1d_update = torch_causal_conv1d_update
    assert gcc._rebind_module(legacy, None)
    assert legacy.causal_conv1d_fn is gcc._legacy_causal_conv1d_fn
    assert gcc._is_marked(legacy.causal_conv1d_update)
    first = legacy.causal_conv1d_update
    gcc._rebind_module(legacy, None)
    assert legacy.causal_conv1d_update is first
    # CPU tensors: the dispatch falls back to the module's own torch update.
    x, state, w = torch.randn(1, 8, 1), torch.randn(1, 8, 4), torch.randn(8, 4)
    legacy.causal_conv1d_update(x, state, w, None, "silu")
    assert seen == [1]


@pytest.fixture
def hub_kernels(monkeypatch):
    hub_kernels = pytest.importorskip("transformers.integrations.hub_kernels")
    if not hasattr(hub_kernels, "use_kernel_func_from_hub_with_fallback"):
        pytest.skip("transformers without use_kernel_func_from_hub_with_fallback")
    import transformers.integrations as integrations
    original = hub_kernels.use_kernel_func_from_hub_with_fallback
    if getattr(original, gcc._HUB_MARK, False):
        original = original.__wrapped__
    monkeypatch.setattr(hub_kernels, "use_kernel_func_from_hub_with_fallback", original)
    monkeypatch.setattr(integrations, "use_kernel_func_from_hub_with_fallback", original)
    monkeypatch.setattr(gcc, "_real_causal_conv1d_available", lambda: False)
    monkeypatch.delenv(gcc._KILL_SWITCH, raising = False)
    return hub_kernels


def _unpatched(modeling, name, decorator):
    fn = getattr(modeling, name)
    if gcc._is_marked(fn):
        dispatch = inspect.unwrap(fn, stop = lambda f: getattr(f, gcc._MARK, False))
        fn = decorator(name, "causal_conv1d")(dispatch.__wrapped__)
    return fn


def test_patch_rebinds_real_update_and_package_wins(hub_kernels, monkeypatch):
    pytest.importorskip("triton")
    modeling = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    if getattr(modeling, "causal_conv1d_update", None) is None:
        pytest.skip("transformers < 5.15 GatedDeltaNet")
    decorator = modeling.use_kernel_func_from_hub_with_fallback
    if getattr(decorator, gcc._HUB_MARK, False):
        decorator = decorator.__wrapped__
    for name in ("causal_conv1d_fn", "causal_conv1d_update"):
        monkeypatch.setattr(modeling, name, _unpatched(modeling, name, decorator))
    monkeypatch.setattr(modeling, "use_kernel_func_from_hub_with_fallback", decorator)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None)

    # A usable causal_conv1d package wins: nothing is rebound.
    monkeypatch.setattr(gcc, "_real_causal_conv1d_available", lambda: True)
    gcc.patch_gdn_causal_conv1d()
    assert not gcc._is_marked(modeling.causal_conv1d_update)
    # Kill switch: nothing is rebound either.
    monkeypatch.setattr(gcc, "_real_causal_conv1d_available", lambda: False)
    monkeypatch.setenv(gcc._KILL_SWITCH, "1")
    gcc.patch_gdn_causal_conv1d()
    assert not gcc._is_marked(modeling.causal_conv1d_update)
    monkeypatch.delenv(gcc._KILL_SWITCH)
    gcc.patch_gdn_causal_conv1d()
    assert gcc._is_marked(modeling.causal_conv1d_update)
    # CPU call through the real hub wrapper reaches the torch fallback, unchanged.
    x, state, w = torch.randn(1, 8, 1), torch.randn(1, 8, 4), torch.randn(8, 4)
    ref_state = state.clone()
    out = modeling.causal_conv1d_update(x, state, w, None, "silu")
    torch.testing.assert_close(out, gcc.causal_conv1d_update_reference(x, ref_state, w, None, "silu"))
    torch.testing.assert_close(state, ref_state)


def _preserve_gdn_modules(monkeypatch):
    """Restore every imported GDN modeling module afterwards: the patch rebinds all of them."""
    import sys
    names = [f"transformers.models.{p}.modeling_{p}" for p in gcc._GDN_MODELING]
    for module in [sys.modules[n] for n in names if n in sys.modules]:
        for name in ("causal_conv1d_fn", "causal_conv1d_update", "use_kernel_func_from_hub_with_fallback"):
            if name in vars(module):
                monkeypatch.setattr(module, name, getattr(module, name))


def _tiny_qwen3_5(modeling_module):
    from transformers import Qwen3_5ForCausalLM
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    config = Qwen3_5TextConfig(
        vocab_size = 256, hidden_size = 64, intermediate_size = 128, num_hidden_layers = 4,
        num_attention_heads = 4, num_key_value_heads = 2, head_dim = 16,
        linear_num_value_heads = 2, linear_num_key_heads = 2,
        linear_key_head_dim = 16, linear_value_head_dim = 16,
        layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    )
    torch.manual_seed(0)
    return Qwen3_5ForCausalLM(config).cuda().eval()


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_tiny_qwen3_5_cached_decode_matches_full_forward(hub_kernels, monkeypatch, dtype):
    modeling = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    if getattr(modeling, "causal_conv1d_update", None) is None:
        pytest.skip("transformers < 5.15 GatedDeltaNet")
    _preserve_gdn_modules(monkeypatch)
    gcc.patch_gdn_causal_conv1d()
    assert gcc._is_marked(modeling.causal_conv1d_update)
    model = _tiny_qwen3_5(modeling).to(dtype)
    ids = torch.randint(0, 256, (2, 24), device = "cuda", generator = torch.Generator(device = "cuda").manual_seed(0))
    prefill = 9
    with torch.no_grad():
        full = model(ids).logits
        before = dict(gcc.GDN_CAUSAL_CONV1D_STATS)
        out = model(ids[:, :prefill], use_cache = True)
        cache, steps = out.past_key_values, [out.logits]
        for t in range(prefill, ids.shape[1]):
            out = model(ids[:, t:t + 1], past_key_values = cache, use_cache = True)
            cache = out.past_key_values
            steps.append(out.logits)
    stepped = torch.cat(steps, dim = 1)
    n_decode = ids.shape[1] - prefill
    # Three linear-attention layers per decode step.
    assert gcc.GDN_CAUSAL_CONV1D_STATS["triton_update"] - before["triton_update"] == 3 * n_decode
    assert gcc.GDN_CAUSAL_CONV1D_STATS["update_fallback"] == before["update_fallback"]
    tol = 1e-4 if dtype == torch.float32 else 3e-2
    assert _rel(stepped, full) < tol
    assert torch.equal(stepped.argmax(-1), full.argmax(-1)) or dtype != torch.float32


@requires_cuda
def test_tiny_qwen3_next_cached_decode_matches_full_forward(monkeypatch):
    # Qwen3-Next on any transformers: 4.57.6 rebinds the module global the layer copies at
    # __init__, 5.x goes through the hub decorator.
    pytest.importorskip("triton")
    modeling = pytest.importorskip("transformers.models.qwen3_next.modeling_qwen3_next")
    from transformers import Qwen3NextConfig, Qwen3NextForCausalLM
    hub_kernels = pytest.importorskip("transformers.integrations.hub_kernels")
    import transformers.integrations as integrations
    _preserve_gdn_modules(monkeypatch)
    for target in (hub_kernels, integrations):
        if hasattr(target, "use_kernel_func_from_hub_with_fallback"):
            monkeypatch.setattr(target, "use_kernel_func_from_hub_with_fallback", target.use_kernel_func_from_hub_with_fallback)
    monkeypatch.setattr(gcc, "_real_causal_conv1d_available", lambda: False)
    monkeypatch.delenv(gcc._KILL_SWITCH, raising = False)
    gcc.patch_gdn_causal_conv1d()
    assert gcc._is_marked(modeling.causal_conv1d_update)
    config = Qwen3NextConfig(
        vocab_size = 256, hidden_size = 64, intermediate_size = 128, num_hidden_layers = 4,
        num_attention_heads = 4, num_key_value_heads = 2, head_dim = 16,
        linear_num_value_heads = 2, linear_num_key_heads = 2,
        linear_key_head_dim = 16, linear_value_head_dim = 16,
        num_experts = 4, num_experts_per_tok = 2, moe_intermediate_size = 32,
        shared_expert_intermediate_size = 32,
        layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    )
    # 4.57.6 calls the nonexistent torch.get_current_dtype for FusedRMSNormGated when unset.
    config.dtype = torch.float32
    torch.manual_seed(0)
    model = Qwen3NextForCausalLM(config).cuda().eval()
    ids = torch.randint(0, 256, (2, 20), device = "cuda", generator = torch.Generator(device = "cuda").manual_seed(0))
    prefill = 7
    with torch.no_grad():
        full = model(ids).logits
        before = dict(gcc.GDN_CAUSAL_CONV1D_STATS)
        out = model(ids[:, :prefill], use_cache = True)
        cache, steps = out.past_key_values, [out.logits]
        for t in range(prefill, ids.shape[1]):
            out = model(ids[:, t:t + 1], past_key_values = cache, use_cache = True)
            cache = out.past_key_values
            steps.append(out.logits)
    n_decode = ids.shape[1] - prefill
    assert gcc.GDN_CAUSAL_CONV1D_STATS["triton_update"] - before["triton_update"] == 3 * n_decode
    assert _rel(torch.cat(steps, dim = 1), full) < 1e-4
