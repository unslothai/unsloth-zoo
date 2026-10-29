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

"""unsloth_zoo.triton_fast_launch: cached eager Triton launches.

CPU cases drive the cache, the bitwise self-check and the fallbacks with a stand-in kernel; the CUDA
cases launch real kernels (the gpt-oss decode ones included) both ways and compare bitwise.
"""
import pytest
import torch

from unsloth_zoo import triton_fast_launch as L


class _Compiled:
    def __init__(self, owner):
        self.owner = owner
        self.function = object()
        self.packed_metadata = object()

    @property
    def run(self):
        return self._run

    def _run(self, g0, g1, g2, stream, function, packed, meta, enter, exit_, *args):
        self.owner.fast_calls.append(args)
        mode = self.owner.fast_mode
        if mode == "raise":
            raise TypeError("launcher takes other arguments")
        if mode == "noop":
            return
        x, out, n, scale, flag = args[:5]
        out.copy_(x * scale + n + (100 if mode == "wrong" else 0))


class _Kernel:
    """kernel(X, OUT, n, scale, flag, BLOCK: constexpr): OUT = X * scale + n."""
    arg_names = ["X", "OUT", "n", "scale", "flag", "BLOCK"]

    def __init__(self, fast_mode = "ok"):
        self.fast_mode = fast_mode
        self.full_calls = []
        self.fast_calls = []

    def __getitem__(self, grid):
        def full(*args, **kwargs):
            self.full_calls.append((args, kwargs))
            x, out, n, scale, flag = args
            out.copy_(x * scale + n)
            return _Compiled(self)
        return full


@pytest.fixture
def fake_cuda(monkeypatch):
    monkeypatch.setattr(L, "_ENABLED", True)
    monkeypatch.setattr(L, "_current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(L, "_raw_stream", lambda device: 0)
    monkeypatch.setattr(L, "_hooks_set", lambda: False)
    monkeypatch.setattr(L, "_PLANS", {})
    monkeypatch.setattr(L, "_KERNEL_OK", {})
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(L, "FAST_LAUNCH_STATS", {"fast": 0, "full": 0})


def _call(kernel, x, out, n = 3, scale = 2.0, flag = True, block = 16):
    return L.launch(kernel, (1,), (x, out, n, scale, flag), dict(BLOCK = block), 0, (out,), num_warps = 4)


def test_first_full_then_checked_then_fast(fake_cuda):
    k = _Kernel()
    x = torch.arange(8, dtype = torch.float32)
    for step in range(4):
        out = torch.empty(8)
        _call(k, x, out)
        assert torch.equal(out, x * 2 + 3), step
    # Call 1 full (cached), call 2 full + checked fast launch, calls 3-4 fast only.
    assert len(k.full_calls) == 2 and len(k.fast_calls) == 3
    assert L._KERNEL_OK[id(k)] is True and L.FAST_LAUNCH_STATS["fast"] == 2
    # The constexpr rides along after the runtime arguments, as Triton passes it.
    assert k.fast_calls[-1][5:] == (16,)


@pytest.mark.parametrize("mode", ["noop", "wrong"])
def test_failed_self_check_keeps_triton_path(fake_cuda, mode):
    k = _Kernel(mode)
    x = torch.arange(8, dtype = torch.float32)
    for _ in range(3):
        out = torch.empty(8)
        _call(k, x, out)
        assert torch.equal(out, x * 2 + 3)  # restored to the full launch's result
    assert L._KERNEL_OK[id(k)] is False
    assert len(k.full_calls) == 3 and len(k.fast_calls) == 1 and L.FAST_LAUNCH_STATS["fast"] == 0


def test_launcher_error_on_self_check_disables_everything(fake_cuda):
    k, other = _Kernel("raise"), _Kernel()
    x = torch.arange(8, dtype = torch.float32)
    _call(other, x, torch.empty(8))
    _call(k, x, torch.empty(8))
    out = torch.empty(8)
    _call(k, x, out)
    assert torch.equal(out, x * 2 + 3)
    assert L._ENABLED is False and not L._PLANS and id(k) not in L._KERNEL_OK
    # Every kernel is back on Triton's path, not only the one that raised.
    n_fast = len(other.fast_calls)
    for _ in range(2):
        _call(other, x, torch.empty(8))
    assert len(other.fast_calls) == n_fast


def test_noop_launch_fails_check_on_nan_output(fake_cuda):
    # A NaN poison would equal a NaN reference bit for bit, so a launch writing nothing would pass.
    k = _Kernel("noop")
    x = torch.full((8,), float("nan"))
    for _ in range(3):
        _call(k, x, torch.empty(8))
    assert L._KERNEL_OK[id(k)] is False and L.FAST_LAUNCH_STATS["fast"] == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16, torch.int32, torch.int64, torch.uint8])
def test_poison_changes_every_element(dtype):
    t = (torch.arange(-4, 12) if dtype != torch.uint8 else torch.arange(16)).to(dtype)
    if dtype.is_floating_point:
        t[0], t[1], t[2] = float("nan"), float("inf"), -0.0
    ref = t.clone()
    L._poison(t)
    assert not (t.view(torch.uint8).view(-1, t.element_size()) == ref.view(torch.uint8).view(-1, t.element_size())).all(1).any()


def test_pre_run_hooks_use_triton(fake_cuda):
    k = _Kernel()
    k.pre_run_hooks = []
    x = torch.arange(8, dtype = torch.float32)
    _call(k, x, torch.empty(8))
    _call(k, x, torch.empty(8))
    k.pre_run_hooks.append(lambda *a, **kw: None)  # JITFunction.add_pre_run_hook
    n_full, n_fast = len(k.full_calls), len(k.fast_calls)
    for _ in range(3):
        _call(k, x, torch.empty(8))
    assert len(k.full_calls) == n_full + 3 and len(k.fast_calls) == n_fast


@pytest.mark.parametrize("knob", ["instrumentation_mode", "debug"])
def test_triton_knobs_use_triton(fake_cuda, monkeypatch, knob):
    from types import SimpleNamespace
    compilation = SimpleNamespace(instrumentation_mode = "consan" if knob == "instrumentation_mode" else "")
    runtime = SimpleNamespace(debug = knob == "debug")
    k = _Kernel()
    x = torch.arange(8, dtype = torch.float32)
    _call(k, x, torch.empty(8))
    _call(k, x, torch.empty(8))
    monkeypatch.setattr(L, "_knobs", (compilation, runtime))
    n_fast = len(k.fast_calls)
    for _ in range(3):
        _call(k, x, torch.empty(8))
    assert len(k.fast_calls) == n_fast
    monkeypatch.setattr(L, "_knobs", (SimpleNamespace(instrumentation_mode = ""), SimpleNamespace(debug = False)))
    _call(k, x, torch.empty(8))
    assert len(k.fast_calls) == n_fast + 1


def test_bias_table_revalidates_container(monkeypatch):
    from unsloth_zoo.temporary_patches import moe_routed as M
    if M._PTRS_EQUAL is None:
        pytest.skip("no _tensors_data_ptrs_at_indices_equal in this torch")
    calls = []
    monkeypatch.setattr(M, "lora_pointer_table", lambda ws: calls.append(len(ws)) or (torch.tensor([w.data_ptr() for w in ws]), 7))
    biases = [torch.randn(4) for _ in range(4)]
    tb = {"bias": biases, "bias_kind": 0}
    M._bias_args(tb)
    assert M._bias_args(tb)[0].numel() == 4 and calls == [4]
    biases.append(torch.randn(4))  # a longer list whose first entries kept their addresses
    assert M._bias_args(tb)[0].numel() == 5 and calls == [4, 5]
    del biases[-2:]
    assert M._bias_args(tb)[0].numel() == 3 and calls == [4, 5, 3]
    tb["bias"] = tuple(biases)
    assert M._bias_args(tb)[0].numel() == 3 and calls == [4, 5, 3, 3]


def test_launcher_error_after_check_disables_everything(fake_cuda):
    k = _Kernel()
    x = torch.arange(8, dtype = torch.float32)
    _call(k, x, torch.empty(8))
    _call(k, x, torch.empty(8))
    k.fast_mode = "raise"
    out = torch.empty(8)
    _call(k, x, out)
    assert torch.equal(out, x * 2 + 3)
    assert L._ENABLED is False and not L._PLANS


def test_specialization_changes_miss(fake_cuda):
    k = _Kernel()
    x = torch.arange(8, dtype = torch.float32)
    _call(k, x, torch.empty(8))
    _call(k, x, torch.empty(8))
    n_full = len(k.full_calls)
    # Another int value, a bool where an int was, another constexpr, dtype, alignment: each a new plan.
    _call(k, x, torch.empty(8), n = 4)
    _call(k, x, torch.empty(8), n = True)
    _call(k, x, torch.empty(8), block = 32)
    _call(k, x.double(), torch.empty(8, dtype = torch.float64))
    _call(k, torch.arange(9, dtype = torch.float32)[1:], torch.empty(8))
    assert len(k.full_calls) == n_full + 5
    # Floats are not specialized by Triton: another value reuses the plan.
    _call(k, x, torch.empty(8), scale = 5.0)
    assert len(k.full_calls) == n_full + 5


def test_bool_and_int_one_do_not_share_a_plan(fake_cuda):
    assert L._spec(True) != L._spec(1)


def test_hooks_or_other_device_use_triton(fake_cuda, monkeypatch):
    k = _Kernel()
    x = torch.arange(8, dtype = torch.float32)
    monkeypatch.setattr(L, "_hooks_set", lambda: True)
    for _ in range(3):
        _call(k, x, torch.empty(8))
    assert len(k.full_calls) == 3 and not k.fast_calls and not L._PLANS
    monkeypatch.setattr(L, "_hooks_set", lambda: False)
    monkeypatch.setattr(L, "_current_device", lambda: 1)
    for _ in range(2):
        L.launch(k, (1,), (x, torch.empty(8), 3, 2.0, True), dict(BLOCK = 16), 0, (), num_warps = 4)
    assert not k.fast_calls and not L._PLANS


def test_constexpr_names_must_follow_args(fake_cuda):
    k = _Kernel()
    x = torch.arange(8, dtype = torch.float32)
    for _ in range(3):
        L.launch(k, (1,), (x, torch.empty(8), 3, 2.0, True), dict(OTHER = 16), 0, (), num_warps = 4)
    assert not L._PLANS and not k.fast_calls


def test_capture_skips_the_check(fake_cuda, monkeypatch):
    k = _Kernel()
    x = torch.arange(8, dtype = torch.float32)
    _call(k, x, torch.empty(8))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    _call(k, x, torch.empty(8))
    assert id(k) not in L._KERNEL_OK and not k.fast_calls


cuda = pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason = "NVIDIA CUDA")


@cuda
def test_real_kernels_fast_path_bit_identical():
    if not L._ENABLED:
        pytest.skip("fast launch off for this Triton")
    from unsloth_zoo import gpt_oss_decode_fused as F
    torch.manual_seed(0)
    x = torch.randn(3, 2880, device = "cuda", dtype = torch.bfloat16)
    r = torch.randn_like(x)
    w = torch.randn(2880, device = "cuda", dtype = torch.bfloat16)
    a = torch.randn(1, 64, 3, 64, device = "cuda", dtype = torch.bfloat16)
    lse = torch.randn(1, 64, 3, device = "cuda")
    sinks = torch.randn(64, device = "cuda", dtype = torch.bfloat16)
    before = dict(L.FAST_LAUNCH_STATS)
    outs = [(F.add_rms_norm(x, r, w, 1e-5), F.sink_scale_output(a, lse, sinks)) for _ in range(4)]
    assert L.FAST_LAUNCH_STATS["fast"] - before["fast"] >= 4  # calls 3-4 of both kernels
    for (y, s), o in outs[1:]:
        assert torch.equal(y, outs[0][0][0]) and torch.equal(s, outs[0][0][1]) and torch.equal(o, outs[0][1])
    assert L._KERNEL_OK[id(F._add_rms_norm_kernel)] and L._KERNEL_OK[id(F._sink_scale_kernel)]


@cuda
def test_bias_pointer_table_follows_recast(monkeypatch):
    from unsloth_zoo.temporary_patches import moe_routed as M
    biases = [torch.randn(64, device = "cuda", dtype = torch.bfloat16) for _ in range(4)]
    params = [torch.nn.Parameter(b, requires_grad = False) for b in biases]
    tb = {"bias": list(params), "bias_kind": 0}
    calls = []
    real = M.lora_pointer_table
    monkeypatch.setattr(M, "lora_pointer_table", lambda ws: calls.append(1) or real(ws))
    t1, kind1 = M._bias_args(tb)
    t2, kind2 = M._bias_args(tb)
    assert t1 is t2 and kind1 == kind2 == M.LORA_KINDS[torch.bfloat16] and len(calls) == 1
    # bitsandbytes' recast: new storage behind the same Parameter.
    params[2].data = params[2].data.clone()
    t3, _ = M._bias_args(tb)
    assert len(calls) == 2 and t3.tolist() == [p.data_ptr() for p in params]
    for p in params:
        p.data = p.data.float()
    t4, kind4 = M._bias_args(tb)
    assert len(calls) == 3 and kind4 == M.LORA_KINDS[torch.float32] and t4.tolist() == [p.data_ptr() for p in params]
    assert M._bias_args(tb)[0] is t4 and len(calls) == 3
