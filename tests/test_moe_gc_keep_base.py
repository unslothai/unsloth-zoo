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

"""Gradient checkpoint replay reuses the first pass's frozen base expert outputs (moe_gc_keep_base): bitwise equal
gradients on the separated LoRA path, one base GEMM / dequant pass fewer, the memory cap, the kill switch."""

import pytest
import torch
import torch.nn as nn

import unsloth_zoo.gradient_checkpointing as GC
from unsloth_zoo.temporary_patches import moe_gc_keep_base as KB
from unsloth_zoo.temporary_patches import moe_grouped_fp16 as MG
from unsloth_zoo.temporary_patches import moe_lora_fold as LF
from unsloth_zoo.temporary_patches import moe_utils as MU

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or not MU._check_torch_grouped_mm_supported(),
    reason = "needs CUDA with torch._grouped_mm",
)
_ENV = ("UNSLOTH_MOE_GC_KEEP_BASE", "UNSLOTH_MOE_GC_KEEP_BASE_MB", "UNSLOTH_MOE_LORA_FOLD", "UNSLOTH_MOE_GC_SAME_ROUTING",
        "UNSLOTH_MOE_GROUPED_TRITON", "UNSLOTH_MOE_RECOMPUTE", "UNSLOTH_MOE_NF4_FUSED", "UNSLOTH_MOE_SELECTIVE_DEQUANT")


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setenv("UNSLOTH_MOE_SELECTIVE_DEQUANT", "0")
    MU._TRITON_GROUPED_MM_POLICY.clear()
    KB.KEEP_BASE_STATS.clear()
    KB._ORDER.clear()
    GC.MOE_GC_ROUTING_STATS.clear()
    yield
    MU._TRITON_GROUPED_MM_POLICY.clear()
    assert getattr(GC._GC_ROUTING_TLS, "frame", None) is None


def _nf4(w):
    from bitsandbytes.nn import Params4bit
    E, N, K = w.shape
    p = Params4bit(w.reshape(-1, K).cpu(), requires_grad = False, quant_type = "nf4", blocksize = 64,
                   compress_statistics = True, quant_storage = torch.uint8).to("cuda")
    p._original_shape = torch.Size((E, N, K))
    p.quant_state.dtype = torch.bfloat16
    return p


class Experts(nn.Module):
    def __init__(self, E, H, I, nf4 = False, seed = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        w1 = (torch.randn(E, 2 * I, H, generator = g) * H ** -0.5).to(torch.bfloat16)
        w2 = (torch.randn(E, H, I, generator = g) * I ** -0.5).to(torch.bfloat16)
        self.num_experts = E
        if nf4:
            self.gate_up_proj, self.down_proj = _nf4(w1), _nf4(w2)
        else:
            self.gate_up_proj = nn.Parameter(w1.cuda(), requires_grad = False)
            self.down_proj = nn.Parameter(w2.cuda(), requires_grad = False)
        self.act_fn = nn.SiLU()


class Layer(nn.Module):
    """Router + experts through forward_moe_backend (routing frame hook), LoRA r on gate_up and down."""

    def __init__(self, E = 8, H = 128, I = 96, k = 2, r = 16, nf4 = False, seed = 0, gu_lora = True, dn_lora = True):
        super().__init__()
        g = torch.Generator().manual_seed(seed + 10)
        self.router = nn.Linear(H, E, bias = False).to("cuda", torch.bfloat16)
        with torch.no_grad():   # seeded, so a routing pattern (and any failure) reproduces across processes
            self.router.weight.copy_(torch.randn(E, H, generator = g) * H ** -0.5)
        self.experts = Experts(E, H, I, nf4, seed)
        self.k = k
        mk = lambda *s: nn.Parameter((torch.randn(*s, generator = g) * 0.05).cuda())
        self.gu = nn.ParameterList([mk(E, H, r), mk(E, r, 2 * I)]) if gu_lora else None
        self.dn = nn.ParameterList([mk(E, I, r), mk(E, r, H)]) if dn_lora else None

    def forward(self, x):
        w, idx = torch.topk(torch.softmax(self.router(x).float(), -1), self.k, -1)
        w = (w / w.sum(-1, keepdim = True)).to(x.dtype)
        for name, lora in (("gate_up_proj", self.gu), ("down_proj", self.dn)):
            setattr(self.experts, MU.moe_lora_stash_name(name), None if lora is None else (lora[0], lora[1], 2.0))
        return x + MU.forward_moe_backend(self.experts, x, idx, w)


def _step(layers, x0, checkpointer = GC.Unsloth_Gradient_Checkpointer):
    for p in layers.parameters():
        p.grad = None
    x = x0.detach().clone().requires_grad_(True)
    h = x
    for layer in layers:
        h = checkpointer.apply(layer, h)
    loss = (h.float() ** 2).mean()
    loss.backward()
    grads = [p.grad.clone() for p in layers.parameters() if p.grad is not None]
    return loss.detach(), x.grad.clone(), grads


def _x(T = 64, H = 128):
    return (torch.randn(T, H, generator = torch.Generator().manual_seed(5)) * 0.5).cuda().to(torch.bfloat16)


@needs_gpu
@pytest.mark.parametrize("checkpointer", [GC.Unsloth_Gradient_Checkpointer, GC.Unsloth_Offloaded_Gradient_Checkpointer])
@pytest.mark.parametrize("nf4", [False, True])
def test_separated_lora_bitwise(monkeypatch, checkpointer, nf4):
    """LoRA fold off: the replay's kept base outputs are the tensors it would have recomputed, so every gradient
    is bitwise equal to keep off; with the switch on, the base GEMM runs in the first pass only."""
    if nf4:
        pytest.importorskip("bitsandbytes")
    monkeypatch.setenv("UNSLOTH_MOE_LORA_FOLD", "0")
    layers = nn.ModuleList([Layer(seed = 0, nf4 = nf4), Layer(seed = 1, nf4 = nf4)]).train()
    x0 = _x()
    calls = []
    real = MU._base_grouped_mm
    monkeypatch.setattr(MU, "_base_grouped_mm", lambda *a, **k: calls.append(torch.is_grad_enabled()) or real(*a, **k))
    monkeypatch.setenv("UNSLOTH_MOE_GC_KEEP_BASE", "0")
    off = _step(layers, x0, checkpointer)
    calls_off = list(calls)
    calls.clear()
    assert KB.KEEP_BASE_STATS["stored"] == 0
    monkeypatch.setenv("UNSLOTH_MOE_GC_KEEP_BASE", "1")
    on = _step(layers, x0, checkpointer)
    assert torch.equal(on[0], off[0]) and torch.equal(on[1], off[1])
    assert len(on[2]) == len(off[2]) == 2 * (1 + 4)
    for a, b in zip(on[2], off[2]):
        assert torch.equal(a, b)
    # 2 layers x 2 projections: first pass + replay without the switch, first pass only with it.
    assert calls_off.count(False) == 4 and calls_off.count(True) == 4
    assert calls == [False] * 4, calls
    assert KB.KEEP_BASE_STATS["stored"] == 4 and KB.KEEP_BASE_STATS["reused"] == 4
    assert KB.kept_base_bytes() == 0   # every kept output was taken back


@needs_gpu
def test_nf4_dequant_passes(monkeypatch):
    """QLoRA: 3 full-stack dequants per projection per step (first pass, replay, dX) become 2."""
    pytest.importorskip("bitsandbytes")
    monkeypatch.setenv("UNSLOTH_MOE_NF4_FUSED", "0")
    layers = nn.ModuleList([Layer(nf4 = True)]).train()
    x0 = _x()
    n = []
    real = MU._dequantize_bnb4bit_full
    monkeypatch.setattr(MU, "_dequantize_bnb4bit_full", lambda p: n.append(1) or real(p))
    for keep, want in (("0", 6), ("1", 4)):
        monkeypatch.setenv("UNSLOTH_MOE_GC_KEEP_BASE", keep)
        n.clear()
        _step(layers, x0)
        assert len(n) == want, (keep, len(n))


@needs_gpu
def test_budget_evicts_oldest_and_stays_exact(monkeypatch):
    monkeypatch.setenv("UNSLOTH_MOE_LORA_FOLD", "0")
    layers = nn.ModuleList([Layer(seed = s) for s in range(3)]).train()
    x0 = _x()
    monkeypatch.setenv("UNSLOTH_MOE_GC_KEEP_BASE", "0")
    off = _step(layers, x0)
    # One layer's gate_up [128, 192] + down [128, 128] bf16 = 80 KiB: room for about one layer.
    monkeypatch.setenv("UNSLOTH_MOE_GC_KEEP_BASE", "1")
    monkeypatch.setenv("UNSLOTH_MOE_GC_KEEP_BASE_MB", str(100 / 1024))
    on = _step(layers, x0)
    assert KB.KEEP_BASE_STATS["evicted"] > 0 and KB.KEEP_BASE_STATS["missing"] > 0
    assert KB.KEEP_BASE_STATS["bytes_peak"] <= 100 * 1024
    for a, b in zip([on[0], on[1], *on[2]], [off[0], off[1], *off[2]]):
        assert torch.equal(a, b)
    monkeypatch.setenv("UNSLOTH_MOE_GC_KEEP_BASE_MB", "0")
    KB.KEEP_BASE_STATS.clear()
    _step(layers, x0)
    assert KB.KEEP_BASE_STATS["stored"] == 0


@needs_gpu
def test_no_routing_frame_no_keep(monkeypatch):
    """Without the routing frame (UNSLOTH_MOE_GC_SAME_ROUTING=0) the replay may route differently: nothing kept."""
    monkeypatch.setenv("UNSLOTH_MOE_GC_SAME_ROUTING", "0")
    layers = nn.ModuleList([Layer()]).train()
    _step(layers, _x())
    assert KB.KEEP_BASE_STATS["stored"] == 0 and KB.KEEP_BASE_STATS["reused"] == 0


@needs_gpu
@pytest.mark.parametrize("gu_lora,dn_lora", [(True, False), (False, True), (False, False)])
def test_partial_lora_bitwise(monkeypatch, gu_lora, dn_lora):
    monkeypatch.setenv("UNSLOTH_MOE_LORA_FOLD", "0")
    layers = nn.ModuleList([Layer(gu_lora = gu_lora, dn_lora = dn_lora)]).train()
    x0 = _x()
    monkeypatch.setenv("UNSLOTH_MOE_GC_KEEP_BASE", "0")
    off = _step(layers, x0)
    monkeypatch.setenv("UNSLOTH_MOE_GC_KEEP_BASE", "1")
    on = _step(layers, x0)
    for a, b in zip([on[0], on[1], *on[2]], [off[0], off[1], *off[2]]):
        assert torch.equal(a, b)
    assert KB.KEEP_BASE_STATS["reused"] == 2


@needs_gpu
@pytest.mark.parametrize("nf4", [False, True])
def test_with_lora_fold(monkeypatch, nf4):
    """Fold on (Triton forced): both passes add the folded LoRA onto the kept base the same way. Two layers for the
    engagement counts; gradients vs main's separated path (fold off, keep off) on one layer, to bf16 rounding. With
    two layers a direct cosine is not a stable gate: rounding in layer 0 can flip a near-tied top-k in layer 1."""
    if nf4:
        pytest.importorskip("bitsandbytes")
    monkeypatch.setenv("UNSLOTH_MOE_GROUPED_TRITON", "1")
    MU._TRITON_GROUPED_MM_POLICY.clear()
    if torch.cuda.get_device_capability() < (8, 0) or not MG.triton_grouped_available(torch.device("cuda")):
        pytest.skip("Triton grouped GEMM unavailable")
    x0 = _x()

    def run(mods, fold, keep):
        monkeypatch.setenv("UNSLOTH_MOE_LORA_FOLD", fold)
        monkeypatch.setenv("UNSLOTH_MOE_GC_KEEP_BASE", keep)
        MU._TRITON_GROUPED_MM_POLICY.clear()
        loss, dx, grads = _step(mods, x0)
        return [loss, dx, *grads]

    two = nn.ModuleList([Layer(seed = 0, nf4 = nf4), Layer(seed = 1, nf4 = nf4)]).train()
    LF.LORA_FOLD_STATS["accumulate"] = 0
    run(two, "1", "1")
    assert LF.LORA_FOLD_STATS["accumulate"] == 8   # 2 layers x 2 projections x (first pass + replay)
    assert KB.KEEP_BASE_STATS["reused"] == 4
    one = nn.ModuleList([Layer(seed = 0, nf4 = nf4)]).train()
    sep = run(one, "0", "0")
    on = run(one, "1", "1")
    for a, b in zip(on, sep):
        cos = torch.nn.functional.cosine_similarity(a.double().flatten(), b.double().flatten(), dim = 0)
        assert cos > 0.9999, cos
        assert ((a.double() - b.double()).norm() / b.double().norm()) < 2e-2


@needs_gpu
def test_nf4_auto_above_cap_takes_kept_path(monkeypatch):
    """UNSLOTH_MOE_NF4_FUSED=auto in eager: above the row cap the regular dequant path runs (so the kept base and the
    LoRA fold apply), at or below it the fused NF4 GEMM does."""
    pytest.importorskip("bitsandbytes")
    from unsloth_zoo.temporary_patches import moe_nf4_grouped as NF
    if not NF._device_ok(torch.device("cuda"), "auto"):
        pytest.skip("the fused NF4 GEMM needs CUDA sm80+ (not ROCm)")
    monkeypatch.setenv("UNSLOTH_MOE_NF4_FUSED", "auto")
    layers = nn.ModuleList([Layer(nf4 = True)]).train()
    x0 = _x()
    monkeypatch.setenv("UNSLOTH_MOE_NF4_AUTO_MAX_ROWS", "1")     # 64 tokens x 2 / 8 experts = 16 rows: above
    before = NF.NF4_FUSED_STATS["launches"]
    _step(layers, x0)
    assert NF.NF4_FUSED_STATS["launches"] == before and KB.KEEP_BASE_STATS["reused"] == 2
    monkeypatch.setenv("UNSLOTH_MOE_NF4_AUTO_MAX_ROWS", "64")    # below: fused, nothing kept
    KB.KEEP_BASE_STATS.clear()
    _step(layers, x0)
    assert NF.NF4_FUSED_STATS["launches"] > before and KB.KEEP_BASE_STATS["stored"] == 0


class _Headroom(nn.Module):
    """Runs `layer` with the auto dequant headroom reading `gib` free, first pass and replay alike."""

    def __init__(self, layer, gib, state, outputs):
        super().__init__()
        self.layer, self.gib, self.state, self.outputs = layer, gib, state, outputs

    def forward(self, x):
        self.state["gib"] = self.gib
        out = self.layer(x)
        self.outputs.setdefault(id(self), []).append(out.detach().clone())
        return out


@needs_gpu
@pytest.mark.parametrize("fold", ["0", "1"])
def test_auto_dequant_chunk_decision_is_per_checkpoint_call(monkeypatch, fold):
    """Two QLoRA layers on one card: the first has headroom (whole stack), the second is chunked;
    each replay reuses its own first-pass decision (bitwise), not a device-wide one."""
    pytest.importorskip("bitsandbytes")
    for name in ("UNSLOTH_MOE_SELECTIVE_DEQUANT", "UNSLOTH_MOE_DEQUANT_CHUNK_EXPERTS", "UNSLOTH_MOE_DEQUANT_MAX_MB"):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setenv("UNSLOTH_MOE_NF4_FUSED", "0")
    monkeypatch.setenv("UNSLOTH_MOE_LORA_FOLD", fold)
    if fold == "1":
        monkeypatch.setenv("UNSLOTH_MOE_GROUPED_TRITON", "1")
        MU._TRITON_GROUPED_MM_POLICY.clear()
        if torch.cuda.get_device_capability() < (8, 0) or not MG.triton_grouped_available(torch.device("cuda")):
            pytest.skip("Triton grouped GEMM unavailable")
    # gate_up per expert 192 x 128 bf16 = 48 KiB, 8 experts = 384 KiB against a 2.5% x 4 MiB = 102 KiB cap.
    index = torch.cuda.current_device()
    monkeypatch.setitem(MU._DEVICE_TOTAL_BYTES, index, 4 * 2**20)
    state = {"gib": 0.0}
    monkeypatch.setattr(MU, "_device_headroom_bytes", lambda i: (int(state["gib"] * 2**30), lambda: int(state["gib"] * 2**30)))
    plans = []
    real_plan = MU._selective_dequant_plan
    monkeypatch.setattr(MU, "_selective_dequant_plan", lambda *a, **k: (lambda p: plans.append((state["gib"], p is None)) or p)(real_plan(*a, **k)))
    outputs = {}
    layers = []
    for seed, gib in ((0, 8.0), (1, 0.0)):
        layer = Layer(seed = seed, nf4 = True)
        layer.experts._unsloth_dequant_scratch_fraction = 0.025
        layers.append(_Headroom(layer, gib, state, outputs))
    layers = nn.ModuleList(layers).train()
    _step(layers, _x())
    # First pass A (whole), B (chunked); replay B then A, each as its first pass.
    assert plans == [(8.0, True), (0.0, False), (0.0, False), (8.0, True)], plans
    for wrapper in layers:
        first, replay = outputs[id(wrapper)]
        assert torch.equal(first, replay)
    assert KB.KEEP_BASE_STATS["reused"] == 2   # layer A's two projections
