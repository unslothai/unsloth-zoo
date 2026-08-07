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

"""Hopper gated-deltanet policy (unslothai/unsloth#5276, fla #640).

fla's gated ``chunk_bwd_dqkwg`` is miscompiled by Triton in [3.4.0, 3.7.1) on
Hopper. Upstream's own bisection pinned the trigger to a single block width --
BK 32 and BK 128 measure clean, BK 64 gives a dk max error of 14.65 -- and
explicitly ruled out the autotune config space. The vendored kernel therefore
steps 64 down to 32 and keeps the Triton fast path instead of raising.

``IS_NVIDIA_HOPPER`` is read at *call* time inside ``chunk_bwd_dqkwg``, so the
whole policy is exercisable on non-Hopper hardware by flipping that flag. That is
what lets these tests cover the fix without an H100.
"""

import os
import sys
import types

import pytest

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
os.environ.setdefault("UNSLOTH_VENDORED_FLA_NO_AUTORUN", "1")

try:
    import unsloth_zoo  # noqa: F401
except ImportError as _e:
    pytest.skip(f"unsloth_zoo unavailable: {_e}", allow_module_level=True)


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _load_vendored_fla():
    """Inject the vendored tree into this interpreter, or skip."""
    from unsloth_zoo.temporary_patches.fla_vendor import (
        _inject_vendored_fla,
        _vendored_injection_supported,
    )

    if not _vendored_injection_supported():
        pytest.skip("vendored fla kernels need CUDA + torch>=2.7 + triton>=3.3")
    injected, _ = _inject_vendored_fla()
    if not injected:
        pytest.skip("vendored fla injection failed on this host")
    import fla.ops.common.chunk_o as chunk_o
    return chunk_o


# ---------------------------------------------------------------------------
# 1. Tile selection: BK must never land on 64 in the affected range
# ---------------------------------------------------------------------------


def _record_bk(chunk_o, K, V, hopper, monkeypatch):
    """Run chunk_bwd_dqkwg's launcher and capture the BK it selects.

    The Triton kernel is swapped for a recorder, so no kernel is compiled or
    launched; only the block-size arithmetic under test runs.
    """
    import torch

    seen = {}

    class _Recorder:
        def __getitem__(self, grid):
            def launch(**kwargs):
                seen["BK"] = kwargs["BK"]
                seen["BV"] = kwargs["BV"]
                seen["grid"] = grid
            return launch

    monkeypatch.setattr(chunk_o, "chunk_bwd_kernel_dqkwg", _Recorder())
    monkeypatch.setattr(chunk_o, "IS_NVIDIA_HOPPER", hopper)

    B, T, H, HV, BT = 1, 128, 2, 2, 64
    dev, dt = "cuda", torch.bfloat16
    q = torch.randn(B, T, H, K, device=dev, dtype=dt)
    k = torch.randn(B, T, H, K, device=dev, dtype=dt)
    v = torch.randn(B, T, HV, V, device=dev, dtype=dt)
    do = torch.randn(B, T, HV, V, device=dev, dtype=dt)
    NT = T // BT
    h = torch.randn(B, NT, HV, K, V, device=dev, dtype=dt)
    dh = torch.randn(B, NT, HV, K, V, device=dev, dtype=dt)
    g = torch.randn(B, T, HV, device=dev, dtype=torch.float32)

    # chunk_bwd_dqkwg is wrapped by @dispatch('common'); call the undecorated
    # implementation so no backend can pre-empt the block-size arithmetic.
    impl = getattr(chunk_o.chunk_bwd_dqkwg, "__wrapped__", chunk_o.chunk_bwd_dqkwg)
    impl(q=q, k=k, v=v, do=do, h=h, dh=dh, g=g, scale=K ** -0.5, chunk_size=BT)
    return seen


@pytest.mark.skipif(not _cuda_available(), reason="needs CUDA")
@pytest.mark.parametrize("K", [16, 32, 60, 64, 65, 100, 128, 256])
def test_bk_64_is_never_selected_on_suspect_hopper(K, monkeypatch):
    """The whole fix in one assertion: on a Hopper host in the bad Triton range the
    launcher must not pick the miscompiled tile, for any head dim."""
    chunk_o = _load_vendored_fla()
    if not (chunk_o.TRITON_ABOVE_3_4_0 and not chunk_o.TRITON_ABOVE_3_7_1):
        pytest.skip("host Triton is outside the affected [3.4.0, 3.7.1) range")

    got = _record_bk(chunk_o, K=K, V=128, hopper=True, monkeypatch=monkeypatch)
    assert got["BK"] != 64, f"K={K} selected the miscompiled BK=64 tile"


@pytest.mark.skipif(not _cuda_available(), reason="needs CUDA")
def test_only_the_bad_tile_is_stepped_down(monkeypatch):
    """The override must be surgical: head dims that already choose a clean tile
    keep it, so Qwen3-Next (K=128 -> BK=128) is completely unaffected."""
    chunk_o = _load_vendored_fla()
    if not (chunk_o.TRITON_ABOVE_3_4_0 and not chunk_o.TRITON_ABOVE_3_7_1):
        pytest.skip("host Triton is outside the affected [3.4.0, 3.7.1) range")

    # K=128 is what Qwen3-Next / Qwen3.5 use (linear_key_head_dim). Upstream
    # measured BK=128 clean, so it must be untouched.
    assert _record_bk(chunk_o, K=128, V=128, hopper=True, monkeypatch=monkeypatch)["BK"] == 128
    assert _record_bk(chunk_o, K=128, V=128, hopper=False, monkeypatch=monkeypatch)["BK"] == 128

    # K=64 is the only shape family that reaches the bad tile, and only on Hopper.
    assert _record_bk(chunk_o, K=64, V=128, hopper=True, monkeypatch=monkeypatch)["BK"] == 32
    assert _record_bk(chunk_o, K=64, V=128, hopper=False, monkeypatch=monkeypatch)["BK"] == 64

    # BV is not implicated by #640 and must not be narrowed (it would cost speed).
    hop = _record_bk(chunk_o, K=64, V=128, hopper=True, monkeypatch=monkeypatch)
    assert hop["BV"] == 128

    # The grid is derived from the *overridden* BK: NK = cdiv(64, 32) = 2.
    assert hop["grid"][0] == 2


# ---------------------------------------------------------------------------
# 2. The override is a tiling change, not an algebra change
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _cuda_available(), reason="needs CUDA")
def test_bk_override_does_not_change_gradients(monkeypatch):
    """Run the real gated-delta backward at BK=64 and at the stepped-down BK=32 on
    hardware where both tiles are known good, and assert the gradients agree.

    This is the strongest statement available without a Hopper GPU: it shows the
    override only repartitions the reduction, so the correctness of BK=32 on SM90
    (measured upstream) carries over to the same kernel invocation we now issue.
    """
    import torch

    chunk_o = _load_vendored_fla()
    if not (chunk_o.TRITON_ABOVE_3_4_0 and not chunk_o.TRITON_ABOVE_3_7_1):
        pytest.skip("host Triton is outside the affected [3.4.0, 3.7.1) range")
    if chunk_o.IS_NVIDIA_HOPPER:
        pytest.skip("on real Hopper BK=64 is the miscompiled tile; nothing to compare")

    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    B, T, H, D = 2, 256, 4, 64      # D=64 -> BK would be 64 without the override

    def run(hopper):
        torch.manual_seed(0)
        dev, dt = "cuda", torch.bfloat16
        q = torch.randn(B, T, H, D, device=dev, dtype=dt, requires_grad=True)
        k = torch.randn(B, T, H, D, device=dev, dtype=dt, requires_grad=True)
        v = torch.randn(B, T, H, D, device=dev, dtype=dt, requires_grad=True)
        beta = torch.rand(B, T, H, device=dev, dtype=dt).requires_grad_(True)
        g = torch.nn.functional.logsigmoid(
            torch.rand(B, T, H, device=dev, dtype=torch.float32)
        ).requires_grad_(True)

        monkeypatch.setattr(chunk_o, "IS_NVIDIA_HOPPER", hopper)
        try:
            # Mirror Qwen3-Next's configuration: without the in-kernel q/k L2norm
            # and beta sigmoid the delta recurrence diverges to NaN over T steps on
            # random inputs, which would test the inputs rather than the tiling.
            o, _ = chunk_gated_delta_rule(
                q=q, k=k, v=v, g=g, beta=beta, scale=D ** -0.5,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
            )
            o.sum().backward()
        finally:
            monkeypatch.undo()
        grads = [t.grad.detach().float() for t in (q, k, v, beta, g)]
        for name, t in zip(("dq", "dk", "dv", "dbeta", "dg"), grads):
            assert torch.isfinite(t).all(), f"{name} is not finite (hopper={hopper})"
        return grads

    baseline = run(hopper=False)   # BK = 64
    stepped = run(hopper=True)     # BK = 32

    names = ("dq", "dk", "dv", "dbeta", "dg")
    for name, a, b in zip(names, baseline, stepped):
        denom = a.norm().clamp_min(1e-12)
        rel = (a - b).norm() / denom
        assert rel < 5e-3, f"{name}: BK=32 diverged from BK=64 (rel L2 {rel:.3e})"


# ---------------------------------------------------------------------------
# 3. No fla is left bound unpatched on a suspect host (the #5276 crash path)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_gated_delta_modeling(monkeypatch):
    """Stand-in transformers gated-delta modeling modules with live fla globals."""
    from unsloth_zoo.temporary_patches.fla_vendor import _GATED_DELTA_MODELING

    made = {}
    for pkg in _GATED_DELTA_MODELING:
        name = f"transformers.models.{pkg}.modeling_{pkg}"
        mod = types.ModuleType(name)
        mod.chunk_gated_delta_rule = lambda *a, **k: None
        mod.fused_recurrent_gated_delta_rule = lambda *a, **k: None
        mod.FusedRMSNormGated = object
        monkeypatch.setitem(sys.modules, name, mod)
        made[pkg] = mod
    return made


@pytest.mark.parametrize(
    "version", ["0.9.0", "0.5.0", None], ids=["newer", "older", "unversioned"],
)
def test_installed_fla_never_stays_bound_on_suspect_hopper(
    version, monkeypatch, fake_gated_delta_modeling,
):
    """Regression test for unslothai/unsloth#5276.

    A pip-installed fla-core carries the #640 miscompile whatever its version, and
    every version reached the RuntimeError before this fix: a *newer* install won
    the deferral check (which runs before the hardware gate), while an *older* one
    lost the deferral but then bailed at the hardware gate, leaving the installed
    fla importable so transformers' own probe still answered True. Neither may
    leave an unpatched fla serving the gated-delta backward.
    """
    from unsloth_zoo.temporary_patches import fla_vendor as fv

    real = types.ModuleType("fla")
    if version is not None:
        real.__version__ = version
    monkeypatch.setitem(sys.modules, "fla", real)
    monkeypatch.setattr(fv, "_hopper_dqkwg_suspect_here", lambda: True)

    injected = {}

    def fake_inject():
        injected["yes"] = True
        monkeypatch.setitem(sys.modules, "fla", types.ModuleType("fla"))
        return True, True

    monkeypatch.setattr(fv, "_inject_vendored_fla", fake_inject)
    monkeypatch.setattr(fv, "_torch_triton_cuda_supported", lambda: True)
    monkeypatch.setattr(fv, "_patch_is_available", lambda probe=None: True)
    monkeypatch.setattr(fv, "_repair_already_imported_modeling", lambda **kw: None)

    fv.patch_vendor_fla()

    assert injected.get("yes"), (
        "a suspect Hopper host must inject the vendored copy, which carries the "
        "BK tile fix, rather than defer to an unpatched user install"
    )


def test_installed_fla_still_wins_on_a_healthy_host(monkeypatch):
    """The Hopper override must not hijack normal hosts: a newer user install is
    still preferred everywhere else."""
    from unsloth_zoo.temporary_patches import fla_vendor as fv

    real = types.ModuleType("fla")
    real.__version__ = "0.9.0"
    monkeypatch.setitem(sys.modules, "fla", real)
    monkeypatch.setattr(fv, "_hopper_dqkwg_suspect_here", lambda: False)

    def fail_inject():
        raise AssertionError("must defer to the newer install, not inject")

    monkeypatch.setattr(fv, "_inject_vendored_fla", fail_inject)
    fv.patch_vendor_fla()
    assert sys.modules["fla"] is real


# ---------------------------------------------------------------------------
# 4. The opt-out
# ---------------------------------------------------------------------------


def test_opt_out_forces_pure_torch(monkeypatch, fake_gated_delta_modeling):
    """UNSLOTH_DISABLE_HOPPER_FLA_BWD=1 must make transformers take its pure-torch
    gated-delta path, which needs both a False availability probe and the module
    global unbound (the layer reads ``chunk_gated_delta_rule or torch_...``)."""
    import transformers.utils.import_utils as iu

    from unsloth_zoo.temporary_patches import fla_vendor as fv

    monkeypatch.setenv("UNSLOTH_DISABLE_HOPPER_FLA_BWD", "1")
    monkeypatch.setattr(fv, "_hopper_dqkwg_suspect_here", lambda: True)
    monkeypatch.setattr(fv, "_FLA_DISABLED_REASON", None)
    monkeypatch.setattr(iu, "is_flash_linear_attention_available", lambda: True, raising=False)

    def fail_inject():
        raise AssertionError("the opt-out must not inject the vendored tree")

    monkeypatch.setattr(fv, "_inject_vendored_fla", fail_inject)

    fv.patch_vendor_fla()

    assert iu.is_flash_linear_attention_available() is False
    reason = fv.fla_unavailable_reason()
    assert reason and "triton>=3.7.1" in reason

    for pkg, mod in fake_gated_delta_modeling.items():
        assert mod.chunk_gated_delta_rule is None, pkg
        # #640 is a chunked-backward bug; decode and the norm stay on fast kernels.
        assert mod.fused_recurrent_gated_delta_rule is not None, pkg
        assert mod.FusedRMSNormGated is not None, pkg


def test_opt_out_covers_models_the_vendored_tree_does_not(monkeypatch):
    """kimi_linear / olmo_hybrid are not vendor-covered, but a user-installed fla
    still exposes them to the same miscompile, so the opt-out must unbind them."""
    from unsloth_zoo.temporary_patches.fla_vendor import (
        _GATED_DELTA_MODELING,
        _REPAIR_MODELING,
    )

    assert set(_REPAIR_MODELING) < set(_GATED_DELTA_MODELING)
    assert {"kimi_linear", "olmo_hybrid"} <= set(_GATED_DELTA_MODELING)


def test_opt_out_outranks_the_source_preference_flags(monkeypatch, fake_gated_delta_modeling):
    """The other two flags choose *which* fla to use; this one is a correctness
    switch, so it has to win over both."""
    from unsloth_zoo.temporary_patches import fla_vendor as fv

    monkeypatch.setattr(fv, "_hopper_dqkwg_suspect_here", lambda: True)

    def fail_inject():
        raise AssertionError("the opt-out must not inject the vendored tree")

    monkeypatch.setattr(fv, "_inject_vendored_fla", fail_inject)

    for other in ("UNSLOTH_DISABLE_VENDORED_FLA", "UNSLOTH_FORCE_VENDORED_FLA"):
        monkeypatch.setenv("UNSLOTH_DISABLE_HOPPER_FLA_BWD", "1")
        monkeypatch.setenv(other, "1")
        for mod in fake_gated_delta_modeling.values():
            mod.chunk_gated_delta_rule = lambda *a, **k: None
        fv.patch_vendor_fla()
        for pkg, mod in fake_gated_delta_modeling.items():
            assert mod.chunk_gated_delta_rule is None, f"{other} defeated the opt-out for {pkg}"
        monkeypatch.delenv(other)


def test_opt_out_is_inert_off_hopper(monkeypatch, fake_gated_delta_modeling):
    """Setting the flag on a non-suspect host must change nothing at all."""
    from unsloth_zoo.temporary_patches import fla_vendor as fv

    monkeypatch.setenv("UNSLOTH_DISABLE_HOPPER_FLA_BWD", "1")
    monkeypatch.setattr(fv, "_hopper_dqkwg_suspect_here", lambda: False)
    monkeypatch.setattr(fv, "_FLA_DISABLED_REASON", None)
    monkeypatch.setattr(fv, "_should_defer_to_installed_fla", lambda: False)
    monkeypatch.setattr(fv, "_torch_triton_cuda_supported", lambda: False)

    fv.patch_vendor_fla()

    assert fv.fla_unavailable_reason() is None
    for mod in fake_gated_delta_modeling.values():
        assert mod.chunk_gated_delta_rule is not None


# ---------------------------------------------------------------------------
# 5. End to end: a real gated-deltanet model trains on a simulated Hopper
# ---------------------------------------------------------------------------


def _has_qwen3_next() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("transformers.models.qwen3_next") is not None
    except Exception:
        return False


@pytest.mark.skipif(not _cuda_available(), reason="needs CUDA")
@pytest.mark.skipif(not _has_qwen3_next(), reason="transformers lacks qwen3_next")
def test_qwen3_next_trains_with_simulated_hopper_tile(monkeypatch):
    """The payoff, end to end: with the Hopper flag forced on, a real Qwen3-Next
    gated-deltanet model must take the fla Triton path (not the pure-torch
    fallback) and produce finite gradients, where it previously raised."""
    import torch

    chunk_o = _load_vendored_fla()
    if not (chunk_o.TRITON_ABOVE_3_4_0 and not chunk_o.TRITON_ABOVE_3_7_1):
        pytest.skip("host Triton is outside the affected [3.4.0, 3.7.1) range")

    from unsloth_zoo.temporary_patches.fla_vendor import _patch_is_available
    _patch_is_available()

    from transformers import Qwen3NextConfig
    from transformers.models.qwen3_next import modeling_qwen3_next as mod

    if getattr(mod, "chunk_gated_delta_rule", None) is None:
        pytest.skip("modeling module was imported before injection; rebinding is "
                    "covered by the force-rebind subprocess test")

    monkeypatch.setattr(chunk_o, "IS_NVIDIA_HOPPER", True)

    cfg = Qwen3NextConfig(
        hidden_size=128, intermediate_size=256, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=256,
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=64, linear_value_head_dim=64,   # 64 -> the bad tile
        num_experts=2, num_experts_per_tok=1, shared_expert_intermediate_size=64,
    )
    # Qwen3NextGatedDeltaNet falls back to torch.get_current_dtype() when the config
    # carries no dtype, which does not exist on every supported torch. Set it so the
    # test exercises the kernels rather than that version mismatch.
    cfg.dtype = torch.bfloat16
    try:
        model = mod.Qwen3NextForCausalLM(cfg).cuda().to(torch.bfloat16)
    except AttributeError as e:
        pytest.skip(f"transformers/torch mismatch building Qwen3-Next: {e}")

    # The layer binds the kernel per-instance; confirm Triton is actually serving
    # this, otherwise a silent fallback would make the test vacuous.
    linear_layers = [
        getattr(layer, "linear_attn", None) for layer in model.model.layers
    ]
    linear_layers = [layer for layer in linear_layers if layer is not None]
    assert linear_layers, "no gated-deltanet layers in this config"
    assert all(
        layer.chunk_gated_delta_rule is not mod.torch_chunk_gated_delta_rule
        for layer in linear_layers
    ), "fell back to the pure-torch path; the Triton kernels were not exercised"

    ids = torch.randint(0, cfg.vocab_size, (1, 256), device="cuda")
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()

    assert torch.isfinite(out.loss), "loss is not finite"
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients were produced"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradients"


# ---------------------------------------------------------------------------
# 6. Non-Hopper hosts are untouched
# ---------------------------------------------------------------------------


def test_this_blackwell_host_is_not_suspect():
    """This host runs Triton 3.5.1, which *is* inside the affected version range,
    so only the architecture check keeps it off the Hopper path. Guards against a
    predicate change that would slow down every Blackwell user."""
    try:
        import torch
        import triton  # noqa: F401
    except Exception:
        pytest.skip("torch/triton unavailable")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if torch.cuda.get_device_capability(0)[0] == 9:
        pytest.skip("this host really is Hopper")

    from unsloth_zoo.temporary_patches.fla_vendor import _hopper_dqkwg_suspect_here

    assert _hopper_dqkwg_suspect_here() is False
