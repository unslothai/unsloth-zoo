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

"""Hopper gated-deltanet policy (unslothai/unsloth 5276, fla 640): fla chunk_bwd_dqkwg is miscompiled by Triton in
[3.4.0, 3.7.1) on Hopper at BK=64, so the vendored kernel steps down to BK=32 and keeps the Triton fast path instead
of raising; IS_NVIDIA_HOPPER is read at call time."""

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


_KEEP = object()


def _record_bk(chunk_o, K, V, hopper, monkeypatch, tensor_hopper=_KEEP):
    """Runs chunk_bwd_dqkwg's launcher with the Triton kernel swapped for a recorder, capturing the BK it selects; nothing is compiled or launched."""
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
    # The guard asks the tensor's device first and falls back to the global, so simulating Hopper sets both;
    # tensor_hopper sets the tensor probe (None means unknown), and _KEEP leaves a caller's _is_hopper_tensor patch in
    # place.
    monkeypatch.setattr(chunk_o, "IS_NVIDIA_HOPPER", hopper)
    if tensor_hopper is not _KEEP:
        monkeypatch.setattr(chunk_o, "_is_hopper_tensor", lambda x, _h=tensor_hopper: _h)

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

    # chunk_bwd_dqkwg is wrapped by @dispatch('common'); call the undecorated __wrapped__ so no backend can pre-empt the block-size arithmetic.
    impl = getattr(chunk_o.chunk_bwd_dqkwg, "__wrapped__", chunk_o.chunk_bwd_dqkwg)
    impl(q=q, k=k, v=v, do=do, h=h, dh=dh, g=g, scale=K ** -0.5, chunk_size=BT)
    return seen


@pytest.mark.skipif(not _cuda_available(), reason="needs CUDA")
@pytest.mark.parametrize("K", [16, 32, 60, 64, 65, 100, 128, 256])
def test_bk_64_is_never_selected_on_suspect_hopper(K, monkeypatch):
    """On a Hopper host in the bad Triton range the launcher must never pick the miscompiled BK=64 tile, for any head dim."""
    chunk_o = _load_vendored_fla()
    if not (chunk_o.TRITON_ABOVE_3_4_0 and not chunk_o.TRITON_ABOVE_3_7_1):
        pytest.skip("host Triton is outside the affected [3.4.0, 3.7.1) range")

    got = _record_bk(chunk_o, K=K, V=128, hopper=True, monkeypatch=monkeypatch, tensor_hopper=True)
    assert got["BK"] != 64, f"K={K} selected the miscompiled BK=64 tile"


@pytest.mark.skipif(not _cuda_available(), reason="needs CUDA")
def test_only_the_bad_tile_is_stepped_down(monkeypatch):
    """The override must be surgical: head dims already on a clean tile keep it, so Qwen3-Next (K=128, BK=128) is unaffected."""
    chunk_o = _load_vendored_fla()
    if not (chunk_o.TRITON_ABOVE_3_4_0 and not chunk_o.TRITON_ABOVE_3_7_1):
        pytest.skip("host Triton is outside the affected [3.4.0, 3.7.1) range")

    # K=128 is what Qwen3-Next and Qwen3.5 use (linear_key_head_dim); upstream measured BK=128 clean, so it must stay untouched.
    assert _record_bk(chunk_o, K=128, V=128, hopper=True, monkeypatch=monkeypatch, tensor_hopper=True)["BK"] == 128
    assert _record_bk(chunk_o, K=128, V=128, hopper=False, monkeypatch=monkeypatch, tensor_hopper=False)["BK"] == 128

    # K=64 is the only shape family that reaches the bad tile, and only on Hopper.
    assert _record_bk(chunk_o, K=64, V=128, hopper=True, monkeypatch=monkeypatch, tensor_hopper=True)["BK"] == 32
    assert _record_bk(chunk_o, K=64, V=128, hopper=False, monkeypatch=monkeypatch, tensor_hopper=False)["BK"] == 64

    # BV is not implicated by #640 and must not be narrowed (it would cost speed).
    hop = _record_bk(chunk_o, K=64, V=128, hopper=True, monkeypatch=monkeypatch, tensor_hopper=True)
    assert hop["BV"] == 128

    # The grid is derived from the *overridden* BK: NK = cdiv(64, 32) = 2.
    assert hop["grid"][0] == 2


@pytest.mark.skipif(not _cuda_available(), reason="needs CUDA")
def test_hopper_at_a_nonzero_device_index_still_steps_down(monkeypatch):
    """Hopper at a nonzero device index must still step down: fla's IS_NVIDIA_HOPPER is frozen from device 0 at import,
    while chunk_bwd_dqkwg picks CONST_TILING per tensor via k.device.index, so a per-device probe is needed on mixed
    hosts."""
    import torch

    chunk_o = _load_vendored_fla()
    if not (chunk_o.TRITON_ABOVE_3_4_0 and not chunk_o.TRITON_ABOVE_3_7_1):
        pytest.skip("host Triton is outside the affected [3.4.0, 3.7.1) range")

    real_cap, real_name = torch.cuda.get_device_capability, torch.cuda.get_device_name
    monkeypatch.setattr(
        torch.cuda, "get_device_capability",
        lambda i=None: (9, 0) if i == 0 else real_cap(i),
    )
    monkeypatch.setattr(
        torch.cuda, "get_device_name",
        lambda i=None: "NVIDIA H100 80GB HBM3" if i == 0 else real_name(i),
    )
    chunk_o._device_is_nvidia_hopper.cache_clear()
    try:
        got = _record_bk(chunk_o, K=64, V=128, hopper=False, monkeypatch=monkeypatch)
        assert got["BK"] == 32, (
            "a Hopper device that the import-time global missed still selected the "
            "miscompiled BK=64 tile"
        )
    finally:
        chunk_o._device_is_nvidia_hopper.cache_clear()

    # With the simulation undone the real devices are not Hopper, so nothing is stepped down; check_shared_mem('hopper')
    # is not a Hopper test since a Blackwell B200 also reports 232448 bytes, so it would wrongly narrow the tile.
    monkeypatch.undo()
    chunk_o._device_is_nvidia_hopper.cache_clear()
    assert _record_bk(chunk_o, K=64, V=128, hopper=False, monkeypatch=monkeypatch, tensor_hopper=False)["BK"] == 64


@pytest.mark.skipif(not _cuda_available(), reason="needs CUDA")
def test_bk_override_does_not_change_gradients(monkeypatch):
    """Runs the gated-delta backward at BK=64 and the stepped-down BK=32 on hardware where both are good; the override
    only repartitions the reduction, so gradients must agree."""
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
            # Mirror Qwen3-Next's config: without the in-kernel q/k L2norm and beta sigmoid the delta recurrence
            # diverges to NaN over T steps on random inputs, testing inputs not tiling.
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
    """Regression for unslothai/unsloth 5276: a pip-installed fla-core carries the fla 640 miscompile at any version, so
    neither a newer nor an older install may stay unpatched and serving the backward on suspect Hopper."""
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
    # Their install cannot be patched in place (a bare stub module here), so the vendored snapshot, which has the fix compiled in, must take over.
    monkeypatch.setattr(fv, "_patch_installed_fla_dqkwg", lambda: False)

    fv.patch_vendor_fla()

    assert injected.get("yes"), (
        "a suspect Hopper host must fall back to the vendored copy, which carries "
        "the BK tile fix, rather than leave an unpatched user install in charge"
    )


def test_installed_fla_is_patched_in_place_when_possible(monkeypatch):
    """On a suspect Hopper host, prefer patching only the miscompiled kernel inside the user's own fla over shadowing their deliberate install."""
    from unsloth_zoo.temporary_patches import fla_vendor as fv

    real = types.ModuleType("fla")
    real.__version__ = "0.9.0"
    monkeypatch.setitem(sys.modules, "fla", real)
    monkeypatch.setattr(fv, "_hopper_dqkwg_suspect_here", lambda: True)
    monkeypatch.setattr(fv, "_patch_installed_fla_dqkwg", lambda: True)

    def fail_inject():
        raise AssertionError("must patch the install in place, not shadow it")

    monkeypatch.setattr(fv, "_inject_vendored_fla", fail_inject)

    def fail_probe(probe=None):
        raise AssertionError(
            "must not repoint transformers' probe at the vendored answer when the "
            "user's own fla is the one in use"
        )

    monkeypatch.setattr(fv, "_patch_is_available", fail_probe)

    fv.patch_vendor_fla()
    assert sys.modules["fla"] is real


def test_installed_fla_still_wins_on_a_healthy_host(monkeypatch):
    """The Hopper override must not hijack normal hosts; a newer user-installed fla is still preferred everywhere else."""
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


@pytest.mark.skipif(not _cuda_available(), reason="needs CUDA")
def test_in_place_patch_fixes_an_unpatched_tree(monkeypatch):
    """_patch_installed_fla_dqkwg must make an fla that has the blanket guard but no tile fix run and give correct
    gradients; tested on the vendored tree with its fix neutralised."""
    import torch

    chunk_o = _load_vendored_fla()
    if not (chunk_o.TRITON_ABOVE_3_4_0 and not chunk_o.TRITON_ABOVE_3_7_1):
        pytest.skip("host Triton is outside the affected [3.4.0, 3.7.1) range")
    if chunk_o.IS_NVIDIA_HOPPER:
        pytest.skip("needs a non-Hopper host to hold a trustworthy reference")

    from unsloth_zoo.temporary_patches import fla_vendor as fv
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    import fla.ops.gated_delta_rule.chunk as gdc

    B, T, H, D = 1, 128, 2, 64      # D=64 is the head dim that reaches the bad tile

    def run():
        torch.manual_seed(0)
        dev, dt = "cuda", torch.bfloat16
        mk = lambda: torch.randn(B, T, H, D, device=dev, dtype=dt, requires_grad=True)
        q, k, v = mk(), mk(), mk()
        beta = torch.rand(B, T, H, device=dev, dtype=dt, requires_grad=True)
        g = torch.nn.functional.logsigmoid(
            torch.rand(B, T, H, device=dev, dtype=torch.float32)).requires_grad_(True)
        o, _ = chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta, scale=D ** -0.5,
            use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True,
        )
        o.sum().backward()
        return [t.grad.detach().float() for t in (q, k, v, beta, g)]

    reference = run()

    # Stand in for an unpatched install by restoring the old blanket guard, which refuses every gated call on Hopper regardless of the tile.
    pristine = chunk_o.chunk_bwd_dqkwg

    def blanket_guard(*args, **kwargs):
        if (kwargs.get("g") is not None and chunk_o.IS_NVIDIA_HOPPER
                and chunk_o.TRITON_ABOVE_3_4_0 and not chunk_o.TRITON_ABOVE_3_7_1):
            raise RuntimeError("Triton >= 3.4.0 and < 3.7.1 on Hopper GPUs ...")
        return pristine(*args, **kwargs)

    monkeypatch.setattr(chunk_o, "chunk_bwd_dqkwg", blanket_guard)
    monkeypatch.setattr(gdc, "chunk_bwd_dqkwg", blanket_guard)
    monkeypatch.setattr(chunk_o, "IS_NVIDIA_HOPPER", True)

    with pytest.raises(RuntimeError):
        run()

    pristine_check = chunk_o.check_shared_mem
    assert fv._patch_installed_fla_dqkwg() is True
    try:
        got = run()
        for name, a, b in zip(("dq", "dk", "dv", "dbeta", "dg"), reference, got):
            assert torch.isfinite(b).all(), f"{name} not finite after the in-place patch"
            rel = (a - b).norm() / a.norm().clamp_min(1e-12)
            assert rel < 5e-3, f"{name} diverged after the in-place patch (rel L2 {rel:.3e})"
        # The patch must not save, mutate and restore module globals per call, since autograd runs one thread per
        # device; the guard flag is cleared once and the tile override is thread-local.
        assert chunk_o.IS_NVIDIA_HOPPER is False
        assert fv._installed_fla_forcing_small_tile() is False
        assert chunk_o.check_shared_mem.__wrapped__ is pristine_check
        # Re-running the idempotent patch must report success, since patch_vendor_fla runs twice (import and
        # TEMPORARY_PATCHES) and False would make the second run shadow the user's fla.
        assert fv._patch_installed_fla_dqkwg() is True
    finally:
        chunk_o.chunk_bwd_dqkwg = pristine
        gdc.chunk_bwd_dqkwg = pristine
        chunk_o.check_shared_mem = pristine_check


def _fake_installed_fla(monkeypatch, version="0.9.0"):
    """Stand-in user-installed fla whose chunk_o has the layout the in-place patch recognises, so the real _patch_installed_fla_dqkwg runs end to end."""
    real = types.ModuleType("fla")
    real.__version__ = version
    chunk_o = types.ModuleType("fla.ops.common.chunk_o")
    chunk_o.IS_NVIDIA_HOPPER = True
    chunk_o.check_shared_mem = lambda arch="none", tensor_idx=0: arch != "nope"
    chunk_o.chunk_bwd_dqkwg = lambda **kw: "original"
    for name, mod in (
        ("fla", real),
        ("fla.ops", types.ModuleType("fla.ops")),
        ("fla.ops.common", types.ModuleType("fla.ops.common")),
        ("fla.ops.common.chunk_o", chunk_o),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    return real, chunk_o


def test_second_patch_run_does_not_shadow_the_users_fla(monkeypatch):
    """patch_vendor_fla runs twice (import and TEMPORARY_PATCHES); the second run must stop at the already-patched
    install rather than fall through to _inject_vendored_fla and purge the user's fla."""
    from unsloth_zoo.temporary_patches import fla_vendor as fv

    real, _chunk_o = _fake_installed_fla(monkeypatch)
    monkeypatch.setattr(fv, "_hopper_dqkwg_suspect_here", lambda: True)
    monkeypatch.setattr(fv, "_torch_triton_cuda_supported", lambda: True)
    monkeypatch.setattr(fv, "_patch_is_available", lambda probe=None: True)
    monkeypatch.setattr(fv, "_repair_already_imported_modeling", lambda **kw: None)

    injected = []

    def fake_inject():
        injected.append(1)
        sys.modules["fla"] = types.ModuleType("fla")
        return True, True

    monkeypatch.setattr(fv, "_inject_vendored_fla", fake_inject)

    fv.patch_vendor_fla()
    assert sys.modules["fla"] is real and not injected

    fv.patch_vendor_fla()
    assert not injected, "the second run shadowed the user's fla with the snapshot"
    assert sys.modules["fla"] is real


def test_purging_an_installed_fla_unbinds_the_models_we_cannot_rebind(
    monkeypatch, fake_gated_delta_modeling,
):
    """Shadowing a user fla on suspect Hopper must unbind models that cannot be rebound (e.g. olmo_hybrid), or they keep
    the unpatched chunk_gated_delta_rule and silently corrupt dk/dg."""
    from unsloth_zoo.temporary_patches import fla_vendor as fv

    real = types.ModuleType("fla")
    real.__version__ = "0.9.0"
    monkeypatch.setitem(sys.modules, "fla", real)
    monkeypatch.setattr(fv, "_hopper_dqkwg_suspect_here", lambda: True)
    monkeypatch.setattr(fv, "_torch_triton_cuda_supported", lambda: True)
    monkeypatch.setattr(fv, "_patch_is_available", lambda probe=None: True)
    monkeypatch.setattr(fv, "_repair_already_imported_modeling", lambda **kw: None)
    # Their install has an unrecognised layout, so the in-place patch cannot apply.
    monkeypatch.setattr(fv, "_patch_installed_fla_dqkwg", lambda: False)
    monkeypatch.setattr(
        fv, "_inject_vendored_fla",
        lambda: (sys.modules.__setitem__("fla", types.ModuleType("fla")), (True, True))[1],
    )

    fv.patch_vendor_fla()

    for pkg in ("olmo_hybrid",):
        assert fake_gated_delta_modeling[pkg].chunk_gated_delta_rule is None, (
            f"{pkg} still points at the purged, unpatched fla kernel"
        )
    # Vendor-covered models are rebound onto the fixed kernels instead, so they must NOT be unbound here.
    for pkg in fv._REPAIR_MODELING:
        assert fake_gated_delta_modeling[pkg].chunk_gated_delta_rule is not None, pkg


def test_in_place_patch_is_thread_safe(monkeypatch):
    """Gated-delta backwards run concurrently on per-device autograd threads, so IS_NVIDIA_HOPPER must not be saved and
    restored per call, or the blanket RuntimeError returns mid-backward."""
    import threading

    from unsloth_zoo.temporary_patches import fla_vendor as fv

    _real, chunk_o = _fake_installed_fla(monkeypatch)
    # The wrapper only overrides the tile for tensors on Hopper, so the fake tensors' device is declared Hopper; without
    # it the concurrency path is never entered on non-Hopper CI.
    monkeypatch.setattr(fv, "_device_index_is_hopper", lambda index: True)

    fast_entered = threading.Event()
    slow_inside = threading.Event()
    fast_done = threading.Event()
    seen = {}

    def original(**kwargs):
        tag = kwargs["tag"]
        if tag == "fast":
            fast_entered.set()
            slow_inside.wait(5)
        else:
            slow_inside.set()
            fast_done.wait(5)
        if chunk_o.IS_NVIDIA_HOPPER:
            raise RuntimeError(f"{tag}: blanket #640 guard fired mid-call")
        seen[tag] = 128 if chunk_o.check_shared_mem("hopper", 0) else 32
        return tag

    chunk_o.chunk_bwd_dqkwg = original
    assert fv._patch_installed_fla_dqkwg() is True
    patched = chunk_o.chunk_bwd_dqkwg

    class _K:
        shape = (1, 128, 2, 64)   # head dim 64 -> the tile that must be stepped down
        device = types.SimpleNamespace(index=0)
        is_cuda = True            # the wrapper skips non-CUDA tensors outright

    errors = []

    def call(tag):
        try:
            patched(g=object(), k=_K(), tag=tag)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    t_fast = threading.Thread(target=call, args=("fast",))
    t_fast.start()
    assert fast_entered.wait(5)
    t_slow = threading.Thread(target=call, args=("slow",))
    t_slow.start()
    t_slow.join(10)
    fast_done.set()
    t_fast.join(10)

    assert not errors, errors
    # Both concurrent K=64 calls must get the safe 32-wide tile; neither override may be cancelled by the other call's restore.
    assert seen == {"fast": 32, "slow": 32}, seen
    assert fv._installed_fla_forcing_small_tile() is False


def test_opt_out_forces_pure_torch(monkeypatch, fake_gated_delta_modeling):
    """UNSLOTH_DISABLE_HOPPER_FLA_BWD=1 forces pure torch only on the pre-47630 Transformers layout (probe False and
    module global unbound), so this test is pinned there; post-47630 see
    test_optout_falls_through_to_protection_on_the_new_layout."""
    import transformers.utils.import_utils as iu

    from unsloth_zoo.temporary_patches import fla_vendor as fv

    monkeypatch.setenv("UNSLOTH_DISABLE_HOPPER_FLA_BWD", "1")
    monkeypatch.setattr(fv, "_hopper_dqkwg_suspect_here", lambda: True)
    monkeypatch.setattr(fv, "_FLA_DISABLED_REASON", None)
    monkeypatch.setattr(fv, "_transformers_uses_availability_probe", lambda: True)
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
    """olmo_hybrid is not vendor-covered but a user fla still exposes it to the miscompile, so the opt-out must unbind
    it; Kimi Linear is left out because its KDA ops never reach chunk_bwd_dqkwg."""
    from unsloth_zoo.temporary_patches.fla_vendor import (
        _GATED_DELTA_MODELING,
        _REPAIR_MODELING,
    )

    assert set(_REPAIR_MODELING) < set(_GATED_DELTA_MODELING)
    assert "olmo_hybrid" in _GATED_DELTA_MODELING
    assert "kimi_linear" not in _GATED_DELTA_MODELING


@pytest.mark.parametrize("old_layout", [True, False], ids=["probe", "kernel-hub"])
def test_opt_out_outranks_the_source_preference_flags(
    old_layout, monkeypatch, fake_gated_delta_modeling,
):
    """The Hopper opt-out outranks both source flags: pre-47630 no gated-delta global may stay bound to fla; post-47630
    neither flag may return early, so the kernel-hub fla stays patched."""
    from unsloth_zoo.temporary_patches import fla_vendor as fv

    monkeypatch.setattr(fv, "_hopper_dqkwg_suspect_here", lambda: True)
    monkeypatch.setattr(fv, "_transformers_uses_availability_probe", lambda: old_layout)

    reached = []
    if old_layout:
        def fail_inject():
            raise AssertionError("the opt-out must not inject the vendored tree")

        monkeypatch.setattr(fv, "_inject_vendored_fla", fail_inject)
    else:
        monkeypatch.setattr(fv, "_warn_hopper_optout_degraded", lambda: None)
        monkeypatch.setattr(fv, "_patch_is_available", lambda probe=None: True)
        monkeypatch.setattr(fv, "_repair_already_imported_modeling", lambda **kw: None)
        monkeypatch.setattr(fv, "_should_defer_to_installed_fla", lambda: False)
        monkeypatch.setattr(fv, "_torch_triton_cuda_supported", lambda: True)
        # An earlier test may already have injected the vendored tree, short-circuiting the injection decision; force
        # the not-yet-injected state so the fall-through is what is tested.
        monkeypatch.setattr(fv, "_vendored_already_injected", lambda: False)
        monkeypatch.setattr(
            fv, "_inject_vendored_fla", lambda: (reached.append(True), (True, False))[1],
        )

    for other in ("UNSLOTH_DISABLE_VENDORED_FLA", "UNSLOTH_FORCE_VENDORED_FLA"):
        monkeypatch.setenv("UNSLOTH_DISABLE_HOPPER_FLA_BWD", "1")
        monkeypatch.setenv(other, "1")
        for mod in fake_gated_delta_modeling.values():
            mod.chunk_gated_delta_rule = lambda *a, **k: None
        reached.clear()
        fv.patch_vendor_fla()
        if old_layout:
            for pkg, mod in fake_gated_delta_modeling.items():
                assert mod.chunk_gated_delta_rule is None, f"{other} defeated the opt-out for {pkg}"
        else:
            assert reached, (
                f"{other} defeated the opt-out: the degraded opt-out must still "
                "reach the protection path, not return and leave the kernel-hub "
                "decorator resolving an unpatched fla"
            )
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


def _has_qwen3_next() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("transformers.models.qwen3_next") is not None
    except Exception:
        return False


@pytest.mark.skipif(not _cuda_available(), reason="needs CUDA")
@pytest.mark.skipif(not _has_qwen3_next(), reason="transformers lacks qwen3_next")
def test_qwen3_next_trains_with_simulated_hopper_tile(monkeypatch):
    """With the Hopper flag forced on, a real Qwen3-Next gated-deltanet model must take the fla Triton path and produce finite gradients instead of raising."""
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
    # Qwen3NextGatedDeltaNet falls back to torch.get_current_dtype() when the config has no dtype, a function missing on
    # some supported torch; set cfg.dtype to exercise the kernels.
    cfg.dtype = torch.bfloat16
    try:
        model = mod.Qwen3NextForCausalLM(cfg).cuda().to(torch.bfloat16)
    except AttributeError as e:
        pytest.skip(f"transformers/torch mismatch building Qwen3-Next: {e}")

    # The layer binds the kernel per instance; confirm Triton actually serves it, since a silent fallback would make the test vacuous.
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


def test_this_blackwell_host_is_not_suspect():
    """This Blackwell host runs Triton 3.5.1, inside the affected range, so only the architecture check keeps it off the
    Hopper path; guards against slowing every Blackwell user."""
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


def test_optout_layout_probe_is_not_fooled_by_our_own_patch(monkeypatch):
    """_transformers_uses_availability_probe must key on the kernel-hub decorator, since
    is_flash_linear_attention_available stays defined after 47630 and _patch_is_available re-adds it."""
    import transformers.utils.import_utils as iu

    from unsloth_zoo.temporary_patches import fla_vendor as fv

    hub = pytest.importorskip("transformers.integrations.hub_kernels")

    # Old layout: no kernel-hub fallback decorator -> the probe still steers things.
    monkeypatch.delattr(hub, "use_kernel_func_from_hub_with_fallback", raising=False)
    assert fv._transformers_uses_availability_probe() is True

    # New layout: the kernel-hub decorator exists so the probe steers nothing, even though the old name stays defined on
    # import_utils and _patch_is_available re-adds it.
    monkeypatch.setattr(
        hub, "use_kernel_func_from_hub_with_fallback", lambda *a, **k: (lambda f: f),
        raising=False,
    )
    monkeypatch.setattr(
        iu, "is_flash_linear_attention_available", lambda: True, raising=False,
    )
    assert fv._transformers_uses_availability_probe() is False


def test_optout_falls_through_to_protection_on_the_new_layout(monkeypatch, fake_gated_delta_modeling):
    """On post-47630 Transformers the opt-out cannot force pure torch, so it must fall through rather than return early,
    or an unpatched BK=64 install would serve the backward."""
    from unsloth_zoo.temporary_patches import fla_vendor as fv

    monkeypatch.setenv("UNSLOTH_DISABLE_HOPPER_FLA_BWD", "1")
    monkeypatch.setattr(fv, "_hopper_dqkwg_suspect_here", lambda: True)
    monkeypatch.setattr(fv, "_FLA_DISABLED_REASON", None)
    monkeypatch.setattr(fv, "_transformers_uses_availability_probe", lambda: False)
    monkeypatch.setattr(fv, "_patch_is_available", lambda probe=None: True)
    monkeypatch.setattr(fv, "_repair_already_imported_modeling", lambda **kw: None)
    monkeypatch.setattr(fv, "_should_defer_to_installed_fla", lambda: False)
    monkeypatch.setattr(fv, "_torch_triton_cuda_supported", lambda: True)
    # An earlier test may already have injected the vendored tree, short-circuiting injection; force the
    # not-yet-injected state so the fall-through is what is tested.
    monkeypatch.setattr(fv, "_vendored_already_injected", lambda: False)

    warned = []
    monkeypatch.setattr(fv, "_warn_hopper_optout_degraded", lambda: warned.append(True))

    reached = []
    monkeypatch.setattr(
        fv, "_inject_vendored_fla", lambda: (reached.append(True), (True, False))[1],
    )

    fv.patch_vendor_fla()

    assert warned, "the user must be told the opt-out could not force pure torch"
    assert reached, (
        "the opt-out must fall through to the normal protection path, not return "
        "and leave the decorator resolving an unpatched fla"
    )
    # fla is not actually disabled on this path, so the loader must not be told it was.
    assert fv.fla_unavailable_reason() is None


def test_optout_does_not_patch_the_kernel_on_the_old_layout(monkeypatch, fake_gated_delta_modeling):
    """Where the availability probe steers selection (pre-47630), the opt-out reaches pure torch and must not touch the installed kernel or warn."""
    from unsloth_zoo.temporary_patches import fla_vendor as fv

    monkeypatch.setenv("UNSLOTH_DISABLE_HOPPER_FLA_BWD", "1")
    monkeypatch.setattr(fv, "_hopper_dqkwg_suspect_here", lambda: True)
    monkeypatch.setattr(fv, "_FLA_DISABLED_REASON", None)
    monkeypatch.setattr(fv, "_transformers_uses_availability_probe", lambda: True)
    monkeypatch.setattr(fv, "_patch_is_available", lambda probe=None: True)

    def fail_patch():
        raise AssertionError("pure torch is reachable; do not touch the kernel")

    def fail_warn(ok):
        raise AssertionError("nothing degraded; do not warn")

    monkeypatch.setattr(fv, "_patch_installed_fla_dqkwg", fail_patch)
    monkeypatch.setattr(fv, "_warn_hopper_optout_degraded", fail_warn)

    fv.patch_vendor_fla()
    for mod in fake_gated_delta_modeling.values():
        assert mod.chunk_gated_delta_rule is None


def test_degraded_optout_outranks_disable_vendored(monkeypatch, fake_gated_delta_modeling):
    """UNSLOTH_DISABLE_VENDORED_FLA is only a source preference; on post-47630 layouts the Hopper opt-out must win, or
    the decorator would resolve an unpatched BK=64 fla."""
    from unsloth_zoo.temporary_patches import fla_vendor as fv

    monkeypatch.setenv("UNSLOTH_DISABLE_HOPPER_FLA_BWD", "1")
    monkeypatch.setenv("UNSLOTH_DISABLE_VENDORED_FLA", "1")
    monkeypatch.setattr(fv, "_hopper_dqkwg_suspect_here", lambda: True)
    monkeypatch.setattr(fv, "_transformers_uses_availability_probe", lambda: False)
    monkeypatch.setattr(fv, "_warn_hopper_optout_degraded", lambda: None)
    monkeypatch.setattr(fv, "_patch_is_available", lambda probe=None: True)
    monkeypatch.setattr(fv, "_repair_already_imported_modeling", lambda **kw: None)
    monkeypatch.setattr(fv, "_vendored_already_injected", lambda: False)
    monkeypatch.setattr(fv, "_should_defer_to_installed_fla", lambda: False)
    monkeypatch.setattr(fv, "_torch_triton_cuda_supported", lambda: True)

    reached = []
    monkeypatch.setattr(
        fv, "_inject_vendored_fla", lambda: (reached.append(True), (True, False))[1],
    )

    fv.patch_vendor_fla()
    assert reached, "disable-vendored must not short-circuit the degraded opt-out"


def test_disable_vendored_still_returns_without_the_optout(monkeypatch):
    """With only UNSLOTH_DISABLE_VENDORED_FLA set, the patch still returns early and leaves a user's own fla exactly as found."""
    from unsloth_zoo.temporary_patches import fla_vendor as fv

    monkeypatch.setenv("UNSLOTH_DISABLE_VENDORED_FLA", "1")
    monkeypatch.delenv("UNSLOTH_DISABLE_HOPPER_FLA_BWD", raising=False)

    def fail_inject():
        raise AssertionError("disable-vendored must still prevent injection")

    monkeypatch.setattr(fv, "_inject_vendored_fla", fail_inject)
    monkeypatch.setattr(fv, "_vendored_already_injected", lambda: False)
    fv.patch_vendor_fla()


def test_degraded_optout_warning_does_not_promise_pure_torch(monkeypatch, caplog):
    """Upgrading Triton clears the miscompile and the opt-out block then skips, so the warning must not suggest the opt-out as a route to pure PyTorch."""
    import logging

    from unsloth_zoo.temporary_patches import fla_vendor as fv

    with caplog.at_level(logging.WARNING):
        fv._warn_hopper_optout_degraded()
    text = caplog.text
    assert "triton>=3.7.1" in text, "the remediation should still be offered"
    assert "neither route gives you the pure-PyTorch path" in text, (
        "the warning must not claim a Triton upgrade reaches the pure-PyTorch path"
    )


@pytest.mark.skipif(not _cuda_available(), reason="needs CUDA")
def test_installed_patch_leaves_non_hopper_tensors_alone(monkeypatch):
    """The installed-fla wrapper applies process-wide when any GPU is Hopper, but fla 640 is Hopper-only; tensors on
    Ada, Ampere or Blackwell cards keep normal tiling, or K=33..64 pays a narrowed BK and BV for nothing."""
    import torch

    chunk_o = _load_vendored_fla()
    from unsloth_zoo.temporary_patches import fla_vendor as fv

    seen = {}

    def fake_original(**kwargs):
        seen["small_tile"] = fv._installed_fla_forcing_small_tile()
        return None

    monkeypatch.setattr(chunk_o, "chunk_bwd_dqkwg", fake_original)
    monkeypatch.setattr(fv, "_device_index_is_hopper", lambda index: False)
    assert fv._patch_installed_fla_dqkwg() is True

    k = torch.zeros(1, 8, 1, 64, device="cuda", dtype=torch.bfloat16)
    g = torch.zeros(1, 8, 1, device="cuda", dtype=torch.float32)
    chunk_o.chunk_bwd_dqkwg(q=k, k=k, v=k, do=k, h=None, dh=None, g=g)
    assert seen["small_tile"] is False, (
        "a non-Hopper tensor must not be forced onto the narrow tile"
    )

    monkeypatch.setattr(fv, "_device_index_is_hopper", lambda index: True)
    chunk_o.chunk_bwd_dqkwg(q=k, k=k, v=k, do=k, h=None, dh=None, g=g)
    assert seen["small_tile"] is True, "a Hopper tensor at K=64 must take the override"


@pytest.mark.skipif(not _cuda_available(), reason="needs CUDA")
def test_vendored_guard_uses_the_tensor_device_not_device_zero(monkeypatch):
    """The vendored guard must ask the tensor's device, not the import-time global frozen from device 0, which is wrong
    in both directions on mixed hosts; only unknown falls back to it."""
    chunk_o = _load_vendored_fla()
    if not (chunk_o.TRITON_ABOVE_3_4_0 and not chunk_o.TRITON_ABOVE_3_7_1):
        pytest.skip("host Triton is outside the affected [3.4.0, 3.7.1) range")

    # Device 0 reports Hopper but this call's tensor is not on Hopper, so the normal tile must survive.
    monkeypatch.setattr(chunk_o, "IS_NVIDIA_HOPPER", True)
    monkeypatch.setattr(chunk_o, "_is_hopper_tensor", lambda x: False)
    assert _record_bk(chunk_o, K=64, V=128, hopper=True, monkeypatch=monkeypatch)["BK"] == 64

    monkeypatch.setattr(chunk_o, "_is_hopper_tensor", lambda x: True)
    assert _record_bk(chunk_o, K=64, V=128, hopper=False, monkeypatch=monkeypatch)["BK"] == 32

    # Undeterminable device: fall back to the global rather than fail open.
    assert _record_bk(
        chunk_o, K=64, V=128, hopper=True, monkeypatch=monkeypatch, tensor_hopper=None,
    )["BK"] == 32
    assert _record_bk(
        chunk_o, K=64, V=128, hopper=False, monkeypatch=monkeypatch, tensor_hopper=None,
    )["BK"] == 64


def test_hopper_probes_report_unknown_rather_than_no(monkeypatch):
    """Both Hopper probes must distinguish not Hopper from could not tell, so a probe failure never silently drops the workaround."""
    from unsloth_zoo.temporary_patches import fla_vendor as fv

    chunk_o = _load_vendored_fla()

    class _Boom:
        is_cuda = True

        @property
        def device(self):
            raise RuntimeError("cannot inspect")

    assert chunk_o._is_hopper_tensor(_Boom()) is None
    assert chunk_o._is_hopper_tensor(None) is None
    # fla_vendor's variant resolves unknown to True, since its wrapper is only installed when some visible GPU is already known to be Hopper.
    assert fv._tensor_on_hopper(_Boom()) is True
    monkeypatch.setattr(fv, "_device_index_is_hopper", lambda index: None)

    class _K:
        is_cuda = True
        device = types.SimpleNamespace(index=0)

    assert fv._tensor_on_hopper(_K()) is True
