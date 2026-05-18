# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Regression guards for upstream-pinned symbols in the MLX / Apple-Silicon /
accelerator-dispatch lanes of unsloth_zoo.

Each test cites the zoo commit that introduced or repaired the symbol it
covers, so a future refactor that renames or silently drops the symbol
fails loudly here. Tests are designed to run on Linux+CUDA via the
``tests/mlx_simulation`` shim and on Apple Silicon natively; CUDA-only
APIs are not exercised directly so the suite is CPU-runnable in CI.
"""

from __future__ import annotations

import sys
import types
from unittest import mock

import pytest
import torch


# ---------------------------------------------------------------------------
# 1. device_type.device_synchronize / device_empty_cache / device_is_bf16_supported
#    must tolerate a partial torch.xpu build that exposes is_available() but
#    lacks the specific call (synchronize / empty_cache / is_bf16_supported).
#
#    Covers commits:
#      - 35dc451 Guard XPU empty_cache call against partial torch.xpu builds
#      - e08c1df Guard XPU synchronize call against partial torch.xpu builds
#      - 2564f39 Route GGUF merge cache flushes and MoE expert merges
#                through active backend (introduced device_empty_cache)
#      - d631837 Route VLM GGUF mmproj bf16 check through active backend
#                (introduced device_is_bf16_supported)
#
#    The existing test_backend_device_helpers.py covers the happy path; this
#    test pins the PARTIAL-BUILD case where torch.xpu.is_available is True
#    but the specific symbol is missing.
# ---------------------------------------------------------------------------

def test_xpu_partial_build_all_three_helpers_silent_no_op():
    """All three device_type helpers must no-op (not AttributeError) on a
    torch.xpu module that lacks synchronize / empty_cache / is_bf16_supported.
    The hasattr-then-call pattern is the exact regression net for the
    e08c1df / 35dc451 / d631837 partial-build crashes seen in the GGUF
    merge and VLM mmproj export paths.
    """
    from unsloth_zoo import device_type as dt

    class PartialXpu:
        """A torch.xpu that knows is_available but nothing else.

        Reflects the upstream IPEX dev build where torch.xpu.is_available is
        True but synchronize / empty_cache / is_bf16_supported are not yet
        wired in. Pre-fix, this raised AttributeError mid-GGUF-export.
        """
        def is_available(self):
            return True

    fake_cuda = mock.MagicMock()
    fake_cuda.is_available.return_value = False

    with mock.patch.object(dt, "DEVICE_TYPE", "xpu"), \
         mock.patch.object(torch, "cuda", fake_cuda), \
         mock.patch.object(torch, "xpu", PartialXpu(), create=True):
        # None of these may raise. The whole regression class is "raises
        # AttributeError because the partial xpu build is missing one of
        # the three call names".
        dt.device_synchronize()
        dt.device_empty_cache()
        assert dt.device_is_bf16_supported() is False


# ---------------------------------------------------------------------------
# 2. saving_utils._active_merge_device() must take NO positional args and
#    cascade cuda -> xpu -> mps -> cpu.
#
#    Covers commit:
#      - fd58aa1 saving_utils: route LoRA merge through accelerator-family probe
#      - 70b93ad fix(mlx): migrate deprecated mx.metal memory APIs + restore
#                device-agnostic LoRA merge
#
#    The pre-fix signature was _active_merge_device(W) which (a) silently
#    dropped MPS, (b) propagated W.device.index across families. This
#    pin asserts the no-arg shape AND the MPS-wins-when-only-mps branch
#    which the previous DEVICE_TYPE_TORCH-only routing dropped.
# ---------------------------------------------------------------------------

def test_active_merge_device_mps_branch_pinned():
    """_active_merge_device() returns "mps" on Apple Silicon (no cuda/xpu).
    This is the exact regression that broke the MLX backend's on-host LoRA
    merge when the helper still routed through DEVICE_TYPE_TORCH.
    """
    from unsloth_zoo.saving_utils import _active_merge_device

    _active_merge_device.cache_clear()
    try:
        # No required positional args. Pre-fix took W; signature change
        # alone would crash every callsite if reverted.
        import inspect
        sig = inspect.signature(_active_merge_device)
        required = [
            p for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        assert required == [], (
            "_active_merge_device() must take no required args; the "
            "pre-fix W-arg signature silently propagated device.index "
            "across accelerator families."
        )

        # Spoof: only MPS available. The cuda-only cascade pre-fix dropped
        # this branch entirely; this assertion is the canary.
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            xpu_ctx = (
                mock.patch.object(torch.xpu, "is_available", return_value=False)
                if hasattr(torch, "xpu") else _NullCtx()
            )
            mps_stub = types.SimpleNamespace(is_available=lambda: True)
            mps_ctx = (
                mock.patch.object(torch.backends.mps, "is_available", return_value=True)
                if hasattr(torch.backends, "mps")
                else mock.patch.object(torch.backends, "mps", mps_stub, create=True)
            )
            with xpu_ctx, mps_ctx:
                _active_merge_device.cache_clear()
                assert _active_merge_device() == "mps"
    finally:
        _active_merge_device.cache_clear()


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


# ---------------------------------------------------------------------------
# 3. MoE-expert _active_merge_device() callsites in saving_utils.py.
#
#    Covers commit:
#      - 2564f39 (introduced)
#      - fd58aa1 (refactored to no-arg helper)
#
#    Pre-fix the five MoE expert helpers (_merge_moe_gate_expert,
#    _merge_moe_up_expert, _merge_moe_down_proj_expert,
#    _merge_moe_fused_gate_up_expert, _merge_moe_fused_down_proj_expert)
#    fell back to CPU on XPU due to hardcoded .to("cuda", ...). This pin
#    asserts those callsites still go through the helper.
# ---------------------------------------------------------------------------

def test_moe_expert_merges_call_active_merge_device():
    """The five MoE-expert merge helpers must route their .to(...) calls
    through _active_merge_device(). A regression to a hardcoded "cuda" or
    DEVICE_TYPE_TORCH inside any one of them silently drops MPS/XPU
    placement and was the exact 2564f39 bug class.

    After unsloth-zoo#647 the gate / up wrappers delegate to a unified
    helper ``_merge_moe_gate_or_up_expert``; the check follows that
    delegation by inspecting the union of each entry-point's source and
    the source of any sibling ``_merge_moe_*`` helper it explicitly
    forwards to.
    """
    import inspect
    import re
    import unsloth_zoo.saving_utils as su

    targets = [
        "_merge_moe_gate_expert",
        "_merge_moe_up_expert",
        "_merge_moe_down_proj_expert",
        "_merge_moe_fused_gate_up_expert",
        "_merge_moe_fused_down_proj_expert",
    ]
    _helper_call_re = re.compile(r"\b(_merge_moe_[A-Za-z0-9_]+)\(")

    def _effective_source(name: str, seen: set) -> str:
        """Return the entry-point's source plus the source of any
        sibling ``_merge_moe_*`` helper it explicitly forwards to.
        One-level follow is enough: zoo never chains wrapper -> wrapper
        -> helper, and the implementations all live in saving_utils."""
        if name in seen:
            return ""
        seen.add(name)
        fn = getattr(su, name, None)
        if fn is None:
            return ""
        src = inspect.getsource(fn)
        callees = set(_helper_call_re.findall(src)) - {name}
        for callee in callees:
            src += "\n" + _effective_source(callee, seen)
        return src

    for name in targets:
        fn = getattr(su, name, None)
        assert fn is not None, (
            f"{name} missing; the MoE-expert merge dispatch surface "
            "shrank without notice — see commit 2564f39."
        )
        src = _effective_source(name, set())
        assert "_active_merge_device(" in src, (
            f"{name} (and any sibling _merge_moe_* it delegates to) no "
            "longer routes through _active_merge_device(). That regresses "
            "2564f39 + fd58aa1: hardcoded 'cuda' breaks Intel XPU and "
            "Apple MPS LoRA merge."
        )
        assert '.to("cuda"' not in src and ".to('cuda'" not in src, (
            f"{name} (or the helper it delegates to) hardcodes "
            ".to('cuda', ...) again — same regression class as commit "
            "2564f39."
        )


# ---------------------------------------------------------------------------
# 4. mx.metal memory APIs migrated to the modern non-namespaced form.
#
#    Covers commit:
#      - 70b93ad fix(mlx): migrate deprecated mx.metal memory APIs +
#                restore device-agnostic LoRA merge
#
#    The deprecated form (mx.metal.set_memory_limit / .set_cache_limit)
#    prints a warning every training run; the modern form is
#    mx.set_memory_limit / mx.set_cache_limit / mx.set_wired_limit.
#    The MLX shim exposes both, so this test pins the trainer source.
# ---------------------------------------------------------------------------

def test_mlx_trainer_uses_modern_memory_apis_only():
    """unsloth_zoo.mlx.trainer must call the non-namespaced memory APIs
    (mx.set_memory_limit, mx.set_cache_limit, mx.set_wired_limit). The
    namespaced mx.metal.set_* forms are deprecated upstream and reverting
    to them resurrects the per-run deprecation warning that 70b93ad fixed.
    """
    import importlib.util
    import pathlib

    pkg_root = pathlib.Path(
        importlib.util.find_spec("unsloth_zoo").submodule_search_locations[0]
    )
    # The MLX path was promoted from a flat module (mlx_trainer.py) to a
    # subpackage (mlx/trainer.py) in e6d8f7f. Accept either layout so the
    # test survives the rename.
    candidates = [pkg_root / "mlx" / "trainer.py", pkg_root / "mlx_trainer.py"]
    mlx_trainer_path = next((c for c in candidates if c.is_file()), None)
    assert mlx_trainer_path is not None, (
        f"Neither {candidates[0]} nor {candidates[1]} exists; the MLX "
        f"trainer module was relocated again. Update this test's path "
        f"candidates."
    )
    src = mlx_trainer_path.read_text()

    # The deprecated forms must NOT appear.
    assert "mx.metal.set_memory_limit" not in src, (
        "Deprecated mx.metal.set_memory_limit call resurfaced; "
        "regresses commit 70b93ad."
    )
    assert "mx.metal.set_cache_limit" not in src, (
        "Deprecated mx.metal.set_cache_limit call resurfaced; "
        "regresses commit 70b93ad."
    )

    # The modern forms must appear.
    for modern in ("mx.set_memory_limit", "mx.set_cache_limit", "mx.set_wired_limit"):
        assert modern in src, f"Expected modern API {modern} missing from {mlx_trainer_path.name}"


# ---------------------------------------------------------------------------
# 5. Apple-Silicon stub injection on __init__ (3 sub-bugs from 2053539).
#
#    Covers commit:
#      - 2053539 fix(mlx): repair stub injection on Apple Silicon (3 sub-bugs)
#
#    Sub-bugs:
#      a. Inverted gate: stubs were inside `if not _SKIP_GPU_INIT:`. Fix
#         moved them under `if _SKIP_GPU_INIT:`.
#      b. Wrong function name: install_*_stub vs the real inject_into_sys_modules.
#      c. _Noop.__call__ silently returned None — fix raises NotImplementedError.
# ---------------------------------------------------------------------------

def test_apple_silicon_stub_injection_entrypoints_pinned():
    """Sub-bugs (a) and (b) of commit 2053539. The init module must gate
    stub injection on `if _SKIP_GPU_INIT:` (NOT the negated form) and call
    inject_into_sys_modules (NOT install_*_stub).
    """
    import importlib.util
    import pathlib

    init_path = pathlib.Path(
        importlib.util.find_spec("unsloth_zoo").submodule_search_locations[0]
    ) / "__init__.py"
    src = init_path.read_text()

    # Sub-bug (b): the real entry point is inject_into_sys_modules.
    assert "inject_into_sys_modules" in src, (
        "Stub injection entry point inject_into_sys_modules vanished from "
        "unsloth_zoo/__init__.py — regresses commit 2053539 sub-bug (b)."
    )
    # Pre-fix names that must NOT come back.
    assert "install_triton_stub" not in src
    assert "install_bitsandbytes_stub" not in src

    # Sub-bug (a): the gate must be positive `if _SKIP_GPU_INIT:` not
    # `if not _SKIP_GPU_INIT:` around the injection block. We look for the
    # exact positive line.
    assert "if _SKIP_GPU_INIT:" in src, (
        "Apple-Silicon stub-injection gate flipped — regresses commit "
        "2053539 sub-bug (a)."
    )


def test_stub_noop_call_raises_not_returns_none():
    """Sub-bug (c) of 2053539. _Noop.__call__ must raise NotImplementedError
    so a stray `bnb.functional.quantize_4bit(weight, ...)` on Apple Silicon
    crashes loudly rather than silently producing None that corrupts the
    downstream tensor pipeline. __bool__ and hasattr probes must still work.
    """
    from unsloth_zoo.stubs import triton_stub, bitsandbytes_stub

    for mod in (triton_stub, bitsandbytes_stub):
        noop = mod._Noop("test.symbol")
        with pytest.raises(NotImplementedError, match="test.symbol"):
            noop()
        # Optional-feature probes still work:
        assert bool(noop) is False  # __bool__ pass-through
        sub = noop.some_attr        # attribute chaining returns another _Noop
        assert sub is not noop
        with pytest.raises(NotImplementedError, match="test.symbol.some_attr"):
            sub()


# ---------------------------------------------------------------------------
# 6. mlx_loader rejects full_finetuning against a pre-quantized repo.
#
#    Covers commit:
#      - 7d2bb95 fix(mlx): reject full_finetuning against pre-quantized
#                repos loudly
#
#    Without this guard, the CCE backward returns mx.zeros for quantized
#    weight grads, so the user "trains" but most of the model never
#    updates. The detection helper is _get_existing_mlx_quantization.
# ---------------------------------------------------------------------------

def test_get_existing_mlx_quantization_detects_both_keys():
    """The detection helper must recognise BOTH the 'quantization' (MLX
    native) and 'quantization_config' (HF style) keys. A regression that
    only checks one silently re-enables the full_finetuning-on-quantized
    foot-gun that 7d2bb95 closed.
    """
    # Import the helper without triggering the heavy mlx_loader import
    # chain on the GPU-free harness. We pull the function directly.
    # Layout was promoted from mlx_loader.py (flat) to mlx/loader.py
    # (subpackage) in e6d8f7f. Try both so the test survives the rename.
    import importlib.util
    import pathlib
    pkg_loc = pathlib.Path(
        importlib.util.find_spec("unsloth_zoo").submodule_search_locations[0]
    )
    candidates = [pkg_loc / "mlx" / "loader.py", pkg_loc / "mlx_loader.py"]
    loader_path = next((c for c in candidates if c.is_file()), None)
    assert loader_path is not None, (
        f"Neither {candidates[0]} nor {candidates[1]} exists; the MLX "
        f"loader module was relocated again. Update this test's path "
        f"candidates."
    )
    src = loader_path.read_text()

    # The function must check BOTH key names; otherwise repos saved by
    # mlx-lm (key "quantization") OR by HF transformers ("quantization_config")
    # slip through the guard.
    assert "config_data.get(\"quantization\"" in src, (
        "_get_existing_mlx_quantization no longer checks 'quantization' "
        "key — regresses commit 7d2bb95."
    )
    assert "config_data.get(\"quantization_config\"" in src, (
        "_get_existing_mlx_quantization no longer checks "
        "'quantization_config' key — regresses commit 7d2bb95."
    )


# ---------------------------------------------------------------------------
# 7. target_modules='all-linear' must collect EVERY nn.Linear name.
#
#    Covers commit:
#      - 7f8b0ca fix(mlx): make target_modules='all-linear' actually mean
#                every nn.Linear
#
#    Pre-fix, "all-linear" was silently rewritten to None and collapsed to
#    the canonical 7-name list. For Qwen3.5 that dropped the GatedDelta
#    in_proj_* and out_proj from LoRA targeting entirely.
# ---------------------------------------------------------------------------

def test_collect_all_linear_target_names_finds_qkv_and_moe():
    """_collect_all_linear_target_names must discover fused-QKV names
    (qkv_proj), GatedDelta projections (in_proj_a, in_proj_b, in_proj_qkv,
    in_proj_z, out_proj), vision tower fused linears, and MoE routers /
    experts — not just the canonical 7. Walks a fake model whose
    named_modules emits the names we care about so we don't need real MLX.
    """
    pytest.importorskip("mlx")  # the helper imports mlx.nn for isinstance
    from unsloth_zoo.mlx_loader import _collect_all_linear_target_names
    import mlx.nn as nn

    class FakeQwen3p5:
        """Minimal model whose named_modules() exposes the leaves that
        triggered the pre-fix silent collapse. Real mlx.nn.Linear types
        are required because the helper's isinstance check uses them.
        """
        def named_modules(self):
            yield ("model.layers.0.self_attn.q_proj", nn.Linear(4, 4))
            yield ("model.layers.0.self_attn.k_proj", nn.Linear(4, 4))
            yield ("model.layers.0.self_attn.v_proj", nn.Linear(4, 4))
            yield ("model.layers.0.self_attn.o_proj", nn.Linear(4, 4))
            yield ("model.layers.0.mlp.gate_proj", nn.Linear(4, 4))
            yield ("model.layers.0.mlp.up_proj",   nn.Linear(4, 4))
            yield ("model.layers.0.mlp.down_proj", nn.Linear(4, 4))
            # GatedDelta projections — the exact 7f8b0ca regression class.
            yield ("model.layers.0.gated_delta.in_proj_qkv", nn.Linear(4, 4))
            yield ("model.layers.0.gated_delta.in_proj_z",   nn.Linear(4, 4))
            yield ("model.layers.0.gated_delta.out_proj",    nn.Linear(4, 4))
            # MoE router + expert — fused QKV — vision tower (numeric leaves
            # are skipped, the *suffix* name is what gets returned).
            yield ("model.layers.0.moe.router", nn.Linear(4, 4))
            yield ("model.layers.0.moe.experts.0.w1", nn.Linear(4, 4))
            yield ("vision_tower.layers.0.attn.qkv", nn.Linear(4, 4))

    names = set(_collect_all_linear_target_names(FakeQwen3p5()))
    canonical = {"q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"}
    # Canonical 7 still resolve.
    assert canonical <= names
    # Plus the extras the pre-fix collapse dropped.
    extras = {"in_proj_qkv", "in_proj_z", "out_proj",
              "router", "w1", "qkv"}
    missing = extras - names
    assert not missing, (
        f"all-linear missed {sorted(missing)} — regresses commit 7f8b0ca; "
        "the silent-collapse-to-canonical-7 bug would skip these layers."
    )


# ---------------------------------------------------------------------------
# 8. patch_gated_delta routes training (state=None) through the efficient
#    custom-VJP path, not the kernel.
#
#    Covers commit:
#      - 46866ce fix(mlx): correct GatedDeltaNet VJP mask handling +
#                actually run it
#
#    Pre-fix patched_gated_delta_update fell through to gated_delta_kernel
#    on Metal (the default use_kernel=True branch), making the custom VJP
#    dead code. The fix unconditionally routes training calls
#    (state is None on entry) through gated_delta_ops_efficient.
# ---------------------------------------------------------------------------

def test_patch_gated_delta_routes_training_through_efficient_path():
    """Pin the routing predicate in patch_gated_delta. The patched
    function MUST call gated_delta_ops_efficient when state is None
    (training entry), even if use_kernel=True and mlx says metal is
    available. Pre-fix the kernel branch shadowed the custom VJP.
    """
    import importlib.util
    import pathlib
    pkg_loc = importlib.util.find_spec("unsloth_zoo").submodule_search_locations[0]
    src = (pathlib.Path(pkg_loc) / "gated_delta_vjp.py").read_text()

    # The training-call routing line is the regression-net.
    # The fix added `is_training_call = state is None` and then the
    # unconditional `if is_training_call: return gated_delta_ops_efficient(...)`
    # branch BEFORE the kernel branch. Both must be present.
    assert "is_training_call" in src, (
        "patch_gated_delta dropped the is_training_call gate; "
        "regresses commit 46866ce — custom VJP becomes dead code under "
        "use_kernel=True."
    )
    assert "gated_delta_ops_efficient" in src
    # And the training branch must come before the kernel fallthrough.
    idx_eff = src.find("if is_training_call:")
    idx_kernel = src.find("gated_delta_kernel(")
    assert idx_eff != -1 and idx_kernel != -1
    assert idx_eff < idx_kernel, (
        "The training-call branch must precede the gated_delta_kernel "
        "fallthrough so the custom VJP actually runs (commit 46866ce)."
    )
