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

"""Tests for the vendored flash-linear-attention (fla) gated-delta kernels.

Two parts:
  * CPU-safe checks that the pruned vendored tree is present, correctly narrowed
    and compiles (no GPU / torch needed).
  * An import-hygiene check that runs the injection in a fresh interpreter (so it
    never contaminates the global sys.modules of the test session) and asserts
    availability + that no heavy / unwanted modules leaked. Skipped when the
    Triton kernels cannot run (no CUDA / torch<2.7 / triton<3.3).
"""

import os
import sys
import subprocess
import pathlib
import textwrap

import pytest

# Importing unsloth_zoo on a GPU host runs its full init, which asserts Unsloth
# is present. Set the flag defensively so the test is self-contained.
os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

# In a zoo-only environment (the separate `unsloth` package not installed),
# unsloth_zoo's init raises on find_spec("unsloth") before any submodule loads,
# which the flag above does not satisfy. Skip the module cleanly there instead of
# erroring at collection; the repo CI installs unsloth so these still run.
try:
    import unsloth_zoo  # noqa: F401
except ImportError as _e:
    pytest.skip(f"unsloth_zoo unavailable: {_e}", allow_module_level=True)

ZOO_ROOT = pathlib.Path(__file__).resolve().parents[1]
VENDORED = ZOO_ROOT / "unsloth_zoo" / "_vendored" / "fla"


def _injection_supported() -> bool:
    # Mirror the production support gate exactly (Python>=3.10, torch/triton
    # minimums, CUDA, and the Hopper/Triton range that needs the pruned TileLang
    # backend). A looser check would run the subprocess tests on hosts where the
    # patch intentionally skips injection, so they would fail instead of skip.
    try:
        from unsloth_zoo.temporary_patches.fla_vendor import (
            _vendored_injection_supported,
        )
        return bool(_vendored_injection_supported())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# CPU-safe structural / symbol-resolution checks
# ---------------------------------------------------------------------------
def test_vendored_tree_layout():
    assert VENDORED.is_dir(), VENDORED
    assert (VENDORED / "LICENSE").is_file()
    assert (VENDORED / "MANIFEST").is_file()

    top = (VENDORED / "__init__.py").read_text()
    assert '__version__ = "0.5.1"' in top
    # narrowed: the eager layers/models imports must be gone
    assert "import fla.layers" not in top
    assert "_import_optional_public_module" not in top

    modules_init = (VENDORED / "modules" / "__init__.py").read_text()
    assert "FusedRMSNormGated" in modules_init

    gdr_init = (VENDORED / "ops" / "gated_delta_rule" / "__init__.py").read_text()
    assert "chunk_gated_delta_rule" in gdr_init
    assert "fused_recurrent_gated_delta_rule" in gdr_init
    # naive reference impl (the only einops dep) is dropped: no import of it in
    # the code (the "Modified by Unsloth" comment may still mention it by name).
    gdr_code = "\n".join(
        ln for ln in gdr_init.splitlines() if not ln.lstrip().startswith("#")
    )
    assert "naive" not in gdr_code


def test_pruned_and_kept_files():
    # naive.py (only einops dependency) dropped
    assert not (VENDORED / "ops" / "gated_delta_rule" / "naive.py").exists()

    tilelang = VENDORED / "ops" / "common" / "backends" / "tilelang"
    assert (tilelang / "__init__.py").is_file()  # guarded wrapper kept
    for dropped in ("chunk_bwd.py", "parallel_attn_fwd.py", "parallel_attn_bwd.py"):
        assert not (tilelang / dropped).exists(), dropped

    # cp kept as-is (imports safely, never executes single-GPU)
    assert (VENDORED / "ops" / "cp" / "chunk_delta_h.py").is_file()


def test_all_vendored_python_compiles():
    import py_compile
    py_files = sorted(VENDORED.rglob("*.py"))
    assert len(py_files) == 42, f"expected 42 vendored .py files, got {len(py_files)}"
    for p in py_files:
        py_compile.compile(str(p), doraise=True)


def test_backported_blackwell_hopper_fixes_present():
    """Guard the three post-v0.5.1 correctness backports against a silent drop on
    re-vendoring (PR #953 Blackwell fwd-h race, #1000 Blackwell bwd hang,
    #983 Hopper Triton>=3.7.1 guard). Source-level so it needs no GPU."""
    cdh = (VENDORED / "ops" / "common" / "chunk_delta_h.py").read_text()
    assert "GATED_DELTA_RULE_FWD_H_NUM_WARPS = [2] if IS_NVIDIA_BLACKWELL else [2, 4]" in cdh
    assert "for num_warps in GATED_DELTA_RULE_FWD_H_NUM_WARPS" in cdh
    # The bwd sibling kernel is intentionally NOT restricted (upstream #953).
    assert cdh.count("for num_warps in [2, 4]") == 1, "bwd-dhu block should keep [2, 4]"

    wy = (VENDORED / "ops" / "gated_delta_rule" / "wy_fast.py").read_text()
    assert "PREPARE_WY_REPR_BWD_NUM_WARPS = [2] if IS_NVIDIA_BLACKWELL else [2, 4]" in wy
    assert "PREPARE_WY_REPR_BWD_NUM_STAGES = [4] if IS_NVIDIA_BLACKWELL else [2, 3, 4]" in wy
    assert "for num_warps in PREPARE_WY_REPR_BWD_NUM_WARPS" in wy
    # The fwd recompute kernel keeps its wider space.
    assert "for num_warps in [2, 4, 8]" in wy

    co = (VENDORED / "ops" / "common" / "chunk_o.py").read_text()
    assert "IS_NVIDIA_HOPPER and TRITON_ABOVE_3_4_0 and not TRITON_ABOVE_3_7_1" in co

    compat = (VENDORED / "utils" / "_compat.py").read_text()
    assert "TRITON_ABOVE_3_7_1 = " in compat
    utils_init = (VENDORED / "utils" / "__init__.py").read_text()
    assert "TRITON_ABOVE_3_7_1," in utils_init


# ---------------------------------------------------------------------------
# Import hygiene (fresh interpreter)
# ---------------------------------------------------------------------------
_HYGIENE_SUBPROCESS = textwrap.dedent(
    """
    import os, sys
    os.environ["UNSLOTH_IS_PRESENT"] = "1"
    os.environ["UNSLOTH_FORCE_VENDORED_FLA"] = "1"

    from unsloth_zoo.temporary_patches.fla_vendor import (
        patch_vendor_fla, _vendored_fla_dir,
    )
    patch_vendor_fla()

    import transformers.utils.import_utils as iu
    assert iu.is_flash_linear_attention_available() is True, "availability not True"

    fla = sys.modules.get("fla")
    assert fla is not None, "fla not injected"
    vend = os.path.realpath(_vendored_fla_dir())
    assert os.path.realpath(fla.__file__).startswith(vend), fla.__file__
    assert hasattr(fla, "__path__"), "fla has no __path__"

    import pkgutil
    names = {m.name for m in pkgutil.iter_modules(list(fla.__path__))}
    assert {"ops", "modules"} <= names, ("__path__ not walkable", names)

    assert "tilelang" not in sys.modules, "external tilelang leaked"
    assert "fla.models" not in sys.modules, "fla.models leaked"
    assert "fla.layers" not in sys.modules, "fla.layers leaked"

    from fla.modules import FusedRMSNormGated
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule, fused_recurrent_gated_delta_rule,
    )
    for fn in (FusedRMSNormGated, chunk_gated_delta_rule, fused_recurrent_gated_delta_rule):
        assert fn is not None
    print("HYGIENE_OK")
    """
)


@pytest.mark.skipif(
    not _injection_supported(),
    reason="vendored fla kernels need CUDA + torch>=2.7 + triton>=3.3",
)
def test_import_hygiene_subprocess():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ZOO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _HYGIENE_SUBPROCESS],
        env=env,
        capture_output=True,
        text=True,
    )
    assert "HYGIENE_OK" in proc.stdout, f"stdout=\n{proc.stdout}\nstderr=\n{proc.stderr}"
    assert proc.returncode == 0, f"stderr=\n{proc.stderr}"


# ---------------------------------------------------------------------------
# Caller-aware availability probe (uncovered models keep the pure-torch path)
# ---------------------------------------------------------------------------

def test_probe_covers_only_vendor_complete_models():
    from unsloth_zoo.temporary_patches.fla_vendor import _vendored_availability_probe

    def call_as(module_name):
        return eval(
            "probe()",
            {"probe": _vendored_availability_probe, "__name__": module_name},
        )

    # Covered gated-deltanet models get the fast path.
    assert call_as("transformers.models.qwen3_5.modeling_qwen3_5") is True
    assert call_as("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe") is True
    assert call_as("transformers.models.qwen3_next.modeling_qwen3_next") is True
    # olmo_hybrid needs ShortConvolution (not vendored): must answer False so its
    # modeling module falls back to pure torch instead of crashing on import.
    assert call_as("transformers.models.olmo_hybrid.modeling_olmo_hybrid") is False
    # Non-modeling callers see the vendored fla as available.
    assert call_as("unsloth.models.loader") is True
    assert call_as("__main__") is True


_OLMO_SUBPROCESS = textwrap.dedent(
    """
    import os
    os.environ["UNSLOTH_FORCE_VENDORED_FLA"] = "1"
    from unsloth_zoo.temporary_patches.fla_vendor import patch_vendor_fla
    patch_vendor_fla()
    import sys
    assert getattr(sys.modules["fla"], "_UNSLOTH_VENDORED_FLA", False) is True

    # Covered model binds the vendored kernels.
    import transformers.models.qwen3_5.modeling_qwen3_5 as q
    assert q.chunk_gated_delta_rule is not None

    # Uncovered model must import cleanly on its pure-torch fallback.
    import transformers.models.olmo_hybrid.modeling_olmo_hybrid as m
    assert m.ShortConvolution is None
    assert m.chunk_gated_delta_rule is None
    print("OLMO_FALLBACK_OK")
    """
)


@pytest.mark.skipif(
    not _injection_supported(),
    reason="vendored fla kernels need CUDA + torch>=2.7 + triton>=3.3",
)
def test_uncovered_model_imports_on_fallback_subprocess():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ZOO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _OLMO_SUBPROCESS],
        env=env,
        capture_output=True,
        text=True,
    )
    assert "OLMO_FALLBACK_OK" in proc.stdout, f"stdout=\n{proc.stdout}\nstderr=\n{proc.stderr}"
    assert proc.returncode == 0, f"stderr=\n{proc.stderr}"


def test_python_39_skips_injection(monkeypatch):
    # The snapshot uses runtime PEP 604 annotations, which raise on 3.9; the
    # support gate must answer False there instead of import-fail + rollback.
    from unsloth_zoo.temporary_patches import fla_vendor

    monkeypatch.setattr(fla_vendor.sys, "version_info", (3, 9, 19, "final", 0))
    assert fla_vendor._torch_triton_cuda_supported() is False


def test_hopper_bad_triton_range_skips_injection():
    # Upstream chunk_bwd_dqkwg raises on Hopper + triton [3.4.0, 3.7.1) and
    # points at the pruned TileLang backend, so injection must bail there.
    import types

    from unsloth_zoo.temporary_patches.fla_vendor import _hopper_triton_needs_tilelang

    def fake_torch(name, major, count=1):
        cuda = types.SimpleNamespace(
            device_count=lambda c=count: c,
            get_device_name=lambda i=0, n=name: n,
            get_device_capability=lambda i=0, m=major: (m, 0),
        )
        return types.SimpleNamespace(cuda=cuda)

    hopper = fake_torch("NVIDIA H100 80GB HBM3", 9)
    blackwell = fake_torch("NVIDIA B200", 10)
    for ver, want_on_hopper in (
        ("3.3.1", False),
        ("3.4.0", True),
        ("3.6.0", True),
        ("3.7.0", True),
        ("3.7.1", False),
        ("3.8.0", False),
    ):
        tri = types.SimpleNamespace(__version__=ver)
        assert _hopper_triton_needs_tilelang(hopper, tri) is want_on_hopper, ver
        assert _hopper_triton_needs_tilelang(blackwell, tri) is False, ver


def test_hopper_at_nonzero_device_index_trips_guard():
    # On a mixed host the model can run on a nonzero Hopper card while cuda:0 is a
    # different architecture; the guard must scan every visible device, not just 0.
    import types

    from unsloth_zoo.temporary_patches.fla_vendor import _hopper_triton_needs_tilelang

    def mixed_torch(caps, names):
        cuda = types.SimpleNamespace(
            device_count=lambda: len(caps),
            get_device_name=lambda i: names[i],
            get_device_capability=lambda i: caps[i],
        )
        return types.SimpleNamespace(cuda=cuda)

    # cuda:0 Ada (sm89), cuda:1 Hopper (sm90). A device-0 probe would say "safe".
    ada_then_hopper = mixed_torch(
        {0: (8, 9), 1: (9, 0)},
        {0: "NVIDIA RTX 6000 Ada Generation", 1: "NVIDIA H100 80GB HBM3"},
    )
    # cuda:0 Ada, cuda:1 Ada (no Hopper anywhere): fast path stays enabled.
    ada_only = mixed_torch(
        {0: (8, 9), 1: (8, 9)},
        {0: "NVIDIA RTX 6000 Ada Generation", 1: "NVIDIA RTX 6000 Ada Generation"},
    )

    bad = types.SimpleNamespace(__version__="3.6.0")   # in [3.4.0, 3.7.1)
    ok = types.SimpleNamespace(__version__="3.7.1")    # patched Triton

    # A Hopper card at cuda:1 must trip the guard in the bad Triton range.
    assert _hopper_triton_needs_tilelang(ada_then_hopper, bad) is True
    # Same host, patched Triton: no skip.
    assert _hopper_triton_needs_tilelang(ada_then_hopper, ok) is False
    # No Hopper on any index: never skip.
    assert _hopper_triton_needs_tilelang(ada_only, bad) is False


def test_blackwell_import_device_scans_visible_devices():
    # fla.utils freezes IS_NVIDIA_BLACKWELL from the current device at import, so
    # on a mixed host the vendored import must run with a Blackwell device current.
    import types

    from unsloth_zoo.temporary_patches.fla_vendor import _blackwell_import_device

    def fake_torch(caps, current):
        cuda = types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: len(caps),
            current_device=lambda: current,
            get_device_capability=lambda i: caps[i],
        )
        return types.SimpleNamespace(cuda=cuda)

    # cuda:0 Ada current, cuda:1 B200 (sm100): switch the import to index 1.
    assert _blackwell_import_device(fake_torch({0: (8, 9), 1: (10, 0)}, 0)) == 1
    # sm120 consumer Blackwell is also covered.
    assert _blackwell_import_device(fake_torch({0: (8, 9), 1: (12, 0)}, 0)) == 1
    # Already Blackwell-current: no switch needed.
    assert _blackwell_import_device(fake_torch({0: (10, 0), 1: (8, 9)}, 0)) is None
    # No Blackwell anywhere: no switch.
    assert _blackwell_import_device(fake_torch({0: (8, 9), 1: (9, 0)}, 0)) is None


# ---------------------------------------------------------------------------
# Pruned TileLang backend cannot import a broken external tilelang
# ---------------------------------------------------------------------------
_TILELANG_NEUTRALIZED_SUBPROCESS = textwrap.dedent(
    """
    import os, sys, pathlib, tempfile
    os.environ["UNSLOTH_IS_PRESENT"] = "1"
    os.environ["UNSLOTH_FORCE_VENDORED_FLA"] = "1"

    # A broken/incompatible tilelang install: importable finder, but executing its
    # __init__ raises a NON-ImportError (e.g. an ABI/CUDA mismatch). The vendored
    # TileLangBackend.is_available() does `import tilelang` catching only
    # ImportError, so an un-neutralized probe would let this abort the call.
    root = pathlib.Path(tempfile.mkdtemp())
    (root / "tilelang").mkdir()
    (root / "tilelang" / "__init__.py").write_text(
        "raise RuntimeError('broken tilelang ABI')\\n"
    )
    sys.path.insert(0, str(root))

    # Sanity: importing it really raises a non-ImportError.
    try:
        import tilelang
        raise AssertionError("expected broken tilelang to raise RuntimeError")
    except RuntimeError:
        pass
    finally:
        sys.modules.pop("tilelang", None)

    from unsloth_zoo.temporary_patches.fla_vendor import patch_vendor_fla
    patch_vendor_fla()

    import fla.ops.common.backends as cb
    from fla.ops.common.backends.tilelang import TileLangBackend

    # Neutralized: the probe answers False without importing the broken tilelang.
    assert TileLangBackend.is_available() is False, "tilelang probe not neutralized"
    assert "tilelang" not in sys.modules, "broken tilelang got imported"

    # Emulate the dispatch loop's `is_available() and is_enabled()` guard across
    # every registered backend: it must not raise even with a broken tilelang on
    # the path, and the tilelang backend stays unusable.
    for be in cb.common_registry._get_sorted_backends():
        usable = be.is_available() and be.is_enabled()
        assert usable in (True, False)
    assert "tilelang" not in sys.modules, "broken tilelang imported during dispatch probe"
    print("TILELANG_NEUTRALIZED_OK")
    """
)


@pytest.mark.skipif(
    not _injection_supported(),
    reason="vendored fla kernels need CUDA + torch>=2.7 + triton>=3.3",
)
def test_broken_tilelang_does_not_abort_dispatch_subprocess():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ZOO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _TILELANG_NEUTRALIZED_SUBPROCESS],
        env=env,
        capture_output=True,
        text=True,
    )
    assert "TILELANG_NEUTRALIZED_OK" in proc.stdout, f"stdout=\n{proc.stdout}\nstderr=\n{proc.stderr}"
    assert proc.returncode == 0, f"stderr=\n{proc.stderr}"


# ---------------------------------------------------------------------------
# Pruned IntraCard CP backend cannot be re-enabled into the missing module
# ---------------------------------------------------------------------------
_INTRACARD_NEUTRALIZED_SUBPROCESS = textwrap.dedent(
    """
    import os, sys
    os.environ["UNSLOTH_IS_PRESENT"] = "1"
    os.environ["UNSLOTH_FORCE_VENDORED_FLA"] = "1"

    from unsloth_zoo.temporary_patches.fla_vendor import patch_vendor_fla
    patch_vendor_fla()

    import fla.ops.common.backends as cb
    from fla.ops.common.backends.intracard import IntraCardCPBackend

    # Injection forces FLA_INTRACARD_CP=0, but a user can flip it back on after
    # import; dispatch reads is_enabled() from the env per call, so the env force
    # alone would re-route varlen inference into the pruned module.
    os.environ["FLA_INTRACARD_CP"] = "1"

    # The pruned module really is absent from the vendored snapshot.
    import importlib.util
    assert importlib.util.find_spec("fla.ops.common.intracard_cp") is None, \\
        "intracard_cp unexpectedly present"

    # Neutralized: the probe answers False even though FLA_INTRACARD_CP=1 enables it.
    assert IntraCardCPBackend.is_enabled() is True, "env flag should read enabled"
    assert IntraCardCPBackend.is_available() is False, "intracard probe not neutralized"

    # The dispatch loop's `is_available() and is_enabled()` guard must never select
    # the intracard backend, so no call imports the missing module.
    for be in cb.common_registry._get_sorted_backends():
        usable = be.is_available() and be.is_enabled()
        assert usable in (True, False)
        if be is IntraCardCPBackend or getattr(be, "backend_type", None) == "intracard_cp":
            assert usable is False, "pruned intracard backend selected by dispatch"
    print("INTRACARD_NEUTRALIZED_OK")
    """
)


@pytest.mark.skipif(
    not _injection_supported(),
    reason="vendored fla kernels need CUDA + torch>=2.7 + triton>=3.3",
)
def test_reenabled_intracard_stays_unavailable_subprocess():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ZOO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _INTRACARD_NEUTRALIZED_SUBPROCESS],
        env=env,
        capture_output=True,
        text=True,
    )
    assert "INTRACARD_NEUTRALIZED_OK" in proc.stdout, f"stdout=\n{proc.stdout}\nstderr=\n{proc.stderr}"
    assert proc.returncode == 0, f"stderr=\n{proc.stderr}"


def test_failed_injection_restores_backend_env(monkeypatch, tmp_path):
    # A vendored import that fails after FLA_TILELANG / FLA_INTRACARD_CP were
    # forced off must restore whatever the user had, so a shadowed real fla is
    # not left with those backends disabled for the rest of the process.
    from unsloth_zoo.temporary_patches import fla_vendor

    bad = tmp_path / "fla"
    bad.mkdir()
    (bad / "__init__.py").write_text("raise RuntimeError('boom')\n")
    monkeypatch.setattr(fla_vendor, "_vendored_fla_dir", lambda: str(bad))

    # One flag pre-set by the user, one unset: both must be returned as found.
    monkeypatch.setenv("FLA_TILELANG", "1")
    monkeypatch.delenv("FLA_INTRACARD_CP", raising=False)

    injected, replaced_real = fla_vendor._inject_vendored_fla()
    assert injected is False
    assert replaced_real is False
    assert os.environ.get("FLA_TILELANG") == "1"
    assert "FLA_INTRACARD_CP" not in os.environ


# ---------------------------------------------------------------------------
# Force-rebind: replacing an already-loaded real fla under the escape hatch
# ---------------------------------------------------------------------------
_FORCE_REBIND_SUBPROCESS = textwrap.dedent(
    """
    import os, sys, types
    os.environ["UNSLOTH_IS_PRESENT"] = "1"
    os.environ["UNSLOTH_FORCE_VENDORED_FLA"] = "1"

    # Stand in for a real (non-vendored) fla install already cached in
    # sys.modules, plus a gated-delta modeling module imported against it whose
    # kernel globals are bound to non-None, non-vendored callables.
    sys.modules["fla"] = types.ModuleType("fla")

    def _old_chunk(*a, **k):
        raise AssertionError("stale real-fla kernel still bound")
    def _old_recurrent(*a, **k):
        raise AssertionError("stale real-fla kernel still bound")
    class _OldRMS:
        pass

    fake = types.ModuleType("transformers.models.qwen3_5.modeling_qwen3_5")
    fake.chunk_gated_delta_rule = _old_chunk
    fake.fused_recurrent_gated_delta_rule = _old_recurrent
    fake.FusedRMSNormGated = _OldRMS
    sys.modules["transformers.models.qwen3_5.modeling_qwen3_5"] = fake

    from unsloth_zoo.temporary_patches.fla_vendor import patch_vendor_fla
    patch_vendor_fla()

    import fla
    assert getattr(fla, "_UNSLOTH_VENDORED_FLA", False) is True, "vendored not injected"

    from fla.modules import FusedRMSNormGated
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule, fused_recurrent_gated_delta_rule,
    )
    # The force flag must rebind the non-None stale globals to the vendored ones.
    assert fake.chunk_gated_delta_rule is chunk_gated_delta_rule, "chunk not rebound"
    assert fake.fused_recurrent_gated_delta_rule is fused_recurrent_gated_delta_rule
    assert fake.FusedRMSNormGated is FusedRMSNormGated
    assert fake.chunk_gated_delta_rule is not _old_chunk
    print("FORCE_REBIND_OK")
    """
)


@pytest.mark.skipif(
    not _injection_supported(),
    reason="vendored fla kernels need CUDA + torch>=2.7 + triton>=3.3",
)
def test_force_rebinds_already_loaded_real_fla_subprocess():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ZOO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _FORCE_REBIND_SUBPROCESS],
        env=env,
        capture_output=True,
        text=True,
    )
    assert "FORCE_REBIND_OK" in proc.stdout, f"stdout=\n{proc.stdout}\nstderr=\n{proc.stderr}"
    assert proc.returncode == 0, f"stderr=\n{proc.stderr}"
