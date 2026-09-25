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

import ast
import os
import sys
import subprocess
import pathlib
import textwrap

import pytest

# torch is intentionally NOT imported at module level: the CPU-safe structural
# tests (AST / source checks) must still collect on a host without torch. The few
# tensor tests below import it locally, and unsloth_zoo's own init pulls it in.

# Importing unsloth_zoo on a GPU host runs its full init, which asserts Unsloth
# is present. Set the flag defensively so the test is self-contained.
os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

# The skip gate below imports fla_vendor to read its support check. On a CUDA host
# that import would otherwise run fla_vendor's import-time patch_vendor_fla() and
# inject fla into this pytest process, contaminating the in-process tests and the
# rest of the session. Suppress only that autorun; the subprocess tests set their
# own env and call patch_vendor_fla() explicitly, so real injection still happens
# there.
os.environ.setdefault("UNSLOTH_VENDORED_FLA_NO_AUTORUN", "1")

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
    # minimums, CUDA). A looser check would run the subprocess tests on hosts where
    # the patch intentionally skips injection, so they would fail instead of skip.
    try:
        from unsloth_zoo.temporary_patches.fla_vendor import (
            _vendored_injection_supported,
        )
        return bool(_vendored_injection_supported())
    except Exception:
        return False


def _transformers_has_models(*names) -> bool:
    # The model-binding subprocess tests import real Transformers modeling modules.
    # Skip cleanly on a supported-but-older Transformers that predates them (e.g.
    # 4.57.x ships qwen3_next but not qwen3_5 / olmo_hybrid) instead of failing with
    # ModuleNotFoundError; a newer Transformers that has them still runs the tests.
    import importlib.util

    for name in names:
        try:
            if importlib.util.find_spec(f"transformers.models.{name}") is None:
                return False
        except Exception:
            return False
    return True


_NO_AUTORUN_SUBPROCESS = textwrap.dedent(
    """
    import os, sys
    os.environ["UNSLOTH_IS_PRESENT"] = "1"
    os.environ["UNSLOTH_VENDORED_FLA_NO_AUTORUN"] = "1"

    # Reading the support gate imports fla_vendor. With the autorun suppressed this
    # import must NOT run patch_vendor_fla() by itself, so no vendored fla is
    # injected into the interpreter merely from evaluating the support marker.
    import unsloth_zoo.temporary_patches.fla_vendor  # noqa: F401

    fla = sys.modules.get("fla")
    assert not getattr(fla, "_UNSLOTH_VENDORED_FLA", False), (
        "autorun injected vendored fla despite UNSLOTH_VENDORED_FLA_NO_AUTORUN"
    )
    print("NO_AUTORUN_OK")
    """
)


def test_no_autorun_suppresses_import_time_injection():
    # Runs in a fresh interpreter so it isolates the autorun-suppression invariant.
    # The in-process pytest session cannot assert this directly: conftest does
    # `import unsloth`, which legitimately applies TEMPORARY_PATCHES and (when an
    # installed fla is not newer than the vendored snapshot) correctly injects the
    # vendored copy. That is real patching, not an autorun leak, so the invariant is
    # checked in a subprocess that does not import unsloth.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ZOO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _NO_AUTORUN_SUBPROCESS],
        env=env,
        capture_output=True,
        text=True,
    )
    assert "NO_AUTORUN_OK" in proc.stdout, f"stdout=\n{proc.stdout}\nstderr=\n{proc.stderr}"
    assert proc.returncode == 0, f"stderr=\n{proc.stderr}"


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
    # num_stages is policy-driven (see test_prepare_wy_repr_bwd_num_stages_policy);
    # the warp pin is the part #1000 actually established, so pin that literally.
    assert "PREPARE_WY_REPR_BWD_NUM_STAGES = _prepare_wy_repr_bwd_num_stages(" in wy
    assert "for num_warps in PREPARE_WY_REPR_BWD_NUM_WARPS" in wy
    # The fwd recompute kernel keeps its wider space.
    assert "for num_warps in [2, 4, 8]" in wy

    co = (VENDORED / "ops" / "common" / "chunk_o.py").read_text()
    # Upstream's guard window, kept intact: [3.4.0, 3.7.1) only.
    assert "and TRITON_ABOVE_3_4_0\n        and not TRITON_ABOVE_3_7_1" in co
    # Hopper is decided per tensor, not from the import-time global. That global is
    # frozen from device 0 and is wrong in both directions on a mixed host: it misses
    # a Hopper card at a nonzero index, and it marks a call on an Ada/Blackwell card
    # as affected when device 0 is the Hopper one. It survives only as the fallback
    # for when the probe cannot tell, so a probe failure never fails open.
    assert "_on_hopper = _is_hopper_tensor(k)" in co
    assert "if _on_hopper is None:\n        _on_hopper = IS_NVIDIA_HOPPER" in co
    assert "def _device_is_nvidia_hopper(index):" in co
    # Capability, not shared memory: check_shared_mem('hopper') is a >=232448-byte
    # tier test that Blackwell B200 also passes, so it cannot detect Hopper.
    assert "torch.cuda.get_device_capability(index)[0] == 9" in co
    # fla #640's root cause is BK == 64 on Hopper, so we step the tile down instead
    # of refusing to run. Pin both halves so a re-vendor cannot silently drop them.
    assert "if HOPPER_DQKWG_BROKEN and BK == 64:\n        BK = 32" in co
    assert co.index("BK = 32") < co.index("NK = triton.cdiv(K, BK)"), (
        "the BK override must run before NK / the dg scratch / the grid are derived"
    )
    # The dead remediation must not come back: the TileLang kernels are pruned and
    # _inject_vendored_fla force-sets FLA_TILELANG=0, so installing it cannot help.
    assert "install tilelang: `pip install tilelang`" not in co
    assert 'pip install -U "triton>=3.7.1"' in co

    compat = (VENDORED / "utils" / "_compat.py").read_text()
    assert "TRITON_ABOVE_3_7_1 = " in compat
    utils_init = (VENDORED / "utils" / "__init__.py").read_text()
    assert "TRITON_ABOVE_3_7_1," in utils_init


def test_prepare_wy_repr_bwd_num_stages_policy():
    """fla PR #1000 pinned Blackwell to num_warps=2 AND num_stages=4. Only num_warps
    was implicated by #999, and that report was on triton 3.3.1, so num_stages is
    allowed to autotune from triton 3.6 onward and the exact upstream pin is kept
    below it. The warp pin is never relaxed. No GPU: the policy is a pure function of
    (is_blackwell, triton_above_3_6_0)."""
    import importlib.util

    src = VENDORED / "ops" / "gated_delta_rule" / "wy_fast.py"
    # Load just the helper, without importing fla (which needs torch + triton + CUDA).
    tree = ast.parse(src.read_text())
    fn = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_prepare_wy_repr_bwd_num_stages"
    )
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(src), "exec"), ns)
    policy = ns["_prepare_wy_repr_bwd_num_stages"]

    # Blackwell + triton < 3.6 -> the exact upstream pin.
    assert policy(True, False) == [4]
    # Blackwell + triton >= 3.6 -> more than one num_stages offered.
    assert len(policy(True, True)) > 1
    assert 4 in policy(True, True), "the validated config must stay in the space"
    # Non-Blackwell is untouched by #1000 on either triton.
    assert policy(False, False) == [2, 3, 4]
    assert policy(False, True) == [2, 3, 4]

    # The warp pin is separate and unconditional.
    wy = src.read_text()
    assert "PREPARE_WY_REPR_BWD_NUM_WARPS = [2] if IS_NVIDIA_BLACKWELL else [2, 4]" in wy
    # The gate is wired to the real constant, not left hardcoded.
    assert "PREPARE_WY_REPR_BWD_NUM_STAGES = _prepare_wy_repr_bwd_num_stages(\n" \
           "    IS_NVIDIA_BLACKWELL, TRITON_ABOVE_3_6_0,\n)" in wy
    compat = (VENDORED / "utils" / "_compat.py").read_text()
    assert 'TRITON_ABOVE_3_6_0 = package_version.parse(triton.__version__) >= ' \
           'package_version.parse("3.6.0")' in compat
    assert "TRITON_ABOVE_3_6_0," in (VENDORED / "utils" / "__init__.py").read_text()


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

def _rdna1_gpu_visible():
    """These subprocess tests assert that the vendored fla IS injected. On a host with an
    RDNA1 GPU visible unsloth routes gated-delta to the pure-torch path on purpose (no dot
    instructions, see _NO_DOT_INSTRUCTION_GFX), so the assertion is wrong there by design."""
    try:
        from unsloth_zoo.temporary_patches.fla_vendor import _gpu_lacks_dot_instructions
        return _gpu_lacks_dot_instructions()
    except Exception:
        return False


_skip_on_rdna1 = pytest.mark.skipif(
    _rdna1_gpu_visible(),
    reason = "an RDNA1 GPU is visible: fla is routed to the pure-torch path here by design",
)

@_skip_on_rdna1
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


# The decode-kernel name Transformers resolves but fla does not export.

class _FakeGatedDeltaModule:
    """Stand-in for fla.ops.gated_delta_rule, so the additivity rules can be
    checked without injecting anything into this interpreter."""

    def __init__(self, **names):
        self.__all__ = list(names)
        for k, v in names.items():
            setattr(self, k, v)


def _alias_into(fake, monkeypatch):
    from unsloth_zoo.temporary_patches import fla_vendor

    monkeypatch.setitem(sys.modules, "fla.ops.gated_delta_rule", fake)
    return fla_vendor._alias_missing_gated_delta_names()


def test_decode_alias_added_when_missing(monkeypatch):
    def fused_recurrent_gated_delta_rule():
        return "fused"

    fake = _FakeGatedDeltaModule(
        chunk_gated_delta_rule=lambda: "chunk",
        fused_recurrent_gated_delta_rule=fused_recurrent_gated_delta_rule,
    )
    added = _alias_into(fake, monkeypatch)

    assert added == ("recurrent_gated_delta_rule",), added
    # This is the exact lookup transformers' use_kernel_func_from_hub_with_fallback
    # performs; before the alias it returns None and the decorator silently keeps
    # the pure-PyTorch loop.
    assert fake.recurrent_gated_delta_rule is fused_recurrent_gated_delta_rule
    assert "recurrent_gated_delta_rule" in fake.__all__


def test_decode_alias_never_overwrites_a_real_export(monkeypatch):
    """A future fla that exports the name itself must keep its own implementation."""
    def upstream():
        return "upstream"

    fake = _FakeGatedDeltaModule(
        fused_recurrent_gated_delta_rule=lambda: "fused",
        recurrent_gated_delta_rule=upstream,
    )
    added = _alias_into(fake, monkeypatch)

    assert added == (), added
    assert fake.recurrent_gated_delta_rule is upstream
    assert fake.__all__.count("recurrent_gated_delta_rule") == 1


def test_decode_alias_noop_on_partial_fla(monkeypatch):
    """Nothing to alias from binds no name: an AttributeError at decoration time
    would be worse than the slow path it replaces."""
    fake = _FakeGatedDeltaModule(chunk_gated_delta_rule=lambda: "chunk")
    added = _alias_into(fake, monkeypatch)

    assert added == (), added
    assert not hasattr(fake, "recurrent_gated_delta_rule")


def test_decode_alias_is_idempotent(monkeypatch):
    fake = _FakeGatedDeltaModule(
        fused_recurrent_gated_delta_rule=lambda: "fused",
    )
    assert _alias_into(fake, monkeypatch) == ("recurrent_gated_delta_rule",)
    assert _alias_into(fake, monkeypatch) == ()
    assert fake.__all__.count("recurrent_gated_delta_rule") == 1


def test_decode_alias_survives_absent_fla(monkeypatch):
    """No fla at all (pure-torch host) must not raise."""
    from unsloth_zoo.temporary_patches import fla_vendor

    monkeypatch.delitem(sys.modules, "fla.ops.gated_delta_rule", raising=False)
    monkeypatch.setattr(
        fla_vendor.importlib, "import_module",
        lambda *a, **k: (_ for _ in ()).throw(ImportError("no fla")),
    )
    assert fla_vendor._alias_missing_gated_delta_names() == ()


_DECODE_ALIAS_SUBPROCESS = textwrap.dedent(
    """
    import os, sys
    os.environ["UNSLOTH_IS_PRESENT"] = "1"
    os.environ.pop("UNSLOTH_VENDORED_FLA_NO_AUTORUN", None)

    from unsloth_zoo.temporary_patches.fla_vendor import patch_vendor_fla
    patch_vendor_fla()

    import fla.ops.gated_delta_rule as gdr
    assert gdr.recurrent_gated_delta_rule is gdr.fused_recurrent_gated_delta_rule

    # The lookup Transformers actually performs, when it is new enough to do it.
    try:
        from transformers.integrations.hub_kernels import resolve_internal_import
    except ImportError:
        print("DECODE_ALIAS_OK (transformers predates the kernel-hub resolver)")
    else:
        import importlib
        resolved = resolve_internal_import(
            importlib.import_module("fla"),
            "ops.gated_delta_rule.recurrent_gated_delta_rule",
        )
        assert resolved is not None, "transformers still cannot resolve the decode kernel"
        assert resolved is gdr.fused_recurrent_gated_delta_rule
        print("DECODE_ALIAS_OK")
    """
)


@pytest.mark.skipif(
    not _injection_supported(),
    reason="vendored fla kernels need CUDA + torch>=2.7 + triton>=3.3",
)
@_skip_on_rdna1
def test_decode_alias_resolves_through_transformers_subprocess():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ZOO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("UNSLOTH_VENDORED_FLA_NO_AUTORUN", None)
    proc = subprocess.run(
        [sys.executable, "-c", _DECODE_ALIAS_SUBPROCESS],
        env=env,
        capture_output=True,
        text=True,
    )
    assert "DECODE_ALIAS_OK" in proc.stdout, f"stdout=\n{proc.stdout}\nstderr=\n{proc.stderr}"
    assert proc.returncode == 0, f"stderr=\n{proc.stderr}"


# Caller-aware availability probe: uncovered models keep the pure-torch path.

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
@pytest.mark.skipif(
    not _transformers_has_models("qwen3_5", "olmo_hybrid"),
    reason="installed transformers lacks the qwen3_5 / olmo_hybrid modeling modules",
)
@_skip_on_rdna1
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


def test_version_strictly_after():
    from unsloth_zoo.temporary_patches.fla_vendor import _version_strictly_after

    assert _version_strictly_after("0.6.0", "0.5.1") is True
    assert _version_strictly_after("0.5.2", "0.5.1") is True
    assert _version_strictly_after("0.5.1", "0.5.1") is False
    assert _version_strictly_after("0.5.0", "0.5.1") is False
    assert _version_strictly_after("0.5.1.dev3", "0.5.1") is False
    assert _version_strictly_after("0.6.0.dev1", "0.5.1") is True
    # Unparseable input is conservatively not-newer.
    assert _version_strictly_after("not-a-version", "0.5.1") is False


def test_defer_to_installed_fla_only_when_strictly_newer(monkeypatch):
    # Auto-detection: use a user-installed fla only when it is strictly newer than
    # the vendored snapshot; an equal or older install is shadowed by the vendored
    # kernels (which carry post-0.5.1 backports). No env flag required.
    import types
    from unsloth_zoo.temporary_patches import fla_vendor as fv

    def set_installed(ver):
        m = types.ModuleType("fla")
        if ver is not None:
            m.__version__ = ver
        # No vendored marker: looks like a genuine user install.
        monkeypatch.setitem(sys.modules, "fla", m)

    set_installed("0.6.0")
    assert fv._should_defer_to_installed_fla() is True          # newer -> use theirs
    set_installed(fv._VENDORED_FLA_VERSION)
    assert fv._should_defer_to_installed_fla() is False         # equal -> use vendored
    set_installed("0.4.0")
    assert fv._should_defer_to_installed_fla() is False         # older -> use vendored
    set_installed(None)
    assert fv._should_defer_to_installed_fla() is True          # unversioned deliberate install

    # Our own vendored module must never be treated as a user install.
    m = types.ModuleType("fla")
    m.__version__ = fv._VENDORED_FLA_VERSION
    setattr(m, fv._VENDORED_MARK, True)
    monkeypatch.setitem(sys.modules, "fla", m)
    assert fv._should_defer_to_installed_fla() is False


def test_python_39_skips_injection(monkeypatch):
    # The snapshot uses runtime PEP 604 annotations, which raise on 3.9; the
    # support gate must answer False there instead of import-fail + rollback.
    from unsloth_zoo.temporary_patches import fla_vendor

    monkeypatch.setattr(fla_vendor.sys, "version_info", (3, 9, 19, "final", 0))
    assert fla_vendor._torch_triton_cuda_supported() is False


def test_hopper_bad_triton_range_is_suspect_but_still_supported():
    # Hopper + triton [3.4.0, 3.7.1) is the fla #640 miscompile range. It used to
    # skip injection entirely; the vendored chunk_bwd_dqkwg now steps around the
    # bad BK=64 tile, so such a host is flagged *suspect* (prefer our copy over an
    # installed fla) while remaining *supported* (keep the Triton fast path).
    import types

    from unsloth_zoo.temporary_patches.fla_vendor import _hopper_dqkwg_suspect

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
        assert _hopper_dqkwg_suspect(hopper, tri) is want_on_hopper, ver
        assert _hopper_dqkwg_suspect(blackwell, tri) is False, ver

    import inspect

    from unsloth_zoo.temporary_patches.fla_vendor import _torch_triton_cuda_supported

    src = inspect.getsource(_torch_triton_cuda_supported)
    assert "_hopper_dqkwg_suspect(" not in src, (
        "the support gate must not disable fla on Hopper; the vendored kernel "
        "avoids the miscompiled tile instead"
    )


def test_rocm_major9_device_not_treated_as_hopper():
    # ROCm/AMD cards can report capability major 9 without being NVIDIA Hopper. On a
    # HIP build (torch.version.hip set) the bare major==9 signal must not trip the
    # Hopper guard, or AMD users lose the vendored path on triton [3.4.0, 3.7.1).
    import types

    from unsloth_zoo.temporary_patches.fla_vendor import _hopper_dqkwg_suspect

    def fake_torch(name, major, hip):
        cuda = types.SimpleNamespace(
            device_count=lambda: 1,
            get_device_name=lambda i=0, n=name: n,
            get_device_capability=lambda i=0, m=major: (m, 0),
        )
        return types.SimpleNamespace(cuda=cuda, version=types.SimpleNamespace(hip=hip))

    bad = types.SimpleNamespace(__version__="3.6.0")  # in [3.4.0, 3.7.1)
    # AMD Instinct on a ROCm build: major 9 but not Hopper -> keep the fast path.
    amd = fake_torch("AMD Instinct MI300X", 9, "6.0.32830")
    assert _hopper_dqkwg_suspect(amd, bad) is False
    # A real Hopper on a CUDA build (hip=None) still trips, by name and by major.
    nvidia_named = fake_torch("NVIDIA H100 80GB HBM3", 9, None)
    nvidia_unnamed = fake_torch("", 9, None)
    assert _hopper_dqkwg_suspect(nvidia_named, bad) is True
    assert _hopper_dqkwg_suspect(nvidia_unnamed, bad) is True


def test_hopper_at_nonzero_device_index_trips_guard():
    # On a mixed host the model can run on a nonzero Hopper card while cuda:0 is a
    # different architecture; the guard must scan every visible device, not just 0.
    import types

    from unsloth_zoo.temporary_patches.fla_vendor import _hopper_dqkwg_suspect

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

    assert _hopper_dqkwg_suspect(ada_then_hopper, bad) is True
    assert _hopper_dqkwg_suspect(ada_then_hopper, ok) is False
    assert _hopper_dqkwg_suspect(ada_only, bad) is False


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


# Pruned TileLang backend cannot import a broken external tilelang.
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
@_skip_on_rdna1
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


# Pruned IntraCard CP backend cannot be re-enabled into the missing module.
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
@_skip_on_rdna1
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


# Force-rebind: replacing an already-loaded real fla under the escape hatch.
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
@_skip_on_rdna1
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


# Kernel-hub closures frozen by an import that happened before fla was live.

def _has_kernel_hub_fallback() -> bool:
    # use_kernel_func_from_hub_with_fallback arrived with huggingface/transformers#47630.
    # On anything older the repair correctly returns () and there is nothing to patch.
    try:
        from transformers.integrations.hub_kernels import (  # noqa: F401
            use_kernel_func_from_hub_with_fallback,
        )
        return True
    except Exception:
        return False


requires_kernel_hub = pytest.mark.skipif(
    not _has_kernel_hub_fallback(),
    reason="transformers predates use_kernel_func_from_hub_with_fallback",
)


def _fake_wrapper(implementation, original, params=("q", "k")):
    """A stand-in for what use_kernel_func_from_hub_with_fallback builds: a closure
    over (applicable_params, implementation) with __wrapped__ set to the original."""
    def wrapped(*args, **kwargs):
        return implementation(*args, **kwargs)
    wrapped.__wrapped__ = original
    # Force a closure with the same shape the real decorator produces.
    def outer(applicable_params, impl):
        def inner(*args, **kwargs):
            return impl(*args, **kwargs)
        return inner
    inner = outer(params, implementation)
    inner.__wrapped__ = original
    return inner


def _fake_modeling(monkeypatch, package, wrapper, attribute="torch_chunk_gated_delta_rule"):
    import types
    module = types.ModuleType(f"transformers.models.{package}.modeling_{package}")
    setattr(module, attribute, wrapper)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def test_resolved_implementation_reads_the_closure():
    from unsloth_zoo.temporary_patches.fla_vendor import _resolved_implementation

    def kernel(): return "kernel"
    def original(): return "torch"

    assert _resolved_implementation(_fake_wrapper(kernel, original)) is kernel
    assert _resolved_implementation(lambda: None) is None
    assert _resolved_implementation(object()) is None


@requires_kernel_hub
def test_late_import_repair_rebinds_the_frozen_fallback(monkeypatch):
    """The whole point: a wrapper still closed over its own torch fallback is rebuilt."""
    from unsloth_zoo.temporary_patches import fla_vendor

    def original(): return "torch"
    def kernel(): return "fla"

    module = _fake_modeling(monkeypatch, "qwen3_5", _fake_wrapper(original, original))

    import transformers.integrations.hub_kernels as hub
    monkeypatch.setattr(
        hub, "use_kernel_func_from_hub_with_fallback",
        lambda name, package, internal_path=None: (
            lambda fn: _fake_wrapper(kernel, fn, params=("q", "k", "v", "g"))
        ),
    )
    repaired = fla_vendor._repair_kernel_hub_closures(packages=("qwen3_5",))

    assert repaired == ("qwen3_5.torch_chunk_gated_delta_rule",), repaired
    assert fla_vendor._resolved_implementation(module.torch_chunk_gated_delta_rule) is kernel


@requires_kernel_hub
def test_late_import_repair_leaves_the_live_kernel_alone(monkeypatch):
    """A module imported in the right order already dispatches to the live fla."""
    from unsloth_zoo.temporary_patches import fla_vendor

    def original(): return "torch"
    def kernel(): return "fla"

    wrapper = _fake_wrapper(kernel, original)
    module = _fake_modeling(monkeypatch, "qwen3_5", wrapper)
    monkeypatch.setattr(fla_vendor, "_live_gated_delta_kernel", lambda name: kernel)

    import transformers.integrations.hub_kernels as hub
    monkeypatch.setattr(
        hub, "use_kernel_func_from_hub_with_fallback",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not rebuild")),
    )
    assert fla_vendor._repair_kernel_hub_closures(packages=("qwen3_5",)) == ()
    assert module.torch_chunk_gated_delta_rule is wrapper


@requires_kernel_hub
def test_late_import_repair_replaces_a_purged_install_kernel(monkeypatch):
    """The case UNSLOTH_FORCE_VENDORED_FLA and the Hopper #640 switch create: the
    wrapper closed over a real install's kernel that has since been purged. That is
    not "already dispatching to a real kernel", it is the miscompiled backward we
    replaced the install to avoid, so it must be rebound onto the live fla."""
    from unsloth_zoo.temporary_patches import fla_vendor

    def original(): return "torch"
    def purged(): return "the install we just deleted"
    def vendored(): return "fla"

    wrapper = _fake_wrapper(purged, original)
    module = _fake_modeling(monkeypatch, "qwen3_5", wrapper)
    monkeypatch.setattr(fla_vendor, "_live_gated_delta_kernel", lambda name: vendored)

    import transformers.integrations.hub_kernels as hub
    monkeypatch.setattr(
        hub, "use_kernel_func_from_hub_with_fallback",
        lambda *a, **k: (lambda fn: _fake_wrapper(vendored, fn)),
    )
    repaired = fla_vendor._repair_kernel_hub_closures(packages=("qwen3_5",))

    assert repaired == ("qwen3_5.torch_chunk_gated_delta_rule",), repaired
    assert fla_vendor._resolved_implementation(
        module.torch_chunk_gated_delta_rule) is vendored


@requires_kernel_hub
def test_late_import_repair_prefers_torch_over_a_purged_kernel(monkeypatch):
    """Stale kernel, and nothing live to swap in. Pure torch beats calling into an
    install that is no longer on sys.modules."""
    from unsloth_zoo.temporary_patches import fla_vendor

    def original(): return "torch"
    def purged(): return "gone"

    module = _fake_modeling(monkeypatch, "qwen3_5", _fake_wrapper(purged, original))
    monkeypatch.setattr(fla_vendor, "_live_gated_delta_kernel", lambda name: None)

    import transformers.integrations.hub_kernels as hub
    monkeypatch.setattr(
        hub, "use_kernel_func_from_hub_with_fallback",
        lambda *a, **k: (lambda fn: _fake_wrapper(fn, fn)),
    )
    repaired = fla_vendor._repair_kernel_hub_closures(packages=("qwen3_5",))

    assert repaired == ("qwen3_5.torch_chunk_gated_delta_rule",), repaired
    assert fla_vendor._resolved_implementation(
        module.torch_chunk_gated_delta_rule) is original


@requires_kernel_hub
def test_late_import_repair_keeps_the_wrapper_when_nothing_to_bind(monkeypatch):
    """If the rebuild still resolves to the fallback, fla genuinely has no kernel:
    keep what was there rather than swapping in an identical object."""
    from unsloth_zoo.temporary_patches import fla_vendor

    def original(): return "torch"
    wrapper = _fake_wrapper(original, original)
    module = _fake_modeling(monkeypatch, "qwen3_5", wrapper)
    monkeypatch.setattr(fla_vendor, "_live_gated_delta_kernel", lambda name: None)

    import transformers.integrations.hub_kernels as hub
    monkeypatch.setattr(
        hub, "use_kernel_func_from_hub_with_fallback",
        lambda *a, **k: (lambda fn: _fake_wrapper(fn, fn)),
    )
    assert fla_vendor._repair_kernel_hub_closures(packages=("qwen3_5",)) == ()
    assert module.torch_chunk_gated_delta_rule is wrapper


@requires_kernel_hub
def test_force_fallback_rebinds_the_compiled_copy(monkeypatch):
    """unsloth runs unsloth_compiled_module_<type>, whose kernel-hub decorator resolves
    fla on its own. On RDNA1 that copy must be forced to the pure-torch fallback too, or
    the compiled forward re-enters fla and aborts with FDOT2."""
    from unsloth_zoo.temporary_patches import fla_vendor

    def original(): return "torch"
    def kernel(): return "fla"

    module = _fake_modeling(monkeypatch, "qwen3_5", _fake_wrapper(kernel, original))

    import types
    compiled_name = "unsloth_compiled_module_qwen3_5"
    compiled = types.ModuleType(compiled_name)
    compiled.torch_chunk_gated_delta_rule = _fake_wrapper(kernel, original)
    monkeypatch.setitem(sys.modules, compiled_name, compiled)

    forced = fla_vendor._force_kernel_hub_fallback(packages=("qwen3_5",))

    assert f"{compiled_name}.torch_chunk_gated_delta_rule" in forced, forced
    assert module.torch_chunk_gated_delta_rule is original
    assert compiled.torch_chunk_gated_delta_rule is original


def test_late_import_repair_skips_undecorated_attributes(monkeypatch):
    """On a transformers that predates #47630 there is no __wrapped__ to rebuild from."""
    from unsloth_zoo.temporary_patches import fla_vendor

    def plain(): return "plain"
    module = _fake_modeling(monkeypatch, "qwen3_5", plain)
    assert fla_vendor._repair_kernel_hub_closures(packages=("qwen3_5",)) == ()
    assert module.torch_chunk_gated_delta_rule is plain


def test_late_import_repair_is_scoped_to_vendor_covered_models():
    """olmo_hybrid imports ShortConvolution, which is not vendored, so its probe must
    stay False; rebinding vendored kernels into it would contradict that."""
    from unsloth_zoo.temporary_patches import fla_vendor
    import inspect

    default = inspect.signature(
        fla_vendor._repair_kernel_hub_closures
    ).parameters["packages"].default
    assert "olmo_hybrid" not in default
    assert default is fla_vendor._REPAIR_MODELING


def test_late_import_repair_survives_missing_hub_kernels(monkeypatch):
    from unsloth_zoo.temporary_patches import fla_vendor

    real_import = __import__

    def fail(name, *args, **kwargs):
        if name == "transformers.integrations.hub_kernels":
            raise ImportError("too old")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fail)
    assert fla_vendor._repair_kernel_hub_closures(packages=("qwen3_5",)) == ()


def test_patch_vendor_fla_survives_a_broken_repair(monkeypatch):
    """The repair runs in the exit path; it must never take patch_vendor_fla down."""
    from unsloth_zoo.temporary_patches import fla_vendor

    monkeypatch.setattr(
        fla_vendor, "_repair_kernel_hub_closures",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(fla_vendor, "_patch_vendor_fla", lambda phase=None: "sentinel")
    assert fla_vendor.patch_vendor_fla() == "sentinel"


# ---------------------------------------------------------------------------
# RDNA1 (gfx1010 / gfx1013) has no dot instructions. Triton emits v_dot2 for a
# 16-bit tl.dot regardless, so every fla chunk kernel aborts the PROCESS in LLVM
# ("Cannot select: AMDGPUISD::FDOT2"). Found on an RX 5700 XT: Qwen3.5 died at the
# first kernel compile; routed to transformers' pure-torch gated-delta path it
# trained 3/3 steps with losses matching an RX 6500 XT on the fla kernels.
# ---------------------------------------------------------------------------

def _fake_torch(hip, archs, available=True):
    """A torch stand-in with only what _gpu_lacks_dot_instructions reads."""
    from types import SimpleNamespace

    def props(i):
        return SimpleNamespace(gcnArchName=archs[i])

    return SimpleNamespace(
        version=SimpleNamespace(hip="7.13.99004" if hip else None),
        cuda=SimpleNamespace(
            is_available=lambda: available,
            device_count=lambda: len(archs),
            get_device_properties=props,
        ),
    )


def test_rdna1_without_dot_instructions_is_detected():
    from unsloth_zoo.temporary_patches.fla_vendor import _gpu_lacks_dot_instructions

    assert _gpu_lacks_dot_instructions(_fake_torch(True, ["gfx1010:xnack-"])) is True
    assert _gpu_lacks_dot_instructions(_fake_torch(True, ["gfx1013"])) is True
    # Nonzero device index counts too: a model can be placed there.
    assert _gpu_lacks_dot_instructions(_fake_torch(True, ["gfx1034", "gfx1010:xnack-"])) is True


def test_gpus_with_dot_instructions_keep_fla():
    from unsloth_zoo.temporary_patches.fla_vendor import _gpu_lacks_dot_instructions

    # gfx1011 / gfx1012 are RDNA1 too but do have dot instructions (LLVM dot1/dot2-insts).
    for arch in ("gfx1011", "gfx1012", "gfx1030", "gfx1034", "gfx1100", "gfx1201", "gfx90a", "gfx942"):
        assert _gpu_lacks_dot_instructions(_fake_torch(True, [arch])) is False, arch


def test_dot_instruction_gate_is_rocm_only_and_fails_open():
    from unsloth_zoo.temporary_patches.fla_vendor import _gpu_lacks_dot_instructions

    # A CUDA build never reports a gfx arch worth acting on.
    assert _gpu_lacks_dot_instructions(_fake_torch(False, ["gfx1010"])) is False
    # No usable accelerator, or an unreadable arch: nothing is narrowed.
    assert _gpu_lacks_dot_instructions(_fake_torch(True, ["gfx1010"], available=False)) is False
    assert _gpu_lacks_dot_instructions(_fake_torch(True, [""])) is False
    assert _gpu_lacks_dot_instructions(_fake_torch(True, [])) is False
    # A torch that raises anywhere answers False rather than propagating.
    class Broken:
        version = None
    assert _gpu_lacks_dot_instructions(Broken()) is False


def test_rdna1_takes_the_pure_torch_gated_delta_path(monkeypatch):
    """On such a GPU patch_vendor_fla must do what the Hopper opt-out does: make the
    availability probe answer False, unbind any already-imported gated-delta module, say
    why, and never reach injection (which would compile the kernels and abort)."""
    from unsloth_zoo.temporary_patches import fla_vendor

    calls = []
    monkeypatch.setattr(fla_vendor, "_gpu_lacks_dot_instructions", lambda torch_mod=None: True)
    monkeypatch.setattr(fla_vendor, "_transformers_uses_availability_probe", lambda: True)
    monkeypatch.setattr(fla_vendor, "_patch_is_available", lambda *a, **k: calls.append(("probe", a)))
    monkeypatch.setattr(
        fla_vendor, "_disable_already_imported_gated_delta", lambda *a, **k: calls.append(("unbind", k))
    )
    monkeypatch.setattr(fla_vendor, "_inject_vendored_fla", lambda: calls.append(("inject", None)) or (False, False))
    monkeypatch.setattr(
        fla_vendor, "_patch_l2norm_fp32_on_torch_path", lambda *a, **k: calls.append(("l2norm", None)) or []
    )
    monkeypatch.setattr(fla_vendor, "_FLA_DISABLED_REASON", None)

    fla_vendor._patch_vendor_fla()

    kinds = [kind for kind, _ in calls]
    assert kinds[:3] == ["probe", "unbind", "l2norm"]
    assert "inject" not in kinds
    assert calls[0][1] == (fla_vendor._unavailable_probe,)
    assert "RDNA1" in calls[1][1]["why"] or "dot instructions" in calls[1][1]["why"]
    reason = fla_vendor.fla_unavailable_reason()
    assert reason and "FDOT2" in reason and "pure-PyTorch" in reason


def test_gpus_with_dot_instructions_do_not_trip_the_rdna1_path(monkeypatch):
    from unsloth_zoo.temporary_patches import fla_vendor

    monkeypatch.setattr(fla_vendor, "_gpu_lacks_dot_instructions", lambda torch_mod=None: False)
    monkeypatch.setattr(fla_vendor, "_FLA_DISABLED_REASON", None)
    marked = []
    monkeypatch.setattr(fla_vendor, "_mark_fla_disabled_no_dot_instructions", lambda: marked.append(1))
    # Stop before the real injection machinery: the Hopper opt-out is off, so the next
    # thing _patch_vendor_fla does is consult the source-preference flags.
    monkeypatch.setattr(fla_vendor, "_flag", lambda name: False)
    monkeypatch.setattr(fla_vendor, "_torch_triton_cuda_supported", lambda: False)

    fla_vendor._patch_vendor_fla()

    assert marked == []
    assert fla_vendor.fla_unavailable_reason() is None


def _transformers_style_l2norm(x, dim = -1, eps = 1e-6):
    """Verbatim shape of transformers' pure-torch l2norm: reduction in the input dtype."""
    import torch
    inv_norm = torch.rsqrt((x * x).sum(dim = dim, keepdim = True) + eps)
    return x * inv_norm


def test_fp32_l2norm_matches_fp32_reference_and_keeps_dtype():
    import torch
    from unsloth_zoo.temporary_patches.fla_vendor import _fp32_l2norm

    x = torch.randn(4, 128, dtype = torch.float16)
    out = _fp32_l2norm(x)
    assert out.dtype == torch.float16
    ref = _transformers_style_l2norm(x.float())
    torch.testing.assert_close(out.float(), ref, atol = 2e-3, rtol = 2e-3)
    # Rows end up unit length, as l2norm promises.
    torch.testing.assert_close(out.float().norm(dim = -1), torch.ones(4), atol = 5e-3, rtol = 0)


def test_fp32_l2norm_survives_the_float16_overflow_that_gives_nan_grads():
    """128 * 300^2 = 1.15e7 overflows float16 (max 65504), so transformers' version stores
    inv_norm = rsqrt(inf) = 0: a finite forward. fp16 training then backpropagates a
    loss-scaled gradient; sum(x * grad) overflows to inf in float16 and inf * 0 inside
    RsqrtBackward is the NaN seen on the RX 5700 XT. The loss itself is float32 in a
    real trainer, so the scale is applied to a float32 sum here as well."""
    import torch
    from unsloth_zoo.temporary_patches.fla_vendor import _fp32_l2norm

    loss_scale = 1024.0
    x = torch.full((2, 128), 300.0, dtype = torch.float16, requires_grad = True)
    bad = _transformers_style_l2norm(x)
    assert torch.isfinite(bad).all()          # the forward looks fine ...
    (bad.float().sum() * loss_scale).backward()
    assert not torch.isfinite(x.grad).all()  # ... the backward is not

    x2 = torch.full((2, 128), 300.0, dtype = torch.float16, requires_grad = True)
    good = _fp32_l2norm(x2)
    (good.float().sum() * loss_scale).backward()
    assert torch.isfinite(good).all()
    assert torch.isfinite(x2.grad).all()
    torch.testing.assert_close(good.float().norm(dim = -1), torch.ones(2), atol = 5e-3, rtol = 0)


def test_l2norm_patch_rebinds_only_imported_gated_delta_modules(monkeypatch):
    import types
    from unsloth_zoo.temporary_patches import fla_vendor

    fake_pkg = "unsloth_test_gated_delta"
    modname = f"transformers.models.{fake_pkg}.modeling_{fake_pkg}"
    mod = types.ModuleType(modname)
    mod.l2norm = _transformers_style_l2norm
    monkeypatch.setitem(sys.modules, modname, mod)
    plain_pkg = "unsloth_test_no_l2norm"
    plain_name = f"transformers.models.{plain_pkg}.modeling_{plain_pkg}"
    plain = types.ModuleType(plain_name)
    monkeypatch.setitem(sys.modules, plain_name, plain)

    # unsloth's compiled copy of a model module carries its own l2norm and is what runs.
    compiled_name = "unsloth_compiled_module_unsloth_test_gated_delta"
    compiled = types.ModuleType(compiled_name)
    compiled.l2norm = _transformers_style_l2norm
    monkeypatch.setitem(sys.modules, compiled_name, compiled)

    patched = fla_vendor._patch_l2norm_fp32_on_torch_path(packages = (fake_pkg, plain_pkg, "never_imported_pkg"))

    assert modname in patched and compiled_name in patched
    assert mod.l2norm is fla_vendor._fp32_l2norm
    assert compiled.l2norm is fla_vendor._fp32_l2norm
    assert not hasattr(plain, "l2norm")
    # Idempotent: a second pass finds nothing left to do.
    assert fla_vendor._patch_l2norm_fp32_on_torch_path(packages = (fake_pkg, plain_pkg)) == []
    assert mod.l2norm is fla_vendor._fp32_l2norm
