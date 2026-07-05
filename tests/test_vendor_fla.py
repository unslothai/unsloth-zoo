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

ZOO_ROOT = pathlib.Path(__file__).resolve().parents[1]
VENDORED = ZOO_ROOT / "unsloth_zoo" / "_vendored" / "fla"


def _cuda_triton_ok() -> bool:
    try:
        import torch
        import triton
        from packaging import version
        return (
            bool(torch.cuda.is_available())
            and version.parse(torch.__version__.split("+")[0]) >= version.parse("2.7")
            and version.parse(triton.__version__.split("+")[0]) >= version.parse("3.3")
        )
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
    not _cuda_triton_ok(),
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
    not _cuda_triton_ok(),
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
