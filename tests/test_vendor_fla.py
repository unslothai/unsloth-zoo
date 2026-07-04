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
