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

"""The vendored fla snapshot shadows an installed fla-core <= 0.5.1 so the gated-delta models
get its backported fixes, but it ships only the gated-delta closure. Kimi delta attention
(glm5_next, kimi_linear) asks transformers for fla.ops.kda, which an installed fla-core 0.5.1
has; that op must still load from the install, while every module the snapshot ships keeps
coming from the snapshot. An install of another version is not mixed in."""

import json
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
os.environ.setdefault("UNSLOTH_VENDORED_FLA_NO_AUTORUN", "1")

try:
    from unsloth_zoo.temporary_patches.fla_vendor import (
        _gpu_lacks_dot_instructions,
        _vendored_injection_supported,
    )
except ImportError as _e:
    pytest.skip(f"unsloth_zoo unavailable: {_e}", allow_module_level = True)

ZOO_ROOT = pathlib.Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not _vendored_injection_supported() or _gpu_lacks_dot_instructions(),
    reason = "the vendored fla is only injected with CUDA + torch>=2.7 + triton>=3.3 (never on RDNA1)",
)


def _fake_install(site, version):
    """A minimal installed fla: an op the snapshot lacks, plus a module the snapshot ships."""
    files = {
        "fla/__init__.py": f'__version__ = "{version}"\n',
        "fla/ops/__init__.py": "",
        "fla/ops/common/__init__.py": "",
        "fla/ops/common/chunk_delta_h.py": 'SOURCE = "installed"\n',
        # A file the snapshot prunes from a package it ships: must stay unimportable.
        "fla/ops/common/intracard_cp.py": 'SOURCE = "installed"\n',
        "fla/ops/kda/__init__.py": textwrap.dedent(
            """
            # Absolute imports inside the install must resolve against the live (vendored) fla.
            import fla.ops.common.chunk_delta_h as _shared
            SHARED_FILE = _shared.__file__

            def chunk_kda(q, k, v, g, beta, **kwargs):
                return "installed chunk_kda"

            def fused_recurrent_kda(q, k, v, g, beta, **kwargs):
                return "installed fused_recurrent_kda"
            """
        ),
        f"flash_linear_attention-{version}.dist-info/METADATA": (
            f"Metadata-Version: 2.1\nName: flash-linear-attention\nVersion: {version}\n"
        ),
    }
    for rel, text in files.items():
        path = site / rel
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_text(text)


_CHILD = textwrap.dedent(
    """
    import importlib, importlib.util, json, sys
    from unsloth_zoo.temporary_patches import fla_vendor
    fla_vendor.patch_vendor_fla()
    import fla
    out = {"vendored": bool(getattr(fla, "_UNSLOTH_VENDORED_FLA", False))}
    import fla.ops.gated_delta_rule as gdr
    import fla.ops.common.chunk_delta_h as shared
    out["gdr_file"] = gdr.__file__
    out["shared_file"] = shared.__file__
    out["pruned_file_spec"] = importlib.util.find_spec("fla.ops.common.intracard_cp") is not None
    try:
        import fla.ops.kda as kda
        out["kda_file"] = kda.__file__
        out["kda_shared_file"] = kda.SHARED_FILE
    except ModuleNotFoundError:
        out["kda_file"] = None
    if importlib.util.find_spec("transformers.models.glm5_next") is not None:
        from transformers.models.glm5_next import modeling_glm5_next as mg
        # The kernel-hub wrapper closes over the implementation it resolved at import.
        impls = [c.cell_contents for c in (mg.chunk_kimi_delta_attention.__closure__ or ())
                 if callable(getattr(c, "cell_contents", None))]
        out["glm5_next_chunk"] = [f"{f.__module__}.{f.__name__}" for f in impls]
    print("@@@" + json.dumps(out))
    """
)


def _run(tmp_path, version, extra_env = None):
    site = tmp_path / "site"
    _fake_install(site, version)
    env = dict(os.environ)
    env["UNSLOTH_IS_PRESENT"] = "1"
    env["UNSLOTH_VENDORED_FLA_NO_AUTORUN"] = "1"
    for name in ("UNSLOTH_DISABLE_VENDORED_FLA", "UNSLOTH_FORCE_VENDORED_FLA", "UNSLOTH_DISABLE_INSTALLED_FLA_FALLBACK"):
        env.pop(name, None)
    env.update(extra_env or {})
    env["PYTHONPATH"] = os.pathsep.join([str(ZOO_ROOT), str(site)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    proc = subprocess.run([sys.executable, "-c", _CHILD], cwd = str(tmp_path), env = env,
                          capture_output = True, text = True, timeout = 600)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    line = next(l for l in proc.stdout.splitlines() if l.startswith("@@@"))
    return json.loads(line[3:]), str(site)


def test_same_version_install_serves_ops_the_snapshot_lacks(tmp_path):
    out, site = _run(tmp_path, "0.5.1")
    assert out["vendored"]
    # The snapshot keeps every module it ships, including the one KDA shares with gated delta.
    assert "_vendored" in out["gdr_file"]
    assert "_vendored" in out["shared_file"]
    assert out["kda_shared_file"] == out["shared_file"]
    assert out["pruned_file_spec"] is False
    # KDA is not in the snapshot, so it comes from the install.
    assert out["kda_file"] is not None and out["kda_file"].startswith(site), out
    if "glm5_next_chunk" in out:
        assert "fla.ops.kda.chunk_kda" in out["glm5_next_chunk"], out


@pytest.mark.parametrize(
    "version, extra_env",
    [("0.5.0", None), ("0.5.1", {"UNSLOTH_DISABLE_INSTALLED_FLA_FALLBACK": "1"})],
    ids = ["older_install", "kill_switch"],
)
def test_other_version_or_kill_switch_is_not_mixed_in(tmp_path, version, extra_env):
    out, _ = _run(tmp_path, version, extra_env)
    assert out["vendored"]
    assert "_vendored" in out["gdr_file"]
    assert out["kda_file"] is None
    if "glm5_next_chunk" in out:
        assert "fla.ops.kda.chunk_kda" not in out["glm5_next_chunk"], out
