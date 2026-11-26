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

"""patch_torch_compile leaves Inductor's autotune_multi_device at the torch default. Forcing it on made the
subprocess autotune pool int()-parse the visible-devices mask, so a UUID / MIG mask raised ValueError, and each rank
tuned on every visible card. CPU-only: runs in a fresh interpreter so the process config stays clean."""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

_REPO = pathlib.Path(__file__).resolve().parents[1]

_PROBE = r"""
import json, os, sys
import torch
import unsloth_zoo, unsloth_zoo.patching_utils as pu
import torch._inductor.config as config
default = config.autotune_multi_device  # fresh interpreter, env override popped: the torch default
pu.patch_torch_compile()
out = {"file": unsloth_zoo.__file__, "value": config.autotune_multi_device, "default": default}
from torch._inductor import autotune_process as ap
# A UUID mask under a private env var name, so the host's real device mask is never touched.
os.environ["UNSLOTH_TEST_VISIBLE_DEVICES"] = "GPU-6f1d3a2c-1b2e-4c3d-9e8f-0a1b2c3d4e5f"
out["masked"] = hasattr(ap, "get_visible_devices_env_var")  # older torch reads CUDA_VISIBLE_DEVICES directly
if out["masked"]:
    ap.get_visible_devices_env_var = lambda *a, **k: "UNSLOTH_TEST_VISIBLE_DEVICES"
try:
    out["devices"] = list(ap.TuningProcessPool.get_device_list())
except Exception as e:
    out["error"] = repr(e)
print("PROBE" + json.dumps(out))
"""


def _probe():
    env = dict(os.environ)
    env["UNSLOTH_IS_PRESENT"] = "1"
    env["PYTHONPATH"] = str(_REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("TORCHINDUCTOR_AUTOTUNE_MULTI_DEVICE", None)
    r = subprocess.run([sys.executable, "-c", _PROBE], env = env, cwd = str(_REPO),
                       capture_output = True, text = True, timeout = 600)
    lines = [l for l in r.stdout.splitlines() if l.startswith("PROBE")]
    assert lines, r.stdout[-2000:] + r.stderr[-4000:]
    return json.loads(lines[-1][len("PROBE"):])


def test_autotune_multi_device_is_torch_default_and_uuid_mask_does_not_raise():
    out = _probe()
    assert pathlib.Path(out["file"]).resolve().is_relative_to(_REPO), out["file"]
    assert out["value"] == out["default"], out
    if out["masked"]:
        assert "error" not in out, out["error"]
        assert out["devices"] == [None], out


def test_patching_utils_does_not_force_autotune_multi_device():
    src = (_REPO / "unsloth_zoo" / "patching_utils.py").read_text()
    assert "config.autotune_multi_device = True" not in src
