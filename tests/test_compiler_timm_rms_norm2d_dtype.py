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

"""RmsNorm2d / RmsNormAct2d keep the input dtype after compile_timm_models (Gemma 3N fp16 on T4).
Runs in a fresh interpreter since the compiler mutates timm."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("unsloth_zoo.compiler")
norm_act = pytest.importorskip("timm.layers.norm_act")
if not hasattr(norm_act, "RmsNormAct2d"):
    pytest.skip("timm without RmsNormAct2d", allow_module_level = True)

ROOT = Path(__file__).resolve().parents[1]


def test_timm_rms_norm2d_modules_keep_input_dtype(tmp_path):
    script = textwrap.dedent(
        f"""
        import json, sys, torch
        sys.path.insert(0, {str(ROOT)!r})
        import unsloth_zoo.compiler as compiler
        import unsloth_zoo
        assert unsloth_zoo.__file__.startswith({str(ROOT)!r}), unsloth_zoo.__file__
        import timm.layers.norm, timm.layers.norm_act
        compiler.compile_timm_models(False, {{}})
        out = {{}}
        torch.manual_seed(0)
        for cls in (timm.layers.norm.RmsNorm2d, timm.layers.norm_act.RmsNormAct2d):
            for dtype in (torch.float16, torch.bfloat16):
                norm = cls(8)
                norm.weight.data.uniform_(0.5, 1.5)
                norm = norm.to(torch.float32)  # Unsloth pins norm weights to float32
                conv = torch.nn.Conv2d(8, 4, 1, bias = False).to(dtype)
                x = torch.randn(2, 8, 5, 5, dtype = dtype)
                key = cls.__name__ + "_" + str(dtype).split(".")[-1]
                y = norm(x)
                row = dict(out_dtype = str(y.dtype))
                try:
                    conv(y)
                    row["conv_ok"] = True
                except RuntimeError as e:
                    row["conv_ok"] = False
                    row["error"] = str(e)[:160]
                # Same math as timm's rms_norm2d in float32, rounded to the input dtype.
                xf = x.float()
                ref = xf * torch.rsqrt(xf.pow(2).mean(dim = 1, keepdim = True) + norm.eps)
                ref = (ref * norm.weight.float().reshape(1, -1, 1, 1)).to(dtype)
                if hasattr(norm, "act"):
                    ref = norm.act(norm.drop(ref))
                row["matches_fp32_reference"] = bool(torch.allclose(y.float(), ref.float(), atol = 1e-2, rtol = 1e-2))
                out[key] = row
        print("RESULT " + json.dumps(out))
        """
    )
    env = dict(os.environ, UNSLOTH_COMPILE_LOCATION = str(tmp_path / "cache"))
    env.setdefault("UNSLOTH_ALLOW_CPU", "1")
    # Eager: CPU inductor is slow and the binding is read the same way.
    env["TORCHDYNAMO_DISABLE"] = "1"
    proc = subprocess.run([sys.executable, "-c", script], capture_output = True, text = True, timeout = 900, env = env)
    line = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert line, proc.stdout[-2000:] + proc.stderr[-3000:]
    out = json.loads(line[-1][len("RESULT "):])
    assert len(out) == 4, out
    for key, row in out.items():
        expected = "torch." + key.rsplit("_", 1)[-1]
        assert row["out_dtype"] == expected, (key, row)
        assert row["conv_ok"], (key, row)
        assert row["matches_fp32_reference"], (key, row)
