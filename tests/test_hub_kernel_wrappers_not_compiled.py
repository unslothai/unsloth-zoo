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

"""Hub-kernel wrapper functions must be emitted bare, not torch.compile'd."""

import json
import os
import subprocess
import sys

import importlib.util
from pathlib import Path
import pytest


def test_detector():
    from unsloth_zoo.compiler import is_hub_kernel_wrapper
    src = (
        '@use_kernel_func_from_hub_with_fallback("mamba_split_conv1d_scan_combined", "mamba_ssm")\n'
        "def mamba2_split_conv1d_scan_combined(zxbcdt, conv1d_weight):\n"
        "    return None\n"
    )
    assert is_hub_kernel_wrapper(src)
    assert is_hub_kernel_wrapper("    " + src.replace("\n", "\n    "))
    assert not is_hub_kernel_wrapper("def rms(x):\n    return x * x.pow(2).mean(-1, keepdim=True).rsqrt()\n")
    assert not is_hub_kernel_wrapper('@use_kernel_forward_from_hub("RMSNorm")\nclass N: pass\n')
    assert not is_hub_kernel_wrapper(None)


_CHILD = r'''
import os, json, io, contextlib, importlib
os.environ["UNSLOTH_COMPILE_LOCATION"] = "unsloth_compiled_cache"
import torch
from unsloth_zoo.compiler import unsloth_compile_transformers
MT = os.environ["UNSLOTH_TEST_MODEL_TYPE"]
modeling = importlib.import_module(f"transformers.models.{MT}.modeling_{MT}")
hub = [n for n in ("causal_conv1d_update", "causal_conv1d_fn", "mamba2_split_conv1d_scan_combined",
                   "mamba2_selective_state_update", "mamba2_chunk_scan",
                   "torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule") if hasattr(modeling, n)]
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    unsloth_compile_transformers(model_type = MT, fast_lora_forwards = False, fullgraph = True,
                                 import_from_cache = False, disable = False, supports_sdpa = [None])
generated = open(os.path.join("unsloth_compiled_cache", f"unsloth_compiled_module_{MT}.py"), encoding = "utf-8").read()
print("@@@" + json.dumps({"generated": generated, "hub": hub, "log": buf.getvalue()[-6000:]}))
'''


def _decorators_of(generated, name):
    where = generated.find(f"\ndef {name}(")
    assert where != -1, f"{name} is not in the generated module"
    head = generated[:where].rstrip("\n").split("\n")
    decs = []
    for line in reversed(head):
        if line.startswith("@"):
            decs.append(line)
        else:
            break
    return decs


def _compile(tmp_path, model_type):
    pytest.importorskip(f"transformers.models.{model_type}.modeling_{model_type}")
    env = dict(os.environ)
    env["UNSLOTH_ALLOW_CPU"] = "1"
    env["UNSLOTH_ZOO_DISABLE_GPU_INIT"] = "1"
    env["UNSLOTH_TEST_MODEL_TYPE"] = model_type
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join([repo_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.pop("UNSLOTH_COMPILE_DISABLE", None)
    r = subprocess.run([sys.executable, "-c", _CHILD], cwd = str(tmp_path), env = env,
                       capture_output = True, text = True, timeout = 1800)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    return json.loads(next(l[3:] for l in r.stdout.splitlines() if l.startswith("@@@")))


def _uses_hub_kernel_fallback(model_type):
    """Whether this transformers wraps any of the model's functions in a hub-kernel fallback."""
    try:
        spec = importlib.util.find_spec(f"transformers.models.{model_type}.modeling_{model_type}")
    except ModuleNotFoundError:  # the model is newer than this transformers
        return False
    if spec is None or spec.origin is None:
        return False
    return "use_kernel_func_from_hub_with_fallback" in Path(spec.origin).read_text(encoding = "utf-8")


def test_nemotron_h_hub_kernel_functions_are_not_compiled(tmp_path):
    # transformers before the hub-kernel wrappers (5.5) has nothing to keep uncompiled. Keyed on
    # its own source, not on `hub` being empty, so a broken discovery still fails on 5.17+.
    if not _uses_hub_kernel_fallback("nemotron_h"):
        pytest.skip("nemotron_h has no hub-kernel wrappers in this transformers")
    payload = _compile(tmp_path, "nemotron_h")
    generated, hub = payload["generated"], payload["hub"]
    assert "mamba2_split_conv1d_scan_combined" in hub
    checked = 0
    for name in hub:
        if f"\ndef {name}(" not in generated:
            continue
        decs = _decorators_of(generated, name)
        assert any("use_kernel_func_from_hub_with_fallback" in d for d in decs), (name, decs)
        assert not any("torch_compile" in d or "torch.compile(" in d for d in decs), (
            f"{name} is torch.compile'd on top of its hub-kernel wrapper: {decs}\n" + payload["log"]
        )
        checked += 1
    assert checked >= 1, "no hub-kernel wrapper reached the generated module, so the test proves nothing"


def test_disable_listed_hub_wrappers_keep_compiler_disable(tmp_path):
    payload = _compile(tmp_path, "qwen3_next")
    generated = payload["generated"]
    names = [n for n in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule")
             if n in payload["hub"] and f"\ndef {n}(" in generated]
    if not names:
        pytest.skip("qwen3_next has no hub-wrapped gated delta rule in this transformers")
    for name in names:
        decs = _decorators_of(generated, name)
        if not any("use_kernel_func_from_hub_with_fallback" in d for d in decs):
            continue
        assert any("torch_compiler_disable_unless_decode" in d for d in decs), (name, decs)
