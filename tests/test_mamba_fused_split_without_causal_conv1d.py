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

"""patch_mamba_fused_split_without_causal_conv1d: split path only when causal_conv1d is unusable, patch before or after
the modeling import. Subprocess per case: stub mamba_ssm failing like the real one, tiny Falcon-H1 on CPU."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

transformers = pytest.importorskip("transformers")
from transformers.integrations import hub_kernels as _hk  # noqa: E402

if not hasattr(_hk, "use_kernel_func_from_hub_with_fallback"):
    pytest.skip("transformers < 5 gates Mamba kernels with is_fast_path_available", allow_module_level = True)
try:
    from transformers import FalconH1Config  # noqa: F401
except ImportError:
    pytest.skip("transformers without Falcon-H1", allow_module_level = True)

MISC = Path(__file__).resolve().parents[1] / "unsloth_zoo" / "temporary_patches" / "misc.py"

STUB_SSD = '''
try:
    from causal_conv1d.cpp_functions import causal_conv1d_fwd_function
except ImportError:
    causal_conv1d_fwd_function = None
def mamba_split_conv1d_scan_combined(*args, **kwargs):
    # mamba_ssm/ops/triton/ssd_combined.py calls this unconditionally
    return causal_conv1d_fwd_function(*args)
'''

# mamba_ssm <= 2.2.x with causal_conv1d < 1.5 (no cpp_functions)
STUB_SSD_LEGACY = '''
try:
    from causal_conv1d import causal_conv1d_fn
    import causal_conv1d_cuda
except ImportError:
    causal_conv1d_fn, causal_conv1d_cuda = None, None
def mamba_split_conv1d_scan_combined(*args, **kwargs):
    return causal_conv1d_cuda.causal_conv1d_fwd(*args)
'''

STUB_CONV = {
    "__init__.py": "causal_conv1d_fn = lambda *a, **k: None\ncausal_conv1d_update = None\n",
    "cpp_functions.py": "def causal_conv1d_fwd_function(*a, **k):\n    raise RuntimeError('FUSED_KERNEL_CALLED')\n",
}

WORKER = r'''
import ast, functools, importlib, json, logging, os, sys
case, misc = sys.argv[1], sys.argv[2]
if not case.endswith("conv_ok"):
    sys.modules["causal_conv1d"] = None
import torch

src = open(misc, encoding = "utf-8").read()
ns = {"torch": torch, "importlib": importlib, "functools": functools, "os": os,
      "logger": logging.getLogger("t")}
wanted = ("_mamba_fused_split_needs_causal_conv1d_unusable",
          "patch_mamba_fused_split_without_causal_conv1d")
for node in ast.parse(src).body:
    if isinstance(node, ast.FunctionDef) and node.name in wanted:
        exec(ast.get_source_segment(src, node), ns)
patch = ns.get("patch_mamba_fused_split_without_causal_conv1d")

if case in ("patch_first", "conv_ok", "legacy_conv_ok", "legacy_patch_first"):
    patch()
import transformers.models.falcon_h1.modeling_falcon_h1 as mf
if case == "patch_after":
    patch()
from transformers import FalconH1Config
cfg = FalconH1Config(vocab_size = 128, hidden_size = 64, intermediate_size = 128,
    num_hidden_layers = 1, num_attention_heads = 2, num_key_value_heads = 1, head_dim = 32,
    mamba_d_ssm = 64, mamba_n_heads = 4, mamba_d_head = 16, mamba_d_state = 16,
    mamba_n_groups = 1, mamba_chunk_size = 16, mamba_d_conv = 4)
torch.manual_seed(0)
model = mf.FalconH1ForCausalLM(cfg).train()
ids = torch.randint(0, 128, (1, 24))
try:
    out = model(input_ids = ids, labels = ids, use_cache = False)
    out.loss.backward()
    res = {"ok": True, "loss": float(out.loss)}
except Exception as e:
    res = {"ok": False, "err": f"{type(e).__name__}: {e}"}
print("RESULT " + json.dumps(res))
'''


def _run(case, tmp_path):
    root = tmp_path / "site"
    ssd = root / "mamba_ssm" / "ops" / "triton"
    ssd.mkdir(parents = True)
    for d in (root / "mamba_ssm", root / "mamba_ssm" / "ops", ssd):
        (d / "__init__.py").write_text("")
    legacy = case.startswith("legacy_")
    (ssd / "ssd_combined.py").write_text(STUB_SSD_LEGACY if legacy else STUB_SSD)
    if case.endswith("conv_ok"):
        conv = root / "causal_conv1d"
        conv.mkdir()
        for name, body in STUB_CONV.items():
            if not (legacy and name == "cpp_functions.py"):
                (conv / name).write_text(body)
        if legacy:
            (root / "causal_conv1d_cuda.py").write_text(
                "def causal_conv1d_fwd(*a, **k):\n    raise RuntimeError('FUSED_KERNEL_CALLED')\n")
    worker = tmp_path / "worker.py"
    worker.write_text(WORKER)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["USE_HUB_KERNELS"] = "0"
    proc = subprocess.run([sys.executable, str(worker), case, str(MISC)],
                          capture_output = True, text = True, env = env, timeout = 600)
    lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert lines, proc.stdout[-2000:] + proc.stderr[-4000:]
    return json.loads(lines[-1][len("RESULT "):])


def test_unpatched_train_forward_hits_the_fused_kernel_and_fails(tmp_path):
    res = _run("no_patch", tmp_path)
    assert not res["ok"]
    assert "NoneType" in res["err"]


@pytest.mark.parametrize("case", ["patch_first", "patch_after", "legacy_patch_first"])
def test_patched_train_forward_takes_the_split_path(case, tmp_path):
    res = _run(case, tmp_path)
    assert res["ok"], res
    assert res["loss"] == res["loss"]  # finite, not NaN


@pytest.mark.parametrize("case", ["conv_ok", "legacy_conv_ok"])
def test_usable_causal_conv1d_keeps_the_fused_kernel(case, tmp_path):
    res = _run(case, tmp_path)
    assert not res["ok"]
    assert "FUSED_KERNEL_CALLED" in res["err"]
