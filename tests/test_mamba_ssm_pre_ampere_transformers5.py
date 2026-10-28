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

"""patch_mamba_ssm_pre_ampere_fallback on transformers 5.15+, which binds mamba_ssm in
`use_kernel_func_from_hub_with_fallback` at modeling import (unsloth-zoo#2090). Subprocess per case:
stub mamba_ssm whose kernels raise, tiny Granite-4.0-H and Falcon-H1 trained one step on CPU."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

transformers = pytest.importorskip("transformers")
from transformers.integrations import hub_kernels as _hk  # noqa: E402

if not hasattr(_hk, "use_kernel_func_from_hub_with_fallback"):
    pytest.skip("transformers < 5.15 gates Mamba kernels with is_fast_path_available", allow_module_level = True)
try:
    from transformers import FalconH1Config, GraniteMoeHybridConfig  # noqa: F401
except ImportError:
    pytest.skip("transformers without Falcon-H1 / Granite-4.0-H", allow_module_level = True)

MISC = Path(__file__).resolve().parents[1] / "unsloth_zoo" / "temporary_patches" / "misc.py"

STUB_SSD = '''
def _kernel(*args, **kwargs):
    raise RuntimeError("KERNEL_CALLED")
mamba_chunk_scan_combined = _kernel
mamba_split_conv1d_scan_combined = _kernel
'''

WORKER = r'''
import ast, functools, importlib, json, logging, os, sys
case, misc = sys.argv[1], sys.argv[2]
import torch
import transformers.integrations.hub_kernels as hk

src = open(misc, encoding = "utf-8").read()
ns = {"torch": torch, "os": os, "importlib": importlib, "functools": functools,
      "logger": logging.getLogger("t")}
prefixes = ("_MAMBA", "_weakest", "_mamba_ssm_fast", "_single_mamba", "_retrim", "_resolves_to",
            "_hub_func", "_bind_mamba", "_force_slow", "patch_mamba_ssm_pre_ampere")
for node in ast.parse(src).body:
    names = [node.name] if isinstance(node, ast.FunctionDef) else [
        t.id for t in getattr(node, "targets", []) if isinstance(t, ast.Name)]
    if any(n.startswith(prefixes) for n in names):
        exec(ast.get_source_segment(src, node), ns)

def patch():
    saved = (torch.cuda.is_available, torch.cuda.device_count, torch.cuda.get_device_capability)
    torch.cuda.is_available = lambda: True
    torch.cuda.device_count = lambda: 1
    torch.cuda.get_device_capability = lambda device = 0: (7, 5)
    try:
        ns["patch_mamba_ssm_pre_ampere_fallback"]()
    finally:
        torch.cuda.is_available, torch.cuda.device_count, torch.cuda.get_device_capability = saved

if case != "patch_after":
    for _ in range(3):
        patch()
import transformers.models.granitemoehybrid.modeling_granitemoehybrid as mg
import transformers.models.falcon_h1.modeling_falcon_h1 as mf
if case == "patch_after":
    patch()

wrappers, f = 0, hk.use_kernel_func_from_hub_with_fallback
while f is not None:
    wrappers += bool(getattr(f, "_unsloth_mamba_ssm_torch", False))
    f = getattr(f, "__wrapped__", None)

from transformers import FalconH1Config, GraniteMoeHybridConfig
common = dict(vocab_size = 128, hidden_size = 64, intermediate_size = 128, num_attention_heads = 2,
              num_key_value_heads = 1, mamba_d_state = 16, mamba_n_groups = 1, mamba_chunk_size = 16,
              mamba_d_conv = 4)
models = {
    "granite": lambda: mg.GraniteMoeHybridForCausalLM(GraniteMoeHybridConfig(
        **common, shared_intermediate_size = 128, num_hidden_layers = 2, layer_types = ["mamba", "attention"],
        mamba_n_heads = 4, mamba_d_head = 16, mamba_expand = 1, num_local_experts = 0)),
    "falcon_h1": lambda: mf.FalconH1ForCausalLM(FalconH1Config(
        **common, num_hidden_layers = 1, head_dim = 32, mamba_d_ssm = 64, mamba_n_heads = 4, mamba_d_head = 16)),
}
res = {"wrappers": wrappers}
for name, build in models.items():
    torch.manual_seed(0)
    model = build().train()
    ids = torch.randint(0, 128, (1, 24))
    try:
        out = model(input_ids = ids, labels = ids, use_cache = False)
        out.loss.backward()
        res[name] = "ok"
    except Exception as e:
        res[name] = f"{type(e).__name__}: {e}"
print("RESULT " + json.dumps(res))
'''


def _run(case, forced, tmp_path):
    root = tmp_path / "site"
    ssd = root / "mamba_ssm" / "ops" / "triton"
    ssd.mkdir(parents = True)
    for d in (root / "mamba_ssm" / "ops", ssd):
        (d / "__init__.py").write_text("")
    (root / "mamba_ssm" / "__init__.py").write_text("import mamba_ssm.ops.triton.ssd_combined\n")
    (ssd / "ssd_combined.py").write_text(STUB_SSD)
    worker = tmp_path / "worker.py"
    worker.write_text(WORKER)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["USE_HUB_KERNELS"] = "0"
    env["UNSLOTH_MAMBA_PRE_AMPERE_FAST"] = forced
    # causal_conv1d stays out of reach, so only mamba_ssm decides the path.
    env["PYTHONNOUSERSITE"] = "1"
    proc = subprocess.run([sys.executable, str(worker), case, str(MISC)],
                          capture_output = True, text = True, env = env, timeout = 600)
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    assert lines, proc.stdout[-2000:] + proc.stderr[-4000:]
    return json.loads(lines[-1][len("RESULT "):])


@pytest.mark.parametrize("case", ["patch_first", "patch_after"])
def test_forced_slow_path_never_reaches_mamba_ssm(case, tmp_path):
    res = _run(case, "0", tmp_path)
    assert res["granite"] == "ok", res
    assert res["falcon_h1"] == "ok", res
    assert res["wrappers"] == 1, res


def test_fast_path_still_binds_mamba_ssm(tmp_path):
    """Negative control: with the fast path kept, the stub kernel is what runs."""
    res = _run("patch_first", "1", tmp_path)
    assert "KERNEL_CALLED" in res["granite"], res
    assert "KERNEL_CALLED" in res["falcon_h1"], res
    assert res["wrappers"] == 0, res
