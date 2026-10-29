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

"""The compiled modeling module and LoRA forwards follow UNSLOTH_FORCE_FLOAT32 in one process.

An fp16 Gemma 3 load sets the flag to "1" and generates
unsloth_compiled_module_gemma3 with the forced float32 MLP and RMSNorm. A later
bf16 load in the same process (flag "0") used to return early on
__UNSLOTH_PATCHED__ and keep running that fp16 module, and patch_lora_forwards
kept the forced float32 LoRA forward. Each scenario mirrors the
loader (pre_compile patches, compile, post_compile patches) in its own process.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = r"""
import inspect, json, os, sys
import transformers
from unsloth_zoo.temporary_patches import TEMPORARY_PATCHES
import unsloth_zoo.compiler as compiler

def run_patches(phase):
    for patch in TEMPORARY_PATCHES:
        try:
            if "phase" in inspect.signature(patch).parameters: patch(phase = phase)
            else: patch()
        except Exception: pass

import peft.tuners.lora.layer
path = os.path.join(os.environ["UNSLOTH_COMPILE_LOCATION"], "unsloth_compiled_module_gemma3.py")
lora_path = os.path.join(os.environ["UNSLOTH_COMPILE_LOCATION"], "Linear_peft_forward.py")
passes = []
for flag in json.loads(sys.argv[1]):
    os.environ["UNSLOTH_FORCE_FLOAT32"] = flag
    run_patches("pre_compile")
    compiler.unsloth_compile_transformers("gemma3")
    run_patches("post_compile")
    mlp = transformers.models.gemma3.modeling_gemma3.Gemma3MLP
    with open(path, encoding = "utf-8") as f: source = f.read()
    with open(lora_path, encoding = "utf-8") as f: lora_source = f.read()
    lora = peft.tuners.lora.layer.Linear.forward
    passes.append({"source": source, "mlp_module": mlp.__module__, "mlp_id": id(mlp),
                   "lora_source": lora_source, "lora_id": id(lora),
                   "lora_flag": getattr(lora, "_unsloth_force_float32", None)})
print("RESULT " + json.dumps(passes))
"""


def _run(flags, cache):
    env = dict(
        os.environ, CUDA_VISIBLE_DEVICES = "", UNSLOTH_IS_PRESENT = "1",
        UNSLOTH_ZOO_DISABLE_GPU_INIT = "1", UNSLOTH_COMPILE_LOCATION = str(cache),
    )
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT), os.environ.get("PYTHONPATH")) if p)
    for key in ("UNSLOTH_FORCE_FLOAT32", "UNSLOTH_FORCE_CUSTOM_DTYPE", "UNSLOTH_COMPILE_OVERWRITE"):
        env.pop(key, None)
    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT, json.dumps(flags)],
        capture_output = True, text = True, timeout = 900, env = env,
    )
    lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-2000:] + proc.stderr[-4000:]
    return json.loads(lines[-1][len("RESULT "):])


def _normalized(source):
    # The modeling-module import list order depends on the module's attribute set
    # (an unstable argsort), not on the flag; compare it as a set.
    out = []
    for line in source.splitlines():
        prefix = "from transformers.models.gemma3.modeling_gemma3 import ("
        if line.startswith(prefix):
            line = prefix + ",".join(sorted(x.strip() for x in line[len(prefix):-1].split(","))) + ")"
        out.append(line)
    return "\n".join(out)


def _forced(source):
    return "_gemma3_rms_norm_float32" in source


@pytest.fixture(scope = "module", autouse = True)
def _needs_gemma3():
    pytest.importorskip("transformers.models.gemma3.modeling_gemma3")


def test_compiled_module_follows_force_float32_in_process(tmp_path):
    fresh_bf16_pass = _run(["0"], tmp_path / "bf16")[0]
    fresh_bf16, lora_bf16 = fresh_bf16_pass["source"], fresh_bf16_pass["lora_source"]
    fresh_fp16_pass = _run(["1"], tmp_path / "fp16")[0]
    fresh_fp16, lora_fp16 = fresh_fp16_pass["source"], fresh_fp16_pass["lora_source"]
    assert _forced(fresh_fp16) and not _forced(fresh_bf16)

    p = _run(["1", "1", "0", "0", "1"], tmp_path / "mixed")
    assert all(x["mlp_module"] == "unsloth_compiled_module_gemma3" for x in p)
    # Same flag: the first pass is reused, not regenerated.
    for i in (1, 3):
        assert p[i]["mlp_id"] == p[i - 1]["mlp_id"] and p[i]["source"] == p[i - 1]["source"]
        assert p[i]["lora_id"] == p[i - 1]["lora_id"]
    # Flag flips: regenerated, and equal to a fresh process with that flag.
    for i in (2, 4):
        assert p[i]["mlp_id"] != p[i - 1]["mlp_id"] and p[i]["lora_id"] != p[i - 1]["lora_id"]
    # The LoRA forward bakes the flag in too (forced float32 variant).
    assert lora_bf16 != lora_fp16
    assert p[0]["lora_source"] == lora_fp16 and p[2]["lora_source"] == lora_bf16
    # Flipping back reuses the flag-1 forward built in pass 0 (live layers keep theirs).
    assert [x["lora_flag"] for x in p] == ["1", "1", "0", "0", "1"]
    assert _normalized(p[0]["source"]) == _normalized(fresh_fp16)
    assert _normalized(p[2]["source"]) == _normalized(fresh_bf16)
    assert _normalized(p[4]["source"]) == _normalized(fresh_fp16)


def test_compiled_cache_written_by_fp16_process_is_regenerated_for_bf16(tmp_path):
    fresh_bf16 = _run(["0"], tmp_path / "bf16")[0]["source"]
    shared = tmp_path / "shared"
    assert _forced(_run(["1"], shared)[0]["source"])
    assert _normalized(_run(["0"], shared)[0]["source"]) == _normalized(fresh_bf16)


CROSS_SCRIPT = r"""
import inspect, json, os, sys
from unsloth_zoo.temporary_patches import TEMPORARY_PATCHES
import unsloth_zoo.compiler as compiler
import peft.tuners.lora.layer

def run_patches(phase):
    for patch in TEMPORARY_PATCHES:
        try:
            if "phase" in inspect.signature(patch).parameters: patch(phase = phase)
            else: patch()
        except Exception: pass

stamps = []
for model_type, flag in json.loads(sys.argv[1]):
    os.environ["UNSLOTH_FORCE_FLOAT32"] = flag
    run_patches("pre_compile")
    compiler.unsloth_compile_transformers(model_type)
    run_patches("post_compile")
    stamps.append(getattr(peft.tuners.lora.layer.Linear.forward, "_unsloth_force_float32", None))
print("RESULT " + json.dumps(stamps))
"""


def test_lora_forwards_follow_the_flag_across_model_families(tmp_path):
    # LoRA forwards are global: llama (0), gemma3 (1), llama (0) must end on the flag-0 forward
    # even though llama's own compiled module is unchanged and returns early.
    env = dict(
        os.environ, CUDA_VISIBLE_DEVICES = "", UNSLOTH_IS_PRESENT = "1",
        UNSLOTH_ZOO_DISABLE_GPU_INIT = "1", UNSLOTH_COMPILE_LOCATION = str(tmp_path / "cache"),
    )
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT), os.environ.get("PYTHONPATH")) if p)
    for key in ("UNSLOTH_FORCE_FLOAT32", "UNSLOTH_FORCE_CUSTOM_DTYPE", "UNSLOTH_COMPILE_OVERWRITE"):
        env.pop(key, None)
    sequence = [["llama", "0"], ["gemma3", "1"], ["llama", "0"]]
    proc = subprocess.run(
        [sys.executable, "-c", CROSS_SCRIPT, json.dumps(sequence)],
        capture_output = True, text = True, timeout = 900, env = env,
    )
    lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-2000:] + proc.stderr[-4000:]
    assert json.loads(lines[-1][len("RESULT "):]) == ["0", "1", "0"]


RETAINED_SCRIPT = r"""
import inspect, json, os, sys
import torch
from unsloth_zoo.temporary_patches import TEMPORARY_PATCHES
import unsloth_zoo.compiler as compiler
from peft import LoraConfig
from peft.tuners.lora.layer import Linear

def run_patches(phase):
    for patch in TEMPORARY_PATCHES:
        try:
            if "phase" in inspect.signature(patch).parameters: patch(phase = phase)
            else: patch()
        except Exception: pass

def compile(model_type, flag):
    os.environ["UNSLOTH_FORCE_FLOAT32"] = flag
    run_patches("pre_compile")
    compiler.unsloth_compile_transformers(model_type)
    run_patches("post_compile")

def layer():
    base = torch.nn.Linear(1, 1, bias = False, dtype = torch.bfloat16)
    lora = Linear(base, "default", config = LoraConfig(r = 1, lora_alpha = 1), r = 1, lora_alpha = 1)
    for p in lora.parameters(): p.data.fill_(1)
    return lora.to(torch.bfloat16)

x = torch.tensor([[100000.0]], dtype = torch.bfloat16)
out = {}
with torch.no_grad():
    compile("llama", "0")
    kept = layer()
    out["kept_before"] = kept(x).item()
    created_unused = layer()
    compile("gemma3", "1")
    out["kept_after"] = kept(x).item()
    out["created_unused_after"] = created_unused(x).item()
    out["new_after"] = layer()(x).item()
    compile("llama", "0")
    out["kept_back"] = kept(x).item()
    out["new_back"] = layer()(x).item()
print("RESULT " + json.dumps({k: str(v) for k, v in out.items()}))
"""


def test_lora_layers_keep_their_flag_when_a_later_load_flips_it(tmp_path):
    # A bf16 model kept alive while an fp16 forced float32 model loads must not change: its LoRA
    # layers keep the flag-0 forward (199680), layers created under flag 1 take the float16 one (inf).
    env = dict(
        os.environ, CUDA_VISIBLE_DEVICES = "", UNSLOTH_IS_PRESENT = "1",
        UNSLOTH_ZOO_DISABLE_GPU_INIT = "1", UNSLOTH_COMPILE_LOCATION = str(tmp_path / "cache"),
    )
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT), os.environ.get("PYTHONPATH")) if p)
    for key in ("UNSLOTH_FORCE_FLOAT32", "UNSLOTH_FORCE_CUSTOM_DTYPE", "UNSLOTH_COMPILE_OVERWRITE"):
        env.pop(key, None)
    proc = subprocess.run(
        [sys.executable, "-c", RETAINED_SCRIPT],
        capture_output = True, text = True, timeout = 900, env = env,
    )
    lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-2000:] + proc.stderr[-4000:]
    assert json.loads(lines[-1][len("RESULT "):]) == {
        "kept_before": "199680.0", "kept_after": "199680.0", "created_unused_after": "199680.0",
        "new_after": "inf", "kept_back": "199680.0", "new_back": "199680.0",
    }
