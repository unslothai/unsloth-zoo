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
"""Qwen3-MoE and Gemma 4 forced float32 patches must not outlive the load that installed them.

Same bug class as Gemma 3 (test_gemma3_forced_float32_undo.py): the loader sets UNSLOTH_FORCE_FLOAT32
per load and reruns every temporary patch, but these patches only returned early on "0". An fp16 load
followed by a bf16 load kept the forced forwards, and the bf16 model raised "expected m1 and m2 to have
the same dtype, but got: c10::Half != c10::BFloat16". Each scenario runs in its own process because the
patches are process global.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

FAMILIES = {
    "qwen3_moe": dict(
        patches = "unsloth_zoo.temporary_patches.qwen3_moe_float32",
        modeling = "transformers.models.qwen3_moe.modeling_qwen3_moe",
        classes = ["Qwen3MoeDecoderLayer", "Qwen3MoeRMSNorm", "Qwen3MoeAttention"],
        model_type = "qwen3_moe",
        config = dict(vocab_size = 128, hidden_size = 32, intermediate_size = 64, moe_intermediate_size = 16,
                      num_hidden_layers = 2, num_attention_heads = 2, num_key_value_heads = 1, head_dim = 16,
                      num_experts = 4, num_experts_per_tok = 2, max_position_embeddings = 64),
        name = "qwen3_moe,",
    ),
    "gemma4": dict(
        patches = "unsloth_zoo.temporary_patches.gemma4_float32",
        modeling = "transformers.models.gemma4.modeling_gemma4",
        classes = ["Gemma4TextScaledWordEmbedding", "Gemma4RMSNorm", "Gemma4ClippableLinear", "Gemma4TextAttention"],
        model_type = "gemma4_text",
        config = dict(vocab_size = 128, hidden_size = 32, intermediate_size = 64, num_hidden_layers = 2,
                      num_attention_heads = 2, num_key_value_heads = 1, head_dim = 16, max_position_embeddings = 64),
        name = "gemma4,gemma4_text,",
    ),
}

SCRIPT = r"""
import importlib, json, os, sys
import torch
import transformers
from unsloth_zoo.temporary_patches.common import TEMPORARY_PATCHES

fam, loads, dtype = json.loads(sys.argv[1]), json.loads(sys.argv[2]), getattr(torch, sys.argv[3])
patch_module = importlib.import_module(fam["patches"])
modeling = importlib.import_module(fam["modeling"])
patches = [p for p in TEMPORARY_PATCHES if getattr(p, "__module__", "") == patch_module.__name__]

def ident(f):
    f = getattr(f, "_torchdynamo_orig_callable", f)
    while hasattr(f, "__wrapped__"): f = f.__wrapped__
    return f.__qualname__ + "@" + str(f.__code__.co_firstlineno)

def installed():
    return {n: ident(getattr(modeling, n).forward) for n in fam["classes"]}

def load(name, flag):
    os.environ["UNSLOTH_MODEL_NAME"] = name
    os.environ["UNSLOTH_FORCE_FLOAT32"] = flag
    for _phase in ("pre_compile", "post_compile"):
        for patch in patches:
            patch()

config = transformers.AutoConfig.for_model(fam["model_type"], **fam["config"])
config._attn_implementation = "sdpa"
torch.manual_seed(0)
model = transformers.AutoModelForCausalLM.from_config(config).to(dtype)
ids = torch.randint(0, config.vocab_size, (2, 12))

def step():
    model.zero_grad(set_to_none = True)
    out = model(input_ids = ids, labels = ids)
    out.loss.backward()
    return float(out.loss), [float(p.grad.double().abs().sum()) for p in model.parameters() if p.grad is not None]

result = {"steps": []}
for name, flag, measure in loads:
    load(name, flag)
    if measure:
        try:
            loss, grads = step()
            result["steps"].append({"installed": installed(), "loss": loss, "grads": grads})
        except Exception as exception:
            result["steps"].append({"installed": installed(), "error": f"{type(exception).__name__}: {exception}"})
print("RESULT " + json.dumps(result))
"""


def _run(family, loads, dtype):
    env = dict(
        os.environ, CUDA_VISIBLE_DEVICES = "", UNSLOTH_IS_PRESENT = "1",
        UNSLOTH_ZOO_DISABLE_GPU_INIT = "1", TORCHDYNAMO_DISABLE = "1",
    )
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT), os.environ.get("PYTHONPATH")) if p)
    for key in ("UNSLOTH_FORCE_FLOAT32", "UNSLOTH_MODEL_NAME"):
        env.pop(key, None)
    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT, json.dumps(FAMILIES[family]), json.dumps(loads), dtype],
        capture_output = True, text = True, timeout = 900, env = env,
    )
    lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-2000:] + proc.stderr[-4000:]
    return json.loads(lines[-1][len("RESULT "):])["steps"]


def _needs(family):
    pytest.importorskip(FAMILIES[family]["modeling"])


@pytest.mark.parametrize("family", list(FAMILIES))
@pytest.mark.parametrize("fp16_passes", [1, 2])
def test_fp16_then_bf16_matches_fresh_bf16(family, fp16_passes):
    _needs(family)
    name = FAMILIES[family]["name"]
    fresh = _run(family, [[name, "0", True]], "bfloat16")[0]
    assert "error" not in fresh, fresh
    after = _run(family, [[name, "1", False]] * fp16_passes + [[name, "0", True]], "bfloat16")[0]
    assert "error" not in after, after["error"]
    assert after == fresh


@pytest.mark.parametrize("family", list(FAMILIES))
def test_bf16_then_fp16_still_installs_forced_set(family):
    _needs(family)
    name = FAMILIES[family]["name"]
    fresh = _run(family, [[name, "1", True]], "float16")[0]
    assert "error" not in fresh, fresh
    clean = _run(family, [[name, "0", True]], "bfloat16")[0]["installed"]
    assert all(fresh["installed"][n] != clean[n] for n in FAMILIES[family]["classes"]), fresh["installed"]
    for loads in ([[name, "0", False], [name, "1", True]], [[name, "1", False], [name, "0", False], [name, "1", True]]):
        after = _run(family, loads, "float16")[0]
        assert after == fresh, loads


@pytest.mark.parametrize("family", list(FAMILIES))
def test_another_family_load_keeps_a_live_fp16_model_patched(family):
    # A bf16 load of another family must not unpatch a live fp16 model of this one.
    _needs(family)
    name = FAMILIES[family]["name"]
    steps = _run(family, [[name, "1", True], ["llama,_load_in_4bit_", "0", True]], "float16")
    assert "error" not in steps[0], steps[0]
    assert steps[1] == steps[0]
