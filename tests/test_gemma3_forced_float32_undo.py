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

"""Gemma3 forced float32 patches must not outlive the load that installed them.

The loader sets UNSLOTH_FORCE_FLOAT32 per load ("1" for an fp16 Gemma3 load, "0"
for bf16) and reruns every temporary patch (pre_compile, post_compile). The
generic set only replaces RMSNorm and attention, so an fp16 load followed by a
bf16 load in one process kept the float32 embedding and the forced MLP, and the
bf16 attention raised "Expected query, key, and value to have the same dtype".

Each scenario runs in its own process because the patches are process global.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = r"""
import json, os, sys
import torch
import transformers
from transformers import Gemma3ForCausalLM, Gemma3TextConfig
import unsloth_zoo.temporary_patches.gemma as gemma_patches
from unsloth_zoo.temporary_patches.common import TEMPORARY_PATCHES

flags, dtype, attn = json.loads(sys.argv[1]), getattr(torch, sys.argv[2]), sys.argv[3]
patches = [p for p in TEMPORARY_PATCHES if getattr(p, "__module__", "") == gemma_patches.__name__]
for flag in flags:
    os.environ["UNSLOTH_FORCE_FLOAT32"] = flag
    for _phase in ("pre_compile", "post_compile"):
        for patch in patches:
            patch()

modeling = transformers.models.gemma3.modeling_gemma3
def ident(f):
    f = getattr(f, "_torchdynamo_orig_callable", f)
    while hasattr(f, "__wrapped__"): f = f.__wrapped__
    return f.__qualname__ + "@" + str(f.__code__.co_firstlineno)
installed = {
    name: ident(getattr(getattr(modeling, name), attr))
    for name, attr in (
        ("Gemma3TextScaledWordEmbedding", "forward"), ("Gemma3MLP", "forward"),
        ("Gemma3RMSNorm", "forward"), ("Gemma3Attention", "forward"),
        ("Gemma3MultiModalProjector", "forward"),
    )
    if hasattr(modeling, name)
}

config = Gemma3TextConfig(
    vocab_size = 128, hidden_size = 32, intermediate_size = 64, num_hidden_layers = 2,
    num_attention_heads = 2, num_key_value_heads = 1, head_dim = 16, sliding_window = 8,
    max_position_embeddings = 64,
)
config._attn_implementation = attn
torch.manual_seed(0)
model = Gemma3ForCausalLM(config).to(dtype)
ids = torch.randint(0, config.vocab_size, (2, 12))
result = {"installed": installed}
try:
    out = model(input_ids = ids, labels = ids)
    out.loss.backward()
    result["loss"] = float(out.loss)
    result["grads"] = [float(p.grad.double().abs().sum()) for p in model.parameters() if p.grad is not None]
except Exception as exception:
    result["error"] = f"{type(exception).__name__}: {exception}"
print("RESULT " + json.dumps(result))
"""


def _run(flags, dtype, attn = "sdpa"):
    env = dict(
        os.environ, CUDA_VISIBLE_DEVICES = "", UNSLOTH_IS_PRESENT = "1",
        UNSLOTH_ZOO_DISABLE_GPU_INIT = "1", TORCHDYNAMO_DISABLE = "1",
    )
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT), os.environ.get("PYTHONPATH")) if p)
    env.pop("UNSLOTH_FORCE_FLOAT32", None)
    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT, json.dumps(flags), dtype, attn],
        capture_output = True, text = True, timeout = 900, env = env,
    )
    lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-2000:] + proc.stderr[-4000:]
    return json.loads(lines[-1][len("RESULT "):])


@pytest.fixture(scope = "module", autouse = True)
def _needs_gemma3():
    pytest.importorskip("transformers.models.gemma3.modeling_gemma3")


@pytest.mark.parametrize("attn", ["sdpa", "eager"])
@pytest.mark.parametrize("fp16_passes", [1, 2])
def test_fp16_then_bf16_matches_fresh_bf16(attn, fp16_passes):
    fresh = _run(["0"], "bfloat16", attn)
    assert "error" not in fresh, fresh
    after = _run(["1"] * fp16_passes + ["0"], "bfloat16", attn)
    assert "error" not in after, after["error"]
    assert after["installed"] == fresh["installed"]
    assert after["loss"] == fresh["loss"]
    assert after["grads"] == fresh["grads"]


def test_bf16_then_fp16_still_installs_forced_set():
    fresh = _run(["1"], "float16")
    assert "error" not in fresh, fresh
    forced = fresh["installed"]
    assert "patch_Gemma3TextScaledWordEmbedding" in forced["Gemma3TextScaledWordEmbedding"]
    assert "patch_Gemma3MLP" in forced["Gemma3MLP"]
    for flags in (["0", "1"], ["1", "0", "1"]):
        after = _run(flags, "float16")
        assert "error" not in after, after["error"]
        assert after["installed"] == forced, flags
        assert after["loss"] == fresh["loss"]
        assert after["grads"] == fresh["grads"]
