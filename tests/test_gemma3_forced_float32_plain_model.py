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

"""Plain bf16/fp32 Gemma 3 keeps dtype after fp16 forced-float32 load; nothing reruns shared-class
patches on plain loads, so final RMSNorm must not emit fp16 (inference, generate crashed)."""

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

flags = json.loads(sys.argv[1])
dtype = getattr(torch, sys.argv[2])
patches = [p for p in TEMPORARY_PATCHES if getattr(p, "__module__", "") == gemma_patches.__name__]
for flag in flags:
    os.environ["UNSLOTH_FORCE_FLOAT32"] = flag
    for _phase in ("pre_compile", "post_compile"):
        for patch in patches:
            patch()
modeling = transformers.models.gemma3.modeling_gemma3

config = Gemma3TextConfig(
    vocab_size = 128, hidden_size = 32, intermediate_size = 64, num_hidden_layers = 2,
    num_attention_heads = 2, num_key_value_heads = 1, head_dim = 16, sliding_window = 8,
    max_position_embeddings = 64,
)
config._attn_implementation = "sdpa"
torch.manual_seed(0)
model = Gemma3ForCausalLM(config).to(dtype)
ids = torch.randint(0, config.vocab_size, (2, 12))
result = {}
try:
    with torch.no_grad():
        out = model(input_ids = ids, output_hidden_states = True)
    result["logits_dtype"] = str(out.logits.dtype)
    result["hidden_dtypes"] = sorted({str(h.dtype) for h in out.hidden_states})
    result["logits"] = out.logits.float().flatten().tolist()
    result["loss"] = float(model(input_ids = ids, labels = ids).loss)
except Exception as exception:
    result["error"] = f"{type(exception).__name__}: {exception}"

# The forced path itself is unchanged: a forced float32 load keeps norms fp16, or fp32 pinned by
# `_pre_set_compute_dtype` (UNSLOTH_HIGH_PRECISION_LAYERNORM), and both still emit fp16.
pinned = modeling.Gemma3RMSNorm(32).to(torch.float32)
pinned._pre_set_compute_dtype = torch.float32
half = modeling.Gemma3RMSNorm(32).to(torch.float16)
result["forced_norm_dtypes"] = [str(pinned(torch.randn(2, 3, 32)).dtype), str(half(torch.randn(2, 3, 32)).dtype)]
print("RESULT " + json.dumps(result))
"""


def _run(flags, dtype):
    env = dict(
        os.environ, CUDA_VISIBLE_DEVICES = "", UNSLOTH_IS_PRESENT = "1",
        UNSLOTH_ZOO_DISABLE_GPU_INIT = "1", TORCHDYNAMO_DISABLE = "1",
    )
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT), os.environ.get("PYTHONPATH")) if p)
    for key in ("UNSLOTH_FORCE_FLOAT32", "UNSLOTH_MODEL_NAME"):
        env.pop(key, None)
    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT, json.dumps(flags), dtype],
        capture_output = True, text = True, timeout = 900, env = env,
    )
    lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-2000:] + proc.stderr[-4000:]
    return json.loads(lines[-1][len("RESULT "):])


@pytest.fixture(scope = "module", autouse = True)
def _needs_gemma3():
    pytest.importorskip("transformers.models.gemma3.modeling_gemma3")


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_plain_gemma3_after_forced_float32_patches(dtype):
    fresh = _run([], dtype)
    assert "error" not in fresh, fresh
    after = _run(["1"], dtype)
    assert "error" not in after, after["error"]
    assert after["logits_dtype"] == f"torch.{dtype}"
    assert after["hidden_dtypes"] == [f"torch.{dtype}"]
    assert after["forced_norm_dtypes"] == ["torch.float16", "torch.float16"]
    # MLP / attention keep their forced float32 internals, so only rounding separates the two.
    assert abs(after["loss"] - fresh["loss"]) < 1e-2 * abs(fresh["loss"])
    max_diff = max(abs(a - b) for a, b in zip(after["logits"], fresh["logits"]))
    assert max_diff < 5e-2, max_diff
