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

"""Gemma3Attention patch must install (not silently fall back) and fill the KV cache under 5.24's `cache`."""

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

flag, dtype = sys.argv[1], getattr(torch, sys.argv[2])
os.environ["UNSLOTH_FORCE_FLOAT32"] = flag
patches = [p for p in TEMPORARY_PATCHES if getattr(p, "__module__", "") == gemma_patches.__name__]
for _phase in ("pre_compile", "post_compile"):
    for patch in patches:
        patch()
modeling = transformers.models.gemma3.modeling_gemma3
f = modeling.Gemma3Attention.forward
f = getattr(f, "_torchdynamo_orig_callable", f)
while hasattr(f, "__wrapped__"): f = f.__wrapped__
config = Gemma3TextConfig(
    vocab_size = 128, hidden_size = 32, intermediate_size = 64, num_hidden_layers = 2,
    num_attention_heads = 2, num_key_value_heads = 1, head_dim = 16, sliding_window = 8,
    max_position_embeddings = 64,
)
config._attn_implementation = "sdpa"
torch.manual_seed(0)
model = Gemma3ForCausalLM(config).to(dtype).eval()
ids = torch.randint(0, config.vocab_size, (1, 12))
with torch.no_grad():
    full = model(input_ids = ids, use_cache = False).logits.float()
    pre = model(input_ids = ids[:, :-2], use_cache = True)
    cache = getattr(pre, "cache", None)
    if cache is None: cache = getattr(pre, "past_key_values", None)
    seq = cache.get_seq_length() if cache is not None else -1
    import inspect
    name = "cache" if "cache" in inspect.signature(Gemma3ForCausalLM.forward).parameters else "past_key_values"
    kw = {name: cache}
    step = model(input_ids = ids[:, -2:], use_cache = True, **kw).logits.float()
print("RESULT " + json.dumps({
    "installed": f.__module__ + "." + f.__qualname__,
    "cache_len_after_prefill": seq,
    "decode_max_err": float((step - full[:, -2:]).abs().max()),
}))
"""


def _run(flag, dtype):
    env = dict(
        os.environ, CUDA_VISIBLE_DEVICES = "", UNSLOTH_IS_PRESENT = "1",
        UNSLOTH_ZOO_DISABLE_GPU_INIT = "1", TORCHDYNAMO_DISABLE = "1",
    )
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT), os.environ.get("PYTHONPATH")) if p)
    for key in ("UNSLOTH_FORCE_FLOAT32", "UNSLOTH_MODEL_NAME"):
        env.pop(key, None)
    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT, flag, dtype],
        capture_output = True, text = True, timeout = 900, env = env,
    )
    lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-2000:] + proc.stderr[-4000:]
    return json.loads(lines[-1][len("RESULT "):])


@pytest.fixture(scope = "module", autouse = True)
def _needs_gemma3():
    pytest.importorskip("transformers.models.gemma3.modeling_gemma3")


@pytest.mark.parametrize("flag,dtype", [("0", "bfloat16"), ("1", "float16")])
def test_gemma3_attention_patch_installs_and_fills_cache(flag, dtype):
    r = _run(flag, dtype)
    assert "unsloth_zoo.temporary_patches.gemma._make_gemma3_attn_forwards" in r["installed"], r
    assert r["cache_len_after_prefill"] == 10, r
    assert r["decode_max_err"] < (5e-2 if dtype == "bfloat16" else 1e-2), r
