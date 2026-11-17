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

"""Gemma3 compiled `prepare` must recompile for decode after prefill (torch 2.9.0 C++ shape guards)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = r"""
import inspect
import torch
import transformers
from transformers import Gemma3ForCausalLM, Gemma3TextConfig
from transformers.cache_utils import DynamicCache
from unsloth_zoo.temporary_patches.gemma import patch_Gemma3Attention_generic

patch_Gemma3Attention_generic()
modeling = transformers.models.gemma3.modeling_gemma3
assert modeling.Gemma3Attention.forward.__module__.startswith("unsloth_zoo"), modeling.Gemma3Attention.forward

torch.manual_seed(0)
config = Gemma3TextConfig(
    vocab_size = 128, hidden_size = 64, intermediate_size = 128, num_hidden_layers = 2,
    num_attention_heads = 4, num_key_value_heads = 1, head_dim = 16, sliding_window = 512,
    layer_types = ["sliding_attention", "full_attention"], max_position_embeddings = 2048,
)
config._attn_implementation = "sdpa"
model = Gemma3ForCausalLM(config).float().eval()
layer = model.model.layers[1]
attn = layer.self_attn
rotary = model.model.rotary_emb
# transformers 5.x keys one rotary module by layer type
ROTARY_KWARGS = {"layer_type": "full_attention"} if "layer_type" in inspect.signature(rotary.forward).parameters else {}
NEG = torch.finfo(torch.float32).min

def run(hidden, start, cache):
    q = hidden.shape[1]
    kv = start + q
    position_ids = torch.arange(start, kv)[None]
    position_embeddings = rotary(hidden, position_ids, **ROTARY_KWARGS)
    mask = torch.zeros(1, 1, q, kv)
    mask.masked_fill_(torch.arange(kv)[None, :] > torch.arange(start, kv)[:, None], NEG)
    out, _ = attn(
        hidden_states = hidden, position_embeddings = position_embeddings,
        attention_mask = mask, past_key_values = cache, cache_position = position_ids[0],
    )
    return out

PREFILL = 300
hidden = torch.randn(1, PREFILL + 3, 64)
with torch.no_grad():
    cache = DynamicCache()
    outputs = [run(hidden[:, :PREFILL], 0, cache)]
    for step in range(3):
        outputs.append(run(hidden[:, PREFILL + step : PREFILL + step + 1], PREFILL + step, cache))
    incremental = torch.cat(outputs, dim = 1)
    full = run(hidden, 0, None)
torch.testing.assert_close(incremental, full, atol = 1e-4, rtol = 1e-4)
if torch.__version__.startswith("2.9.0"):
    import torch._dynamo.config as dynamo_config
    assert dynamo_config.enable_cpp_symbolic_shape_guards is False
print("OK")
"""


def test_gemma3_prepare_decode_after_prefill():
    pytest.importorskip("transformers.models.gemma3.modeling_gemma3")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES = "", UNSLOTH_IS_PRESENT = "1", UNSLOTH_ZOO_DISABLE_GPU_INIT = "1")
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT), os.environ.get("PYTHONPATH")) if p)
    env.pop("UNSLOTH_COMPILE_DISABLE", None)
    env.pop("UNSLOTH_FORCE_FLOAT32", None)
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT], capture_output = True, text = True, timeout = 900, env = env,
    )
    assert result.returncode == 0 and "OK" in result.stdout, (result.stdout[-2000:] + result.stderr[-4000:])
