# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# transformers 5 dropped GptOssDecoderLayer.attention_type, so the patched forward sent every
# layer the full causal mask (gpt-oss-20b wikitext PPL 1082.7 vs stock 214.7). CPU, tiny config.
from __future__ import annotations

import os
import subprocess
import sys
import types

import pytest
import torch


def _gpt_oss():
    from unsloth_zoo.temporary_patches import gpt_oss
    return gpt_oss


def _layer(attention_type = None, sliding_window = None):
    layer = types.SimpleNamespace(self_attn = types.SimpleNamespace(sliding_window = sliding_window))
    if attention_type is not None: layer.attention_type = attention_type
    return layer


def test_layer_type_matches_every_transformers_layout():
    f = _gpt_oss()._gpt_oss_layer_attention_type
    cfg = types.SimpleNamespace(layer_types = ["sliding_attention", "full_attention"])
    assert f(_layer(attention_type = "full_attention"), cfg, 0) == "full_attention"
    assert f(_layer(), cfg, 0) == "sliding_attention"
    assert f(_layer(), cfg, 1) == "full_attention"
    bare = types.SimpleNamespace()
    assert f(_layer(sliding_window = 128), bare, 0) == "sliding_attention"
    assert f(_layer(), bare, 0) == "full_attention"


def test_select_mask_never_tests_tensor_truthiness():
    f = _gpt_oss()._gpt_oss_select_mask
    full = torch.zeros(1, 1, 8, 8)
    sliding = torch.ones(1, 1, 8, 8)
    masks = {"full_attention": full, "sliding_attention": sliding}
    assert f(masks, "sliding_attention") is sliding
    assert f(masks, "full_attention") is full
    assert f({"full_attention": full, "sliding_attention": None}, "sliding_attention") is None
    assert f(masks, "chunked_attention") is full
    assert f(full, "sliding_attention") is full
    assert f(None, "sliding_attention") is None


_RUNTIME = r"""
import os, sys, json
os.environ['UNSLOTH_ALLOW_CPU'] = '1'
os.environ['UNSLOTH_MODEL_NAME'] = 'unsloth/gpt-oss-test'
os.environ['UNSLOTH_COMPILE_DISABLE'] = '0'
import torch
def _id_compile(model=None, *a, **k):
    return (lambda fn: fn) if model is None else model
torch.compile = _id_compile
if not torch.cuda.is_available():
    import torch.cuda.memory as _cm
    _cm.mem_get_info = lambda *a, **k: (0, 80 * 1024**3)
    torch.cuda.device_count = lambda: 1
    torch.cuda.get_device_capability = lambda *a, **k: (8, 0)
    class _P:
        major = 8; minor = 0; total_memory = 80 * 1024**3
        multi_processor_count = 108; name = 'stub'
    torch.cuda.get_device_properties = lambda *a, **k: _P()
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
import transformers.models.gpt_oss.modeling_gpt_oss as M

STOCK_ATTENTION = sys.argv[1] == 'stock_attention'
L, T = 4, 16
def make(layer_types):
    c = GptOssConfig(
        hidden_size=64, intermediate_size=64, num_hidden_layers=L,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        num_local_experts=4, num_experts_per_tok=2, vocab_size=128,
        max_position_embeddings=128, sliding_window=4, layer_types=layer_types,
    )
    c._attn_implementation = 'eager'
    return M.GptOssForCausalLM(c).eval()
torch.manual_seed(0)
model = make(['sliding_attention', 'full_attention'] * (L // 2))
with torch.no_grad():
    for n, p in model.named_parameters():
        p.normal_(0, 0.3 if 'sinks' in n else 0.08)
g = torch.Generator().manual_seed(1)
ids = torch.randint(0, 128, (2, T), generator=g)
pad = torch.ones(2, T, dtype=torch.long); pad[1, :5] = 0
prompt = torch.randint(0, 128, (1, 10), generator=g)

def run(m):
    out = {}
    with torch.no_grad():
        out['eval'] = m(input_ids=ids, use_cache=False).logits
        out['eval_padded'] = m(input_ids=ids, attention_mask=pad, use_cache=False).logits
        out['eval_cache'] = m(input_ids=ids, use_cache=True).logits
        o = m(input_ids=ids[:1, :10], use_cache=True)
        pkv, steps = o.past_key_values, [o.logits[:, -1]]
        for t in range(10, T):
            o = m(input_ids=ids[:1, t:t + 1], past_key_values=pkv, use_cache=True)
            pkv = o.past_key_values; steps.append(o.logits[:, -1])
        out['decode'] = torch.stack(steps, 1)
        out['generate'] = m.generate(prompt, max_new_tokens=14, do_sample=False, pad_token_id=0).float()
        m.train()
        out['train'] = m(input_ids=ids, use_cache=False).logits
        m.eval()
    return out

ref = run(model)
full = make(['full_attention'] * L); full.load_state_dict(model.state_dict())
with torch.no_grad():
    oracle = float((full(input_ids=ids, use_cache=False).logits - ref['eval']).abs().max())

from unsloth_zoo.temporary_patches import gpt_oss as G
stock_attention_forward = M.GptOssAttention.forward
G.patch_GptOssAttention()
G.patch_GptOssModel()
if STOCK_ATTENTION:
    M.GptOssAttention.forward = stock_attention_forward
    G._GPT_OSS_FLEX_SINK_ATTENTION_INSTALLED = False
got = run(model)
fwd = M.GptOssModel.forward
res = {
    'oracle': oracle,
    'model_patch_live': getattr(fwd, '__wrapped__', fwd).__module__ == G.__name__,
    'diff': {k: float((ref[k] - got[k]).abs().max()) for k in ref},
}
print('RESULT ' + json.dumps(res))
"""


@pytest.mark.parametrize("attention", ["unsloth_attention", "stock_attention"])
def test_patched_forward_matches_stock_with_sliding_window(attention):
    env = dict(os.environ)
    # CPU keeps parity exact; NVML would report GPUs CUDA_VISIBLE_DEVICES hides.
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.pop("PYTORCH_NVML_BASED_CUDA_CHECK", None)
    proc = subprocess.run(
        [sys.executable, "-c", _RUNTIME, attention],
        capture_output = True, text = True, timeout = 600, env = env,
    )
    lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, (
        f"exit={proc.returncode}\nstdout:\n{proc.stdout[-3000:]}\nstderr:\n{proc.stderr[-6000:]}"
    )
    import json
    res = json.loads(lines[-1][len("RESULT "):])
    # Ignoring the window must move the logits, else the parity check is vacuous.
    assert res["oracle"] > 1e-3, res
    for key, diff in res["diff"].items():
        assert diff < 1e-5, (key, res)

