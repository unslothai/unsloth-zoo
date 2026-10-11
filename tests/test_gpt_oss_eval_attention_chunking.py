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

"""Eval gpt-oss eager attention splits long prefills into query chunks with unchanged output.

A 4 x 4608 token gpt-oss-120b prefill held ~32 GiB of logits + float32 softmax in one layer.
Runs in a subprocess so torch.compile is identity before unsloth_zoo imports (as in
tests/test_gpt_oss_attention_mask.py). CPU-only.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

_SUBPROCESS = textwrap.dedent(r"""
import os, sys
os.environ['UNSLOTH_ALLOW_CPU'] = '1'
os.environ['UNSLOTH_MODEL_NAME'] = 'unsloth/gpt-oss-test'
os.environ['UNSLOTH_COMPILE_DISABLE'] = '0'

import torch
def _id_compile(model=None, *a, **k):
    if model is None:
        return lambda fn: fn
    return model
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

import unsloth_zoo  # noqa
import unsloth_zoo.temporary_patches.gpt_oss as G
G.patch_GptOssAttention()
import transformers.models.gpt_oss.modeling_gpt_oss as M
from torch.utils._python_dispatch import TorchDispatchMode

assert M.GptOssAttention.forward.__module__ == G.__name__, "attention patch not installed"

class CountSoftmax(TorchDispatchMode):
    def __init__(self):
        super().__init__(); self.n = 0
    def __torch_dispatch__(self, func, types, args = (), kwargs = None):
        if func.overloadpacket is torch.ops.aten._softmax:
            self.n += 1
        return func(*args, **(kwargs or {}))

config = M.GptOssConfig(
    hidden_size = 64, intermediate_size = 128, num_hidden_layers = 1,
    num_attention_heads = 4, num_key_value_heads = 2, head_dim = 16,
    num_local_experts = 2, num_experts_per_tok = 1, vocab_size = 100,
    max_position_embeddings = 128, sliding_window = 8, layer_types = ['full_attention'],
)
config._attn_implementation = 'eager'
torch.manual_seed(0)
attn = M.GptOssAttention(config, layer_idx = 0).eval()
with torch.no_grad():
    attn.sinks.normal_()
rope = M.GptOssRotaryEmbedding(config)

B, T = 2, 24
x = torch.randn(B, T, 64)
pos = torch.arange(T).unsqueeze(0).expand(B, -1)
cos_sin = rope(x, pos)
# Causal + left padding (row 1 starts at 5), additive float mask [B, 1, T, T].
keep = torch.ones(B, T, T, dtype = torch.bool).tril()
keep[1, :, :5] = False
keep[1, :5, :] = torch.eye(T, dtype = torch.bool)[:5]
mask = torch.zeros(B, 1, T, T).masked_fill(~keep.unsqueeze(1), torch.finfo(torch.float32).min)

def run():
    with torch.no_grad(), CountSoftmax() as c:
        out = attn(x.clone(), position_embeddings = cos_sin, attention_mask = mask)[0]
    return out, c.n

full, n_full = run()
G._EAGER_ATTENTION_CHUNK_BYTES = 1  # one query row per chunk
chunked, n_chunked = run()
assert n_full == 1, n_full
assert n_chunked == T, n_chunked
diff = (full - chunked).abs().max().item()
assert torch.allclose(full, chunked, rtol = 1e-5, atol = 1e-6), diff

# A broadcast [B, 1, 1, T] mask keeps its single query row in every chunk.
mask1 = mask[:, :, -1:, :].clone()
with torch.no_grad():
    G._EAGER_ATTENTION_CHUNK_BYTES = 1 << 40
    a = attn(x.clone(), position_embeddings = cos_sin, attention_mask = mask1)[0]
    G._EAGER_ATTENTION_CHUNK_BYTES = 1
    b = attn(x.clone(), position_embeddings = cos_sin, attention_mask = mask1)[0]
assert torch.allclose(a, b, rtol = 1e-5, atol = 1e-6)
print('CHUNK_OK', n_full, n_chunked, diff)
""")


def test_eval_attention_chunks_match_single_shot():
    r = subprocess.run([sys.executable, "-c", _SUBPROCESS], capture_output = True, text = True, timeout = 600)
    assert r.returncode == 0 and "CHUNK_OK" in r.stdout, (r.stdout[-3000:], r.stderr[-3000:])
