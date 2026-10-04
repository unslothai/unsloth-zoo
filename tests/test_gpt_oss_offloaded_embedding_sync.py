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

# Guards: a non_blocking input_ids copy to the CPU-offloaded embedding raced generate()
# on transformers 5 (no per-step sync), embedding the previous token: garbage output.
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import torch


_RUNTIME = r"""
import os, sys, json
os.environ['UNSLOTH_MODEL_NAME'] = 'unsloth/gpt-oss-test'
os.environ['UNSLOTH_COMPILE_DISABLE'] = '0'
import torch
def _id_compile(model=None, *a, **k):
    return (lambda fn: fn) if model is None else model
torch.compile = _id_compile
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
import transformers.models.gpt_oss.modeling_gpt_oss as M
from unsloth_zoo.temporary_patches import gpt_oss as G
G.patch_GptOssModel()
if 'patch_GptOssModel' not in getattr(M.GptOssModel.forward, '__qualname__', ''):
    print('SKIP patched GptOssModel.forward not installed on this transformers'); raise SystemExit(0)

L, T = 2, 8
c = GptOssConfig(
    hidden_size=64, intermediate_size=64, num_hidden_layers=L,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    num_local_experts=4, num_experts_per_tok=2, vocab_size=128,
    max_position_embeddings=128, sliding_window=4,
    layer_types=['sliding_attention', 'full_attention'] * (L // 2),
)
c._attn_implementation = 'eager'
torch.manual_seed(0)
model = M.GptOssModel(c).cuda().eval()
model.embed_tokens.cpu()
ids = torch.randint(0, 128, (1, T), generator=torch.Generator().manual_seed(1)).cuda()
mask = torch.ones_like(ids)  # generate passes a mask; without one, a padding check syncs first

from torch.overrides import TorchFunctionMode

class _RecordTo(TorchFunctionMode):  # every non-blocking CUDA -> CPU Tensor.to
    def __init__(self):
        super().__init__(); self.unsafe = []
    def __torch_function__(self, func, types, args = (), kwargs = None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        if func is torch.Tensor.to and isinstance(out, torch.Tensor) and args[0].is_cuda and out.device.type == 'cpu':
            if kwargs.get('non_blocking') or any(a is True for a in args[1:]):
                self.unsafe.append(str(tuple(args[0].shape)))
        return out

with torch.no_grad():
    ref = model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state.float().cpu()
    with _RecordTo() as rec:
        out = model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state.float().cpu()
print('RESULT', json.dumps({'unsafe_copies': rec.unsafe, 'max_diff': (out - ref).abs().max().item()}))
"""


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA for an asynchronous device -> host copy")
def test_offloaded_embedding_reads_finished_input_ids():
    proc = subprocess.run([sys.executable, "-c", _RUNTIME], capture_output = True, text = True, timeout = 600)
    skip = [l for l in proc.stdout.splitlines() if l.startswith("SKIP ")]
    if skip:
        pytest.skip(skip[0][len("SKIP "):])
    lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-2000:] + proc.stderr[-4000:]
    res = json.loads(lines[-1][len("RESULT "):])
    assert res["unsafe_copies"] == [], res
    assert res["max_diff"] == 0.0, res


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_offloaded_embedding_op_under_cuda_graphs():
    """The offloaded lookup op used inside a compiled, CUDA graphed step: fullgraph, and the same
    rows as eager while the ids change every step. (The race it removes, a CPU lookup reading ids
    a graph partition had not copied back yet, showed on a whole gpt-oss decode step at batch 4;
    this small graph does not reproduce it.)"""
    from unsloth_zoo.temporary_patches.gpt_oss import _offloaded_embedding
    if _offloaded_embedding is None:
        pytest.skip("torch.library.custom_op unavailable")
    torch.manual_seed(0)
    V, H = 50021, 64
    weight = torch.randn(V, H, dtype = torch.bfloat16)
    proj = torch.randn(H, V, device = "cuda", dtype = torch.bfloat16)

    def step(logits):
        # Next ids come from a GPU op, as in generate, then go through the CPU table.
        ids = logits.argmax(-1, keepdim = True)
        emb = _offloaded_embedding(ids, weight)
        return (emb[:, 0].float() @ proj.float()).to(torch.bfloat16)

    compiled = torch.compile(step, fullgraph = True, mode = "reduce-overhead")
    logits = torch.randn(4, V, device = "cuda", dtype = torch.bfloat16)
    ref = logits.clone()
    with torch.no_grad():
        for _ in range(12):
            torch.compiler.cudagraph_mark_step_begin()
            logits = compiled(logits).clone()
            ref = step(ref)
            torch.testing.assert_close(logits, ref)
