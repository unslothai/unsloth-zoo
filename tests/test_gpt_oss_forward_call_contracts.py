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

"""The patched GptOssModel.forward keeps two stock call contracts on transformers 4.x and 5.x:
an omitted use_cache follows config.use_cache in eval (a manual prefill then decode keeps the
prefix), and caller-supplied inputs_embeds keep their graph in eval (input gradients)."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

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
import transformers.models.gpt_oss.modeling_gpt_oss as M
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig

c = GptOssConfig(
    hidden_size=32, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
    num_key_value_heads=1, head_dim=16, num_local_experts=4, num_experts_per_tok=2,
    vocab_size=64, max_position_embeddings=64, sliding_window=4,
    layer_types=['sliding_attention', 'full_attention'], use_cache=True,
)
c._attn_implementation = 'eager'
torch.manual_seed(0)
lm = M.GptOssForCausalLM(c)
with torch.no_grad():
    for n, p in lm.named_parameters():
        p.normal_(0, 0.05)
        if 'norm' in n:
            p.add_(1)
from unsloth_zoo.temporary_patches import gpt_oss as G
G.patch_GptOssModel()
fwd = M.GptOssModel.forward
res = {'live': getattr(fwd, '__wrapped__', fwd).__module__ == G.__name__}
lm.eval()
ids = torch.tensor([[1, 2, 3]])
with torch.no_grad():
    full = lm(ids, use_cache=False).logits[:, -1]
    prefix = lm(ids[:, :2])
    step = lm(ids[:, 2:], past_key_values=prefix.past_key_values).logits[:, -1]
res['prefix_has_cache'] = prefix.past_key_values is not None
res['decode_err'] = float((full - step).abs().max())
lm.requires_grad_(False)
emb = lm.model.embed_tokens(ids).detach().requires_grad_(True)
out = lm(inputs_embeds=emb, use_cache=False).logits
res['embeds_out_requires_grad'] = out.requires_grad
if out.requires_grad:
    out.square().sum().backward()
res['embeds_grad'] = emb.grad is not None and bool(emb.grad.abs().sum() > 0)
# A single-token grad-bearing input must not reach the in-place decode fast path.
emb1 = lm.model.embed_tokens(ids[:, :1]).detach().float().requires_grad_(True)
try:
    out1 = lm(inputs_embeds=emb1, use_cache=False).logits
    out1.square().sum().backward()
    res['embeds1_grad'] = emb1.grad is not None and bool(emb1.grad.abs().sum() > 0)
except RuntimeError as e:
    res['embeds1_grad'] = repr(e)[:300]
# Embeddings the model makes itself (an input-require-grads hook) are still detached in eval.
lm.enable_input_require_grads()
res['ids_out_requires_grad'] = lm(ids, use_cache=False).logits.requires_grad
print('RESULT ' + json.dumps(res))
"""


def test_patched_forward_keeps_cache_default_and_input_grads():
    pytest.importorskip("transformers.models.gpt_oss.modeling_gpt_oss")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.pop("PYTORCH_NVML_BASED_CUDA_CHECK", None)
    env.pop("UNSLOTH_RETURN_HIDDEN_STATES", None)
    proc = subprocess.run(
        [sys.executable, "-c", _RUNTIME],
        capture_output = True, text = True, timeout = 600, env = env,
    )
    lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, (
        f"exit={proc.returncode}\nstdout:\n{proc.stdout[-3000:]}\nstderr:\n{proc.stderr[-6000:]}"
    )
    res = json.loads(lines[-1][len("RESULT "):])
    assert res["live"], res
    assert res["prefix_has_cache"], res
    assert res["decode_err"] < 1e-4, res
    assert res["embeds_out_requires_grad"] and res["embeds_grad"], res
    assert res["embeds1_grad"] is True, res
    assert res["ids_out_requires_grad"] is False, res
