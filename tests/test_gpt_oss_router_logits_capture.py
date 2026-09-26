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

# Guards: patched forward dropped router_logits, so aux_loss.to() raised (TRL >= 1.7 MoE).
from __future__ import annotations

import json
import os
import subprocess
import sys


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

L, T = 4, 12
c = GptOssConfig(
    hidden_size=64, intermediate_size=64, num_hidden_layers=L,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    num_local_experts=4, num_experts_per_tok=2, vocab_size=128,
    max_position_embeddings=128, sliding_window=4,
    layer_types=['sliding_attention', 'full_attention'] * (L // 2),
)
c._attn_implementation = 'eager'
torch.manual_seed(0)
model = M.GptOssForCausalLM(c).train()
with torch.no_grad():
    for n, p in model.named_parameters():
        p.normal_(0, 0.3 if 'sinks' in n else 0.08)
ids = torch.randint(0, 128, (2, T), generator=torch.Generator().manual_seed(1))

def run(m):
    out = m(input_ids=ids, labels=ids, use_cache=False, output_router_logits=True)
    out.loss.backward()
    grad = m.model.layers[0].mlp.router.weight.grad.clone()
    m.zero_grad()
    plain = m(input_ids=ids, labels=ids, use_cache=False)
    return {
        'aux_type': type(out.aux_loss).__name__,
        'aux': float(out.aux_loss) if isinstance(out.aux_loss, torch.Tensor) else None,
        'loss': float(out.loss),
        'n_router_logits': None if out.router_logits is None else len(out.router_logits),
        'router_logits': None if out.router_logits is None else [r.detach() for r in out.router_logits],
        'router_grad': grad,
        'plain_router_logits_is_none': plain.router_logits is None,
        'plain_aux_is_none': plain.aux_loss is None,
    }

ref = run(model)
# The stock run installs transformers' own permanent capture hooks; count ours against that.
hooks_before = sum(len(l.mlp.router._forward_hooks) for l in model.model.layers)
from unsloth_zoo.temporary_patches import gpt_oss as G
G.patch_GptOssModel()
fwd = M.GptOssModel.forward
try:
    got = run(model)
    err = None
except Exception as e:
    got, err = None, f'{type(e).__name__}: {e}'
# TRL leaves config.output_router_logits on after training; generating right after must still work
# when zoo's generation path hands the model a per-layer-type mask mapping.
model.config.output_router_logits = True
model.eval()
try:
    with torch.no_grad():
        gen = model.generate(ids[:1, :6], max_new_tokens=4, do_sample=False, pad_token_id=0)
    gen_err = None if gen.shape == (1, 10) else f'shape {tuple(gen.shape)}'
    # On GPU zoo's generation path passes the per-type mapping into the causal-LM forward.
    with torch.no_grad():
        model(input_ids=ids[:1, :6], attention_mask={'full_attention': None, 'sliding_attention': None}, use_cache=False)
except Exception as e:
    gen_err = f'{type(e).__name__}: {e}'
res = {'model_patch_live': getattr(fwd, '__wrapped__', fwd).__module__ == G.__name__, 'error': err, 'generate_error': gen_err}
if got is not None:
    res.update({
        'aux_type': got['aux_type'], 'n_router_logits': got['n_router_logits'],
        'ref_n_router_logits': ref['n_router_logits'],
        'aux_diff': abs(ref['aux'] - got['aux']) if got['aux'] is not None else None,
        'loss_diff': abs(ref['loss'] - got['loss']),
        'router_logits_diff': max(float((a - b).abs().max()) for a, b in zip(ref['router_logits'], got['router_logits'])) if got['router_logits'] else None,
        'router_grad_diff': float((ref['router_grad'] - got['router_grad']).abs().max()),
        'plain_router_logits_is_none': got['plain_router_logits_is_none'],
        'plain_aux_is_none': got['plain_aux_is_none'],
        'hooks_left': sum(len(l.mlp.router._forward_hooks) for l in model.model.layers) - hooks_before,
    })
print('RESULT ' + json.dumps(res))
"""


def test_patched_forward_returns_router_logits_like_stock():
    env = dict(os.environ)
    # CPU keeps parity exact; NVML would report GPUs CUDA_VISIBLE_DEVICES hides.
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.pop("PYTORCH_NVML_BASED_CUDA_CHECK", None)
    proc = subprocess.run(
        [sys.executable, "-c", _RUNTIME],
        capture_output = True, text = True, timeout = 600, env = env,
    )
    lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, (
        f"exit={proc.returncode}\nstdout:\n{proc.stdout[-3000:]}\nstderr:\n{proc.stderr[-6000:]}"
    )
    res = json.loads(lines[-1][len("RESULT "):])
    assert res["model_patch_live"], res
    assert res["error"] is None, res
    assert res["generate_error"] is None, res
    assert res["n_router_logits"] == res["ref_n_router_logits"] == 4, res
    assert res["aux_type"] == "Tensor", res
    assert res["aux_diff"] < 1e-5 and res["loss_diff"] < 1e-5, res
    assert res["router_logits_diff"] < 1e-5 and res["router_grad_diff"] < 1e-5, res
    assert res["plain_router_logits_is_none"] and res["plain_aux_is_none"], res
    assert res["hooks_left"] == 0, res
