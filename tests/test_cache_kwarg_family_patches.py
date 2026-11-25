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

"""Family attention / decoder patches keep the KV cache across transformers' past_key_values -> cache rename.

transformers 5.24 passes the cache as `cache=`. A replacement declaring `past_key_values` received it in
**kwargs and ran with no cache, so a cached decode step attended to the new token only. Each case
applies the real patch to a tiny random model and checks that the patch is live and that one cached
decode step matches the full forward.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

_RUNTIME = r"""
import os, sys, json, warnings
os.environ['UNSLOTH_ALLOW_CPU'] = '1'
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
import importlib
import transformers

case = os.environ['_CASE']
common = dict(vocab_size=64, hidden_size=32, intermediate_size=32, num_hidden_layers=2,
              num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=64,
              use_cache=True)
if case.startswith('gemma3'):
    M = importlib.import_module('transformers.models.gemma3.modeling_gemma3')
    from transformers import Gemma3TextConfig as C
    config = C(**common, sliding_window=4, layer_types=['sliding_attention', 'full_attention'])
    model_cls, targets = M.Gemma3ForCausalLM, [M.Gemma3Attention]
    from unsloth_zoo.temporary_patches import gemma as P
    patches = [P.patch_Gemma3Attention if case == 'gemma3_float32' else P.patch_Gemma3Attention_generic]
elif case.startswith('ministral'):
    M = importlib.import_module('transformers.models.ministral.modeling_ministral')
    from transformers import MinistralConfig as C
    kw = dict(common)
    if case == 'ministral_no_window':
        config = C(**kw, sliding_window=None, layer_types=['full_attention', 'full_attention'])
    else:
        config = C(**kw, sliding_window=4, layer_types=['sliding_attention', 'full_attention'])
    model_cls, targets = M.MinistralForCausalLM, [M.MinistralAttention, M.MinistralModel]
    from unsloth_zoo.temporary_patches import ministral as P
    patches = [P.patch_MinistralAttention, P.patch_MinistralModel_forward]
elif case == 'qwen3_moe_float32':
    M = importlib.import_module('transformers.models.qwen3_moe.modeling_qwen3_moe')
    from transformers import Qwen3MoeConfig as C
    config = C(**common, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16,
               decoder_sparse_step=1, mlp_only_layers=[])
    model_cls, targets = M.Qwen3MoeForCausalLM, [M.Qwen3MoeAttention, M.Qwen3MoeDecoderLayer]
    from unsloth_zoo.temporary_patches import qwen3_moe_float32 as P
    patches = [P.patch_Qwen3MoeDecoderLayer_float32, P.patch_Qwen3MoeAttention_float32]
elif case == 'gemma4_float32':
    M = importlib.import_module('transformers.models.gemma4.modeling_gemma4')
    from transformers import Gemma4TextConfig as C
    config = C(**common, sliding_window=4, layer_types=['sliding_attention', 'full_attention'],
               num_kv_shared_layers=0, hidden_size_per_layer_input=0, global_head_dim=16)
    model_cls, targets = M.Gemma4ForCausalLM, [M.Gemma4TextAttention]
    from unsloth_zoo.temporary_patches import gemma4_float32 as P
    patches = [P.patch_Gemma4TextAttention]
elif case == 'qwen3_moe_hidden_states_wrapper':
    M = importlib.import_module('transformers.models.qwen3_moe.modeling_qwen3_moe')
    from transformers import Qwen3MoeConfig as C
    config = C(**common, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16,
               decoder_sparse_step=1, mlp_only_layers=[])
    model_cls, targets = M.Qwen3MoeForCausalLM, [M.Qwen3MoeForCausalLM]
    from unsloth_zoo.temporary_patches import qwen3_moe as P
    output_cls = getattr(M, 'MoeCausalLMOutputWithPast')
    patches = [lambda: P._patch_causal_lm_forward_for_hidden_states(M.Qwen3MoeForCausalLM, output_cls, 'Qwen3MoeForCausalLM')]
elif case == 'cpu_chunked_ce':
    M = importlib.import_module('transformers.models.llama.modeling_llama')
    from transformers import LlamaConfig as C
    config = C(**common)
    model_cls, targets = M.LlamaForCausalLM, []
    from unsloth_zoo.cpu import ops as P
    patches = [lambda: P.patch_chunked_causal_lm_loss(lm)]
else:
    raise SystemExit('unknown case ' + case)

config._attn_implementation = 'eager'
torch.manual_seed(0)
lm = model_cls(config).float()
with torch.no_grad():
    for n, p in lm.named_parameters():
        p.normal_(0, 0.05)
        if 'norm' in n:
            p.add_(1)
stock = {cls.__name__: cls.forward for cls in targets}
for patch in patches:
    patch()
res = {'live': {cls.__name__: cls.forward is not stock[cls.__name__] for cls in targets}}
if case == 'cpu_chunked_ce':
    res['live'] = {'instance_forward': 'forward' in vars(lm)}
# Decode with the cache under the running transformers' own keyword (`cache` from 5.24), as generate does.
import inspect
_stock_causal = getattr(model_cls.forward, '__wrapped__', model_cls.forward)
cache_kw = 'cache' if 'cache' in inspect.signature(_stock_causal).parameters else 'past_key_values'
lm.eval()
ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
with torch.no_grad(), warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    full = lm(ids, use_cache=False).logits[:, -1].float()
    prefix = lm(ids[:, :-1], use_cache=True)
    cache = prefix.get('cache', prefix.get('past_key_values'))
    step = lm(ids[:, -1:], use_cache=True, **{cache_kw: cache}).logits[:, -1].float()
res['decode_err'] = float((full - step).abs().max())
res['scale'] = float(full.abs().max())
res['both_set_warnings'] = sum('Both `' in str(w.message) for w in caught)
print('RESULT ' + json.dumps(res))
"""

_CASES = {
    # case: (model module that must exist, extra env)
    "gemma3_generic": ("gemma3", {"UNSLOTH_MODEL_NAME": "unsloth/gemma3-test", "UNSLOTH_FORCE_FLOAT32": "0"}),
    "gemma3_float32": ("gemma3", {"UNSLOTH_MODEL_NAME": "unsloth/gemma3-test", "UNSLOTH_FORCE_FLOAT32": "1"}),
    "ministral": ("ministral", {"UNSLOTH_MODEL_NAME": "unsloth/ministral-test"}),
    "ministral_no_window": ("ministral", {"UNSLOTH_MODEL_NAME": "unsloth/ministral-test"}),
    "qwen3_moe_float32": ("qwen3_moe", {"UNSLOTH_MODEL_NAME": "unsloth/qwen3-moe-test", "UNSLOTH_FORCE_FLOAT32": "1"}),
    "gemma4_float32": ("gemma4", {"UNSLOTH_MODEL_NAME": "unsloth/gemma4-test", "UNSLOTH_FORCE_FLOAT32": "1"}),
    "qwen3_moe_hidden_states_wrapper": ("qwen3_moe", {"UNSLOTH_MODEL_NAME": "unsloth/qwen3-moe-test"}),
    "cpu_chunked_ce": ("llama", {"UNSLOTH_MODEL_NAME": "unsloth/llama-test"}),
}


@pytest.mark.parametrize("case", list(_CASES))
def test_patched_forward_is_live_and_keeps_the_kv_cache(case):
    family, extra = _CASES[case]
    pytest.importorskip(f"transformers.models.{family}.modeling_{family}")
    env = dict(os.environ)
    env.update(extra)
    env["_CASE"] = case
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.pop("PYTORCH_NVML_BASED_CUDA_CHECK", None)
    env.pop("UNSLOTH_COMPILE_DISABLE", None)
    proc = subprocess.run(
        [sys.executable, "-c", _RUNTIME],
        capture_output = True, text = True, timeout = 600, env = env,
    )
    lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, (
        f"exit={proc.returncode}\nstdout:\n{proc.stdout[-3000:]}\nstderr:\n{proc.stderr[-6000:]}"
    )
    res = json.loads(lines[-1][len("RESULT "):])
    assert all(res["live"].values()), res
    # Forced-float32 paths cast to fp16 around the projections; the rest are float32 end to end.
    tolerance = 5e-2 if "float32" in case else 1e-4
    assert res["decode_err"] < tolerance * max(1.0, res["scale"]), res
    assert res["both_set_warnings"] == 0, res
