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

# Guards: after patch_gpt_oss_bnb4bit swaps GptOssTopKRouter, the stock GptOssModel.forward (used
# under UNSLOTH_COMPILE_DISABLE) recorded no router_logits, so load_balancing_loss_func raised
# IndexError on the first training step.
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
os.environ['UNSLOTH_COMPILE_DISABLE'] = '1'
import torch
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
from unsloth_zoo.temporary_patches import gpt_oss as G

def target():
    return M.GptOssPreTrainedModel._can_record_outputs['router_logits'].target_class

L, T = 4, 12
def build():
    c = GptOssConfig(
        hidden_size=64, intermediate_size=64, num_hidden_layers=L,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        num_local_experts=4, num_experts_per_tok=2, vocab_size=128,
        max_position_embeddings=128, sliding_window=4,
        layer_types=['sliding_attention', 'full_attention'] * (L // 2),
    )
    c._attn_implementation = 'eager'
    torch.manual_seed(0)
    return M.GptOssForCausalLM(c).train()

ids = torch.randint(0, 128, (2, T), generator=torch.Generator().manual_seed(1))
def run(m):
    # Independent reference: the router outputs seen by our own hooks, one per layer.
    seen = []
    def keep(module, args, output):
        seen.append((output[0] if isinstance(output, tuple) else output).detach())
    routers = [layer.mlp.router for layer in m.model.layers]
    saved = lambda r: getattr(r, 'modules_to_save', None)
    routers = [saved(r)['default'] if saved(r) is not None and 'default' in saved(r) else r for r in routers]
    handles = [r.register_forward_hook(keep) for r in routers]
    try:
        out = m(input_ids=ids, labels=ids, use_cache=False, output_router_logits=True)
        out.loss.backward()
        got = [r.detach() for r in (out.router_logits or ())]
        return {'error': None, 'n_router_logits': len(got),
                'match_reference': len(got) == len(seen) and all(torch.equal(a, b) for a, b in zip(got, seen)),
                'aux_is_tensor': isinstance(out.aux_loss, torch.Tensor),
                'loss_finite': bool(torch.isfinite(out.loss))}
    except Exception as e:
        return {'error': f'{type(e).__name__}: {e}'}
    finally:
        for h in handles:
            h.remove()

stock_cls = M.GptOssTopKRouter
res = {'compile_disabled': bool(G.UNSLOTH_COMPILE_DISABLE)}
G.patch_gpt_oss_bnb4bit()
res['router_swapped'] = M.GptOssTopKRouter is not stock_cls
# Still named like the stock class, so name-based retargets (compiler, patch mappings) keep working.
res['target_widened'] = target().__name__ == stock_cls.__name__ and target() is not stock_cls
m = build()
res['model_router_is_patched'] = type(m.model.layers[0].mlp.router) is M.GptOssTopKRouter
res['patched'] = run(m)
res['patched_again'] = run(m)
# A PEFT wrapper around one router (wrapped before the first forward, as training does) must not
# add a second capture for that layer: neither the patched nor the restored stock router.
from peft import LoraConfig, get_peft_model
def wrapped():
    pm = get_peft_model(build(), LoraConfig(r=2, target_modules=['q_proj'], modules_to_save=['layers.0.mlp.router']))
    res['peft_wrapped'] = type(pm.base_model.model.model.layers[0].mlp.router).__name__
    return pm.base_model.model
res['patched_peft'] = run(wrapped())
G.restore_gpt_oss_original()
res['stock'] = run(build())
res['stock_peft'] = run(wrapped())
print('RESULT ' + json.dumps(res))
"""


def test_bnb4bit_router_swap_keeps_router_logits_without_compile():
    M = pytest.importorskip("transformers.models.gpt_oss.modeling_gpt_oss")
    pytest.importorskip("peft")
    if not isinstance(getattr(M.GptOssPreTrainedModel, "_can_record_outputs", None), dict):
        pytest.skip("this transformers has no class-keyed output recorders")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.pop("PYTORCH_NVML_BASED_CUDA_CHECK", None)
    env.pop("UNSLOTH_RETURN_HIDDEN_STATES", None)
    proc = subprocess.run(
        [sys.executable, "-c", _RUNTIME],
        capture_output = True, text = True, timeout = 600, env = env,
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, (
        f"exit={proc.returncode}\nstdout:\n{proc.stdout[-3000:]}\nstderr:\n{proc.stderr[-6000:]}"
    )
    res = json.loads(lines[-1][len("RESULT "):])
    assert res["compile_disabled"], res
    assert res["router_swapped"] and res["model_router_is_patched"], res
    assert res["patched"]["error"] is None, res
    assert res["patched"]["n_router_logits"] == 4, res
    assert res["patched"]["aux_is_tensor"] and res["patched"]["loss_finite"], res
    assert res["patched"]["match_reference"], res
    assert res["patched_again"]["n_router_logits"] == 4, res
    assert res["peft_wrapped"] == "ModulesToSaveWrapper", res
    assert res["patched_peft"]["n_router_logits"] == 4 and res["patched_peft"]["match_reference"], res
    assert res["stock"]["error"] is None and res["stock"]["n_router_logits"] == 4, res
    assert res["stock"]["match_reference"], res
    assert res["stock_peft"]["n_router_logits"] == 4 and res["stock_peft"]["match_reference"], res
    assert res["target_widened"], res
