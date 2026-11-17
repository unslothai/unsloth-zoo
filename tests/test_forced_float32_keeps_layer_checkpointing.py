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

"""Forced float32 patches must keep transformers >= 5.23's `forward_with_checkpointing` wrapper outermost,
else checkpointing silently turns off (Gemma-4 E2B LoRA: 27 GB instead of 10 GB)."""

import json
import os
import subprocess
import sys
import textwrap

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCRIPT = r'''
import json, os, torch
import unsloth_zoo, unsloth_zoo.temporary_patches
from unsloth_zoo.temporary_patches.common import TEMPORARY_PATCHES
assert unsloth_zoo.__file__.startswith(os.environ["REPO"]), unsloth_zoo.__file__
import transformers.models.gemma4.modeling_gemma4 as g
from transformers.modeling_layers import GradientCheckpointingLayer
stock = g.Gemma4TextDecoderLayer.__dict__["forward"]

def apply():
    for fn in TEMPORARY_PATCHES:
        try:
            fn()
        except Exception:
            pass

def run():
    cfg = g.Gemma4TextConfig(vocab_size = 256, hidden_size = 64, intermediate_size = 48,
        num_hidden_layers = 4, num_attention_heads = 4, num_key_value_heads = 2, head_dim = 16, global_head_dim = 16,
        max_position_embeddings = 128, num_kv_shared_layers = 2, hidden_size_per_layer_input = 8,
        vocab_size_per_layer_input = 256, layer_types = ["sliding_attention", "full_attention"] * 2,
        sliding_window = 64, tie_word_embeddings = False, use_cache = False, pad_token_id = 0, bos_token_id = 1,
        eos_token_id = 1)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    model = g.Gemma4ForCausalLM(cfg).to(torch.float16).train()
    for p in model.parameters():
        p.requires_grad_(False)
    for mod in model.model.layers.modules():
        if isinstance(mod, torch.nn.Linear):
            mod.weight.requires_grad_(True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
    calls = [0]
    for layer in model.model.layers:
        layer.mlp.register_forward_hook(lambda *a: calls.__setitem__(0, calls[0] + 1))
    ids = torch.randint(2, 256, (2, 12), generator = torch.Generator().manual_seed(0))
    out = model(input_ids = ids, labels = ids)
    forward_calls = calls[0]
    out.loss.backward()
    return {"layers": len(model.model.layers), "forward": forward_calls, "total": calls[0], "loss": out.loss.item(),
            "finite": all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)}

apply()
res = {"checkpoints_in_forward": "__init_subclass__" in GradientCheckpointingLayer.__dict__,
       "boundary": bool(getattr(g.Gemma4TextDecoderLayer.forward, "_unsloth_gemma4_boundary", False)),
       "patched": run()}
apply()  # idempotent: a second pass must not stack another scope or checkpoint wrapper
res["patched_twice"] = run()
os.environ["UNSLOTH_FORCE_FLOAT32"] = "0"  # a later flag "0" load undoes the scope and restores the stock forward
apply()
res["undone_is_stock"] = g.Gemma4TextDecoderLayer.__dict__["forward"] is stock
res["undone"] = run()
os.environ["UNSLOTH_FORCE_FLOAT32"] = "1"
apply()
import transformers.models.qwen3_moe.modeling_qwen3_moe  # noqa: F401  (patched by qwen3_moe_float32)
apply()
def subclasses(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from subclasses(sub)
res["lost_checkpointing"] = sorted(
    f"{cls.__module__}.{cls.__qualname__}" for cls in set(subclasses(GradientCheckpointingLayer))
    if "forward" in cls.__dict__ and getattr(cls.__dict__["forward"].__code__, "co_name", "") != "forward_with_checkpointing"
) if res["checkpoints_in_forward"] else []
print(json.dumps(res))
'''


def _run(out):
    env = {k: v for k, v in os.environ.items() if not k.startswith("UNSLOTH_")}
    env.update(HF_HUB_OFFLINE = "1", OMP_NUM_THREADS = "2", REPO = REPO,
               UNSLOTH_COMPILE_LOCATION = os.path.join(out, "ucc"),
               PYTHONPATH = os.pathsep.join(p for p in (REPO, os.environ.get("PYTHONPATH")) if p),
               CUDA_VISIBLE_DEVICES = "", UNSLOTH_IS_PRESENT = "1", UNSLOTH_ALLOW_CPU = "1",
               UNSLOTH_ZOO_DISABLE_GPU_INIT = "1", UNSLOTH_FORCE_FLOAT32 = "1", TORCHDYNAMO_DISABLE = "1")
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(SCRIPT)], env = env, cwd = out,
                          capture_output = True, text = True, timeout = 900)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    return json.loads([ln for ln in proc.stdout.splitlines() if ln.startswith("{")][-1])


def test_gemma4_forced_float32_layers_are_recomputed(tmp_path):
    try:
        import transformers.models.gemma4.modeling_gemma4  # noqa: F401
    except Exception:
        pytest.skip("transformers without Gemma-4")
    res = _run(str(tmp_path))
    assert res["boundary"], res
    for key in ("patched", "patched_twice", "undone"):
        r = res[key]
        assert r["finite"], (key, res)
        assert r["forward"] == r["layers"], (key, res)
        assert r["total"] == 2 * r["layers"], (key, res)
    assert res["patched_twice"]["loss"] == res["patched"]["loss"], res
    assert res["undone_is_stock"], res
    assert res["lost_checkpointing"] == [], res["lost_checkpointing"]
