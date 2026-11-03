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

"""Qwen3-MoE is in FORCE_FLOAT32: on a GPU without bfloat16 an fp16 request sets UNSLOTH_FORCE_FLOAT32=1, and full
finetuning then trains the model in float32 with no autocast. The float32 patches hard-cast the RMSNorm and attention
outputs to float16, so the first forward died with "expected mat1 and mat2 to have the same dtype ... Half != float"
at q_proj (Kaggle T4, compose harness). Now each Linear boundary follows its weights: a float32 model matches vanilla
transformers in float32, and the fp16 paths (autocast training, no-autocast eval) are unchanged.

CPU subprocesses: the patches gate on UNSLOTH_FORCE_FLOAT32 at patch time, and the reference never imports the zoo.
"""

import json
import os
import subprocess
import sys
import textwrap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COMMON = r'''
import json, os, sys, torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
OUT = sys.argv[1]
IDS = torch.randint(2, 256, (2, 12), generator = torch.Generator().manual_seed(0))

def build(dtype):
    cfg = Qwen3MoeConfig(vocab_size = 256, hidden_size = 64, intermediate_size = 48, moe_intermediate_size = 40,
        num_hidden_layers = 2, num_attention_heads = 4, num_key_value_heads = 2, head_dim = 16, num_experts = 4,
        num_experts_per_tok = 2, max_position_embeddings = 128, tie_word_embeddings = False, use_cache = False,
        pad_token_id = 0, mlp_only_layers = [1])
    torch.manual_seed(0)
    return Qwen3MoeForCausalLM(cfg).to(dtype)

def full_ft(model):
    model.train()
    out = model(input_ids = IDS, labels = IDS)
    out.loss.backward()
    grads = {n: p.grad.float() for n, p in model.named_parameters() if p.grad is not None}
    return out.loss.item(), grads
'''

PATCHED = COMMON + r'''
import unsloth_zoo, unsloth_zoo.temporary_patches
from unsloth_zoo.temporary_patches.common import TEMPORARY_PATCHES
assert unsloth_zoo.__file__.startswith(os.environ["REPO"]), unsloth_zoo.__file__
for fn in TEMPORARY_PATCHES:
    try:
        fn()
    except Exception:
        pass
import transformers.models.qwen3_moe.modeling_qwen3_moe as q
assert "float32" in q.Qwen3MoeAttention.forward.__code__.co_filename, "float32 patches not applied"
loss, grads = full_ft(build(torch.float32))
torch.save(grads, os.path.join(OUT, "patched.pt"))
# The fp16 paths the patches exist for must still run: autocast training and no-autocast eval.
m16 = build(torch.float16)
with torch.autocast("cpu", dtype = torch.float16):
    loss16 = m16(input_ids = IDS, labels = IDS).loss.item()
m16.eval()
with torch.no_grad():
    eval_dtype = str(m16(input_ids = IDS).logits.dtype)
print(json.dumps({"loss": loss, "loss16": loss16, "eval_dtype": eval_dtype}))
'''

VANILLA = COMMON + r'''
assert "unsloth_zoo" not in sys.modules
loss, grads = full_ft(build(torch.float32))
torch.save(grads, os.path.join(OUT, "vanilla.pt"))
print(json.dumps({"loss": loss}))
'''


def _run(code, out, zoo):
    env = {k: v for k, v in os.environ.items() if not k.startswith("UNSLOTH_")}
    env.update(HF_HUB_OFFLINE = "1", OMP_NUM_THREADS = "4", REPO = REPO,
               UNSLOTH_COMPILE_LOCATION = os.path.join(out, "ucc"))
    if zoo:
        env.update(PYTHONPATH = REPO, UNSLOTH_IS_PRESENT = "1", UNSLOTH_FORCE_FLOAT32 = "1")
    else:
        env.pop("PYTHONPATH", None)
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(code), out], env = env, cwd = out,
                          capture_output = True, text = True, timeout = 1800)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    return json.loads([ln for ln in proc.stdout.splitlines() if ln.startswith("{")][-1])


def test_forced_float32_full_finetune_matches_vanilla_float32(tmp_path):
    import torch

    out = str(tmp_path)
    patched = _run(PATCHED, out, zoo = True)
    vanilla = _run(VANILLA, out, zoo = False)
    assert abs(patched["loss"] - vanilla["loss"]) <= 1e-4 * abs(vanilla["loss"]), (patched, vanilla)
    got = torch.load(os.path.join(out, "patched.pt"))
    want = torch.load(os.path.join(out, "vanilla.pt"))
    assert set(got) == set(want)
    for name, g in want.items():
        torch.testing.assert_close(got[name], g, rtol = 1e-3, atol = 1e-5 * float(g.abs().max() or 1.0), msg = name)
    assert abs(patched["loss16"] - vanilla["loss"]) < 0.05, patched
    assert patched["eval_dtype"] == "torch.float16", patched


INTERLEAVED = COMMON + r'''
import threading
import unsloth_zoo
assert unsloth_zoo.__file__.startswith(os.environ["REPO"]), unsloth_zoo.__file__
import unsloth_zoo.temporary_patches.qwen3_moe_float32 as patch
for fn in (patch.patch_Qwen3MoeDecoderLayer_float32, patch.patch_Qwen3MoeRMSNorm_float32,
           patch.patch_Qwen3MoeAttention_float32):
    fn()
m16, m32 = build(torch.float16).eval(), build(torch.float32).eval()
ready, done, errors = threading.Event(), threading.Event(), {}
def hold(module, args):  # the fp16 model waits at its final norm while the fp32 model runs every layer
    ready.set()
    assert done.wait(60)
m16.model.norm.register_forward_pre_hook(hold)
def half():
    try:
        with torch.no_grad(): errors["half_dtype"] = str(m16(IDS).logits.dtype)
    except Exception as e: errors["half"] = str(e)
def full():
    try:
        assert ready.wait(60)
        with torch.no_grad(): m32(IDS)
    except Exception as e: errors["float"] = str(e)
    finally: done.set()
a, b = threading.Thread(target = half), threading.Thread(target = full)
a.start(); b.start(); a.join(); b.join()
print(json.dumps(errors))
'''


def test_float16_and_float32_models_in_two_threads_keep_their_own_norm_dtype(tmp_path):
    # Failing before: the boundary dtype was one process-wide value, so a float32 model running between an fp16
    # model's last layer and its final norm made that norm return float32 to the fp16 lm_head ("float != Half").
    got = _run(INTERLEAVED, str(tmp_path), zoo = True)
    assert got == {"half_dtype": "torch.float16"}, got
