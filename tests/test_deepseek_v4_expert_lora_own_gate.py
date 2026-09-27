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

"""DeepSeek-V4's bf16 experts have their own clamped-SwiGLU `_apply_gate`. The dispatcher sent such
classes to transformers' grouped_mm, which never reads Unsloth's expert LoRA: the routed-expert
lora_A / lora_B got no gradient and the experts output ignored the adapter. They must go through the
moe_utils backend with the class's own gate, matching the eager forward on W + PEFT's delta."""

import json
import os
import subprocess
from pathlib import Path
import sys

import pytest
import torch

pytest.importorskip("transformers.models.deepseek_v4.modeling_deepseek_v4")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "FastModel needs a GPU")

_CHILD = r'''
import os, sys, json, copy
out_dir = sys.argv[1]
os.environ["UNSLOTH_COMPILE_LOCATION"] = os.path.join(out_dir, "unsloth_compiled_cache")
import unsloth
from unsloth import FastModel
import torch
from transformers import DeepseekV4Config, DeepseekV4ForCausalLM, PreTrainedTokenizerFast
from transformers.models.deepseek_v4 import modeling_deepseek_v4 as M
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

ckpt = os.path.join(out_dir, "tiny_dsv4")
cfg = DeepseekV4Config(vocab_size = 512, hidden_size = 128, num_hidden_layers = 4, num_attention_heads = 4,
    num_key_value_heads = 1, head_dim = 64, qk_rope_head_dim = 16, q_lora_rank = 64, o_lora_rank = 32, o_groups = 2,
    n_routed_experts = 8, n_shared_experts = 1, num_experts_per_tok = 2, moe_intermediate_size = 96,
    index_n_heads = 2, index_head_dim = 32, index_topk = 4, sliding_window = 16, compress_ratios = [0, 128, 4, 128],
    num_hash_layers = 1, max_position_embeddings = 512, bos_token_id = 0, eos_token_id = 1, pad_token_id = 2)
torch.manual_seed(0)
DeepseekV4ForCausalLM(cfg).to(torch.bfloat16).save_pretrained(ckpt)
vocab = {f"t{i}": i for i in range(512)}
tk = Tokenizer(WordLevel(vocab, unk_token = "t3")); tk.pre_tokenizer = Whitespace()
PreTrainedTokenizerFast(tokenizer_object = tk, bos_token = "t0", eos_token = "t1", pad_token = "t2",
                        unk_token = "t3").save_pretrained(ckpt)

model, _ = FastModel.from_pretrained(ckpt, max_seq_length = 128, dtype = torch.bfloat16, load_in_4bit = False,
                                     load_in_16bit = True, device_map = {"": 0})
model = FastModel.get_peft_model(model, r = 8, lora_alpha = 16, lora_dropout = 0, bias = "none",
    target_modules = ["q_a_proj", "q_b_proj", "kv_proj", "o_b_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing = False, random_state = 3407)
torch.manual_seed(1)
for n, p in model.named_parameters():
    if "lora_B" in n:
        p.data.normal_(0, 0.02)
experts = [(n, m) for n, m in model.named_modules() if type(m).__name__ == "DeepseekV4Experts"]
cap = {}
def hook(name):
    def f(mod, args, kwargs, out):
        cap[name] = ([t.detach().clone() for t in args], out.detach().clone())
    return f
for n, m in experts:
    m.register_forward_hook(hook(n), with_kwargs = True)
ids = torch.randint(4, 500, (2, 48), device = "cuda")
model.train()
with torch.autocast("cuda", dtype = torch.bfloat16):
    loss = model(input_ids = ids, labels = ids).loss
loss.backward()
grads = [float(p.grad.abs().max()) if p.grad is not None else 0.0
         for n, p in model.named_parameters() if "lora_" in n and ".mlp.experts." in n]
wrappers = {n: m for n, m in model.named_modules() if type(m).__name__ == "ParamWrapper"}
eager = getattr(M.DeepseekV4Experts.forward, "__wrapped__", M.DeepseekV4Experts.forward)
rel = []
for n, m in experts:
    (h, idx, w), got = cap[n]
    ref = copy.deepcopy(m).float()
    prefix = n.rsplit(".experts", 1)[0] + ".experts"
    for wn, wm in wrappers.items():
        if wn.startswith(prefix):
            getattr(ref, wm.parameter_name).data += wm.get_delta_weight("default").float()
    with torch.no_grad():
        want = eager(ref, h.float(), idx, w.float())
    rel.append(float((got.float() - want).norm() / want.norm()))
print("RESULT " + json.dumps({"n_expert_lora": len(grads), "grads": grads, "rel": rel}))
'''


def test_deepseek_v4_bf16_expert_lora_is_applied(tmp_path):
    script = tmp_path / "child.py"
    script.write_text(_CHILD)
    # The script lives in tmp_path, so without this the child would import the installed
    # unsloth_zoo rather than the tree under test.
    env = dict(os.environ)
    repo_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join([repo_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    proc = subprocess.run([sys.executable, str(script), str(tmp_path)], capture_output = True, text = True,
                          timeout = 900, env = env)
    line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
    assert line is not None, proc.stdout[-3000:] + proc.stderr[-3000:]
    r = json.loads(line[len("RESULT "):])
    assert r["n_expert_lora"] == 16  # 4 layers x (gate_up, down) x (A, B)
    assert all(g > 0 for g in r["grads"]), r["grads"]
    # bf16 backend vs fp32 eager reference on W + delta; ignoring the adapter is about 0.5
    assert max(r["rel"]) < 0.02, r["rel"]
