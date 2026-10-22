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

"""Expert LoRA trains with no graph breaks on experts-interface families outside the own-gate route: default
gate (GLM-4.7-Flash), non-gated stacks (NemotronH), a cast router (Ernie 4.5) and Jamba's decoder layer mapping."""

import json
import os
import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "FastModel needs a GPU")

_CONFIGS = {
    "glm4_moe_lite": dict(hidden_size = 64, intermediate_size = 128, moe_intermediate_size = 64, num_hidden_layers = 2,
        mlp_layer_types = ["dense", "sparse"], num_attention_heads = 4, num_key_value_heads = 4, kv_lora_rank = 32,
        q_lora_rank = 32, qk_nope_head_dim = 16, qk_rope_head_dim = 16, v_head_dim = 16, n_routed_experts = 4,
        num_experts_per_tok = 2, n_group = 1, topk_group = 1),
    "nemotron_h": dict(hidden_size = 64, intermediate_size = 128, moe_intermediate_size = 64, head_dim = 16,
        layers_block_type = ["linear_attention", "moe", "full_attention", "mlp"], mamba_num_heads = 8,
        mamba_head_dim = 16, n_groups = 2, ssm_state_size = 16, num_attention_heads = 4, num_key_value_heads = 2,
        n_routed_experts = 4, num_experts_per_tok = 2, moe_shared_expert_intermediate_size = 128, n_group = 1,
        topk_group = 1, chunk_size = 16),
    "ernie4_5_moe": dict(hidden_size = 64, intermediate_size = 128, moe_intermediate_size = 64, num_hidden_layers = 2,
        num_attention_heads = 4, num_key_value_heads = 2, moe_num_experts = 4, moe_k = 2, moe_layer_start_index = 0,
        moe_num_shared_experts = 2),
    "jamba": dict(hidden_size = 64, intermediate_size = 128, num_hidden_layers = 2, num_attention_heads = 4,
        num_key_value_heads = 2, num_experts = 4, num_experts_per_tok = 2, attn_layer_offset = 1, attn_layer_period = 2,
        expert_layer_offset = 1, expert_layer_period = 2, mamba_dt_rank = 8, use_mamba_kernels = False),
}

_CHILD = r'''
import os, sys, json, copy
model_type, out_dir = sys.argv[1], sys.argv[2]
overrides = json.loads(sys.argv[3])
os.environ["UNSLOTH_COMPILE_LOCATION"] = os.path.join(out_dir, "unsloth_compiled_cache")
import unsloth
from unsloth import FastModel
import torch
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

ckpt = os.path.join(out_dir, "tiny")
cfg = AutoConfig.for_model(model_type, vocab_size = 512, bos_token_id = 0, eos_token_id = 1, pad_token_id = 2, **overrides)
torch.manual_seed(0)
AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16).save_pretrained(ckpt)
tk = Tokenizer(WordLevel({f"t{i}": i for i in range(512)}, unk_token = "t3")); tk.pre_tokenizer = Whitespace()
PreTrainedTokenizerFast(tokenizer_object = tk, bos_token = "t0", eos_token = "t1", pad_token = "t2",
                        unk_token = "t3").save_pretrained(ckpt)

model, _ = FastModel.from_pretrained(ckpt, max_seq_length = 128, dtype = torch.bfloat16, load_in_4bit = False,
                                     load_in_16bit = True, device_map = {"": 0})
model = FastModel.get_peft_model(model, r = 8, lora_alpha = 16, lora_dropout = 0, bias = "none",
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing = "unsloth", random_state = 3407)
torch.manual_seed(1)
for n, p in model.named_parameters():
    if "lora_B" in n:
        p.data.normal_(0, 0.02)
experts = [(n, m) for n, m in model.named_modules()
           if type(m).__name__.endswith("Experts") and (hasattr(m, "gate_up_proj") or hasattr(m, "up_proj"))]
cap = {}
def hook(name):
    def f(mod, args, kwargs, out):
        if torch.is_grad_enabled():
            cap[name] = ([t.detach().clone() for t in args], out.detach().clone())
    return f
for n, m in experts:
    m.register_forward_hook(hook(n), with_kwargs = True)
from torch._dynamo.utils import counters
counters.clear()
model.train()
ids = torch.randint(4, 500, (2, 48), device = "cuda")
with torch.autocast("cuda", dtype = torch.bfloat16):
    loss = model(input_ids = ids, labels = ids).loss
loss.backward()
grads = [float(p.grad.abs().max()) if p.grad is not None else 0.0
         for n, p in model.named_parameters() if "lora_" in n and ".experts." in n]
wrappers = {n: m for n, m in model.named_modules() if type(m).__name__ == "ParamWrapper"}
rel = []
for n, m in experts:
    (args, got) = cap[n]
    ref = copy.deepcopy(m).float()
    prefix = n.rsplit(".experts", 1)[0] + ".experts"
    for wn, wm in wrappers.items():
        if wn.startswith(prefix):
            getattr(ref, wm.parameter_name).data += wm.get_delta_weight("default").float()
    eager = getattr(type(m).forward, "__wrapped__", type(m).forward)
    with torch.no_grad():
        want = eager(ref, args[0].float(), args[1], args[2].float())
    rel.append(float((got.float() - want).norm() / want.norm()))
print("RESULT " + json.dumps({"experts": len(experts), "grads": grads, "rel": rel, "loss": float(loss),
                              "breaks": [str(k)[:300] for k in counters["graph_break"]]}))
'''


@pytest.mark.parametrize("model_type", sorted(_CONFIGS))
def test_expert_lora_trains_without_graph_breaks(tmp_path, model_type):
    pytest.importorskip(f"transformers.models.{model_type}.modeling_{model_type}")
    script = tmp_path / "child.py"
    script.write_text(_CHILD)
    env = dict(os.environ)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join([repo_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    # Compiling is what is under test; an inherited UNSLOTH_COMPILE_DISABLE=1 makes "no graph breaks" vacuous.
    env.pop("UNSLOTH_COMPILE_DISABLE", None)
    proc = subprocess.run([sys.executable, str(script), model_type, str(tmp_path), json.dumps(_CONFIGS[model_type])],
                          capture_output = True, text = True, timeout = 1200, env = env)
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
    assert line is not None, proc.stdout[-3000:] + proc.stderr[-3000:]
    r = json.loads(line[len("RESULT "):])
    if r["experts"] == 0:
        # transformers 4.57 keeps per-expert ModuleLists here: the step ran, nothing stacked to check.
        pytest.skip(reason = f"{model_type} has no stacked experts in this transformers")
    assert r["grads"] and all(g > 0 for g in r["grads"]), r["grads"]
    assert max(r["rel"]) < 0.02, r["rel"]
    assert r["breaks"] == [], r["breaks"]
