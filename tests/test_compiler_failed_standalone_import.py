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

"""A torch module whose standalone compiled copy fails must still be importable from the generated module.
On transformers 4.x, Qwen3-MoE float16 (FORCE_FLOAT32 patches) failed to copy Qwen3MoeRMSNorm, which was then
neither defined nor imported, and loading died with NameError in Qwen3MoeAttention.__init__."""

import json
import os
import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "FastLanguageModel needs a GPU")

_CHILD = r'''
import os, sys, json
dtype_name, out_dir = sys.argv[1], sys.argv[2]
os.environ["UNSLOTH_COMPILE_LOCATION"] = os.path.join(out_dir, "unsloth_compiled_cache")
import unsloth
from unsloth import FastLanguageModel
import torch
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
dtype = getattr(torch, dtype_name)
ckpt = os.path.join(out_dir, "tiny")
cfg = AutoConfig.for_model("qwen3_moe", vocab_size = 512, bos_token_id = 0, eos_token_id = 1, pad_token_id = 2,
                           hidden_size = 64, intermediate_size = 128, moe_intermediate_size = 64, num_hidden_layers = 2,
                           num_attention_heads = 4, num_key_value_heads = 2, head_dim = 16, num_experts = 4,
                           num_experts_per_tok = 2)
torch.manual_seed(0)
AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16).save_pretrained(ckpt)
tk = Tokenizer(WordLevel({f"t{i}": i for i in range(512)}, unk_token = "t3")); tk.pre_tokenizer = Whitespace()
PreTrainedTokenizerFast(tokenizer_object = tk, bos_token = "t0", eos_token = "t1", pad_token = "t2",
                        unk_token = "t3").save_pretrained(ckpt)
err, loss = None, None
try:
    model, _ = FastLanguageModel.from_pretrained(ckpt, max_seq_length = 128, dtype = dtype, load_in_4bit = False)
    model = FastLanguageModel.get_peft_model(model, r = 8, lora_alpha = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    ids = torch.randint(4, 500, (2, 32), device = "cuda")
    out = model(input_ids = ids, labels = ids)
    out.loss.backward()
    loss = float(out.loss)
except Exception as e:
    err = f"{type(e).__name__}: {str(e)[:300]}"
print("RESULT " + json.dumps({"err": err, "loss": loss}))
'''


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_qwen3_moe_loads_and_steps(tmp_path, dtype_name):
    pytest.importorskip("transformers.models.qwen3_moe.modeling_qwen3_moe")
    script = tmp_path / "child.py"
    script.write_text(_CHILD)
    env = dict(os.environ)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join([repo_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.pop("UNSLOTH_COMPILE_DISABLE", None)   # the generated module is what is under test
    proc = subprocess.run([sys.executable, str(script), dtype_name, str(tmp_path)],
                          capture_output = True, text = True, timeout = 1200, env = env)
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
    assert line is not None, proc.stdout[-3000:] + proc.stderr[-3000:]
    r = json.loads(line[len("RESULT "):])
    assert r["err"] is None, r["err"]
    assert r["loss"] is not None and r["loss"] == r["loss"] and r["loss"] < float("inf"), r
