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

"""MoE expert LoRA must apply exactly the adapters PEFT would; checked on CPU against PEFT per adapter and merged."""

import json
import os
import subprocess
import sys
import textwrap

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CODE = r'''
import json, os, sys
import torch
family, work, compiled = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
import unsloth_zoo, unsloth_zoo.temporary_patches
from unsloth_zoo.temporary_patches.common import TEMPORARY_PATCHES
assert unsloth_zoo.__file__.startswith(os.environ["REPO"]), unsloth_zoo.__file__
for fn in TEMPORARY_PATCHES:
    try:
        fn()
    except Exception:
        pass
from peft import LoraConfig, PeftModel, get_peft_model

IDS = torch.randint(2, 256, (2, 12), generator = torch.Generator().manual_seed(0))

def build():
    torch.manual_seed(0)
    if family == "llama4":
        from transformers import Llama4TextConfig, Llama4ForCausalLM
        cfg = Llama4TextConfig(vocab_size = 256, hidden_size = 64, intermediate_size = 40, intermediate_size_mlp = 64,
            num_hidden_layers = 2, num_attention_heads = 4, num_key_value_heads = 2, head_dim = 16,
            num_local_experts = 4, num_experts_per_tok = 1, moe_layers = [0, 1], no_rope_layers = [1, 1],
            attention_chunk_size = 64, max_position_embeddings = 128, tie_word_embeddings = False, use_cache = False,
            pad_token_id = 0, bos_token_id = 1, eos_token_id = 1)
        return Llama4ForCausalLM._from_config(cfg, dtype = torch.float32)
    if family == "qwen3_moe":
        from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
        cfg = Qwen3MoeConfig(vocab_size = 256, hidden_size = 64, intermediate_size = 48, moe_intermediate_size = 40,
            num_hidden_layers = 2, num_attention_heads = 4, num_key_value_heads = 2, head_dim = 16, num_experts = 4,
            num_experts_per_tok = 2, max_position_embeddings = 128, use_cache = False, pad_token_id = 0,
            tie_word_embeddings = False)
        return Qwen3MoeForCausalLM._from_config(cfg, dtype = torch.float32)
    from transformers import GptOssConfig, GptOssForCausalLM
    cfg = GptOssConfig(vocab_size = 256, hidden_size = 64, intermediate_size = 40, num_hidden_layers = 2,
        num_attention_heads = 4, num_key_value_heads = 2, head_dim = 16, num_local_experts = 4,
        num_experts_per_tok = 2, max_position_embeddings = 128, use_cache = False, pad_token_id = 0,
        tie_word_embeddings = False, layer_types = ["full_attention", "sliding_attention"], sliding_window = 8)
    model = GptOssForCausalLM._from_config(cfg, dtype = torch.float32)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "experts" in name:
                p.normal_(0, 0.05)
    return model

experts = dict(target_modules = ["q_proj"], target_parameters = ["experts.gate_up_proj", "experts.down_proj"])
kinds = {"moe": (experts, 3), "moe2": (experts, 4), "attn": (dict(target_modules = ["q_proj", "v_proj"]), 5)}
for kind, (targets, seed) in kinds.items():
    model = build()
    torch.manual_seed(seed)
    get_peft_model(model, LoraConfig(r = 3, lora_alpha = 8, init_lora_weights = False, **targets)) \
        .save_pretrained(os.path.join(work, kind))

def run(model):
    model.eval()
    fn = torch.compile(model, backend = "aot_eager", dynamic = False) if compiled else model
    with torch.no_grad():
        return fn(input_ids = IDS).logits.float()

err = lambda x, y: float((x - y).abs().max())
want = {kind: run(PeftModel.from_pretrained(build(), os.path.join(work, kind))) for kind in kinds}
want["base"] = run(build())
out = {}
for a, b in (("moe", "moe2"), ("attn", "moe"), ("moe", "attn")):
    tag = a + "+" + b
    model = PeftModel.from_pretrained(build(), os.path.join(work, a), adapter_name = "a")
    model.load_adapter(os.path.join(work, b), adapter_name = "b")
    for step, name in enumerate(("a", "b", "a")):
        model.set_adapter(name)
        out[f"peft {tag} {step}{name}"] = err(run(model), want[a if name == "a" else b])
    model.set_adapter("b")
    with model.disable_adapter():
        out[f"peft {tag} disabled"] = err(run(model), want["base"])
    model.merge_adapter()
    out[f"peft {tag} merged b"] = err(run(model), want[b])
    model.unmerge_adapter()
    out[f"peft {tag} unmerged b"] = err(run(model), want[b])
    model.base_model.set_adapter(["a", "b"])
    both = run(model)
    model.merge_adapter()
    out[f"peft {tag} both active vs merged"] = err(both, run(model))
    model.unmerge_adapter()

    # transformers' own adapter API drives the same layers.
    model = build()
    model.load_adapter(os.path.join(work, a), adapter_name = "a")
    model.load_adapter(os.path.join(work, b), adapter_name = "b")
    for step, name in enumerate(("a", "b", "a")):
        model.set_adapter(name)
        out[f"transformers {tag} {step}{name}"] = err(run(model), want[a if name == "a" else b])
    model.disable_adapters()
    out[f"transformers {tag} disabled"] = err(run(model), want["base"])
print(json.dumps(out))
'''


def _multi_target_parameter_adapters_supported():
    # PEFT 0.18 refuses a second target_parameters adapter on one model.
    try:
        import peft
        from packaging.version import Version
    except Exception:
        return False
    return Version(peft.__version__) >= Version("0.19")


def _run(family, work, compiled):
    env = {k: v for k, v in os.environ.items() if not k.startswith("UNSLOTH_")}
    env.update(HF_HUB_OFFLINE = "1", OMP_NUM_THREADS = "4", REPO = REPO, PYTHONPATH = REPO,
               UNSLOTH_IS_PRESENT = "1", UNSLOTH_COMPILE_LOCATION = os.path.join(work, "ucc"),
               CUDA_VISIBLE_DEVICES = "")
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(CODE), family, work, "1" if compiled else "0"],
                          env = env, cwd = work, capture_output = True, text = True, timeout = 1800)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    assert lines, proc.stdout[-2000:] + proc.stderr[-4000:]
    return json.loads(lines[-1])


@pytest.mark.skipif(not _multi_target_parameter_adapters_supported(), reason = "PEFT < 0.19: one target_parameters adapter per model")
@pytest.mark.parametrize(
    "family, compiled",
    [("llama4", False), ("qwen3_moe", False), ("gpt_oss", False), ("qwen3_moe", True)],
    ids = ["llama4", "qwen3_moe", "gpt_oss", "qwen3_moe-compiled"],
)
def test_moe_expert_lora_follows_active_adapter(tmp_path, family, compiled):
    out = _run(family, str(tmp_path), compiled)
    assert len(out) == 3 * 11, out
    wrong = {key: value for key, value in out.items() if not value <= 1e-5}
    assert not wrong, wrong
