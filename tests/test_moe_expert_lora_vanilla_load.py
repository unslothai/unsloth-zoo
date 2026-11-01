"""An expert-LoRA adapter trained under Unsloth on Llama-4 / Qwen3-VL-MoE must load in vanilla transformers + PEFT
and give the model Unsloth trained, and must still reload under Unsloth.

Unsloth flags Llama4TextExperts is_transposed and stores Qwen3VLMoeTextExperts transposed, so PEFT factors their
stacks over swapped dims; without moe_lora_factor_swap the saved lora_A / lora_B do not fit vanilla PEFT (size
mismatch). Three CPU subprocesses: vanilla builds and saves a tiny non-square checkpoint, Unsloth (zoo patches)
loads it, attaches a random expert LoRA, saves it and its logits, vanilla reloads the adapter.
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COMMON = r'''
import json, os, sys, torch
FAMILY, WORK = sys.argv[1], sys.argv[2]
CKPT = os.path.join(WORK, "ckpt")
ADAPTER = os.path.join(WORK, "adapter")
IDS = torch.randint(2, 256, (2, 12), generator = torch.Generator().manual_seed(0))

def config():
    if FAMILY == "llama4":
        from transformers import Llama4TextConfig
        return Llama4TextConfig(vocab_size = 256, hidden_size = 64, intermediate_size = 40, intermediate_size_mlp = 64,
            num_hidden_layers = 2, num_attention_heads = 4, num_key_value_heads = 2, head_dim = 16,
            num_local_experts = 4, num_experts_per_tok = 1, moe_layers = [0, 1], no_rope_layers = [1, 1],
            attention_chunk_size = 64, max_position_embeddings = 128, tie_word_embeddings = False, use_cache = False,
            pad_token_id = 0, bos_token_id = 1, eos_token_id = 1)
    from transformers import Qwen3VLMoeConfig
    text = dict(vocab_size = 256, hidden_size = 64, intermediate_size = 48, moe_intermediate_size = 40,
        num_hidden_layers = 2, num_attention_heads = 4, num_key_value_heads = 2, head_dim = 16, num_experts = 4,
        num_experts_per_tok = 2, max_position_embeddings = 128, use_cache = False, pad_token_id = 0,
        rope_parameters = {"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [2, 3, 3]})
    vision = dict(depth = 1, hidden_size = 32, intermediate_size = 64, num_heads = 2, out_hidden_size = 64,
        deepstack_visual_indexes = [0], num_position_embeddings = 16)
    return Qwen3VLMoeConfig(text_config = text, vision_config = vision, tie_word_embeddings = False)

def model_class():
    if FAMILY == "llama4":
        from transformers import Llama4ForCausalLM
        return Llama4ForCausalLM
    from transformers import Qwen3VLMoeForConditionalGeneration
    return Qwen3VLMoeForConditionalGeneration

def logits(model):
    model.eval()
    with torch.no_grad():
        return model(input_ids = IDS).logits.float()
'''

VANILLA_SAVE = COMMON + r'''
torch.manual_seed(0)
model = model_class()._from_config(config(), dtype = torch.float32)
model.save_pretrained(CKPT)
'''

UNSLOTH_TRAIN = COMMON + r'''
import unsloth_zoo, unsloth_zoo.temporary_patches
from unsloth_zoo.temporary_patches.common import TEMPORARY_PATCHES
assert unsloth_zoo.__file__.startswith(os.environ["REPO"]), unsloth_zoo.__file__
for fn in TEMPORARY_PATCHES:
    try:
        fn()
    except Exception:
        pass
from peft import LoraConfig, PeftModel, get_peft_model
model = model_class().from_pretrained(CKPT, dtype = torch.float32)
torch.manual_seed(1)
cfg = LoraConfig(r = 4, lora_alpha = 8, target_modules = ["q_proj"], init_lora_weights = False,
                 target_parameters = ["experts.gate_up_proj", "experts.down_proj"])
model = get_peft_model(model, cfg)
want = logits(model)
model.save_pretrained(ADAPTER)
torch.save(want, os.path.join(WORK, "unsloth_logits.pt"))
fresh = PeftModel.from_pretrained(model_class().from_pretrained(CKPT, dtype = torch.float32), ADAPTER)
print(json.dumps({"reload_max_abs": float((logits(fresh) - want).abs().max())}))
'''

VANILLA_LOAD = COMMON + r'''
assert "unsloth_zoo" not in sys.modules
from peft import PeftModel
model = PeftModel.from_pretrained(model_class().from_pretrained(CKPT, dtype = torch.float32), ADAPTER)
want = torch.load(os.path.join(WORK, "unsloth_logits.pt"))
print(json.dumps({"vanilla_max_abs": float((logits(model) - want).abs().max())}))
'''


def _run(code, family, work, zoo):
    env = {k: v for k, v in os.environ.items() if not k.startswith("UNSLOTH_")}
    env.update(HF_HUB_OFFLINE = "1", OMP_NUM_THREADS = "4", REPO = REPO,
               UNSLOTH_COMPILE_LOCATION = os.path.join(work, "ucc"))
    if zoo:
        env.update(PYTHONPATH = REPO, UNSLOTH_IS_PRESENT = "1")
    else:
        env.pop("PYTHONPATH", None)
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(code), family, work], env = env, cwd = work,
                          capture_output = True, text = True, timeout = 1800)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    return json.loads(lines[-1]) if lines else {}


@pytest.mark.parametrize("family", ["llama4", "qwen3_vl_moe"])
def test_unsloth_expert_lora_loads_in_vanilla_peft(tmp_path, family):
    work = str(tmp_path)
    _run(VANILLA_SAVE, family, work, zoo = False)
    trained = _run(UNSLOTH_TRAIN, family, work, zoo = True)
    assert trained["reload_max_abs"] <= 1e-4, trained
    loaded = _run(VANILLA_LOAD, family, work, zoo = False)
    assert loaded["vanilla_max_abs"] <= 1e-4, loaded


def test_factor_swap_is_an_involution_and_keeps_the_delta():
    import torch
    from unsloth_zoo.temporary_patches.moe_lora_factor_swap import swap_expert_lora_factors

    E, R, d1, d2 = 3, 2, 5, 7
    A, B = torch.randn(E * R, d1), torch.randn(d2, R * E)
    A2, B2 = swap_expert_lora_factors(A, B, E)
    assert A2.shape == (E * R, d2) and B2.shape == (d1, R * E)
    A3, B3 = swap_expert_lora_factors(A2, B2, E)
    assert torch.equal(A3, A) and torch.equal(B3, B)
    # PEFT packing: A as (E, R, in), B as (out, R, E); delta_e = B_e @ A_e.
    def delta(a, b):
        a = a.reshape(E, R, -1)
        b = b.reshape(b.shape[0], R, E).permute(2, 0, 1)
        return b @ a
    torch.testing.assert_close(delta(A2, B2), delta(A, B).transpose(-2, -1))
