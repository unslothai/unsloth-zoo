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
out = {"reload_max_abs": float((logits(fresh) - want).abs().max())}
# Hotswap: load another adapter, then swap the saved one in (peft.utils.hotswap bypasses set_peft_model_state_dict).
other = get_peft_model(model_class().from_pretrained(CKPT, dtype = torch.float32), cfg)
other.save_pretrained(os.path.join(WORK, "other"))
from peft.utils.hotswap import hotswap_adapter
swapped = PeftModel.from_pretrained(model_class().from_pretrained(CKPT, dtype = torch.float32), os.path.join(WORK, "other"))
hotswap_adapter(swapped, ADAPTER, adapter_name = "default")
out["hotswap_max_abs"] = float((logits(swapped) - want).abs().max())
merged = swapped.merge_and_unload()
out["merged_max_abs"] = float((logits(merged) - want).abs().max())
# A PeftModel checkpoint (keys prefixed base_model.model.) loaded into a model from inject_adapter_in_model.
from peft import inject_adapter_in_model, set_peft_model_state_dict
from safetensors.torch import load_file
injected = inject_adapter_in_model(cfg, model_class().from_pretrained(CKPT, dtype = torch.float32))
set_peft_model_state_dict(injected, load_file(os.path.join(ADAPTER, "adapter_model.safetensors")))
out["injected_max_abs"] = float((logits(injected) - want).abs().max())
# transformers' own load_adapter (5.x loads the tensors itself, not through set_peft_model_state_dict).
direct = model_class().from_pretrained(CKPT, dtype = torch.float32)
direct.load_adapter(ADAPTER)
out["load_adapter_max_abs"] = float((logits(direct) - want).abs().max())
# The same through a .bin adapter (transformers reads those itself too).
model.save_pretrained(os.path.join(WORK, "adapter_bin"), safe_serialization = False)
direct_bin = model_class().from_pretrained(CKPT, dtype = torch.float32)
direct_bin.load_adapter(os.path.join(WORK, "adapter_bin"), use_safetensors = False)
out["load_adapter_bin_max_abs"] = float((logits(direct_bin) - want).abs().max())
# Failing before: a compiled wrapper's `_orig_mod.` names matched no key, and a hotswap into a lower-rank adapter
# never converted (orientation was read from the whole shape, rank included).
small = LoraConfig(r = 2, lora_alpha = 8, target_modules = ["q_proj"], init_lora_weights = False,
                   target_parameters = ["experts.gate_up_proj", "experts.down_proj"])
low = get_peft_model(model_class().from_pretrained(CKPT, dtype = torch.float32), small)
low.save_pretrained(os.path.join(WORK, "low"))
low = PeftModel.from_pretrained(model_class().from_pretrained(CKPT, dtype = torch.float32), os.path.join(WORK, "low"))
compiled = torch.compile(low, backend = "eager")
hotswap_adapter(compiled, ADAPTER, adapter_name = "default")
# Unsloth's forward reads the factors by shape, so check the merge, which needs PEFT's own layout.
try:
    out["compiled_rank_hotswap_max_abs"] = float((logits(low.merge_and_unload()) - want).abs().max())
except Exception as e:
    out["compiled_rank_hotswap_max_abs"] = f"{type(e).__name__}: {e}"[:200]
print(json.dumps(out))
'''

VANILLA_LOAD = COMMON + r'''
assert "unsloth_zoo" not in sys.modules
from peft import PeftModel
model = PeftModel.from_pretrained(model_class().from_pretrained(CKPT, dtype = torch.float32), ADAPTER)
want = torch.load(os.path.join(WORK, "unsloth_logits.pt"))
print(json.dumps({"vanilla_max_abs": float((logits(model) - want).abs().max())}))
'''


def _run(code, family, work, zoo, **extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("UNSLOTH_")}
    env.update(HF_HUB_OFFLINE = "1", OMP_NUM_THREADS = "4", REPO = REPO,
               UNSLOTH_COMPILE_LOCATION = os.path.join(work, "ucc"))
    if zoo:
        env.update(PYTHONPATH = REPO, UNSLOTH_IS_PRESENT = "1", **extra)
    else:
        env.pop("PYTHONPATH", None)
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(code), family, work], env = env, cwd = work,
                          capture_output = True, text = True, timeout = 1800)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    return json.loads(lines[-1]) if lines else {}


@pytest.mark.parametrize("b_layout", ["rank_major", "grouped_by_expert"])
@pytest.mark.parametrize("family", ["llama4", "qwen3_vl_moe"])
def test_unsloth_expert_lora_loads_in_vanilla_peft(tmp_path, family, b_layout):
    # grouped_by_expert (the legacy lora_B packing, UNSLOTH_MOE_LORA_B_LAYOUT) failing before: the swap read its B as
    # PEFT's rank-major packing and mixed experts, so vanilla PEFT loaded it silently with wrong logits. Failing
    # before too: transformers' load_adapter loaded the swapped factors unconverted (size mismatch).
    work = str(tmp_path)
    _run(VANILLA_SAVE, family, work, zoo = False)
    trained = _run(UNSLOTH_TRAIN, family, work, zoo = True, UNSLOTH_MOE_LORA_B_LAYOUT = b_layout)
    assert trained["reload_max_abs"] <= 1e-4, trained
    # Failing before: hotswap grew lora_A / padded lora_B instead of mapping the factors back, and prefixed keys did
    # not match an injected model's module names (size mismatch).
    for key in ("hotswap_max_abs", "merged_max_abs", "injected_max_abs", "compiled_rank_hotswap_max_abs",
                "load_adapter_max_abs", "load_adapter_bin_max_abs"):
        assert isinstance(trained[key], float) and trained[key] <= 1e-4, trained
    if b_layout == "grouped_by_expert":
        # Legacy packing is left as trained: vanilla PEFT refuses it by shape (never a silent wrong load).
        with pytest.raises(AssertionError, match = "size mismatch"):
            _run(VANILLA_LOAD, family, work, zoo = False)
        return
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


def test_llama4_factors_follow_pefts_is_transposed_support(monkeypatch):
    # Failing before: PEFT 0.18 ignores is_transposed, so Llama-4 factors already matched vanilla and swapping them
    # on save broke vanilla loading there. A physically transposed stack (Qwen3-VL-MoE) is swapped on any PEFT.
    from unsloth_zoo.temporary_patches import moe_lora_factor_swap as swap

    class Relabelled:
        _unsloth_expert_lora_factors_swapped = "is_transposed"

    class Transposed:
        _unsloth_expert_lora_factors_swapped = True

    monkeypatch.setattr(swap, "_PEFT_READS_IS_TRANSPOSED", [False])
    assert not swap._factors_swapped(Relabelled) and swap._factors_swapped(Transposed)
    monkeypatch.setattr(swap, "_PEFT_READS_IS_TRANSPOSED", [True])
    assert swap._factors_swapped(Relabelled) and swap._factors_swapped(Transposed)


QWEN3_VL_NATIVE_LAYOUT = r'''
import json, os, sys, torch
import transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe as m
LAYOUT = sys.argv[1]
if LAYOUT == "hidden_first":
    # transformers 4.x builds the stacks as (E, H, 2I) / (E, I, H): the layout Unsloth stores.
    def init(self, config):
        torch.nn.Module.__init__(self)
        self.num_experts = config.num_experts
        self.gate_up_proj = torch.nn.Parameter(torch.empty(config.num_experts, config.hidden_size, 2 * config.moe_intermediate_size))
        self.down_proj = torch.nn.Parameter(torch.empty(config.num_experts, config.moe_intermediate_size, config.hidden_size))
    m.Qwen3VLMoeTextExperts.__init__ = init
import unsloth_zoo
assert unsloth_zoo.__file__.startswith(os.environ["REPO"]), unsloth_zoo.__file__
from unsloth_zoo.temporary_patches.qwen3_vl_moe import patch_qwen3_vl_moe
patch_qwen3_vl_moe()
patch_qwen3_vl_moe()  # a re-patch must not read Unsloth's own __init__ as transformers'
print(json.dumps({"swapped": m.Qwen3VLMoeTextExperts.__dict__.get("_unsloth_expert_lora_factors_swapped")}))
'''


@pytest.mark.parametrize("layout, swapped", [("transformers", True), ("hidden_first", False)])
def test_qwen3_vl_moe_swaps_factors_only_when_transformers_stores_another_layout(tmp_path, layout, swapped):
    # Failing before: the flag was set unconditionally, so on transformers 4.x (whose own stacks are already
    # (E, H, 2I)) saves were swapped and vanilla PEFT refused a Qwen3-VL-MoE adapter it loaded before (size mismatch).
    import transformers

    if int(transformers.__version__.split(".")[0]) < 5 and layout == "transformers":
        swapped = False  # transformers 4.x itself builds the hidden-first layout
    got = _run(QWEN3_VL_NATIVE_LAYOUT.replace("sys.argv[1]", repr(layout)), "qwen3_vl_moe", str(tmp_path), zoo = True)
    assert got == {"swapped": swapped}, got
