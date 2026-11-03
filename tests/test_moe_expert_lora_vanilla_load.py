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
import transformers
TRANSFORMERS_5 = int(transformers.__version__.split(".")[0]) >= 5

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
    if not TRANSFORMERS_5:  # 4.x reads rope_scaling and rope_theta, not rope_parameters
        text.update(rope_scaling = {"rope_type": "default", "mrope_section": [2, 3, 3]}, rope_theta = 10000.0)
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
# 4.x has no use_safetensors here: PEFT's load_peft_weights falls back to the .bin itself.
direct_bin.load_adapter(os.path.join(WORK, "adapter_bin"), **({"use_safetensors": False} if TRANSFORMERS_5 else {}))
out["load_adapter_bin_max_abs"] = float((logits(direct_bin) - want).abs().max())
# A compiled wrapper (`_orig_mod.` names) hotswapped into a lower-rank adapter. PEFT < 0.20 refuses that swap
# for expert ParamWrappers even without Unsloth (Incompatible shapes), so there is nothing to check there.
import peft
from packaging.version import Version
if Version(peft.__version__) < Version("0.20"):
    out["compiled_rank_hotswap_max_abs"] = None
    print(json.dumps(out))
    sys.exit(0)
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


UNSLOTH_RELOAD = COMMON + r'''
import unsloth_zoo, unsloth_zoo.temporary_patches
from unsloth_zoo.temporary_patches.common import TEMPORARY_PATCHES
assert unsloth_zoo.__file__.startswith(os.environ["REPO"]), unsloth_zoo.__file__
for fn in TEMPORARY_PATCHES:
    try:
        fn()
    except Exception:
        pass
from peft import PeftModel
from safetensors.torch import load_file
saved = load_file(os.path.join(ADAPTER, "adapter_model.safetensors"))
expert_B = sorted({tuple(v.shape) for k, v in saved.items() if ".experts." in k and ".lora_B." in k})
model = PeftModel.from_pretrained(model_class().from_pretrained(CKPT, dtype = torch.float32), ADAPTER)
want = torch.load(os.path.join(WORK, "unsloth_logits.pt"))
print(json.dumps({"reload_max_abs": float((logits(model) - want).abs().max()), "expert_lora_B_shapes": expert_B}))
'''


LEGACY_TRAIN = COMMON + r"""
import unsloth_zoo, unsloth_zoo.temporary_patches
from unsloth_zoo.temporary_patches.common import TEMPORARY_PATCHES
assert unsloth_zoo.__file__.startswith(os.environ["REPO"]), unsloth_zoo.__file__
assert os.environ["UNSLOTH_MOE_LORA_B_LAYOUT"] == "grouped_by_expert"
for fn in TEMPORARY_PATCHES:
    try:
        fn()
    except Exception:
        pass
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file
cfg = LoraConfig(r = 4, lora_alpha = 8, target_modules = ["q_proj"], init_lora_weights = False,
                 target_parameters = ["experts.gate_up_proj", "experts.down_proj"])
for seed, name in ((1, "adapter"), (2, "other")):
    torch.manual_seed(seed)
    model = get_peft_model(model_class().from_pretrained(CKPT, dtype = torch.float32), cfg)
    torch.save(logits(model), os.path.join(WORK, name + "_logits.pt"))
    model.save_pretrained(os.path.join(WORK, name))
# An older zoo wrote the same bytes as 2D [out, R*E].
import shutil
shutil.copytree(ADAPTER, ADAPTER + "_2d")
weights = load_file(os.path.join(ADAPTER, "adapter_model.safetensors"))
shapes = sorted({tuple(v.shape) for k, v in weights.items() if ".experts." in k and ".lora_B." in k})
save_file({k: v.reshape(v.shape[0], -1).contiguous() if v.dim() == 3 else v for k, v in weights.items()},
          os.path.join(ADAPTER + "_2d", "adapter_model.safetensors"))
marker = json.load(open(os.path.join(ADAPTER, "adapter_config.json"))).get("lora_B_layout")
print(json.dumps({"expert_lora_B_shapes": shapes, "marker": marker}))
"""

LEGACY_LOAD_UNSET = COMMON + r"""
import unsloth_zoo, unsloth_zoo.temporary_patches
from unsloth_zoo.temporary_patches.common import TEMPORARY_PATCHES
assert unsloth_zoo.__file__.startswith(os.environ["REPO"]), unsloth_zoo.__file__
assert "UNSLOTH_MOE_LORA_B_LAYOUT" not in os.environ
for fn in TEMPORARY_PATCHES:
    try:
        fn()
    except Exception:
        pass
from peft import PeftModel
from peft.utils.hotswap import hotswap_adapter
SOURCE = ADAPTER + os.environ["LEGACY_SUFFIX"]
OTHER = os.path.join(WORK, "other")
want = torch.load(os.path.join(WORK, "adapter_logits.pt"))
base = lambda: model_class().from_pretrained(CKPT, dtype = torch.float32)
out = {}
out["peft_from_pretrained"] = float((logits(PeftModel.from_pretrained(base(), SOURCE)) - want).abs().max())
direct = base()
direct.load_adapter(SOURCE)
out["transformers_load_adapter"] = float((logits(direct) - want).abs().max())
if TRANSFORMERS_5:
    swapped = base()
    swapped.load_adapter(OTHER)
    swapped.load_adapter(SOURCE, adapter_name = "default", hotswap = True)
    out["transformers_hotswap"] = float((logits(swapped) - want).abs().max())
peft_swapped = PeftModel.from_pretrained(base(), OTHER)
hotswap_adapter(peft_swapped, SOURCE, adapter_name = "default")
out["peft_hotswap_adapter"] = float((logits(peft_swapped) - want).abs().max())
print(json.dumps(out))
"""


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


def _transformers_at_least_5():
    import transformers
    return int(transformers.__version__.split(".")[0]) >= 5


def _peft_at_least_0_20():
    import peft
    from packaging.version import Version
    return Version(peft.__version__) >= Version("0.20")


@pytest.mark.parametrize("b_layout", ["rank_major", "grouped_by_expert"])
@pytest.mark.parametrize("family", ["llama4", "qwen3_vl_moe"])
def test_unsloth_expert_lora_loads_in_vanilla_peft(tmp_path, family, b_layout):
    # Legacy grouped_by_expert must never load silently wrong; a factor swap alone is not enough (4.x / PEFT 0.18 swap nothing).
    work = str(tmp_path)
    _run(VANILLA_SAVE, family, work, zoo = False)
    trained = _run(UNSLOTH_TRAIN, family, work, zoo = True, UNSLOTH_MOE_LORA_B_LAYOUT = b_layout)
    assert trained["reload_max_abs"] <= 1e-4, trained
    keys = ["hotswap_max_abs", "merged_max_abs", "load_adapter_max_abs", "load_adapter_bin_max_abs"]
    if _peft_at_least_0_20():
        # PEFT < 0.20 gets both of these wrong for expert ParamWrappers without Unsloth too: a PeftModel checkpoint
        # into inject_adapter_in_model loads off by up to ~1 in the logits, and a compiled lower-rank hotswap raises.
        keys += ["injected_max_abs", "compiled_rank_hotswap_max_abs"]
    else:
        assert trained["compiled_rank_hotswap_max_abs"] is None, trained
    for key in keys:
        assert isinstance(trained[key], float) and trained[key] <= 1e-4, (key, trained)
    if b_layout == "grouped_by_expert":
        with pytest.raises(AssertionError, match = "size mismatch"):
            _run(VANILLA_LOAD, family, work, zoo = False)
        reloaded = _run(UNSLOTH_RELOAD, family, work, zoo = True)
        assert reloaded["reload_max_abs"] <= 1e-4, reloaded
        assert reloaded["expert_lora_B_shapes"] and all(len(s) == 3 for s in reloaded["expert_lora_B_shapes"]), reloaded
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


def test_legacy_grouped_lora_B_saves_3d_and_loads_back_as_trained(monkeypatch):
    import torch
    from unsloth_zoo.temporary_patches import moe_lora_factor_swap as swap

    E, R, out = 3, 2, 5
    grouped = torch.randn(out, E * R)

    class ParamWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_experts = E
            self.base_layer = torch.nn.Identity()
            self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(E * R, out, bias = False)})

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = ParamWrapper()

    model = Model()
    state = {"experts.lora_A.weight": torch.randn(E * R, 4), "experts.lora_B.weight": grouped}
    monkeypatch.setattr(swap, "_legacy_grouped", lambda wrapper, name: True)
    saved = swap._to_vanilla(model, state, "default")
    assert saved["experts.lora_B.weight"].shape == (out, E, R)
    assert torch.equal(saved["experts.lora_B.weight"][:, 1, 0], grouped[:, 1 * R + 0])
    for legacy in (True, False):
        monkeypatch.setattr(swap, "_legacy_grouped", lambda wrapper, name, legacy = legacy: legacy)
        assert torch.equal(swap._from_vanilla(model, saved, "default")["experts.lora_B.weight"], grouped)
    assert swap._to_vanilla(model, state, "default")["experts.lora_B.weight"] is grouped


def test_llama4_factors_follow_pefts_is_transposed_support(monkeypatch):
    # PEFT 0.18 ignores is_transposed, so Llama-4 factors already match vanilla there; a physically transposed
    # stack (Qwen3-VL-MoE) is swapped on any PEFT.
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
    # transformers 4.x stacks are already (E, H, 2I), so saves there must not be swapped.
    import transformers

    if int(transformers.__version__.split(".")[0]) < 5 and layout == "transformers":
        swapped = False  # transformers 4.x itself builds the hidden-first layout
    got = _run(QWEN3_VL_NATIVE_LAYOUT.replace("sys.argv[1]", repr(layout)), "qwen3_vl_moe", str(tmp_path), zoo = True)
    assert got == {"swapped": swapped}, got


def test_load_adapter_wrapper_loads_once_and_raises_real_errors(monkeypatch):
    # A real load error must propagate once, not be retried.
    from unsloth_zoo.temporary_patches import moe_lora_factor_swap as swap

    calls = []

    class _Model:
        @classmethod
        def _load_pretrained_model(cls, **kwargs):
            calls.append(kwargs)
            raise RuntimeError("real load error")

        def modules(self):
            return iter(())

        def named_modules(self):
            return iter(())

    def original(self, peft_model_id = None, adapter_name = None):
        return self._load_pretrained_model(model = self, state_dict = None, checkpoint_files = [])

    model = _Model()
    with pytest.raises(RuntimeError, match = "real load error"):
        swap._wrap_load_adapter(original)(model, "x")
    assert len(calls) == 1
    assert "_load_pretrained_model" not in model.__dict__


def test_load_adapter_keeps_the_file_path_when_nothing_converts(monkeypatch):
    # Disk offload cannot index an in-memory state dict, so unconverted loads keep the checkpoint files.
    from unsloth_zoo.temporary_patches import moe_lora_factor_swap as swap

    kwargs = {"state_dict": None, "checkpoint_files": [__file__]}
    monkeypatch.setattr(swap, "_from_vanilla", lambda model, sd, name: sd)
    monkeypatch.setattr("transformers.modeling_utils.load_state_dict", lambda f, map_location = "cpu": {"k": 1})
    assert swap._converted_load_kwargs(object(), dict(kwargs), "default") == kwargs

    class _Offloaded:
        hf_device_map = {"model.layers.0": "disk"}
    assert swap._converted_load_kwargs(_Offloaded(), dict(kwargs), "default") == kwargs


def test_load_adapter_under_disk_offload_passes_a_converted_file(monkeypatch, tmp_path):
    # Under disk offload the converted factors go to a scratch safetensors file that disk offload can index.
    import torch
    from safetensors.torch import load_file
    from unsloth_zoo.temporary_patches import moe_lora_factor_swap as swap

    class _Offloaded:
        hf_device_map = {"model": "cpu", "lm_head": "disk"}

    src = {"k": torch.ones(2)}
    monkeypatch.setattr("transformers.modeling_utils.load_state_dict", lambda f, map_location = "cpu": dict(src))
    monkeypatch.setattr(swap, "_from_vanilla", lambda model, sd, name: {"k": torch.zeros(2)})
    out = swap._converted_load_kwargs(
        _Offloaded(), {"state_dict": None, "checkpoint_files": [__file__]}, "default", str(tmp_path)
    )
    assert out["state_dict"] is None and len(out["checkpoint_files"]) == 1
    assert torch.equal(load_file(out["checkpoint_files"][0])["k"], torch.zeros(2))


@pytest.mark.parametrize("file_kind", ["3d", "2d"])
@pytest.mark.parametrize("family", ["llama4", "qwen3_vl_moe"])
def test_legacy_adapter_loads_as_trained_on_every_path_with_the_switch_unset(tmp_path, family, file_kind):
    # The config marker, not the env switch, drives the repack on every load path, for 3D and older 2D files.
    work = str(tmp_path)
    _run(VANILLA_SAVE, family, work, zoo = False)
    trained = _run(LEGACY_TRAIN, family, work, zoo = True, UNSLOTH_MOE_LORA_B_LAYOUT = "grouped_by_expert")
    assert trained["marker"] == "grouped_by_expert", trained
    assert trained["expert_lora_B_shapes"] and all(len(s) == 3 for s in trained["expert_lora_B_shapes"]), trained
    loaded = _run(LEGACY_LOAD_UNSET, family, work, zoo = True, LEGACY_SUFFIX = "" if file_kind == "3d" else "_2d")
    keys = ["peft_from_pretrained", "transformers_load_adapter", "peft_hotswap_adapter"]
    if _transformers_at_least_5():
        keys.append("transformers_hotswap")
    for key in keys:
        assert loaded[key] <= 1e-4, (key, loaded)


def test_load_adapter_source_reads_the_adapter_location_across_signatures():
    from types import SimpleNamespace
    from unsloth_zoo.temporary_patches import moe_lora_factor_swap as swap

    def v5(self, peft_model_id = None, adapter_name = None, peft_config = None, adapter_state_dict = None,
           hotswap = "auto", local_files_only = False, adapter_kwargs = None, load_config = None, **kwargs):
        pass

    def v4(self, peft_model_id = None, adapter_name = None, revision = None, token = None, peft_config = None,
           adapter_state_dict = None, hotswap = "auto", adapter_kwargs = None):
        pass

    model_id, hub = swap._load_adapter_source(v5, (None, "org/adapter"), {"adapter_kwargs": {"subfolder": "s"},
                                              "local_files_only": True, "token": "t"})
    assert model_id == "org/adapter" and hub["subfolder"] == "s" and hub["local_files_only"] and hub["token"] == "t"
    from_config = SimpleNamespace(pretrained_model_name_or_path = "org/x", download_kwargs = {"cache_dir": "/c"})
    model_id, hub = swap._load_adapter_source(v5, (None,), {"load_config": from_config})
    assert model_id == "org/x" and hub["cache_dir"] == "/c"
    assert swap._load_adapter_source(v5, (None, None), {"peft_config": object(), "adapter_state_dict": {}}) == (None, {})
    model_id, hub = swap._load_adapter_source(v4, (None, "dir", "default", "rev"), {})
    assert model_id == "dir" and hub["revision"] == "rev"
    model_id, hub = swap._load_adapter_source(v5, (None, "org/a"), {
        "load_config": from_config, "download_kwargs": {"subfolder": "saved", "revision": "branch"},
        "adapter_kwargs": {"revision": "pinned"},
    })
    assert hub["subfolder"] == "saved" and hub["revision"] == "pinned" and "cache_dir" not in hub
    assert "download_kwargs" not in hub and "load_config" not in hub
