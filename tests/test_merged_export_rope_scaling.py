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

"""merged_16bit writes the base checkpoint's config (#1073), but Unsloth's loader puts RoPE scaling and the widened
max_position_embeddings on the in-memory config only. The export must reload with the RoPE the weights trained with."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace as NS

import pytest
import torch

import _merge_e2e_helpers as H

NATIVE, WIDE = H._POS, 4 * H._POS


def _set_base_rope(base_dir, native):
    """Give the saved base checkpoint a native RoPE scheme, under the key this transformers writes."""
    path = os.path.join(base_dir, "config.json")
    with open(path) as f:
        config = json.load(f)
    if "rope_parameters" in config:  # transformers 5 keeps rope_theta inside
        config["rope_parameters"] = {**native, "rope_theta": config["rope_parameters"].get("rope_theta", 10000.0)}
    else:
        config["rope_scaling"] = dict(native)
    with open(path, "w") as f:
        json.dump(config, f)


def _export(tmp_path, family, rope=None, wide=WIDE, in_memory_rope=None, rotary=None, base_rope=None,
            rotary_from_trained=False):
    """Load like Unsloth's loader (scaling and window on the in-memory config), merge, and reload the export."""
    import transformers as T
    from peft import LoraConfig, get_peft_model

    H.set_offline_cpu_env()
    base_dir, out_dir = str(tmp_path / "base"), str(tmp_path / "out")
    H.build_and_save_base(H.make_spec(family), base_dir)
    if base_rope is not None:
        _set_base_rope(base_dir, base_rope)
    kwargs = {"rope_scaling": rope} if rope else {}
    model = T.AutoModelForCausalLM.from_pretrained(
        base_dir, max_position_embeddings=wide, dtype=torch.float32, **kwargs)
    # transformers 4.x reads rope_scaling only, which already holds the loader's dict from the kwarg.
    if in_memory_rope is not None and getattr(model.config, "rope_parameters", None) is not None:
        model.config.rope_parameters = in_memory_rope
    if rotary is not None:
        model.model.rotary_emb = rotary(model.config if rotary_from_trained else T.AutoConfig.from_pretrained(base_dir))
    pm = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj"]))
    H.seed_lora(pm)
    pm.eval()
    ids = torch.randint(0, H._VOCAB, (1, 48), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        trained = pm(ids).logits
    H.run_merge(pm, base_dir, out_dir, save_dtype=torch.float32)
    with open(os.path.join(out_dir, "config.json")) as f:
        saved = json.load(f)
    with open(os.path.join(base_dir, "config.json")) as f:
        base = json.load(f)
    with torch.no_grad():
        reloaded = T.AutoModelForCausalLM.from_pretrained(out_dir, dtype=torch.float32).eval()(ids).logits
    return saved, base, (reloaded - trained).abs().max().item()


def _rope(config):
    return config.get("rope_parameters") or config.get("rope_scaling") or {}


def _theta(config):
    # transformers 5 keeps rope_theta in rope_parameters, 4.x at the top level (the loader also puts it in rope_scaling).
    return _rope(config).get("rope_theta", config.get("rope_theta"))


# As the loader builds them: rope_theta inside, since on transformers 5 rope_scaling replaces all of rope_parameters.
LINEAR = {"rope_type": "linear", "factor": 4.0, "rope_theta": 10000.0}
YARN = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": NATIVE, "rope_theta": 10000.0}


@pytest.mark.parametrize("rope,in_memory_rope", [
    # The automatic extension past the native window: the loader's own dict, with "type" and no "rope_type".
    (LINEAR, {"type": "linear", "factor": 4.0, "rope_theta": 10000.0}),
    (YARN, None),
], ids=["linear", "yarn"])
def test_merged_export_keeps_trained_rope_scaling(tmp_path, rope, in_memory_rope):
    saved, base, diff = _export(tmp_path, "llama", rope, in_memory_rope=in_memory_rope)
    got = _rope(saved)
    assert (got.get("rope_type") or got.get("type")) == rope["rope_type"] and got.get("factor") == 4.0, got
    assert _theta(saved) == _theta(base) is not None
    assert saved["max_position_embeddings"] == WIDE
    assert diff < 1e-4, f"export reloads with different logits than the trained model: {diff:.3e}"


def test_merged_export_without_scaling_keeps_base_config(tmp_path):
    saved, base, diff = _export(tmp_path, "llama", wide=NATIVE)
    assert _rope(saved) == _rope(base)
    assert saved["max_position_embeddings"] == base["max_position_embeddings"] == NATIVE
    assert diff < 1e-4, diff


def test_rope_the_model_ignored_is_not_exported(tmp_path):
    """Unsloth's Gemma 2 rotary reads only rope_theta, so linear scaling on the config never reached training."""
    from transformers.models.gemma2.modeling_gemma2 import Gemma2RotaryEmbedding

    class GemmaFixedRotaryEmbedding(Gemma2RotaryEmbedding):
        """Stand-in for unsloth/models/gemma.py's class: built from the unscaled base config."""

    saved, base, diff = _export(tmp_path, "gemma2", LINEAR, rotary=GemmaFixedRotaryEmbedding)
    assert _rope(saved) == _rope(base)
    assert saved["max_position_embeddings"] == WIDE
    assert diff < 1e-4, f"export reloads with different logits than the trained model: {diff:.3e}"


def test_text_only_carry_over_targets_the_nested_text_config():
    from unsloth_zoo.saving_utils import _carry_over_rope_scaling

    yarn = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": NATIVE, "rope_theta": 1e6}
    trained = NS(rope_parameters=yarn, max_position_embeddings=WIDE)
    text = NS(rope_parameters={"rope_type": "default", "rope_theta": 1e6}, max_position_embeddings=NATIVE)
    base = NS(text_config=text, vision_config=NS())
    _carry_over_rope_scaling(base, trained, torch.nn.Module())
    assert text.rope_parameters == yarn and text.rope_parameters is not yarn
    assert text.max_position_embeddings == WIDE
    assert not hasattr(base, "rope_parameters") and not hasattr(base, "max_position_embeddings")


def test_rope_a_fixed_gemma_rotary_applied_is_exported(tmp_path):
    """A Gemma rotary that applied the loader's linear scaling records it as scaling_factor: export it."""
    from copy import deepcopy
    from transformers.models.gemma2.modeling_gemma2 import Gemma2RotaryEmbedding

    class GemmaFixedRotaryEmbedding(Gemma2RotaryEmbedding):
        """Stand-in for a fixed unsloth/models/gemma.py: built from the loader's scaled config."""

        def __init__(self, config):
            config = deepcopy(config)
            # transformers 5 reads rope_parameters, 4.x rope_scaling.
            field = "rope_parameters" if getattr(config, "rope_parameters", None) is not None else "rope_scaling"
            setattr(config, field, dict(LINEAR))
            config.max_position_embeddings = WIDE
            super().__init__(config)
            self.scaling_factor = LINEAR["factor"]

    saved, base, diff = _export(tmp_path, "gemma2", LINEAR, rotary=GemmaFixedRotaryEmbedding)
    got = _rope(saved)
    assert (got.get("rope_type") or got.get("type")) == "linear" and got.get("factor") == 4.0, got
    assert diff < 1e-4, f"export reloads with different logits than the trained model: {diff:.3e}"


def test_native_llama3_past_the_window_keeps_its_rope_and_the_window(tmp_path):
    """The loader keeps a native llama3 scheme and only widens max_position_embeddings past the window."""
    llama3 = {"rope_type": "llama3", "factor": 4.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
              "original_max_position_embeddings": NATIVE // 4}
    saved, base, diff = _export(tmp_path, "llama", base_rope=llama3)
    assert _rope(saved) == _rope(base) and _rope(saved)["rope_type"] == "llama3", _rope(saved)
    assert saved["max_position_embeddings"] == WIDE
    assert diff < 1e-4, diff


@pytest.mark.parametrize("native", [
    {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": NATIVE // 4},
    {"rope_type": "dynamic", "factor": 2.0},
], ids=["yarn", "dynamic"])
def test_native_rope_below_the_window_is_untouched(tmp_path, native):
    saved, base, diff = _export(tmp_path, "llama", wide=NATIVE, base_rope=native)
    assert _rope(saved) == _rope(base) and (_rope(saved).get("rope_type") or _rope(saved).get("type")) == native["rope_type"]
    assert saved["max_position_embeddings"] == NATIVE
    assert diff < 1e-4, diff


def _unsloth_llama_rotary(applied):
    """Stand-in for unsloth/models/llama.py's LlamaRotaryEmbedding, which keeps its frequencies as
    _unsloth_inv_freq_cpu; when transformers cannot build the scaled ones it falls back to unscaled RoPE."""
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding as HFLlamaRotaryEmbedding

    class LlamaRotaryEmbedding(HFLlamaRotaryEmbedding):
        def __init__(self, config):
            # Built from the trained config when it applied the scaling, else from the unscaled base config.
            super().__init__(config)
            self.dim, self.base = H._H // H._HEADS, 10000.0
            self._unsloth_inv_freq_cpu = self.inv_freq.detach().clone()
            if not applied:
                # The fallback's own expression, as unsloth/models/llama.py computes it.
                self._unsloth_inv_freq_cpu = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))

    return LlamaRotaryEmbedding


@pytest.mark.parametrize("applied", [False, True], ids=["fell_back_unscaled", "applied"])
def test_rope_a_llama_rotary_could_not_apply_is_not_exported(tmp_path, applied):
    """transformers 4.57 cannot build YaRN for a config whose head_dim is None (Mistral), and Unsloth's rotary then
    trains unscaled: exporting the config's YaRN would reload a different model."""
    saved, base, diff = _export(tmp_path, "llama", YARN, rotary=_unsloth_llama_rotary(applied), rotary_from_trained=applied)
    got = _rope(saved)
    if applied:
        assert got.get("rope_type") == "yarn" and got.get("factor") == 4.0, got
    else:
        assert _rope(saved) == _rope(base), got
    assert saved["max_position_embeddings"] == WIDE
    assert diff < 1e-4, f"export reloads with different logits than the trained model: {diff:.3e}"


def test_checkpoint_key_restore_keeps_the_carried_rope(tmp_path):
    """A repo-code checkpoint whose keys the built-in class renamed is written back from its own config.json; the RoPE
    and window the export carried over from training must survive that, and nothing else changes."""
    from unsloth_zoo.saving_utils import _restore_checkpoint_config_keys

    class AliasConfig:
        attribute_map = {"norm_topk_prob": "moe_renormalize"}

    source = {"auto_map": {}, "norm_topk_prob": True, "max_position_embeddings": NATIVE, "rope_scaling": None}
    written = {"auto_map": {}, "moe_renormalize": True, "max_position_embeddings": WIDE,
               "rope_scaling": {"type": "linear", "factor": 4.0}}
    for carried, want in [((), source), (("rope_scaling", "max_position_embeddings"), None)]:
        path = tmp_path / "config.json"
        path.write_text(json.dumps(written))
        assert _restore_checkpoint_config_keys(path, json.loads(json.dumps(source)), AliasConfig, carried=carried)
        got = json.loads(path.read_text())
        if want is not None:
            assert got == want
        else:
            assert got == {**source, "max_position_embeddings": WIDE, "rope_scaling": written["rope_scaling"]}, got


@pytest.mark.parametrize("attention_scaling,exported", [(1.0, False), (1.1386, True)], ids=["fallback", "yarn_mscale_only"])
def test_yarn_that_only_scales_attention_still_counts_as_applied(attention_scaling, exported):
    """With a long original window every frequency can stay unscaled under YaRN and only cos / sin scale (mscale), as
    on Qwen2 (theta 1e6, original 32768): that is applied scaling, unlike the fallback, which leaves both at 1."""
    from unsloth_zoo.saving_utils import _carry_over_rope_scaling

    class LlamaRotaryEmbedding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dim, self.base, self.attention_scaling = 8, 1e6, attention_scaling
            self._unsloth_inv_freq_cpu = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))

    yarn = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768, "rope_theta": 1e6}
    default = {"rope_type": "default", "rope_theta": 1e6}
    trained = NS(rope_parameters=yarn, max_position_embeddings=4 * 32768)
    base = NS(rope_parameters=dict(default), max_position_embeddings=32768)
    model = torch.nn.Module()
    model.rotary_emb = LlamaRotaryEmbedding()
    _carry_over_rope_scaling(base, trained, model)
    assert base.rope_parameters == (yarn if exported else default)
    assert base.max_position_embeddings == 4 * 32768


def test_wrapper_whose_text_config_is_itself_moves_rope_into_llm_config():
    """A repo-code composite (InternVL style) whose get_text_config() returns the wrapper keeps the language model's
    RoPE and window in llm_config; a text-only export trains that config, and the reload reads llm_config."""
    import transformers as T
    from unsloth_zoo.saving_utils import _carry_over_rope_scaling

    llm = T.LlamaConfig(max_position_embeddings=NATIVE)
    base = NS(llm_config=llm, vision_config=NS())
    base.get_text_config = lambda: base
    trained = T.LlamaConfig(max_position_embeddings=WIDE)
    field = "rope_parameters" if getattr(llm, "rope_parameters", None) is not None else "rope_scaling"
    setattr(trained, field, {"rope_type": "linear", "factor": 4.0, "rope_theta": 10000.0})
    _carry_over_rope_scaling(base, trained, torch.nn.Module())
    got = getattr(llm, field) or {}
    assert (got.get("rope_type") or got.get("type")) == "linear" and got.get("factor") == 4.0, got
    assert llm.max_position_embeddings == WIDE
    assert not hasattr(base, "rope_parameters") and not hasattr(base, "rope_scaling")
    assert not hasattr(base, "max_position_embeddings")
