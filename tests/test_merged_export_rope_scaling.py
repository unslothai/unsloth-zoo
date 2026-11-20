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


def _export(tmp_path, family, rope=None, wide=WIDE, in_memory_rope=None, rotary=None):
    """Load like Unsloth's loader (scaling and window on the in-memory config), merge, and reload the export."""
    import transformers as T
    from peft import LoraConfig, get_peft_model

    H.set_offline_cpu_env()
    base_dir, out_dir = str(tmp_path / "base"), str(tmp_path / "out")
    H.build_and_save_base(H.make_spec(family), base_dir)
    kwargs = {"rope_scaling": rope} if rope else {}
    model = T.AutoModelForCausalLM.from_pretrained(
        base_dir, max_position_embeddings=wide, dtype=torch.float32, **kwargs)
    if in_memory_rope is not None:
        model.config.rope_parameters = in_memory_rope
    if rotary is not None:
        model.model.rotary_emb = rotary(T.AutoConfig.from_pretrained(base_dir))
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
    assert got.get("rope_theta") == _rope(base).get("rope_theta")
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
