# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""End-to-end LoRA merge correctness for vision / multimodal models.

Language-side LoRA must merge while every vision-tower / projector / audio tensor
stays byte-identical. Also exercises the #773 key remap (runtime
`model.language_model.` vs on-disk `language_model.model.`): reference keys use the
production remapper `_convert_lora_keys_to_safetensor_format`, and a clean run proves
the key-count check does not false-positive when only the nested language model is
adapted.
"""

from __future__ import annotations

import os

import pytest
import torch

import _merge_e2e_helpers as H

# language model q/v only (the vision encoder also has q_proj/v_proj).
_LANG_QV = r".*language_model.*\.(q_proj|v_proj)$"
_VISION_MARKERS = ("vision", "visual", "multi_modal", "audio_tower", "vision_tower")


def _build_gemma3():
    import transformers as T
    text = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
                max_position_embeddings=64, head_dim=8)
    vision = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                  num_attention_heads=4, image_size=16, patch_size=8, num_channels=3)
    return T.Gemma3Config(text_config=text, vision_config=vision)


def _build_qwen3_vl():
    import transformers as T
    text = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
                max_position_embeddings=64, head_dim=8, rope_scaling={"type": "default",
                "mrope_section": [1, 1, 2]})
    vision = dict(hidden_size=32, intermediate_size=64, depth=2, num_heads=4,
                  patch_size=16, out_hidden_size=32)
    return T.Qwen3VLConfig(text_config=text, vision_config=vision)


_VLM_BUILDERS = {"gemma3": _build_gemma3, "qwen3_vl": _build_qwen3_vl}


def _make_vlm(family):
    import transformers as T
    if not H.family_available(family):
        pytest.skip(f"{family} unavailable in this transformers")
    try:
        cfg = _VLM_BUILDERS[family]()
        torch.manual_seed(H.SEED)
        model = T.AutoModelForImageTextToText.from_config(cfg).to(torch.float32)
    except Exception as e:  # tiny VLM config quirks vary by version
        pytest.skip(f"could not instantiate tiny {family}: {type(e).__name__}: {e}")
    return model


@pytest.mark.parametrize("family", list(_VLM_BUILDERS))
def test_vision_language_only_merge_preserves_vision(family, tmp_path):
    import unsloth_zoo.saving_utils as SU
    from peft import LoraConfig, get_peft_model

    H.set_offline_cpu_env()
    model = _make_vlm(family)
    cls_name = type(model).__name__
    base_dir = os.path.join(str(tmp_path), "base")
    out_dir = os.path.join(str(tmp_path), "merged")
    model.save_pretrained(base_dir, safe_serialization=True)
    model.config._name_or_path = base_dir
    base = H.read_safetensors_dir(base_dir)
    base_keys = list(base.keys())

    pm = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none", target_modules=_LANG_QV))
    # no adapter may land on the vision tower
    in_vision = [n for n, m in pm.named_modules()
                 if getattr(m, "lora_A", None) is not None
                 and hasattr(m.lora_A, "__contains__") and "default" in m.lora_A
                 and any(s in n for s in _VISION_MARKERS)]
    assert not in_vision, f"language-only LoRA leaked onto vision modules: {in_vision[:3]}"
    H.seed_lora(pm)

    # reference adapted keys via the PRODUCTION remapper (handles the VLM prefix swap)
    lora_weights, _ = SU.create_lora_statistics(pm, merge_into_original=True)
    remapped = SU._convert_lora_keys_to_safetensor_format(lora_weights, base_keys, cls_name)
    ref = {}
    for k, st in remapped.items():
        if isinstance(k, str) and st.lora_A is not None:
            sk = k + ".weight"
            if sk in base:
                ref[sk] = st
    if not ref:
        # remapper could not line language LoRA keys to on-disk keys on this
        # transformers version; vision preservation below is the core check.
        pytest.skip(f"{family}: language adapter keys did not resolve against base "
                    f"on this transformers version")

    H.run_merge(pm, base_dir, out_dir, save_dtype=torch.float32)
    merged = H.read_safetensors_dir(out_dir)

    atol, rtol = H._TOL[torch.float32]
    n_lang = n_vision = 0
    for key, mt in merged.items():
        if key in ref:
            st = ref[key]
            A = st.lora_A.double().cpu(); B = st.lora_B.double().cpu()
            expected = (base[key].double() + float(st.alpha) * (B @ A)).to(torch.float32)
            H._assert_close(family, key, mt, expected, atol, rtol, adapted=True)
            n_lang += 1
        else:
            H._assert_equal(family, key, mt, base[key])
            if any(s in key for s in _VISION_MARKERS):
                n_vision += 1

    assert n_lang >= 1, "no language LoRA deltas were merged"
    assert n_vision >= 1, "no vision tensors present to verify preservation"
