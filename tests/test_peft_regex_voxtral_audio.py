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

"""finetune_audio_layers=True on Voxtral used to leave the audio_tower and projector without LoRA."""
import torch.nn as nn

from unsloth_zoo.peft_utils import get_peft_regex
from test_peft_regex_audio import FakeModel, linear_names, matched, QWEN2_AUDIO


def _voxtral(model_type="voxtral"):
    lin = []
    for i in range(2):
        lin += [f"language_model.layers.{i}.self_attn.{p}" for p in ("q_proj", "k_proj", "v_proj", "o_proj")]
        lin += [f"language_model.layers.{i}.mlp.{p}" for p in ("gate_proj", "up_proj", "down_proj")]
        lin += [f"audio_tower.layers.{i}.self_attn.{p}" for p in ("q_proj", "k_proj", "v_proj", "out_proj")]
        lin += [f"audio_tower.layers.{i}.fc1", f"audio_tower.layers.{i}.fc2"]
    lin += ["multi_modal_projector.linear_1", "multi_modal_projector.linear_2"]
    m = FakeModel(lin, name="mistralai/Voxtral-Mini-3B-2507", model_type=model_type)
    m.lm_head = nn.Linear(8, 8, bias=False)
    return m


def _flags(**kw):
    d = dict(finetune_vision_layers=False, finetune_language_layers=True,
             finetune_attention_modules=True, finetune_mlp_modules=True)
    d.update(kw)
    return d


def test_voxtral_audio_flag_attaches_audio_tower_and_projector():
    model = _voxtral()
    ns = linear_names(model)
    on = matched(get_peft_regex(model, **_flags(finetune_audio_layers=True)), ns)
    audio = {n for n in on if ".audio_tower." in n}
    assert len(audio) == 12, sorted(audio)
    assert {"model.multi_modal_projector.linear_1", "model.multi_modal_projector.linear_2"} <= on
    assert sum(".language_model." in n for n in on) == 14
    assert not any(n.endswith("lm_head") for n in on)


def test_voxtral_audio_flag_off_unchanged():
    model = _voxtral()
    ns = linear_names(model)
    off = matched(get_peft_regex(model, **_flags(finetune_audio_layers=False)), ns)
    assert not any(".audio_tower." in n or "multi_modal_projector" in n for n in off)


def test_voxtral_audio_respects_attn_mlp_flags_and_explicit_targets():
    model = _voxtral()
    ns = linear_names(model)
    attn = matched(get_peft_regex(model, **_flags(finetune_audio_layers=True, finetune_mlp_modules=False)), ns)
    assert not any(n.endswith(("fc1", "fc2", "linear_1", "linear_2")) for n in attn)
    assert any(n.endswith("audio_tower.layers.0.self_attn.out_proj") for n in attn)
    only_q = matched(get_peft_regex(model, **_flags(finetune_audio_layers=True), target_modules=["q_proj"]), ns)
    assert {n.rsplit(".", 1)[-1] for n in only_q} == {"q_proj"}


def test_other_audio_models_untouched():
    model = FakeModel(QWEN2_AUDIO, model_type="qwen2_audio")
    assert get_peft_regex(model, finetune_audio_layers=True) == get_peft_regex(model, finetune_audio_layers=False)
