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

"""Where a VLM checkpoint keeps its text weights, across every layout we have seen (#969).

A `text_only` export has to find the text tensors inside a composite checkpoint before it can
drop the rest, and there is no one place they live. The layout is fixed when the checkpoint is
written, not by the transformers doing the reading, so a published gemma3 and one saved locally
by the same code disagree. Hardcoding any single prefix silently mismaps the others, which is
the failure #969 reported in the first place.

The key sets below are trimmed to one layer, but the prefixes are the real ones, read off the
published `model.safetensors.index.json` of each checkpoint. Each case pins both halves of the
answer: a rule that maps the text weights correctly and still keeps the vision tower is wrong.
"""

from __future__ import annotations

import pytest

from unsloth_zoo.saving_utils import TextOnlyRemapError, _text_only_key_map

TEXT_KEYS = {
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.0.input_layernorm.weight",
    "model.norm.weight",
    "lm_head.weight",
}


def _text(text_prefix, base_prefix, head = None):
    """The text tensors after `text_prefix` is swapped for `base_prefix`, plus the head, if any.

    That swap is the whole shape of the problem: some layouts prepend a component to every text
    key, others replace the leading `model.` with a deeper path, and the head can be tied away,
    left at the top level, or carried along under the prefix.
    """
    keys = {base_prefix + k[len(text_prefix):] for k in TEXT_KEYS - {"lm_head.weight"}}
    return keys if head is None else keys | {head}


def _vision(prefix):
    return {prefix + "encoder.layers.0.self_attn.q_proj.weight", prefix + "post_layernorm.weight"}


LAYOUTS = [
    # unsloth/gemma-3-4b-it as published: no lm_head anywhere, because the head is tied.
    ("gemma3-published", True,
     _text("", "language_model."),
     _vision("vision_tower.vision_model.") | {"multi_modal_projector.mm_input_projection_weight"}),

    # The same family written by save_pretrained on transformers 5.5.0: one component deeper.
    ("gemma3-save-pretrained", True,
     _text("", "model.language_model."),
     _vision("model.vision_tower.vision_model.") |
     {"model.multi_modal_projector.mm_input_projection_weight"}),

    # An in-memory Gemma3ForConditionalGeneration, which keeps lm_head at the top level.
    ("gemma3-in-memory", False,
     _text("model.", "model.language_model.", head = "lm_head.weight"),
     _vision("model.vision_tower.vision_model.") |
     {"model.multi_modal_projector.mm_input_projection_weight"}),

    # HuggingFaceM4/Idefics3-8B-Llama3: text under model.text_model, head unprefixed.
    ("idefics3", False,
     _text("model.", "model.text_model.", head = "lm_head.weight"),
     _vision("model.vision_model.") | {"model.connector.modality_projection.proj.weight"}),

    # Qwen/Qwen2.5-Omni-3B: the text model is one of five top-level towers, audio included.
    ("qwen-omni", False,
     _text("", "thinker.", head = "thinker.lm_head.weight"),
     _vision("thinker.visual.") | {
         "thinker.audio_tower.layers.0.self_attn.q_proj.weight",
         "talker.model.layers.0.self_attn.q_proj.weight",
         "token2wav.code2wav_bigvgan_model.conv_pre.weight",
     }),
]


@pytest.mark.parametrize("name, tied, text_base, other", LAYOUTS, ids = [l[0] for l in LAYOUTS])
def test_the_text_weights_are_found_and_only_those_are_kept(name, tied, text_base, other):
    key_map = _text_only_key_map(TEXT_KEYS, text_base | other, tie_word_embeddings = tied)

    expected = TEXT_KEYS - {"lm_head.weight"} if tied else TEXT_KEYS
    assert set(key_map) == expected, f"{name}: the reload would be missing {expected - set(key_map)}"
    assert set(key_map.values()) == text_base, (
        f"{name}: kept {sorted(set(key_map.values()) - text_base)}, "
        f"lost {sorted(text_base - set(key_map.values()))}")
    assert not (set(key_map.values()) & other), f"{name}: a vision or audio tensor was kept"


def test_the_tied_head_is_left_out_rather_than_invented():
    """gemma3 ships 883 tensors and not one is an lm_head, because the head is tied."""
    _, tied, text_base, other = LAYOUTS[0]
    key_map = _text_only_key_map(TEXT_KEYS, text_base | other, tie_word_embeddings = tied)
    assert "lm_head.weight" not in key_map, "the export would write a head the base does not have"


def test_a_missing_head_that_is_not_tied_is_an_error():
    """Without the tie there is no tensor to fall back on, so guessing would ship a broken head."""
    _, _, text_base, other = LAYOUTS[0]
    with pytest.raises(TextOnlyRemapError):
        _text_only_key_map(TEXT_KEYS, text_base | other, tie_word_embeddings = False)


def test_text_weights_under_two_prefixes_are_an_error():
    """No single substitution reaches both halves, and picking one would drop the other."""
    base_keys = {
        "language_model.model.embed_tokens.weight",
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "decoder.model.layers.0.mlp.down_proj.weight",
        "decoder.model.layers.0.input_layernorm.weight",
        "language_model.model.norm.weight",
        "language_model.lm_head.weight",
    }
    with pytest.raises(TextOnlyRemapError):
        _text_only_key_map(TEXT_KEYS, base_keys, tie_word_embeddings = False)


def test_a_text_key_with_no_counterpart_is_an_error():
    """A tensor the reload needs and the checkpoint lacks is exactly #969, so refuse to write it."""
    _, _, text_base, other = LAYOUTS[4]
    base_keys = {k for k in text_base | other if "mlp.down_proj" not in k}
    with pytest.raises(TextOnlyRemapError):
        _text_only_key_map(TEXT_KEYS, base_keys, tie_word_embeddings = False)


def test_two_complete_readings_are_an_error():
    """Qwen Omni carries a second decoder under `talker`, and it only fails to match because
    it is a different shape. One that matched would give two equally valid answers, and
    picking either would be a coin flip over which model the user gets.
    """
    base_keys = set()
    for tower in ("thinker.", "talker."):
        base_keys |= {tower + k for k in TEXT_KEYS - {"lm_head.weight"}}
        base_keys.add(tower + "lm_head.weight")
    with pytest.raises(TextOnlyRemapError):
        _text_only_key_map(TEXT_KEYS, base_keys, tie_word_embeddings = False)


def test_an_already_text_only_checkpoint_is_an_error():
    """There is nothing to drop, and an identity map would rewrite every shard to achieve it."""
    with pytest.raises(TextOnlyRemapError):
        _text_only_key_map(TEXT_KEYS, set(TEXT_KEYS), tie_word_embeddings = False)
