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

"""Qwen/Qwen3-ASR-0.6B and -1.7B store weights in the qwen-asr package layout (thinker.*). Unsloth loads them into
transformers' native Qwen3-ASR names, so a merged export rereading the original checkpoint must map the adapters back,
or refuse, instead of writing the base weights as a trained model."""

from __future__ import annotations

import json
import re

import pytest
import torch

import _merge_e2e_helpers as H

import unsloth_zoo.saving_utils as SU


_INVERSE = [
    (r"^model\.multi_modal_projector\.linear_1\.", "thinker.audio_tower.proj1."),
    (r"^model\.multi_modal_projector\.linear_2\.", "thinker.audio_tower.proj2."),
    (r"^model\.audio_tower\.", "thinker.audio_tower."),
    (r"^model\.language_model\.", "thinker.model."),
    (r"^lm_head\.", "thinker.lm_head."),
]


def _to_package(key):
    for pattern, repl in _INVERSE:
        if re.match(pattern, key):
            return re.sub(pattern, repl, key, count = 1)
    return key


def _package_checkpoint(base_dir):
    """A tiny native Qwen3-ASR, re-saved the way the qwen-asr package does."""
    import transformers as T
    from safetensors.torch import save_file

    Config = getattr(T, "Qwen3ASRConfig", None)
    Model = getattr(T, "Qwen3ASRForConditionalGeneration", None)
    if Config is None or Model is None:
        pytest.skip("transformers has no Qwen3-ASR")
    audio = dict(d_model = 32, encoder_layers = 1, encoder_attention_heads = 2, encoder_ffn_dim = 64,
                 output_dim = 32, downsample_hidden_size = 16, num_mel_bins = 128)
    text = dict(hidden_size = 32, intermediate_size = 64, num_hidden_layers = 1, num_attention_heads = 2,
                num_key_value_heads = 1, head_dim = 16, vocab_size = 256)
    config = Config(audio_config = audio, text_config = text, audio_token_id = 5, pad_token_id = 0,
                    eos_token_id = 1, tie_word_embeddings = False)
    torch.manual_seed(H.SEED)
    model = Model(config).eval()
    tensors = {_to_package(k): v.contiguous().clone() for k, v in model.state_dict().items()}
    save_file(tensors, str(base_dir / "model.safetensors"), metadata = {"format": "pt"})
    thinker = dict(model_type = "qwen3_asr", audio_config = audio, text_config = text, audio_token_id = 5,
                   pad_token_id = 0, eos_token_id = 1, tie_word_embeddings = False)
    (base_dir / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3ASRForConditionalGeneration"],
        "model_type": "qwen3_asr",
        "thinker_config": thinker,
    }))
    model.config._name_or_path = str(base_dir)
    return model


def _trained(base_dir):
    from peft import LoraConfig, get_peft_model

    model = _package_checkpoint(base_dir)
    peft_model = get_peft_model(model, LoraConfig(
        r = 2, lora_alpha = 4, lora_dropout = 0.0,
        target_modules = ["q_proj", "v_proj", "linear_1"],
    ))
    H.seed_lora(peft_model)
    return peft_model


def test_package_layout_merge_writes_the_trained_weights(tmp_path):
    H.set_offline_cpu_env()
    base_dir, out_dir = tmp_path / "base", tmp_path / "out"
    base_dir.mkdir()
    peft_model = _trained(base_dir)
    adapted = H.extract_adapted(peft_model)
    # The audio projector is a rename, not a prefix move: thinker.audio_tower.proj1 -> multi_modal_projector.linear_1.
    assert any(".multi_modal_projector.linear_1." in k for k in adapted)
    assert any(k.startswith("model.audio_tower.") for k in adapted)
    assert any(k.startswith("model.language_model.") for k in adapted)

    base = H.read_safetensors_dir(str(base_dir))
    H.run_merge(peft_model, str(base_dir), str(out_dir), save_dtype = torch.float32)
    merged = H.read_safetensors_dir(str(out_dir))

    for key, a in adapted.items():
        disk = _to_package(key)
        assert disk in merged, (key, disk, sorted(merged)[:8])
        want = H._ref_dense(base[disk], a).to(torch.float32)
        assert not torch.equal(merged[disk], base[disk]), f"{disk} kept the base weight"
        torch.testing.assert_close(merged[disk], want, atol = 1e-5, rtol = 1e-5)
    untouched = set(base) - {_to_package(k) for k in adapted}
    assert untouched and all(torch.equal(merged[k], base[k]) for k in untouched)


def test_merge_with_no_placeable_adapter_is_refused(tmp_path, monkeypatch):
    H.set_offline_cpu_env()
    base_dir, out_dir = tmp_path / "base", tmp_path / "out"
    base_dir.mkdir()
    peft_model = _trained(base_dir)
    # Without the layout's mapping no adapter has a checkpoint tensor; 0 placed must not pass as 0 expected.
    monkeypatch.setattr(SU, "_LOADER_CHECKPOINT_CONVERSION_MAPPINGS", {}, raising = False)
    monkeypatch.delenv("UNSLOTH_ALLOW_PARTIAL_LORA_MERGE", raising = False)
    with pytest.raises(SU.PartialLoraMergeError, match = "None of the"):
        H.run_merge(peft_model, str(base_dir), str(out_dir), save_dtype = torch.float32)
