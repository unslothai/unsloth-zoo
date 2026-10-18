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

"""LoRA on embed_tokens / lm_head after add_new_tokens: the merged rows past the base
checkpoint's vocab must be the live base rows plus the delta, not zeros plus the delta."""

from __future__ import annotations

import copy
import os

import pytest
import torch

import _merge_e2e_helpers as H

NEW_VOCAB = 72  # base vocab is 64 (H._VOCAB)


def _merged_vs_peft(tmp_path, target_modules, new_vocab = NEW_VOCAB, tie = False, inactive_targets = None,
                    config = None):
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    H.set_offline_cpu_env()
    if not H.family_available("llama"):
        pytest.skip(reason = "installed transformers has no llama model_type")
    base_dir = os.path.join(str(tmp_path), "base")
    out_dir = os.path.join(str(tmp_path), "merged")

    torch.manual_seed(H.SEED)
    config = config or H.make_spec("llama").config
    config.tie_word_embeddings = tie
    model = AutoModelForCausalLM.from_config(config).to(torch.float32)
    model.save_pretrained(base_dir, safe_serialization = True)
    model.config._name_or_path = base_dir

    # Offset so neither a zero row nor the checkpoint row can pass.
    model.resize_token_embeddings(new_vocab)
    new_rows = slice(H._VOCAB, new_vocab) if new_vocab > H._VOCAB else slice(new_vocab - 4, new_vocab)
    with torch.no_grad():
        for w in (model.get_input_embeddings().weight, model.get_output_embeddings().weight):
            w[new_rows] = w[:new_rows.start].mean(0) + 0.5

    torch.manual_seed(H.SEED)
    pm = get_peft_model(model, LoraConfig(
        r = 4, lora_alpha = 8, lora_dropout = 0.0, bias = "none", target_modules = target_modules))
    if inactive_targets is not None:
        pm.add_adapter("other", LoraConfig(r = 4, lora_alpha = 8, target_modules = inactive_targets))
        pm.set_adapter("default")
    H.seed_lora(pm)
    with torch.no_grad():
        for name, p in pm.named_parameters():
            if "lora_embedding" in name:
                p.copy_(torch.randn_like(p) * 0.02 + 0.01)

    expected = {
        k: v.detach().clone() for k, v in copy.deepcopy(pm).merge_and_unload().state_dict().items()
    }
    H.run_merge(pm, base_dir, out_dir, save_dtype = torch.float32)
    return H.read_safetensors_dir(out_dir), expected, out_dir


@pytest.mark.parametrize("target_modules", [
    ["embed_tokens", "lm_head", "q_proj"], ["lm_head"], ["embed_tokens"],
])
def test_new_token_rows_keep_their_base_values(tmp_path, target_modules):
    from transformers import AutoModelForCausalLM

    merged, expected, out_dir = _merged_vs_peft(tmp_path, target_modules)
    for key in ("model.embed_tokens.weight", "lm_head.weight"):
        assert merged[key].shape[0] == NEW_VOCAB
        new_rows, want = merged[key][H._VOCAB:], expected[key][H._VOCAB:]
        assert torch.allclose(new_rows, want, atol = 1e-5, rtol = 1e-5), (
            f"{key} new rows: max diff {(new_rows - want).abs().max().item():.4f}"
        )
        assert torch.allclose(merged[key][:H._VOCAB], expected[key][:H._VOCAB], atol = 1e-5, rtol = 1e-5)
    AutoModelForCausalLM.from_pretrained(out_dir)


def test_shrunk_vocab_merges_onto_the_kept_rows():
    from unsloth_zoo.saving_utils import _merge_lora, LoraStats

    torch.manual_seed(0)
    W = torch.randn(8, 4)
    A, B = torch.randn(2, 4), torch.randn(6, 2)
    stats = LoraStats(None, A, B, 0.5)
    merged = _merge_lora(W.clone(), stats, "lm_head")
    assert torch.allclose(merged.cpu(), W[:6] + 0.5 * (B @ A), atol = 1e-6)


@pytest.mark.parametrize("quant_type", [None, "fp8", "mxfp4"])
def test_untargeted_vocab_override_reaches_every_shard_writer(tmp_path, quant_type):
    from collections import defaultdict
    from safetensors import safe_open
    from safetensors.torch import save_file
    from unsloth_zoo.saving_utils import _merge_and_overwrite_lora, LoraStats

    torch.manual_seed(0)
    old = torch.randn(64, 8, dtype = torch.bfloat16)
    live = torch.cat([old, torch.randn(8, 8, dtype = torch.bfloat16)])
    shard = tmp_path / "model.safetensors"
    head = torch.nn.Linear(8, 72, dtype = torch.bfloat16)
    with torch.no_grad():
        head.weight.copy_(live)
    tensors = {"lm_head.weight": old, "lm_head.bias": head.bias.detach()[:64].clone(),
               "model.norm.weight": torch.ones(8, dtype = torch.bfloat16)}
    if quant_type == "fp8":
        tensors["model.layers.0.mlp.down_proj.weight"] = torch.randn(8, 8).to(torch.float8_e4m3fn)
        tensors["model.layers.0.mlp.down_proj.weight_scale"] = torch.ones(8, 1, dtype = torch.bfloat16)
    save_file(tensors, str(shard), metadata = {"format": "pt"})

    _merge_and_overwrite_lora(
        save_directory = str(tmp_path),
        filename = "model.safetensors",
        lora_weights = defaultdict(lambda: LoraStats(None, None, None, 0)),
        output_dtype = torch.bfloat16,
        model_class_name = "LlamaForCausalLM",
        base_model_is_quantized = quant_type is not None,
        quant_type = quant_type,
        vocab_modules = {"lm_head": (head.weight, head.bias)},
    )
    with safe_open(str(shard), framework = "pt", device = "cpu") as f:
        assert torch.equal(f.get_tensor("lm_head.weight"), live)
        assert torch.equal(f.get_tensor("lm_head.bias"), head.bias.detach())


@pytest.mark.parametrize("target_modules", [["embed_tokens", "lm_head", "q_proj"], ["lm_head"]])
def test_new_tokens_on_spare_padding_rows_keep_their_seeded_values(tmp_path, target_modules):
    merged, expected, _ = _merged_vs_peft(tmp_path, target_modules, new_vocab = H._VOCAB)
    for key in ("model.embed_tokens.weight", "lm_head.weight"):
        assert torch.allclose(merged[key], expected[key], atol = 1e-5, rtol = 1e-5), key


def test_tied_shared_vocab_grows_without_a_vocab_adapter(tmp_path):
    from transformers import AutoModelForCausalLM

    merged, expected, out_dir = _merged_vs_peft(tmp_path, ["q_proj"], tie = True)
    key = "model.embed_tokens.weight"
    assert merged[key].shape[0] == NEW_VOCAB
    assert torch.allclose(merged[key], expected[key], atol = 1e-5, rtol = 1e-5)
    AutoModelForCausalLM.from_pretrained(out_dir)


def test_vocab_wrapped_only_by_an_inactive_adapter_still_grows(tmp_path):
    from transformers import AutoModelForCausalLM

    merged, expected, out_dir = _merged_vs_peft(
        tmp_path, ["q_proj"], inactive_targets = ["embed_tokens", "lm_head"])
    for key in ("model.embed_tokens.weight", "lm_head.weight"):
        assert merged[key].shape[0] == NEW_VOCAB
        assert torch.allclose(merged[key], expected[key], atol = 1e-5, rtol = 1e-5), key
    AutoModelForCausalLM.from_pretrained(out_dir)


def test_live_vocab_rows_edit_in_place_without_copying_the_table():
    from unsloth_zoo.saving_utils import _with_live_vocab_rows

    W = torch.randn(16, 4).to(torch.bfloat16).float()
    assert _with_live_vocab_rows(W, W.to(torch.bfloat16)) is W
    live = W.to(torch.bfloat16).clone()
    live[3] += 1
    before = W.clone()
    out = _with_live_vocab_rows(W, live)
    assert out is W and torch.equal(out[3], live[3].float())
    assert torch.equal(out[:3], before[:3]) and torch.equal(out[4:], before[4:])


def test_float8_live_codes_are_never_copied_over_the_checkpoint():
    from unsloth_zoo.saving_utils import _with_live_vocab_rows

    W = torch.randn(16, 4)
    assert _with_live_vocab_rows(W, (W * 7).to(torch.float8_e4m3fn)) is W


def _tiny(arch):
    import transformers as T

    common = dict(vocab_size = H._VOCAB, hidden_size = 32, intermediate_size = 64, num_hidden_layers = 2,
                  num_attention_heads = 4, max_position_embeddings = 64)
    if not hasattr(T, arch):
        pytest.skip(reason = f"installed transformers has no {arch}")
    return getattr(T, arch)(**common)


@pytest.mark.parametrize("arch, target_modules, new_vocab, tie", [
    ("PhiConfig", ["q_proj"], NEW_VOCAB, False),
    ("PhiConfig", ["lm_head", "q_proj"], NEW_VOCAB, False),
    (None, ["lm_head"], H._VOCAB, True),
])
def test_every_exported_tensor_matches_peft_after_new_tokens(tmp_path, arch, target_modules, new_vocab, tie):
    from transformers import AutoModelForCausalLM

    config = _tiny(arch) if arch else None
    _, expected, out_dir = _merged_vs_peft(
        tmp_path, target_modules, new_vocab = new_vocab, tie = tie, config = config)
    reloaded = AutoModelForCausalLM.from_pretrained(out_dir, dtype = torch.float32).state_dict()
    for key, want in expected.items():
        assert torch.allclose(reloaded[key].float(), want.float(), atol = 1e-5, rtol = 1e-5), key
