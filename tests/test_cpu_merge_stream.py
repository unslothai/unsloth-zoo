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

"""CPU merged save streamed from the live model: same checkpoint as merge_and_unload()."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


_TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"


def _tiny_model():
    from transformers import AutoModelForCausalLM
    try:
        return AutoModelForCausalLM.from_pretrained(_TINY, dtype = torch.float32, attn_implementation = "sdpa")
    except Exception as error:  # offline without the tiny model cached
        pytest.skip(f"{_TINY} unavailable: {error}")


def _read_checkpoint(directory):
    import glob
    import json
    from safetensors.torch import load_file
    tensors = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.safetensors"))):
        tensors.update(load_file(path))
    with open(os.path.join(directory, "config.json")) as f:
        config = json.load(f)
    config.pop("transformers_version", None)
    return tensors, config


def _lora_model(name, dtype, frozen_bf16 = False):
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    from unsloth_zoo.cpu.ops import store_frozen_linears_in_bf16
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype = torch.bfloat16 if frozen_bf16 else dtype)
    except Exception as error:
        pytest.skip(f"{name} unavailable: {error}")
    if frozen_bf16:
        store_frozen_linears_in_bf16(model)
    torch.manual_seed(0)
    config = LoraConfig(r = 4, lora_alpha = 8, target_modules = ["q_proj", "v_proj", "up_proj", "down_proj"],
                        init_lora_weights = False)
    return get_peft_model(model, config)


@pytest.mark.parametrize("case", ["bf16", "fp32", "fp32_frozen_bf16", "full_ft_fp32", "sharded"])
@pytest.mark.parametrize("save_method", ["merged_16bit", "merged_32bit"])
def test_streamed_merge_matches_the_copying_path(tmp_path, monkeypatch, case, save_method):
    from unsloth_zoo.cpu import loader
    if case == "full_ft_fp32":
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(_TINY, dtype = torch.float32)
    else:
        model = _lora_model(_TINY, torch.float32 if case.startswith("fp32") else torch.bfloat16,
                            frozen_bf16 = case == "fp32_frozen_bf16")
    shard = "1MB" if case == "sharded" else None
    before = {k: v.clone() for k, v in model.state_dict().items()}
    reference, streamed = tmp_path / "copy", tmp_path / "stream"
    merged, _ = loader._merged_copy(model, save_method)
    merged.save_pretrained(reference, **({"max_shard_size": shard} if shard else {}))
    del merged
    assert loader._stream_merged_save(model, str(streamed), save_method, shard) is True
    ref_tensors, ref_config = _read_checkpoint(reference)
    new_tensors, new_config = _read_checkpoint(streamed)
    assert set(ref_tensors) == set(new_tensors)
    for key, value in ref_tensors.items():
        assert new_tensors[key].dtype == value.dtype, key
        assert torch.equal(new_tensors[key], value), key
    assert new_config == ref_config
    if case == "sharded":
        assert os.path.exists(streamed / "model.safetensors.index.json")
    after = model.state_dict()
    assert set(after) == set(before) and all(torch.equal(after[k], before[k]) for k in before)


def test_streamed_merge_refuses_what_it_cannot_reproduce(tmp_path):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    from unsloth_zoo.cpu import loader
    model = AutoModelForCausalLM.from_pretrained(_TINY, dtype = torch.float32)
    lora = get_peft_model(model, LoraConfig(r = 4, target_modules = ["q_proj"], modules_to_save = ["lm_head"]))
    assert loader._stream_merged_save(lora, str(tmp_path / "x"), "merged_16bit") is False


def test_streamed_merge_tied_embeddings_match(tmp_path):
    """Tied lm_head / embed_tokens: one name on disk, as save_pretrained writes it."""
    import transformers
    from unsloth_zoo.cpu import loader
    config = transformers.LlamaConfig(hidden_size = 64, intermediate_size = 128, num_hidden_layers = 2,
                                      num_attention_heads = 4, num_key_value_heads = 2, vocab_size = 256,
                                      tie_word_embeddings = True)
    torch.manual_seed(0)
    transformers.LlamaForCausalLM(config).to(torch.bfloat16).save_pretrained(tmp_path / "base")
    model = _lora_model(str(tmp_path / "base"), torch.bfloat16)
    merged, _ = loader._merged_copy(model, "merged_16bit")
    merged.save_pretrained(tmp_path / "copy")
    assert loader._stream_merged_save(model, str(tmp_path / "stream"), "merged_16bit") is True
    ref, ref_config = _read_checkpoint(tmp_path / "copy")
    new, new_config = _read_checkpoint(tmp_path / "stream")
    assert set(ref) == set(new) and "lm_head.weight" not in new
    assert all(torch.equal(ref[k], new[k]) for k in ref) and ref_config == new_config


def _tiny_llama():
    import transformers
    config = transformers.LlamaConfig(hidden_size = 64, intermediate_size = 128, num_hidden_layers = 2,
                                      num_attention_heads = 4, num_key_value_heads = 2, vocab_size = 256)
    torch.manual_seed(0)
    return transformers.LlamaForCausalLM(config)


def _assert_save_merged_matches_copy(model, tmp_path):
    from unsloth_zoo.cpu import loader
    merged, _ = loader._merged_copy(model, "merged_16bit")
    merged.save_pretrained(tmp_path / "copy")
    del merged
    loader._save_merged(model, str(tmp_path / "out"), "merged_16bit")
    ref, ref_config = _read_checkpoint(tmp_path / "copy")
    new, new_config = _read_checkpoint(tmp_path / "out")
    assert set(new) == set(ref)
    assert all(torch.equal(ref[k], new[k]) for k in ref) and ref_config == new_config


@pytest.mark.parametrize("adapter", ["trainable_token_indices", "ia3", "dora"])
def test_streamed_merge_falls_back_for_peft_it_cannot_merge(tmp_path, adapter):
    """Token adapters, IA3 and DoRA: the streamed writer would leave adapter tensors / unmerged or wrong weights."""
    from peft import IA3Config, LoraConfig, get_peft_model
    from unsloth_zoo.cpu import loader
    if adapter == "trainable_token_indices":
        config = LoraConfig(r = 4, target_modules = ["q_proj"], trainable_token_indices = [0, 1])
    elif adapter == "ia3":
        config = IA3Config(target_modules = ["k_proj", "v_proj", "down_proj"], feedforward_modules = ["down_proj"],
                           init_ia3_weights = False)
    else:
        config = LoraConfig(r = 4, target_modules = ["q_proj"], use_dora = True, init_lora_weights = False)
    torch.manual_seed(0)
    model = get_peft_model(_tiny_llama(), config)
    if adapter == "trainable_token_indices":
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "trainable_tokens_delta" in name:
                    parameter.add_(1.0)
    assert loader._stream_merge_source(model) is None
    assert loader._stream_merged_save(model, str(tmp_path / "x"), "merged_16bit") is False
    _assert_save_merged_matches_copy(model, tmp_path)


def test_streamed_merge_keeps_checkpoint_conversions(tmp_path):
    """transformers 5 saves Qwen3MoE experts back in the per-expert checkpoint layout; the merged save must too."""
    import transformers
    from peft import LoraConfig, get_peft_model
    if not hasattr(transformers, "Qwen3MoeConfig"):
        pytest.skip("no Qwen3MoE in this transformers")
    config = transformers.Qwen3MoeConfig(hidden_size = 64, intermediate_size = 128, moe_intermediate_size = 32,
                                         num_hidden_layers = 2, num_attention_heads = 4, num_key_value_heads = 2,
                                         vocab_size = 256, num_experts = 4, num_experts_per_tok = 2, head_dim = 16)
    torch.manual_seed(0)
    model = get_peft_model(transformers.Qwen3MoeForCausalLM(config),
                           LoraConfig(r = 4, target_modules = ["q_proj"], init_lora_weights = False))
    _assert_save_merged_matches_copy(model, tmp_path)


def test_streamed_merge_still_streams_plain_lora():
    from unsloth_zoo.cpu import loader
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(_tiny_llama(), LoraConfig(r = 4, target_modules = ["q_proj", "v_proj"]))
    assert loader._stream_merge_source(model) is not None


def test_streamed_merge_reexport_removes_obsolete_shards(tmp_path):
    """Re-export leaves only the new weights, like save_pretrained; unrelated files stay."""
    import json
    from unsloth_zoo.cpu import loader
    model = _lora_model(_TINY, torch.float32)
    out = tmp_path / "merged"
    out.mkdir()
    unrelated = ["adapter_model.safetensors", "tokenizer.json", "other_model.safetensors", "model-notes.txt",
                 "model-00001-of-00002.bin"]
    for name in unrelated:
        (out / name).write_text("keep")

    def weights():
        return sorted(f for f in os.listdir(out) if f not in unrelated and f.startswith("model"))

    def tensors():
        from safetensors.torch import load_file
        loaded = {}
        for name in weights():
            if name.endswith(".safetensors"):
                loaded.update(load_file(str(out / name)))
        return loaded

    def expected_shards():
        with open(out / "model.safetensors.index.json") as f:
            return sorted(set(json.load(f)["weight_map"].values()))

    assert loader._stream_merged_save(model, str(out), "merged_16bit", 1000) is True
    many = expected_shards()
    assert len(many) > 2 and weights() == sorted(many + ["model.safetensors.index.json"])
    assert loader._stream_merged_save(model, str(out), "merged_16bit", 4000) is True
    fewer = expected_shards()
    assert 1 < len(fewer) < len(many)
    assert weights() == sorted(fewer + ["model.safetensors.index.json"])  # no shard of the earlier export is left
    assert loader._stream_merged_save(model, str(out), "merged_16bit", None) is True
    assert weights() == ["model.safetensors"]  # single file: the stale index and shards are gone
    single = tensors()
    assert loader._stream_merged_save(model, str(out), "merged_16bit", 1000) is True
    assert weights() == sorted(expected_shards() + ["model.safetensors.index.json"])  # and the other way round
    sharded = tensors()
    assert set(sharded) == set(single) and all(torch.equal(sharded[k], single[k]) for k in single)
    assert all((out / name).read_text() == "keep" for name in unrelated)


def _file_hashes(directory):
    import hashlib
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in sorted(os.listdir(directory))}


def _lora_llama(dtype = torch.float32, vocab = 256, hidden = 64, mixed = False):
    import transformers
    from peft import LoraConfig, get_peft_model
    config = transformers.LlamaConfig(hidden_size = hidden, intermediate_size = 2 * hidden, num_hidden_layers = 2,
                                      num_attention_heads = 4, num_key_value_heads = 2, vocab_size = vocab)
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(config).to(dtype)
    if mixed:
        model.model.norm.float()  # bf16 and fp32 tensors in one file
    torch.manual_seed(1)
    return get_peft_model(model, LoraConfig(r = 4, target_modules = ["q_proj", "v_proj", "down_proj"],
                                            init_lora_weights = False))


@pytest.mark.parametrize("case", ["single", "sharded", "mixed_dtype"])
@pytest.mark.parametrize("save_method", ["merged_16bit", "merged_32bit"])
def test_streamed_merge_files_are_byte_identical(tmp_path, case, save_method):
    """Not only the tensors: every file (safetensors layout, shard grouping, index) matches save_pretrained's."""
    from unsloth_zoo.cpu import loader
    model = _lora_llama(torch.bfloat16 if case == "mixed_dtype" else torch.float32,
                        vocab = 4096 if case == "sharded" else 256, hidden = 128 if case == "sharded" else 64,
                        mixed = case == "mixed_dtype")
    kwargs = {"max_shard_size": 300000} if case == "sharded" else {}
    merged, _ = loader._merged_copy(model, save_method)
    merged.save_pretrained(tmp_path / "copy", **kwargs)
    del merged
    assert loader._stream_merged_save(model, str(tmp_path / "stream"), save_method, kwargs.get("max_shard_size"))
    expected = _file_hashes(tmp_path / "copy")
    if case == "sharded":
        assert "model.safetensors.index.json" in expected
    assert _file_hashes(tmp_path / "stream") == expected


def test_streamed_merge_default_shard_size_follows_save_pretrained(tmp_path):
    """No max_shard_size: shard as this transformers' save_pretrained would (5GB on 4.x, 50GB on 5.x)."""
    import transformers
    from unsloth_zoo.cpu import loader

    class SmallShards(transformers.LlamaForCausalLM):
        def save_pretrained(self, save_directory, max_shard_size = 300000, **kwargs):
            return super().save_pretrained(save_directory, max_shard_size = max_shard_size, **kwargs)

    import inspect
    default = inspect.signature(transformers.PreTrainedModel.save_pretrained).parameters["max_shard_size"].default
    assert loader._default_shard_size(_tiny_llama()) == default
    config = transformers.LlamaConfig(hidden_size = 128, intermediate_size = 256, num_hidden_layers = 2,
                                      num_attention_heads = 4, num_key_value_heads = 2, vocab_size = 4096)
    torch.manual_seed(0)
    model = SmallShards(config)
    model.save_pretrained(tmp_path / "copy")
    assert loader._stream_merged_save(model, str(tmp_path / "stream"), "merged_32bit") is True
    assert _file_hashes(tmp_path / "stream") == _file_hashes(tmp_path / "copy")


_SOURCE_DIR_SCRIPT = r"""
import sys, torch, transformers
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from unsloth_zoo.cpu import loader
src, ref, dtype = sys.argv[1], sys.argv[2], getattr(torch, sys.argv[3])
config = transformers.LlamaConfig(hidden_size = 256, intermediate_size = 512, num_hidden_layers = 2,
                                  num_attention_heads = 4, num_key_value_heads = 2, vocab_size = 2048)
torch.manual_seed(0)
transformers.LlamaForCausalLM(config).to(dtype).save_pretrained(src)
model = transformers.AutoModelForCausalLM.from_pretrained(src, dtype = dtype)  # weights memory-mapped from src
torch.manual_seed(1)
model = get_peft_model(model, LoraConfig(r = 4, target_modules = ["q_proj", "v_proj"], init_lora_weights = False))
loader.save_pretrained_merged(model, ref, save_method = "merged_16bit")
loader.save_pretrained_merged(model, src, save_method = "merged_16bit")
a, b = load_file(ref + "/model.safetensors"), load_file(src + "/model.safetensors")
assert set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a), "merged export differs"
print("OK")
"""


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_streamed_merge_into_the_directory_the_model_was_loaded_from(tmp_path, dtype):
    """In-place overwrite of mmapped weights destroyed the source (SIGBUS); write aside, move."""
    script = tmp_path / "run.py"
    script.write_text(_SOURCE_DIR_SCRIPT)
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1]) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    result = subprocess.run([sys.executable, str(script), str(tmp_path / "src"), str(tmp_path / "ref"), dtype],
                            capture_output = True, text = True, env = env, timeout = 600)
    assert result.returncode == 0 and "OK" in result.stdout, (result.returncode, result.stderr[-2000:])


def test_streamed_merge_interrupted_keeps_the_previous_export(tmp_path):
    """An export that fails part way leaves the previous export loadable and no partial or temporary file."""
    from unsloth_zoo.cpu import loader
    model = _lora_llama(vocab = 4096, hidden = 128)
    out = tmp_path / "out"
    loader.save_pretrained_merged(model, str(out), save_method = "merged_16bit", max_shard_size = 300000)
    before = _file_hashes(out)
    original, calls = loader._merged_weight, []

    def failing(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise KeyboardInterrupt("simulated")
        return original(*args, **kwargs)

    loader._merged_weight = failing
    try:
        with pytest.raises(KeyboardInterrupt):
            loader.save_pretrained_merged(model, str(out), save_method = "merged_16bit")
    finally:
        loader._merged_weight = original
    assert _file_hashes(out) == before
