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


"""CPU float32 full finetuning warns before loading when weights, grads and AdamW states do not fit in free RAM (it
would swap: Qwen3-0.6B on a 7 GB Apple M1 took 266 s per step). Every checkpoint is built locally: no hub access."""

import pytest
import torch

_RANK_VARS = ("LOCAL_WORLD_SIZE", "MPI_LOCALNRANKS", "OMPI_COMM_WORLD_LOCAL_SIZE", "MV2_COMM_WORLD_LOCAL_SIZE",
              "WORLD_SIZE", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE")


def _checkpoint(path, dtype):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    config = Qwen3Config(vocab_size = 128, hidden_size = 32, intermediate_size = 64, num_hidden_layers = 2,
                         num_attention_heads = 2, num_key_value_heads = 1, head_dim = 16, max_position_embeddings = 64,
                         tie_word_embeddings = True)
    torch.manual_seed(0)
    Qwen3ForCausalLM(config).to(dtype).save_pretrained(path)
    vocab = {"<unk>": 0, "<eos>": 1, **{f"t{i}": i for i in range(2, 128)}}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token = "<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object = tokenizer, unk_token = "<unk>", eos_token = "<eos>").save_pretrained(path)
    return str(path)


@pytest.fixture(scope = "module")
def tiny(tmp_path_factory):
    return _checkpoint(tmp_path_factory.mktemp("tiny-bf16") / "tiny", torch.bfloat16)


@pytest.fixture(scope = "module")
def tiny_fp32(tmp_path_factory):
    return _checkpoint(tmp_path_factory.mktemp("tiny-fp32") / "tiny", torch.float32)


@pytest.fixture(scope = "module")
def tiny_fp16(tmp_path_factory):
    return _checkpoint(tmp_path_factory.mktemp("tiny-fp16") / "tiny", torch.float16)


@pytest.fixture(autouse = True)
def _single_process(monkeypatch):
    for name in _RANK_VARS:
        monkeypatch.delenv(name, raising = False)


def _needed(path):
    from transformers import AutoConfig
    from unsloth_zoo.cpu import loader
    return loader._dense_parameter_count(AutoConfig.from_pretrained(path)) * 16


class _Captured(Exception):
    pass


def _load(monkeypatch, path, ram, bf16, **kwargs):
    """What the loader printed and the dtype it hands transformers, without building the model."""
    import transformers
    from unsloth_zoo.cpu import loader
    seen, printed = {}, []

    def capture(name, *args, **load_kwargs):
        seen["dtype"] = load_kwargs.get(loader._DTYPE_KWARG)
        raise _Captured

    monkeypatch.setattr(loader, "available_ram", lambda: ram)
    monkeypatch.setattr(loader, "cpu_bf16_supported", lambda: bf16)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", capture)
    monkeypatch.setattr(loader, "_print", printed.append)
    with pytest.raises(_Captured):
        loader.FastCPUModel.from_pretrained(path, load_in_4bit = False, local_files_only = True, **kwargs)
    return seen["dtype"], "\n".join(printed)


def _warned(text):
    return "float32 full finetuning needs about" in text and "will swap" in text


def test_no_native_bf16_full_finetuning_warns_when_it_does_not_fit(tiny, monkeypatch):
    dtype, out = _load(monkeypatch, tiny, _needed(tiny) // 2, False, full_finetuning = True)
    assert dtype == torch.float32
    assert _warned(out), out
    assert "get_peft_model" in out


@pytest.mark.parametrize("factor", [1, 4])
def test_no_warning_when_it_fits(tiny, monkeypatch, factor):
    _, out = _load(monkeypatch, tiny, _needed(tiny) * factor, False, full_finetuning = True)
    assert not _warned(out), out


def test_forced_float32_master_weights_warn_when_they_do_not_fit(tiny, monkeypatch):
    dtype, out = _load(monkeypatch, tiny, _needed(tiny) // 2, True, full_finetuning = True,
                       float32_mixed_precision = True)
    assert dtype == torch.float32
    assert _warned(out), out


def test_bfloat16_fallback_does_not_warn(tiny, monkeypatch):
    """Native bf16 and float32 too big: the loader already picks bfloat16 weights and says so."""
    dtype, out = _load(monkeypatch, tiny, _needed(tiny) // 2, True, full_finetuning = True)
    assert dtype == torch.bfloat16
    assert not _warned(out), out


@pytest.mark.parametrize("ram", [None, 1])
def test_lora_and_unknown_ram_do_not_warn(tiny, monkeypatch, ram):
    _, out = _load(monkeypatch, tiny, ram, False)
    assert not _warned(out), out
    if ram is None:
        _, out = _load(monkeypatch, tiny, None, False, full_finetuning = True)
        assert not _warned(out), out


def test_several_processes_do_not_warn(tiny, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    _, out = _load(monkeypatch, tiny, 1, False, full_finetuning = True)
    assert not _warned(out), out


@pytest.mark.parametrize("key", ["dtype", "torch_dtype"])
def test_auto_dtype_sizes_the_dtype_that_loads(tiny, tiny_fp32, tiny_fp16, monkeypatch, key):
    """dtype = "auto" loads the checkpoint's dtype: a bf16 checkpoint stays bf16 (no float32 state, no warning) even on a
    CPU without native bf16, a float32 one is float32 even on a native-bf16 CPU, and float16 becomes the CPU's."""
    low = _needed(tiny) // 2
    _, out = _load(monkeypatch, tiny, low, False, full_finetuning = True, **{key: "auto"})
    assert not _warned(out), out
    _, out = _load(monkeypatch, tiny_fp32, low, True, full_finetuning = True, **{key: "auto"})
    assert _warned(out), out
    _, out = _load(monkeypatch, tiny_fp16, low, False, full_finetuning = True, **{key: "auto"})
    assert _warned(out), out
    _, out = _load(monkeypatch, tiny_fp16, low, True, full_finetuning = True, **{key: "auto"})
    assert not _warned(out), out
