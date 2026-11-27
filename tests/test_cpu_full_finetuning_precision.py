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

"""CPU full finetuning precision: float32 master weights by default when they fit in RAM on a native-bf16 CPU, and
always with float32_mixed_precision = True; bfloat16 weights with float32_mixed_precision = False or an explicit
dtype. Every checkpoint is built locally: no hub access."""

import os
import types

import pytest
import torch

_RANK_VARS = ("LOCAL_WORLD_SIZE", "MPI_LOCALNRANKS", "OMPI_COMM_WORLD_LOCAL_SIZE", "MV2_COMM_WORLD_LOCAL_SIZE",
              "WORLD_SIZE", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE")
_PLENTY = 1024 ** 4


def _write_checkpoint(path, dtype, sentinel = None):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size = 128, hidden_size = 32, intermediate_size = 64, num_hidden_layers = 2, num_attention_heads = 2,
        num_key_value_heads = 1, head_dim = 16, max_position_embeddings = 64, tie_word_embeddings = True,
    )
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(config).to(dtype)
    if sentinel is not None:
        with torch.no_grad():
            model.get_input_embeddings().weight[0, 0] = sentinel
    model.save_pretrained(path)
    vocab = {"<unk>": 0, "<eos>": 1, **{f"t{i}": i for i in range(2, 128)}}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token = "<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object = tokenizer, unk_token = "<unk>", eos_token = "<eos>").save_pretrained(path)
    return str(path)


@pytest.fixture(scope = "module")
def tiny(tmp_path_factory):
    return _write_checkpoint(tmp_path_factory.mktemp("tiny-bf16") / "tiny", torch.bfloat16)


@pytest.fixture(autouse = True)
def _single_process_plenty_of_ram(monkeypatch):
    """One process with ample RAM, whatever the runner's launcher and free RAM: each test sets its own budget."""
    from unsloth_zoo.cpu import loader
    for name in _RANK_VARS:
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(loader, "available_ram", lambda: _PLENTY)


@pytest.fixture
def bf16_cpu(monkeypatch):
    """The dtype policy under test is the native-bf16 one, whatever this runner's CPU."""
    from unsloth_zoo.cpu import loader
    monkeypatch.setattr(loader, "cpu_bf16_supported", lambda: True)


def _ram(monkeypatch, value):
    from unsloth_zoo.cpu import loader
    monkeypatch.setattr(loader, "available_ram", lambda: value)


def _load(path, **kwargs):
    from unsloth_zoo.cpu import loader
    return loader.FastCPUModel.from_pretrained(path, load_in_4bit = False, local_files_only = True, **kwargs)


def _dtypes(model):
    return {p.dtype for p in model.parameters()}


def _bytes_needed(path, **overrides):
    from transformers import AutoConfig
    from unsloth_zoo.cpu import loader
    config = AutoConfig.from_pretrained(path, **overrides)
    return loader._dense_parameter_count(config) * 16


class _Captured(Exception):
    pass


def _load_dtype(monkeypatch, path, **kwargs):
    """The dtype the loader hands transformers, and what it printed, without building the model."""
    import transformers
    from unsloth_zoo.cpu import loader

    seen, printed = {}, []

    def capture(name, *args, **load_kwargs):
        seen["dtype"] = load_kwargs.get(loader._DTYPE_KWARG)
        raise _Captured

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", capture)
    monkeypatch.setattr(loader, "_print", printed.append)
    with pytest.raises(_Captured):
        _load(path, **kwargs)
    return seen["dtype"], "\n".join(printed)


@pytest.mark.usefixtures("bf16_cpu")
@pytest.mark.parametrize("dtype", [torch.bfloat16, None, "auto"])
def test_float32_mixed_precision_true_always_keeps_float32_master_weights(tiny, dtype, monkeypatch, capsys):
    _ram(monkeypatch, 0)  # forced: no RAM check
    model, _ = _load(tiny, full_finetuning = True, dtype = dtype, float32_mixed_precision = True)
    assert _dtypes(model) == {torch.float32}
    assert all(p.requires_grad for p in model.parameters())
    assert "float32_mixed_precision = True: full finetuning keeps float32 weights" in capsys.readouterr().out


@pytest.mark.usefixtures("bf16_cpu")
def test_default_full_finetuning_takes_float32_master_weights_when_they_fit(tiny, capsys):
    model, _ = _load(tiny, full_finetuning = True)
    assert _dtypes(model) == {torch.float32}
    out = capsys.readouterr().out
    assert "full finetuning keeps float32 master weights" in out and "fit in RAM" in out
    assert "float32_mixed_precision = False (or dtype = torch.bfloat16) keeps bfloat16 weights" in out


@pytest.mark.usefixtures("bf16_cpu")
def test_default_full_finetuning_stays_bfloat16_when_float32_does_not_fit(tiny, monkeypatch, capsys):
    _ram(monkeypatch, 0)
    model, _ = _load(tiny, full_finetuning = True)
    assert _dtypes(model) == {torch.bfloat16}
    out = capsys.readouterr().out
    assert "do not fit in RAM" in out and "rounded away at small learning rates" in out


@pytest.mark.usefixtures("bf16_cpu")
def test_fit_check_counts_weights_grads_and_adamw_states(tiny, monkeypatch):
    needed = _bytes_needed(tiny)
    _ram(monkeypatch, 2 * needed)
    assert _load_dtype(monkeypatch, tiny, full_finetuning = True)[0] == torch.float32
    _ram(monkeypatch, 2 * needed - 1)
    assert _load_dtype(monkeypatch, tiny, full_finetuning = True)[0] == torch.bfloat16


@pytest.mark.usefixtures("bf16_cpu")
@pytest.mark.parametrize("kwargs", [dict(float32_mixed_precision = False), dict(dtype = torch.bfloat16),
                                    dict(torch_dtype = torch.bfloat16)])
def test_user_choice_of_bfloat16_wins(tiny, kwargs, capsys):
    model, _ = _load(tiny, full_finetuning = True, **kwargs)
    assert _dtypes(model) == {torch.bfloat16}
    out = capsys.readouterr().out
    assert "float32_mixed_precision = True keeps float32 master weights" in out
    assert "fit in RAM" not in out


@pytest.mark.usefixtures("bf16_cpu")
@pytest.mark.parametrize("value", [None, True])
def test_lora_is_unchanged(tiny, value, capsys):
    model, _ = _load(tiny, full_finetuning = False, float32_mixed_precision = value)
    assert _dtypes(model) == {torch.bfloat16}
    assert "master weights" not in capsys.readouterr().out


def test_explicit_float32_unchanged(tiny, capsys):
    model, _ = _load(tiny, full_finetuning = True, dtype = torch.float32)
    assert _dtypes(model) == {torch.float32}
    assert "bfloat16 full finetuning" not in capsys.readouterr().out


def test_no_native_bfloat16_keeps_float32(tiny, monkeypatch, capsys):
    from unsloth_zoo.cpu import loader
    monkeypatch.setattr(loader, "cpu_bf16_supported", lambda: False)
    _ram(monkeypatch, 0)
    model, _ = _load(tiny, full_finetuning = True)
    assert _dtypes(model) == {torch.float32}
    assert "master weights" not in capsys.readouterr().out


def test_decision_lives_only_in_the_cpu_loader():
    """The GPU loaders never see it: nothing outside unsloth_zoo/cpu names the CPU precision helpers."""
    import pathlib
    import unsloth_zoo
    root = pathlib.Path(unsloth_zoo.__file__).parent
    users = sorted(path.relative_to(root).as_posix() for path in root.rglob("*.py")
                   if any(name in path.read_text(errors = "ignore") for name in ("_fp32_master_fit", "_print_full_finetuning_precision")))
    assert users == ["cpu/loader.py"]


# The forced path loads float32 straight from the checkpoint.

@pytest.mark.usefixtures("bf16_cpu")
def test_forced_auto_keeps_float16_checkpoint_values_exact(tmp_path):
    # 1 + 2**-10 is exact in float16 and rounds to 1.0 in bfloat16.
    path = _write_checkpoint(tmp_path / "fp16", torch.float16, sentinel = 1.0009765625)
    model, _ = _load(path, full_finetuning = True, dtype = "auto", float32_mixed_precision = True)
    assert _dtypes(model) == {torch.float32}
    assert model.get_input_embeddings().weight[0, 0].item() == 1.0009765625
    assert model.config.torch_dtype == torch.float32


@pytest.mark.usefixtures("bf16_cpu")
@pytest.mark.parametrize("dtype", [torch.float64, torch.float16, "float16"])
def test_forced_float32_overrides_any_explicit_dtype(tiny, dtype, monkeypatch):
    loaded, printed = _load_dtype(monkeypatch, tiny, full_finetuning = True, dtype = dtype,
                                  float32_mixed_precision = True)
    assert loaded == torch.float32
    assert "using torch.bfloat16 instead" not in printed



@pytest.mark.usefixtures("bf16_cpu")
def test_fit_check_counts_a_resized_vocabulary(tiny, monkeypatch):
    needed = _bytes_needed(tiny)
    _ram(monkeypatch, 2 * needed)  # the saved vocabulary fits exactly
    assert _load_dtype(monkeypatch, tiny, full_finetuning = True)[0] == torch.float32
    assert _load_dtype(monkeypatch, tiny, full_finetuning = True, resize_model_vocab = 4096)[0] == torch.bfloat16


@pytest.mark.usefixtures("bf16_cpu")
@pytest.mark.parametrize("vocab_size, resize, expected", [
    (256, None, torch.float32), (256, 128, torch.float32),       # the override sizes the load
    (4096, None, torch.bfloat16), (4096, 256, torch.bfloat16),   # a larger override than the resize counts
    (128, 4096, torch.bfloat16),                                 # a larger resize than the override counts
])
def test_fit_check_takes_a_vocab_size_override(tiny, monkeypatch, vocab_size, resize, expected):
    # vocab_size is a config override of the load, never a duplicate of the pending resize argument.
    _ram(monkeypatch, 2 * _bytes_needed(tiny, vocab_size = 256))
    extra = {} if resize is None else dict(resize_model_vocab = resize)
    loaded, _ = _load_dtype(monkeypatch, tiny, full_finetuning = True, vocab_size = vocab_size,
                            ignore_mismatched_sizes = True, **extra)
    assert loaded == expected


@pytest.mark.usefixtures("bf16_cpu")
def test_fit_check_applies_config_overrides(tiny, monkeypatch):
    _ram(monkeypatch, 2 * _bytes_needed(tiny))
    loaded, _ = _load_dtype(monkeypatch, tiny, full_finetuning = True, intermediate_size = 4096,
                            ignore_mismatched_sizes = True)
    assert loaded == torch.bfloat16


@pytest.mark.usefixtures("bf16_cpu")
def test_fit_check_sizes_the_checkpoint_behind_a_path_valued_config(tiny, monkeypatch):
    # AutoModelForCausalLM reads the checkpoint's own config for a config= that is not a config object.
    assert _load_dtype(monkeypatch, tiny, full_finetuning = True, config = os.path.join(tiny, "config.json"))[0] \
        == torch.float32


@pytest.mark.usefixtures("bf16_cpu")
def test_fit_check_sizes_the_base_of_a_prequantized_name(tiny, monkeypatch):
    # Only the base checkpoint is present: the -unsloth-bnb-4bit name redirects to it before sizing.
    assert _load_dtype(monkeypatch, tiny + "-unsloth-bnb-4bit", full_finetuning = True)[0] == torch.float32


@pytest.mark.usefixtures("bf16_cpu")
def test_fit_check_forwards_code_revision(tiny, monkeypatch):
    import transformers
    calls = []
    original = transformers.AutoConfig.from_pretrained

    def record(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **{k: v for k, v in kwargs.items() if k != "code_revision"})

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", record)
    _load_dtype(monkeypatch, tiny, full_finetuning = True, code_revision = "pinned")
    assert any(call.get("code_revision") == "pinned" for call in calls)


@pytest.mark.usefixtures("bf16_cpu")
@pytest.mark.parametrize("env", [dict(LOCAL_WORLD_SIZE = "2"), dict(OMPI_COMM_WORLD_LOCAL_SIZE = "2"),
                                 dict(MPI_LOCALNRANKS = "2"), dict(WORLD_SIZE = "2"), dict(PMI_SIZE = "2")])
def test_several_processes_keep_bfloat16_unless_forced(tiny, env, monkeypatch, capsys):
    # Each rank reads its own free RAM: an automatic choice could differ across ranks and break DDP's sync.
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    model, _ = _load(tiny, full_finetuning = True)
    assert _dtypes(model) == {torch.bfloat16}
    out = capsys.readouterr().out
    assert "training processes" in out and "float32_mixed_precision = True forces" in out
    assert _load_dtype(monkeypatch, tiny, full_finetuning = True, float32_mixed_precision = True)[0] == torch.float32


@pytest.mark.parametrize("fields", [
    dict(intermediate_size = [64, 128]),  # per-layer widths
    dict(vocab_size_per_layer_input = 262144, hidden_size_per_layer_input = 256),  # per-layer embeddings
    dict(head_dim = [16, 32]),
])
def test_unsupported_layouts_are_not_sized(fields):
    from unsloth_zoo.cpu import loader
    base = dict(hidden_size = 32, num_hidden_layers = 2, vocab_size = 128, num_attention_heads = 2,
                num_key_value_heads = 1, head_dim = 16, intermediate_size = 64)
    assert loader._dense_parameter_count(types.SimpleNamespace(**base)) is not None
    assert loader._dense_parameter_count(types.SimpleNamespace(**dict(base, **fields))) is None


def test_a_config_that_raises_is_not_sized():
    from unsloth_zoo.cpu import loader

    class Heterogeneous:
        hidden_size, num_hidden_layers, vocab_size, num_attention_heads = 32, 2, 128, 2

        @property
        def head_dim(self):
            raise ValueError("head_dim differs per layer")

    assert loader._dense_parameter_count(Heterogeneous()) is None


@pytest.mark.usefixtures("bf16_cpu")
def test_float32_master_weights_save_and_reload_as_float32(tiny, tmp_path):
    from safetensors.torch import load_file
    model, tokenizer = _load(tiny, full_finetuning = True)
    assert model.config.torch_dtype == torch.float32
    model.save_pretrained(tmp_path / "out")
    saved = load_file(str(tmp_path / "out" / "model.safetensors"))
    assert {t.dtype for t in saved.values()} == {torch.float32}


def _changed_fraction(model, lr):
    """One bf16-autocast step at lr: returns the moved-weight fraction and AdamW state dtypes."""
    torch.manual_seed(0)
    ids = torch.randint(3, 100, (2, 16))
    before = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, fused = True)
    with torch.autocast("cpu", dtype = torch.bfloat16):
        loss = model(input_ids = ids, labels = ids).loss
    loss.backward()
    optimizer.step()
    moved = sum(int((p.detach() != b).sum()) for p, b in zip(model.parameters(), before))
    states = {v.dtype for s in optimizer.state.values() for k, v in s.items() if k in ("exp_avg", "exp_avg_sq")}
    return moved / sum(b.numel() for b in before), states


def test_default_master_weights_take_updates_bfloat16_weights_drop(tiny):
    from unsloth_zoo.cpu.runtime import cpu_bf16_supported
    if not cpu_bf16_supported():
        pytest.skip("no native bfloat16: full finetuning is float32 already")
    default, _ = _load(tiny, full_finetuning = True)
    bf16, _ = _load(tiny, full_finetuning = True, float32_mixed_precision = False)
    assert _dtypes(default) == {torch.float32} and _dtypes(bf16) == {torch.bfloat16}
    default.load_state_dict({k: v.float() for k, v in bf16.state_dict().items()})
    # Adam's first step moves every weight by about lr; below a bf16 weight's spacing that rounds away.
    fraction, states = _changed_fraction(default, 1e-5)
    assert fraction > 0.99 and states == {torch.float32}
    fraction, states = _changed_fraction(bf16, 1e-5)
    assert fraction < 0.5 and states == {torch.bfloat16}
