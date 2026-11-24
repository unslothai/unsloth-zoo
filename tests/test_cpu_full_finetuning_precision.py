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

"""CPU full finetuning honours float32_mixed_precision = True as the GPU loaders do: float32 master weights."""

import pytest
import torch

TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"


def _load(**kwargs):
    from unsloth_zoo.cpu import loader
    try:
        return loader.FastCPUModel.from_pretrained(TINY, load_in_4bit = False, **kwargs)
    except OSError as error:  # offline without the tiny model cached
        pytest.skip(f"{TINY} unavailable: {error}")


def _bf16():
    from unsloth_zoo.cpu.runtime import cpu_bf16_supported
    return cpu_bf16_supported()


def _dtypes(model):
    return {p.dtype for p in model.parameters()}


def _no_free_ram(monkeypatch):
    from unsloth_zoo.cpu import loader
    monkeypatch.setattr(loader, "available_ram", lambda: 1024 ** 2)


def _needs_bf16():
    if not _bf16():
        pytest.skip("no native bfloat16: full finetuning is float32 already")


@pytest.mark.parametrize("dtype", [torch.bfloat16, None, "auto"])
def test_float32_mixed_precision_true_always_keeps_float32_master_weights(dtype, monkeypatch, capsys):
    _no_free_ram(monkeypatch)  # forced: no RAM check
    model, _ = _load(full_finetuning = True, dtype = dtype, float32_mixed_precision = True)
    assert _dtypes(model) == {torch.float32}
    assert all(p.requires_grad for p in model.parameters())
    if _bf16():
        assert "float32_mixed_precision = True: full finetuning keeps float32 weights" in capsys.readouterr().out


def test_default_full_finetuning_takes_float32_master_weights_when_they_fit(capsys):
    _needs_bf16()
    model, _ = _load(full_finetuning = True)
    assert _dtypes(model) == {torch.float32}
    out = capsys.readouterr().out
    assert "full finetuning keeps float32 master weights" in out and "fit in RAM" in out
    assert "float32_mixed_precision = False (or dtype = torch.bfloat16) keeps bfloat16 weights" in out


def test_default_full_finetuning_stays_bfloat16_when_float32_does_not_fit(monkeypatch, capsys):
    _needs_bf16()
    _no_free_ram(monkeypatch)
    model, _ = _load(full_finetuning = True)
    assert _dtypes(model) == {torch.bfloat16}
    out = capsys.readouterr().out
    assert "do not fit in RAM" in out and "rounded away at small learning rates" in out


def test_fit_check_counts_weights_grads_and_adamw_states(monkeypatch):
    from transformers import AutoConfig
    from unsloth_zoo.cpu import loader
    try:
        config = AutoConfig.from_pretrained(TINY)
    except OSError as error:
        pytest.skip(f"{TINY} unavailable: {error}")
    needed = loader._dense_parameter_count(config) * 16
    monkeypatch.setattr(loader, "available_ram", lambda: 2 * needed)
    assert loader._fp32_master_fit(TINY, config, None, None, False)[2] is True
    monkeypatch.setattr(loader, "available_ram", lambda: 2 * needed - 1)
    assert loader._fp32_master_fit(TINY, config, None, None, False)[2] is False
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setattr(loader, "available_ram", lambda: 2 * needed)
    assert loader._fp32_master_fit(TINY, config, None, None, False)[2] is False


@pytest.mark.parametrize("kwargs", [dict(float32_mixed_precision = False), dict(dtype = torch.bfloat16),
                                    dict(torch_dtype = torch.bfloat16)])
def test_user_choice_of_bfloat16_wins(kwargs, capsys):
    _needs_bf16()
    model, _ = _load(full_finetuning = True, **kwargs)
    assert _dtypes(model) == {torch.bfloat16}
    out = capsys.readouterr().out
    assert "float32_mixed_precision = True keeps float32 master weights" in out
    assert "fit in RAM" not in out


@pytest.mark.parametrize("value", [None, True])
def test_lora_is_unchanged(value, capsys):
    _needs_bf16()
    model, _ = _load(full_finetuning = False, float32_mixed_precision = value)
    assert _dtypes(model) == {torch.bfloat16}
    assert "master weights" not in capsys.readouterr().out


def test_explicit_float32_unchanged(capsys):
    model, _ = _load(full_finetuning = True, dtype = torch.float32)
    assert _dtypes(model) == {torch.float32}
    assert "bfloat16 full finetuning" not in capsys.readouterr().out


def test_decision_lives_only_in_the_cpu_loader():
    """The GPU loaders never see it: nothing outside unsloth_zoo/cpu names the CPU precision helpers."""
    import pathlib
    import unsloth_zoo
    root = pathlib.Path(unsloth_zoo.__file__).parent
    users = sorted(str(path.relative_to(root)) for path in root.rglob("*.py")
                   if any(name in path.read_text(errors = "ignore") for name in ("_fp32_master_fit", "_print_full_finetuning_precision")))
    assert users == ["cpu/loader.py"]


def _changed_fraction(model, lr):
    """One bf16-autocast forward / backward and one fused AdamW step at ``lr``: the fraction of weights that moved."""
    torch.manual_seed(0)
    ids = torch.randint(3, 100, (2, 16))
    before = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, fused = True)
    with torch.autocast("cpu", dtype = torch.bfloat16):
        loss = model(input_ids = ids, labels = ids).loss
    loss.backward()
    optimizer.step()
    moved = sum(int((p.detach() != b).sum()) for p, b in zip(model.parameters(), before))
    return moved / sum(b.numel() for b in before)


def test_default_master_weights_take_updates_bfloat16_weights_drop():
    _needs_bf16()
    default, _ = _load(full_finetuning = True)
    bf16, _ = _load(full_finetuning = True, float32_mixed_precision = False)
    assert _dtypes(default) == {torch.float32} and _dtypes(bf16) == {torch.bfloat16}
    default.load_state_dict({k: v.float() for k, v in bf16.state_dict().items()})
    # Adam's first step moves every weight by about lr; below a bf16 weight's spacing that rounds away.
    assert _changed_fraction(default, 1e-5) > 0.99
    assert _changed_fraction(bf16, 1e-5) < 0.5
