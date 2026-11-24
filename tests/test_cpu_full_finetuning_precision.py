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


@pytest.mark.parametrize("dtype", [torch.bfloat16, None, "auto"])
def test_float32_mixed_precision_keeps_float32_master_weights(dtype, capsys):
    model, _ = _load(full_finetuning = True, dtype = dtype, float32_mixed_precision = True)
    assert _dtypes(model) == {torch.float32}
    assert all(p.requires_grad for p in model.parameters())
    if _bf16():
        assert "float32_mixed_precision = True: full finetuning keeps float32 weights" in capsys.readouterr().out


@pytest.mark.parametrize("value", [None, False])
def test_bfloat16_full_finetuning_stays_bfloat16_and_says_so(value, capsys):
    if not _bf16():
        pytest.skip("no native bfloat16: full finetuning is float32 already")
    model, _ = _load(full_finetuning = True, float32_mixed_precision = value)
    assert _dtypes(model) == {torch.bfloat16}
    assert "float32_mixed_precision = True keeps float32 weights" in capsys.readouterr().out


def test_lora_ignores_float32_mixed_precision():
    if not _bf16():
        pytest.skip("no native bfloat16")
    model, _ = _load(full_finetuning = False, float32_mixed_precision = True)
    assert _dtypes(model) == {torch.bfloat16}


def test_explicit_float32_unchanged(capsys):
    model, _ = _load(full_finetuning = True, dtype = torch.float32)
    assert _dtypes(model) == {torch.float32}
    assert "bfloat16 full finetuning" not in capsys.readouterr().out


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


def test_float32_master_weights_take_updates_bfloat16_weights_drop():
    if not _bf16():
        pytest.skip("no native bfloat16")
    bf16, _ = _load(full_finetuning = True)
    fp32, _ = _load(full_finetuning = True, float32_mixed_precision = True)
    fp32.load_state_dict({k: v.float() for k, v in bf16.state_dict().items()})
    # Adam's first step moves every weight by about lr; below a bf16 weight's spacing that rounds away.
    assert _changed_fraction(fp32, 1e-5) > 0.99
    assert _changed_fraction(bf16, 1e-5) < 0.5
