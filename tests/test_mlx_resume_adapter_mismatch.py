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

"""Resuming from an adapter that does not cover every LoRA module now attached."""

import warnings

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

RANK, HIDDEN = 4, 8


def _lora_module(wrapped=False):
    module = nn.Linear(HIDDEN, HIDDEN)
    if wrapped:                     # adapters held as modules, not arrays
        module.lora_a = nn.Linear(HIDDEN, RANK)
        module.lora_b = nn.Linear(RANK, HIDDEN)
    else:
        module.lora_a = mx.zeros((HIDDEN, RANK))
        module.lora_b = mx.zeros((RANK, HIDDEN))
    return module


def _model(names, wrapped=False):
    model = type("M", (nn.Module,), {})()
    for name in names:
        setattr(model, name, _lora_module(wrapped))
    return model


def _checkpoint(tmp_path, names, wrapped=False):
    path = tmp_path / "adapters.safetensors"
    suffix = ".weight" if wrapped else ""
    mx.save_safetensors(str(path), {
        f"{name}.{leaf}{suffix}": mx.zeros((HIDDEN, RANK) if leaf == "lora_a"
                                           else (RANK, HIDDEN))
        for name in names for leaf in ("lora_a", "lora_b")})
    return str(path)


def _warn_on(model, path):
    from unsloth_zoo.mlx.trainer import _warn_resume_adapter_mismatch
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _warn_resume_adapter_mismatch(model, path)
    return [str(w.message) for w in caught]


# A wrapper holding its adapters as modules stores them one level down, so the
# same coverage is spelled `q_proj.lora_a.weight` rather than `q_proj.lora_a`.
@pytest.mark.parametrize("wrapped", [False, True],
                         ids=["adapters as arrays", "adapters as modules"])
def test_a_checkpoint_covering_every_module_is_silent(tmp_path, wrapped):
    names = ["q_proj", "o_proj"]
    assert _warn_on(_model(names, wrapped),
                    _checkpoint(tmp_path, names, wrapped)) == []


def test_a_narrower_checkpoint_names_what_it_does_not_cover(tmp_path):
    # What an adapter written when selection was narrower looks like on resume.
    path = _checkpoint(tmp_path, ["q_proj"])
    said = _warn_on(_model(["q_proj", "in_proj_qkv", "out_proj"]), path)
    assert len(said) == 1
    assert "2 of the LoRA modules" in said[0]
    assert "in_proj_qkv" in said[0] and "out_proj" in said[0]
    assert "q_proj" not in said[0].split("for example")[1]


def test_a_missing_checkpoint_is_not_an_error(tmp_path):
    # Reporting must never be the thing that stops a resume.
    assert _warn_on(_model(["q_proj"]), str(tmp_path / "absent.safetensors")) == []
