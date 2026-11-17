# SPDX-License-Identifier: AGPL-3.0-or-later
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

"""UNSLOTH_QLORA_MERGE_BASE picks dequant(W4) or W16 as the QLoRA merge base."""

from types import SimpleNamespace

import pytest

from unsloth_zoo import saving_utils as S


def _model(model_type):
    return SimpleNamespace(config=SimpleNamespace(model_type=model_type))


@pytest.mark.parametrize("value", [None, "auto", ""])
def test_default_keeps_per_architecture_choice(value, monkeypatch):
    if value is None:
        monkeypatch.delenv("UNSLOTH_QLORA_MERGE_BASE", raising=False)
    else:
        monkeypatch.setenv("UNSLOTH_QLORA_MERGE_BASE", value)
    assert S._model_type_needs_dequant_merge_base(_model("falcon_h1")) is True
    assert S._model_type_needs_dequant_merge_base(_model("qwen3")) is False


@pytest.mark.parametrize("value", ["dequant", "DEQUANT", "4bit"])
def test_dequant_forces_the_trained_base_for_every_architecture(value, monkeypatch):
    monkeypatch.setenv("UNSLOTH_QLORA_MERGE_BASE", value)
    for mt in ("qwen3", "llama", "gemma3", "falcon_h1"):
        assert S._model_type_needs_dequant_merge_base(_model(mt)) is True


def test_original_forces_the_downloaded_16bit_base(monkeypatch):
    monkeypatch.setenv("UNSLOTH_QLORA_MERGE_BASE", "original")
    assert S._model_type_needs_dequant_merge_base(_model("falcon_h1")) is False


def test_every_lora_merge_of_the_16bit_path_forwards_the_choice():
    import ast
    import inspect
    import textwrap

    for function, callee in ((S._merge_and_overwrite_lora, "_merge_moe_experts_file"), (S._merge_moe_experts_file, "_merge_lora")):
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == callee]
        assert calls, callee
        for call in calls:
            assert any(k.arg == "use_dequant_base" for k in call.keywords), (function.__name__, callee)
