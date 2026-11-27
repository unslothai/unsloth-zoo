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

"""Class-assigned forwards keep past_key_values; standalone 5.24 declares cache, so pass cache."""

import importlib.util
import sys
import textwrap

import pytest
import torch

pytest.importorskip("transformers.utils.deprecation")

from unsloth_zoo import compiler
from unsloth_zoo.temporary_patches.utils import torch_compiler_disable_unless_decode

_FAKE = """
    import torch
    from torch import nn
    from transformers.utils.deprecation import deprecate_kwarg


    def zoo_ple_layer_forward(self, hidden_states, input_ids, past_key_values, conv_mask = None):
        return self.proj(hidden_states), past_key_values


    def zoo_ple_layer_forward_new(self, hidden_states, input_ids, cache, conv_mask = None):
        return self.proj(hidden_states), cache


    class FakePleLayer(nn.Module):
        def __init__(self, config = None):
            super().__init__()
            self.proj = nn.Linear(4, 4, bias = False)

        # As transformers 5.24 declares it.
        @deprecate_kwarg("past_key_values", "5.40.0", new_name = "cache")
        def forward(
            self,
            hidden_states: torch.Tensor,
            input_ids: torch.Tensor,
            cache,
            conv_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            return self.proj(hidden_states), cache
"""


def _generate(tmp_path, monkeypatch, replacement):
    name = "fake_ple_direct_assign_modeling"
    path = tmp_path / f"{name}.py"
    path.write_text(textwrap.dedent(_FAKE), encoding = "utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(compiler, name, module, raising = False)
    fn = getattr(module, replacement)
    monkeypatch.setattr(module.FakePleLayer, "forward", fn)  # no patch_function, so no adapter
    generated = compiler.create_standalone_class("FakePleLayer", name, dir(module), disable = True)
    namespace = {
        "torch": torch, "nn": torch.nn, "torch_compile_options": {},
        "torch_compiler_disable_unless_decode": torch_compiler_disable_unless_decode,
        "deprecate_kwarg": module.deprecate_kwarg, replacement: fn,
    }
    exec(compile(generated, "<ple-direct-assign-codegen>", "exec"), namespace)
    return generated, namespace["FakePleLayer"]()


def test_direct_assignment_passes_the_stock_cache_name(tmp_path, monkeypatch):
    generated, layer = _generate(tmp_path, monkeypatch, "zoo_ple_layer_forward")
    assert "past_key_values=cache" in generated
    x, ids, cache = torch.randn(1, 2, 4), torch.zeros(1, 2, dtype = torch.long), object()
    assert layer(x, ids, cache = cache)[1] is cache
    assert layer(x, ids, cache)[1] is cache
    assert layer(x, ids, None)[1] is None


def test_replacement_already_on_the_new_name_is_untouched(tmp_path, monkeypatch):
    generated, layer = _generate(tmp_path, monkeypatch, "zoo_ple_layer_forward_new")
    assert "cache=cache" in generated and "past_key_values=" not in generated
    x, ids, cache = torch.randn(1, 2, 4), torch.zeros(1, 2, dtype = torch.long), object()
    assert layer(x, ids, cache = cache)[1] is cache
