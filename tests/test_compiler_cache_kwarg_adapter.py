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

"""The compiler's standalone class must re-apply the 5.24 `cache` adapter or `cache=` is lost in **kwargs."""

import importlib.util
import sys
import textwrap

import pytest
import torch

pytest.importorskip("transformers.utils.deprecation")

from unsloth_zoo import compiler
from unsloth_zoo.temporary_patches import utils as U
from unsloth_zoo.temporary_patches.utils import torch_compiler_disable_unless_decode

_FAKE = """
    import torch
    from torch import nn
    from transformers.utils.deprecation import deprecate_kwarg


    def zoo_attention_forward(self, hidden_states, position_embeddings = None, past_key_values = None, **kwargs):
        seen = self.proj(hidden_states)
        return seen, past_key_values


    class FakeCacheAttention(nn.Module):
        def __init__(self, config = None, layer_idx = 0):
            super().__init__()
            self.proj = nn.Linear(4, 4, bias = False)

        # As transformers 5.24 declares it.
        @deprecate_kwarg("past_key_values", "5.40.0", new_name = "cache")
        def forward(self, hidden_states, position_embeddings = None, cache = None, **kwargs):
            return self.proj(hidden_states), cache
"""


def _fake_module(tmp_path, monkeypatch):
    name = "fake_cache_kwarg_modeling"
    path = tmp_path / f"{name}.py"
    path.write_text(textwrap.dedent(_FAKE), encoding = "utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(compiler, name, module, raising = False)
    return name, module


def test_standalone_class_keeps_the_cache_keyword_adapter(tmp_path, monkeypatch):
    name, module = _fake_module(tmp_path, monkeypatch)
    cls = module.FakeCacheAttention
    monkeypatch.setattr(cls, "forward", cls.forward)
    assert U.patch_function(cls, "forward", module.zoo_attention_forward)
    assert getattr(cls.forward, "_unsloth_cache_kwarg_adapter", None) == ("past_key_values", "cache")

    generated = compiler.create_standalone_class("FakeCacheAttention", name, dir(module), disable = True)
    namespace = {
        "torch": torch, "nn": torch.nn, "torch_compile_options": {},
        "torch_compiler_disable_unless_decode": torch_compiler_disable_unless_decode,
        "deprecate_kwarg": module.deprecate_kwarg, "zoo_attention_forward": module.zoo_attention_forward,
    }
    exec(compile(generated, "<cache-adapter-codegen>", "exec"), namespace)
    generated_cls = namespace["FakeCacheAttention"]

    layer = generated_cls()
    x, cache = torch.randn(1, 2, 4), object()
    assert layer(x, cache = cache)[1] is cache, "a 5.24 caller's cache= must reach the replacement"
    assert layer(x, None, cache)[1] is cache, "positional"
    assert layer(x, past_key_values = cache)[1] is cache, "zoo's own past_key_values= callers"
    assert layer(x)[1] is None


def test_plain_forward_codegen_is_unchanged(tmp_path, monkeypatch):
    name, module = _fake_module(tmp_path, monkeypatch)
    generated = compiler.create_standalone_class("FakeCacheAttention", name, dir(module), disable = True)
    assert "_unsloth_ensure_cache_kwarg" not in generated
