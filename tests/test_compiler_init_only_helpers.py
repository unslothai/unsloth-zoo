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

"""Init-only module helpers (Qwen4Exp `_build_layer_multipliers`) must not be compiled."""
import importlib
import inspect

import pytest

SOURCE = '''
def _mix(value):
    return value * 3

def _build_layer_multipliers(vocab, n, index, seed: int):
    return torch.tensor([_mix(seed + i) for i in range(n)])

def rotate_half(x):
    return x

def used_in_forward(x):
    return x

TABLE = _mix(3)

class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.buf = _build_layer_multipliers(config.v, 3, 0, config.seed)

    def forward(self, x):
        return rotate_half(used_in_forward(x))

class Other(nn.Module):
    def _init_weights(self, module):
        module.buf = _build_layer_multipliers(1, 2, 3, 4)
'''


def test_init_only_helpers_detected():
    from unsloth_zoo.compiler import function_only_called_at_init
    assert function_only_called_at_init(SOURCE, "_build_layer_multipliers")
    # _mix is also called at module import time (TABLE), so it is not init-only.
    assert not function_only_called_at_init(SOURCE, "_mix")
    assert not function_only_called_at_init(SOURCE, "rotate_half")
    assert not function_only_called_at_init(SOURCE, "used_in_forward")
    assert not function_only_called_at_init(SOURCE, "missing_function")
    assert not function_only_called_at_init("not python (", "_mix")


def test_transitive_init_only_helper():
    from unsloth_zoo.compiler import function_only_called_at_init
    src = SOURCE.replace("TABLE = _mix(3)\n", "")
    assert function_only_called_at_init(src, "_mix")


def test_qwen4_exp_ngram_helpers_are_init_only():
    try:
        modeling = importlib.import_module("transformers.models.qwen4_exp.modeling_qwen4_exp")
    except Exception:
        pytest.skip("transformers without qwen4_exp")
    from unsloth_zoo.compiler import function_only_called_at_init
    source = inspect.getsource(modeling)
    for name in ("_build_layer_multipliers", "_find_nth_prime_after", "_is_prime", "_splitmix64"):
        assert function_only_called_at_init(source, name), name
    for name in ("rotate_half", "apply_rotary_pos_emb", "l2norm", "repeat_kv"):
        assert not function_only_called_at_init(source, name), name
