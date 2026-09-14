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

"""higher_precision_layernorms must read only the norm class it found.

It picks UNSLOTH_HIGH_PRECISION_LAYERNORM from markers like `self.weight.float()` in the
RMSNorm source. That slice used to run one class too far, so a neighbour's markers decided.
"""

import os

import pytest

# Before the import: CI installs unsloth with `|| true`, and zoo raises without this flag.
#
# Put back afterwards, which the plain setdefault did not do. This runs at COLLECTION, and
# pytest imports every selected module before running any test, so leaving it set handed the
# flag to the whole xdist worker: `__init__.py:179` reads it into _SKIP_GPU_INIT, which skips
# the import-time allocator block and pins DEVICE_TYPE to "cpu". The suite then reported
# test_alloc_conf_platform_matrix.py as "the product wrote no allocator vars" and
# test_temporary_patches_imports.py as "device_memory is 0", neither of which was true of the
# product. Only the import needs the flag, and _SKIP_GPU_INIT is read once at zoo import.
_GPU_INIT_GATE = "UNSLOTH_ZOO_DISABLE_GPU_INIT"
_previous_gate = os.environ.get(_GPU_INIT_GATE)
try:
    os.environ.setdefault(_GPU_INIT_GATE, "1")
    from unsloth_zoo.compiler import higher_precision_layernorms  # noqa: E402
finally:
    if _previous_gate is None:
        os.environ.pop(_GPU_INIT_GATE, None)
    else:
        os.environ[_GPU_INIT_GATE] = _previous_gate

# Llama 4 shape: the weight multiplies in the input dtype.
FLOAT16_NORM = """
class Llama4TextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
"""

FLOAT32_MARKER_NEIGHBOUR = """

class Llama4TextExperts(nn.Module):
    def forward(self, x):
        gate = self.weight.float()
        return gate * x
"""

TRAILING_CLASS = """

class Llama4TextAttention(nn.Module):
    def forward(self, x):
        return x
"""


@pytest.fixture
def precision_flag(monkeypatch):
    monkeypatch.setitem(os.environ, "UNSLOTH_HIGH_PRECISION_LAYERNORM", "0")
    return lambda: os.environ["UNSLOTH_HIGH_PRECISION_LAYERNORM"]


def test_marker_in_the_next_class_does_not_upcast(precision_flag):
    higher_precision_layernorms(FLOAT16_NORM + FLOAT32_MARKER_NEIGHBOUR + TRAILING_CLASS)

    assert precision_flag() == "0"


def test_marker_in_the_norm_itself_still_upcasts(precision_flag):
    upcasting_norm = FLOAT16_NORM.replace(
        "output = self._norm(x.float()).type_as(x)",
        "output = self._norm(x.float()).type_as(x)\n        scale = self.weight.float()",
    )
    higher_precision_layernorms(upcasting_norm + FLOAT32_MARKER_NEIGHBOUR + TRAILING_CLASS)

    assert precision_flag() == "1"


def test_a_marker_in_the_next_class_can_also_suppress_a_real_upcast(precision_flag):
    # The ladder is priority ordered, so a float16 marker next door hides a real float32 norm.
    # Qwen4Exp is the live case, and it was silently losing its upcast.
    float32_norm = FLOAT16_NORM.replace(
        "return output * self.weight", "return output * (1.0 + self.weight.float())"
    )
    float16_marker_neighbour = """

class Llama4TextRMSNormGated(nn.Module):
    def forward(self, hidden_states, gate):
        return self.weight * hidden_states.to(input_dtype)
"""
    higher_precision_layernorms(float32_norm + float16_marker_neighbour + TRAILING_CLASS)

    assert precision_flag() == "1"


def test_decorated_next_class_does_not_leak_either(precision_flag):
    # Stop at the next class's "\nclass", not its decorator. Kimi Linear and Cohere2 MoE.
    decorated_neighbour = '''

@use_kernel_forward_from_hub("RMSNormGated")
class Llama4TextRMSNormGated(nn.Module):
    def forward(self, hidden_states):
        return self.weight.to(torch.float32) * hidden_states
'''
    higher_precision_layernorms(FLOAT16_NORM + decorated_neighbour + TRAILING_CLASS)

    assert precision_flag() == "0"


def test_the_norm_class_is_read_in_full(precision_flag):
    norm_with_trailing_method = FLOAT16_NORM + """
    def extra_repr(self):
        scale = self.weight.float()
        return f"{tuple(self.weight.shape)}"
"""
    higher_precision_layernorms(norm_with_trailing_method + TRAILING_CLASS)

    assert precision_flag() == "1"


def test_the_flag_is_never_downgraded(precision_flag, monkeypatch):
    # loader.py hardcodes "1" for Gemma 3/3n/4 and Granite-4 before this runs; never undo it.
    monkeypatch.setitem(os.environ, "UNSLOTH_HIGH_PRECISION_LAYERNORM", "1")
    higher_precision_layernorms(FLOAT16_NORM + FLOAT32_MARKER_NEIGHBOUR + TRAILING_CLASS)

    assert precision_flag() == "1"


@pytest.mark.parametrize(
    "model, expected",
    [
        # Real sources the old slice got wrong: a neighbour leaked its marker in.
        ("cohere2_moe", "0"),  # Cohere2MoeLayerNorm
        ("kimi_linear", "0"),  # KimiLinearRMSNormGated
        ("qwen4_exp", "1"),  # Qwen4ExpTextRMSNormGated masked a real float32 norm
        # Controls it already got right.
        ("llama4", "0"),
        ("llama", "0"),
        ("gemma3", "1"),
        ("olmo2", "1"),  # (self.weight * hidden_states).to(...): weight used in float32
    ],
)
def test_real_transformers_sources(precision_flag, model, expected):
    import importlib
    import inspect

    try:
        module = importlib.import_module(f"transformers.models.{model}.modeling_{model}")
    except ImportError:
        pytest.skip(f"transformers has no {model}")

    higher_precision_layernorms(inspect.getsource(module))

    assert precision_flag() == expected
