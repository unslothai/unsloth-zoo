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

It decides UNSLOTH_HIGH_PRECISION_LAYERNORM from markers like `self.weight.float()`
in the RMSNorm source. The slice it matches those against used to run to the end of
the class after the norm one, so a marker anywhere in that neighbour upcast the
layernorm weights of a model that never asked for it.
"""

import os

import pytest

from unsloth_zoo.compiler import higher_precision_layernorms

# Llama 4 shape: the norm multiplies in the input dtype, so this is a float16 norm.
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

# A neighbour that is not a norm at all, but does contain the float32 marker.
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
    # The control: move the same marker into the norm class and the decision must flip.
    upcasting_norm = FLOAT16_NORM.replace(
        "output = self._norm(x.float()).type_as(x)",
        "output = self._norm(x.float()).type_as(x)\n        scale = self.weight.float()",
    )
    higher_precision_layernorms(upcasting_norm + FLOAT32_MARKER_NEIGHBOUR + TRAILING_CLASS)

    assert precision_flag() == "1"


def test_a_norm_class_with_nothing_after_it_is_still_read(precision_flag):
    # find() returns -1 when the norm class ends the file, which used to slice off the last
    # character instead of reading to the end.
    upcasting_norm = FLOAT16_NORM.replace(
        "return output * self.weight", "return output * self.weight.float()"
    )
    higher_precision_layernorms(upcasting_norm + TRAILING_CLASS)

    assert precision_flag() == "1"
