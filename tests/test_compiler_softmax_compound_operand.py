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

"""higher_precision_softmax appends `.to(<first argument>.dtype)`, so compound first arguments (Falcon's `scores +
mask`, SLANet's `torch.stack(xs, dim=1)`) must be left as written."""

import ast

import pytest
import torch
import torch.nn.functional as F

from unsloth_zoo import compiler


FALCON_LINES = [
    "attention_scores = F.softmax(attention_scores + attention_mask, dim=-1, dtype=hidden_states.dtype)\n",
    "attention_probs = F.softmax(attention_logits + attention_mask, dim=-1, dtype=hidden_states.dtype)\n",
]

SLANET_LINE = (
    "structure_preds = F.softmax(torch.stack(structure_preds_list, dim=1), dim=-1, dtype=torch.float32).to(\n"
    "    structure_preds_list[0].dtype\n"
    ")\n"
)


@pytest.mark.parametrize("source", FALCON_LINES + [SLANET_LINE])
def test_compound_first_argument_is_left_as_written(source):
    assert compiler.higher_precision_softmax(source) == source


def test_falcon_line_still_runs_after_the_rewrite():
    source = FALCON_LINES[0]
    scores = torch.randn(1, 2, 3, 3, dtype = torch.bfloat16)
    namespace = {
        "F": F,
        "torch": torch,
        "attention_scores": scores,
        "attention_mask": torch.zeros(1, 1, 3, 3, dtype = torch.bfloat16),
        "hidden_states": torch.zeros(1, 3, 4, dtype = torch.bfloat16),
    }
    exec(compiler.higher_precision_softmax(source), namespace)
    assert namespace["attention_scores"].dtype == torch.bfloat16


def test_cut_call_does_not_become_a_syntax_error():
    ast.parse(compiler.higher_precision_softmax(SLANET_LINE))


@pytest.mark.parametrize(
    "variable",
    [
        "attn_weights",
        "self.layer_weights",
        "weights.flatten(2)",
        "router_logits.float()",
        "layer_gate.to(compute_device)",
        "hidden[0]",
    ],
)
def test_single_operand_is_still_upcast(variable):
    out = compiler.higher_precision_softmax(f"x = nn.functional.softmax({variable}, dim=-1)\n")
    assert out == f"x = nn.functional.softmax({variable}, dim=-1, dtype = torch.float32).to({variable}.dtype)\n"
