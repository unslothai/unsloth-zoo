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

"""Layer input op registers on torch >= 2.4; each module copy owns its own op."""

import importlib.util
import sys

import torch

import unsloth_zoo.routing_replay as rr


def test_layer_inputs_op_registers_and_records():
    op = rr._layer_inputs_op()  # torch < 2.7: ValueError from infer_schema on list[torch.Tensor]
    x = torch.randn(3)
    op(5, [x])
    assert rr._STATE.layer_inputs[5] == frozenset({x.data_ptr()})
    rr._STATE.layer_inputs.pop(5, None)


def test_second_module_copy_gets_its_own_op():
    first = rr._layer_inputs_op()
    spec = importlib.util.spec_from_file_location("_rr_second_copy", rr.__file__)
    copy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = copy
    try:
        spec.loader.exec_module(copy)
        second = copy._layer_inputs_op()
        assert str(second) != str(first)
        x = torch.randn(2)
        first(7, [x])
        assert rr._STATE.layer_inputs.pop(7) == frozenset({x.data_ptr()})
        assert 7 not in copy._STATE.layer_inputs
    finally:
        sys.modules.pop(spec.name, None)
