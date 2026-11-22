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

"""Inkling 2 block attention residuals stay eager: compiled on their own they give wrong input grads on CUDA."""

import importlib

import pytest

from unsloth_zoo.compiler import DISABLE_COMPILE_MODULES

_CLASSES = {
    "inkling2": "Inkling2BlockAttentionResidual",
    "inkling2_loop": "Inkling2LoopBlockAttentionResidual",
}


@pytest.mark.parametrize("model_type", sorted(_CLASSES))
def test_block_attention_residual_is_not_compiled(model_type):
    name = _CLASSES[model_type]
    assert name in DISABLE_COMPILE_MODULES
    # The compiler matches by class-name suffix: the name must not also cover the message residual, which compiles fine.
    assert not any("MessageAttentionResidual".endswith(x) for x in DISABLE_COMPILE_MODULES)
    try:
        modeling = importlib.import_module(f"transformers.models.{model_type}.modeling_{model_type}")
    except ImportError:
        pytest.skip(f"transformers without {model_type}")
    assert hasattr(modeling, name), f"{name} renamed upstream: update DISABLE_COMPILE_MODULES"
