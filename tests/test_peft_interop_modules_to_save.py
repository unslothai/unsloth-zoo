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

"""group_peft_lora_pairs credits modules_to_save the way peft matches them: at component boundaries
((^|.*\\.)name($|\\..*)), never on a leaf that merely ends with the name. numpy-only; no mlx or torch needed."""

import numpy as np

from unsloth_zoo.mlx.peft_interop import group_peft_lora_pairs

_PREFIX = "base_model.model."


def _lora_pair(path = "model.layers.0.q_proj"):
    return {
        f"{_PREFIX}{path}.lora_A.weight": np.zeros((2, 4), dtype = np.float32),
        f"{_PREFIX}{path}.lora_B.weight": np.zeros((4, 2), dtype = np.float32),
    }


def test_peft_m2s_component_boundary_not_a_suffix_substring():
    """Declares `lm_head` but ships only `model.xlm_head` tensors: a different module whose leaf ends with the
    declared name. The loose endswith check credited it anyway, so the inconsistent adapter was accepted."""
    tensors = {
        f"{_PREFIX}model.xlm_head.modules_to_save.default.weight": np.zeros((4, 4), dtype = np.float32),
        **_lora_pair(),
    }
    _, _, rejected = group_peft_lora_pairs(
        tensors, {"modules_to_save": ["lm_head"], "use_dora": False},
    )
    assert "modules_to_save:lm_head" in rejected, (
        f"declared lm_head has no tensors but was credited by xlm_head: {rejected}"
    )


def test_peft_m2s_still_credits_the_declared_module():
    """The boundary fix must not drop real coverage: `model.lm_head` is `lm_head` at a boundary."""
    tensors = {
        f"{_PREFIX}model.lm_head.modules_to_save.default.weight": np.zeros((4, 4), dtype = np.float32),
        **_lora_pair(),
    }
    _, full_state, rejected = group_peft_lora_pairs(
        tensors, {"modules_to_save": ["lm_head"], "use_dora": False},
    )
    assert "modules_to_save:lm_head" not in rejected
    assert any(path.endswith("lm_head") for path in full_state)
