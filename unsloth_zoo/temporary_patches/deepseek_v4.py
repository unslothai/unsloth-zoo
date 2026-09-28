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

"""transformers keeps DeepSeek-V4 norm weights in float32 and returns `weight * x.to(input_dtype)`, so a bf16
model's next Linear gets float32 input. DeepSeek's reference (`inference/model.py`) returns `(weight * x).to(dtype)`."""
import inspect

import torch

from .common import TEMPORARY_PATCHES, UNSLOTH_ENABLE_LOGGING
from .utils import logger

__all__ = ["patch_deepseek_v4_rmsnorm_dtype"]

_BUGGY_RETURN = "return self.weight * hidden_states.to(input_dtype)"


def patch_deepseek_v4_rmsnorm_dtype():
    try:
        from transformers.models.deepseek_v4 import modeling_deepseek_v4
    except Exception:
        return
    cls = getattr(modeling_deepseek_v4, "DeepseekV4RMSNorm", None)
    if cls is None or getattr(cls.forward, "_unsloth_patched", False):
        return
    try:
        source = inspect.getsource(cls.forward)
    except Exception:
        return
    if _BUGGY_RETURN not in source:
        return

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)

    forward._unsloth_patched = True
    cls.forward = forward
    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: Patched DeepseekV4RMSNorm to return the input dtype like DeepSeek's reference.")
pass

TEMPORARY_PATCHES.append(patch_deepseek_v4_rmsnorm_dtype)
