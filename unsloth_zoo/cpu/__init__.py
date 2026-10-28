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

"""CPU training backend, used only when torch sees no accelerator or UNSLOTH_FORCE_CPU=1: stock transformers/PEFT/TRL
with CPU dtype policy, bitsandbytes CPU QLoRA and chunked CE; not imported on GPU hosts."""

from .runtime import (
    apply_cpu_security_patches,
    cpu_capabilities,
    cpu_bf16_supported,
    cpu_bnb_4bit_supported,
)
from .loader import FastCPUModel
from .trainer import install_cpu_training_defaults

apply_cpu_security_patches()
