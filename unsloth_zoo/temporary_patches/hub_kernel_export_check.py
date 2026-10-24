# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


"""Make transformers' kernel-hub fallback wrapper see real torch.export state under torch.compile.

`use_kernel_func_from_hub_with_fallback` runs its pure-torch reference whenever
`is_torchdynamo_exporting()` is True. On torch < 2.12 Dynamo constant-folds
`torch.compiler.is_exporting()` to True in every trace (pytorch 2.12 reads
`torch.compiler._is_exporting_flag` instead), so a compiled caller silently skips fla /
mamba_ssm / causal_conv1d and runs the slow reference (glm5_next and kimi_linear KDA).
Only the hub_kernels module global is swapped; modeling files keep their own reference.
"""

__all__ = [
    "patch_hub_kernel_export_check",
]

import torch

from .common import TEMPORARY_PATCHES, UNSLOTH_ENABLE_LOGGING, logger

_MARK = "_unsloth_real_export_check"


def _exporting_folded_in_compile():
    """True on torch versions whose Dynamo folds is_exporting() to True (fixed in 2.12)."""
    try:
        from packaging.version import Version
        return Version(torch.__version__).release[:2] < (2, 12)
    except Exception:
        return False


def _is_exporting():
    # torch.export sets this flag for its trace; Dynamo reads the global, as torch >= 2.12 does.
    return bool(torch.compiler._is_exporting_flag)


_is_exporting.__dict__[_MARK] = True


def patch_hub_kernel_export_check():
    if not _exporting_folded_in_compile():
        return False
    if not isinstance(getattr(torch.compiler, "_is_exporting_flag", None), bool):
        return False
    try:
        import transformers.integrations.hub_kernels as hub_kernels
    except Exception:
        return False
    current = getattr(hub_kernels, "is_torchdynamo_exporting", None)
    if current is None or not hasattr(hub_kernels, "use_kernel_func_from_hub_with_fallback"):
        return False
    if getattr(current, _MARK, False):
        return True
    hub_kernels.is_torchdynamo_exporting = _is_exporting
    if UNSLOTH_ENABLE_LOGGING:
        logger.info(
            "Unsloth: kernel-hub fallback wrappers now read torch.export state directly, so "
            "torch.compile keeps their fla / mamba kernels on this torch version."
        )
    return True


TEMPORARY_PATCHES.append(patch_hub_kernel_export_check)

try:
    patch_hub_kernel_export_check()
except Exception as _e:
    if UNSLOTH_ENABLE_LOGGING:
        logger.warning(f"Unsloth: could not patch the kernel-hub export check: {_e}")
