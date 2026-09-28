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

"""Repair remote NemotronH NemotronHMamba2Mixer.torch_forward (the no-mamba_ssm path): the inter-chunk sum must
reduce the source chunk axis (dim=3; decay_chunk is [b, h, target, source], not transposed), and dt must clamp to
time_step_limit like the remote fused path, not floor at time_step_min. Remote classes load via
transformers.dynamic_module_utils.get_class_in_module, so that is hooked. UNSLOTH_REMOTE_MAMBA2_FIX=0 disables.
"""

import functools
import inspect
import os
import re
import sys
import textwrap

from .common import TEMPORARY_PATCHES

__all__ = [
    "repair_remote_mamba2_torch_forward",
    "repair_remote_mamba2_modules",
]

_MAMBA2_BAD_SUM = "states_permuted[:, :, None, ...]).sum(dim=2)"
_MAMBA2_BAD_CLAMP = re.compile(r"torch\.clamp\(\s*dt\s*,\s*self\.time_step_min\s*\)")
_MAMBA2_GOOD_CLAMP = (
    "(torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1]) "
    "if getattr(self, 'time_step_limit', None) is not None else torch.clamp(dt, self.time_step_min))"
)


def repair_remote_mamba2_torch_forward(cls):
    if not isinstance(cls, type) or os.environ.get("UNSLOTH_REMOTE_MAMBA2_FIX", "1") == "0":
        return False
    fn = cls.__dict__.get("torch_forward", None)
    if fn is None or getattr(fn, "_unsloth_mamba2_fixed", False):
        return False
    try:
        source = inspect.getsource(fn)
    except Exception:
        return False
    if _MAMBA2_BAD_SUM not in source or "decay_chunk = torch.exp(segment_sum(" not in source \
            or "segment_sum(nn.functional.pad(A_cumsum[:, :, :, -1], (1, 0)))).transpose" in source:
        return False
    new_source = source.replace(_MAMBA2_BAD_SUM, "states_permuted[:, :, None, ...]).sum(dim=3)")
    new_source = _MAMBA2_BAD_CLAMP.sub(_MAMBA2_GOOD_CLAMP, new_source)
    module = sys.modules.get(cls.__module__, None)
    if module is None:
        return False
    # Module globals: the rewrite sees later rebinds like the original.
    local_namespace = {}
    exec(compile(textwrap.dedent(new_source), f"<unsloth remote mamba2 {cls.__module__}>", "exec"),
         module.__dict__, local_namespace)
    fixed = local_namespace["torch_forward"]
    fixed.__qualname__ = fn.__qualname__
    fixed.__module__ = fn.__module__
    fixed._unsloth_mamba2_fixed = True
    fixed._unsloth_original = fn
    cls.torch_forward = fixed
    return True
pass


_REPAIRED_MODULES = set()


def repair_remote_mamba2_modules():
    repaired = []
    for name, module in list(sys.modules.items()):
        if module is None or not name.startswith("transformers_modules"):
            continue
        # A changed remote file is re-executed into the same module object with a new hash.
        key = (name, id(module), getattr(module, "__transformers_module_hash__", None))
        if key in _REPAIRED_MODULES:
            continue
        _REPAIRED_MODULES.add(key)
        for value in list(vars(module).values()):
            if not isinstance(value, type) or getattr(value, "__module__", None) != name:
                continue
            try:
                if repair_remote_mamba2_torch_forward(value):
                    repaired.append(f"{value.__name__}.torch_forward")
            except Exception:
                continue
    return repaired
pass


def patch_remote_mamba2_torch_forward():
    if os.environ.get("UNSLOTH_REMOTE_MAMBA2_FIX", "1") == "0":
        return
    try:
        import transformers.dynamic_module_utils as dynamic_module_utils
    except Exception:
        return
    original = getattr(dynamic_module_utils, "get_class_in_module", None)
    if original is None or getattr(original, "_unsloth_remote_mamba2", False):
        return

    @functools.wraps(original)
    def get_class_in_module(*args, **kwargs):
        cls = original(*args, **kwargs)
        try:
            repair_remote_mamba2_modules()
        except Exception:
            pass
        return cls

    get_class_in_module._unsloth_remote_mamba2 = True
    dynamic_module_utils.get_class_in_module = get_class_in_module
    try:
        repair_remote_mamba2_modules()
    except Exception:
        pass
pass
TEMPORARY_PATCHES.append(patch_remote_mamba2_torch_forward)
