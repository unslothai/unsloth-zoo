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

"""The GPT-OSS patches never lower the process-wide dynamo recompile limit (zoo sets 1024). CPU-only; runs in a
subprocess."""
from __future__ import annotations

import subprocess
import sys


_GPT_OSS_SUBPROCESS = r"""
import os, sys
os.environ['UNSLOTH_ALLOW_CPU'] = '1'
os.environ['UNSLOTH_MODEL_NAME'] = 'unsloth/gpt-oss-test'
os.environ['UNSLOTH_COMPILE_DISABLE'] = '0'

import torch
def _id_compile(model=None, *a, **k):
    if model is None:
        return lambda fn: fn
    return model
torch.compile = _id_compile

if not torch.cuda.is_available():
    import torch.cuda.memory as _cm
    _cm.mem_get_info = lambda *a, **k: (0, 80 * 1024**3)
    torch.cuda.device_count = lambda: 1
    torch.cuda.get_device_capability = lambda *a, **k: (8, 0)
    class _P:
        major = 8; minor = 0; total_memory = 80 * 1024**3
        multi_processor_count = 108; name = 'stub'
    torch.cuda.get_device_properties = lambda *a, **k: _P()
    torch.cuda.current_device = lambda: 0

import unsloth_zoo  # noqa
from unsloth_zoo.temporary_patches.gpt_oss import patch_GptOssAttention, patch_GptOssModel
torch._dynamo.config.cache_size_limit = 1024
patch_GptOssAttention()
patch_GptOssModel()
print('LIMIT', torch._dynamo.config.cache_size_limit)
torch._dynamo.config.cache_size_limit = 8
patch_GptOssAttention()
print('LIMIT', torch._dynamo.config.cache_size_limit)
"""


def test_gpt_oss_patches_never_lower_the_recompile_limit():
    proc = subprocess.run(
        [sys.executable, "-c", _GPT_OSS_SUBPROCESS], capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    limits = [line.split()[1] for line in proc.stdout.splitlines() if line.startswith("LIMIT ")]
    # A larger limit survives; a smaller one is still raised to the patch's 256.
    assert limits == ["1024", "256"], proc.stdout
