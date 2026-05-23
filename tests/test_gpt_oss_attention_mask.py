# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Runtime regression check for the gpt-oss BlockMask leak (PR #690).

The behavioural sibling tests (static AST guards over wrap and the
patched GptOssModel.forward, plus the PR #691 _align_kv_to_mask static
check, plus the inspect-driven mask-kwargs filter check) live in
tests/test_zoo_history_regressions.py so they auto-run under the
consolidated Tests CI workflow.

This file is the *runtime* invariant: actually call the wrapped
transformers.masking_utils.create_causal_mask with a tiny GptOssConfig
that has _attn_implementation="flex_attention" and assert the return is
not a BlockMask. The check runs in a clean subprocess so torch.compile
can be replaced with identity BEFORE any unsloth_zoo import -- the wrap
captures _torch_compile = functools.partial(torch.compile, ...) at
module load time and on CPU torch the compiled wrapper drops kwarg
names, which would otherwise mask the real BlockMask invariant with a
misleading "missing positional argument" error.

CPU-only; no GPU required. The repo's existing tests/conftest.py GPU-free
harness (device_type preload, mem_get_info / get_device_capability
stubs, UNSLOTH_ALLOW_CPU=1) handles the inner subprocess via env
inheritance.
"""
from __future__ import annotations

_RUNTIME_SUBPROCESS = r"""
import os, sys, traceback
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

try:
    import unsloth_zoo  # noqa
    from unsloth_zoo.temporary_patches.gpt_oss import (
        patch_GptOssAttention, patch_GptOssModel,
    )
    patch_GptOssAttention()
    patch_GptOssModel()

    import inspect
    import transformers.masking_utils as MU
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
    from torch.nn.attention.flex_attention import BlockMask

    config = GptOssConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        num_local_experts=2, num_experts_per_tok=1, vocab_size=1000,
        max_position_embeddings=64, sliding_window=32,
        layer_types=['full_attention', 'sliding_attention'],
    )
    config._attn_implementation = 'flex_attention'

    inputs_embeds = torch.zeros((1, 8, 64), dtype=torch.float32, requires_grad=False)
    cache_position = torch.arange(8)

    underlying = getattr(MU, '_old_create_causal_mask', MU.create_causal_mask)
    sig = inspect.signature(underlying)
    kwargs = {'config': config, 'attention_mask': None, 'past_key_values': None}
    if 'inputs_embeds' in sig.parameters: kwargs['inputs_embeds'] = inputs_embeds
    if 'input_embeds' in sig.parameters:  kwargs['input_embeds'] = inputs_embeds
    if 'cache_position' in sig.parameters: kwargs['cache_position'] = cache_position

    result = MU.create_causal_mask(**kwargs)

    if isinstance(result, BlockMask):
        print('RUNTIME_FAIL: got BlockMask (regression).')
        sys.exit(2)
    print('RUNTIME_OK', type(result).__name__)
    sys.exit(0)
except Exception:
    traceback.print_exc()
    sys.exit(3)
"""


def test_pr690_runtime_blockmask_does_not_reach_eager_inference_path():
    """Runtime invariant: with _attn_implementation='flex_attention' and
    a non-grad inputs_embeds, the patched create_causal_mask must return
    a tensor (or None), never a BlockMask. Without PR #690 this returns
    a BlockMask and gpt-oss generate() crashes with
        TypeError: unsupported operand type(s) for +=:
            'Tensor' and 'BlockMask'
    """
    import subprocess
    import sys as _sys

    proc = subprocess.run(
        [_sys.executable, "-c", _RUNTIME_SUBPROCESS],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, (
        "Runtime BlockMask invariant failed.\n"
        f"exit={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "RUNTIME_OK" in proc.stdout, (
        f"missing RUNTIME_OK token in stdout: {proc.stdout}"
    )
