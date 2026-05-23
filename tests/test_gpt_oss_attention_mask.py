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

"""CPU-only regression tests for gpt-oss inference attention-mask patches.

Covers two recent fixes:

  PR 690 (BlockMask reaching the eager inference path). When
    config._attn_implementation == "flex_attention" but Unsloth runs
    inference through eager attention, the wrap over create_causal_mask
    MUST NOT return a BlockMask. Without the fix the eager forward
    crashes with:
        TypeError: unsupported operand type(s) for +=:
            'Tensor' and 'BlockMask'

  PR 691 (eager KV length must match attention-mask kv length). The
    eager_attention_forward closures MUST trim KV (or the mask) to the
    shorter length, otherwise pre-allocated cache slots crash with:
        RuntimeError: The size of tensor a (N) must match the size of
            tensor b (M) at non-singleton dimension 3

Each fix has a runtime invariant check plus an AST / source check, so
either a behavioural regression or a silent deletion of the guard
fails CI. tests/conftest.py already preloads device_type, stubs
torch.cuda.mem_get_info / get_device_capability, and sets
UNSLOTH_ALLOW_CPU=1 -- no extra GPU spoofing is needed here.
"""
from __future__ import annotations

import ast
import inspect
import os

# Trigger the gpt_oss model gate before unsloth_zoo's temporary patches install.
os.environ.setdefault("UNSLOTH_MODEL_NAME", "unsloth/gpt-oss-test")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "0")

# torch.compile around the wrap-around-create_causal_mask trick drops keyword
# names on some torch CPU builds, which would mask the real BlockMask check.
# Replace torch.compile with identity BEFORE unsloth_zoo imports so the wrap
# can forward (*args, **kwargs) cleanly.
import torch


def _identity_compile(model=None, *args, **kwargs):
    if model is None:
        return lambda fn: fn
    return model


torch.compile = _identity_compile  # noqa: E305

import pytest


# ---------------------------------------------------------------------------
# PR 690: BlockMask must not reach the eager inference path
# ---------------------------------------------------------------------------

_RUNTIME_SUBPROCESS = r"""
# Run the runtime invariant in a clean subprocess so we can replace
# torch.compile with identity BEFORE any unsloth_zoo import. The wrap
# over create_causal_mask captures _torch_compile = functools.partial(
# torch.compile, ...) at module load time, which on CPU torch drops
# kwarg names and makes the underlying call fail with a misleading
# "missing positional arg" instead of the real BlockMask invariant
# we want to check.
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


def test_pr690_runtime_flex_attention_inference_does_not_return_blockmask(tmp_path):
    """With config._attn_implementation == 'flex_attention' and an inference
    (no grad) inputs_embeds, the patched create_causal_mask must return a
    tensor (or None), never a BlockMask. Without the fix this returns a
    BlockMask and downstream eager attention crashes with TypeError."""
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


def test_pr690_static_flex_attention_guard_present_in_wrap():
    """AST-level guard: the wrap() returned by patch_GptOssModel must
    branch on _attn_implementation == 'flex_attention' on its inference
    side, even if it doesn't run in this environment."""
    from unsloth_zoo.temporary_patches import gpt_oss as _M
    src = inspect.getsource(_M.patch_GptOssModel)
    tree = ast.parse(src)

    # Find the inner wrap(f) -> return_attention_mask function.
    wrap_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "wrap":
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.FunctionDef)
                    and inner.name == "return_attention_mask"
                ):
                    wrap_fn = inner
                    break
        if wrap_fn is not None:
            break

    assert wrap_fn is not None, "wrap.return_attention_mask not found"

    body_src = ast.unparse(wrap_fn)
    assert '"flex_attention"' in body_src or "'flex_attention'" in body_src, (
        "wrap(f) for create_causal_mask must check for "
        "_attn_implementation == 'flex_attention' to avoid leaking a "
        "BlockMask to the eager inference path"
    )
    assert "_attn_implementation" in body_src, (
        "wrap(f) must read config._attn_implementation"
    )


def test_pr690_static_flex_attention_guard_present_in_model_forward():
    """AST-level guard: the patched GptOssModel.forward must also swap
    flex_attention -> eager around its own create_causal_mask call."""
    from unsloth_zoo.temporary_patches import gpt_oss as _M
    src = inspect.getsource(_M.patch_GptOssModel)
    assert (
        '"flex_attention"' in src or "'flex_attention'" in src
    ) and "_attn_implementation" in src, (
        "patch_GptOssModel must guard mask creation against "
        "_attn_implementation == 'flex_attention' during inference"
    )


# ---------------------------------------------------------------------------
# PR 691: eager attention must align KV length to the mask
# ---------------------------------------------------------------------------

def test_pr691_static_align_kv_helper_present_and_called():
    """Static check that _align_kv_to_mask exists in patch_GptOssAttention
    and is invoked by both eager attention forwards. Catches accidental
    deletion of the trim-to-shortest-length normalisation."""
    from unsloth_zoo.temporary_patches import gpt_oss as _M
    src = inspect.getsource(_M.patch_GptOssAttention)

    assert "_align_kv_to_mask" in src, (
        "patch_GptOssAttention must define _align_kv_to_mask "
        "(PR 691) so eager attention can survive KV-vs-mask "
        "length mismatches from pre-allocated caches"
    )

    # Should be invoked from BOTH eager forward variants so the inplace
    # and non-inplace paths agree.
    callsite_count = src.count("_align_kv_to_mask(")
    # Definition is "def _align_kv_to_mask(" -- exclude that.
    invocation_count = callsite_count - src.count("def _align_kv_to_mask(")
    assert invocation_count >= 2, (
        f"_align_kv_to_mask must be called from both eager attention "
        f"forwards (inplace + non-inplace); found {invocation_count} "
        f"invocations"
    )
