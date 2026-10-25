# SPDX-License-Identifier: AGPL-3.0-only
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
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""The default fused packed MXFP4 decode returns the activation dtype even with float32 expert biases.

transformers 5.x keeps MXFP4 expert biases in float32; the fused decode added them before its fixed-order
sum and returned that promoted dtype, so a bf16 eval forward without autocast handed float32 to the next
layer's bf16 projections ("mat1 and mat2 must have the same dtype").
"""

import json
import os
import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "the fused MXFP4 decode runs on CUDA only")

from unsloth_zoo.temporary_patches import gpt_oss  # noqa: E402
from test_mxfp4_packed_experts import _tiny_gpt_oss  # noqa: E402


def _float32_biases(experts):
    for name in ("gate_up_proj_bias", "down_proj_bias"):
        setattr(experts, name, torch.nn.Parameter(getattr(experts, name).float(), requires_grad = False))


@pytest.fixture
def fused(monkeypatch):
    monkeypatch.delenv("UNSLOTH_MXFP4_OGS", raising = False)
    monkeypatch.delenv("UNSLOTH_MXFP4_FUSED_GEMM", raising = False)
    model, experts = _tiny_gpt_oss("cuda")
    _float32_biases(experts)
    if not gpt_oss._mxfp4_fused_decode_enabled(experts.gate_up_proj, torch.bfloat16):
        pytest.skip("the fused packed MXFP4 decode is not available on this GPU")
    yield model, experts
    torch._dynamo.reset()


def _reference(mlp, experts, h):
    """The same experts dequantized into plain bf16 stacks with bf16 biases (the dense kernel needs one dtype)."""
    names = ("gate_up_proj", "down_proj", "gate_up_proj_bias", "down_proj_bias")
    saved = [getattr(experts, n) for n in names]
    for n, p in zip(names, saved):
        dense = p.dequantize() if n.endswith("proj") else p.to(torch.bfloat16)
        setattr(experts, n, torch.nn.Parameter(dense, requires_grad = False))
    try:
        with torch.autocast("cuda", enabled = False):
            return gpt_oss.moe_forward_inference_bf16(mlp, h).float()
    finally:
        for n, p in zip(names, saved):
            setattr(experts, n, p)


@pytest.mark.parametrize("autocast", [False, True])
def test_fused_decode_keeps_the_activation_dtype_with_float32_biases(fused, autocast):
    model, experts = fused
    mlp = model.model.layers[0].mlp
    g = torch.Generator(device = "cuda").manual_seed(13)
    for batch in (1, 4):
        h = torch.randn(batch, 1, 128, device = "cuda", dtype = torch.bfloat16, generator = g)
        with torch.no_grad(), torch.autocast("cuda", dtype = torch.bfloat16, enabled = autocast):
            assert gpt_oss._mxfp4_ogs_decode(mlp, h) is None  # auto: the fused route, not matmul_ogs
            torch.compiler.cudagraph_mark_step_begin()
            out = gpt_oss.moe_forward_inference_bf16(mlp, h).clone()  # a CUDA graph output, reused next replay
            want = _reference(mlp, experts, h)
        assert out.dtype == torch.bfloat16 and out.shape == h.shape
        assert torch.isfinite(out).all()
        assert (out.float() - want).norm() / want.norm() < 2e-2


_MODEL = r"""
import os, json
os.environ['UNSLOTH_MODEL_NAME'] = 'unsloth/gpt-oss-20b'
os.environ.pop('UNSLOTH_MXFP4_OGS', None)
os.environ.pop('UNSLOTH_MXFP4_FUSED_GEMM', None)
import torch
from transformers import GptOssConfig, GptOssForCausalLM
import transformers.models.gpt_oss.modeling_gpt_oss as M
from unsloth_zoo.temporary_patches import gpt_oss as G
from unsloth_zoo.mxfp4_dequant import Mxfp4ExpertParam
G.patch_GptOssModel()
if 'patch_GptOssModel' not in getattr(M.GptOssModel.forward, '__qualname__', ''):
    print('SKIP patched GptOssModel.forward not installed on this transformers'); raise SystemExit(0)
config = GptOssConfig(
    vocab_size=128, hidden_size=128, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
    num_key_value_heads=1, head_dim=64, num_local_experts=4, num_experts_per_tok=2,
    layer_types=['full_attention'] * 2, max_position_embeddings=128,
)
torch.manual_seed(0)
model = GptOssForCausalLM(config).to(device='cuda', dtype=torch.bfloat16).eval()
for layer in model.model.layers:
    e = layer.mlp.experts
    for name, n, k in (('gate_up_proj', 128, 128), ('down_proj', 128, 64)):
        blocks = torch.randint(0, 256, (4, n, k // 32, 16), device='cuda', dtype=torch.uint8)
        scales = torch.full((4, n, k // 32), 120, device='cuda', dtype=torch.uint8)
        setattr(e, name, Mxfp4ExpertParam(blocks, mxfp4_scales=scales))
        # transformers 5.x keeps MXFP4 expert biases in float32.
        setattr(e, name + '_bias', torch.nn.Parameter(getattr(e, name + '_bias').float(), requires_grad=False))
if not G._mxfp4_fused_decode_enabled(model.model.layers[0].mlp.experts.gate_up_proj, torch.bfloat16):
    print('SKIP the fused packed MXFP4 decode is not available on this GPU'); raise SystemExit(0)
ids = torch.tensor([[1]], device='cuda')
with torch.no_grad():  # no autocast
    out = model(input_ids=ids, use_cache=True)
print('RESULT', json.dumps({'dtype': str(out.logits.dtype), 'finite': bool(torch.isfinite(out.logits).all())}))
"""


def test_bf16_eval_decode_without_autocast_runs_on_packed_experts():
    pytest.importorskip("transformers").GptOssConfig
    proc = subprocess.run(
        [sys.executable, "-c", _MODEL], capture_output = True, text = True, timeout = 600,
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)},
    )
    line = next((l for l in proc.stdout.splitlines() if l.startswith(("RESULT", "SKIP"))), None)
    assert line is not None, proc.stdout[-2000:] + proc.stderr[-4000:]
    if line.startswith("SKIP"):
        pytest.skip(line[5:])
    result = json.loads(line[len("RESULT "):])
    assert result == {"dtype": "torch.bfloat16", "finite": True}
