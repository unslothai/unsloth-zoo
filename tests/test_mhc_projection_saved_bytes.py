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

"""The eager mHC mixer saves its bf16 input for backward, not the FP32 upcast, bit for bit.

transformers' DeepSeek-V4 / GLM-5.3-Flash mixer runs `input_norm` and the `fn` projection in FP32
on `hidden_streams.flatten(2).float()`; plain autograd keeps that `[tokens, hc_mult * hidden]`
FP32 copy (twice the bf16 streams) per mixer site. `MODULE_FORWARD_EAGER_REWRITES` routes the
two lines through `unsloth_mhc_projection`, which reruns them in backward instead.
"""

import importlib
import inspect
import textwrap

import pytest
import torch
import torch.nn.functional as F

from unsloth_zoo.compiler import (
    MODULE_FORWARD_EAGER_REWRITES,
    module_forward_eager_rewrites,
    module_forward_source_rewrites,
)
from unsloth_zoo.temporary_patches.mhc_sinkhorn import (
    MHC_PROJECTION_SOURCE,
    MHC_SINKHORN_SOURCE,
    unsloth_mhc_projection,
)

# (model_type, mixer class, config kwargs for a small mixer)
CASES = [
    ("deepseek_v4", "DeepseekV4HyperConnection", "deepseek_v4"),
    ("glm5_next", "Glm5NextTextHyperConnection", "glm5_next_text"),
]


def _modeling(model_type):
    try:
        return importlib.import_module(f"transformers.models.{model_type}.modeling_{model_type}")
    except Exception:
        pytest.skip(f"transformers has no {model_type}")


def _mixer(model_type, cls_name, config_type):
    import transformers

    modeling = _modeling(model_type)
    cls = getattr(modeling, cls_name, None)
    if cls is None or MHC_PROJECTION_SOURCE not in inspect.getsource(cls.forward):
        pytest.skip(f"{cls_name} has a different forward in this transformers")
    config = transformers.AutoConfig.for_model(config_type, hidden_size = 64, hc_mult = 4, hc_sinkhorn_iters = 20)
    torch.manual_seed(0)
    mixer = cls(config)
    with torch.no_grad():
        for p in mixer.parameters():
            p.normal_(0, 0.05)
    namespace = dict(vars(modeling))
    source = module_forward_eager_rewrites(modeling, model_type)[cls_name]
    exec(compile(textwrap.dedent(source), f"<{cls_name}>", "exec"), namespace)
    return cls, mixer, namespace["forward"], config


def _run(forward, mixer, x0, grads):
    x = x0.clone().requires_grad_()
    for p in mixer.parameters():
        p.grad = None
    saved = {}

    def pack(t):
        saved[t.untyped_storage().data_ptr()] = (t.untyped_storage().nbytes(), t.dtype)
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        outs = forward(mixer, x)
    torch.autograd.backward([o.float() for o in outs], grads)
    fp32 = sum(n for n, dtype in saved.values() if dtype == torch.float32)
    params = {n: None if p.grad is None else p.grad.clone() for n, p in mixer.named_parameters()}
    return [o.detach() for o in outs], x.grad, params, fp32


@pytest.mark.parametrize("model_type, cls_name, config_type", CASES)
def test_registry_rewrite_applies_and_keeps_the_sinkhorn_rewrite(model_type, cls_name, config_type):
    modeling = _modeling(model_type)
    if cls_name not in MODULE_FORWARD_EAGER_REWRITES.get(model_type, {}):
        pytest.fail(f"{cls_name} missing from MODULE_FORWARD_EAGER_REWRITES")
    cls = getattr(modeling, cls_name, None)
    if cls is None or MHC_PROJECTION_SOURCE not in inspect.getsource(cls.forward):
        pytest.skip(f"{cls_name} has a different forward in this transformers")
    source = module_forward_eager_rewrites(modeling, model_type)[cls_name]
    assert "unsloth_mhc_projection(" in source and "self.fn.float()" not in source
    assert source.count(MHC_SINKHORN_SOURCE) == 1
    assert modeling.unsloth_mhc_projection is unsloth_mhc_projection
    compiled = module_forward_source_rewrites(modeling, model_type).get(cls_name)
    if compiled is not None:
        assert "unsloth_mhc_projection(" in compiled


@pytest.mark.parametrize("fn_trainable", [False, True])
@pytest.mark.parametrize("scale", [1e-3, 1.0, 30.0])
@pytest.mark.parametrize("model_type, cls_name, config_type", CASES)
def test_rewrite_is_bitwise_and_drops_the_fp32_copy(model_type, cls_name, config_type, scale, fn_trainable):
    cls, mixer, rewritten, config = _mixer(model_type, cls_name, config_type)
    mixer.fn.requires_grad_(fn_trainable)
    g = torch.Generator().manual_seed(1)
    tokens = 96
    x0 = (torch.randn(2, tokens, config.hc_mult, config.hidden_size, generator = g) * scale).bfloat16()
    with torch.no_grad():
        shapes = [o.shape for o in cls.forward(mixer, x0)]
    grads = [torch.randn(s, generator = g) for s in shapes]

    out0, gx0, gp0, fp32_0 = _run(cls.forward, mixer, x0, grads)
    out1, gx1, gp1, fp32_1 = _run(rewritten, mixer, x0, grads)

    assert all(torch.equal(a, b) for a, b in zip(out0, out1))
    assert torch.equal(gx0, gx1) and gx1.dtype == torch.bfloat16
    for name in gp0:
        assert (gp0[name] is None) == (gp1[name] is None), name
        if gp0[name] is not None:
            assert torch.equal(gp0[name], gp1[name]), name
    upcast = x0.numel() * 4
    assert fp32_0 - fp32_1 >= upcast, (fp32_0, fp32_1, upcast)


def test_helper_plain_paths_match():
    torch.manual_seed(0)
    x = torch.randn(3, 5, 4, 16).bfloat16()
    fn = torch.randn(24, 64) * 0.05

    def norm(t):
        return t * torch.rsqrt(t.float().square().mean(-1, keepdim = True) + 1e-6).to(t.dtype)

    ref = F.linear(norm(x.flatten(start_dim = 2).float()), fn.float())
    with torch.no_grad():
        assert torch.equal(unsloth_mhc_projection(x, norm, fn), ref)
    assert torch.equal(unsloth_mhc_projection(x.float(), norm, fn), ref)
    assert torch.equal(unsloth_mhc_projection(x, norm, fn), ref)  # nothing requires grad
