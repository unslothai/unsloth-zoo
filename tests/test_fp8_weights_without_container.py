# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""A pre-quantized FP8 checkpoint whose expert stacks live on a module
transformers cannot give an FP8 container (stepfun-ai/Step-3.7-Flash-FP8's
`MoELinear.weight`) must be dequantized into that module's dtype at load,
while `FP8Linear` targets keep their packed weight and scale.
"""
import json
import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

transformers = pytest.importorskip("transformers")
pytest.importorskip("transformers.integrations.finegrained_fp8")

import unsloth_zoo  # noqa: F401  registers the temporary patches
from unsloth_zoo.temporary_patches.fp8_uncontained_weights import (
    patch_fp8_dequantize_weights_without_container,
    _make_op,
    _target_owns_scale,
)

from transformers.integrations.finegrained_fp8 import Fp8Dequantize, FP8Linear
from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer
from transformers.utils.quantization_config import FineGrainedFP8Config

E4M3 = torch.float8_e4m3fn
patch_fp8_dequantize_weights_without_container()


def _block_quantize(weight, block):
    """(rows, cols) or (E, rows, cols) fp32 -> e4m3 plus a per-block fp32 scale grid."""
    rows, cols = weight.shape[-2:]
    br, bc = block
    assert rows % br == 0 and cols % bc == 0
    lead = weight.shape[:-2]
    w = weight.reshape(*lead, rows // br, br, cols // bc, bc)
    amax = w.abs().amax(dim = (-3, -1), keepdim = True).clamp(min = 1e-12)
    scale = amax / 448.0
    q = (w / scale).to(E4M3)
    return q.reshape(weight.shape), scale.reshape(*lead, rows // br, cols // bc)


def _reference_dequant(q, scale):
    rows, cols = q.shape[-2:]
    sr, sc = scale.shape[-2:]
    lead = q.shape[:-2]
    w = q.to(torch.float32).reshape(*lead, sr, rows // sr, sc, cols // sc)
    s = scale.reshape(*lead, sr, 1, sc, 1)
    return (w * s).reshape(q.shape)


class _MoELinear(nn.Module):
    """Step-3.7's expert stack: a bare 3-D parameter indexed by expert id."""
    def __init__(self, num_experts, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, out_features, in_features))

    def forward(self, x, expert_id):
        return F.linear(x.float(), self.weight[expert_id].float())


def _quantizer(pre_quantized = True, dequantize = False):
    config = FineGrainedFP8Config(weight_block_size = (4, 4), dequantize = dequantize)
    q = FineGrainedFP8HfQuantizer(config)
    q.pre_quantized = pre_quantized
    return q


class _Holder(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = _MoELinear(3, 8, 8).to(torch.bfloat16)
        self.proj = FP8Linear(8, 8, block_size = (4, 4))
        self.norm = nn.LayerNorm(8)


def test_uncontained_target_is_dequantized_into_its_dtype():
    torch.manual_seed(0)
    model = _Holder()
    op = _make_op(Fp8Dequantize)(_quantizer())
    full = torch.randn(3, 8, 8)
    q, scale = _block_quantize(full, (4, 4))
    out = op.convert({"weight$": [q], "weight_scale_inv": [scale]}, full_layer_name = "experts.weight", model = model)
    assert list(out) == ["experts.weight"]
    got = out["experts.weight"]
    assert got.dtype == torch.bfloat16 and got.shape == (3, 8, 8)
    torch.testing.assert_close(got.float(), _reference_dequant(q, scale).to(torch.bfloat16).float())


def test_container_target_passes_weight_and_scale_through():
    model = _Holder()
    op = _make_op(Fp8Dequantize)(_quantizer())
    q, scale = _block_quantize(torch.randn(8, 8), (4, 4))
    out = op.convert({"weight$": [q], "weight_scale_inv": [scale]}, full_layer_name = "proj.weight", model = model)
    assert set(out) == {"proj.weight", "proj.weight_scale_inv"}
    assert out["proj.weight"] is q and out["proj.weight_scale_inv"] is scale


def test_container_without_a_checkpoint_scale_gets_ones():
    model = _Holder()
    op = _make_op(Fp8Dequantize)(_quantizer())
    w = torch.randn(8, 8, dtype = torch.bfloat16)
    out = op.convert({"weight$": [w]}, full_layer_name = "proj.weight", model = model)
    assert out["proj.weight"] is w
    ones = out["proj.weight_scale_inv"]
    assert ones.shape == model.proj.weight_scale_inv.shape and ones.dtype == torch.float32
    assert bool((ones == 1).all())


def test_weight_without_scale_on_a_plain_module_is_untouched():
    model = _Holder()
    op = _make_op(Fp8Dequantize)(_quantizer())
    w = torch.randn(8, dtype = torch.bfloat16)
    out = op.convert({"weight$": [w]}, full_layer_name = "norm.weight", model = model)
    assert list(out) == ["norm.weight"] and out["norm.weight"] is w


def test_target_owns_scale_predicate():
    model = _Holder()
    assert _target_owns_scale(model, "proj.weight")
    assert not _target_owns_scale(model, "experts.weight")
    assert not _target_owns_scale(model, "norm.weight")
    assert not _target_owns_scale(model, "missing.weight")
    assert not _target_owns_scale(None, "proj.weight")


@pytest.mark.parametrize(
    "pre_quantized, dequantize, appended",
    [(True, False, True), (True, True, False), (False, False, False)],
)
def test_fallback_converter_only_for_prequantized_packed_loads(pre_quantized, dequantize, appended):
    from transformers.core_model_loading import WeightConverter
    q = _quantizer(pre_quantized = pre_quantized, dequantize = dequantize)
    conversions = q.update_weight_conversions([])
    ours = [
        c for c in conversions
        if isinstance(c, WeightConverter)
        and any(type(op).__name__ == "Fp8DequantizeWithoutContainer" for op in c.operations)
    ]
    assert (len(ours) == 1) == appended
    if appended:
        assert ours[0] is conversions[-1]


# ---------------------------------------------------------------------------
# End to end through from_pretrained: a tiny remote-code shaped model.
# ---------------------------------------------------------------------------
from transformers import PretrainedConfig, PreTrainedModel


class _TinyConfig(PretrainedConfig):
    model_type = "unsloth_test_uncontained_fp8"

    def __init__(self, hidden = 16, num_experts = 3, **kwargs):
        self.hidden = hidden
        self.num_experts = num_experts
        super().__init__(**kwargs)


class _TinyModel(PreTrainedModel):
    config_class = _TinyConfig
    base_model_prefix = "core"
    _no_split_modules = []

    def __init__(self, config):
        super().__init__(config)
        h = config.hidden
        self.experts = _MoELinear(config.num_experts, h, h)
        self.proj = nn.Linear(h, h, bias = False)
        self.vis = nn.Linear(h, h, bias = False)
        self.post_init()

    def forward(self, x, expert_id = 0):
        y = self.experts(x, expert_id).to(x.dtype)
        return self.vis(self.proj(y))


def _write_fp8_checkpoint(reference, path, block):
    """Experts and proj quantized with block scales; vis left bf16 and left out
    of modules_to_not_convert, the way the Step-3.7 checkpoint forgets its
    vision tower."""
    from safetensors.torch import save_file
    sd = {}
    q, s = _block_quantize(reference.experts.weight.detach().float(), block)
    sd["experts.weight"] = q.contiguous()
    sd["experts.weight_scale_inv"] = s.contiguous()
    q, s = _block_quantize(reference.proj.weight.detach().float(), block)
    sd["proj.weight"] = q.contiguous()
    sd["proj.weight_scale_inv"] = s.contiguous()
    sd["vis.weight"] = reference.vis.weight.detach().to(torch.bfloat16).contiguous()
    os.makedirs(path, exist_ok = True)
    save_file(sd, os.path.join(path, "model.safetensors"), metadata = {"format": "pt"})
    config = reference.config.to_dict()
    config["quantization_config"] = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "weight_block_size": list(block),
        "modules_to_not_convert": [],
    }
    config["architectures"] = ["_TinyModel"]
    config["dtype"] = "bfloat16"
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "FP8 loading needs a CUDA device")
def test_from_pretrained_gives_the_dequantized_experts(tmp_path):
    torch.manual_seed(0)
    block = (8, 8)
    reference = _TinyModel(_TinyConfig(hidden = 16, num_experts = 3)).to(torch.bfloat16)
    with torch.no_grad():
        for p in reference.parameters():
            p.copy_(torch.randn_like(p, dtype = torch.float32).to(torch.bfloat16))
    ckpt = str(tmp_path / "fp8")
    _write_fp8_checkpoint(reference, ckpt, block)

    loaded = _TinyModel.from_pretrained(ckpt, dtype = torch.bfloat16, device_map = {"": 0})

    # Experts: bare parameter, so dequantized into bf16 with the scale folded in.
    expert_w = loaded.experts.weight
    assert expert_w.dtype == torch.bfloat16 and expert_w.shape == (3, 16, 16)
    q, s = _block_quantize(reference.experts.weight.detach().float(), block)
    torch.testing.assert_close(expert_w.detach().cpu().float(), _reference_dequant(q, s).to(torch.bfloat16).float())
    # Without the scale folded in the values are off by the per-block factor.
    assert not torch.allclose(expert_w.detach().cpu().float(), q.float().to(torch.bfloat16).float())

    # proj: an FP8Linear keeps its packed weight and the checkpoint's scale.
    assert isinstance(loaded.proj, FP8Linear)
    assert loaded.proj.weight.dtype == E4M3
    torch.testing.assert_close(loaded.proj.weight_scale_inv.detach().cpu().float(), _block_quantize(reference.proj.weight.detach().float(), block)[1])

    # vis: FP8Linear the checkpoint forgot to exclude; scale of ones, not garbage.
    assert isinstance(loaded.vis, FP8Linear)
    assert bool((loaded.vis.weight_scale_inv.detach().cpu() == 1).all())

    # The expert forward matches the reference up to e4m3 rounding. (FP8Linear's own
    # forward wants the `kernels` package, which is not what this test is about.)
    x = torch.randn(4, 16, dtype = torch.bfloat16)
    with torch.no_grad():
        want = reference.experts(x, expert_id = 1).float()
        got = loaded.experts(x.to("cuda:0"), expert_id = 1).float().cpu()
    torch.testing.assert_close(got, want, rtol = 0.15, atol = 0.15 * want.abs().max().item())
    # And the unscaled bytes would not: the block scales are far from one.
    with torch.no_grad():
        wrong = F.linear(x.float(), q[1].float())
    assert not torch.allclose(wrong, want, rtol = 0.15, atol = 0.15 * want.abs().max().item())
