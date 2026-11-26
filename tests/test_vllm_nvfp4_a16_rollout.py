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

"""fast_inference on compressed-tensors NVFP4: W4A16 vLLM config and shared compressed-tensors tensors."""

import copy
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from unsloth_zoo import vllm_utils as vu


def _nvfp4_args(**extra):
    args = dict(num_bits = 4, type = "float", symmetric = True, group_size = 16, strategy = "tensor_group",
                dynamic = False, actorder = None, observer = "minmax")
    args.update(extra)
    return args


def _ct_config(input_activations = "nvfp4", weights = None):
    acts = _nvfp4_args(dynamic = "local") if input_activations == "nvfp4" else input_activations
    quant = {
        "quant_method": "compressed-tensors",
        "format": "nvfp4-pack-quantized",
        "ignore": ["lm_head"],
        "config_groups": {"group_0": {
            "format": "nvfp4-pack-quantized", "targets": ["Linear"],
            "weights": weights if weights is not None else _nvfp4_args(),
            "input_activations": acts, "output_activations": None,
        }},
    }
    return SimpleNamespace(quantization_config = quant)


FP8_WEIGHTS = dict(num_bits = 8, type = "float", symmetric = True, group_size = None, strategy = "channel", dynamic = False)


def test_w4a4_checkpoint_gets_a16_config_without_mutating_it(monkeypatch):
    monkeypatch.delenv("UNSLOTH_VLLM_NVFP4_A16", raising = False)
    config = _ct_config()
    before = copy.deepcopy(config.quantization_config)
    quant = vu._vllm_nvfp4_a16_quantization_config(config)
    assert quant["config_groups"]["group_0"]["input_activations"] is None
    assert quant["config_groups"]["group_0"]["weights"] == before["config_groups"]["group_0"]["weights"]
    assert quant["ignore"] == ["lm_head"] and quant["quant_method"] == "compressed-tensors"
    assert config.quantization_config == before


def test_kill_switch_keeps_the_checkpoint_activations(monkeypatch):
    monkeypatch.setenv("UNSLOTH_VLLM_NVFP4_A16", "0")
    assert vu._vllm_nvfp4_a16_quantization_config(_ct_config()) is None


@pytest.mark.parametrize("config", [
    _ct_config(input_activations = None),  # already NVFP4A16
    _ct_config(input_activations = dict(num_bits = 8, type = "float", strategy = "token", dynamic = True)),
    _ct_config(weights = FP8_WEIGHTS, input_activations = None),
    SimpleNamespace(quantization_config = {"quant_method": "fp8", "weight_block_size": [128, 128]}),
    SimpleNamespace(quantization_config = None),
    SimpleNamespace(),
])
def test_nothing_to_rewrite(monkeypatch, config):
    monkeypatch.delenv("UNSLOTH_VLLM_NVFP4_A16", raising = False)
    assert vu._vllm_nvfp4_a16_quantization_config(config) is None


def test_nvfp4_group_detection():
    assert list(vu._ct_nvfp4_groups(_ct_config())) == ["group_0"]
    assert vu._ct_nvfp4_groups(_ct_config(weights = FP8_WEIGHTS)) == {}
    # A QuantizationConfig object (not a dict) is read through to_dict().
    obj = SimpleNamespace(quantization_config = SimpleNamespace(to_dict = lambda: _ct_config().quantization_config))
    assert list(vu._ct_nvfp4_groups(obj)) == ["group_0"]
    assert not vu._is_ct_nvfp4_args(_nvfp4_args(group_size = 32))
    assert not vu._is_ct_nvfp4_args(_nvfp4_args(symmetric = False))


def _fused_proj(sizes, k = 32, bias = False):
    n = sum(sizes)
    packed = torch.randint(0, 255, (n, k // 2), dtype = torch.uint8)
    scale = torch.randn(n, k // 16).to(torch.float8_e4m3fn)
    glob = torch.tensor([100.0 + i for i in range(len(sizes))])
    proj = SimpleNamespace(_unsloth_nvfp4_ct = (packed, scale, glob),
                           output_sizes = list(sizes) if len(sizes) > 1 else None,
                           bias = torch.randn(n) if bias else None)
    return proj


def test_fused_slices_are_views_of_the_kept_tensors():
    proj = _fused_proj([8, 4, 4], bias = True)
    packed, scale, glob = proj._unsloth_nvfp4_ct
    for kk, (lo, hi) in enumerate([(0, 8), (8, 12), (12, 16)]):
        sd, qsd = {}, {}
        vu._nvfp4_ct_state_dict(proj, "p", kk, True, sd, qsd)
        assert sd.keys() == qsd.keys() == {"p.weight_packed", "p.weight_scale", "p.weight_global_scale", "p.bias"}
        assert "p.weight" not in sd
        assert torch.equal(sd["p.weight_packed"], packed[lo:hi])
        assert sd["p.weight_packed"].data_ptr() == packed[lo:hi].data_ptr()
        assert sd["p.weight_packed"].untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()
        assert sd["p.weight_scale"].untyped_storage().data_ptr() == scale.untyped_storage().data_ptr()
        assert sd["p.weight_global_scale"].tolist() == [100.0 + kk]
        assert torch.equal(sd["p.bias"], proj.bias[lo:hi])


def test_unfused_layer_takes_everything():
    proj = _fused_proj([12])
    sd = {}
    vu._nvfp4_ct_state_dict(proj, "o", 0, False, sd, {})
    assert sd["o.weight_packed"].shape == (12, 16) and sd["o.weight_global_scale"].tolist() == [100.0]


def test_rebuilt_layer_is_what_compressed_tensors_leaves():
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.layers[0].q_proj = torch.nn.Linear(2, 2)
    proj = _fused_proj([12], bias = True)
    qsd = {}
    vu._nvfp4_ct_state_dict(proj, "layers.0.q_proj", 0, False, qsd, {})
    scheme = vu._nvfp4_ct_scheme(_ct_config())
    vu._set_nvfp4_ct_linear(model, "layers.0.q_proj", qsd, scheme)
    layer = model.layers[0].q_proj
    assert type(layer) is torch.nn.Linear and "weight" not in layer._parameters
    assert (layer.in_features, layer.out_features) == (32, 12)
    for name in ("weight_packed", "weight_scale", "weight_global_scale"):
        assert getattr(layer, name).data_ptr() == qsd[f"layers.0.q_proj.{name}"].data_ptr()
        assert not getattr(layer, name).requires_grad
    assert torch.equal(layer.bias, proj.bias)
    assert str(getattr(layer.quantization_status, "value", "")) == "compressed"
    weights = layer.quantization_scheme.weights
    assert getattr(weights, "num_bits") == 4 and getattr(weights, "group_size") == 16


def test_load_hook_keeps_copies_only_while_flagged(monkeypatch):
    name = "vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4"

    class CompressedTensorsW4A4Fp4:
        def __init__(self, use_a16 = False):
            self.use_a16 = use_a16

        def create_weights(self, layer, output_partition_sizes, input_size_per_partition, params_dtype,
                           weight_loader, **kwargs):
            layer.weight_packed = torch.full((4, 8), 7, dtype = torch.uint8)
            layer.weight_scale = torch.ones(4, 1)
            layer.weight_global_scale = torch.tensor([4.0, 2.0])
            if not self.use_a16:
                layer.input_global_scale = torch.ones(len(output_partition_sizes))

        def process_weights_after_loading(self, layer):
            # Like vLLM: invert the global scale and repack the weight in place.
            layer.weight_global_scale.data.reciprocal_()
            layer.weight_packed.data.zero_()

    class PerTensorScaleParameter(torch.nn.Parameter):
        def __new__(cls, data, weight_loader):
            return super().__new__(cls, data, requires_grad = False)

    class Layer(torch.nn.Module):
        pass

    monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(CompressedTensorsW4A4Fp4 = CompressedTensorsW4A4Fp4))
    monkeypatch.setitem(sys.modules, "vllm.model_executor.parameter",
                        types.SimpleNamespace(PerTensorScaleParameter = PerTensorScaleParameter))
    assert vu._patch_vllm_nvfp4_keep_compressed_tensors_weights()
    assert vu._patch_vllm_nvfp4_keep_compressed_tensors_weights()  # idempotent
    assert CompressedTensorsW4A4Fp4.process_weights_after_loading._unsloth_keeps_ct_weights

    def load(scheme):
        layer = Layer()
        scheme.create_weights(layer = layer, output_partition_sizes = [2, 2], input_size_per_partition = 16,
                              params_dtype = torch.bfloat16, weight_loader = None)
        registered = "input_global_scale" in layer._parameters
        scheme.process_weights_after_loading(layer)
        return layer, registered

    plain, registered = load(CompressedTensorsW4A4Fp4(use_a16 = True))
    assert not hasattr(plain, "_unsloth_nvfp4_ct") and not registered

    monkeypatch.setattr(vu, "_UNSLOTH_KEEP_NVFP4_CT_WEIGHTS", True)
    monkeypatch.setattr(vu, "_UNSLOTH_NVFP4_DROP_INPUT_SCALES", True)
    kept, registered = load(CompressedTensorsW4A4Fp4(use_a16 = True))
    # The checkpoint's input_global_scale had a place to load into, then it is gone.
    assert registered and not hasattr(kept, "input_global_scale")
    packed, scale, glob = kept._unsloth_nvfp4_ct
    assert torch.all(packed == 7) and glob.tolist() == [4.0, 2.0]  # the pre-conversion values
    assert torch.all(kept.weight_packed == 0)
    # W4A4 (kill switch) keeps its own input scales untouched.
    w4a4, registered = load(CompressedTensorsW4A4Fp4(use_a16 = False))
    assert not registered and torch.equal(w4a4.input_global_scale, torch.ones(2))


def test_return_args_skips_the_in_process_nvfp4_rewrite():
    # return_args hands the args to a separate `vllm serve`, where the input_global_scale hook is not installed:
    # the A16 rewrite and its load flags must only apply to engines built in this process.
    import inspect
    from unsloth_zoo import vllm_utils
    source = inspect.getsource(vllm_utils.load_vllm)
    assert "if _ct_nvfp4_groups(config) and not return_args:" in source
