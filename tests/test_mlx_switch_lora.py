# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team.
# Licensed under the GNU Affero General Public License, version 3 or later.

"""MLX LoRA coverage for routed SwitchLinear expert projections."""

import sys
import types

import pytest


pytest.importorskip("mlx")


class _Model:
    def __init__(self, modules):
        self._modules = modules
        self.layers = [self]

    def named_modules(self):
        yield from self._modules


def test_mlx_lm_switch_targets_are_discovered():
    import mlx.nn as nn
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchLinear
    from unsloth_zoo.mlx.loader import (
        _collect_all_linear_target_names,
        _resolve_lora_keys,
    )

    modules = [
        ("mlp.gate_proj", SwitchLinear(8, 16, 2, bias=False)),
        ("mlp.down_proj", QuantizedSwitchLinear(64, 16, 2, bias=False)),
        ("mlp.quant_proj", nn.QuantizedLinear(64, 16, bias=False)),
    ]
    model = _Model(modules)
    assert _collect_all_linear_target_names(model) == [
        "down_proj", "gate_proj", "quant_proj",
    ]
    assert _resolve_lora_keys(model, ["gate_proj", "down_proj", "quant_proj"]) == {
        name for name, _ in modules
    }


def _install_vlm_types(monkeypatch):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear
    from mlx_lm.tuner.lora import LoRASwitchLinear

    class VLMSwitchLinear(nn.Module):
        def __init__(self, input_dims=64, output_dims=16, num_experts=2):
            super().__init__()
            self.weight = mx.random.normal((num_experts, output_dims, input_dims))

        input_dims = property(lambda self: self.weight.shape[2])
        output_dims = property(lambda self: self.weight.shape[1])
        num_experts = property(lambda self: self.weight.shape[0])

        def __call__(self, x, indices, sorted_indices=False):
            return mx.gather_mm(
                x, self.weight.swapaxes(-1, -2), rhs_indices=indices,
                sorted_indices=sorted_indices,
            )

    class VLMQuantizedSwitchLinear(nn.Module):
        def __init__(self, input_dims=64, output_dims=16, num_experts=2):
            super().__init__()
            source = QuantizedSwitchLinear(
                input_dims, output_dims, num_experts, bias=False,
            )
            self.weight = source.weight
            self.scales = source.scales
            self.biases = source.biases
            self.group_size = source.group_size
            self.bits = source.bits
            self.mode = source.mode
            self.freeze()

        input_dims = property(lambda self: self.scales.shape[2] * self.group_size)
        output_dims = property(lambda self: self.weight.shape[1])
        num_experts = property(lambda self: self.weight.shape[0])

        def __call__(self, x, indices, sorted_indices=False):
            return mx.gather_qmm(
                x, self.weight, self.scales, self.biases,
                rhs_indices=indices, transpose=True,
                group_size=self.group_size, bits=self.bits, mode=self.mode,
                sorted_indices=sorted_indices,
            )

    class VLMLoRASwitchLinear(LoRASwitchLinear):
        @staticmethod
        def from_base(linear, r=8, dropout=0.0, scale=20.0):
            wrapped = VLMLoRASwitchLinear(
                linear.input_dims, linear.output_dims, linear.num_experts,
                r=r, dropout=dropout, scale=scale,
            )
            wrapped.linear = linear
            return wrapped

    switch_module = types.ModuleType("mlx_vlm.models.switch_layers")
    switch_module.SwitchLinear = VLMSwitchLinear
    switch_module.QuantizedSwitchLinear = VLMQuantizedSwitchLinear
    lora_module = types.ModuleType("mlx_vlm.trainer.lora_layers")
    lora_module.LoRASwitchLinear = VLMLoRASwitchLinear
    monkeypatch.setitem(sys.modules, switch_module.__name__, switch_module)
    monkeypatch.setitem(sys.modules, lora_module.__name__, lora_module)
    return VLMSwitchLinear, VLMQuantizedSwitchLinear, VLMLoRASwitchLinear


class _Block:
    def __new__(cls, projection):
        import mlx.nn as nn

        block = nn.Module()
        block.proj = projection
        return block


class _TinyModel:
    def __new__(cls, projections):
        import mlx.nn as nn

        model = nn.Module()
        model.model = nn.Module()
        model.model.layers = [_Block(projection) for projection in projections]
        model._unsloth_full_finetuning = False
        model._is_vlm_model = False
        return model


def _direct_layer_model(layers):
    import mlx.nn as nn

    class DirectLayerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Module()
            self.backbone.blocks = layers
            self._unsloth_full_finetuning = False
            self._is_vlm_model = False

        @property
        def layers(self):
            return self.backbone.blocks

    return DirectLayerModel()


@pytest.mark.parametrize("quantized", [False, True])
def test_type_driven_peft_trains_routed_expert_without_unfreezing_base(quantized):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchLinear
    from mlx_lm.tuner.lora import LoRALinear, LoRASwitchLinear
    from unsloth_zoo.mlx.loader import FastMLXModel

    switch_type = QuantizedSwitchLinear if quantized else SwitchLinear
    model = _TinyModel([
        nn.Linear(64, 16, bias=False),
        switch_type(64, 16, 2, bias=False),
    ])
    dense_x = mx.random.normal((3, 64))
    switch_x = mx.random.normal((1, 3, 1, 1, 64))
    indices = mx.array([[[0], [0], [0]]])
    dense_before = model.model.layers[0].proj(dense_x)
    switch_before = model.model.layers[1].proj(switch_x, indices)
    mx.eval(dense_before, switch_before)

    FastMLXModel.get_peft_model(
        model, r=4, lora_alpha=4, target_modules=["proj"],
        use_gradient_checkpointing=False,
    )
    dense = model.model.layers[0].proj
    switch = model.model.layers[1].proj
    assert isinstance(dense, LoRALinear)
    assert isinstance(switch, LoRASwitchLinear)
    assert bool(mx.allclose(dense(dense_x), dense_before))
    assert bool(mx.allclose(switch(switch_x, indices), switch_before))
    trainable = dict(tree_flatten(model.trainable_parameters()))
    assert trainable and all(name.endswith(("lora_a", "lora_b")) for name in trainable)

    base_before = mx.array(switch.linear.weight)
    adapter_before = mx.array(switch.lora_b)

    def loss_fn(active_model, inputs, expert_indices):
        return mx.sum(active_model.model.layers[1].proj(inputs, expert_indices))

    _, grads = nn.value_and_grad(model, loss_fn)(model, switch_x, indices)
    b_grad = next(
        value for name, value in tree_flatten(grads)
        if ".layers.1." in name and name.endswith("lora_b")
    )
    mx.eval(b_grad)
    assert float(mx.sum(mx.abs(b_grad[0]))) > 0.0
    assert float(mx.sum(mx.abs(b_grad[1]))) == 0.0

    optimizer = optim.SGD(learning_rate=0.1)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    assert not bool(mx.array_equal(switch.lora_b, adapter_before))
    assert bool(mx.array_equal(switch.linear.weight, base_before))


def test_custom_to_lora_dispatch_preserves_fused_outputs():
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.models.afm7 import FusedLinear, FusedLoRALinear
    from unsloth_zoo.mlx.loader import FastMLXModel

    model = _TinyModel([FusedLinear(64, [8, 8])])
    x = mx.random.normal((2, 64))
    before = model.model.layers[0].proj(x)
    FastMLXModel.get_peft_model(
        model, r=2, lora_alpha=2, target_modules=["proj"],
        use_gradient_checkpointing=False,
    )
    wrapped = model.model.layers[0].proj
    assert isinstance(wrapped, FusedLoRALinear)
    assert all(
        bool(mx.allclose(left, right))
        for left, right in zip(before, wrapped(x))
    )
    trainable = dict(tree_flatten(model.trainable_parameters()))
    assert any("lora_a." in name for name in trainable)
    assert any("lora_b." in name for name in trainable)


def test_direct_layers_respect_finetune_last_n_layers():
    from mlx_lm.models.switch_layers import SwitchLinear
    from mlx_lm.tuner.lora import LoRASwitchLinear
    from unsloth_zoo.mlx.loader import FastMLXModel

    model = _direct_layer_model([
        _Block(SwitchLinear(8, 16, 2, bias=False)) for _ in range(3)
    ])
    FastMLXModel.get_peft_model(
        model, r=2, lora_alpha=2, target_modules=["proj"],
        finetune_last_n_layers=1, use_gradient_checkpointing=False,
    )
    assert all(
        not isinstance(layer.proj, LoRASwitchLinear)
        for layer in model.layers[:2]
    )
    assert isinstance(model.layers[2].proj, LoRASwitchLinear)


@pytest.mark.parametrize("quantized", [False, True])
def test_independent_vlm_switch_uses_matching_wrapper(monkeypatch, quantized):
    import mlx.core as mx
    import mlx.nn as nn
    from unsloth_zoo.mlx.loader import (
        FastMLXModel,
        _collect_all_linear_target_names,
        _resolve_lora_keys,
    )

    switch_type, quantized_type, wrapper_type = _install_vlm_types(monkeypatch)
    base_type = quantized_type if quantized else switch_type
    language_model = _TinyModel([base_type()])
    assert _collect_all_linear_target_names(language_model) == ["proj"]
    assert _resolve_lora_keys(language_model, ["proj"]) == {"proj"}

    model = nn.Module()
    model.language_model = language_model
    model._unsloth_full_finetuning = False
    model._is_vlm_model = True
    x = mx.random.normal((1, 2, 1, 1, 64))
    indices = mx.array([[[0], [1]]])
    before = language_model.model.layers[0].proj(x, indices)
    FastMLXModel.get_peft_model(
        model, r=2, lora_alpha=2, target_modules=["proj"],
        use_gradient_checkpointing=False,
    )
    wrapped = model.language_model.model.layers[0].proj
    assert isinstance(wrapped, wrapper_type)
    assert bool(mx.allclose(wrapped(x, indices), before))
