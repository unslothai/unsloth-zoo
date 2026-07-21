# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team.
# Licensed under the GNU Affero General Public License, version 3 or later.

"""MLX LoRA coverage for routed SwitchLinear expert projections."""

import json
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
@pytest.mark.parametrize("init_lora_weights", [True, "gaussian", False])
def test_type_driven_peft_trains_routed_expert_without_unfreezing_base(
    quantized, init_lora_weights,
):
    import math
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
        init_lora_weights=init_lora_weights,
        use_gradient_checkpointing=False,
    )
    dense = model.model.layers[0].proj
    switch = model.model.layers[1].proj
    assert isinstance(dense, LoRALinear)
    assert isinstance(switch, LoRASwitchLinear)
    if init_lora_weights is not False:
        assert bool(mx.allclose(dense(dense_x), dense_before))
        assert bool(mx.allclose(switch(switch_x, indices), switch_before))
    if init_lora_weights == "gaussian":
        assert 0.15 < float(mx.std(switch.lora_a)) < 0.35
    elif init_lora_weights is False:
        assert float(mx.max(mx.abs(switch.lora_a))) <= 1.0 / math.sqrt(64)
        assert float(mx.max(mx.abs(switch.lora_b))) <= 1.0 / math.sqrt(4)
        assert bool(mx.any(switch.lora_b != 0))
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


def test_legacy_quant_map_may_omit_switch_paths():
    import mlx.nn as nn
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear
    from unsloth_zoo.mlx.loader import (
        _effective_mlx_quantization_map,
        _validate_mlx_adapter_base,
    )

    model = nn.Module()
    model.dense = nn.QuantizedLinear(64, 16)
    model.experts = QuantizedSwitchLinear(64, 16, 2, bias=False)
    live_map = _effective_mlx_quantization_map(model)
    legacy_config = {
        "base_resolved_quantization_map": {"dense": live_map["dense"]},
    }

    _validate_mlx_adapter_base(model, legacy_config)

    marked_config = {
        **legacy_config,
        "base_resolved_quantization_map_supports_switch": True,
    }
    with pytest.raises(ValueError, match="unexpected quantized modules"):
        _validate_mlx_adapter_base(model, marked_config)

    model.extra_dense = nn.QuantizedLinear(64, 16)
    with pytest.raises(ValueError, match="unexpected quantized modules"):
        _validate_mlx_adapter_base(model, legacy_config)

    switch_aware = nn.Module()
    switch_aware.experts = QuantizedSwitchLinear(64, 16, 2, bias=False)
    switch_aware.extra_experts = QuantizedSwitchLinear(64, 16, 2, bias=False)
    switch_map = _effective_mlx_quantization_map(switch_aware)
    with pytest.raises(ValueError, match="unexpected quantized modules"):
        _validate_mlx_adapter_base(switch_aware, {
            "base_resolved_quantization_map": {
                "experts": switch_map["experts"],
            },
        })


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


@pytest.mark.parametrize(
    "backend,quantized,corrupt",
    [
        ("mlx_lm", False, None),
        ("mlx_lm", True, None),
        ("mlx_vlm", False, None),
        ("mlx_vlm", True, None),
        ("mlx_vlm", False, "fp32"),
        ("mlx_lm", False, "bf16"),
        ("mlx_lm", False, "missing_b"),
        ("mlx_lm", False, "legacy_flattened"),
        ("mlx_lm", False, "pathless"),
    ],
)
def test_switch_adapter_public_lifecycle(
    monkeypatch, tmp_path, backend, quantized, corrupt,
):
    import mlx.core as mx
    import mlx_lm.utils as mlx_lm_utils
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchLinear
    from mlx_lm.tuner.lora import LoRASwitchLinear
    import unsloth_zoo.mlx.loader as loader
    from unsloth_zoo.mlx.utils import save_lora_adapters, save_merged_model

    switch, quantized_switch, wrapper = (
        _install_vlm_types(monkeypatch)
        if backend == "mlx_vlm"
        else (SwitchLinear, QuantizedSwitchLinear, LoRASwitchLinear)
    )
    projection_type = quantized_switch if quantized else switch

    def make_model():
        mx.random.seed(17)
        projection = (
            projection_type()
            if backend == "mlx_vlm"
            else projection_type(64, 16, 2, bias=False)
        )
        model = _direct_layer_model([_Block(projection)])
        return model

    trained, base = make_model(), make_model()
    loader.FastMLXModel.get_peft_model(
        trained, r=2, lora_alpha=2, target_modules=["proj"],
        use_gradient_checkpointing=False,
    )
    trained.layers[0].proj.lora_b = mx.ones_like(
        trained.layers[0].proj.lora_b,
    ) * 0.125
    trained._hf_repo = "test/switch-base"
    x = mx.random.normal((1, 2, 1, 1, 64))
    indices = mx.array([[[0], [1]]])
    expected = trained.layers[0].proj(x, indices)
    adapter = tmp_path / f"{backend}-{'q' if quantized else 'dense'}-{corrupt}"
    save_lora_adapters(trained, adapter)
    config = json.loads((adapter / "adapter_config.json").read_text())
    path = "backbone.blocks.0.proj"
    assert config["unsloth_mlx_lora_module_paths"] == [path]
    if quantized:
        assert config["base_resolved_quantization_map"] == {
            path: {"bits": 4, "group_size": 64, "mode": "affine"},
        }
        assert config["base_resolved_quantization_map_supports_switch"] is True
    weights_path = adapter / "adapters.safetensors"
    weights = mx.load(str(weights_path))
    assert {value.ndim for value in weights.values()} == {3}
    if corrupt == "pathless":
        config.pop("unsloth_mlx_lora_module_paths")
        (adapter / "adapter_config.json").write_text(json.dumps(config))
    elif corrupt:
        if corrupt == "missing_b":
            weights.pop(f"{path}.lora_b")
        elif corrupt == "legacy_flattened":
            lora_a = weights[f"{path}.lora_a"]
            weights[f"{path}.lora_a"] = lora_a.reshape((-1, lora_a.shape[-1]))
            config.pop("unsloth_mlx_lora_module_paths")
            (adapter / "adapter_config.json").write_text(json.dumps(config))
        else:
            weights[f"{path}.lora_b"] = weights[f"{path}.lora_b"][..., :1]
        if corrupt == "bf16":
            weights = {key: value.astype(mx.bfloat16) for key, value in weights.items()}
        mx.save_safetensors(str(weights_path), weights)

    original = loader.FastMLXModel.from_pretrained
    tokenizer = types.SimpleNamespace()
    monkeypatch.setattr(mlx_lm_utils, "_download", lambda *args, **kwargs: adapter)

    def load_base(*args, **kwargs):
        if quantized:
            assert kwargs["mlx_quantization_config"]["quantize_modules"] == [path]
        return base, tokenizer

    monkeypatch.setattr(
        loader.FastMLXModel, "from_pretrained", staticmethod(load_base),
    )
    if corrupt not in (None, "pathless"):
        with pytest.raises(RuntimeError, match="partial adapter"):
            original(str(adapter), load_in_4bit=False)
        return

    loaded, tokenizer = original(str(adapter), load_in_4bit=False)
    assert isinstance(loaded.layers[0].proj, wrapper)
    assert bool(mx.allclose(loaded.layers[0].proj(x, indices), expected))
    if not quantized:
        monkeypatch.setattr(mlx_lm_utils, "save_model", lambda *args, **kwargs: None)
        monkeypatch.setattr(mlx_lm_utils, "create_model_card", lambda *args, **kwargs: None)
        tokenizer.save_pretrained = lambda *args, **kwargs: None
        save_merged_model(loaded, tokenizer, tmp_path / f"merged-{backend}")
        assert not hasattr(loaded.layers[0].proj, "lora_a")
        assert bool(mx.allclose(loaded.layers[0].proj(x, indices), expected, atol=1e-5))


def test_legacy_switch_rank_layout(tmp_path):
    import math
    import mlx.core as mx
    import unsloth_zoo.mlx.loader as loader
    import unsloth_zoo.mlx.utils as mlx_utils

    module = types.SimpleNamespace(
        lora_a=mx.zeros((4, 64)), lora_b=mx.zeros((2, 16, 2)),
        num_experts=2,
    )
    assert mlx_utils._infer_mlx_lora_rank(module) == 2

    model = _Model([("proj", module)])
    mx.random.seed(31)
    loader._apply_mlx_lora_initialization(model, "gaussian")
    assert 0.35 < float(mx.std(module.lora_a)) < 0.65
    mx.random.seed(31)
    loader._apply_mlx_lora_initialization(model, False)
    assert float(mx.max(mx.abs(module.lora_a))) <= 1.0 / math.sqrt(64)
    assert float(mx.max(mx.abs(module.lora_b))) <= 1.0 / math.sqrt(2)
    assert bool(mx.any(module.lora_b != 0))

    weights = tmp_path / "legacy.safetensors"
    mx.save_safetensors(str(weights), {
        "proj.lora_a": module.lora_a.astype(mx.bfloat16),
        "proj.lora_b": module.lora_b.astype(mx.bfloat16),
    })
    assert loader._infer_rank_from_saved_adapter(str(weights), "proj") == 2
