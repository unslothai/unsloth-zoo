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

"""Deeper MLX component exercises: trainer, compile discovery,
cce backward, and quantization helpers, beyond just imports.

If a test fails, the failing component identifies the next gap.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch


@pytest.fixture(autouse=True, scope="module")
def _install_shim():
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()


# ---------------------------------------------------------------------------
# 1. MLXTrainingConfig: full surface check.
# ---------------------------------------------------------------------------

def test_mlx_training_config_is_dataclass_with_all_fields():
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig
    assert dataclasses.is_dataclass(MLXTrainingConfig)
    fields = {f.name for f in dataclasses.fields(MLXTrainingConfig)}
    # Required SFT-compat fields
    for must_have in (
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "max_steps",
        "learning_rate",
        "lr_scheduler_type",
        "optim",
        "weight_decay",
        "max_grad_norm",
        "max_grad_leaf_norm",
        "seed",
        "logging_steps",
        "output_dir",
        "max_seq_length",
        "use_cce",
        "compile",
        "gradient_checkpointing",
        "dataset_order",
        "preserve_dataset_order",
        "completion_only_loss",
    ):
        assert must_have in fields, f"missing field: {must_have}"


def test_mlx_training_config_exposes_completion_only_loss():
    from unsloth_zoo.mlx.trainer import (
        MLXTrainingConfig,
        _text_completion_only_loss_arg,
    )

    assert _text_completion_only_loss_arg(
        MLXTrainingConfig(completion_only_loss=False)
    ) is False
    assert _text_completion_only_loss_arg(
        MLXTrainingConfig(completion_only_loss=True)
    ) is True
    assert _text_completion_only_loss_arg(
        MLXTrainingConfig(train_on_completions=True)
    ) is True


@pytest.mark.parametrize("optim_name", ["adamw", "adam", "sgd", "adafactor"])
def test_mlx_training_config_each_optim(optim_name):
    """Every supported optim string constructs cleanly in config."""
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig
    cfg = MLXTrainingConfig(optim=optim_name)
    assert cfg.optim == optim_name


def test_trainer_drives_dynamic_lr_outside_optimizer_scheduler():
    from unsloth_zoo.mlx.trainer import (
        MLXTrainer,
        MLXTrainingConfig,
    )

    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer.args = MLXTrainingConfig(
        learning_rate=5e-5,
        lr_scheduler_type="linear",
        warmup_steps=5,
    )
    schedule = trainer._build_schedule(total_steps=8)
    def value_at(step):
        value = schedule(step)
        return value.item() if hasattr(value, "item") else float(value)

    assert value_at(0) == pytest.approx(0.0)
    assert value_at(1) > value_at(0)
    assert value_at(4) < trainer.args.learning_rate
    assert value_at(5) == pytest.approx(trainer.args.learning_rate)

    trainer.model = object()
    optimizer = trainer._build_optimizer(total_steps=8)
    assert not callable(optimizer.learning_rate)
    first_lr = float(optimizer.learning_rate)
    trainer._set_optimizer_lr_for_step(optimizer, 1)
    second_lr = float(optimizer.learning_rate)
    assert second_lr > first_lr


def test_adamw_weight_decay_uses_hf_bias_norm_filter():
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    class DummyModel:
        def trainable_parameters(self):
            return {}

    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer.model = DummyModel()
    trainer.args = MLXTrainingConfig(
        optim="adamw",
        weight_decay=0.1,
    )

    optimizer = trainer._build_optimizer(total_steps=8)

    assert trainer._manual_weight_decay == pytest.approx(0.1)
    if hasattr(optimizer, "_kw"):
        assert optimizer._kw["weight_decay"] == 0.0
    assert MLXTrainer._should_apply_weight_decay("layers.0.mlp.down_proj.weight")
    assert not MLXTrainer._should_apply_weight_decay("layers.0.mlp.down_proj.bias")
    assert not MLXTrainer._should_apply_weight_decay("layers.0.input_layernorm.weight")
    assert not MLXTrainer._should_apply_weight_decay("vision.blocks.0.norm1.weight")


@pytest.mark.parametrize("optim_name", ["muon", "lion"])
def test_decoupled_optimizers_use_hf_parity_manual_decay(optim_name):
    """Muon and Lion mirror the AdamW pattern: zero out the optimizer's
    built-in `weight_decay` and let `_apply_manual_weight_decay` own the
    decoupled decay so bias and norm params are excluded."""
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    class DummyModel:
        def trainable_parameters(self):
            return {}

    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer.model = DummyModel()
    trainer.args = MLXTrainingConfig(
        optim=optim_name,
        weight_decay=0.05,
    )

    optimizer = trainer._build_optimizer(total_steps=4)

    assert trainer._manual_weight_decay == pytest.approx(0.05)
    assert trainer._coupled_weight_decay == pytest.approx(0.0)
    if hasattr(optimizer, "_kw"):
        assert optimizer._kw["weight_decay"] == 0.0


def test_sgd_weight_decay_is_coupled_not_decoupled():
    """SGD must use coupled decay (folded into the gradient before momentum)
    to match HF/PyTorch SGD, not the AdamW-style decoupled parameter shrink."""
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    class DummyModel:
        def trainable_parameters(self):
            return {}

    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer.model = DummyModel()
    trainer.args = MLXTrainingConfig(optim="sgd", weight_decay=0.05)

    optimizer = trainer._build_optimizer(total_steps=4)

    assert trainer._coupled_weight_decay == pytest.approx(0.05)
    assert trainer._manual_weight_decay == pytest.approx(0.0)
    if hasattr(optimizer, "_kw"):
        assert optimizer._kw["weight_decay"] == 0.0


def test_norm_clip_dtype_restore_keeps_lora_and_norms_promotable():
    from unsloth_zoo.mlx.trainer import MLXTrainer

    def should_restore_original_dtype(name):
        return (
            not MLXTrainer._is_norm_parameter_name(name)
            and not MLXTrainer._is_lora_parameter_name(name)
        )

    assert should_restore_original_dtype("model.layers.0.mlp.down_proj.weight")
    assert not should_restore_original_dtype("model.layers.0.self_attn.q_proj.lora_a")
    assert not should_restore_original_dtype("model.layers.0.self_attn.q_proj.lora_b")
    assert not should_restore_original_dtype("model.layers.0.input_layernorm.weight")
    assert not should_restore_original_dtype("vision.blocks.0.norm1.weight")


def test_global_norm_clip_reduces_in_float32():
    import inspect

    from unsloth_zoo.mlx.trainer import _clip_grad_norm_fp32

    source = inspect.getsource(_clip_grad_norm_fp32)

    assert "g.astype(mx.float32)" in source
    assert "scale.astype(g.dtype)" in source
    assert "tree_reduce" in source


@pytest.mark.parametrize(
    ("scheduler", "warmup"),
    [
        ("linear", 0),
        ("linear", 5),
        ("cosine", 0),
        ("cosine", 5),
        ("constant", 0),
        ("constant", 5),
    ],
)
def test_scheduler_lr_matches_expected_optimizer_update_steps(scheduler, warmup):
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    total_steps = 8
    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer.args = MLXTrainingConfig(
        learning_rate=5e-5,
        lr_scheduler_type=scheduler,
        warmup_steps=warmup,
    )
    schedule = trainer._build_schedule(total_steps=total_steps)

    if callable(schedule):
        raw_values = [schedule(step) for step in range(total_steps)]
    else:
        raw_values = [schedule] * total_steps
    values = [
        value.item() if hasattr(value, "item") else float(value)
        for value in raw_values
    ]

    if scheduler == "linear" and warmup == 0:
        # Match `transformers.get_scheduler("linear", num_warmup_steps=0,
        # num_training_steps=total_steps)` as seen by optimizer steps across
        # Transformers 4.56.1 through 5.5.0: step 1 uses base LR, then decays.
        lr = trainer.args.learning_rate
        expected = [lr * (total_steps - step) / total_steps for step in range(total_steps)]
        assert values == pytest.approx(expected)
    elif warmup > 0:
        assert values[0] == pytest.approx(0.0)
        assert all(value > 0.0 for value in values[1:])
    else:
        assert all(value > 0.0 for value in values)


def test_mlx_text_dataset_does_not_append_eos(monkeypatch):
    """Studio formatting owns EOS decisions; MLX batching must not add one."""
    import sys

    class CacheDataset:
        def __init__(self, data):
            self._data = data
            self._cache = {}

        def __len__(self):
            return len(self._data)

        def __getitem__(self, idx):
            if idx not in self._cache:
                self._cache[idx] = self._data.process(self._data[idx])
            return self._cache[idx]

        def itemlen(self, idx):
            return len(self[idx][0])

    monkeypatch.setattr(sys.modules["mlx_lm.tuner.datasets"], "CacheDataset", CacheDataset)

    from unsloth_zoo.mlx.utils import _prepare_dataset

    class Tokenizer:
        eos_token_id = 99
        chat_template = None

        def encode(self, text):
            assert text == "hello"
            return [1, 2, 3]

    # append_eos=False is what Studio passes (chat-template renders EOS).
    dataset = _prepare_dataset([{"text": "hello"}], Tokenizer(), append_eos=False)
    assert dataset[0] == ([1, 2, 3], 0)

    # Default (mlx-lm parity for direct MLX text fine-tuning callers)
    # appends the tokenizer EOS so a raw `{"text": str}` row still
    # trains the model to predict EOS.
    dataset_default = _prepare_dataset([{"text": "hello"}], Tokenizer())
    assert dataset_default[0] == ([1, 2, 3, 99], 0)


def test_encode_mlx_text_keeps_raw_text_bos_when_template_has_bos():
    from unsloth_zoo.mlx.utils import encode_mlx_text

    class Tokenizer:
        bos_token = "<s>"
        chat_template = "{{ bos_token }}{{ messages }}"

        def __init__(self):
            self.add_special_tokens_seen = []

        def encode(self, text, add_special_tokens=True):
            self.add_special_tokens_seen.append(add_special_tokens)
            return [1, 2, 3]

    tokenizer = Tokenizer()

    encode_mlx_text(tokenizer, "raw text")
    encode_mlx_text(tokenizer, "<s>rendered text")

    assert tokenizer.add_special_tokens_seen == [True, False]


def test_mlx_text_loss_masks_exclude_position_at_sequence_length():
    import inspect
    from unsloth_zoo.mlx import utils as mlx_utils

    source = inspect.getsource(mlx_utils.make_baseline_loss_fn)
    assert "steps < lengths[:, 1:]" in source


def test_train_on_responses_only_forwards_last_response_only(monkeypatch):
    import unsloth_zoo.dataset_utils as dataset_utils
    from unsloth_zoo.mlx.trainer import train_on_responses_only

    class CallableTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2, 3]}

    received = {}

    def fake_hf(trainer, *, instruction_part=None, response_part=None,
                force_match=True, tokenizer=None, return_function=False,
                num_proc=None, last_response_only=False):
        received["last_response_only"] = last_response_only
        return lambda batch: batch

    monkeypatch.setattr(dataset_utils, "train_on_responses_only", fake_hf)
    train_on_responses_only(
        None,
        instruction_part="<user>",
        response_part="<assistant>",
        tokenizer=CallableTokenizer(),
        return_function=True,
        last_response_only=True,
    )

    assert received["last_response_only"] is True


def test_response_mask_tokenizer_rejects_encode_only_tokenizer():
    from unsloth_zoo.mlx.trainer import _resolve_response_mask_tokenizer

    class EncodeOnlyTokenizer:
        def encode(self, text):
            return [1, 2, 3]

        def convert_tokens_to_ids(self, token):
            return 1

    with pytest.raises(TypeError, match="requires a callable"):
        _resolve_response_mask_tokenizer(EncodeOnlyTokenizer())


def test_vlm_eval_batches_define_completion_only_loss_before_use():
    import inspect

    from unsloth_zoo.mlx.trainer import MLXTrainer

    source = inspect.getsource(MLXTrainer._train_inner)
    definition = source.index("text_completion_only_loss = _text_completion_only_loss_arg(args)")
    eval_use = source.index("completion_only_loss=text_completion_only_loss")
    assert definition < eval_use


def test_evaluate_dict_eval_datasets_records_split_metrics():
    import mlx.core as mx

    from unsloth_zoo.mlx.trainer import MLXTrainer

    class Model:
        def __init__(self):
            self.modes = []

        def eval(self):
            self.modes.append("eval")

        def train(self):
            self.modes.append("train")

    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer.model = Model()
    trainer.stop_requested = False

    def loss_fn(_model, name, _lengths, _labels):
        if name == "small":
            return mx.array(1.0), mx.array(2)
        return mx.array(3.0), mx.array(6)

    loss, ppl = trainer._evaluate(
        {"small": [("small", None, None)], "large": [("large", None, None)]},
        loss_fn,
        is_vlm=False,
    )

    assert loss == pytest.approx(2.5)
    assert ppl == pytest.approx(__import__("math").exp(2.5))
    assert trainer._last_eval_metrics["eval_small_loss"] == pytest.approx(1.0)
    assert trainer._last_eval_metrics["eval_large_loss"] == pytest.approx(3.0)
    assert trainer._last_eval_metrics["eval_loss"] == pytest.approx(2.5)
    assert trainer.model.modes == ["eval", "train"]


def test_vlm_cce_prefers_collated_position_ids_for_cuda_parity():
    import inspect
    from unsloth_zoo.mlx import utils as mlx_utils

    forward_source = inspect.getsource(mlx_utils._vlm_cce_forward)
    unpack_source = inspect.getsource(mlx_utils._unpack_embed_result)
    prepare_source = inspect.getsource(mlx_utils._prepare_vlm_batch_for_compile)
    assert '"_unsloth_collated_position_ids"' in prepare_source
    assert 'not k.startswith("_unsloth_")' in forward_source
    assert 'use_collated_position_ids and "position_ids" in extra_kwargs' in forward_source
    assert 'lm is not None and "position_ids" not in backbone_kwargs' in unpack_source


def test_mlx_train_result_reports_base_quantization():
    import inspect
    from unsloth_zoo.mlx.trainer import MLXTrainer

    source = inspect.getsource(MLXTrainer._train_inner)
    assert '"base_quantization_config"' in source
    assert '"base_quantization_policy"' in source
    assert '"base_quantized_source"' in source


def test_mlx_loader_exposes_dense_nf4_diagnostic_mode():
    import mlx.core as mx
    from unsloth_zoo.mlx.loader import (
        _MLX_QUANT_MODE_DEFAULTS,
        _nf4_dense_dequantize_weight,
    )

    assert _MLX_QUANT_MODE_DEFAULTS["nf4_dense"] == (64, 4)

    weight = mx.array([[-1.0, -0.6961928, 0.0, 0.72295684]], dtype=mx.float32)
    dequantized = _nf4_dense_dequantize_weight(weight, group_size=4)
    assert dequantized.shape == weight.shape
    assert dequantized.reshape((-1,)).tolist() == pytest.approx(
        weight.reshape((-1,)).tolist()
    )


def test_mlx_loader_keeps_norm_parameters_float32():
    import mlx.core as mx
    from unsloth_zoo.mlx.loader import _keep_norm_parameters_float32

    class TinyModel:
        def __init__(self):
            self._parameters = {
                "vision_tower": {
                    "blocks": {
                        "0": {
                            "norm1": {
                                "weight": mx.array([1.0], dtype=mx.bfloat16),
                                "bias": mx.array([0.0], dtype=mx.bfloat16),
                            },
                            "attn": {
                                "qkv": {
                                    "weight": mx.array([[1.0]], dtype=mx.bfloat16),
                                },
                            },
                        },
                    },
                },
                "language_model": {
                    "model": {
                        "layers": {
                            "0": {
                                "input_layernorm": {
                                    "weight": mx.array([1.0], dtype=mx.bfloat16),
                                },
                            },
                        },
                    },
                },
            }

        def parameters(self):
            return self._parameters

        def update(self, parameters):
            self._parameters = parameters

    model = TinyModel()
    _keep_norm_parameters_float32(model)
    params = model.parameters()

    assert params["vision_tower"]["blocks"]["0"]["norm1"]["weight"].dtype == mx.float32
    assert params["vision_tower"]["blocks"]["0"]["norm1"]["bias"].dtype == mx.float32
    assert (
        params["language_model"]["model"]["layers"]["0"]["input_layernorm"]["weight"].dtype
        == mx.float32
    )
    assert (
        params["vision_tower"]["blocks"]["0"]["attn"]["qkv"]["weight"].dtype
        == mx.bfloat16
    )


def test_mlx_loader_fixes_gemma3_vision_post_layernorm_eps():
    from types import SimpleNamespace

    from unsloth_zoo.mlx.loader import _fix_gemma3_vision_post_layernorm_eps

    post_layernorm = SimpleNamespace(eps=1e-5)
    model = SimpleNamespace(
        config=SimpleNamespace(
            vision_config=SimpleNamespace(layer_norm_eps=1e-6),
        ),
        vision_tower=SimpleNamespace(
            vision_model=SimpleNamespace(post_layernorm=post_layernorm),
        ),
    )

    assert _fix_gemma3_vision_post_layernorm_eps(model) is True
    assert post_layernorm.eps == 1e-6
    assert model._unsloth_gemma3_vision_post_layernorm_eps == 1e-6


def test_mlx_loader_patches_gemma3_vision_attention_fp32_sdpa():
    import inspect

    import unsloth_zoo.mlx.loader as loader
    from unsloth_zoo.mlx.loader import _fix_gemma3_vision_attention_fp32_sdpa

    patched = _fix_gemma3_vision_attention_fp32_sdpa()
    assert patched in {True, False}

    source = inspect.getsource(loader._fix_gemma3_vision_attention_fp32_sdpa)
    assert "scaled_dot_product_attention" in source
    assert "astype(mx.float32)" in source
    assert "output.astype(orig_dtype)" in source


def test_mlx_loader_patches_gemma3_text_rmsnorm_fp32():
    import inspect

    import unsloth_zoo.mlx.loader as loader
    from unsloth_zoo.mlx.loader import _fix_gemma3_text_rmsnorm_fp32

    patched = _fix_gemma3_text_rmsnorm_fp32()
    assert patched in {True, False}

    source = inspect.getsource(loader._fix_gemma3_text_rmsnorm_fp32)
    assert "x.astype(mx.float32)" in source
    assert "mx.rsqrt(mx.mean(x_f * x_f" in source
    assert "return y.astype(orig_dtype)" in source
    assert "_unsloth_fp32_rmsnorm_patched" in source


def test_vlm_hidden_stack_preserves_inputs_embed_dtype():
    import inspect

    import unsloth_zoo.mlx.utils as utils

    source = inspect.getsource(utils._run_hidden_stack)
    assert "h = inputs_embeds" in source
    assert "inputs_embeds.astype(norm_weight.dtype)" not in source


def test_mlx_loader_patches_gemma3_vision_mlp_fp32_activation():
    import inspect

    import unsloth_zoo.mlx.loader as loader
    from unsloth_zoo.mlx.loader import _fix_gemma3_vision_mlp_fp32_activation

    patched = _fix_gemma3_vision_mlp_fp32_activation()
    assert patched in {True, False}

    source = inspect.getsource(loader._fix_gemma3_vision_mlp_fp32_activation)
    assert "activation_fn(x.astype(mx.float32)).astype(orig_dtype)" in source
    assert "_unsloth_fp32_activation_patched" in source


def test_mlx_loader_patches_gemma3_vision_encoder_fp32_layernorm():
    import inspect

    import unsloth_zoo.mlx.loader as loader
    from unsloth_zoo.mlx.loader import _fix_gemma3_vision_encoder_fp32_layernorm

    patched = _fix_gemma3_vision_encoder_fp32_layernorm()
    assert patched in {True, False}

    source = inspect.getsource(loader._fix_gemma3_vision_encoder_fp32_layernorm)
    assert "x.astype(mx.float32)" in source
    assert "return y.astype(orig_dtype)" in source
    assert "_unsloth_fp32_layernorm_patched" in source


def test_mlx_loader_patches_gemma3_vision_post_layernorm_fp32():
    import inspect

    import unsloth_zoo.mlx.loader as loader
    from unsloth_zoo.mlx.loader import _fix_gemma3_vision_post_layernorm_fp32

    patched = _fix_gemma3_vision_post_layernorm_fp32()
    assert patched in {True, False}

    source = inspect.getsource(loader._fix_gemma3_vision_post_layernorm_fp32)
    assert "pooler_output = torch_like_layer_norm" in source
    assert "return y.astype(orig_dtype)" in source
    assert "_unsloth_fp32_post_layernorm_patched" in source


def test_mlx_loader_patches_gemma3_image_feature_scale():
    import inspect

    import mlx.core as mx
    import unsloth_zoo.mlx.loader as loader
    from unsloth_zoo.mlx.loader import _fix_gemma3_multimodal_image_feature_scale

    patched = _fix_gemma3_multimodal_image_feature_scale()
    assert patched in {True, False}

    source = inspect.getsource(loader._fix_gemma3_multimodal_image_feature_scale)
    assert "embed_dim = image_features.shape[-1]" in source
    assert "image_features / (embed_dim**0.5)" in source
    assert "del hidden_size" in source

    if patched:
        from mlx_vlm.models.gemma3.gemma3 import Model

        image_token_id = 99
        input_ids = mx.array([[1, image_token_id, image_token_id]])
        inputs_embeds = mx.ones((1, 3, 4))
        image_features = mx.ones((1, 2, 4))
        attention_mask = mx.ones((1, 3))

        embeds, _ = Model.prepare_inputs_for_multimodal(
            9,
            0,
            image_token_id,
            image_features,
            inputs_embeds,
            input_ids,
            attention_mask,
        )

        assert mx.allclose(embeds[0, 1:], mx.full((2, 4), 0.5))


def test_qwen3_vl_vision_rotary_uses_transformers_fp32_math():
    import inspect
    import unsloth_zoo.mlx.compile as mc

    source = inspect.getsource(mc._install_qwen3_family_compile_patches)

    assert "def _qwen3_vision_rotary_fp32" in source
    assert "tensor_f = tensor.astype(mx.float32)" in source
    assert "freqs_f = freqs.astype(mx.float32)" in source
    assert "return rotated.astype(orig_dtype)" in source
    assert "q = _qwen3_vision_rotary_fp32(q, rotary_pos_emb)" in source
    assert "k = _qwen3_vision_rotary_fp32(k, rotary_pos_emb)" in source


def test_qwen3_vl_vision_block_mlp_fp32_guard_for_fp16():
    """Pin the fp16 MLP overflow guard in patched_qwen3_vision_block_call.

    On M1/M2 Macs (no native bf16), MLX defaults to float16 for the vision
    tower. The vision block's MLP linear_fc1 (up-projection) produces output
    magnitudes that exceed fp16's 65504 ceiling for some inputs; downcasting
    to fp16 saturates to inf and cascades to NaN in the backward.

    Fix: when activation dtype is fp16, upcast the MLP input to fp32 so the
    entire MLP (fc1, GELU, fc2) runs in fp32. The output is cast back to
    source dtype at the residual add. bf16/fp32 keep the original path.
    """
    import inspect
    import unsloth_zoo.mlx.compile as mc

    source = inspect.getsource(mc._install_qwen3_family_compile_patches)

    # Guard is present
    assert "linear_fc1 (up-projection) overflows fp16" in source, (
        "Missing comment documenting the fp16 overflow rationale"
    )
    # Dtype-conditional branch keys on residual_dtype (the activation dtype)
    assert "if residual_dtype == mx.float16:" in source, (
        "MLP fp32 guard must be gated on residual_dtype == mx.float16"
    )
    # fp16 path: upcast input to fp32 before calling self.mlp
    assert "self.mlp(mlp_norm_out.astype(mx.float32))" in source, (
        "fp16 branch must upcast mlp input to fp32"
    )
    # non-fp16 path: original (cheaper) cast-only flow preserved
    assert "self.mlp(mlp_norm_out)" in source, (
        "bf16/fp32 path must keep the original self.mlp(...) call"
    )


def test_qwen3_vl_training_compile_verified():
    import unsloth_zoo.mlx.compile as mc

    assert "qwen3_vl" in mc._VERIFIED_TRAINING_ARCHES
    assert "qwen3_vl_moe" in mc._VERIFIED_TRAINING_ARCHES


def test_quantized_cce_uses_layer_mode_and_affine_bias_guard():
    import inspect
    import unsloth_zoo.mlx.utils as mlx_utils

    source = inspect.getsource(mlx_utils.make_vlm_cce_loss_fn)
    assert 'quant_mode = getattr(lm_layer, "mode", "affine")' in source
    assert "mode=quant_mode" in source
    assert 'if bi is None and quant_mode == "affine":' in source
    assert "bi = mx.zeros_like(sc)" in source


def test_gemma3_training_compile_verified():
    import unsloth_zoo.mlx.compile as mc

    assert "gemma3" in mc._VERIFIED_TRAINING_ARCHES


# ---------------------------------------------------------------------------
# 2. compile module-level discovery functions return sensible defaults
#    on a host with no real MLX architectures.
# ---------------------------------------------------------------------------

def test_compile_discovers_no_archs_under_shim():
    """No real mlx_vlm.models.* installed -> empty discovery, not crash."""
    import unsloth_zoo.mlx.compile as mc
    archs = mc.discover_architectures()
    assert isinstance(archs, tuple)


def test_compile_patch_primitives_exist():
    import unsloth_zoo.mlx.compile as mc
    primitives = mc.list_compile_patch_primitives()
    assert len(primitives) > 0


def test_compile_protocol_requirements_exist():
    import unsloth_zoo.mlx.compile as mc
    reqs = mc.list_protocol_requirements()
    assert len(reqs) > 0


def test_compile_summarize_qualifications_returns_dict():
    import unsloth_zoo.mlx.compile as mc
    s = mc.summarize_compile_qualifications()
    assert isinstance(s, dict)
    assert "architectures" in s


# ---------------------------------------------------------------------------
# 3. CCE backward via the pure-Python fallback.
# ---------------------------------------------------------------------------

def test_cce_backward_via_torch_autograd():
    """Build a tiny CCE forward and verify torch.autograd traverses it."""
    from unsloth_zoo.mlx.cce.runtime_cce import _forward_chunked_fused_finalize

    torch.manual_seed(0)
    n, hd, vocab = 4, 8, 32
    hidden = torch.randn(n, hd, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(vocab, hd, dtype=torch.float32) * 0.1
    weight.requires_grad_(True)
    targets = torch.tensor([3, 17, 5, 29], dtype=torch.int32)

    loss, _ = _forward_chunked_fused_finalize(
        hidden, weight, targets,
        scales=None, biases=None, group_size=None, bits=None, mode="affine",
        ignore_index=-100, logit_softcap=0.0, chunk_size=16,
        forward_update_kernel=None, forward_update_finalize_kernel=None,
    )
    loss.sum().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()


# ---------------------------------------------------------------------------
# 4. mx.dequantize cross-validation against the helper's output.
# ---------------------------------------------------------------------------

def test_mx_dequantize_with_nonzero_bias_and_scale():
    import mlx.core as mx

    bits, group_size = 4, 8
    elements_per_word = 32 // bits
    packed_value = 0
    for i, v in enumerate([0, 1, 2, 3, 4, 5, 6, 7]):
        packed_value |= v << (i * bits)
    packed = torch.tensor([[packed_value]], dtype=torch.int32)
    scale = 0.5
    bias = -1.0
    scales = torch.tensor([[scale]])
    biases = torch.tensor([[bias]])

    out = mx.dequantize(packed, scales, biases, group_size=group_size,
                       bits=bits, mode="affine")
    expected = torch.tensor([[v * scale + bias for v in range(8)]],
                            dtype=scales.dtype)
    torch.testing.assert_close(out, expected)


# ---------------------------------------------------------------------------
# 5. mx.fast.scaled_dot_product_attention works for a small attention.
# ---------------------------------------------------------------------------

def test_mx_fast_sdpa_works():
    import mlx.core as mx
    B, H, T, D = 1, 2, 4, 8
    q = torch.randn(B, H, T, D, dtype=torch.float32)
    k = torch.randn(B, H, T, D, dtype=torch.float32)
    v = torch.randn(B, H, T, D, dtype=torch.float32)
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / (D ** 0.5))
    assert out.shape == (B, H, T, D)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 6. Tree utilities round-trip.
# ---------------------------------------------------------------------------

def test_tree_flatten_unflatten_roundtrip():
    from mlx.utils import tree_flatten, tree_unflatten

    tree = {"a": {"b": torch.tensor([1.0]), "c": torch.tensor([2.0])},
            "d": torch.tensor([3.0])}
    flat = tree_flatten(tree)
    keys = sorted(k for k, _ in flat)
    assert keys == ["a.b", "a.c", "d"]

    rebuilt = tree_unflatten(flat)
    assert set(rebuilt.keys()) == {"a", "d"}
    torch.testing.assert_close(rebuilt["d"], torch.tensor([3.0]))


# ---------------------------------------------------------------------------
# 7. Quantized layer __call__ works (forward through nn.QuantizedLinear).
# ---------------------------------------------------------------------------

def test_quantized_linear_forward():
    import mlx.nn as nn
    bits, group_size = 4, 8

    # 4-bit, in_features=8, out_features=2.
    elements_per_word = 32 // bits
    packed_value = 0
    for i, v in enumerate([0, 1, 2, 3, 4, 5, 6, 7]):
        packed_value |= v << (i * bits)
    packed_row = torch.tensor([[packed_value]], dtype=torch.int32)
    packed = torch.cat([packed_row, packed_row], dim=0)  # (2, 1)
    scales = torch.ones((2, 1), dtype=torch.float32)
    biases = torch.zeros((2, 1), dtype=torch.float32)

    layer = nn.QuantizedLinear(8, 2, bias=False, group_size=group_size,
                                bits=bits, mode="affine")
    layer.weight = packed
    layer.scales = scales
    layer.biases = biases

    x = torch.ones((1, 8), dtype=torch.float32)
    # x @ W.T  with W = [[0,1,2,3,4,5,6,7], [0,1,2,3,4,5,6,7]] = [28, 28]
    out = layer(x)
    torch.testing.assert_close(out, torch.tensor([[28.0, 28.0]]))
