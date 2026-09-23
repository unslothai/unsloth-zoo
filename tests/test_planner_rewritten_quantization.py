# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""The planner rebuilds a checkpoint's config from its name. An NVIDIA ModelOpt FP8
checkpoint serializes `quant_method: modelopt`, which transformers cannot build a
quantizer for, so planning failed and the load fell back to `sequential`. Unsloth
rewrites that block into the transformers fp8 form for the load and hands the same
form to the planner, which has to size with it."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from unsloth_zoo import device_map_planner as planner  # noqa: E402

_MODELOPT = {
    "quant_method": "modelopt",
    "quant_algo": "FP8",
    "producer": {"name": "modelopt"},
    "ignore": ["lm_head"],
}
_FP8_PLAN = {
    "quant_method": "fp8",
    "activation_scheme": "static",
    "weight_block_size": None,
    "modules_to_not_convert": ["lm_head"],
}


def _fp8_plan_supported():
    try:
        from transformers.utils.quantization_config import FineGrainedFP8Config

        FineGrainedFP8Config(**{k: v for k, v in _FP8_PLAN.items() if k != "quant_method"})
    except Exception:
        return False
    return True


needs_per_tensor_fp8 = pytest.mark.skipif(
    not _fp8_plan_supported(), reason = "this transformers has no per-tensor fp8 config"
)


def _modelopt_checkpoint(tmp_path):
    from transformers import LlamaConfig

    config = LlamaConfig(
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        vocab_size = 256,
    )
    config.quantization_config = dict(_MODELOPT)
    config.save_pretrained(tmp_path)
    return str(tmp_path)


def test_unknown_methods_are_recognised():
    assert not planner._quantization_method_is_known(_MODELOPT)
    assert planner._quantization_method_is_known(_FP8_PLAN)
    assert planner._quantization_method_is_known({"load_in_4bit": True})
    # Left to transformers: no method, or an already built config object.
    assert planner._quantization_method_is_known({"bits": 4})
    assert planner._quantization_method_is_known(object())


@needs_per_tensor_fp8
def test_a_rewritten_modelopt_checkpoint_is_sized_with_the_callers_config(tmp_path):
    path = _modelopt_checkpoint(tmp_path)
    model, hf_quantizer, config = planner.build_meta_model(
        path, quantization_config = dict(_FP8_PLAN)
    )
    assert type(hf_quantizer).__name__ == "FineGrainedFP8HfQuantizer"
    assert hf_quantizer.pre_quantized
    assert config.quantization_config["quant_method"] == "fp8"
    # The quantized Linear classes are swapped in, so the size table is the fp8 one.
    proj = model.model.layers[0].self_attn.q_proj
    assert type(proj).__name__ != "Linear"
    assert type(model.lm_head).__name__ == "Linear"


def test_an_unknown_method_without_a_callers_config_still_refuses(tmp_path):
    path = _modelopt_checkpoint(tmp_path)
    with pytest.raises(ValueError, match = "modelopt"):
        planner.build_meta_model(path)


@needs_per_tensor_fp8
def test_a_known_serialized_method_still_wins_over_the_callers_config(tmp_path):
    from transformers import LlamaConfig

    config = LlamaConfig(
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 1,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        vocab_size = 256,
    )
    config.quantization_config = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    }
    config.save_pretrained(tmp_path)
    from transformers.utils.quantization_config import FineGrainedFP8Config

    runtime = FineGrainedFP8Config(activation_scheme = "static", weight_block_size = None)
    _, hf_quantizer, built = planner.build_meta_model(str(tmp_path), quantization_config = runtime)
    assert type(hf_quantizer).__name__ == "FineGrainedFP8HfQuantizer"
    # merge_quantization_configs keeps the checkpoint's own block size.
    assert list(hf_quantizer.quantization_config.weight_block_size) == [128, 128]
    assert built.quantization_config["quant_method"] == "fp8"
