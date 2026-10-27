# SPDX-License-Identifier: AGPL-3.0-only

import pytest

mx = pytest.importorskip("mlx.core")
from mlx_simulation import mlx_is_simulated

if mlx_is_simulated():
    pytest.skip("Requires native MLX", allow_module_level = True)
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason = "Requires Metal")

import mlx.nn as nn
from mlx_lm.models import llama
from unsloth_zoo.mlx.loader import _apply_mlx_quantization, _resolve_mlx_quantization_spec

_LOAD_FLAGS = dict(
    load_in_4bit = True, load_in_8bit = False, load_in_16bit = False, load_in_fp8 = False,
    load_in_mxfp4 = False, load_in_nvfp4 = False, full_finetuning = False, q_bits = None,
    q_group_size = None, q_mode = None, mlx_quantization_config = None, quantization_config = None,
    quant_predicate = None, quantize_modules = None, force_requantize = False,
)


def _quantized(**overrides):
    args = llama.ModelArgs(
        model_type = "llama", hidden_size = 128, num_hidden_layers = 2, intermediate_size = 256,
        num_attention_heads = 4, num_key_value_heads = 4, rms_norm_eps = 1e-5, vocab_size = 512,
        tie_word_embeddings = False,
    )
    spec = _resolve_mlx_quantization_spec(**{**_LOAD_FLAGS, **overrides})
    model, _ = _apply_mlx_quantization(llama.Model(args), {}, spec, is_vlm = False)
    return model


def test_runtime_4bit_keeps_embeddings_full_precision_by_default():
    model = _quantized()
    assert type(model.model.embed_tokens) is nn.Embedding
    assert type(model.lm_head) is nn.Linear
    assert isinstance(model.model.layers[0].self_attn.q_proj, nn.QuantizedLinear)


def test_quantize_embeddings_quantizes_embed_tokens_and_lm_head():
    model = _quantized(quantize_embeddings = True)
    assert isinstance(model.model.embed_tokens, nn.QuantizedEmbedding)
    assert isinstance(model.lm_head, nn.QuantizedLinear)
    assert isinstance(model.model.layers[0].self_attn.q_proj, nn.QuantizedLinear)


def test_quantize_embeddings_does_nothing_for_full_finetuning():
    model = _quantized(quantize_embeddings = True, full_finetuning = True)
    assert type(model.model.embed_tokens) is nn.Embedding
    assert type(model.lm_head) is nn.Linear
