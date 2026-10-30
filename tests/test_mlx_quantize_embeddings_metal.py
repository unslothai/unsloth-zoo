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


def test_quantized_embeddings_keep_mllama_image_attention_dtype():
    from mlx.utils import tree_map
    from mlx_vlm.models.mllama import mllama
    from mlx_vlm.models.mllama.config import ModelConfig, TextConfig, VisionConfig
    from unsloth_zoo.mlx.compile import _install_llama_pixtral_mistral_compile_patches

    _install_llama_pixtral_mistral_compile_patches()
    text = TextConfig(
        vocab_size = 512, hidden_size = 128, intermediate_size = 256, num_hidden_layers = 2,
        num_attention_heads = 4, num_key_value_heads = 4, cross_attention_layers = [1],
    )
    vision = VisionConfig(
        image_size = 28, patch_size = 14, hidden_size = 64, intermediate_size = 128,
        num_hidden_layers = 2, num_attention_heads = 4, max_num_tiles = 1, num_global_layers = 1,
        vision_output_dim = 128, intermediate_layers_indices = [0], max_aspect_ratio_id = 1,
    )
    model = mllama.Model(ModelConfig(
        text_config = text, vision_config = vision, model_type = "mllama",
        vocab_size = 512, image_token_index = 511,
    ))
    model.update(tree_map(
        lambda p: p.astype(mx.bfloat16) if mx.issubdtype(p.dtype, mx.floating) else p,
        model.parameters(),
    ))
    spec = _resolve_mlx_quantization_spec(**{**_LOAD_FLAGS, "quantize_embeddings": True})
    model, _ = _apply_mlx_quantization(model, {}, spec, is_vlm = True)
    assert isinstance(model.language_model.model.embed_tokens, nn.QuantizedEmbedding)
    out = model(
        mx.array([[511, 1, 2, 3]]),
        pixel_values = mx.zeros((1, 1, 1, 3, 28, 28), dtype = mx.bfloat16),
        aspect_ratio_ids = mx.array([[1]]),
        aspect_ratio_mask = mx.ones((1, 1), dtype = mx.int32),
        cross_attention_mask = mx.ones((1, 4, 1, 1), dtype = mx.int32),
    )
    logits = getattr(out, "logits", out)
    mx.eval(logits)
    assert logits.dtype == mx.bfloat16


def test_bitsandbytes_checkpoint_keeps_quantize_embeddings(monkeypatch, tmp_path):
    import json
    import shutil
    from pathlib import Path
    import mlx_lm.utils as mlx_lm_utils
    import unsloth_zoo.mlx.loader as loader
    from mlx_lm.utils import save_model
    from tokenizers import Tokenizer, models, pre_tokenizers

    config = dict(
        model_type = "llama", hidden_size = 128, num_hidden_layers = 2, intermediate_size = 256,
        num_attention_heads = 4, num_key_value_heads = 4, rms_norm_eps = 1e-5, vocab_size = 512,
        tie_word_embeddings = False,
    )
    dense = tmp_path / "dense"
    dense.mkdir()
    model = llama.Model(llama.ModelArgs(**config))
    mx.eval(model.parameters())
    save_model(str(dense), model)
    (dense / "config.json").write_text(json.dumps({**config, "architectures": ["LlamaForCausalLM"]}))
    vocab = {f"t{i}": i for i in range(511)}
    vocab["[UNK]"] = 511
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token = "[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(dense / "tokenizer.json"))
    (dense / "tokenizer_config.json").write_text(json.dumps(
        {"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "[UNK]", "eos_token": "t0"}
    ))

    repo = tmp_path / "bnb"
    repo.mkdir()
    (repo / "config.json").write_text(json.dumps({
        **config, "architectures": ["LlamaForCausalLM"],
        "quantization_config": {"quant_method": "bitsandbytes", "load_in_4bit": True},
    }))
    monkeypatch.setattr(mlx_lm_utils, "_download", lambda path, *a, **k: Path(path))
    monkeypatch.setattr(
        loader, "_dequantize_bnb_to_tempdir",
        lambda *a, **k: str(shutil.copytree(dense, tmp_path / "scratch")),
    )
    model, _ = loader.FastMLXModel.from_pretrained(
        str(repo), load_in_4bit = True, quantize_embeddings = True, text_only = True,
    )
    assert isinstance(model.model.embed_tokens, nn.QuantizedEmbedding)
    assert isinstance(model.lm_head, nn.QuantizedLinear)


def test_mixed_bit_adapter_reload_keeps_quantized_embeddings(tmp_path):
    import json
    from mlx_lm.models import qwen3_moe
    from mlx_lm.utils import save_model
    from tokenizers import Tokenizer, models, pre_tokenizers
    from unsloth_zoo.mlx.loader import FastMLXModel
    from unsloth_zoo.mlx.utils import save_lora_adapters

    config = dict(
        model_type = "qwen3_moe", hidden_size = 128, num_hidden_layers = 2, intermediate_size = 256,
        num_attention_heads = 4, num_key_value_heads = 4, rms_norm_eps = 1e-6, vocab_size = 512,
        head_dim = 32, rope_theta = 10000.0, max_position_embeddings = 512, tie_word_embeddings = False,
        num_experts = 4, num_experts_per_tok = 2, decoder_sparse_step = 1, mlp_only_layers = [],
        moe_intermediate_size = 128, norm_topk_prob = True,
    )
    base = tmp_path / "base"
    base.mkdir()
    model = qwen3_moe.Model(qwen3_moe.ModelArgs(**config))
    mx.eval(model.parameters())
    save_model(str(base), model)
    (base / "config.json").write_text(json.dumps({**config, "architectures": ["Qwen3MoeForCausalLM"]}))
    vocab = {f"t{i}": i for i in range(511)}
    vocab["[UNK]"] = 511
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token = "[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(base / "tokenizer.json"))
    (base / "tokenizer_config.json").write_text(json.dumps(
        {"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "[UNK]", "eos_token": "t0"}
    ))

    model, _ = FastMLXModel.from_pretrained(str(base), load_in_4bit = True, quantize_embeddings = True)
    assert model.model.layers[0].mlp.gate.bits == 8
    FastMLXModel.get_peft_model(
        model, r = 2, lora_alpha = 2, target_modules = ["q_proj", "v_proj"],
        use_gradient_checkpointing = False,
    )
    adapter = tmp_path / "adapter"
    save_lora_adapters(model, str(adapter))
    reloaded, _ = FastMLXModel.from_pretrained(str(adapter))
    assert isinstance(reloaded.model.embed_tokens, nn.QuantizedEmbedding)
    assert isinstance(reloaded.lm_head, nn.QuantizedLinear)
