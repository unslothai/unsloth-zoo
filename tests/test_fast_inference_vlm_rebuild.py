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

"""fast_inference vLLM -> HF rebuild for gemma4_unified (gemma-4-12b) and muse_glimmer, on CPU.

The fake vLLM models mirror the module names of vllm/model_executor/models/gemma4_unified.py
(language model from gemma4.py) and muse_glimmer.py; the reference is the transformers model.
"""

import copy
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

transformers = pytest.importorskip("transformers")


def _needs(name):
    # Gemma 4 and Muse Glimmer arrive in transformers 5.x; the 4.57.6 floor has neither.
    return pytest.mark.skipif(not hasattr(transformers, name), reason = f"transformers lacks {name}")


_needs_gemma4_unified = _needs("Gemma4UnifiedForConditionalGeneration")
_needs_muse_glimmer = _needs("MuseGlimmerForConditionalGeneration")
_needs_gemma4 = _needs("Gemma4ForConditionalGeneration")


def _linear(weight, bias = None, output_sizes = None):
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(weight.detach().clone(), requires_grad = False)
    module.bias = None if bias is None else torch.nn.Parameter(bias.detach().clone(), requires_grad = False)
    if output_sizes is not None:
        module.output_sizes = list(output_sizes)
    return module


def _stack(sd, names, bias = False):
    weights = [sd[f"{n}.weight"] for n in names]
    biases = torch.cat([sd[f"{n}.bias"] for n in names]) if bias else None
    return _linear(torch.cat(weights), biases, [w.shape[0] for w in weights])


def _llm(root):
    engine = types.SimpleNamespace(model_executor = types.SimpleNamespace(
        driver_worker = types.SimpleNamespace(model_runner = types.SimpleNamespace(model = root))))
    return types.SimpleNamespace(llm_engine = engine)


def _randomize(model):
    torch.manual_seed(0)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn_like(param) * 0.05 + (1.0 if param.ndim == 1 else 0.0))
    return model.eval()


def _rebuild(monkeypatch, llm, config):
    from unsloth_zoo import vllm_utils
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 0))
    monkeypatch.setattr(vllm_utils, "get_target_device", lambda index = 0: torch.device("cpu"))
    _, quant_state_dict = vllm_utils._get_vllm_state_dict(
        llm, config = config, is_vision_model = True,
    )
    rebuilt = vllm_utils.convert_vllm_to_huggingface(
        quant_state_dict, copy.deepcopy(config), dtype = torch.float32, is_vision_model = True,
    )
    return quant_state_dict, rebuilt


def _assert_same_weights(reference, rebuilt):
    expected = reference.state_dict()
    got = rebuilt.state_dict()
    assert sorted(set(expected) - set(got)) == []
    assert sorted(set(got) - set(expected)) == []
    for key, value in expected.items():
        assert got[key].shape == value.shape, key
        assert torch.equal(got[key].float(), value.float()), key


# ---------------------------------------------------------------- gemma4_unified

def _gemma4_unified_config():
    from transformers.models.gemma4_unified.configuration_gemma4_unified import Gemma4UnifiedConfig
    layer_types = ["sliding_attention", "sliding_attention", "full_attention"]
    config = Gemma4UnifiedConfig(
        text_config = dict(
            vocab_size = 128, vocab_size_per_layer_input = 128, hidden_size = 32, intermediate_size = 64,
            num_hidden_layers = 3, layer_types = layer_types, num_attention_heads = 2, num_key_value_heads = 1,
            num_global_key_value_heads = 1, head_dim = 16, global_head_dim = 32, attention_k_eq_v = True,
            hidden_size_per_layer_input = 0, num_kv_shared_layers = 0, sliding_window = 8,
            max_position_embeddings = 64, tie_word_embeddings = True,
        ),
        vision_config = dict(patch_size = 2, pooling_kernel_size = 2, mm_embed_dim = 24, mm_posemb_size = 8,
                             output_proj_dims = 24, num_soft_tokens = 4),
        audio_config = dict(audio_embed_dim = 12),
    )
    config.architectures = ["Gemma4UnifiedForConditionalGeneration"]
    return config


def _vllm_gemma4_unified(hf):
    """Mirrors Gemma4UnifiedForConditionalGeneration (no .model; language_model.model = gemma4.Gemma4Model)."""
    sd = hf.state_dict()
    text_config = hf.config.text_config
    lm = "model.language_model"
    text = torch.nn.Module()
    text.embed_tokens = _linear(sd[f"{lm}.embed_tokens.weight"])
    text.norm = _linear(sd[f"{lm}.norm.weight"])
    layers = torch.nn.ModuleList()
    for kk, layer_type in enumerate(text_config.layer_types):
        p = f"{lm}.layers.{kk}"
        layer = torch.nn.Module()
        attn = torch.nn.Module()
        q, k = sd[f"{p}.self_attn.q_proj.weight"], sd[f"{p}.self_attn.k_proj.weight"]
        # attention_k_eq_v full-attention layers: vLLM loads k_proj into both the K and V slots
        v = sd.get(f"{p}.self_attn.v_proj.weight", k)
        attn.qkv_proj = _linear(torch.cat([q, k, v]), output_sizes = [q.shape[0], k.shape[0], v.shape[0]])
        attn.o_proj = _linear(sd[f"{p}.self_attn.o_proj.weight"])
        attn.q_norm = _linear(sd[f"{p}.self_attn.q_norm.weight"])
        attn.k_norm = _linear(sd[f"{p}.self_attn.k_norm.weight"])
        layer.self_attn = attn
        layer.mlp = torch.nn.Module()
        layer.mlp.gate_up_proj = _stack(sd, [f"{p}.mlp.gate_proj", f"{p}.mlp.up_proj"])
        layer.mlp.down_proj = _linear(sd[f"{p}.mlp.down_proj.weight"])
        for norm in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"):
            setattr(layer, norm, _linear(sd[f"{p}.{norm}.weight"]))
        layer.layer_scalar = torch.nn.Parameter(sd[f"{p}.layer_scalar"].clone(), requires_grad = False)
        layers.append(layer)
    text.layers = layers
    root = torch.nn.Module()
    root.config = hf.config
    root.language_model = torch.nn.Module()
    root.language_model.model = text
    root.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"], "gate_up_proj": ["gate_proj", "up_proj"]}
    root.vision_embedder = torch.nn.Module()
    for name in ("patch_ln1", "patch_dense", "patch_ln2", "pos_norm"):
        setattr(root.vision_embedder, name, _linear(sd[f"model.embed_vision.{name}.weight"], sd[f"model.embed_vision.{name}.bias"]))
    root.vision_embedder.pos_embedding = torch.nn.Parameter(sd["model.embed_vision.pos_embedding"].clone(), requires_grad = False)
    root.embed_vision = torch.nn.Module()
    root.embed_vision.embedding_projection = _linear(sd["model.embed_vision.multimodal_embedder.embedding_projection.weight"])
    root.embed_audio = torch.nn.Module()
    root.embed_audio.embedding_projection = _linear(sd["model.embed_audio.embedding_projection.weight"])
    return _llm(root)


@_needs_gemma4_unified
def test_gemma4_unified_is_a_gemma4_config():
    from unsloth_zoo.empty_model import _is_gemma4_config
    assert _is_gemma4_config(_gemma4_unified_config())
    assert _is_gemma4_config(types.SimpleNamespace(model_type = "gemma4_unified_text"))
    assert not _is_gemma4_config(types.SimpleNamespace(model_type = "gemma3"))


@_needs_gemma4_unified
def test_gemma4_unified_vllm_rebuild_matches_hf(monkeypatch, tmp_path):
    from safetensors.torch import save_file
    from transformers import Gemma4UnifiedForConditionalGeneration
    config = _gemma4_unified_config()
    reference = _randomize(Gemma4UnifiedForConditionalGeneration(copy.deepcopy(config)))
    # vLLM serves embed_audio, but the HF audio projection is read from the checkpoint (as for dense Gemma-4)
    audio_key = "model.embed_audio.embedding_projection.weight"
    save_file({audio_key: reference.state_dict()[audio_key].contiguous()}, str(tmp_path / "model.safetensors"))
    config._name_or_path = str(tmp_path)

    quant_state_dict, rebuilt = _rebuild(monkeypatch, _vllm_gemma4_unified(reference), config)
    assert "model.embed_vision.embedding_projection.weight" not in quant_state_dict
    assert "model.embed_vision.patch_dense.bias" in quant_state_dict
    _assert_same_weights(reference, rebuilt)
    assert rebuilt.model.language_model.layers[2].self_attn.v_proj is None

    input_ids = torch.tensor([[2, 5, 9, 17, 33, 64, 100, 7]])
    with torch.no_grad():
        torch.testing.assert_close(rebuilt(input_ids = input_ids).logits, reference(input_ids = input_ids).logits)
        patches = torch.randn(1, 4, 3 * 4 * 4)
        positions = torch.tensor([[[0, 0], [0, 1], [1, 0], [1, 1]]])
        torch.testing.assert_close(
            rebuilt.model.embed_vision(patches, positions), reference.model.embed_vision(patches, positions),
        )


@_needs_gemma4_unified
def test_gemma4_unified_bnb_patch_dense_is_dequantized(monkeypatch, tmp_path):
    # load_in_4bit: vLLM builds vision_embedder.patch_dense with quant_config; its key is installed
    # as a plain tensor, so a packed weight must come back dequantized, not as uint8.
    bnb_functional = pytest.importorskip("bitsandbytes.functional")
    from safetensors.torch import save_file
    from transformers import Gemma4UnifiedForConditionalGeneration
    config = _gemma4_unified_config()
    reference = _randomize(Gemma4UnifiedForConditionalGeneration(copy.deepcopy(config)))
    audio_key = "model.embed_audio.embedding_projection.weight"
    save_file({audio_key: reference.state_dict()[audio_key].contiguous()}, str(tmp_path / "model.safetensors"))
    config._name_or_path = str(tmp_path)
    llm = _vllm_gemma4_unified(reference)
    patch_dense = llm.llm_engine.model_executor.driver_worker.model_runner.model.vision_embedder.patch_dense
    packed, quant_state = bnb_functional.quantize_4bit(patch_dense.weight.data, quant_type = "nf4")
    patch_dense.weight = torch.nn.Parameter(packed, requires_grad = False)
    patch_dense.weight.bnb_quant_state = {0: quant_state}
    expected = bnb_functional.dequantize_4bit(packed, quant_state)

    with torch.inference_mode():  # get_vllm_state_dict runs under vLLM's inference_mode
        _, rebuilt = _rebuild(monkeypatch, llm, config)
    got = rebuilt.model.embed_vision.patch_dense.weight
    assert got.dtype == torch.float32 and torch.equal(got, expected)
    assert not got.is_inference()  # trainable (vision LoRA / full vision fine-tuning)
    # NF4 rounds to within half its widest level gap (-1.0 to -0.696, so 0.152 x the block absmax).
    # Checked through a flat dequantize: bitsandbytes' CPU dequantize_4bit reads the wrong block's
    # absmax for 2-D shapes whose rows are not a multiple of the block size (this one is 24 x 12).
    original = reference.model.embed_vision.patch_dense.weight
    flat_state = copy.copy(quant_state)
    flat_state.shape = torch.Size([1, original.numel()])
    unpacked = bnb_functional.dequantize_4bit(packed, flat_state).view_as(original)
    torch.testing.assert_close(unpacked, original, atol = 0.16 * original.abs().max().item(), rtol = 0)
    if torch.equal(expected, unpacked):
        torch.testing.assert_close(got, original, atol = 0.16 * original.abs().max().item(), rtol = 0)


# ---------------------------------------------------------------- muse_glimmer

def _muse_glimmer_config():
    from transformers import MuseGlimmerConfig
    config = MuseGlimmerConfig(
        text_config = dict(
            vocab_size = 128, hidden_size = 32, intermediate_size = 48, num_hidden_layers = 4,
            num_attention_heads = 4, num_key_value_heads = 2, head_dim = 8, sliding_window = 8,
            max_position_embeddings = 64, tie_word_embeddings = False,
            layer_types = ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
            layer_rope_theta = [10000.0, 10000.0, 10000.0, 0],
        ),
        vision_config = dict(
            hidden_size = 16, intermediate_size = 32, num_hidden_layers = 2, num_attention_heads = 2,
            layer_types = ["window_attention", "full_attention"], patch_size = 2, patch_temporal = 2,
            pos_emb_height = 4, pos_emb_width = 4, merge_size = 2, max_position_embeddings = 16,
        ),
        out_hidden_size = 64, projector_hidden_size = 24,
    )
    config.architectures = ["MuseGlimmerForConditionalGeneration"]
    return config


def _vllm_muse_glimmer(hf):
    """Mirrors MuseGlimmerForCausalLM: .model (text), vision_encoder, vision_adapter, vision_projection, lm_head."""
    sd = hf.state_dict()
    lm = "model.language_model"
    text = torch.nn.Module()
    text.embed_tokens = _linear(sd[f"{lm}.embed_tokens.weight"])
    text.norm = _linear(sd[f"{lm}.norm.weight"])
    layers = torch.nn.ModuleList()
    for kk in range(hf.config.text_config.num_hidden_layers):
        p = f"{lm}.layers.{kk}"
        layer = torch.nn.Module()
        attn = torch.nn.Module()
        attn.qkv_proj = _stack(sd, [f"{p}.self_attn.{x}_proj" for x in "qkv"])
        attn.o_proj = _linear(sd[f"{p}.self_attn.o_proj.weight"])
        attn.output_gate_proj = _linear(sd[f"{p}.self_attn.gate_proj.weight"])
        attn.qk_norm = torch.nn.Module()  # weightless (with_scale = False)
        layer.self_attn = attn
        layer.mlp = torch.nn.Module()
        layer.mlp.gate_up_proj = _stack(sd, [f"{p}.mlp.gate_proj", f"{p}.mlp.up_proj"])
        layer.mlp.down_proj = _linear(sd[f"{p}.mlp.down_proj.weight"])
        for norm in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"):
            setattr(layer, norm, _linear(sd[f"{p}.{norm}.weight"]))
        layers.append(layer)
    text.layers = layers
    root = torch.nn.Module()
    root.config = hf.config
    root.model = text
    root.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"], "gate_up_proj": ["gate_proj", "up_proj"]}
    root.lm_head = _linear(sd["lm_head.weight"])

    tower = "model.vision_tower"
    encoder = torch.nn.Module()
    encoder.conv1_linear = _linear(sd[f"{tower}.patch_embedder.patch_embedding.weight"])
    encoder.positional_embedding_vlm = torch.nn.Parameter(
        sd[f"{tower}.patch_embedder.position_embedding_table.weight"].clone(), requires_grad = False)
    encoder.ln_pre = _linear(sd[f"{tower}.ln_pre.weight"], sd[f"{tower}.ln_pre.bias"])
    encoder.ln_post = _linear(sd[f"{tower}.ln_post.weight"], sd[f"{tower}.ln_post.bias"])
    blocks = torch.nn.ModuleList()
    for kk in range(hf.config.vision_config.num_hidden_layers):
        p = f"{tower}.layers.{kk}"
        block = torch.nn.Module()
        block.ln_1 = _linear(sd[f"{p}.norm1.weight"], sd[f"{p}.norm1.bias"])
        block.ln_2 = _linear(sd[f"{p}.norm2.weight"], sd[f"{p}.norm2.bias"])
        block.attn = torch.nn.Module()
        block.attn.qkv_proj = _stack(sd, [f"{p}.attn.{x}_proj" for x in "qkv"], bias = True)
        block.attn.o_proj = _linear(sd[f"{p}.attn.proj.weight"], sd[f"{p}.attn.proj.bias"])
        block.mlp = torch.nn.Module()
        block.mlp.c_fc = _linear(sd[f"{p}.mlp.fc1.weight"], sd[f"{p}.mlp.fc1.bias"])
        block.mlp.c_proj = _linear(sd[f"{p}.mlp.fc2.weight"], sd[f"{p}.mlp.fc2.bias"])
        blocks.append(block)
    encoder.transformer = blocks
    root.vision_encoder = encoder
    root.vision_adapter = torch.nn.Module()
    root.vision_adapter.c_fc = _linear(sd["model.vision_adapter.fc1.weight"])
    root.vision_adapter.c_proj = _linear(sd["model.vision_adapter.fc2.weight"])
    root.vision_projection = _linear(sd["model.vision_projection.weight"])
    return _llm(root)


@_needs_muse_glimmer
def test_muse_glimmer_vllm_rebuild_matches_hf(monkeypatch):
    from transformers import MuseGlimmerForConditionalGeneration
    config = _muse_glimmer_config()
    reference = _randomize(MuseGlimmerForConditionalGeneration(copy.deepcopy(config)))

    quant_state_dict, rebuilt = _rebuild(monkeypatch, _vllm_muse_glimmer(reference), config)
    assert "model.language_model.layers.0.self_attn.gate_proj.weight" in quant_state_dict
    assert not any(k.startswith("model.layers.") for k in quant_state_dict)
    _assert_same_weights(reference, rebuilt)

    input_ids = torch.tensor([[2, 5, 9, 17, 33, 64, 100, 7]])
    with torch.no_grad():
        torch.testing.assert_close(rebuilt(input_ids = input_ids).logits, reference(input_ids = input_ids).logits)
        grid_thw = torch.tensor([[1, 4, 4]])
        pixel_values = torch.randn(16, 2 * 3 * 2 * 2)
        torch.testing.assert_close(
            rebuilt.model.get_image_features(pixel_values, grid_thw).pooler_output,
            reference.model.get_image_features(pixel_values, grid_thw).pooler_output,
        )


def test_text_only_muse_glimmer_config_keeps_model_prefix():
    # A bare MuseGlimmerTextConfig has no text_config: the HF causal LM is model.layers.*
    from unsloth_zoo import vllm_utils
    import inspect
    source = inspect.getsource(vllm_utils._get_vllm_state_dict)
    assert 'model_type == "muse_glimmer" and hasattr(config, "text_config")' in source


# ---------------------------------------------------------------- dense gemma4 (E2B / E4B layout)

def _gemma4_dense_config():
    from transformers import Gemma4Config
    config = Gemma4Config(
        text_config = dict(
            vocab_size = 128, vocab_size_per_layer_input = 128, hidden_size = 32, intermediate_size = 64,
            hidden_size_per_layer_input = 8, num_hidden_layers = 4,
            layer_types = ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
            num_kv_shared_layers = 2, use_double_wide_mlp = True, num_attention_heads = 2,
            num_key_value_heads = 1, head_dim = 16, global_head_dim = 32, attention_k_eq_v = False,
            sliding_window = 8, max_position_embeddings = 64, tie_word_embeddings = True,
        ),
        vision_config = dict(
            hidden_size = 16, intermediate_size = 32, num_hidden_layers = 2, num_attention_heads = 2,
            num_key_value_heads = 2, head_dim = 8, global_head_dim = 8, patch_size = 2, pooling_kernel_size = 2,
            position_embedding_size = 16, default_output_length = 4, use_clipped_linears = True,
        ),
        audio_config = dict(
            hidden_size = 16, num_hidden_layers = 1, num_attention_heads = 2, subsampling_conv_channels = [4, 4],
            output_proj_dims = 16, use_clipped_linears = True,
        ),
    )
    config.architectures = ["Gemma4ForConditionalGeneration"]
    return config


def _vllm_gemma4_dense(hf):
    """Mirrors gemma4_mm.Gemma4ForConditionalGeneration: HF vision tower module, vLLM gemma4.Gemma4Model text."""
    sd = hf.state_dict()
    text_config = hf.config.text_config
    first_shared = text_config.num_hidden_layers - text_config.num_kv_shared_layers
    lm = "model.language_model"
    text = torch.nn.Module()
    text.embed_tokens = _linear(sd[f"{lm}.embed_tokens.weight"])
    text.embed_tokens_per_layer = _linear(sd[f"{lm}.embed_tokens_per_layer.weight"])
    text.per_layer_model_projection = _linear(sd[f"{lm}.per_layer_model_projection.weight"])
    text.per_layer_projection_norm = _linear(sd[f"{lm}.per_layer_projection_norm.weight"])
    text.norm = _linear(sd[f"{lm}.norm.weight"])
    layers = torch.nn.ModuleList()
    for kk in range(text_config.num_hidden_layers):
        p = f"{lm}.layers.{kk}"
        layer = torch.nn.Module()
        attn = torch.nn.Module()
        if kk >= first_shared:
            # vLLM >= 0.30 builds KV-shared layers with q_proj only
            attn.q_proj = _linear(sd[f"{p}.self_attn.q_proj.weight"])
        else:
            attn.qkv_proj = _stack(sd, [f"{p}.self_attn.{x}_proj" for x in "qkv"])
            attn.k_norm = _linear(sd[f"{p}.self_attn.k_norm.weight"])
        attn.o_proj = _linear(sd[f"{p}.self_attn.o_proj.weight"])
        attn.q_norm = _linear(sd[f"{p}.self_attn.q_norm.weight"])
        layer.self_attn = attn
        layer.mlp = torch.nn.Module()
        layer.mlp.gate_up_proj = _stack(sd, [f"{p}.mlp.gate_proj", f"{p}.mlp.up_proj"])
        layer.mlp.down_proj = _linear(sd[f"{p}.mlp.down_proj.weight"])
        for norm in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                     "post_feedforward_layernorm", "post_per_layer_input_norm"):
            setattr(layer, norm, _linear(sd[f"{p}.{norm}.weight"]))
        layer.per_layer_input_gate = _linear(sd[f"{p}.per_layer_input_gate.weight"])
        layer.per_layer_projection = _linear(sd[f"{p}.per_layer_projection.weight"])
        layer.layer_scalar = torch.nn.Parameter(sd[f"{p}.layer_scalar"].clone(), requires_grad = False)
        layers.append(layer)
    text.layers = layers
    root = torch.nn.Module()
    root.config = hf.config
    root.language_model = torch.nn.Module()
    root.language_model.model = text
    root.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"], "gate_up_proj": ["gate_proj", "up_proj"]}
    root.vision_tower = copy.deepcopy(hf.model.vision_tower)
    root.embed_vision = copy.deepcopy(hf.model.embed_vision)
    return _llm(root)


@_needs_gemma4
def test_gemma4_dense_vllm_rebuild_matches_hf(monkeypatch, tmp_path):
    from safetensors.torch import save_file
    from transformers import Gemma4ForConditionalGeneration
    config = _gemma4_dense_config()
    reference = _randomize(Gemma4ForConditionalGeneration(copy.deepcopy(config)))
    with torch.no_grad():
        for name, buffer in reference.named_buffers():
            if name.endswith(("input_min", "output_min")):
                buffer.fill_(-0.5)
            if name.endswith(("input_max", "output_max")):
                buffer.fill_(0.5)
    audio = {k: v.contiguous() for k, v in reference.state_dict().items() if k.startswith(("model.audio_tower.", "model.embed_audio."))}
    save_file(audio, str(tmp_path / "model.safetensors"))
    config._name_or_path = str(tmp_path)

    quant_state_dict, rebuilt = _rebuild(monkeypatch, _vllm_gemma4_dense(reference), config)
    assert "model.vision_tower.encoder.layers.1.mlp.down_proj.output_max" in quant_state_dict
    _assert_same_weights(reference, rebuilt)

    input_ids = torch.tensor([[2, 5, 9, 17, 33, 64, 100, 7]])
    with torch.no_grad():
        torch.testing.assert_close(rebuilt(input_ids = input_ids).logits, reference(input_ids = input_ids).logits)


def test_set_module_attribute_keeps_buffers_as_buffers():
    # gemma-4-26B-A4B: std_bias / std_scale are vision-tower buffers; set as Parameters the
    # rebuilt-tower census failed ("unexpected parameter", "buffer missing").
    from unsloth_zoo.empty_model import _set_module_attribute
    root = torch.nn.Module()
    root.tower = torch.nn.Module()
    root.tower.register_buffer("std_bias", torch.zeros(4))
    root.tower.proj = torch.nn.Linear(4, 4)
    value = torch.nn.Parameter(torch.arange(4.0), requires_grad = False)
    _set_module_attribute(root, "tower.std_bias", value)
    assert "std_bias" in dict(root.tower.named_buffers())
    assert "std_bias" not in dict(root.tower.named_parameters())
    assert torch.equal(root.tower.std_bias, torch.arange(4.0))
    weight = torch.nn.Parameter(torch.ones(4, 4), requires_grad = False)
    _set_module_attribute(root, "tower.proj.weight", weight)
    assert root.tower.proj.weight is weight
