# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""fast_inference for DeepSeek-V4.1 (model_type deepseek_v41).

vLLM's DeepseekV41ForCausalLM keeps the text model at language_model.model with the release's layer names
(attn.fused_wqa_wkv, attn.wo_a, ffn.experts.routed_experts.w13_weight, hc_attn_fn, engram.embed_tokens, ...).
Unsloth trains the native text model (unsloth.models.deepseek_v41). Every training tensor is handed out here as
a view of vLLM's storage under the native name, and the native model is rebuilt around those views, so one copy
of the weights serves both rollouts and training. Anything that cannot be a view is refused, never copied.
"""

__all__ = [
    "is_deepseek_v41_config",
    "deepseek_v41_fast_inference_refusal",
    "deepseek_v41_engine_args",
    "deepseek_v41_attention_backend",
    "get_deepseek_v41_vllm_state_dict",
    "convert_deepseek_v41_vllm_to_huggingface",
]

import os
from collections import OrderedDict
import torch

DEEPSEEK_V41_MODEL_TYPES = ("deepseek_v41", "deepseek_v41_text")
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _text_config(config):
    return getattr(config, "text_config", None) or config
pass


def is_deepseek_v41_config(config):
    # All Unsloth Zoo code licensed under LGPLv3
    if config is None: return False
    return getattr(config, "model_type", None) in DEEPSEEK_V41_MODEL_TYPES or \
        getattr(_text_config(config), "model_type", None) in DEEPSEEK_V41_MODEL_TYPES
pass


def _quant_dict(config):
    quant = getattr(config, "quantization_config", None) or getattr(_text_config(config), "quantization_config", None)
    if quant is None: return {}
    if not isinstance(quant, dict):
        quant = quant.to_dict() if hasattr(quant, "to_dict") else dict(vars(quant))
    return quant
pass


def deepseek_v41_fast_inference_refusal(config, load_in_4bit = False, load_in_8bit = False):
    """Why fast_inference cannot share this DeepSeek-V4.1 checkpoint with training, else None. Config only, so it
    runs before vLLM allocates anything."""
    # All Unsloth Zoo code licensed under LGPLv3
    if not is_deepseek_v41_config(config): return None
    if load_in_4bit or load_in_8bit:
        return (
            "bitsandbytes weights are not shared: vLLM's bitsandbytes MoE experts cannot be aliased by training "
            "and vLLM serves no LoRA on them (as for every sparse MoE model)"
        )
    quant = _quant_dict(config)
    method = str(quant.get("quant_method", "") or "").lower()
    if method:
        experts = quant.get("expert_dtype", None)
        return (
            f"the checkpoint is quantized ({method}, weight_block_size {quant.get('weight_block_size')}, "
            f"expert_dtype {experts}). vLLM runs it with MXFP8 linear kernels and MXFP4 / FP8 expert kernels that "
            "repack the weights and scales into their own layout at load, so the training model cannot alias them, "
            "and Unsloth trains this release dequantized to 16-bit"
        )
    return None
pass


def deepseek_v41_engine_args(config, is_vision_model = False):
    """vLLM engine arguments for a DeepSeek-V4.1 text load: Unsloth's native port is a text model, so the vision
    tower is not built (vLLM stubs it when the image limit is 0)."""
    # All Unsloth Zoo code licensed under LGPLv3
    if is_vision_model:
        raise NotImplementedError(
            "Unsloth: fast_inference = True loads DeepSeek-V4.1 as a text model; its vision tower is not "
            "trained by Unsloth. Load it as a language model, or set fast_inference = False."
        )
    args = {"language_model_only": True, "limit_mm_per_prompt": {"image": 0}}
    backend = deepseek_v41_attention_backend()
    if backend is not None:
        args["attention_backend"] = backend
    return args
pass


def deepseek_v41_attention_backend(capability = None):
    """vLLM attention backend for rollouts that should match training. vLLM's default DeepSeek-V4.1 backends
    (FlashMLA, mega attention) keep the KV cache in fp8 (fp8_ds_mla), which the 16-bit training model does not
    see; on SM10x FlashInfer's sparse MLA keeps it in bfloat16. UNSLOTH_DSV41_VLLM_ATTENTION_BACKEND picks one
    ("vllm" keeps vLLM's choice)."""
    # All Unsloth Zoo code licensed under LGPLv3
    chosen = os.environ.get("UNSLOTH_DSV41_VLLM_ATTENTION_BACKEND", "").strip()
    if chosen.lower() in ("vllm", "default"): return None
    if chosen: return chosen
    if capability is None:
        try:
            if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is not None: return None
            capability = torch.cuda.get_device_capability()
        except Exception:
            return None
    return "FLASHINFER_MLA_SPARSE_DSV41" if capability[0] == 10 else None
pass


def _refuse(message):
    raise NotImplementedError(f"Unsloth: fast_inference = True cannot share DeepSeek-V4.1 with vLLM: {message}. "
                              "Set fast_inference = False.")
pass


def _base(module):
    return getattr(module, "base_layer", module)  # vLLM LoRA wrappers keep the weight on base_layer
pass


def _float_tensor(tensor, where):
    if not isinstance(tensor, torch.Tensor):
        _refuse(f"{where} is missing in vLLM's model")
    if tensor.dtype not in _FLOAT_DTYPES:
        _refuse(f"{where} is {tensor.dtype} in vLLM (quantized or repacked), so it cannot be a training tensor")
    return tensor.data
pass


def _linear_weight(module, where):
    if module is None: _refuse(f"{where} is missing in vLLM's model")
    base = _base(module)
    for attr in ("weight_scale", "weight_scale_inv", "weight_packed", "qweight"):
        if getattr(base, attr, None) is not None:
            _refuse(f"{where} is quantized in vLLM ({attr} present)")
    if getattr(base, "bias", None) is not None:
        _refuse(f"{where} has a bias in vLLM, which the native DeepSeek-V4.1 layer does not have")
    return _float_tensor(getattr(base, "weight", None), where)
pass


def _split_rows(module, where, widths):
    """Row views of a vLLM merged linear, checked against its own output_sizes."""
    weight = _linear_weight(module, where)
    sizes = list(getattr(_base(module), "output_sizes", None) or [weight.shape[0]])
    if sum(sizes) != weight.shape[0] or sizes[:len(widths)] != list(widths) or len(sizes) != len(widths):
        _refuse(f"{where} has output sizes {sizes} (rows {weight.shape[0]}), expected {list(widths)}")
    out, start = [], 0
    for width in widths:
        out.append(weight[start : start + width])
        start += width
    return out
pass


def _vllm_text_model(vllm_internals):
    language_model = getattr(vllm_internals, "language_model", vllm_internals)
    text_model = getattr(language_model, "model", None)
    lm_head = getattr(language_model, "lm_head", None)
    if text_model is None or not hasattr(text_model, "layers") or lm_head is None:
        _refuse(f"vLLM's {type(vllm_internals).__name__} has no language_model.model.layers / lm_head")
    return language_model, text_model, lm_head
pass


def get_deepseek_v41_vllm_state_dict(vllm_internals, config):
    """Native-name views of every tensor in vLLM's DeepSeek-V4.1 text model. Refuses quantized or repacked
    weights, and any vLLM language-model tensor this mapping does not cover (an unshared tensor would silently
    diverge between rollouts and training)."""
    # All Unsloth Zoo code licensed under LGPLv3
    from unsloth_zoo.empty_model import vllm_moe_expert_descriptor

    text_config = _text_config(config)
    language_model, text_model, lm_head = _vllm_text_model(vllm_internals)
    n_heads = int(text_config.num_attention_heads)
    q_lora_rank = int(text_config.q_lora_rank)
    head_dim = int(text_config.head_dim)
    vocab_size = int(text_config.vocab_size)
    state_dict = OrderedDict()

    def put(name, tensor, where):
        state_dict[name] = _float_tensor(tensor, where)

    embed = _float_tensor(getattr(_base(text_model.embed_tokens), "weight", None), "embed_tokens")
    state_dict["model.embed_tokens.weight"] = embed[:vocab_size]
    if getattr(text_model, "start_layer", 0) != 0 or getattr(text_model, "end_layer", len(text_model.layers)) != len(text_model.layers):
        _refuse("vLLM split the layers across pipeline stages")

    for kk, layer in enumerate(text_model.layers):
        p = f"model.layers.{kk}"
        where = f"layer {kk}"
        attn = getattr(layer, "attn", None)
        ffn = getattr(layer, "ffn", None)
        if attn is None or ffn is None:
            _refuse(f"{where} is a {type(layer).__name__} without attn / ffn")

        # Attention: wq_a and wkv are fused as fused_wqa_wkv = [q_a_proj | kv_proj].
        q_a, kv = _split_rows(attn.fused_wqa_wkv, f"{where} attn.fused_wqa_wkv", (q_lora_rank, head_dim))
        state_dict[f"{p}.self_attn.q_a_proj.weight"] = q_a
        state_dict[f"{p}.self_attn.kv_proj.weight"] = kv
        put(f"{p}.self_attn.q_a_norm.weight", attn.q_norm.weight, f"{where} attn.q_norm")
        put(f"{p}.self_attn.kv_norm.weight", attn.kv_norm.weight, f"{where} attn.kv_norm")
        state_dict[f"{p}.self_attn.q_b_proj.weight"] = _linear_weight(attn.wq_b, f"{where} attn.wq_b")
        state_dict[f"{p}.self_attn.o_a_proj.weight"] = _linear_weight(attn.wo_a, f"{where} attn.wo_a")
        state_dict[f"{p}.self_attn.o_b_proj.weight"] = _linear_weight(attn.wo_b, f"{where} attn.wo_b")
        sinks = _float_tensor(getattr(attn, "attn_sink", None), f"{where} attn.attn_sink")
        if sinks.dim() != 1 or sinks.shape[0] < n_heads:
            _refuse(f"{where} attn.attn_sink has shape {tuple(sinks.shape)} for {n_heads} heads")
        state_dict[f"{p}.self_attn.sinks"] = sinks[:n_heads]  # vLLM pads the heads to the kernel's 64 / 128

        compressor = getattr(attn, "compressor", None)
        if compressor is not None:
            fused = compressor.fused_wkv_wgate
            sizes = list(getattr(_base(fused), "output_sizes", None) or [])
            names = ("kv_proj", "gate_proj")[:len(sizes)]
            for name, view in zip(names, _split_rows(fused, f"{where} compressor.fused_wkv_wgate", sizes)):
                state_dict[f"{p}.self_attn.compressor.{name}.weight"] = view
            put(f"{p}.self_attn.compressor.kv_norm.weight", compressor.norm.weight, f"{where} compressor.norm")
            for name, value in compressor.named_parameters(recurse = False):
                _refuse(f"{where} compressor.{name} has no native DeepSeek-V4.1 counterpart")
        indexer = getattr(attn, "indexer", None)
        if indexer is not None:
            state_dict[f"{p}.self_attn.indexer.q_b_proj.weight"] = _linear_weight(indexer.wq_b, f"{where} indexer.wq_b")
            state_dict[f"{p}.self_attn.indexer.weights_proj.weight"] = _linear_weight(
                indexer.weights_proj, f"{where} indexer.weights_proj"
            )
            if getattr(indexer, "wk", None) is not None:
                state_dict[f"{p}.self_attn.indexer.k_proj.weight"] = _linear_weight(indexer.wk, f"{where} indexer.wk")
                put(f"{p}.self_attn.indexer.k_norm.weight", indexer.k_norm.weight, f"{where} indexer.k_norm")

        # mHC mixing (float32 in both).
        for hc, native in (("attn", "attn_hc"), ("ffn", "ffn_hc")):
            for leaf in ("fn", "base", "scale"):
                put(f"{p}.{native}.{leaf}", getattr(layer, f"hc_{hc}_{leaf}", None), f"{where} hc_{hc}_{leaf}")
        put(f"{p}.input_layernorm.weight", layer.attn_norm.weight, f"{where} attn_norm")
        put(f"{p}.post_attention_layernorm.weight", layer.ffn_norm.weight, f"{where} ffn_norm")

        # MoE: router, shared expert, routed experts.
        gate = ffn.gate
        state_dict[f"{p}.mlp.gate.weight"] = _linear_weight(gate, f"{where} ffn.gate")
        put(f"{p}.mlp.gate.e_score_correction_bias", getattr(gate, "e_score_correction_bias", None),
            f"{where} ffn.gate.e_score_correction_bias")
        if getattr(gate, "bias_vl", None) is not None:
            put(f"{p}.mlp.gate.e_score_correction_bias_vl", gate.bias_vl, f"{where} ffn.gate.bias_vl")
        shared = getattr(ffn, "shared_experts", None)
        experts = getattr(ffn, "experts", None)
        if shared is None or getattr(_base(experts), "has_fused_shared_experts", False):
            _refuse(f"{where}'s shared expert is fused into the routed experts")
        inter = _linear_weight(shared.down_proj, f"{where} shared_experts.down_proj").shape[1]
        gp, up = _split_rows(shared.gate_up_proj, f"{where} shared_experts.gate_up_proj", (inter, inter))
        state_dict[f"{p}.mlp.shared_experts.gate_proj.weight"] = gp
        state_dict[f"{p}.mlp.shared_experts.up_proj.weight"] = up
        state_dict[f"{p}.mlp.shared_experts.down_proj.weight"] = _linear_weight(
            shared.down_proj, f"{where} shared_experts.down_proj"
        )
        if getattr(ffn, "use_native_mega_moe", False):
            _refuse(f"{where} runs vLLM's MegaMoE expert layout")
        found = vllm_moe_expert_descriptor(experts, f"{where} experts", text_config)
        if found["scheme"] != "16bit":
            _refuse(f"{where}'s routed experts are {found['scheme']} in vLLM")
        state_dict[f"{p}.mlp.experts.gate_up_proj"] = found["w13"].data
        state_dict[f"{p}.mlp.experts.down_proj"] = found["w2"].data

        engram = getattr(layer, "engram", None)
        if engram is not None:
            state_dict[f"{p}.engram.wkv.weight"] = _linear_weight(engram.wkv, f"{where} engram.wkv")
            put(f"{p}.engram.q_weight", engram.q_weight, f"{where} engram.q_weight")
            put(f"{p}.engram.k_weight", engram.k_weight, f"{where} engram.k_weight")
            table = engram.embed_tokens
            weight, scale = getattr(table, "weight", None), getattr(table, "weight_scale_inv", None)
            fp8 = getattr(torch, "float8_e4m3fn", None)
            e8m0 = getattr(torch, "float8_e8m0fnu", None)
            if not isinstance(weight, torch.Tensor) or weight.dtype != fp8 or not isinstance(scale, torch.Tensor) \
                or scale.dtype not in (torch.uint8, e8m0) or weight.device.type == "meta":
                _refuse(f"{where} engram table is {getattr(weight, 'dtype', None)} / {getattr(scale, 'dtype', None)}, "
                        "not vLLM's resident fp8 rows + e8m0 scales")
            if weight.shape[0] != getattr(table, "num_embeddings", weight.shape[0]) or \
                getattr(table, "_shared_memory", None) is not None:
                _refuse(f"{where} engram table is partitioned or held in shared host memory")
            # Raw e8m0 bytes are stored as uint8 by vLLM's loader: a dtype view, no copy.
            state_dict[f"model.engram_tables.{kk}.weight"] = weight.data
            state_dict[f"model.engram_tables.{kk}.weight_scale_inv"] = \
                scale.data.view(e8m0) if scale.dtype == torch.uint8 else scale.data
    pass
    put("model.norm.weight", text_model.norm.weight, "norm")
    state_dict["lm_head.weight"] = _float_tensor(getattr(_base(lm_head), "weight", None), "lm_head")[:vocab_size]

    # Every vLLM language-model parameter must back a training tensor.
    used = {t.untyped_storage().data_ptr() for t in state_dict.values()}
    unshared = [
        name for name, param in language_model.named_parameters()
        if param.numel() and param.untyped_storage().data_ptr() not in used
        and not name.startswith(("mtp.", "model.mtp."))
        and ".lora_" not in name and "lora_a_stacked" not in name and "lora_b_stacked" not in name
    ]
    if unshared:
        _refuse(f"{len(unshared)} vLLM tensors have no native DeepSeek-V4.1 name, e.g. {unshared[:5]}")
    token_map = getattr(getattr(text_model, "engram_hash", None), "token_map", None)
    if token_map is not None:
        state_dict._unsloth_dsv41_token_map = token_map
    return state_dict
pass


def _native_model_class_and_config(config):
    """Unsloth's native DeepSeek-V4.1 CausalLM (or transformers' own, when it ships one) and the config in its
    class: vLLM registers its own DeepseekV41Config with AutoConfig, which the native model does not accept."""
    from unsloth_zoo.temporary_patches.deepseek_v41 import _modeling
    modeling = _modeling()
    model_class = getattr(modeling, "DeepseekV41ForCausalLM", None) if modeling is not None else None
    if model_class is None:
        _refuse("no DeepSeek-V4.1 modeling code is loaded (import unsloth first)")
    config_class = getattr(model_class, "config_class", None)
    if config_class is not None and not isinstance(config, config_class):
        values = config.to_dict()
        values.pop("model_type", None)
        values.pop("transformers_version", None)
        native = config_class(**values)
        for key in ("_name_or_path", "name_or_path", "model_name"):
            if hasattr(config, key):
                try: setattr(native, key, getattr(config, key))
                except Exception: pass
        config = native
    return model_class, config
pass


def _set_param(model, name, tensor):
    owner_name, _, leaf = name.rpartition(".")
    owner = model.get_submodule(owner_name)
    owner._parameters[leaf] = torch.nn.Parameter(tensor, requires_grad = False)
pass


@torch.inference_mode()
def convert_deepseek_v41_vllm_to_huggingface(quant_state_dict, config, dtype = torch.bfloat16):
    """Build Unsloth's native DeepSeek-V4.1 text model on vLLM's tensors (views, no copies)."""
    # All Unsloth Zoo code licensed under LGPLv3
    from accelerate import init_empty_weights
    model_class, config = _native_model_class_and_config(config)
    with init_empty_weights(include_buffers = False):
        model = model_class._from_config(config, torch_dtype = dtype)
    tables = {k: v for k, v in quant_state_dict.items() if k.startswith("model.engram_tables.")}
    missing = []
    for name, param in list(model.named_parameters()):
        tensor = quant_state_dict.get(name, None)
        if tensor is None:
            missing.append(name)
            continue
        if tuple(tensor.shape) != tuple(param.shape):
            _refuse(f"{name} is {tuple(tensor.shape)} in vLLM, {tuple(param.shape)} in the native model")
        _set_param(model, name, tensor)
    if missing:
        _refuse(f"{len(missing)} native tensors have no vLLM counterpart, e.g. {missing[:5]}")
    used = {n for n, _ in model.named_parameters()}
    extra = sorted(set(quant_state_dict) - used - set(tables))
    if extra:
        _refuse(f"vLLM tensors {extra[:5]} have no place in the native model")

    # Parameter-free engram tables keep their rows in a store: bind it to vLLM's fp8 rows + scales.
    text_model = model.model
    for layer_id in getattr(text_model, "engram_tables", {}) or {}:
        weight = tables.get(f"model.engram_tables.{layer_id}.weight", None)
        scale = tables.get(f"model.engram_tables.{layer_id}.weight_scale_inv", None)
        if weight is None:
            _refuse(f"engram table {layer_id} is not in vLLM's model")
        table = text_model.engram_tables[layer_id]
        if any(True for _ in table.parameters(recurse = False)):
            continue  # Parameter-holding table, bound above
        text_model.set_engram_table(int(layer_id), weight, scale)
    token_map = getattr(quant_state_dict, "_unsloth_dsv41_token_map", None)
    if token_map is not None and hasattr(text_model, "bind_token_map"):
        text_model.bind_token_map(token_map.detach().cpu())

    device = next(iter(quant_state_dict.values())).device
    for module in model.modules():
        for key, buffer in list(module._buffers.items()):
            if isinstance(buffer, torch.Tensor) and buffer.device != device and buffer.device.type != "meta":
                module._buffers[key] = buffer.to(device)
    model.eval()
    model._unsloth_vllm_dtype = dtype
    # DeepseekV41RMSNorm applies weight.float() itself, so Unsloth's float32 norm upcast (a copy) is not needed.
    model._unsloth_vllm_shared_norms = True
    model._unsloth_vllm_nvfp4_layers = 0
    return model
pass
