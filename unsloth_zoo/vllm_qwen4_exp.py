# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""fast_inference for Qwen3.8-Flash-Next (model_type qwen4_exp).

vLLM >= 0.34 serves Qwen4Exp (Gated DeltaNet + QSA sparse attention + hyper-connections + a PLE n-gram table).
This module maps every vLLM tensor onto the transformers names as views of vLLM's storage, refuses what cannot
be shared, and carries the one vLLM 0.34.0 load fix the model needs.
"""

__all__ = [
    "is_qwen4_exp_config",
    "check_vllm_supports_qwen4_exp",
    "patch_vllm_qwen4_exp",
    "qwen4_exp_layer_templates",
    "qwen4_exp_empty_config_attrs",
    "extract_qwen4_exp_layer",
    "extract_qwen4_exp_model",
    "install_qwen4_exp_shared_tensors",
]

import functools

import torch

from .log import logger


def is_qwen4_exp_config(config):
    text_config = getattr(config, "text_config", None) or config
    model_type = str(getattr(text_config, "model_type", "") or getattr(config, "model_type", "") or "")
    return model_type.startswith("qwen4_exp")
pass


def _patch_gated_residual_padding(cls):
    """vLLM 0.34.0 builds the hyper-connection down + inject GEMM with a 16-row alignment pad registered as a
    third checkpoint shard (`_input_mix_padding`). No checkpoint ships it, so the unquantized load fails its
    weight tracking ('...input_mix_weight_down_block_inject.weight.2 not initialized'), and the pad rows are
    torch.empty. vLLM 0.34.1 zeroes the pad and drops it from the shard ids; do the same on older builds."""
    # All Unsloth Zoo code licensed under LGPLv3
    if getattr(cls.__init__, "_unsloth_hc_padding", False): return False
    original_init = cls.__init__

    @functools.wraps(original_init)
    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        linear = getattr(self, "input_mix_weight_down_block_inject", None)
        pad = int(getattr(self, "pad_size", 0) or 0)
        if linear is None or pad <= 0 or getattr(linear, "padding_size", 0): return
        shard_ids = tuple(getattr(linear, "shard_ids", ()))
        if len(shard_ids) == 3:
            linear.shard_ids = shard_ids[:2]
        weight = getattr(linear, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.shape[0] > pad:
            with torch.no_grad():
                weight.data[-pad:].zero_()
    __init__._unsloth_hc_padding = True
    cls.__init__ = __init__
    return True
pass


def _vllm_supported_architectures():
    try:
        from vllm.model_executor.models.registry import ModelRegistry
        return set(ModelRegistry.get_supported_archs())
    except Exception:
        return None
pass


def check_vllm_supports_qwen4_exp(config, supported = None):
    """Refuse before vLLM starts when it has no Qwen4Exp implementation (added in vLLM 0.34)."""
    # All Unsloth Zoo code licensed under LGPLv3
    if supported is None: supported = _vllm_supported_architectures()
    if supported is None: return
    archs = list(getattr(config, "architectures", None) or ["Qwen4ExpForConditionalGeneration"])
    missing = [a for a in archs if a not in supported]
    if missing:
        try:
            from importlib.metadata import version
            found = version("vllm")
        except Exception:
            found = "unknown"
        raise NotImplementedError(
            f"Unsloth: fast_inference = True needs vLLM's Qwen4Exp implementation ({', '.join(missing)}), "
            f"which vLLM {found} does not have; it arrived in vLLM 0.34. Upgrade vLLM (pip install --upgrade vllm), "
            "or load with fast_inference = False."
        )
pass


def _mark_stacked_moe_lora(module):
    """transformers keeps Qwen4Exp's routed experts stacked (gate_up_proj [E, 2I, H]); PEFT adapts them through
    target_parameters, which only vLLM's FusedMoE3DWithLoRA reads. vLLM picks it when the model class sets
    is_3d_moe_weight (as Unsloth does for Qwen3-MoE and DeepSeek-V2 / V3)."""
    # All Unsloth Zoo code licensed under LGPLv3
    marked = []
    for name in ("Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"):
        cls = getattr(module, name, None)
        if cls is not None and getattr(cls, "is_3d_moe_weight", False) is not True:
            cls.is_3d_moe_weight = True
            marked.append(f"{module.__name__}.{name}")
    return marked
pass


def patch_vllm_qwen4_exp(config = None, enable_lora = False):
    """Idempotent vLLM patches for Qwen4Exp. Returns the names of the classes patched."""
    # All Unsloth Zoo code licensed under LGPLv3
    if config is not None: check_vllm_supports_qwen4_exp(config)
    patched = []
    if enable_lora:
        for module_name in ("vllm.models.qwen4_exp.nvidia.model", "vllm.models.qwen4_exp.amd.model"):
            try:
                patched += _mark_stacked_moe_lora(__import__(module_name, fromlist = ["Qwen4ExpForCausalLM"]))
            except Exception:
                continue
    for module_name in ("vllm.models.qwen4_exp.nvidia.hyperconnection", "vllm.models.qwen4_exp.amd.hyperconnection"):
        try:
            module = __import__(module_name, fromlist = ["GatedResidual"])
        except Exception:
            continue
        cls = getattr(module, "GatedResidual", None)
        if cls is not None and _patch_gated_residual_padding(cls):
            patched.append(f"{module_name}.GatedResidual")
    if patched:
        logger.info(f"Unsloth: patched vLLM Qwen4Exp classes {patched}.")
    return patched
pass


# HF names of the Qwen4Exp tensors the generic layer loop does not know. Linears go through the standard Linear /
# Linear4bit rebuild, norms through the layernorm path (vLLM keeps the same module paths for every norm here), and
# the PLE depthwise conv weight is a bare parameter on the HF Conv1d (its groups / dilation come from the meta model).
# The n-gram table and the int hash buffers are installed after finalize: the table can live in pinned host memory
# (vLLM's PLE CPU offload), and finalize's .to(device) would copy it onto the GPU.
_QWEN4_EXP_LINEARS = (
    "attn_hyper_connection.input_mix_weight_down",
    "attn_hyper_connection.input_mix_weight_up",
    "attn_hyper_connection.block_inject_weight",
    "mlp_hyper_connection.input_mix_weight_down",
    "mlp_hyper_connection.input_mix_weight_up",
    "mlp_hyper_connection.block_inject_weight",
    "ple.key_proj",
    "ple.value_proj",
    "self_attn.indexer.index_qk_proj",
)
_QWEN4_EXP_NORMS = (
    "attn_hyper_connection.hc_norm",
    "mlp_hyper_connection.hc_norm",
    "ple.norm_key",
    "ple.norm_query",
    "ple.norm_conv",
    "self_attn.indexer.q_layernorm",
    "self_attn.indexer.k_layernorm",
)
_QWEN4_EXP_PARAMS = (
    "ple.conv1d.weight",
)
_QWEN4_EXP_TABLE = "ple.ple_embedding.ngram_embedding.weight"
_QWEN4_EXP_HASH_BUFFERS = (
    "ple.ple_embedding.layer_multipliers",
    "ple.ple_embedding.ngram_heads_vocab_sizes",
    "ple.ple_embedding.ngram_heads_offsets",
)


def qwen4_exp_layer_templates():
    """{category: set of templates} merged into empty_model.get_model_layer_config()."""
    # Qwen4ExpForConditionalGeneration nests the text model at model.language_model, Qwen4ExpForCausalLM at model
    prefixes = ("model.language_model.layers.{kk}.", "model.layers.{kk}.")
    return {
        "standard_layers": {p + x for p in prefixes for x in _QWEN4_EXP_LINEARS + _QWEN4_EXP_PARAMS},
        "layernorms": {p + x for p in prefixes for x in _QWEN4_EXP_NORMS},
    }
pass


def qwen4_exp_empty_config_attrs(text_config):
    """Shrink the placeholder model: the PLE table is sum(primes near ngram_vocab_size_base) rows (about 320M on the
    real model) and must not be allocated for the empty model. Shapes are restored from the meta model."""
    heads = max(1, (int(getattr(text_config, "ngram_size", 2) or 2) - 1) * int(getattr(text_config, "heads_per_ngram", 1) or 1))
    return {
        "ple_embed_dim": heads,
        "ngram_vocab_size_base": 3,
        "make_ngram_vocab_size_divisible_by": 1,
        "hc_lowrank": 1,
    }
pass


def _refuse(message):
    raise NotImplementedError(f"Unsloth: {message} Load with fast_inference = False.")
pass


def _store(state_dict, quant_state_dict, name, value):
    value = getattr(value, "data", value)
    state_dict[name] = value
    quant_state_dict[name] = value
pass


def _extract_gated_residual(hc, prefix, state_dict, quant_state_dict, get_state_dict):
    """One vLLM GatedResidual (hyper-connection) onto HF names. vLLM packs [input_mix_weight_down | block_inject_weight
    | 16-row pad] into one MergedColumnParallelLinear; the HF rows are views of its first two shards."""
    # All Unsloth Zoo code licensed under LGPLv3
    if hc is None: _refuse(f"vLLM's Qwen4Exp layer has no hyper-connection at {prefix}.")
    fused = getattr(hc, "input_mix_weight_down_block_inject", None)
    if fused is not None:
        base = getattr(fused, "base_layer", fused)
        sizes = list(getattr(base, "output_sizes", None) or [])
        lowrank, count = int(getattr(hc, "hc_lowrank", -1)), int(getattr(hc, "hc_count", -1))
        if len(sizes) < 2 or sizes[0] != lowrank or sizes[1] != count:
            _refuse(f"cannot split vLLM's {prefix}.input_mix_weight_down_block_inject (output sizes {sizes}).")
        get_state_dict(f"{prefix}.input_mix_weight_down", 0, state_dict, fused)
        get_state_dict(f"{prefix}.block_inject_weight",   1, state_dict, fused)
    else:
        get_state_dict(f"{prefix}.input_mix_weight_down", 0, state_dict, hc.input_mix_weight_down, slice_weights = False)
    get_state_dict(f"{prefix}.input_mix_weight_up", 0, state_dict, hc.input_mix_weight_up, slice_weights = False)
    # hc_norm comes through the layernorm templates (per-layer) or below (final mixer).
pass


def extract_qwen4_exp_layer(layer, prefix, state_dict, quant_state_dict, get_state_dict):
    """Hyper-connections, PLE and the QSA indexer of one vLLM Qwen4ExpDecoderLayer. `prefix` is the HF layer prefix
    (model.language_model.layers.{kk}). Attention, GDN and MoE go through the generic extractors."""
    # All Unsloth Zoo code licensed under LGPLv3
    for name in ("attn_hyper_connection", "mlp_hyper_connection"):
        _extract_gated_residual(getattr(layer, name, None), f"{prefix}.{name}", state_dict, quant_state_dict, get_state_dict)

    ple = getattr(layer, "ple", None)
    if ple is not None:
        kv_proj = getattr(ple, "kv_proj", None)
        if kv_proj is None: _refuse(f"cannot find vLLM's {prefix}.ple.kv_proj.")
        get_state_dict(f"{prefix}.ple.key_proj",   0, state_dict, kv_proj)
        get_state_dict(f"{prefix}.ple.value_proj", 1, state_dict, kv_proj)
        conv = getattr(ple, "conv1d", None)
        _store(state_dict, quant_state_dict, f"{prefix}.ple.conv1d.weight", conv.weight)
        embedding = getattr(getattr(ple, "ple_embedding", None), "ngram_embedding", None)
        table = getattr(embedding, "weight", None)
        if not isinstance(table, torch.Tensor):
            _refuse(f"cannot find vLLM's PLE n-gram table at {prefix}.ple.")
        if table.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            _refuse(f"vLLM stores the PLE n-gram table at {prefix}.ple in {table.dtype}; training needs it in 16-bit.")
        if int(getattr(embedding, "tp_size", 1) or 1) != 1:
            _refuse("vLLM shards the PLE n-gram table across tensor-parallel ranks; fast_inference needs it whole.")
        _store(state_dict, quant_state_dict, f"{prefix}.{_QWEN4_EXP_TABLE}", table)
        for buffer_name in _QWEN4_EXP_HASH_BUFFERS:
            owner, _, attr = f"{prefix}.{buffer_name}".rpartition(".")
            value = getattr(ple.ple_embedding, attr, None)
            if isinstance(value, torch.Tensor):
                _store(state_dict, quant_state_dict, f"{prefix}.{buffer_name}", value)

    indexer = getattr(getattr(layer, "self_attn", None), "indexer", None)
    if indexer is not None:
        index_qk = getattr(indexer, "index_qk_proj", None)
        if index_qk is not None:
            get_state_dict(f"{prefix}.self_attn.indexer.index_qk_proj", 0, state_dict, index_qk, slice_weights = False)
        else:
            # Without LoRA vLLM folds index_qk_proj into qkv_proj as a fourth shard.
            qkv = getattr(layer.self_attn, "qkv_proj", None)
            sizes = list(getattr(getattr(qkv, "base_layer", qkv), "output_sizes", None) or [])
            if len(sizes) != 4:
                _refuse(f"cannot find vLLM's {prefix}.self_attn.indexer.index_qk_proj (qkv_proj sizes {sizes}).")
            get_state_dict(f"{prefix}.self_attn.indexer.index_qk_proj", 3, state_dict, qkv)
pass


def extract_qwen4_exp_model(vllm_text_model, prefix, state_dict, quant_state_dict, get_state_dict):
    """Model-level Qwen4Exp tensors: the final hyper-connection mixer (Qwen4Exp has no final RMSNorm)."""
    # All Unsloth Zoo code licensed under LGPLv3
    mixer = getattr(vllm_text_model, "hyper_connection_mixer", None)
    if mixer is None: _refuse("vLLM's Qwen4Exp model has no hyper_connection_mixer (pipeline parallel is not supported).")
    _extract_gated_residual(mixer, f"{prefix}.hyper_connection_mixer", state_dict, quant_state_dict, get_state_dict)
    _store(state_dict, quant_state_dict, f"{prefix}.hyper_connection_mixer.hc_norm.weight", mixer.hc_norm.weight)
pass


def _storage_key(tensor):
    try:
        return (tensor.device.type, tensor.untyped_storage().data_ptr())
    except Exception:
        return None
pass


def _module_and_attr(root, dotted):
    parent_path, _, attr = dotted.rpartition(".")
    module = root
    for part in parent_path.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module, attr
pass


@torch.no_grad()
def install_qwen4_exp_shared_tensors(new_model, quant_state_dict, config):
    """After finalize: install the tensors finalize must not move or cast (the PLE n-gram table, which may live in
    pinned host memory, and GDN A_log, which vLLM keeps in float32), share the PLE hash buffers, then check every
    language-model parameter is a view of a vLLM tensor. Anything unshared is refused. Returns the number installed."""
    # All Unsloth Zoo code licensed under LGPLv3
    def unwrap(v): return getattr(v, "data", v)
    inner = getattr(new_model, "model", new_model)
    if hasattr(inner, "language_model"):
        text_prefix, language_model = "model.language_model.", inner.language_model
    else:
        text_prefix, language_model = "model.", inner
    installed = 0
    for key, value in quant_state_dict.items():
        value = unwrap(value)
        if not isinstance(value, torch.Tensor) or not key.startswith(text_prefix): continue
        is_table, is_buffer = key.endswith(_QWEN4_EXP_TABLE), key.endswith(_QWEN4_EXP_HASH_BUFFERS)
        if not (is_table or is_buffer or key.endswith(".linear_attn.A_log")): continue
        try:
            module, attr = _module_and_attr(new_model, key)
        except (AttributeError, IndexError):
            _refuse(f"the training model has no {key} for vLLM's tensor.")
        if is_buffer:
            current = module._buffers.get(attr)
            if current is None or tuple(current.shape) != tuple(value.shape) or \
                not torch.equal(current.to(value.device, value.dtype), value):
                _refuse(f"vLLM's PLE hash buffer {key} differs from transformers'; the rollout would hash n-grams differently.")
            module._buffers[attr] = value
            installed += 1
            continue
        if is_table:
            rows = int(getattr(module, "num_embeddings", value.shape[0]))
            if value.shape[0] < rows or value.shape[1] != int(getattr(module, "embedding_dim", value.shape[1])):
                _refuse(f"vLLM's PLE n-gram table {tuple(value.shape)} does not cover transformers' ({rows}, {module.embedding_dim}).")
            value = value[:rows]
        module._parameters[attr] = torch.nn.Parameter(value, requires_grad = False)
        installed += 1

    shared = {k for k in (_storage_key(unwrap(v)) for v in quant_state_dict.values() if isinstance(unwrap(v), torch.Tensor)) if k}
    roots = [(text_prefix, language_model), ("lm_head.", getattr(new_model, "lm_head", None))]
    unshared = []
    for root_prefix, root in roots:
        if root is None: continue
        for name, param in root.named_parameters():
            if _storage_key(unwrap(param)) not in shared:
                unshared.append(root_prefix + name)
    if unshared:
        _refuse(
            f"{len(unshared)} Qwen4Exp training tensors are not views of vLLM's weights (e.g. {sorted(unshared)[:6]}), "
            "so training and the vLLM rollout would drift apart."
        )
    return installed
pass
