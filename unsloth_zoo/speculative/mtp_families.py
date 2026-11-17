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

"""Per-family native MTP/NextN head data; every llama.cpp-drafted family shares eh_proj(cat[enorm(embed(x)), hnorm(h)])
-> target's decoder block(s) -> head norm -> lm_head; canonical names
heads.{i}.{eh_proj,enorm,hnorm,shared_head_norm} and heads.{i}.block.{j}.<layer names>."""

__all__ = [
    "MTPFamily",
    "MTP_FAMILIES",
    "get_mtp_family",
    "mtp_family_for_config",
]

import re
from dataclasses import dataclass

_HEAD_KEYS = ("eh_proj", "enorm", "hnorm")


@dataclass(frozen = True)
class MTPFamily:
    name: str
    model_types: tuple
    module: str                  # transformers.models.<module>.modeling_<module>
    layer_cls: str
    norm_cls: str
    rotary_cls: str = None       # None = NoPE attention
    # Checkpoint layout by style: layers = model.layers.{N+i}.* (DeepSeek / GLM), qwen = mtp.fc / mtp.layers.0.*,
    # nemotron = mtp.layers.{j}.* sub-blocks of one head.
    style: str = "layers"
    count_keys: tuple = ("num_nextn_predict_layers",)
    # "final" = target hidden after the final norm, "last_layer" = before it (step35 / mimo2).
    h_source: str = "final"
    # Recurrent state fed to the next step: "normed" (after the head norm) or "raw".
    feedback: str = "normed"
    chained: bool = False        # several heads, draft step k runs head k
    max_heads: int = 1           # what llama.cpp's graph accepts
    # Per block of one head: {config list attribute: value at that layer index}.
    blocks: tuple = ({},)
    head_norm: str = "rms"       # "rms" (family norm class) or "layer" (weight-only LayerNorm)
    shared_head_norm_name: str = "shared_head.norm"
    shared_head_head_name: str = "shared_head.head"
    # (checkpoint suffix regex, module suffix): renames the stock loader applies.
    renames: tuple = ()
    logit_scale_key: str = None
    mrope: bool = False          # rotary takes [3, B, S] position ids (text tokens: same position x3)
    rotary_layer_type: bool = False  # rotary(x, position_ids, layer_type) (per layer-type rope)
    config_overrides: tuple = ()     # (attribute, value) set on the head config
    # Per-layer head-count override for sliding MTP layers (Step's per_layer_config).
    sliding_heads_attr: str = None
    # llama.cpp status from drafting tiny random models via convert_hf_to_gguf.py + llama-server draft-mtp: drafts (our
    # drafts match), partial, no_graph (no MTP graph or converter drops the head), unverified.
    llama_cpp: str = "drafts"

    def num_heads(self, config):
        for key in self.count_keys:
            value = getattr(config, key, None)
            if value:
                return int(value)
        return 0

    # ---- checkpoint naming ----------------------------------------------------------------

    def mtp_name_re(self, num_hidden_layers):
        if self.style == "layers":
            return re.compile(r"^((?:(?:model|language_model)\.)*)layers\.(\d+)\.(.+)$")
        return re.compile(r"^((?:(?:model|language_model)\.)*mtp\.)(.+)$")

    def is_mtp_name(self, name, num_hidden_layers):
        m = self.mtp_name_re(num_hidden_layers).match(name)
        if m is None:
            return False
        if self.style == "layers":
            return int(m.group(2)) >= num_hidden_layers
        return True

    def from_ckpt(self, name, num_hidden_layers, num_blocks = 1):
        """(prefix, canonical name) for one checkpoint tensor, or None if it is not MTP."""
        m = self.mtp_name_re(num_hidden_layers).match(name)
        if m is None:
            return None
        if self.style == "layers":
            prefix, layer, rest = m.group(1), int(m.group(2)), m.group(3)
            if layer < num_hidden_layers:
                return None
            i = layer - num_hidden_layers
            if rest.split(".")[0] in _HEAD_KEYS:
                return prefix, f"heads.{i}.{rest}"
            for own, canon in ((self.shared_head_norm_name, "shared_head_norm"),
                               (self.shared_head_head_name, "shared_head_head"),
                               ("embed_tokens", "embed_tokens")):
                if rest.startswith(own + "."):
                    return prefix, f"heads.{i}.{canon}{rest[len(own):]}"
            return prefix, f"heads.{i}.block.0.{self._rename_in(rest)}"
        prefix, rest = m.group(1), m.group(2)
        if self.style == "qwen":
            for own, canon in (("fc", "eh_proj"), ("pre_fc_norm_embedding", "enorm"),
                               ("pre_fc_norm_hidden", "hnorm"), ("norm", "shared_head_norm")):
                if rest.startswith(own + "."):
                    return prefix, f"heads.0.{canon}{rest[len(own):]}"
            lm = re.match(r"^layers\.(\d+)\.(.+)$", rest)
            if lm:
                return prefix, f"heads.0.block.{lm.group(1)}.{self._rename_in(lm.group(2))}"
            return prefix, f"heads.0.{rest}"
        if self.style == "nemotron":
            lm = re.match(r"^layers\.(\d+)\.(.+)$", rest)
            if lm is None:
                return prefix, f"heads.0.{rest}"
            j, sub = int(lm.group(1)), lm.group(2)
            if sub.split(".")[0] in _HEAD_KEYS:
                return prefix, f"heads.0.{sub}"
            if sub.startswith("final_layernorm."):
                return prefix, f"heads.0.shared_head_norm{sub[len('final_layernorm'):]}"
            return prefix, f"heads.0.block.{j}.{self._rename_in(sub)}"
        raise ValueError(f"Unsloth: unknown MTP naming style {self.style!r}.")

    def to_ckpt(self, canon, prefix, num_hidden_layers):
        """Inverse of from_ckpt for module-layout names (experts already unpacked)."""
        m = re.match(r"^heads\.(\d+)\.(.+)$", canon)
        if m is None:
            raise ValueError(f"Unsloth: not a canonical MTP name: {canon}")
        i, rest = int(m.group(1)), m.group(2)
        bm = re.match(r"^block\.(\d+)\.(.+)$", rest)
        if self.style == "layers":
            base = f"{prefix}layers.{num_hidden_layers + i}."
            if bm:
                return base + self._rename_out(bm.group(2))
            for canon_key, own in (("shared_head_norm", self.shared_head_norm_name),
                                   ("shared_head_head", self.shared_head_head_name)):
                if rest.startswith(canon_key + "."):
                    return base + own + rest[len(canon_key):]
            return base + rest
        if self.style == "qwen":
            if bm:
                return f"{prefix}layers.{bm.group(1)}.{self._rename_out(bm.group(2))}"
            for canon_key, own in (("eh_proj", "fc"), ("enorm", "pre_fc_norm_embedding"),
                                   ("hnorm", "pre_fc_norm_hidden"), ("shared_head_norm", "norm")):
                if rest.startswith(canon_key + "."):
                    return prefix + own + rest[len(canon_key):]
            return prefix + rest
        if self.style == "nemotron":
            if bm:
                return f"{prefix}layers.{bm.group(1)}.{self._rename_out(bm.group(2))}"
            if rest.startswith("shared_head_norm."):
                last = len(self.blocks) - 1
                return f"{prefix}layers.{last}.final_layernorm{rest[len('shared_head_norm'):]}"
            return f"{prefix}layers.0.{rest}"
        raise ValueError(f"Unsloth: unknown MTP naming style {self.style!r}.")

    def default_prefix(self):
        return "model." if self.style == "layers" else "mtp."

    def _rename_in(self, name):
        for pattern, target in self.renames:
            name = re.sub(pattern, target, name)
        return name

    def _rename_out(self, name):
        for pattern, target in self.renames:
            # Renames are literal suffix swaps written as escaped regexes.
            name = name.replace(target, pattern.replace("\\", ""))
        return name

    # ---- head construction ----------------------------------------------------------------

    def head_config(self, config):
        """Deep copy of the target text config whose per-layer lists cover the MTP layers."""
        import copy
        n = int(config.num_hidden_layers)
        heads = max(1, self.num_heads(config))
        head_config = copy.deepcopy(config)
        total = n + heads * len(self.blocks)
        # Per-layer lists (layer_types, indexer_types, swiglu limits, ...) must cover the MTP layers: transformers
        # truncates them to num_hidden_layers, so raw config.json values past it win, then the family's, then a default.
        raw = _raw_config_lists(config)
        for attr, values in list(vars(head_config).items()):
            if not isinstance(values, (list, tuple)) or len(values) != n or not values:
                continue
            values = list(values)
            # Layer 0 is never a "shared" indexer / special layer; other lists default to the last layer.
            default = values[0] if attr == "indexer_types" else values[-1]
            stored = raw.get(attr)
            # transformers keeps the MTP layers' entries separately on some configs (Step, EXAONE).
            mtp_values = getattr(config, f"mtp_{attr}", None)
            for i in range(heads):
                for j, spec in enumerate(self.blocks):
                    idx = self.block_layer_idx(n, i, j)
                    if isinstance(mtp_values, (list, tuple)) and len(mtp_values) > idx - n:
                        values.append(mtp_values[idx - n])
                    elif isinstance(stored, list) and len(stored) > idx:
                        values.append(stored[idx])
                    elif attr in spec:
                        values.append(spec[attr])
                    else:
                        values.append(default)
            setattr(head_config, attr, values)
        head_config.num_hidden_layers = total
        for attr, value in self.config_overrides:
            setattr(head_config, attr, value)
        if self.sliding_heads_attr and getattr(head_config, self.sliding_heads_attr, None):
            overrides = {int(k): v for k, v in (head_config.to_dict().get("per_layer_config") or {}).items()}
            types = getattr(head_config, "layer_types", None) or []
            for idx in range(n, total):
                if idx < len(types) and types[idx] == "sliding_attention":
                    overrides[idx] = {"num_attention_heads": getattr(head_config, self.sliding_heads_attr)}
            head_config.per_layer_config = overrides
        head_config._attn_implementation = "sdpa"
        if hasattr(head_config, "use_cache"):
            head_config.use_cache = True
        return head_config

    def block_layer_idx(self, num_hidden_layers, head, block):
        return num_hidden_layers + head * len(self.blocks) + block


def _raw_config_lists(config):
    """List-valued fields of the checkpoint's own config.json (local folder or HF cache only)."""
    import json
    import os
    name = getattr(config, "_name_or_path", None) or ""
    path = None
    if name and os.path.isfile(os.path.join(name, "config.json")):
        path = os.path.join(name, "config.json")
    elif name:
        try:
            from huggingface_hub import try_to_load_from_cache
            cached = try_to_load_from_cache(name, "config.json")
            path = cached if isinstance(cached, str) else None
        except Exception:
            path = None
    if path is None:
        return {}
    try:
        with open(path, "r", encoding = "utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    holder = data.get("text_config") if isinstance(data.get("text_config"), dict) else data
    return {k: v for k, v in holder.items() if isinstance(v, list)}


def _nemotron_blocks(config):
    pattern = getattr(config, "mtp_hybrid_override_pattern", None) or "*E"
    kinds = {"*": "full_attention", "E": "moe", "-": "mlp"}
    if any(c not in kinds for c in pattern):
        raise NotImplementedError(f"Unsloth: Nemotron-H MTP pattern {pattern!r} is not supported.")
    return tuple({"layer_types": kinds[c]} for c in pattern)


MTP_FAMILIES = {}


def _register(family):
    for model_type in family.model_types:
        MTP_FAMILIES[model_type] = family
    return family


_register(MTPFamily(
    name = "glm4_moe", model_types = ("glm4_moe",), module = "glm4_moe",
    layer_cls = "Glm4MoeDecoderLayer", norm_cls = "Glm4MoeRMSNorm", rotary_cls = "Glm4MoeRotaryEmbedding",
))
_register(MTPFamily(
    name = "glm4_moe_lite", model_types = ("glm4_moe_lite",), module = "glm4_moe_lite",
    layer_cls = "Glm4MoeLiteDecoderLayer", norm_cls = "Glm4MoeLiteRMSNorm",
    rotary_cls = "Glm4MoeLiteRotaryEmbedding", blocks = ({"mlp_layer_types": "sparse"},),
))
_register(MTPFamily(
    name = "deepseek_v3", model_types = ("deepseek_v3",), module = "deepseek_v3",
    layer_cls = "DeepseekV3DecoderLayer", norm_cls = "DeepseekV3RMSNorm", rotary_cls = "DeepseekV3RotaryEmbedding",
    llama_cpp = "no_graph",
))
_register(MTPFamily(
    name = "deepseek_v32", model_types = ("deepseek_v32",), module = "deepseek_v32",
    layer_cls = "DeepseekV32DecoderLayer", norm_cls = "DeepseekV32RMSNorm",
    rotary_cls = "DeepseekV32RotaryEmbedding",
    blocks = ({"layer_types": "deepseek_sparse_attention", "mlp_layer_types": "sparse"},),
    config_overrides = (("index_topk", 1 << 30),),
    llama_cpp = "unverified",
))
_register(MTPFamily(
    name = "glm_moe_dsa", model_types = ("glm_moe_dsa",), module = "glm_moe_dsa",
    layer_cls = "GlmMoeDsaDecoderLayer", norm_cls = "GlmMoeDsaRMSNorm", rotary_cls = "GlmMoeDsaRotaryEmbedding",
    blocks = ({"layer_types": "deepseek_sparse_attention", "mlp_layer_types": "sparse"},),
    config_overrides = (("index_topk", 1 << 30),),
))
_register(MTPFamily(
    name = "hy_v3", model_types = ("hy_v3",), module = "hy_v3",
    layer_cls = "HYV3DecoderLayer", norm_cls = "HYV3RMSNorm", rotary_cls = "HYV3RotaryEmbedding",
    blocks = ({"mlp_layer_types": "sparse"},), shared_head_norm_name = "final_layernorm",
    renames = ((r"mlp\.router\.gate\.", "mlp.gate."), (r"mlp\.expert_bias", "mlp.e_score_correction_bias"),
               (r"mlp\.shared_mlp\.", "mlp.shared_experts.")),
))
# Qwen3.5/3.6 train through Qwen35MTPHead (mtp.py); registered here for naming and as the reference the generic engine is tested against.
_register(MTPFamily(
    name = "qwen3_5", model_types = ("qwen3_5_text",), module = "qwen3_5",
    layer_cls = "Qwen3_5DecoderLayer", norm_cls = "Qwen3_5RMSNorm", rotary_cls = "Qwen3_5TextRotaryEmbedding",
    style = "qwen", count_keys = ("mtp_num_hidden_layers",), blocks = ({"layer_types": "full_attention"},),
    mrope = True,
))
_register(MTPFamily(
    name = "qwen3_5_moe", model_types = ("qwen3_5_moe_text",), module = "qwen3_5_moe",
    layer_cls = "Qwen3_5MoeDecoderLayer", norm_cls = "Qwen3_5MoeRMSNorm", rotary_cls = "Qwen3_5MoeTextRotaryEmbedding",
    style = "qwen", count_keys = ("mtp_num_hidden_layers",), blocks = ({"layer_types": "full_attention"},),
    mrope = True,
))
_register(MTPFamily(
    name = "qwen3_next", model_types = ("qwen3_next",), module = "qwen3_next",
    layer_cls = "Qwen3NextDecoderLayer", norm_cls = "Qwen3NextRMSNorm", rotary_cls = "Qwen3NextRotaryEmbedding",
    style = "qwen", count_keys = ("num_nextn_predict_layers", "mtp_num_hidden_layers"),
    blocks = ({"layer_types": "full_attention"},),
    llama_cpp = "partial",
))
_register(MTPFamily(
    name = "exaone_moe", model_types = ("exaone_moe",), module = "exaone_moe",
    layer_cls = "ExaoneMoeDecoderLayer", norm_cls = "ExaoneMoeRMSNorm", rotary_cls = "ExaoneMoeRotaryEmbedding",
    style = "qwen", blocks = ({"layer_types": "full_attention", "mlp_layer_types": "dense"},),
    renames = ((r"mlp\.e_score_correction_bias", "mlp.gate.e_score_correction_bias"),),
    llama_cpp = "no_graph",
))
# Step-3.5/3.7: three chained sliding-window heads with dense MLPs; trunk feeds and feeds back the pre-norm residual
# (llama.cpp step35.cpp, vLLM step3p5_mtp.py).
_register(MTPFamily(
    name = "step3p5", model_types = ("step3p5",), module = "step3p7",
    layer_cls = "Step3p7DecoderLayer", norm_cls = "Step3p7RMSNorm", rotary_cls = "Step3p7RotaryEmbedding",
    h_source = "last_layer", feedback = "raw", chained = True, max_heads = 3,
    blocks = ({"layer_types": "sliding_attention", "mlp_layer_types": "dense",
               "swiglu_limits": 0.0, "swiglu_limits_shared": 0.0},),
    shared_head_norm_name = "transformer.shared_head.norm", shared_head_head_name = "transformer.shared_head.output",
    rotary_layer_type = True, sliding_heads_attr = "num_sliding_attention_heads",
))
_register(MTPFamily(
    name = "nemotron_h", model_types = ("nemotron_h",), module = "nemotron_h",
    layer_cls = "NemotronHBlock", norm_cls = "NemotronHRMSNorm", rotary_cls = None,
    style = "nemotron", blocks = ({"layer_types": "full_attention"}, {"layer_types": "moe"}),
    # NVIDIA/vLLM use RMSNorm; llama.cpp currently uses LayerNorm here.
    llama_cpp = "partial",
))


def get_mtp_family(model_type):
    return MTP_FAMILIES.get(model_type)


def mtp_family_for_config(config):
    """Family for a (text) config, with per-config block layouts resolved."""
    model_type = getattr(config, "model_type", None)
    family = MTP_FAMILIES.get(model_type)
    if family is None:
        return None
    if family.name == "nemotron_h":
        import dataclasses
        family = dataclasses.replace(family, blocks = _nemotron_blocks(config))
    if family.name == "exaone_moe":
        types = getattr(config, "mtp_layer_types", None)
        if types:
            import dataclasses
            family = dataclasses.replace(family, blocks = ({"layer_types": types[0], "mlp_layer_types": "dense"},))
    return family
