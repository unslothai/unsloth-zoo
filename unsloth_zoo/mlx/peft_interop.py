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


# ---------------------------------------------------------------------------
# PEFT <-> MLX LoRA adapter interop. The two formats encode the same math in
# different layouts:
#
#     peft {p}.lora_A.weight [r, in]   ==  mlx {p}.lora_a [in, r], transposed
#     peft {p}.lora_B.weight [out, r]  ==  mlx {p}.lora_b [r, out], transposed
#     effective scale = lora_alpha / r     (rsLoRA: lora_alpha / sqrt(r))
#
# Conversion is rename + transpose only, so a round trip is bitwise-identical
# for the supported subset: linear LoRA, linear DoRA, and full-module state
# (modules_to_save snapshots and peft's auto-saved embeddings) tracked through
# origin-tagged full_state_modules maps. Everything else is refused by name.
# Tensor IO prefers mlx.core for native bfloat16, falling back to
# safetensors.torch off Apple hardware.
# ---------------------------------------------------------------------------

import json
import math
import os
import re
import shutil
import uuid

PEFT_WEIGHTS_FILE = "adapter_model.safetensors"
MLX_WEIGHTS_FILE = "adapters.safetensors"
_PEFT_PREFIX = "base_model.model."

# peft resolves rank_pattern / alpha_pattern by ordered first-match, and a key
# may be a regex fragment; mirror its expression exactly.
def _resolve_pattern(patterns, module_path):
    if not patterns:
        return None
    for key, value in patterns.items():
        try:
            if re.match(rf"(.*\.)?({key})$", module_path):
                return value
        except re.error:
            # A malformed regex key cannot silently mean "no override".
            raise ValueError(
                f"Unsloth MLX: PEFT pattern key {key!r} is not a valid "
                f"regular expression; fix rank_pattern/alpha_pattern in "
                f"adapter_config.json."
            )
    return None


def _tensor_backend():
    """Return (name, load_file, save_file, transpose) for the host."""
    try:
        import mlx.core as mx

        def _load(path):
            return dict(mx.load(str(path)))

        def _save(path, tensors):
            mx.save_safetensors(str(path), tensors)

        return "mlx", _load, _save, lambda t: t.T
    except ImportError:
        pass
    try:
        import torch  # noqa: F401
        from safetensors.torch import load_file, save_file

        def _transpose(t):
            return t.transpose(0, 1).contiguous()

        def _save(path, tensors):
            # safetensors.torch.save_file takes (tensors, filename); normalize
            # to the (path, tensors) order the mlx branch uses.
            save_file(tensors, str(path))

        return "torch", load_file, _save, _transpose
    except ImportError:
        raise ImportError(
            "Unsloth MLX: adapter conversion needs mlx (Apple hosts) or "
            "torch + safetensors (other hosts) for bf16-safe tensor IO; "
            "the safetensors numpy backend cannot represent bfloat16."
        )


def detect_adapter_format(path):
    """Return 'peft' or 'mlx' for an adapter directory; refuse ambiguity."""
    has_peft = os.path.exists(os.path.join(path, PEFT_WEIGHTS_FILE))
    has_mlx = os.path.exists(os.path.join(path, MLX_WEIGHTS_FILE))
    if has_peft and has_mlx:
        raise ValueError(
            f"Unsloth MLX: {path!r} contains both {PEFT_WEIGHTS_FILE} and "
            f"{MLX_WEIGHTS_FILE}; cannot tell which adapter is current. "
            "Remove one of the weight files."
        )
    if has_peft:
        return "peft"
    if has_mlx:
        return "mlx"
    raise FileNotFoundError(
        f"Unsloth MLX: no adapter weights ({PEFT_WEIGHTS_FILE} or "
        f"{MLX_WEIGHTS_FILE}) found in {path!r}."
    )


# Config keys this converter implements or may safely ignore. Anything else set
# to a non-empty value is refused by name so a feature is never silently dropped
# (an aLoRA adapter would otherwise become an always-on LoRA).
_PEFT_HANDLED_KEYS = {
    "peft_type", "r", "lora_alpha", "use_rslora", "lora_dropout",
    "target_modules", "base_model_name_or_path", "revision",
    "rank_pattern", "alpha_pattern", "layers_to_transform", "layers_pattern",
    # Linear DoRA converts (magnitude <-> m); embedding DoRA refuses later.
    "use_dora",
    # Full-weight snapshots convert; peft strips the modules_to_save infix at
    # save, so detection is config-driven.
    "modules_to_save",
}
# peft applies DoRA dropout inside the correction term, mlx-lm scales
# base-plus-dropped-update together: continued training would diverge.
def _reject_dora_dropout(cfg):
    if cfg.get("use_dora") and float(cfg.get("lora_dropout") or 0.0) > 0:
        raise ValueError(
            "Unsloth MLX: use_dora with nonzero lora_dropout cannot be "
            "converted; peft and mlx-lm apply DoRA dropout with different "
            "training-mode formulas. Zero the dropout before converting."
        )
_PEFT_NEUTRAL_KEYS = {
    # task_type is checked explicitly above; listed here so the unknown-field
    # sweep does not reject the CAUSAL_LM value that check accepts.
    "task_type", "inference_mode", "init_lora_weights", "peft_version",
    "auto_mapping", "fan_in_fan_out", "exclude_modules",
    # EVA redistributes ranks/alphas through the supported pattern fields and
    # leaves base weights alone; peft serializes eva_config even at defaults.
    "eva_config",
    # Only meaningful when use_qalora is set, which is rejected below.
    "qalora_group_size",
    # megatron_core is set by default; only megatron_config makes it megatron.
    "megatron_core",
}
_PEFT_REJECTED_KEYS = {
    "target_parameters": "expert-parameter (MoE) LoRA is not supported on "
                         "the MLX backend yet",
    "alora_invocation_tokens": "aLoRA adapters activate conditionally; "
                               "converting would silently make them "
                               "always-on",
    "trainable_token_indices": "trainable token indices are not supported "
                               "on the MLX backend",
    "megatron_config": "megatron-format adapters are not supported on the "
                       "MLX backend",
    # Replicated/reordered stacks change which base layer a path denotes, so
    # strict binding would attach weights to the wrong layers.
    "layer_replication": "layer_replication adapters remap base layers and "
                         "cannot bind onto the unmodified model",
    # QALoRA pools/reshapes A-side inputs; not plain LoRA factors.
    "use_qalora": "QALoRA adapters are not plain LoRA and are not supported "
                  "on the MLX backend",
    # Adds a trainable bias to lora_B; mlx-lm LoRA layers have no such term.
    "lora_bias": "lora_bias (trainable LoRA bias) is not supported on the "
                 "MLX backend",
}


def _is_empty(value):
    return value in (None, False, 0, 0.0, "", "none") or value == [] or value == {}


def normalize_peft_adapter_config(cfg, adapter_dir=None):
    """Validate a PEFT LoraConfig dict for MLX import; return a copy with
    zoo-loader key aliases (``base_model_revision``) filled in. Raises
    ValueError naming the first unsupported field, so an adapter is refused
    before the expensive base-model load.
    """
    # A dict straight from LoraConfig.to_dict() carries peft's enums, whose
    # str() is "PeftType.LORA"; adapter JSON carries the bare value.
    peft_type = cfg.get("peft_type", "")
    if str(getattr(peft_type, "value", peft_type)).upper() != "LORA":
        raise ValueError(
            f"Unsloth MLX: only peft_type='LORA' adapters can be imported; "
            f"got {cfg.get('peft_type')!r}."
        )
    # The MLX backend builds a causal LM, so another task's adapter would
    # attach its backbone factors and then answer with vocabulary logits.
    task_type = cfg.get("task_type")
    task_type = getattr(task_type, "value", task_type)
    if task_type is not None and str(task_type).upper() != "CAUSAL_LM":
        raise ValueError(
            f"Unsloth MLX: task_type={task_type!r} adapters cannot be "
            "imported; the MLX backend serves CAUSAL_LM only."
        )
    _reject_dora_dropout(cfg)
    if cfg.get("bias") not in (None, "none"):
        raise ValueError(
            "Unsloth MLX: PEFT LoRA bias terms (bias="
            f"{cfg.get('bias')!r}) are not supported on the MLX backend."
        )
    for key, reason in _PEFT_REJECTED_KEYS.items():
        if not _is_empty(cfg.get(key)):
            raise ValueError(
                f"Unsloth MLX: cannot import this PEFT adapter: {reason} "
                f"(adapter_config.json field {key!r})."
            )
    # PiSSA/OLoRA/CorDA/LoftQ factors only reproduce the trained model on the
    # base they mutated at init, which this importer never applies. peft can
    # convert them to plain LoRA at save time; require that.
    init_mode = cfg.get("init_lora_weights")
    # peft matches these case-insensitively, dispatching corda by prefix.
    init_norm = init_mode.lower() if isinstance(init_mode, str) else ""
    if not _is_empty(cfg.get("loftq_config")) or init_norm == "loftq":
        _named = "LoftQ"
    elif init_norm.startswith(("pissa", "corda")) or init_norm == "olora":
        _named = f"init_lora_weights={init_mode!r}"
    else:
        _named = None
    if _named:
        raise ValueError(
            f"Unsloth MLX: {_named} adapters assume a base mutated at "
            "initialization; re-save converted to plain LoRA."
        )
    known = _PEFT_HANDLED_KEYS | _PEFT_NEUTRAL_KEYS | set(_PEFT_REJECTED_KEYS)
    known |= {"bias", "loftq_config"}
    for key, value in cfg.items():
        if key in known or _is_empty(value):
            continue
        raise ValueError(
            f"Unsloth MLX: PEFT adapter_config.json field {key!r}="
            f"{value!r} is not understood by the MLX importer; refusing to "
            "import an adapter whose semantics would be dropped."
        )
    normalized = dict(cfg)
    if cfg.get("revision") and not normalized.get("base_model_revision"):
        normalized["base_model_revision"] = cfg["revision"]
    normalized["_unsloth_peft_import"] = True
    if adapter_dir is not None:
        index_file = os.path.join(adapter_dir, "adapter_model.safetensors.index.json")
        if os.path.exists(index_file):
            raise ValueError(
                "Unsloth MLX: sharded PEFT adapters "
                "(adapter_model.safetensors.index.json) are not supported."
            )
    return normalized


def _effective_scale(cfg, module_path, rank):
    alpha = _resolve_pattern(cfg.get("alpha_pattern"), module_path)
    if alpha is None:
        alpha = cfg.get("lora_alpha", rank)
    denom = math.sqrt(rank) if cfg.get("use_rslora") else rank
    return float(alpha) / float(denom)


def group_peft_lora_pairs(tensors, cfg=None):
    """Split raw PEFT tensors into (pairs, full_state, rejected).

    ``pairs`` maps module paths to {'A','B'[,'M']}. ``full_state`` maps to
    {tensor_name: tensor} for modules_to_save snapshots (peft strips the infix
    at save, so detection is config-driven) and auto-saved embedding weights.
    The wrapper prefix is stripped exactly once; keys without it are rejected.
    """
    cfg = cfg or {}
    m2s_paths = set(cfg.get("modules_to_save") or [])

    def _m2s_hit(cand):
        # peft selects with a RAW key.endswith(name) — no dot boundary, so
        # ["head"] wraps lm_head — and a composite module's snapshot arrives as
        # submodule keys, so a name may suffix-match a dotted ancestor too.
        for m2s in m2s_paths:
            if cand.endswith(m2s):
                return m2s
            for j, ch in enumerate(cand):
                if ch == "." and cand[:j].endswith(m2s):
                    return m2s
        return None

    pairs = {}
    full_state = {}
    rejected = {}
    for key, tensor in tensors.items():
        if re.match(
            rf"^{re.escape(_PEFT_PREFIX)}(.+)\.lora_embedding_(A|B)$", key
        ):
            rejected[key] = (
                "embedding LoRA, which this converter does not support; "
                "retrain without the embedding in target_modules"
            )
            continue
        mag = re.match(
            rf"^{re.escape(_PEFT_PREFIX)}(.+)\.lora_magnitude_vector(?:\.weight)?$",
            key,
        )
        if mag is not None:
            pairs.setdefault(mag.group(1), {})["M"] = tensor
            continue
        m = re.match(
            rf"^{re.escape(_PEFT_PREFIX)}(.+)\.lora_(A|B)\.weight$", key
        )
        if m is not None:
            pairs.setdefault(m.group(1), {})[m.group(2)] = tensor
            continue
        fs = re.match(
            rf"^{re.escape(_PEFT_PREFIX)}(.+)\.(weight|bias)$", key
        )
        if fs is not None:
            fs_path, tname = fs.group(1), fs.group(2)
            if fs_path.endswith(".base_layer"):
                # peft saves it under the wrapper's inner path; normalize to
                # the module path the MLX tree uses.
                fs_path = fs_path[: -len(".base_layer")]
                if not _is_auto_saved_module_path(fs_path):
                    # Every LoRA target has a .base_layer, so without this any
                    # of them could replace its own base weights.
                    rejected[key] = (
                        "auto-saved base state on a module that is not the "
                        "input embedding or output head"
                    )
                    continue
                full_state.setdefault(fs_path, {})[tname] = tensor
                full_state[fs_path]["__origin__"] = "embedding_auto"
                continue
            # peft's embedding auto-save copies raw wrapper internals:
            # `{p}.modules_to_save[.name].{t}` duplicates the snapshot and
            # `{p}.original_module[.name].{t}` is the pristine base. Fold the
            # former, drop the latter — before the general modules_to_save
            # match, which would adopt them as real submodules.
            for _marker in (".modules_to_save", ".original_module"):
                _idx = fs_path.find(_marker)
                if _idx > 0 and _m2s_hit(fs_path[:_idx]) is not None:
                    _rem = fs_path[_idx + len(_marker):].lstrip(".")
                    if _rem and "." in _rem:
                        # Cannot attribute a composite child to one snapshot
                        # tensor; surface it rather than fold it wrongly.
                        rejected[key] = (
                            "wrapper-internal duplicate of a composite "
                            "modules_to_save entry cannot be attributed"
                        )
                        fs_path = None
                        break
                    _base_path = fs_path[:_idx]
                    if _marker == ".modules_to_save":
                        # The duplicate counts as a snapshot; the pristine
                        # copy does not — alone it means state was lost.
                        _slot = full_state.setdefault(_base_path, {})
                        if tname in _slot and not _tensors_equal(
                            _slot[tname], tensor
                        ):
                            rejected[key] = (
                                "disagrees with the module's canonical "
                                "snapshot tensor; the adapter file is "
                                "inconsistent"
                            )
                        else:
                            _slot.setdefault(tname, tensor)
                            _slot["__origin__"] = "modules_to_save"
                    fs_path = None
                    break
            if fs_path is None:
                continue
            if _m2s_hit(fs_path) is not None:
                _slot = full_state.setdefault(fs_path, {})
                if tname in _slot and not _tensors_equal(_slot[tname], tensor):
                    rejected[key] = (
                        "disagrees with the module's duplicate snapshot "
                        "tensor; the adapter file is inconsistent"
                    )
                    continue
                _slot[tname] = tensor
                _slot["__origin__"] = "modules_to_save"
                continue
            if _is_auto_saved_module_path(fs_path):
                # Auto-saved embedding/head state; biased heads include bias.
                full_state.setdefault(fs_path, {})[tname] = tensor
                full_state[fs_path]["__origin__"] = "embedding_auto"
                continue
        rejected[key] = "not a recognized PEFT adapter key"
        continue
    has_mag = {p for p, v in pairs.items() if "M" in v}
    for path, pair in pairs.items():
        if "A" not in pair or "B" not in pair:
            rejected[f"{_PEFT_PREFIX}{path}.lora_*"] = (
                "incomplete lora_A/lora_B pair"
            )
    # use_dora is global: a partial magnitude set means file and config
    # disagree about what the adapter is.
    _linear_pairs = {p for p, v in pairs.items() if "A" in v}
    if has_mag and has_mag != _linear_pairs:
        rejected[f"{_PEFT_PREFIX}<mixed>.lora_magnitude_vector"] = (
            "magnitudes present on only some modules; peft DoRA is all-or-"
            "nothing"
        )
        pairs = {}
    for _p, _v in pairs.items():
        _bad = sorted(
            k for k in ("A", "B", "M")
            if k in _v and not _is_float_dtype(_v[k])
        )
        if _bad:
            rejected[f"{_PEFT_PREFIX}{_p}.lora_*"] = (
                f"non-floating adapter factor dtype(s) "
                f"({', '.join(str(_v[k].dtype) for k in _bad)}); LoRA "
                "factors must be floating point"
            )
    pairs = {p: v for p, v in pairs.items() if "A" in v and "B" in v}
    # An entry with no tensors would silently downgrade to plain LoRA. Credit
    # every declaration a snapshot satisfies: overlapping names like
    # ["head", "lm_head"] select the same module.
    _m2s_snapshot_paths = [
        p for p, v in full_state.items()
        if v.get("__origin__") == "modules_to_save"
    ]

    def _name_credited(name):
        for cand in _m2s_snapshot_paths:
            if cand.endswith(name):
                return True
            for j, ch in enumerate(cand):
                if ch == "." and cand[:j].endswith(name):
                    return True
        return False

    for _name in sorted(m2s_paths):
        if not _name_credited(_name):
            rejected[f"modules_to_save:{_name}"] = (
                "adapter_config lists this module under modules_to_save "
                "but the weight file carries no tensors for it"
            )
    return pairs, full_state, rejected


def _raise_rejected(rejected, where):
    preview = "; ".join(f"{k}: {v}" for k, v in list(rejected.items())[:5])
    if len(rejected) > 5:
        preview += f"; ... (+{len(rejected) - 5} more)"
    raise ValueError(
        f"Unsloth MLX: {where} contains adapter tensors that cannot be "
        f"converted as plain linear LoRA ({preview}). No partial conversion "
        "is performed."
    )


def _require_fresh_destination(dst):
    # Early refusal including dangling symlinks; the atomic publish rename at
    # the end of conversion makes the claim final.
    if os.path.lexists(dst):
        raise ValueError(
            f"Unsloth MLX: destination {dst!r} already exists. Adapter "
            "conversion creates its own fresh directory so two formats can "
            "never mix in one place; pass a path that does not exist yet."
        )


def _num_hidden_layers(base_config):
    for holder in (base_config, base_config.get("text_config") or {}):
        for key in ("num_hidden_layers", "num_layers"):
            if isinstance(holder, dict) and holder.get(key) is not None:
                return int(holder[key])
    raise ValueError(
        "Unsloth MLX: base config lacks num_hidden_layers (top-level or "
        "text_config); cannot synthesize mlx-lm adapter metadata."
    )


def convert_peft_dir_to_mlx(src, dst, base_config):
    """Convert a PEFT LoRA directory to an mlx-lm adapter directory."""
    # Raises on a both-formats directory; the source must be unambiguous.
    detect_adapter_format(src)
    backend, load_file, save_file, transpose = _tensor_backend()
    with open(os.path.join(src, "adapter_config.json"), "r") as f:
        cfg = normalize_peft_adapter_config(json.load(f), adapter_dir=src)
    tensors = load_file(os.path.join(src, PEFT_WEIGHTS_FILE))
    pairs, full_state, rejected = group_peft_lora_pairs(tensors, cfg)
    if rejected:
        _raise_rejected(rejected, f"{src!r}")
    if not pairs:
        raise ValueError(f"Unsloth MLX: {src!r} has no LoRA tensor pairs.")

    out = {}
    ranks, scales = {}, {}
    fs_origins = {}
    _base_ties = _config_ties_word_embeddings(base_config)
    if _base_ties:
        _m2s_emb = sorted(
            p for p, v in full_state.items()
            if v.get("__origin__") == "modules_to_save"
            and _names_embedding_or_head(p, v.get("weight"), base_config)
        )
        if _m2s_emb:
            raise ValueError(
                f"Unsloth MLX: {_m2s_emb} carry {_TIED_FULL_STATE_REASON}."
            )
    for path, entries in sorted(full_state.items()):
        if (
            _base_ties
            and entries["__origin__"] == "embedding_auto"
            and _leaf_in(path, _OUTPUT_HEAD_LEAF_NAMES)
        ):
            _emb_matches = [
                v for p, v in full_state.items()
                if _leaf_in(p, _EMBEDDING_LEAF_NAMES)
            ]
            if len(_emb_matches) > 1:
                raise ValueError(
                    f"Unsloth MLX: {path} is a tied output-head snapshot "
                    "but the artifact carries multiple embed_tokens "
                    "snapshots; the duplicate cannot be attributed."
                )
            _err = _tied_head_fold_error(
                path, entries, (_emb_matches or [{}])[0].get("weight"),
                _tensors_equal,
            )
            if _err:
                raise ValueError(f"Unsloth MLX: {_err}.")
            continue
        fs_origins[path] = entries["__origin__"]
        # A reload wraps LoRA-paired modules, so base tensors live under the
        # wrapper's inner module.
        _kp = f"{path}.linear" if path in pairs else path
        for tname, tensor in entries.items():
            if tname == "__origin__":
                continue
            if not _is_float_dtype(tensor):
                raise ValueError(
                    f"Unsloth MLX: {path}.{tname} holds non-floating "
                    f"({getattr(tensor, 'dtype', '?')}) full-module state; "
                    "the artifact cannot restore as module state."
                )
            out[f"{_kp}.{tname}"] = tensor
    for path in sorted(pairs):
        a, b = pairs[path]["A"], pairs[path]["B"]
        rank = _lora_pair_rank(path, a, b, 0)
        out[f"{path}.lora_a"] = transpose(a)
        out[f"{path}.lora_b"] = transpose(b)
        mag = pairs[path].get("M")
        if cfg.get("use_dora") and mag is None:
            raise ValueError(
                f"Unsloth MLX: use_dora is set but {path} has no magnitude "
                "vector; the config and weights disagree."
            )
        if mag is not None and not cfg.get("use_dora"):
            raise ValueError(
                f"Unsloth MLX: {path} carries a DoRA magnitude but the "
                "config does not set use_dora; the artifact is inconsistent."
            )
        if mag is not None:
            _check_dora_magnitude(path, mag, int(b.shape[0]))
            out[f"{path}.m"] = mag
        ranks[path] = rank
        scales[path] = _effective_scale(cfg, path, rank)
    uniform_rank = len(set(ranks.values())) == 1
    uniform_scale = len(set(scales.values())) == 1
    any_path = next(iter(sorted(ranks)))
    _is_dora = bool(cfg.get("use_dora"))
    mlx_cfg = {
        "fine_tune_type": "dora" if _is_dora else "lora",
        "peft_type": "LORA",
        "num_layers": _num_hidden_layers(base_config),
        "base_model_name_or_path": cfg.get("base_model_name_or_path", ""),
        "lora_parameters": {
            "rank": ranks[any_path],
            "scale": scales[any_path],
            "dropout": float(cfg.get("lora_dropout") or 0.0),
            # Pin the exact module set: without "keys", mlx-lm wraps every
            # projection in the selected layers with zero-initialized extras.
            "keys": sorted(ranks),
        },
        "unsloth_mlx_lora_module_paths": sorted(ranks),
    }
    if cfg.get("base_model_revision"):
        mlx_cfg["base_model_revision"] = cfg["base_model_revision"]
    # Converted artifacts carry peft-parity guarantees native ones do not;
    # reload validation keys off this stamp.
    mlx_cfg["unsloth_peft_converted"] = True
    if fs_origins:
        mlx_cfg["full_state_modules"] = fs_origins
    if not (uniform_rank and uniform_scale):
        # Stock mlx-lm rebuilds from one global rank/scale; record the
        # per-module truth and flag that reloads need the unsloth loader.
        mlx_cfg["unsloth_mlx_lora_module_ranks"] = ranks
        mlx_cfg["unsloth_mlx_lora_module_scales"] = scales
        mlx_cfg["unsloth_mlx_requires_unsloth_loader"] = True
        print(
            "Unsloth: this adapter uses per-module ranks/alphas; the MLX "
            "copy needs Unsloth to load (stock mlx-lm assumes one global "
            "rank/scale)."
        )
    return _publish_adapter_dir(dst, MLX_WEIGHTS_FILE, out, mlx_cfg, save_file)


def _names_embedding_or_head(path, weight=None, config=None):
    """Whether a full-state path is the embedding or the output head. Callers
    decide tie-ness from the config; this only identifies the module, by a row
    count matching the declared vocabulary when a weight is at hand, else by
    leaf spelling."""
    shape = tuple(getattr(weight, "shape", ()) or ())
    if len(shape) == 2:
        for scope in ((config or {}).get("text_config") or {}, config or {}):
            if isinstance(scope, dict) and scope.get("vocab_size"):
                return int(shape[0]) == int(scope["vocab_size"]) or _leaf_in(
                    path, _EMBEDDING_LEAF_NAMES + _OUTPUT_HEAD_LEAF_NAMES
                )
    return _leaf_in(path, _EMBEDDING_LEAF_NAMES + _OUTPUT_HEAD_LEAF_NAMES)


def _full_state_owner(path, recorded):
    """The recorded full-state module a path belongs to: the path itself, or
    the module whose LoRA wrapper keeps its base under .embedding/.linear."""
    if path in recorded:
        return path
    for inner in (".embedding", ".linear"):
        if path.endswith(inner) and path[: -len(inner)] in recorded:
            return path[: -len(inner)]
    return None


def _validate_full_state_origins(fs_origins):
    bad = {
        p: o for p, o in fs_origins.items()
        if o not in ("modules_to_save", "embedding_auto")
    }
    if bad:
        raise ValueError(
            f"Unsloth MLX: full_state_modules carries unknown origin tag(s) "
            f"{bad}; expected 'modules_to_save' or 'embedding_auto'."
        )


def _tensors_equal(a, b):
    if tuple(getattr(a, "shape", ())) != tuple(getattr(b, "shape", ())):
        return False
    if getattr(a, "dtype", None) != getattr(b, "dtype", None):
        return False
    return bool((a == b).all())


def _tied_head_fold_error(path, entries, emb_weight, equal):
    """Why a tied base's auto-saved output-head snapshot cannot fold into the
    embedding, or None when ``equal`` accepts it as a duplicate. On a tied base
    the head tensor IS the embedding weight, so emitting it would restore a
    duplicate the MLX tree may not even carry. ``equal`` is the caller's:
    the converter demands an identical dtype, while a live attach compares in
    the live dtype, since the loader may have cast the base."""
    extra = sorted(set(entries) - {"__origin__", "weight"})
    if extra:
        return (
            f"{path} (tied output-head snapshot carries {extra}; a tied "
            "embedding has no slot for them)"
        )
    snapshot = entries.get("weight")
    if snapshot is None or not _is_float_dtype(snapshot):
        return (
            f"{path} (tied output-head snapshot is missing or non-floating)"
        )
    if emb_weight is None or not equal(emb_weight, snapshot):
        return (
            f"{path} (tied output-head snapshot differs from the tied "
            "embedding; a tied model has no independent head)"
        )
    return None


def _lora_pair_rank(path, a, b, rank_axis):
    """The rank a LoRA factor pair agrees on. ``rank_axis`` is 0 for peft's
    [r,in]/[out,r] layout and 1 for mlx-lm's [in,r]/[r,out]; ``b`` is indexed
    on the other axis. Messages use the direction's own key spelling."""
    _a, _b, _why = (
        ("lora_A", "lora_B", "not plain linear LoRA") if rank_axis == 0
        else ("lora_a", "lora_b", "switch/MoE LoRA cannot be exported")
    )
    if getattr(a, "ndim", 0) != 2 or getattr(b, "ndim", 0) != 2:
        raise ValueError(
            f"Unsloth MLX: {path} carries non-2-D LoRA factors ({_a} ndim "
            f"{getattr(a, 'ndim', '?')}, {_b} ndim {getattr(b, 'ndim', '?')}); "
            f"{_why}."
        )
    rank = int(a.shape[rank_axis])
    other = int(b.shape[1 - rank_axis])
    if other != rank:
        raise ValueError(
            f"Unsloth MLX: {path} has {_a} rank {rank} but {_b} rank "
            f"{other}; the adapter file is inconsistent."
        )
    return rank


def _check_dora_magnitude(path, mag, out_dim):
    if getattr(mag, "ndim", 0) != 1 or int(mag.shape[0]) != out_dim:
        raise ValueError(
            f"Unsloth MLX: {path} DoRA magnitude shape "
            f"{tuple(getattr(mag, 'shape', ()))} does not match the module "
            f"out-dim {out_dim}."
        )


def _publish_adapter_dir(dst, weights_name, tensors, cfg, save_file):
    """Stage the whole artifact in a unique sibling directory and publish it
    with ONE atomic rename, so a crash leaves either nothing at ``dst`` or the
    complete adapter. Claim-first designs strand a partial destination that
    blocks retries. The rename refuses a concurrently created non-empty dst,
    so content is never replaced and formats never mix; adopting a zero-data
    entry (empty directory or bare symlink) created in the race window is
    accepted. dst is checked in the caller's spelling BEFORE resolving, so a
    pre-planted dangling symlink is rejected rather than followed — trailing
    separators go first, since lexists("link/") follows the terminal symlink —
    and again after, in case a parent symlink retargeted it."""
    dst = os.fspath(dst)
    _require_fresh_destination(dst.rstrip(os.sep) or dst)
    dst = os.path.realpath(dst)
    _require_fresh_destination(dst)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    tmp = f"{dst}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    created = False
    try:
        os.makedirs(tmp)
        created = True
        save_file(os.path.join(tmp, weights_name), tensors)
        with open(os.path.join(tmp, "adapter_config.json"), "w") as f:
            json.dump(cfg, f, indent=2, sort_keys=True)
        try:
            os.rename(tmp, dst)
        except OSError as exc:
            raise ValueError(
                f"Unsloth MLX: destination {dst!r} appeared concurrently "
                "and is not empty; refusing to mix adapter artifacts."
            ) from exc
    finally:
        if created:
            shutil.rmtree(tmp, ignore_errors=True)
    return dst


def _is_float_dtype(tensor):
    """Full-module state must be float; packed or boolean tensors cannot restore."""
    return "float" in str(getattr(tensor, "dtype", "")).lower()


# Embedding/head module spellings across mlx-lm text models, used when a
# tensor-shape test cannot decide.
_EMBEDDING_LEAF_NAMES = (
    "embed_tokens", "tok_embeddings", "token_embeddings", "wte",
    "embed_in", "word_embeddings", "embeddings", "embedding",
)
_OUTPUT_HEAD_LEAF_NAMES = ("lm_head", "output", "embed_out")


# peft trains an untied copy of a tied embedding or head, while the tied MLX
# module both looks up inputs and projects output logits.
_TIED_FULL_STATE_REASON = (
    "full-module state for a tied embedding or output head, which peft and "
    "mlx-lm cannot represent the same way"
)


def _leaf_in(path, names):
    return path.rsplit(".", 1)[-1] in names


def _is_auto_saved_module_path(path):
    """Whether a path could name the input embedding or output head, the two
    modules peft auto-saves full base state for. These are Hugging Face paths,
    whose spellings for those two are a small fixed set, but some are reused
    deeper in the tree: OLMo names both its root head and every block's MLP
    output ff_out, and `output` also spells an attention projection. Neither
    of those sits at the root, so a path through a numbered layer is never
    one of them."""
    if any(segment.isdigit() for segment in path.split(".")):
        return False
    # ff_out is hf_olmo's head, kept local: the tied-embedding checks read the
    # shared tuple without the numbered-segment rule.
    return _leaf_in(
        path, _EMBEDDING_LEAF_NAMES + _OUTPUT_HEAD_LEAF_NAMES + ("ff_out",)
    )


def _config_ties_word_embeddings(config):
    """True when a config ties word embeddings. A nested text config governs
    when it carries the key (embedding adapters live in the language tower, so
    an outer multimodal false must not shadow it). An ABSENT key counts as
    tied: PretrainedConfig defaults it True and always-tied architectures may
    omit it."""
    config = config or {}
    text_config = config.get("text_config")
    if isinstance(text_config, dict) and "tie_word_embeddings" in text_config:
        return bool(text_config["tie_word_embeddings"])
    if "tie_word_embeddings" in config:
        return bool(config["tie_word_embeddings"])
    return True


def convert_mlx_dir_to_peft(src, dst, module_types=None):
    """Convert an mlx-lm adapter directory to a PEFT LoRA directory.

    ``module_types`` optionally declares LoRA target paths as ``"linear"``.
    Such an entry asserts what a weights file cannot: that the same dotted
    path exists in the Hugging Face tree. mlx-lm renames some roots (GPT-2's
    ``model.h.*`` vs HF's ``transformer.h.*``), so a mis-asserted path yields
    an adapter peft cannot bind. Without the map only root-anchored
    ``model.layers.N.`` paths convert and anything else is refused.
    """
    for _path, _kind in (module_types or {}).items():
        if _kind != "linear":
            raise ValueError(
                f"Unsloth MLX: module_types[{_path!r}]={_kind!r}; expected "
                "'linear'."
            )
    # Raises on a both-formats directory; the source must be unambiguous.
    detect_adapter_format(src)
    backend, load_file, save_file, transpose = _tensor_backend()
    with open(os.path.join(src, "adapter_config.json"), "r") as f:
        cfg = json.load(f)
    _ft = str(cfg.get("fine_tune_type", "lora")).lower()
    if _ft not in ("lora", "dora"):
        raise ValueError(
            f"Unsloth MLX: fine_tune_type={cfg.get('fine_tune_type')!r} "
            "adapters cannot be exported to PEFT format."
        )
    fs_origins = dict(cfg.get("full_state_modules") or {})
    # PEFT expresses trainable module state only as modules_to_save, which
    # cannot coexist with a LoRA target on one module, so a trainable
    # auto-saved replacement has no faithful PEFT form.
    _trainable_auto = sorted(
        p for p in (cfg.get("full_state_trainable") or [])
        if fs_origins.get(p) != "modules_to_save"
    )
    if _trainable_auto:
        raise ValueError(
            f"Unsloth MLX: {_trainable_auto} carry trainable auto-saved "
            "full-module state, which PEFT cannot represent. Export the MLX "
            "adapter instead."
        )
    _validate_full_state_origins(fs_origins)
    tensors = load_file(os.path.join(src, MLX_WEIGHTS_FILE))
    pairs = {}
    rejected = {}
    full_state_out = {}
    for key, tensor in tensors.items():
        fsm = re.match(r"^(.+)\.(weight|bias)$", key)
        if fsm is not None:
            _fs_path = _full_state_owner(fsm.group(1), fs_origins)
            if _fs_path is not None:
                if not _is_float_dtype(tensor):
                    raise ValueError(
                        f"Unsloth MLX: {key} holds non-floating "
                        f"({getattr(tensor, 'dtype', '?')}) full-module "
                        "state; dequantize before exporting to PEFT format."
                    )
                _slot = full_state_out.setdefault(_fs_path, {})
                _prev = _slot.get(fsm.group(2))
                if _prev is not None and not _tensors_equal(_prev, tensor):
                    raise ValueError(
                        f"Unsloth MLX: {key} disagrees with another "
                        "spelling of the same module tensor; the adapter "
                        "file is inconsistent."
                    )
                _slot[fsm.group(2)] = tensor
                continue
        magm = re.match(r"^(.+)\.m$", key)
        if magm is not None and str(cfg.get("fine_tune_type", "")).lower() == "dora":
            pairs.setdefault(magm.group(1), {})["m_vec"] = tensor
            continue
        m = re.match(r"^(.+)\.lora_(a|b)$", key)
        if m is None:
            rejected[key] = (
                "not a LoRA/DoRA adapter tensor for this artifact's "
                "fine_tune_type; full-weight state cannot be exported"
            )
            continue
        path = m.group(1)
        if (module_types or {}).get(path) is None and re.match(
            # Root-anchored `model.layers.N.` only: the layout mlx-lm mirrors
            # from HF, so emitted paths bind in peft. GPT-2-style stacks rename
            # roots, so an unanchored heuristic would emit unloadable adapters.
            r"^model\.layers\.\d+\.", path
        ) is None:
            rejected[key] = (
                "outside a Hugging-Face-mirroring transformer stack, so PEFT "
                "export could emit an adapter that binds to nothing — save "
                "this adapter in MLX format instead"
            )
            continue
        pairs.setdefault(path, {})[m.group(2)] = tensor
    for path, pair in list(pairs.items()):
        if "a" not in pair or "b" not in pair:
            rejected[f"{path}.lora_*"] = "incomplete lora_a/lora_b pair"
            pairs.pop(path)
            continue
        _bad = sorted(
            k for k in ("a", "b", "m_vec")
            if k in pair and not _is_float_dtype(pair[k])
        )
        if _bad:
            rejected[f"{path}.lora_*"] = (
                f"non-floating adapter factor dtype(s) "
                f"({', '.join(str(pair[k].dtype) for k in _bad)}); LoRA "
                "factors must be floating point"
            )
            pairs.pop(path)
    if rejected:
        _raise_rejected(rejected, f"{src!r}")
    if not pairs:
        raise ValueError(f"Unsloth MLX: {src!r} has no LoRA tensor pairs.")

    scale_map = cfg.get("unsloth_mlx_lora_module_scales") or {}
    lora_params = cfg.get("lora_parameters") or {}
    global_scale = float(lora_params.get("scale", cfg.get("scale", 1.0)))
    dropout = float(lora_params.get("dropout", cfg.get("dropout", 0.0)) or 0.0)

    out = {}
    ranks, alphas = {}, {}
    if _ft == "dora" and dropout > 0:
        raise ValueError(
            "Unsloth MLX: DoRA adapters with nonzero dropout cannot be "
            "exported; peft and mlx-lm apply DoRA dropout with different "
            "training-mode formulas."
        )
    _mag_paths = {p for p, v in pairs.items() if "m_vec" in v}
    if _ft == "dora" and _mag_paths != set(pairs):
        raise ValueError(
            "Unsloth MLX: DoRA magnitudes present on only some modules; "
            "peft applies use_dora globally, so a partial set cannot be "
            "exported."
        )
    for path in sorted(pairs):
        a, b = pairs[path]["a"], pairs[path]["b"]
        rank = _lora_pair_rank(path, a, b, 1)
        out[f"{_PEFT_PREFIX}{path}.lora_A.weight"] = transpose(a)
        out[f"{_PEFT_PREFIX}{path}.lora_B.weight"] = transpose(b)
        if "m_vec" in pairs[path]:
            _mv = pairs[path]["m_vec"]
            _check_dora_magnitude(path, _mv, int(b.shape[1]))
            # peft's on-disk form has no .weight suffix (stripped at save).
            out[f"{_PEFT_PREFIX}{path}.lora_magnitude_vector"] = _mv
        ranks[path] = rank
        scale = float(scale_map.get(path, global_scale))
        alphas[path] = scale * rank
    # Modal values minimize pattern entries; peft's ^-anchored matcher gives
    # exact-path semantics, so overlapping dotted suffixes cannot shadow.
    from collections import Counter
    ref_rank = Counter(ranks.values()).most_common(1)[0][0]
    ref_alpha = Counter(alphas.values()).most_common(1)[0][0]
    peft_cfg = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": ref_rank,
        "lora_alpha": ref_alpha,
        "lora_dropout": dropout,
        "bias": "none",
        "use_rslora": False,
        "use_dora": _ft == "dora",
        # Full module paths, matched exactly by peft's list semantics: leaf
        # names would suffix-match every layer's projection, growing a
        # layer-subset adapter into a different topology.
        "target_modules": sorted(ranks),
        "base_model_name_or_path": cfg.get("base_model_name_or_path", ""),
    }
    revision = cfg.get("base_model_revision") or cfg.get("base_model_commit_hash")
    if revision:
        peft_cfg["revision"] = revision
    if len(set(ranks.values())) > 1:
        peft_cfg["rank_pattern"] = {
            "^" + re.escape(p): r for p, r in ranks.items()
            if r != peft_cfg["r"]
        }
    if len(set(alphas.values())) > 1:
        peft_cfg["alpha_pattern"] = {
            "^" + re.escape(p): a for p, a in alphas.items()
            if a != peft_cfg["lora_alpha"]
        }
    _missing_fs = set(fs_origins) - set(full_state_out)
    if _missing_fs:
        raise ValueError(
            "Unsloth MLX: full_state_modules lists "
            f"{sorted(_missing_fs)} but the weight file carries no "
            "tensors for them; the artifact is inconsistent."
        )
    _m2s_list = sorted(
        p for p, origin in fs_origins.items() if origin == "modules_to_save"
    )
    if _m2s_list:
        peft_cfg["modules_to_save"] = _m2s_list
    for _p, _entries in sorted(full_state_out.items()):
        # peft's on-disk forms: plain {path}.{tensor} for snapshots and
        # untargeted auto-saved embeddings, but a LoRA-wrapped module's base
        # state sits under .base_layer, since a plain key would not bind.
        _wrapped = _p in pairs
        for _tname, _tensor in sorted(_entries.items()):
            if _wrapped:
                out[f"{_PEFT_PREFIX}{_p}.base_layer.{_tname}"] = _tensor
            else:
                out[f"{_PEFT_PREFIX}{_p}.{_tname}"] = _tensor
    return _publish_adapter_dir(dst, PEFT_WEIGHTS_FILE, out, peft_cfg, save_file)


def export_peft_adapter(src, dst, *, base_config=None, module_types=None):
    """Public wrapper: convert an mlx-lm adapter directory to PEFT format.

    ``module_types`` follows convert_mlx_dir_to_peft. ``base_config`` adds the
    tie_word_embeddings check that a modules_to_save snapshot of a tied
    embedding cannot be exported.
    """
    if base_config is not None and _config_ties_word_embeddings(base_config):
        try:
            with open(os.path.join(os.fspath(src), "adapter_config.json")) as _f:
                _src_fs = dict(json.load(_f).get("full_state_modules") or {})
        except OSError:
            _src_fs = {}
        _m2s_emb = sorted(
            p for p, o in _src_fs.items()
            if o == "modules_to_save"
            and _names_embedding_or_head(p)
        )
        if _m2s_emb:
            raise ValueError(
                f"Unsloth MLX: {_m2s_emb} carry {_TIED_FULL_STATE_REASON}."
            )
    return convert_mlx_dir_to_peft(src, dst, module_types=module_types)
