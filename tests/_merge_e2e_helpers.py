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

"""Shared helpers for the end-to-end LoRA merge-to-16bit correctness suite.

Drives the real merge path on tiny architecturally-real models and checks each
saved tensor against a reference built independently from the live PEFT LoRA
(adapted: base + scale*(B@A); pass-through: byte-identical). Distinct hidden /
intermediate sizes keep fused-expert orientation unambiguous. Not a test_ module.
"""

from __future__ import annotations

import os
import json
import hashlib
from dataclasses import dataclass, field

import torch
from safetensors import safe_open

from unsloth_zoo.saving_utils import merge_and_overwrite_lora


SEED = 1234

# (atol, rtol) for adapted tensors; pass-through is always byte-exact.
_TOL = {
    torch.float32:  (1e-5, 1e-5),
    torch.bfloat16: (2e-2, 2e-2),
    torch.float16:  (5e-3, 5e-3),
}


def set_offline_cpu_env():
    """Make the merge run offline and CPU-tolerant for CI."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("UNSLOTH_ALLOW_CPU", "1")
    os.environ.setdefault("UNSLOTH_DISABLE_AUTO_UPDATES", "1")


@dataclass
class _Adapted:
    key: str
    lora_A: torch.Tensor
    lora_B: torch.Tensor
    alpha: float
    fused: bool         # grouped 3D expert tensor (gate_up_proj/down_proj)


def _strip_peft_prefix(name: str) -> str:
    for pfx in ("base_model.model.", "base_model."):
        if name.startswith(pfx):
            return name[len(pfx):]
    return name


def _iter_lora_modules(peft_model):
    """Yield (name, module) for every LoRA-bearing module."""
    for name, mod in peft_model.named_modules():
        la = getattr(mod, "lora_A", None)
        lb = getattr(mod, "lora_B", None)
        if la is None or lb is None:
            continue
        if not (hasattr(la, "__contains__") and "default" in la and "default" in lb):
            continue
        yield name, mod


def _module_to_key(name: str, param_name) -> str:
    """Map a PEFT module name to the safetensor key it adapts."""
    base = _strip_peft_prefix(name)
    if param_name:
        if base.endswith(".base_layer"):
            base = base[: -len(".base_layer")]
        return f"{base}.{param_name}"
    return f"{base}.weight"


def seed_lora(peft_model, seed: int = SEED):
    """Fill lora_A/lora_B with deterministic asymmetric nonzero values.

    Seeded by safetensor key (hashlib, since hash() is per-process random) so the
    result is independent of module-iteration order; the shaped ramp makes
    transpose/slice bugs visible on tiny tensors.
    """
    with torch.no_grad():
        for name, mod in _iter_lora_modules(peft_model):
            key = _module_to_key(name, getattr(mod, "parameter_name", None))
            for tag, bank in (("A", mod.lora_A), ("B", mod.lora_B)):
                w = bank["default"].weight
                h = hashlib.sha256(f"{key}|{tag}|{seed}".encode()).digest()
                g = torch.Generator(device="cpu").manual_seed(
                    int.from_bytes(h[:7], "big")
                )
                vals = torch.randn(w.numel(), generator=g)
                ramp = torch.arange(w.numel(), dtype=torch.float32) * 1.0e-3
                w.copy_((vals * 0.02 + ramp).reshape(w.shape).to(w.dtype))


def extract_adapted(peft_model) -> dict[str, _Adapted]:
    """Build {safetensor_key: _Adapted} from the live PEFT model."""
    out: dict[str, _Adapted] = {}
    for name, mod in _iter_lora_modules(peft_model):
        param_name = getattr(mod, "parameter_name", None)
        key = _module_to_key(name, param_name)
        A = mod.lora_A["default"].weight.detach().cpu()
        B = mod.lora_B["default"].weight.detach().cpu()
        alpha = float(mod.scaling["default"])
        out[key] = _Adapted(key=key, lora_A=A, lora_B=B, alpha=alpha,
                            fused=bool(param_name))
    return out


def _ref_dense(base: torch.Tensor, a: _Adapted) -> torch.Tensor:
    """2D Linear / per-expert merge, with vocab-grow zero-pad when B has more rows."""
    A = a.lora_A.to(torch.float64)
    B = a.lora_B.to(torch.float64)
    W = base.to(torch.float64)
    if B.shape[0] != W.shape[0]:
        new = torch.zeros(B.shape[0], W.shape[1], dtype=torch.float64)
        new[: W.shape[0]] = W
        W = new
    return W + a.alpha * (B @ A)


def _ref_fused(base3d: torch.Tensor, a: _Adapted) -> torch.Tensor:
    """Grouped 3D expert merge; per expert delta=B_e@A_e in the orientation that
    fits base[e] (auto-detected from distinct dims)."""
    E = base3d.shape[0]
    r = a.lora_A.shape[0] // E
    d1, d2 = base3d.shape[1], base3d.shape[2]
    out = base3d.to(torch.float64).clone()
    A = a.lora_A.to(torch.float64)
    B = a.lora_B.to(torch.float64)
    for e in range(E):
        s, t = e * r, (e + 1) * r
        delta = B[:, s:t] @ A[s:t, :]
        if tuple(delta.shape) == (d1, d2):
            out[e] += a.alpha * delta
        elif tuple(delta.shape) == (d2, d1):
            out[e] += a.alpha * delta.T
        else:
            raise AssertionError(
                f"fused delta {tuple(delta.shape)} fits neither {(d1, d2)} nor "
                f"{(d2, d1)} for key {a.key}; use distinct hidden/intermediate dims"
            )
    return out


def read_safetensors_dir(d: str) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(d, f), framework="pt", device="cpu") as g:
                for k in g.keys():
                    out[k] = g.get_tensor(k)
    return out


def read_index(d: str):
    p = os.path.join(d, "model.safetensors.index.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def run_merge(peft_model, base_dir, out_dir, *, save_dtype, tokenizer=None,
              low_disk_space_usage=False):
    """Canonical local-dir merge invocation. Returns out_dir."""
    res = merge_and_overwrite_lora(
        get_model_name=lambda *a, **k: base_dir,
        model=peft_model,
        tokenizer=tokenizer,
        save_directory=out_dir,
        save_method="merged_16bit",
        output_dtype=save_dtype,
        low_disk_space_usage=low_disk_space_usage,
        push_to_hub=False,
    )
    return res


_LORA_REMNANT_MARKERS = (".lora_A", ".lora_B", ".lora_embedding",
                         ".base_layer", "modules_to_save", "original_module")


class KeyResolutionError(AssertionError):
    """Adapted module could not be mapped to a base safetensor key.

    Raised (not for value mismatches) when a tiny config's on-disk key layout
    differs from the LoRA module path (composite VLM prefixes, or fused-vs-per-
    expert serialization that varies by transformers version). Tests skip on it.
    """


def _resolve_key(adapted_key: str, base_keys: set[str]) -> str | None:
    """Resolve an adapted key to a base key, tolerating an extra prefix (VLM
    composite models; mirrors _infer_prefix_and_remap)."""
    if adapted_key in base_keys:
        return adapted_key
    cands = [k for k in base_keys if k.endswith("." + adapted_key) or k.endswith(adapted_key)]
    cands = [k for k in cands if k != adapted_key]
    if len(cands) == 1:
        return cands[0]
    return None


def _load_named_params(model_dir: str) -> dict[str, torch.Tensor]:
    """Load a saved model and return {param_name: cpu tensor}. In-memory parameter
    names line up with the PEFT/adapted keys even when the checkpoint uses a
    different on-disk layout (composite-VLM prefixes, per-expert expert shards):
    transformers re-packs the experts on load."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    return {n: p.detach().cpu() for n, p in model.named_parameters()}


def assert_merge_correct_model_space(*, family, base_dir, out_dir,
                                     adapted: dict[str, _Adapted], save_dtype):
    """Verify a merge in the in-memory (re-packed) parameter space.

    Some architectures keep experts packed in memory (mlp.experts.gate_up_proj)
    but serialize them per-expert on disk (mlp.experts.<e>.gate_proj.weight), so
    the packed adapted key maps to no single safetensor. Loading base + merged
    re-packs the experts, so adapted keys line up with parameter names and
    _ref_fused / _ref_dense apply directly. The merge does not modify base_dir,
    so it is loaded as the pre-merge reference.
    """
    from unsloth_zoo.saving_utils import _MOE_MERGE_STATE

    base_p = _load_named_params(base_dir)
    merged_p = _load_named_params(out_dir)
    assert set(base_p) == set(merged_p), (
        f"[{family}] merge changed the parameter set "
        f"(+{sorted(set(merged_p) - set(base_p))[:4]} "
        f"-{sorted(set(base_p) - set(merged_p))[:4]})"
    )

    # map each adapted key onto its parameter name (usually identical)
    resolved: dict[str, _Adapted] = {}
    for ak, a in adapted.items():
        if ak in base_p:
            pname = ak
        else:
            cands = [n for n in base_p if n == ak or n.endswith("." + ak) or n.endswith(ak)]
            cands = [n for n in cands if n != ak]
            if len(cands) != 1:
                raise KeyResolutionError(
                    f"[{family}] adapted key {ak!r} maps to {len(cands)} params {cands[:4]}"
                )
            pname = cands[0]
        resolved[pname] = a

    atol, rtol = _TOL[save_dtype]
    n_adapted = n_passthru = 0
    for name, base in base_p.items():
        got = merged_p[name]
        if name in resolved:
            a = resolved[name]
            ref = (_ref_fused(base, a) if a.fused else _ref_dense(base, a)).to(save_dtype)
            _assert_close(family, name, got.to(save_dtype), ref, atol, rtol, adapted=True)
            n_adapted += 1
        else:
            _assert_equal(family, name, got, base)
            n_passthru += 1

    assert n_adapted >= 1, f"[{family}] no adapted params were checked (LoRA not attached?)"
    assert _MOE_MERGE_STATE.get("fallback", 0) == 0, (
        f"[{family}] MoE merge fell back: {_MOE_MERGE_STATE}"
    )
    return n_adapted, n_passthru


def assert_merge_correct(*, family, base_tensors, out_dir, save_dtype,
                         adapted: dict[str, _Adapted], allow_missing_adapted=False,
                         base_dir=None):
    """Check every merged tensor against the independent reference.

    base_tensors : {key: tensor} snapshot of the base BEFORE merge.
    adapted      : {safetensor_key: _Adapted} from extract_adapted().
    """
    from unsloth_zoo.saving_utils import _MOE_MERGE_STATE

    merged = read_safetensors_dir(out_dir)
    assert merged, f"[{family}] no safetensors written to {out_dir}"
    base_keys = set(base_tensors.keys())

    # no adapter remnants leaked
    leaked = [k for k in merged if any(m in k for m in _LORA_REMNANT_MARKERS)]
    assert not leaked, f"[{family}] adapter remnants in merged output: {leaked[:5]}"

    # Fused experts can serialize per-expert on disk (e.g. qwen3_5_moe keeps
    # mlp.experts.gate_up_proj packed in memory but writes experts.<e>.gate_proj
    # .weight shards), so the packed adapted key resolves to no single
    # safetensor. The merge handles this (saving_utils._merge_moe_*); verify in
    # the in-memory fused space, where param names line up with the adapted keys.
    if any(a.fused and _resolve_key(ak, base_keys) is None for ak, a in adapted.items()):
        assert base_dir is not None, (
            f"[{family}] base_dir is required to verify per-expert-serialized "
            f"fused experts in model space"
        )
        return assert_merge_correct_model_space(
            family=family, base_dir=base_dir, out_dir=out_dir,
            adapted=adapted, save_dtype=save_dtype,
        )

    # resolve adapted keys against real base keys (VLM prefix tolerance)
    resolved: dict[str, _Adapted] = {}
    for ak, a in adapted.items():
        rk = _resolve_key(ak, base_keys)
        if rk is None:
            if allow_missing_adapted:
                # target absent from base (e.g. vision-tower adapter): merge skips
                # it, nothing to check (mirrors PR #773).
                continue
            raise KeyResolutionError(
                f"[{family}] adapted key {ak!r} not found among base keys "
                f"(sample: {sorted(base_keys)[:4]})"
            )
        resolved[rk] = a

    atol, rtol = _TOL[save_dtype]
    n_adapted = n_passthru = 0
    for key, mt in merged.items():
        if key in resolved:
            a = resolved[key]
            base = base_tensors[key]
            ref = (_ref_fused(base, a) if a.fused else _ref_dense(base, a)).to(save_dtype)
            _assert_close(family, key, mt, ref, atol, rtol, adapted=True)
            n_adapted += 1
        else:
            assert key in base_tensors, (
                f"[{family}] merged has unexpected key {key!r} absent from base"
            )
            _assert_equal(family, key, mt, base_tensors[key])
            n_passthru += 1

    # every adapted target appeared and was checked
    if not allow_missing_adapted:
        missing = [k for k in resolved if k not in merged]
        assert not missing, f"[{family}] adapted targets missing from merged output: {missing}"
    assert n_adapted >= 1, f"[{family}] no adapted tensors were checked (LoRA not attached?)"

    # MoE merge must not have fallen back
    assert _MOE_MERGE_STATE.get("fallback", 0) == 0, (
        f"[{family}] MoE merge fell back: {_MOE_MERGE_STATE}"
    )

    # index consistency
    idx = read_index(out_dir)
    if idx is not None:
        wm = idx.get("weight_map", {})
        for k, shard in wm.items():
            assert os.path.exists(os.path.join(out_dir, shard)), (
                f"[{family}] index points at missing shard {shard} for {k}"
            )
            assert k in merged, f"[{family}] index key {k} absent from shards"
    return n_adapted, n_passthru


def _diag(family, key, got, ref, adapted):
    diff = (got.to(torch.float64) - ref.to(torch.float64)).abs()
    maxe = diff.max().item() if diff.numel() else 0.0
    denom = ref.to(torch.float64).abs().clamp_min(1e-12)
    rele = (diff / denom).max().item() if diff.numel() else 0.0
    flat = (got != ref).reshape(-1).nonzero()
    first = int(flat[0].item()) if flat.numel() else -1
    return (f"[{family}] key={key} adapted={adapted} shape={tuple(got.shape)} "
            f"dtype={got.dtype} max_abs_err={maxe:.3e} max_rel_err={rele:.3e} "
            f"first_diff_idx={first}")


def _assert_close(family, key, got, ref, atol, rtol, *, adapted):
    try:
        torch.testing.assert_close(got.to(torch.float32), ref.to(torch.float32),
                                   atol=atol, rtol=rtol)
    except AssertionError as e:
        raise AssertionError(_diag(family, key, got, ref, adapted) + "\n" + str(e))


def _assert_equal(family, key, got, base):
    if not torch.equal(got, base):
        raise AssertionError("PASS-THROUGH NOT BYTE-IDENTICAL " +
                             _diag(family, key, got, base, adapted=False))


@dataclass
class FamilySpec:
    family: str
    kind: str                       # dense | per_expert | fused
    config: object
    auto: str = "causal"            # causal | image_text
    attn_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")
    mlp_modules: tuple = ("gate_proj", "up_proj", "down_proj")
    expert_modules: tuple = ()      # per-expert MoE target_modules
    fused_params: tuple = ()        # fused-expert target_parameters


# distinct dims so fused-expert orientation is unambiguous (transpose bugs show)
_H = 32
_I = 64
_MOE_I = 48      # distinct from _H and 2*_MOE_I
_HEADS = 4
_KV = 2
_VOCAB = 64
_POS = 64
_EXPERTS = 4


def _common(**over):
    d = dict(hidden_size=_H, intermediate_size=_I, num_hidden_layers=2,
             num_attention_heads=_HEADS, num_key_value_heads=_KV, vocab_size=_VOCAB,
             max_position_embeddings=_POS, tie_word_embeddings=False)
    d.update(over)
    return d


def family_available(family: str) -> bool:
    """True iff the installed transformers exposes this model_type."""
    try:
        from transformers import AutoConfig
        AutoConfig.for_model(family)
        return True
    except Exception:
        return False


def make_spec(family: str) -> FamilySpec:
    """Build a tiny FamilySpec for `family`. Raises if the arch is unavailable."""
    import transformers as T

    if family == "llama":
        cfg = T.LlamaConfig(**_common())
        return FamilySpec(family, "dense", cfg)
    if family == "qwen3":
        cfg = T.Qwen3Config(**_common(head_dim=_H // _HEADS))
        return FamilySpec(family, "dense", cfg)
    if family == "mistral":
        cfg = T.MistralConfig(**_common())
        return FamilySpec(family, "dense", cfg)
    if family == "gemma2":
        cfg = T.Gemma2Config(**_common(head_dim=_H // _HEADS, query_pre_attn_scalar=_H // _HEADS))
        return FamilySpec(family, "dense", cfg)

    if family == "qwen3_moe":
        cfg = T.Qwen3MoeConfig(**_common(moe_intermediate_size=_MOE_I, num_experts=_EXPERTS,
                                         num_experts_per_tok=2, decoder_sparse_step=1,
                                         head_dim=_H // _HEADS))
        return FamilySpec(family, "per_expert", cfg,
                          expert_modules=("gate_proj", "up_proj", "down_proj"))
    if family == "glm4_moe":
        cfg = T.Glm4MoeConfig(**_common(moe_intermediate_size=_MOE_I, n_routed_experts=_EXPERTS,
                                        n_shared_experts=1, num_experts_per_tok=2,
                                        first_k_dense_replace=0, head_dim=_H // _HEADS))
        return FamilySpec(family, "per_expert", cfg,
                          expert_modules=("gate_proj", "up_proj", "down_proj"))
    if family == "granitemoe":
        cfg = T.GraniteMoeConfig(**_common(num_local_experts=_EXPERTS, num_experts_per_tok=2))
        return FamilySpec(family, "per_expert", cfg,
                          expert_modules=("input_linear", "output_linear"))

    if family == "gpt_oss":
        cfg = T.GptOssConfig(**_common(intermediate_size=_MOE_I, num_local_experts=_EXPERTS,
                                       num_experts_per_tok=2))
        return FamilySpec(family, "fused", cfg,
                          fused_params=("mlp.experts.gate_up_proj", "mlp.experts.down_proj"))

    # transformers 5.x only families
    if family == "qwen3_5_moe":
        cfg = T.AutoConfig.for_model("qwen3_5_moe")
        return _shrink_generic(family, "fused", cfg,
                               fused_params=("mlp.experts.gate_up_proj", "mlp.experts.down_proj"))
    if family == "gemma4":
        cfg = T.AutoConfig.for_model("gemma4")
        return _shrink_generic(family, "fused", cfg,
                               fused_params=("experts.gate_up_proj", "experts.down_proj"))
    if family == "lfm2_moe":
        cfg = T.AutoConfig.for_model("lfm2_moe")
        return _shrink_generic(family, "fused", cfg,
                               fused_params=("feed_forward.experts.gate_up_proj",
                                             "feed_forward.experts.down_proj"))

    raise ValueError(f"unknown family {family!r}")


def _shrink_generic(family, kind, cfg, **kw):
    """Best-effort shrink of an arbitrary config (used for 5.x-only fused MoEs)."""
    text = getattr(cfg, "text_config", cfg)
    objs = [cfg] if text is cfg else [cfg, text]
    n_layers = 2
    for obj in objs:
        for attr, val in (("hidden_size", _H), ("intermediate_size", _I),
                          ("moe_intermediate_size", _MOE_I), ("num_hidden_layers", n_layers),
                          ("num_attention_heads", _HEADS), ("num_key_value_heads", _KV),
                          ("vocab_size", _VOCAB), ("max_position_embeddings", _POS),
                          ("num_experts", _EXPERTS), ("num_local_experts", _EXPERTS),
                          ("num_experts_per_tok", 2), ("top_k_experts", 2),
                          ("decoder_sparse_step", 1)):
            if hasattr(obj, attr):
                setattr(obj, attr, val)
        # Some families (gemma4) gate MoE behind a flag; enable it so the tiny
        # config materializes experts to exercise the fused-merge path.
        if hasattr(obj, "enable_moe_block"):
            obj.enable_moe_block = True
        # strict/hybrid archs (lfm2_moe, qwen3_5_moe) need an explicit per-layer
        # schedule matching num_hidden_layers or instantiation raises.
        if hasattr(obj, "layer_types") and getattr(obj, "num_hidden_layers", None):
            try:
                obj.layer_types = ["full_attention"] * obj.num_hidden_layers
            except Exception:
                pass
        if hasattr(obj, "tie_word_embeddings"):
            obj.tie_word_embeddings = False
    return FamilySpec(family, kind, cfg, **kw)


def build_and_save_base(spec: FamilySpec, base_dir: str, *, dtype=torch.float32,
                        max_shard_size="5GB"):
    """Instantiate the tiny model, save 16bit shards + index, return the model."""
    from transformers import AutoModelForCausalLM
    torch.manual_seed(SEED)
    cfg = spec.config
    try:
        model = AutoModelForCausalLM.from_config(cfg).to(dtype)
    except AttributeError:
        # Composite (multimodal) MoE configs keep vocab_size under text_config;
        # some transformers releases build the text backbone straight from the
        # top-level config and raise AttributeError on the missing attribute
        # (e.g. qwen3_5_moe on 5.5.0). Mirror unsloth_zoo.create_empty_causal_lm
        # and retry from text_config so the fused-expert path is still exercised.
        if hasattr(cfg, "vocab_size") or not hasattr(cfg, "text_config"):
            raise
        model = AutoModelForCausalLM.from_config(cfg.text_config).to(dtype)
    model.save_pretrained(base_dir, safe_serialization=True, max_shard_size=max_shard_size)
    model.config._name_or_path = base_dir
    return model


def attach_lora(model, spec: FamilySpec, scenario: str, *, r=8, lora_alpha=16,
                alpha_pattern=None, rank_pattern=None):
    """Attach a PEFT LoRA adapter (scenario in {full, attn_only, mlp_only,
    expert_only}) and seed it nonzero. Returns the PeftModel."""
    from peft import LoraConfig, get_peft_model
    tm: list[str] = []
    tp: list[str] = []
    if scenario in ("full", "attn_only"):
        tm += list(spec.attn_modules)
    if scenario in ("full", "mlp_only") and spec.kind == "dense":
        tm += list(spec.mlp_modules)
    if scenario in ("full", "expert_only"):
        if spec.kind == "per_expert":
            tm += list(spec.expert_modules)
        elif spec.kind == "fused":
            tp += list(spec.fused_params)
    kw = dict(r=r, lora_alpha=lora_alpha, lora_dropout=0.0, bias="none",
              target_modules=tm)
    if tp:
        kw["target_parameters"] = tp
    if alpha_pattern:
        kw["alpha_pattern"] = alpha_pattern
    if rank_pattern:
        kw["rank_pattern"] = rank_pattern
    torch.manual_seed(SEED)
    peft_model = get_peft_model(model, LoraConfig(**kw))
    seed_lora(peft_model)
    return peft_model


def run_case(family, scenario, work_dir, *, dtype=torch.float32, max_shard_size="5GB",
             allow_missing_adapted=False, **attach_kw):
    """Build base -> attach LoRA -> real merge -> assert. Returns
    (n_adapted_checked, n_passthrough_checked); raises on any mismatch."""
    set_offline_cpu_env()
    spec = make_spec(family)
    base_dir = os.path.join(work_dir, "base")
    out_dir = os.path.join(work_dir, "merged")
    model = build_and_save_base(spec, base_dir, dtype=dtype, max_shard_size=max_shard_size)
    base_tensors = read_safetensors_dir(base_dir)
    peft_model = attach_lora(model, spec, scenario, **attach_kw)
    adapted = extract_adapted(peft_model)
    run_merge(peft_model, base_dir, out_dir, save_dtype=dtype)
    return assert_merge_correct(
        family=family, base_tensors=base_tensors, out_dir=out_dir,
        save_dtype=dtype, adapted=adapted, allow_missing_adapted=allow_missing_adapted,
        base_dir=base_dir,
    )
