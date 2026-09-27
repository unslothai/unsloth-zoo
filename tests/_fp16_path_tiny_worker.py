# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Worker for test_fp16_path_tiny_models.py, in its own process because temporary
patches rewrite transformers classes process-wide. Runs every arch in bf16 first,
then re-applies the patches with UNSLOTH_FORCE_FLOAT32=1 (as the loader does per
load) and runs the T4 path: forced-float32 cast (or plain fp16) + fp16 autocast.

argv: arch names. Prints one JSON line per (mode, arch).
"""

import importlib
import importlib.util
import json
import math
import os
import sys
import traceback

ARCHS = sys.argv[1:]
MODE = "bf16"
os.environ["UNSLOTH_FORCE_FLOAT32"] = "0"
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # no Hub kernel lookups from CI
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")  # CPU Inductor builds C++ probes; compile is GPU-tested
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("UNSLOTH_ALLOW_CPU", "1")
os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

import torch  # noqa: E402

torch.set_num_threads(4)  # 96 threads on tiny tensors cost more than they save
torch.manual_seed(0)
SMALL = dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_attention_heads=2,
             num_key_value_heads=1, head_dim=16)


def _mod(model_type):
    return (importlib.import_module(f"transformers.models.{model_type}.modeling_{model_type}"),
            importlib.import_module(f"transformers.models.{model_type}.configuration_{model_type}"))


def _gemma4(model_type, prefix, **extra):
    m, c = _mod(model_type)
    kw = dict(SMALL, num_hidden_layers=4, global_head_dim=16, num_global_key_value_heads=1,
              layer_types=["sliding_attention", "full_attention"] * 2, sliding_window=4)
    kw.update(extra)
    config = getattr(c, prefix + "TextConfig")(**kw)
    return getattr(m, prefix + "ForCausalLM")(config)


def _qwen3_5(model_type, prefix, **extra):
    m, c = _mod(model_type)
    kw = dict(SMALL, num_hidden_layers=2, layer_types=["linear_attention", "full_attention"],
              linear_conv_kernel_dim=2, linear_key_head_dim=8, linear_value_head_dim=8,
              linear_num_key_heads=2, linear_num_value_heads=2)
    kw.update(extra)
    config = getattr(c, prefix + "TextConfig")(**kw)
    return getattr(m, prefix + "ForCausalLM")(config)


def _muse_glimmer():
    m, c = _mod("muse_glimmer")
    text = dict(SMALL, num_hidden_layers=2, layer_types=["sliding_attention", "full_attention"],
                sliding_window=4)
    config = c.MuseGlimmerConfig(text_config=text, vision_config=dict(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2))
    return m.MuseGlimmerForConditionalGeneration(config)


# name -> (builder, forced float32 on a T4). Names track the real checkpoints they shrink.
BUILDERS = {
    "gemma4_e2b": (lambda: _gemma4("gemma4", "Gemma4", hidden_size_per_layer_input=8,
                                   vocab_size_per_layer_input=64, num_kv_shared_layers=2), True),
    "gemma4_26b_a4b_moe": (lambda: _gemma4("gemma4", "Gemma4", hidden_size_per_layer_input=0,
                                           enable_moe_block=True, num_experts=4, top_k_experts=2,
                                           moe_intermediate_size=16, attention_k_eq_v=True), True),
    "gemma4_unified_12b": (lambda: _gemma4("gemma4_unified", "Gemma4Unified", attention_k_eq_v=True), True),
    "qwen3_5_fla": (lambda: _qwen3_5("qwen3_5", "Qwen3_5"), True),
    "qwen3_6_moe": (lambda: _qwen3_5("qwen3_5_moe", "Qwen3_5Moe", num_experts=4, num_experts_per_tok=2,
                                     moe_intermediate_size=16, shared_expert_intermediate_size=16), True),
    "muse_glimmer": (_muse_glimmer, False),
}


# Patch modules these architectures route through; the rest (vLLM, TRL, datasets) cost
# seconds and touch none of them.
_PATCH_MODULES = ("common", "misc", "fla_vendor", "gemma4", "gemma4_float32", "gemma4_moe",
                  "gemma4_banded_attention", "gemma4_flash_sliding", "qwen3_5_moe",
                  "muse_glimmer_banded_attention", "moe_experts_interface", "moe_grouped_modulelist")


def _strict(fn, name):
    # On GPU the T4 path reaches flex attention, which is not an autocast op and
    # rejects mixed q/k/v dtypes. SDPA/eager under CPU autocast would unify them
    # silently, so check the invariant at the call instead.
    def wrapper(*args, **kwargs):
        q, k, v = (args[i] if len(args) > i else kwargs[n] for i, n in ((0, "query"), (1, "key"), (2, "value")))
        if isinstance(q, torch.nn.Module):  # attention-interface signature: (module, q, k, v, ...)
            q, k, v = args[1], args[2], args[3]
        if not (q.dtype == k.dtype == v.dtype):
            raise RuntimeError(f"{name}: mixed q/k/v dtypes {q.dtype} {k.dtype} {v.dtype}")
        return fn(*args, **kwargs)
    return wrapper


torch.nn.functional.scaled_dot_product_attention = _strict(torch.nn.functional.scaled_dot_product_attention, "sdpa")
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402

for _name in list(ALL_ATTENTION_FUNCTIONS.keys()):
    ALL_ATTENTION_FUNCTIONS[_name] = _strict(ALL_ATTENTION_FUNCTIONS[_name], _name)


MODEL_TYPES = {"gemma4_e2b": "gemma4", "gemma4_26b_a4b_moe": "gemma4", "gemma4_unified_12b": "gemma4_unified",
               "qwen3_5_fla": "qwen3_5", "qwen3_6_moe": "qwen3_5_moe", "muse_glimmer": "muse_glimmer"}


def _apply_temporary_patches():
    from unsloth_zoo.temporary_patches import TEMPORARY_PATCHES
    for patch in TEMPORARY_PATCHES:
        if patch.__module__.rsplit(".", 1)[-1] not in _PATCH_MODULES:
            continue
        try:
            patch()
        except Exception:
            pass  # CUDA/Triton-only patches; the loader tolerates the same failures.


def run(arch):
    builder, forced = BUILDERS[arch]
    out = {"arch": arch, "mode": MODE, "ok": False, "error": None}
    model_type = MODEL_TYPES[arch]
    if importlib.util.find_spec(f"transformers.models.{model_type}") is None:
        out["error"] = f"UNAVAILABLE: transformers has no {model_type}"
        return out
    try:
        model = builder().to(torch.bfloat16)  # build errors on an installed arch are drift: fail
        if MODE == "t4" and forced:
            from unsloth_zoo.patching_utils import patch_model_and_tokenizer
            patch_model_and_tokenizer(model, None, do_forced_float32 = True)
            autocast = torch.float16
        elif MODE == "t4":
            model = model.to(torch.float16)
            autocast = torch.float16
        else:
            autocast = torch.bfloat16
        model.train()
        ids = torch.randint(3, 64, (2, 8))
        with torch.autocast("cpu", dtype = autocast):
            loss = model(input_ids = ids, labels = ids, use_cache = False).loss
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        dtypes = sorted({str(p.dtype) for p in model.parameters()})
        out.update(
            loss = float(loss),
            n_grads = len(grads),
            grads_finite = all(torch.isfinite(g).all().item() for g in grads),
            any_grad_nonzero = any(g.abs().sum().item() > 0 for g in grads),
            dtypes = dtypes,
        )
        out["ok"] = (math.isfinite(out["loss"]) and out["grads_finite"] and out["any_grad_nonzero"]
                     and not (MODE == "t4" and "torch.bfloat16" in dtypes))
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"[:600]
        out["trace"] = traceback.format_exc()[-1500:]
    return out


import time  # noqa: E402

for MODE in ("bf16", "t4"):
    os.environ["UNSLOTH_FORCE_FLOAT32"] = "1" if MODE == "t4" else "0"
    _apply_temporary_patches()
    for arch in ARCHS:
        _t = time.perf_counter()
        result = run(arch)
        result["secs"] = round(time.perf_counter() - _t, 2)
        print("RESULT " + json.dumps(result), flush = True)
