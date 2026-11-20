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

"""Forced float32 Gemma-4: residual post-norms must return float32 so the residual gradient never hits fp16."""

import json
import os
import subprocess
import sys
import textwrap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCRIPT = r'''
import json, os, sys, torch
import unsloth_zoo, unsloth_zoo.temporary_patches
from unsloth_zoo.temporary_patches.common import TEMPORARY_PATCHES
assert unsloth_zoo.__file__.startswith(os.environ["REPO"]), unsloth_zoo.__file__
for fn in TEMPORARY_PATCHES:
    try:
        fn()
    except Exception:
        pass
import transformers.models.gemma4.modeling_gemma4 as g
assert "gemma4_float32" in g.Gemma4RMSNorm.forward.__code__.co_filename, "float32 patches not applied"
POST = ("post_attention_layernorm", "post_feedforward_layernorm", "post_per_layer_input_norm")
IDS = torch.randint(2, 256, (2, 12), generator = torch.Generator().manual_seed(0))
res = {}
for moe in (False, True):
    cfg = g.Gemma4TextConfig(vocab_size = 256, hidden_size = 64, intermediate_size = 48, moe_intermediate_size = 40,
        num_hidden_layers = 2, num_attention_heads = 4, num_key_value_heads = 2, head_dim = 16, global_head_dim = 16,
        max_position_embeddings = 128, enable_moe_block = moe, num_experts = 4, top_k_experts = 2,
        num_kv_shared_layers = 0, hidden_size_per_layer_input = 8, vocab_size_per_layer_input = 256,
        layer_types = ["sliding_attention", "full_attention"], sliding_window = 64, tie_word_embeddings = False,
        use_cache = False, pad_token_id = 0, bos_token_id = 1, eos_token_id = 1)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    model = g.Gemma4ForCausalLM(cfg).to(torch.float16).train()
    with torch.no_grad():
        # Small post-norm gains put the sub-layer gradients far below the residual gradient; a large final-norm
        # gain makes the residual gradient the largest in the backward, as in the real model.
        for layer in model.model.layers:
            for name in POST:
                if getattr(layer, name, None) is not None:
                    getattr(layer, name).weight.fill_(2.0 ** -14)
        model.model.norm.weight.fill_(50.0)
    for p in model.parameters():  # as LoRA: only the decoder projections get gradients, norms and embeddings stay frozen
        p.requires_grad_(False)
    for mod in model.model.layers.modules():
        if isinstance(mod, torch.nn.Linear):
            mod.weight.requires_grad_(True)
    seen, grads_out, f16_amax = {}, [], [0.0, ""]

    def track(mod, args, out):
        if isinstance(out, torch.Tensor) and out.requires_grad and out.dtype == torch.float16:
            def h(gr, mod = mod):
                v = float(gr.float().abs().nan_to_num(0, 0, 0).max())
                if v > f16_amax[0]:
                    f16_amax[:] = [v, type(mod).__name__]
            out.register_hook(h)

    for layer in model.model.layers:
        for name in POST:
            mod = getattr(layer, name, None)
            if mod is None:
                continue
            def hook(mod, args, out, name = name):
                seen.setdefault(name, set()).add(str(out.dtype))
                if out.requires_grad:
                    out.register_hook(lambda gr: grads_out.append(float(gr.float().abs().max())))
            mod.register_forward_hook(hook)
    for mod in model.model.layers.modules():
        if not any(mod is getattr(layer, name, None) for layer in model.model.layers for name in POST):
            mod.register_forward_hook(track)
    scale = float(os.environ["LOSS_SCALE"])
    out = model(input_ids = IDS, labels = IDS)
    (out.loss * scale).backward()
    finite = all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
    res[str(int(moe))] = {"loss": out.loss.item(), "dtypes": {k: sorted(v) for k, v in seen.items()},
                          "finite": finite, "residual_grad_amax": max(grads_out), "fp16_grad_amax": f16_amax[0], "fp16_grad_argmax": f16_amax[1]}
print(json.dumps(res))
'''


def _run(out, scale, post_norm_fp32):
    env = {k: v for k, v in os.environ.items() if not k.startswith("UNSLOTH_")}
    env.update(HF_HUB_OFFLINE = "1", OMP_NUM_THREADS = "4", REPO = REPO, LOSS_SCALE = str(scale),
               UNSLOTH_COMPILE_LOCATION = os.path.join(out, "ucc"),
               PYTHONPATH = os.pathsep.join(p for p in (REPO, os.environ.get("PYTHONPATH")) if p),
               CUDA_VISIBLE_DEVICES = "", UNSLOTH_IS_PRESENT = "1", UNSLOTH_ALLOW_CPU = "1",
               UNSLOTH_ZOO_DISABLE_GPU_INIT = "1", UNSLOTH_FORCE_FLOAT32 = "1", TORCHDYNAMO_DISABLE = "1",
               UNSLOTH_GEMMA4_FP32_POST_NORM = "1" if post_norm_fp32 else "0")
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(SCRIPT)], env = env, cwd = out,
                          capture_output = True, text = True, timeout = 1800)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    return json.loads([ln for ln in proc.stdout.splitlines() if ln.startswith("{")][-1])


def test_gemma4_post_norms_return_float32_and_keep_the_residual_gradient_out_of_float16(tmp_path):
    import pytest

    try:
        import transformers.models.gemma4.modeling_gemma4  # noqa: F401
    except Exception:
        pytest.skip("transformers without Gemma-4")
    out = str(tmp_path)
    probe = _run(out, 1.0, True)
    fp16_max = 65504.0
    for moe, r in probe.items():
        assert r["finite"], (moe, r)
        assert set(r["dtypes"]) == {"post_attention_layernorm", "post_feedforward_layernorm", "post_per_layer_input_norm"}, r
        assert all(v == ["torch.float32"] for v in r["dtypes"].values()), (moe, r["dtypes"])
    # Residual gradient past fp16 max, every fp16 gradient 4x below it.
    res_amax = min(r["residual_grad_amax"] for r in probe.values())
    f16_amax = max(r["fp16_grad_amax"] for r in probe.values())
    scale = 2.0 ** int(__import__("math").ceil(__import__("math").log2(2 * fp16_max / res_amax)))
    assert f16_amax * scale < fp16_max / 4, (probe, scale)
    head = _run(out, scale, True)
    base = _run(out, scale, False)
    for moe in ("0", "1"):
        assert head[moe]["finite"], (moe, head[moe], scale)
        assert not base[moe]["finite"], (moe, base[moe], scale)  # kill switch: float16 post-norm output overflows
        assert all(v == ["torch.float16"] for v in base[moe]["dtypes"].values()), base[moe]["dtypes"]
        assert abs(head[moe]["loss"] - base[moe]["loss"]) < 1e-2 * abs(base[moe]["loss"]), (head[moe], base[moe])
