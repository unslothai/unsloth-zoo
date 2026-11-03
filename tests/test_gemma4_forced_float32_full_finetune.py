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

"""Gemma-4 is in FORCE_FLOAT32: on a GPU without bfloat16 an fp16 request sets UNSLOTH_FORCE_FLOAT32=1, and full
finetuning then trains the model in float32 with no autocast. The patched RMSNorm always returned float16, and every
decoder boundary casts back to its Linear's dtype, but the final norm feeds lm_head directly, so the first forward
died with "expected mat1 and mat2 to have the same dtype, but got: c10::Half != float" at lm_head (Kaggle T4, MoE
matrix, Gemma-4 loaded alone). Now the norms follow the projections' dtype when no autocast is active: a float32
model matches vanilla transformers in float32, and the fp16 paths (autocast training, no-autocast eval) still run.

CPU subprocesses: the patches gate on UNSLOTH_FORCE_FLOAT32 at patch time, and the reference never imports the zoo.
"""

import json
import os
import subprocess
import sys
import textwrap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COMMON = r'''
import json, os, sys, torch
OUT = sys.argv[1]
IDS = torch.randint(2, 256, (2, 12), generator = torch.Generator().manual_seed(0))

def build(dtype, moe):
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextConfig, Gemma4ForCausalLM
    cfg = Gemma4TextConfig(vocab_size = 256, hidden_size = 64, intermediate_size = 48, moe_intermediate_size = 40,
        num_hidden_layers = 2, num_attention_heads = 4, num_key_value_heads = 2, head_dim = 16, global_head_dim = 16,
        max_position_embeddings = 128, enable_moe_block = moe, num_experts = 4, top_k_experts = 2,
        num_kv_shared_layers = 0, hidden_size_per_layer_input = 0, layer_types = ["sliding_attention", "full_attention"],
        sliding_window = 64, tie_word_embeddings = False, use_cache = False, pad_token_id = 0, bos_token_id = 1,
        eos_token_id = 1)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    return Gemma4ForCausalLM(cfg).to(dtype)

def full_ft(model):
    model.train()
    out = model(input_ids = IDS, labels = IDS)
    out.loss.backward()
    grads = {n: p.grad.float() for n, p in model.named_parameters() if p.grad is not None}
    return out.loss.item(), grads
'''

PATCHED = COMMON + r'''
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
res = {}
# First forward of the process, through inputs_embeds (the embedding never sets the boundary) under non-reentrant
# checkpointing: the first input_layernorm must pick the same kernel when it is recomputed (else CheckpointError).
gc_model = build(torch.float32, False)
gc_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
gc_model.train()
emb = gc_model.get_input_embeddings()
embeds = (torch.nn.functional.embedding(IDS, emb.weight) * emb.embed_scale).detach()  # not through the module
gc_model(inputs_embeds = embeds, labels = IDS).loss.backward()
for moe in (False, True):
    loss, grads = full_ft(build(torch.float32, moe))
    torch.save(grads, os.path.join(OUT, f"patched_{int(moe)}.pt"))
    # The fp16 paths the patches exist for must still run: autocast training and no-autocast eval.
    m16 = build(torch.float16, moe)
    with torch.autocast("cpu", dtype = torch.float16):
        loss16 = m16(input_ids = IDS, labels = IDS).loss.item()
    m16.eval()
    with torch.no_grad():
        eval_dtype = str(m16(input_ids = IDS).logits.dtype)
    res[str(int(moe))] = {"loss": loss, "loss16": loss16, "eval_dtype": eval_dtype}
print(json.dumps(res))
'''

VANILLA = COMMON + r'''
assert "unsloth_zoo" not in sys.modules
res = {}
for moe in (False, True):
    loss, grads = full_ft(build(torch.float32, moe))
    torch.save(grads, os.path.join(OUT, f"vanilla_{int(moe)}.pt"))
    res[str(int(moe))] = {"loss": loss}
print(json.dumps(res))
'''


def _run(code, out, zoo):
    env = {k: v for k, v in os.environ.items() if not k.startswith("UNSLOTH_")}
    env.update(HF_HUB_OFFLINE = "1", OMP_NUM_THREADS = "4", REPO = REPO,
               UNSLOTH_COMPILE_LOCATION = os.path.join(out, "ucc"))
    if zoo:
        # A fresh interpreter lacks conftest.py's mocks: configure the CPU import here.
        env.update(PYTHONPATH = os.pathsep.join(p for p in (REPO, os.environ.get("PYTHONPATH")) if p),
                   CUDA_VISIBLE_DEVICES = "", UNSLOTH_IS_PRESENT = "1", UNSLOTH_ALLOW_CPU = "1",
                   UNSLOTH_ZOO_DISABLE_GPU_INIT = "1", UNSLOTH_FORCE_FLOAT32 = "1")
    else:
        env.pop("PYTHONPATH", None)
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(code), out], env = env, cwd = out,
                          capture_output = True, text = True, timeout = 1800)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    return json.loads([ln for ln in proc.stdout.splitlines() if ln.startswith("{")][-1])


def test_gemma4_forced_float32_full_finetune_matches_vanilla_float32(tmp_path):
    import pytest
    import torch

    try:
        import transformers.models.gemma4.modeling_gemma4  # noqa: F401
    except Exception:
        pytest.skip("transformers without Gemma-4")
    out = str(tmp_path)
    patched = _run(PATCHED, out, zoo = True)
    vanilla = _run(VANILLA, out, zoo = False)
    for moe in ("0", "1"):
        p, v = patched[moe], vanilla[moe]
        assert abs(p["loss"] - v["loss"]) <= 1e-4 * abs(v["loss"]), (moe, p, v)
        got = torch.load(os.path.join(out, f"patched_{moe}.pt"))
        want = torch.load(os.path.join(out, f"vanilla_{moe}.pt"))
        assert set(got) == set(want), moe
        for name, g in want.items():
            torch.testing.assert_close(got[name], g, rtol = 1e-3, atol = 1e-5 * float(g.abs().max() or 1.0),
                                       msg = f"moe={moe} {name}")
        assert abs(p["loss16"] - v["loss"]) < 0.05, (moe, p)
        assert p["eval_dtype"] == "torch.float16", (moe, p)


def test_decoder_boundary_patch_is_undone_by_a_flag_off_load(monkeypatch):
    import pytest

    try:
        import transformers.models.gemma4.modeling_gemma4 as g
    except Exception:
        pytest.skip("transformers without Gemma-4")
    from unsloth_zoo.temporary_patches import gemma4_float32 as p

    original = g.Gemma4TextDecoderLayer.forward
    monkeypatch.delenv("UNSLOTH_MODEL_NAME", raising = False)
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    p.patch_Gemma4TextDecoderLayerBoundary()
    p.patch_Gemma4TextDecoderLayerBoundary()  # idempotent: one wrapper
    assert g.Gemma4TextDecoderLayer.forward.__wrapped__ is original
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "0")
    p.patch_Gemma4TextDecoderLayerBoundary()
    assert g.Gemma4TextDecoderLayer.forward is original


VISION = r'''
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
text = g.Gemma4TextConfig(vocab_size = 256, hidden_size = 64, intermediate_size = 48, num_hidden_layers = 2,
    num_attention_heads = 4, num_key_value_heads = 2, head_dim = 16, global_head_dim = 16, max_position_embeddings = 128,
    enable_moe_block = False, num_kv_shared_layers = 0, hidden_size_per_layer_input = 0,
    layer_types = ["sliding_attention", "full_attention"], sliding_window = 64, tie_word_embeddings = False,
    use_cache = False, pad_token_id = 0, bos_token_id = 1, eos_token_id = 1)
vision = g.Gemma4VisionConfig(hidden_size = 32, intermediate_size = 64, num_attention_heads = 2, num_key_value_heads = 2,
    head_dim = 16, num_hidden_layers = 1, patch_size = 2, pooling_kernel_size = 2, standardize = False,
    position_embedding_size = 8)
config = g.Gemma4Config(text_config = text, vision_config = vision, audio_config = None, image_token_id = 250,
    video_token_id = 251, audio_token_id = 252, boi_token_id = 253, eoi_token_id = 254)
config._attn_implementation = "sdpa"
torch.manual_seed(0)
model = g.Gemma4ForConditionalGeneration(config).float().train()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
# Image features before the text embedding: the vision norms run first with no boundary set by the text tower.
features = model.get_image_features(pixel_values = torch.rand(1, 4, 12),
    image_position_ids = torch.tensor([[[0, 0], [1, 0], [0, 1], [1, 1]]]), return_dict = True).pooler_output
embeddings = model.get_input_embeddings()(torch.tensor([[3, 4, 5, 6]]))
(sum(f.square().mean() for f in features) + embeddings.square().mean()).backward()
print(json.dumps({"ok": True}))
'''


def test_gemma4_vision_layers_recompute_with_the_same_norm_kernel(tmp_path):
    # Guards the scoped design: without a vision scope, norms run fp16 then float32 on recomputation (CheckpointError).
    import pytest

    try:
        import transformers.models.gemma4.modeling_gemma4  # noqa: F401
    except Exception:
        pytest.skip("transformers without Gemma-4")
    assert _run(VISION, str(tmp_path), zoo = True) == {"ok": True}


SCOPE = r'''
import json, os, sys, torch
import unsloth_zoo, unsloth_zoo.temporary_patches
from unsloth_zoo.temporary_patches.common import TEMPORARY_PATCHES
for fn in TEMPORARY_PATCHES:
    try:
        fn()
    except Exception:
        pass
from unsloth_zoo.temporary_patches import gemma4_float32 as p
import transformers.models.gemma4.modeling_gemma4 as g
cfg = g.Gemma4TextConfig(vocab_size = 256, hidden_size = 64, intermediate_size = 48, num_hidden_layers = 2,
    num_attention_heads = 4, num_key_value_heads = 2, head_dim = 16, global_head_dim = 16, max_position_embeddings = 128,
    enable_moe_block = False, num_kv_shared_layers = 0, hidden_size_per_layer_input = 0,
    layer_types = ["sliding_attention", "full_attention"], sliding_window = 64, tie_word_embeddings = False,
    use_cache = False, pad_token_id = 0, bos_token_id = 1, eos_token_id = 1)
cfg._attn_implementation = "eager"
model = g.Gemma4ForCausalLM(cfg).float()
ids = torch.randint(2, 256, (1, 8))
model(input_ids = ids, labels = ids).loss.backward()
model.get_input_embeddings()(ids)  # a standalone embedding call
# Outside the text model nothing is left behind: a norm feeding an fp16 projector stays fp16.
print(json.dumps({"after": str(getattr(p._GEMMA4_BOUNDARY, "dtype", None))}))
'''


def test_gemma4_boundary_does_not_leak_past_its_scope(tmp_path):
    # Failing before: a stale float32 boundary made the multimodal embedder norm return float32 into its fp16
    # projector ("expected m1 and m2 to have the same dtype, but got: float != c10::Half").
    import pytest

    try:
        import transformers.models.gemma4.modeling_gemma4  # noqa: F401
    except Exception:
        pytest.skip("transformers without Gemma-4")
    assert _run(SCOPE, str(tmp_path), zoo = True) == {"after": "None"}
