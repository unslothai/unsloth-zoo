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

"""fp16 full finetuning of Gemma3-VL / Gemma4-VL on a GPU without bfloat16 (T4, V100).

Those models force float32: the forced RMSNorm patches compute in float32 and emit fp16,
which matches the fp16 Linears of LoRA. Full finetuning upcasts the weights to float32 and,
without bfloat16, trains with no autocast, so every consumer of a norm the text-path fix
(test_gemma3_forced_float32_boundary_dtype.py) did not cover died with
"expected mat1 and mat2 to have the same dtype ... Half != float":
  * Gemma3MultiModalProjector: torch.matmul(mm_soft_emb_norm(x), mm_input_projection_weight)
  * Gemma4ClippableLinear (vision tower q / k / v / o / MLP projections)
  * Gemma4TextMLP (gate_proj) and the forced Gemma4 text attention (q_proj in, o_proj out)
  * SDPA in Gemma4VisionAttention: RoPE leaves q / k float32 while v_norm emits fp16
The real patched forwards run on CPU (dtype errors raise there too). fp16 weights (LoRA)
must stay bit-for-bit what they were.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")

from unsloth_zoo.temporary_patches import gemma as gemma_patches
from unsloth_zoo.temporary_patches import utils as patch_utils

try:
    from transformers.models.gemma3 import modeling_gemma3 as gemma3
    from transformers.models.gemma3 import Gemma3Config
    HAS_GEMMA3 = True
except Exception:  # pragma: no cover
    HAS_GEMMA3 = False

try:
    from transformers.models.gemma4 import modeling_gemma4 as gemma4
    from transformers.models.gemma4 import Gemma4TextConfig, Gemma4VisionConfig
    from unsloth_zoo.temporary_patches import gemma4 as gemma4_patches
    from unsloth_zoo.temporary_patches import gemma4_float32 as gemma4_float32_patches
    from unsloth_zoo.temporary_patches import gemma4_flash_sliding
    HAS_GEMMA4 = True
except Exception:  # pragma: no cover
    HAS_GEMMA4 = False

requires_gemma3 = pytest.mark.skipif(not HAS_GEMMA3, reason = "transformers gemma3 not installed")
requires_gemma4 = pytest.mark.skipif(not HAS_GEMMA4, reason = "transformers gemma4 not installed")


@pytest.fixture
def forced(monkeypatch):
    """UNSLOTH_FORCE_FLOAT32 on, compile off, every class forward restored at teardown."""
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    monkeypatch.setattr(patch_utils, "UNSLOTH_COMPILE_DISABLE", True)
    monkeypatch.setattr(torch._dynamo.config, "disable", True)
    targets = []
    if HAS_GEMMA3:
        targets += [gemma3.Gemma3RMSNorm, gemma3.Gemma3MultiModalProjector]
    if HAS_GEMMA4:
        targets += [gemma4.Gemma4RMSNorm, gemma4.Gemma4ClippableLinear, gemma4.Gemma4TextMLP]
        # The forced Gemma4 patches also install on the gemma4_unified copies; restore those too.
        for module, prefix in gemma4_float32_patches._gemma4_text_variants():
            for suffix in ("RMSNorm", "ClippableLinear"):
                cls = getattr(module, prefix + suffix, None)
                if cls is not None and cls not in targets:
                    targets.append(cls)
    originals = {t: t.forward for t in targets}
    for t in targets:
        monkeypatch.setattr(t, "forward", t.forward)
    return originals


def _apply(patches, name):
    """Run a zoo patch by name; a tree without it runs stock transformers, so the test shows
    the real failure there instead of an AttributeError."""
    patch = getattr(patches, name, None)
    if patch is not None:
        patch()


def _seen_dtypes(module):
    seen = []
    module.register_forward_pre_hook(lambda _m, args: seen.append(args[0].dtype))
    return seen


def _gemma3_projector(dtype):
    config = Gemma3Config(
        text_config = dict(hidden_size = 32, intermediate_size = 64, num_hidden_layers = 1,
                           num_attention_heads = 2, num_key_value_heads = 1, head_dim = 16, vocab_size = 64),
        vision_config = dict(hidden_size = 16, intermediate_size = 32, num_hidden_layers = 1,
                             num_attention_heads = 2, image_size = 28, patch_size = 7),
        mm_tokens_per_image = 4,
    )
    torch.manual_seed(0)
    projector = gemma3.Gemma3MultiModalProjector(config)
    with torch.no_grad():
        projector.mm_input_projection_weight.normal_(0, 0.05)
    return projector.to(dtype), torch.randn(2, 16, 16, dtype = dtype)


@requires_gemma3
def test_gemma3_projector_float32_full_finetune(forced):
    gemma_patches.patch_Gemma3RMSNorm()
    _apply(gemma_patches, "patch_Gemma3MultiModalProjector")
    projector, vision_outputs = _gemma3_projector(torch.float32)
    # The forced norm really does hand the matmul fp16: the crash precondition.
    assert projector.mm_soft_emb_norm(torch.randn(1, 4, 16)).dtype == torch.float16

    out = projector(vision_outputs)

    assert out.dtype == torch.float32 and out.shape == (2, 4, 32)
    assert torch.isfinite(out).all()


@requires_gemma3
def test_gemma3_projector_float16_lora_is_bitwise_unchanged(forced):
    gemma_patches.patch_Gemma3RMSNorm()
    projector, vision_outputs = _gemma3_projector(torch.float16)
    expected = forced[gemma3.Gemma3MultiModalProjector](projector, vision_outputs)
    _apply(gemma_patches, "patch_Gemma3MultiModalProjector")

    out = projector(vision_outputs)

    assert out.dtype == torch.float16
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


def _gemma4_vision_config():
    return Gemma4VisionConfig(hidden_size = 16, intermediate_size = 32, num_hidden_layers = 1,
                              num_attention_heads = 2, num_key_value_heads = 2, head_dim = 8,
                              use_clipped_linears = True)


@requires_gemma4
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_gemma4_clippable_linear_meets_its_weight(forced, dtype):
    torch.manual_seed(0)
    layer = gemma4.Gemma4ClippableLinear(_gemma4_vision_config(), 16, 8).to(dtype)
    x = torch.randn(2, 3, 16, dtype = torch.float16)
    expected = forced[gemma4.Gemma4ClippableLinear](layer, x) if dtype == torch.float16 else None
    gemma4_float32_patches.patch_Gemma4RMSNorm()
    _apply(gemma4_float32_patches, "patch_Gemma4ClippableLinear")
    seen = _seen_dtypes(layer.linear)

    out = layer(x)

    assert seen == [dtype] and out.dtype == dtype
    if expected is not None:
        assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


@requires_gemma4
def test_gemma4_text_mlp_float32_full_finetune(forced):
    config = Gemma4TextConfig(hidden_size = 16, intermediate_size = 32, num_hidden_layers = 1,
                              num_attention_heads = 2, num_key_value_heads = 1, head_dim = 8,
                              vocab_size = 64, vocab_size_per_layer_input = 64, hidden_size_per_layer_input = 8)
    torch.manual_seed(0)
    mlp = gemma4.Gemma4TextMLP(config, layer_idx = 0).to(torch.float32)
    _apply(gemma4_patches, "patch_Gemma4TextMLP")
    seen = _seen_dtypes(mlp.gate_proj)

    out = mlp(torch.randn(2, 3, 16, dtype = torch.float16))

    assert seen == [torch.float32] and out.dtype == torch.float32
    assert torch.isfinite(out).all()


@requires_gemma4
def test_gemma4_sdpa_promotes_mixed_qkv_and_leaves_matching_dtypes_alone(monkeypatch):
    calls = []

    def fake_sdpa(module, query, key, value, attention_mask, **kwargs):
        calls.append((query.dtype, key.dtype, value.dtype))
        return query, None

    monkeypatch.setattr(gemma4_flash_sliding, "_ORIG_SDPA", [fake_sdpa])
    module = torch.nn.Module()
    q = torch.randn(1, 2, 4, 8)
    v16 = torch.randn(1, 2, 4, 8, dtype = torch.float16)
    gemma4_flash_sliding._sdpa_maybe_flash_sliding(module, q, q, v16, None)
    gemma4_flash_sliding._sdpa_maybe_flash_sliding(module, v16, v16, v16, None)
    assert calls == [(torch.float32,) * 3, (torch.float16,) * 3]


@requires_gemma4
@pytest.mark.parametrize("autocast", [False, True], ids = ["no_autocast", "autocast"])
def test_gemma4_text_attention_meets_full_finetune_weights_only_without_autocast(forced, monkeypatch, autocast):
    # fp32 weights (full finetuning). Without autocast the fp16 norm output meets them; under autocast
    # (fp16 forced on a bf16 GPU) the Linears cast for themselves, so the inputs stay fp16 as before.
    monkeypatch.setattr(gemma4.Gemma4TextAttention, "forward", gemma4.Gemma4TextAttention.forward)
    config = Gemma4TextConfig(
        vocab_size = 32, hidden_size = 16, intermediate_size = 32, num_hidden_layers = 2,
        num_attention_heads = 2, num_key_value_heads = 1, head_dim = 8, global_head_dim = 8,
        num_global_key_value_heads = 1, layer_types = ["sliding_attention", "full_attention"],
        sliding_window = 2, vocab_size_per_layer_input = 32, hidden_size_per_layer_input = 8,
    )
    config._attn_implementation = "sdpa"
    gemma4_float32_patches.patch_Gemma4TextAttention()
    torch.manual_seed(0)
    attention = gemma4.Gemma4TextAttention(config, 0).to(torch.float32)
    q_seen, o_seen = _seen_dtypes(attention.q_proj), _seen_dtypes(attention.o_proj)
    angles = torch.randn(1, 3, 8)
    positions = torch.arange(3)
    mask = ((positions[:, None] >= positions[None, :]) & (positions[:, None] - positions[None, :] < 2))[None, None]
    inputs = torch.randn(1, 3, 16, dtype = torch.float16)
    if autocast:
        with torch.autocast("cpu", dtype = torch.float16):
            out, _ = attention(inputs, (angles.cos(), angles.sin()), mask, shared_kv_states = {})
        assert q_seen == [torch.float16] and o_seen == [torch.float16]
    else:
        out, _ = attention(inputs, (angles.cos(), angles.sin()), mask, shared_kv_states = {})
        assert q_seen == [torch.float32] and o_seen == [torch.float32]
    assert torch.isfinite(out).all()
