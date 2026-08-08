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

"""CPU regression tests for the Gemma3 UNSLOTH_FORCE_FLOAT32 Linear boundary dtype fix.

The forced-float32 Gemma3 patches run RMSNorm / SwiGLU / attention reductions in
float32 and hand a float16 activation to every Linear. That hard-coded float16
assumed the projection weights were float16 too: true for LoRA / QLoRA, false for
full finetuning, which upcasts the trainable weights to float32. A float16
activation then met a float32 weight and the matmul died with
"expected mat1 and mat2 to have the same dtype, but got: c10::Half != float"
(the CPU wording is "m1 and m2", same error).

``_linear_boundary_dtype`` reads the dtype off the projections that actually do
the multiply and ``_to_boundary_dtype`` moves the activation there, so:

  * float32 projections (full finetuning) no longer crash, and stay in float32;
  * float16 projections (LoRA / QLoRA) hit an identity branch, so the common path
    keeps bit-for-bit the numerics it had before the fix;
  * a bitsandbytes 4bit uint8 weight blob is not floating point, so it falls
    through to the float16 default that path always used.

Everything here is behavioural: the real patched ``Gemma3MLP`` / ``Gemma3Attention``
forwards are executed on CPU with real transformers modules built from a tiny real
Gemma3 text config. No test inspects the source text of the patch. dtype-mismatch
errors raise on CPU, so no GPU is needed.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")

try:
    from transformers.models.gemma3 import modeling_gemma3 as gemma3
    from transformers.models.gemma3 import Gemma3TextConfig
    HAS_GEMMA3 = True
except Exception:  # pragma: no cover - transformers too old for gemma3
    gemma3 = None
    Gemma3TextConfig = None
    HAS_GEMMA3 = False

from unsloth_zoo.temporary_patches import gemma as gemma_patches
from unsloth_zoo.temporary_patches import utils as patch_utils
from unsloth_zoo.temporary_patches.gemma import (
    _linear_boundary_dtype,
    _to_boundary_dtype,
)

requires_gemma3 = pytest.mark.skipif(not HAS_GEMMA3, reason="transformers gemma3 not installed")

# Tiny but real Gemma3 text config: head_dim * num_attention_heads == hidden_size.
HIDDEN_SIZE, INTERMEDIATE_SIZE, HEADS, KV_HEADS, HEAD_DIM = 16, 32, 2, 1, 8


def _text_config():
    return Gemma3TextConfig(
        vocab_size            = 64,
        hidden_size           = HIDDEN_SIZE,
        intermediate_size     = INTERMEDIATE_SIZE,
        num_hidden_layers     = 1,
        num_attention_heads   = HEADS,
        num_key_value_heads   = KV_HEADS,
        head_dim              = HEAD_DIM,
    )


def _record_input_dtypes(module):
    """Collect the dtype every call actually hands to ``module``."""
    seen = []
    module.register_forward_pre_hook(lambda _module, args: seen.append(args[0].dtype))
    return seen


def _install(monkeypatch, force_float32, patchers):
    """Run the real zoo patch bodies against transformers, then undo them.

    ``monkeypatch.setattr(cls, "forward", cls.forward)`` records the pre-patch
    function so teardown restores it: these patches mutate the global transformers
    classes and must not leak into other test files. torch.compile is switched off
    so the assertions see the patched Python itself rather than a Dynamo graph
    (and so the bitwise test compares against eager arithmetic).
    """
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1" if force_float32 else "0")
    monkeypatch.setattr(patch_utils, "UNSLOTH_COMPILE_DISABLE", True)
    monkeypatch.setattr(gemma_patches, "torch_compile", lambda function, *args, **kwargs: function)

    targets = (gemma3.Gemma3MLP, gemma3.Gemma3Attention)
    for target in targets:
        monkeypatch.setattr(target, "forward", target.forward)
    before = {target: target.forward for target in targets}

    for patcher in patchers:
        patcher()

    # Guard against a vacuous run: patch_function() silently declines to patch when
    # the upstream signature drifts, and then these tests would exercise stock
    # transformers and pass no matter what the fix does.
    patched = {target for target in targets if target.forward is not before[target]}
    return patched


@pytest.fixture
def forced_float32_patches(monkeypatch):
    patched = _install(
        monkeypatch,
        force_float32 = True,
        patchers = (gemma_patches.patch_Gemma3MLP, gemma_patches.patch_Gemma3Attention),
    )
    assert gemma3.Gemma3MLP in patched, "patch_Gemma3MLP did not install its forward"
    assert gemma3.Gemma3Attention in patched, "patch_Gemma3Attention did not install its forward"
    yield


@pytest.fixture
def generic_patches(monkeypatch):
    patched = _install(
        monkeypatch,
        force_float32 = False,
        patchers = (gemma_patches.patch_Gemma3Attention_generic,),
    )
    assert gemma3.Gemma3Attention in patched, "patch_Gemma3Attention_generic did not install its forward"
    yield


@requires_gemma3
def test_forced_float32_mlp_runs_float16_activation_into_float32_projections(forced_float32_patches):
    """Full finetuning: fp32 MLP weights fed the fp16 activation RMSNorm emits.

    Before the fix ``self.gate_proj(x)`` raised RuntimeError here.
    """
    torch.manual_seed(0)
    mlp = gemma3.Gemma3MLP(_text_config()).to(torch.float32)
    gate_seen = _record_input_dtypes(mlp.gate_proj)
    up_seen = _record_input_dtypes(mlp.up_proj)
    down_seen = _record_input_dtypes(mlp.down_proj)
    x = torch.randn(2, 3, HIDDEN_SIZE, dtype = torch.float16)

    out = mlp(x)

    assert out.dtype == torch.float32
    assert out.shape == (2, 3, HIDDEN_SIZE)
    assert torch.isfinite(out).all()
    # Every Linear must have been handed its own weight dtype, not a bare float16.
    assert gate_seen == [torch.float32]
    assert up_seen == [torch.float32]
    assert down_seen == [torch.float32]


@requires_gemma3
def test_forced_float32_mlp_float16_weights_are_bitwise_identical_to_pre_fix(forced_float32_patches):
    """LoRA / QLoRA path: fp16 weights must reproduce the old algorithm exactly."""
    torch.manual_seed(0)
    mlp = gemma3.Gemma3MLP(_text_config()).to(torch.float16)
    gate_seen = _record_input_dtypes(mlp.gate_proj)
    down_seen = _record_input_dtypes(mlp.down_proj)
    x = torch.randn(2, 3, HIDDEN_SIZE, dtype = torch.float16)

    out = mlp(x)

    # Pre-fix algorithm, inline: projections in fp16, upcast, act_fn and product in
    # fp32, bare .to(torch.float16), down_proj.
    gate_proj_out = mlp.gate_proj(x)
    up_proj_out = mlp.up_proj(x)
    gate_proj_fp32 = gate_proj_out.to(torch.float32)
    up_proj_fp32 = up_proj_out.to(torch.float32)
    activated_fp32 = mlp.act_fn(gate_proj_fp32)
    intermediate_fp32 = activated_fp32 * up_proj_fp32
    intermediate_fp16 = intermediate_fp32.to(torch.float16)
    expected = mlp.down_proj(intermediate_fp16)

    assert out.dtype == torch.float16
    assert expected.dtype == torch.float16
    # Bitwise, via the raw fp16 bit patterns rather than float comparison.
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
    assert gate_seen[0] == torch.float16
    assert down_seen[0] == torch.float16


@requires_gemma3
def test_forced_float32_attention_runs_float16_hidden_states_into_float32_projections(forced_float32_patches):
    """The exact crash site from the traceback: fp32 q_proj fed fp16 hidden_states."""
    torch.manual_seed(0)
    attention = gemma3.Gemma3Attention(_text_config(), layer_idx = 0).to(torch.float32)
    q_seen = _record_input_dtypes(attention.q_proj)
    o_seen = _record_input_dtypes(attention.o_proj)
    hidden_states = torch.randn(1, 4, HIDDEN_SIZE, dtype = torch.float16)
    cos = torch.randn(1, 4, HEAD_DIM, dtype = torch.float32)
    sin = torch.randn(1, 4, HEAD_DIM, dtype = torch.float32)

    attn_output, _attn_weights = attention(
        hidden_states,
        position_embeddings = (cos, sin),
        attention_mask = None,
    )

    assert attn_output.dtype == torch.float32
    assert attn_output.shape == (1, 4, HIDDEN_SIZE)
    assert torch.isfinite(attn_output).all()
    assert q_seen == [torch.float32]
    assert o_seen == [torch.float32]


@requires_gemma3
def test_generic_attention_runs_float16_hidden_states_into_float32_projections(generic_patches):
    """The second (non forced-float32) Gemma3Attention forward needs the same fix."""
    torch.manual_seed(0)
    attention = gemma3.Gemma3Attention(_text_config(), layer_idx = 0).to(torch.float32)
    q_seen = _record_input_dtypes(attention.q_proj)
    hidden_states = torch.randn(1, 4, HIDDEN_SIZE, dtype = torch.float16)
    cos = torch.randn(1, 4, HEAD_DIM, dtype = torch.float32)
    sin = torch.randn(1, 4, HEAD_DIM, dtype = torch.float32)

    attn_output, _attn_weights = attention(
        hidden_states,
        position_embeddings = (cos, sin),
        attention_mask = None,
    )

    assert attn_output.dtype == torch.float32
    assert attn_output.shape == (1, 4, HIDDEN_SIZE)
    assert torch.isfinite(attn_output).all()
    assert q_seen == [torch.float32]


class _Projections:
    """Bare attribute holder: ``_linear_boundary_dtype`` only reads ``.weight``."""

    def __init__(self, **projections):
        for name, value in projections.items():
            setattr(self, name, value)


def test_linear_boundary_dtype_reads_the_first_floating_point_projection_weight():
    names = ("q_proj", "k_proj", "v_proj", "o_proj")

    float32_module = _Projections(**{name: torch.nn.Linear(4, 4, bias = False, dtype = torch.float32) for name in names})
    assert _linear_boundary_dtype(float32_module, *names) == torch.float32

    float16_module = _Projections(**{name: torch.nn.Linear(4, 4, bias = False, dtype = torch.float16) for name in names})
    assert _linear_boundary_dtype(float16_module, *names) == torch.float16

    bfloat16_module = _Projections(**{name: torch.nn.Linear(4, 4, bias = False, dtype = torch.bfloat16) for name in names})
    assert _linear_boundary_dtype(bfloat16_module, *names) == torch.bfloat16

    # bitsandbytes 4bit keeps the base weight as a packed uint8 blob, which is not
    # floating point: fall through to the float16 default this path always used.
    quantized_module = _Projections(**{name: _Projections(weight = torch.zeros(8, 1, dtype = torch.uint8)) for name in names})
    assert _linear_boundary_dtype(quantized_module, *names) == torch.float16

    # A uint8 blob must be skipped rather than end the search: the fp32 weight wins.
    mixed_module = _Projections(
        q_proj = _Projections(weight = torch.zeros(8, 1, dtype = torch.uint8)),
        k_proj = torch.nn.Linear(4, 4, bias = False, dtype = torch.float32),
    )
    assert _linear_boundary_dtype(mixed_module, *names) == torch.float32

    # No projections at all, and projections explicitly set to None.
    assert _linear_boundary_dtype(_Projections(), *names) == torch.float16
    assert _linear_boundary_dtype(_Projections(**{name: None for name in names}), *names) == torch.float16


def test_to_boundary_dtype_casts_to_the_weight_dtype_and_is_identity_when_it_matches():
    float32_activation = torch.randn(2, 3, dtype = torch.float32)
    float16_activation = torch.randn(2, 3, dtype = torch.float16)

    # Matching dtype: exact same tensor object back, no copy, no numerics change.
    assert _to_boundary_dtype(float32_activation, torch.float32) is float32_activation
    assert _to_boundary_dtype(float16_activation, torch.float16) is float16_activation

    # Mismatched dtype: moved onto the Linear's dtype.
    assert _to_boundary_dtype(float16_activation, torch.float32).dtype == torch.float32
    assert _to_boundary_dtype(float32_activation, torch.float16).dtype == torch.float16
    assert torch.equal(
        _to_boundary_dtype(float16_activation, torch.float32),
        float16_activation.to(torch.float32),
    )
