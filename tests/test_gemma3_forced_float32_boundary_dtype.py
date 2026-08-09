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
    _to_forced_output_dtype,
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
    # floating point: no weight answers, so leave the activation alone.
    quantized_module = _Projections(**{name: _Projections(weight = torch.zeros(8, 1, dtype = torch.uint8)) for name in names})
    assert _linear_boundary_dtype(quantized_module, *names) is None

    # A uint8 blob must be skipped rather than end the search: the fp32 weight wins.
    mixed_module = _Projections(
        q_proj = _Projections(weight = torch.zeros(8, 1, dtype = torch.uint8)),
        k_proj = torch.nn.Linear(4, 4, bias = False, dtype = torch.float32),
    )
    assert _linear_boundary_dtype(mixed_module, *names) == torch.float32

    # No projections at all, and projections explicitly set to None.
    assert _linear_boundary_dtype(_Projections(), *names) is None
    assert _linear_boundary_dtype(_Projections(**{name: None for name in names}), *names) is None


_FLOAT8_DTYPES = [
    getattr(torch, name) for name in
    ("float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz")
    if hasattr(torch, name)
]


@pytest.mark.parametrize("float8_dtype", _FLOAT8_DTYPES, ids = lambda d: str(d).split(".")[-1])
def test_a_float8_storage_weight_is_not_read_as_an_activation_dtype(float8_dtype):
    """`torch.float8_e4m3fn.is_floating_point` is True, so a plain floating-point
    test hands back float8 and the caller casts the hidden states straight to it.

    transformers' FP8Linear / FbgemmFp8Linear take bfloat16 or float16 in and do
    their own scaled quantization, so an unscaled cast loses values before the
    scaling meant to preserve them, and the Q/K norm and SDPA downstream get a
    dtype they do not support.
    """
    names = ("q_proj", "k_proj", "v_proj", "o_proj")
    fp8_module = _Projections(**{
        name: _Projections(weight = torch.zeros(8, 8).to(float8_dtype)) for name in names
    })
    assert float8_dtype.is_floating_point, "the decoy must look like a float to the old test"
    assert _linear_boundary_dtype(fp8_module, *names) is None

    # It is skipped, not treated as terminal: a real compute dtype further along wins.
    mixed = _Projections(
        q_proj = _Projections(weight = torch.zeros(8, 8).to(float8_dtype)),
        k_proj = torch.nn.Linear(4, 4, bias = False, dtype = torch.bfloat16),
    )
    assert _linear_boundary_dtype(mixed, *names) == torch.bfloat16


def test_a_bfloat16_activation_survives_a_4bit_model():
    """The generic (non-forced) path installs on 4bit QLoRA with bfloat16
    activations. Every projection is a uint8 blob there, so nothing answers, and
    a float16 fallback would narrow bfloat16 to float16 on a forward the
    unpatched model passed through untouched, costing exactly the exponent range
    bfloat16 is chosen for.
    """
    names = ("q_proj", "k_proj", "v_proj", "o_proj")
    quantized = _Projections(**{
        name: _Projections(weight = torch.zeros(8, 1, dtype = torch.uint8)) for name in names
    })
    hidden_states = torch.randn(2, 3, dtype = torch.bfloat16)
    boundary = _linear_boundary_dtype(quantized, *names)
    out = _to_boundary_dtype(hidden_states, boundary)
    assert out is hidden_states, "the activation was copied or narrowed"
    assert out.dtype == torch.bfloat16


def test_every_patch_installing_a_boundary_forward_publishes_the_helpers():
    """The auto-compiler serializes these forwards into unsloth_compiled_cache and
    resolves their free names by importing from the modeling module. A helper that
    lives only in the patch module is not importable there, so the generated
    forward dies with NameError on its first call.

    Read off the source rather than asserted by name, so adding a fourth patch
    that uses the helpers without publishing them fails here.
    """
    import ast
    import pathlib

    source = pathlib.Path(gemma_patches.__file__).read_text(encoding = "utf-8")
    tree = ast.parse(source)
    helpers = {"_linear_boundary_dtype", "_to_boundary_dtype", "_to_forced_output_dtype"}

    offenders = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("patch_"):
            continue
        called = {
            n.func.id for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        if not (helpers & called):
            continue
        if "_publish_boundary_helpers" not in called:
            offenders.append(node.name)

    assert not offenders, (
        f"{offenders} install a forward calling {sorted(helpers)} without calling "
        f"_publish_boundary_helpers, so the compiled copy raises NameError"
    )


def test_the_publish_check_actually_finds_the_patches():
    """Guard the guard: if the AST walk stopped matching, the test above would
    pass with an empty offender list and prove nothing."""
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path(gemma_patches.__file__).read_text(encoding = "utf-8"))
    helpers = {"_linear_boundary_dtype", "_to_boundary_dtype", "_to_forced_output_dtype"}
    using = [
        node.name for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("patch_")
        and (helpers & {
            n.func.id for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        })
    ]
    assert len(using) >= 3, f"expected the MLP and both attention patches, found {using}"


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


class _FakeLinear4bit(torch.nn.Module):
    """A stand-in for bitsandbytes `Linear4bit`, in the two ways that matter here.

    bitsandbytes stores the base weight as a packed `uint8` blob
    (`Params4bit(quant_storage=torch.uint8)`), so `_linear_boundary_dtype` finds
    no floating-point weight and answers None. And its forward ends in
    `bnb.matmul_4bit(...).to(inp_dtype)`, so the result carries the caller's
    input dtype straight back out. Together those two facts are what turn a
    float32 activation at a forced output boundary into a float32 `down_proj` /
    `o_proj` result, and thus into changed QLoRA forward and gradient dtypes.
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.register_buffer("weight", torch.zeros(out_features * in_features // 2, 1, dtype = torch.uint8))
        self.register_buffer("_dequantized", torch.randn(out_features, in_features, dtype = torch.float16))

    def forward(self, x):
        inp_dtype = x.dtype
        return (x.to(torch.float16) @ self._dequantized.T).to(inp_dtype)


def _quantize_projections(module, names):
    for name in names:
        linear = getattr(module, name)
        setattr(module, name, _FakeLinear4bit(linear.in_features, linear.out_features))
pass


@requires_gemma3
def test_forced_float32_mlp_downcasts_to_float16_when_4bit_weights_cannot_answer(forced_float32_patches):
    """4bit QLoRA: no projection weight is floating point, so the boundary dtype
    is None. The forced output boundary must still fall back to the float16 the
    bare `.to(torch.float16)` always produced, or `down_proj` receives float32
    and hands float32 back.
    """
    torch.manual_seed(0)
    mlp = gemma3.Gemma3MLP(_text_config()).to(torch.float16)
    _quantize_projections(mlp, ("gate_proj", "up_proj", "down_proj"))
    assert _linear_boundary_dtype(mlp, "gate_proj", "up_proj", "down_proj") is None

    gate_seen = _record_input_dtypes(mlp.gate_proj)
    down_seen = _record_input_dtypes(mlp.down_proj)
    x = torch.randn(2, 3, HIDDEN_SIZE, dtype = torch.float16)

    out = mlp(x)

    assert gate_seen == [torch.float16], "the fp16 input boundary must stay fp16"
    assert down_seen == [torch.float16], "the forced output boundary leaked the fp32 reduction"
    assert out.dtype == torch.float16, "Linear4bit returns the caller's dtype, so down_proj went fp32"


@requires_gemma3
def test_forced_float32_attention_downcasts_to_float16_when_4bit_weights_cannot_answer(forced_float32_patches):
    """Same forced output boundary, on `o_proj` at the attention exit."""
    torch.manual_seed(0)
    attention = gemma3.Gemma3Attention(_text_config(), layer_idx = 0).to(torch.float16)
    _quantize_projections(attention, ("q_proj", "k_proj", "v_proj", "o_proj"))
    assert _linear_boundary_dtype(attention, "q_proj", "k_proj", "v_proj", "o_proj") is None

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

    assert q_seen == [torch.float16], "the fp16 input boundary must stay fp16"
    assert o_seen == [torch.float16], "the forced output boundary leaked the fp32 reduction"
    assert attn_output.dtype == torch.float16, "Linear4bit returns the caller's dtype, so o_proj went fp32"


def test_to_forced_output_dtype_defaults_to_float16_and_is_not_identity_on_none():
    """The whole distinction between the two helpers, asserted directly."""
    float32_reduction = torch.randn(2, 3, dtype = torch.float32)

    # No weight answered: the forced output boundary is the old bare fp16 cast...
    assert _to_forced_output_dtype(float32_reduction, None).dtype == torch.float16
    assert torch.equal(
        _to_forced_output_dtype(float32_reduction, None),
        float32_reduction.to(torch.float16),
    )
    # ...while the generic input helper is deliberately identity on None.
    assert _to_boundary_dtype(float32_reduction, None) is float32_reduction

    # A weight that did answer still wins over the fp16 default.
    assert _to_forced_output_dtype(float32_reduction, torch.float32) is float32_reduction
    assert _to_forced_output_dtype(float32_reduction, torch.bfloat16).dtype == torch.bfloat16

    # Already fp16 with no answer: same tensor back, no pointless copy.
    float16_reduction = torch.randn(2, 3, dtype = torch.float16)
    assert _to_forced_output_dtype(float16_reduction, None) is float16_reduction


class _QuantizedWeight(torch.Tensor):
    """A tensor that carries a `quant_state`, the way `Params4bit` does.

    bitsandbytes only fills `quant_state` in once the parameter has been moved
    to an accelerator, so a CPU-only test cannot build a real packed weight.
    What the helper actually reads is just `.dtype` and `.quant_state`, so a
    tensor subclass carrying both reproduces the case exactly and keeps this
    test running everywhere. The GPU test below asserts the same thing against
    a real `Params4bit`.
    """

    quant_state = None

    @staticmethod
    def __new__(cls, data, quant_state):
        instance = torch.Tensor._make_subclass(cls, data, False)
        instance.quant_state = quant_state
        return instance


_QUANT_STORAGE_DTYPES = [torch.uint8, torch.bfloat16, torch.float16, torch.float32]


@pytest.mark.parametrize("storage_dtype", _QUANT_STORAGE_DTYPES,
                         ids = lambda d: str(d).split(".")[-1])
def test_a_quantized_weight_never_answers_whatever_its_storage_dtype_is(storage_dtype):
    """`bnb_4bit_quant_storage` is a public knob, and it is not the compute dtype.

    It defaults to uint8, which is not floating point and so was already
    skipped. But FSDP can only shard float dtypes, so FSDP-QLoRA users are
    told to set it to bfloat16, and `vllm_utils` and the bnb MoE loaders plumb
    the configured value through. Then `Params4bit.weight.dtype` is bfloat16
    while the tensor is still packed 4bit, and a plain floating-point test
    reads a storage container as the activation dtype. `Linear4bit` returns
    the caller's input dtype, so answering here would change QLoRA forward and
    gradient numerics on a weight that is still quantized.
    """
    names = ("q_proj", "k_proj", "v_proj", "o_proj")
    quantized = _Projections(**{
        name: _Projections(weight = _QuantizedWeight(
            torch.zeros(8, 1, dtype = storage_dtype), quant_state = object(),
        )) for name in names
    })
    assert _linear_boundary_dtype(quantized, *names) is None

    # Skipped, not terminal: a real unquantized compute dtype further along wins.
    mixed = _Projections(
        q_proj = _Projections(weight = _QuantizedWeight(
            torch.zeros(8, 1, dtype = storage_dtype), quant_state = object(),
        )),
        k_proj = torch.nn.Linear(4, 4, bias = False, dtype = torch.float32),
    )
    assert _linear_boundary_dtype(mixed, *names) == torch.float32


@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float16, torch.float32],
                         ids = lambda d: str(d).split(".")[-1])
def test_an_unquantized_weight_still_answers_and_the_skip_is_not_blanket(weight_dtype):
    """The discriminator, asserted from the other side.

    Skipping quantized weights must key off `quant_state`, not off the dtype.
    A bfloat16 weight with no `quant_state` is an ordinary LoRA / QLoRA base
    weight or a bf16 full finetune, and it must still answer, or a float32 full
    finetune goes back to the "expected mat1 and mat2 to have the same dtype"
    crash this helper exists to fix.
    """
    names = ("q_proj", "k_proj", "v_proj", "o_proj")
    module = _Projections(**{
        name: torch.nn.Linear(4, 4, bias = False, dtype = weight_dtype) for name in names
    })
    assert getattr(module.q_proj.weight, "quant_state", None) is None
    assert _linear_boundary_dtype(module, *names) == weight_dtype

    # A plain nn.Parameter has no `quant_state` attribute at all: getattr, not
    # hasattr-then-read, so this stays a no-op on any non-bitsandbytes backend.
    bare = _Projections(**{name: torch.nn.Parameter(torch.zeros(4, 4, dtype = weight_dtype))
                           for name in names})
    assert _linear_boundary_dtype(bare, *names) == weight_dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs an accelerator to quantize")
@pytest.mark.parametrize("storage_dtype", _QUANT_STORAGE_DTYPES,
                         ids = lambda d: str(d).split(".")[-1])
def test_a_real_bitsandbytes_params4bit_never_answers(storage_dtype):
    """The same assertion against real bitsandbytes, not a stand-in.

    bitsandbytes quantizes on the move to the accelerator, so this is where a
    genuine packed `Params4bit` exists and where its dtype really does report
    the storage container.
    """
    bnb = pytest.importorskip("bitsandbytes")
    names = ("q_proj", "k_proj", "v_proj", "o_proj")

    weight = bnb.nn.Params4bit(
        torch.randn(64, 64, dtype = torch.float16),
        quant_type = "nf4", quant_storage = storage_dtype, requires_grad = False,
    ).cuda()
    assert weight.quant_state is not None, "bitsandbytes did not quantize"
    assert weight.dtype == storage_dtype, "quant_storage is meant to set the storage dtype"

    module = _Projections(**{name: _Projections(weight = weight) for name in names})
    assert _linear_boundary_dtype(module, *names) is None
