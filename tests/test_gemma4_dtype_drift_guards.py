"""CPU/CI regression guards for the Gemma 4 dtype fixes (PR #925).

These lock in the three failures we traced:

  1. Audio merge: upstream `Gemma4Model.forward` cast `audio_features.to(device)`
     without dtype (transformers #45192), while image/video cast dtype too ->
     `masked_scatter_: expected self and source to have same dtypes`.
  2. Forced-float32 PLE: fp32 residual into the fp16 PLE Linears (unsloth-zoo #866
     forced fp32 but left PLE inputs uncast) -> `mat1 and mat2 ... float != Half`.
  3. Cross-path NameError: the eager patch rewrote calls to one helper name while
     the compiler emitted another; a compile after the eager patch called an
     undefined helper.

Everything here is CPU-only (dtype-check errors raise on CPU), no GPU required.
The real-source canaries import the real transformers gemma4 module and skip
cleanly when it is absent; when present (transformers >= 5.5.0, as on CI) they
catch upstream source drift that would silently disable the patch.
"""
import inspect
import re

import pytest
import torch

from unsloth_zoo.temporary_patches import gemma4_float32 as g4f
from unsloth_zoo.temporary_patches.gemma4_float32 import _unsloth_gemma4_ple_cast_input
from unsloth_zoo import compiler as C
from unsloth_zoo.compiler import _GEMMA4_PLE_CAST_HELPER, fix_gemma4_forced_float32_ple_dtype

CANONICAL_HELPER = "_unsloth_gemma4_ple_cast_input"

try:
    from transformers.models.gemma4 import modeling_gemma4 as real_gemma4
    HAS_GEMMA4 = True
except Exception:
    real_gemma4 = None
    HAS_GEMMA4 = False

requires_gemma4 = pytest.mark.skipif(not HAS_GEMMA4, reason="transformers gemma4 not installed")


# ---------------------------------------------------------------------------
# Guard 3: the eager and compiler PLE helpers must share ONE name (string/AST).
# This is what broke as the cross-path NameError. Any reintroduction of a second
# spelling fails here instantly, with no GPU / model needed.
# ---------------------------------------------------------------------------
def _ple_cast_identifiers(source):
    return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*_ple_cast_input", source))


def test_ple_cast_helper_has_exactly_one_name_across_eager_and_compiler():
    eager_src = inspect.getsource(g4f)
    compiler_src = inspect.getsource(C)
    names = _ple_cast_identifiers(eager_src) | _ple_cast_identifiers(compiler_src)
    assert names == {CANONICAL_HELPER}, (
        f"PLE cast helper name diverged across eager/compiler paths: {sorted(names)}. "
        f"Both must use exactly {CANONICAL_HELPER!r} or a compile after the eager "
        f"patch will call an undefined helper (NameError)."
    )


def test_compiler_generated_helper_defines_the_name_it_calls():
    # The name the compiler REWRITE inserts must equal the name the appended
    # helper DEFINES. Extract both from the compiler source directly.
    on_flag = _run_ple_rewrite("Gemma4TextDecoderLayer")
    called = set(re.findall(r"([A-Za-z_][A-Za-z0-9_]*_ple_cast_input)\(", on_flag))
    defined = set(re.findall(r"def ([A-Za-z_][A-Za-z0-9_]*_ple_cast_input)\b", _GEMMA4_PLE_CAST_HELPER))
    assert called and defined and called == defined == {CANONICAL_HELPER}, (
        f"compiler rewrite calls {called} but helper defines {defined}"
    )


def _run_ple_rewrite(module, monkey_env="1"):
    import os
    prev = os.environ.get("UNSLOTH_FORCE_FLOAT32")
    os.environ["UNSLOTH_FORCE_FLOAT32"] = monkey_env
    try:
        src = (
            "def forward(self, hidden_states):\n"
            "    hidden_states = self.per_layer_input_gate(hidden_states)\n"
            "    return self.per_layer_projection(hidden_states)\n"
        )
        return fix_gemma4_forced_float32_ple_dtype(src, module)
    finally:
        if prev is None:
            os.environ.pop("UNSLOTH_FORCE_FLOAT32", None)
        else:
            os.environ["UNSLOTH_FORCE_FLOAT32"] = prev


# ---------------------------------------------------------------------------
# Guard: eager and compiler helper implementations must stay behaviourally
# identical (both copies are hand-maintained; catch silent drift between them).
# ---------------------------------------------------------------------------
def _compiler_helper_callable():
    ns = {}
    exec(_GEMMA4_PLE_CAST_HELPER, ns)
    return ns[CANONICAL_HELPER]


class _ModWithWeight:
    def __init__(self, weight): self.weight = weight
class _WeightStub:
    pass


def _dtype_cases():
    cases = []
    for name, dt in (("f16", torch.float16), ("bf16", torch.bfloat16), ("f32", torch.float32)):
        w = torch.nn.Linear(3, 3, bias=False, dtype=dt)
        cases.append((f"dense_{name}", w))
    q = torch.zeros(3, 3, dtype=torch.uint8); q.quant_state = object()
    cases.append(("bnb4bit", _ModWithWeight(q)))
    cases.append(("int8", _ModWithWeight(torch.zeros(3, 3, dtype=torch.int8))))
    fp8 = getattr(torch, "float8_e4m3fn", None)
    if fp8 is not None:
        s = _WeightStub(); s.dtype = fp8
        cases.append(("fp8", _ModWithWeight(s)))
    cases.append(("no_dtype", _ModWithWeight(_WeightStub())))
    return cases


@pytest.mark.parametrize("label,module", _dtype_cases())
def test_eager_and_compiler_helpers_agree(label, module):
    eager = _unsloth_gemma4_ple_cast_input
    comp = _compiler_helper_callable()
    x = torch.randn(2, 3, dtype=torch.float32)
    e, c = eager(module, x), comp(module, x)
    assert (e is x) == (c is x), f"{label}: identity divergence"
    assert e.dtype == c.dtype, f"{label}: dtype divergence {e.dtype} vs {c.dtype}"


# ---------------------------------------------------------------------------
# Guard 1 & 2 behaviour (pure CPU torch): prove the fix actually resolves the
# two dtype crashes. No gemma4 needed - exercises the exact failing op.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dst_dt", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("src_dt", [torch.float16, torch.bfloat16, torch.float32])
def test_audio_dtype_cast_fixes_masked_scatter(dst_dt, src_dt):
    dst = torch.zeros(1, 4, dtype=dst_dt)
    mask = torch.ones(1, 4, dtype=torch.bool)
    src = torch.arange(4, dtype=src_dt).reshape(1, 4)
    if dst_dt != src_dt:
        # BEFORE the fix (device-only cast) the merge raises.
        with pytest.raises(RuntimeError, match="same dtype"):
            dst.masked_scatter(mask, src.to(dst.device))
    # AFTER the fix (device + dtype) it works and preserves values.
    out = dst.masked_scatter(mask, src.to(dst.device, dst.dtype))
    assert out.dtype == dst_dt
    torch.testing.assert_close(out, src.to(dst_dt))


@pytest.mark.parametrize("w_dt", [torch.float16, torch.bfloat16])
def test_ple_helper_fixes_fp32_into_lowprec_linear(w_dt):
    lin = torch.nn.Linear(3, 3, bias=False, dtype=w_dt)
    x = torch.randn(2, 3, dtype=torch.float32, requires_grad=True)
    # BEFORE: fp32 activation into low-precision weight raises.
    with pytest.raises(RuntimeError, match="same dtype"):
        lin(x)
    # AFTER: helper casts the input to the weight dtype; forward + backward work.
    out = lin(_unsloth_gemma4_ple_cast_input(lin, x))
    assert out.dtype == w_dt
    out.float().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_ple_helper_never_casts_to_fp8_or_packed_weight():
    x = torch.randn(2, 3, dtype=torch.float32)
    # fp8 packed storage must be left alone (fp8 kernels scale internally).
    fp8 = getattr(torch, "float8_e4m3fn", None)
    if fp8 is not None:
        s = _WeightStub(); s.dtype = fp8
        assert _unsloth_gemma4_ple_cast_input(_ModWithWeight(s), x) is x
    # bnb 4-bit: uint8 + quant_state -> unchanged (never cast activation to uint8).
    q = torch.zeros(3, 3, dtype=torch.uint8); q.quant_state = object()
    assert _unsloth_gemma4_ple_cast_input(_ModWithWeight(q), x) is x


# ---------------------------------------------------------------------------
# Real-source drift canaries: the highest-value "never again" guards. They read
# the REAL transformers gemma4 module FILE FROM DISK (no mutation, no GPU) and
# assert the exact call sites the unsloth patches target are still present.
#
# Reading from disk is deliberate: importing unsloth_zoo monkeypatches
# Gemma4Model.forward (the KV-carrier wrapper) and can poison linecache for the
# PLE methods, so inspect.getsource() on the class attributes returns the
# unsloth-wrapped source, not upstream. The on-disk file is pristine upstream.
#
# If a future transformers reshapes a call site so the patch can no longer match
# AND upstream has not fixed the dtype itself, these FAIL loudly instead of the
# patch silently no-op-ing and the dtype crash quietly returning.
# ---------------------------------------------------------------------------
def _gemma4_modeling_source():
    import pathlib
    return pathlib.Path(inspect.getsourcefile(real_gemma4)).read_text()


@requires_gemma4
def test_real_gemma4_audio_merge_site_is_recognized():
    src = _gemma4_modeling_source()
    buggy = "audio_features.to(inputs_embeds.device)"
    fixed = "audio_features.to(inputs_embeds.device, inputs_embeds.dtype)"
    assert (buggy in src) or (fixed in src), (
        "Gemma4 audio merge site drifted: neither the known device-only pattern "
        "nor the fixed device+dtype pattern is present in modeling_gemma4.py. The "
        "unsloth audio patch would silently no-op and the masked_scatter dtype crash "
        "would return. Update _patch_gemma4_audio_feature_dtype_on_class (eager) and "
        "fix_gemma4_audio_feature_dtype (compiler)."
    )


@requires_gemma4
def test_real_gemma4_image_and_video_merges_still_cast_dtype():
    # Regression anchor: image/video have always cast dtype; if upstream ever
    # drops it there too, that is a new modality that also needs patching.
    src = _gemma4_modeling_source()
    assert "image_features.to(inputs_embeds.device, inputs_embeds.dtype)" in src
    assert "video_features.to(inputs_embeds.device, inputs_embeds.dtype)" in src


@requires_gemma4
def test_real_gemma4_ple_call_sites_are_recognized():
    src = _gemma4_modeling_source()

    def _ok(attr, arg):
        raw = f"self.{attr}({arg})"
        wrapped = f"self.{attr}({CANONICAL_HELPER}(self.{attr}, {arg})"
        return (raw in src) or (wrapped in src)

    assert _ok("per_layer_model_projection", "inputs_embeds"), (
        "Gemma4TextModel.project_per_layer_inputs PLE call drifted; PLE dtype patch "
        "would silently no-op under UNSLOTH_FORCE_FLOAT32."
    )
    assert _ok("per_layer_input_gate", "hidden_states"), "per_layer_input_gate call drifted"
    assert _ok("per_layer_projection", "hidden_states"), "per_layer_projection call drifted"


@requires_gemma4
def test_real_gemma4_audio_compiler_transform_emits_dtype_cast():
    # Drive the compiler's audio rewrite against the REAL upstream source; either
    # upstream already casts dtype, or our regex still matches and inserts it.
    src = _gemma4_modeling_source()
    out = C.fix_gemma4_audio_feature_dtype(src)
    assert "audio_features.to(inputs_embeds.device, inputs_embeds.dtype)" in out, (
        "fix_gemma4_audio_feature_dtype no longer produces the dtype-aligned cast on "
        "upstream source (regex drift or multiple matches)."
    )
