# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Regression pins for past zoo bugs. Each test maps to a SHIPPED fix on
`main` and catches the bug class if it re-appears (not the fix path itself);
each cites the commit/PR that introduced the regression."""

from __future__ import annotations

import importlib

import pytest


# ---------------------------------------------------------------------------
# Regression: a missing comma in temporary_patches/utils.py `__all__` silently
# concatenates adjacent string literals ("raise_errorUnpack"), making the
# public names un-importable under `from ... import *`. No syntax error fires.
# Source: 2e36f32 fix(temporary_patches/utils): add missing comma in __all__
# between 'raise_error' and 'Unpack' (#617)
# ---------------------------------------------------------------------------


def test_temporary_patches_utils_all_entries_are_valid_attributes():
    """Every name in `__all__` must be a real attribute on the module."""
    from unsloth_zoo.temporary_patches import utils

    missing = [
        name for name in utils.__all__
        if not hasattr(utils, name)
    ]
    assert not missing, (
        f"temporary_patches.utils.__all__ lists names that are not "
        f"actual module attributes: {missing}. Most likely cause: a "
        "missing comma between two entries causing Python to "
        "silently concatenate the two strings (the regression "
        "fixed in #617)."
    )


def test_temporary_patches_utils_no_concatenated_all_entries():
    """No `__all__` entry should look like two jammed names (e.g.
    `raise_errorUnpack`). Heuristic: a name with BOTH a snake_case underscore
    AND a lowercase->uppercase transition. Pure ALL_CAPS like `KWARGS_TYPE`
    has no such transition, so it doesn't match.
    """
    from unsloth_zoo.temporary_patches import utils
    import re

    # lowercase->uppercase = a snake_case + CamelCase concatenation.
    camel_boundary = re.compile(r"[a-z][A-Z]")
    suspicious = []
    for name in utils.__all__:
        if name.startswith("_"):
            continue
        if "_" not in name:
            continue  # pure CamelCase or lowercase -- not the bug class.
        if camel_boundary.search(name):
            suspicious.append(name)
    assert not suspicious, (
        "Suspicious __all__ entries (likely two concatenated names): "
        f"{suspicious}. This is the bug class fixed in zoo PR #617 "
        "(missing comma between adjacent string literals in __all__)."
    )


def test_temporary_patches_utils_known_public_names_present():
    """Pin specific public names that downstream patches import."""
    from unsloth_zoo.temporary_patches import utils

    expected = ["raise_error", "Unpack", "patch_function"]
    for name in expected:
        assert name in utils.__all__, (
            f"Public name {name!r} missing from temporary_patches.utils.__all__"
        )
        assert hasattr(utils, name), (
            f"Public name {name!r} listed in __all__ but not on module"
        )


# ---------------------------------------------------------------------------
# Regression: `compiler.higher_precision_softmax` wasn't idempotent -- running
# it twice appended a duplicate `.to(x.dtype).to(x.dtype)` (the
# already-rewritten lookahead was missing). The compiler runs on user source
# mid-training and may re-run (e.g. checkpoint reload), so emitted source must
# not drift.
# Source: f98dbbc fix(compiler): make higher_precision_softmax idempotent (#631)
# ---------------------------------------------------------------------------


def test_higher_precision_softmax_idempotent():
    """`higher_precision_softmax(higher_precision_softmax(src))` must
    equal `higher_precision_softmax(src)`.
    """
    from unsloth_zoo.compiler import higher_precision_softmax

    # Sample source pulled from the docstring of the function itself.
    src = (
        "attn_weights = nn.functional.softmax(attn_weights, dim=-1)\n"
        "probs = F.softmax(combined_logits, dim=-1)\n"
        "routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)\n"
    )

    once  = higher_precision_softmax(src)
    twice = higher_precision_softmax(once)
    assert once == twice, (
        "higher_precision_softmax is NOT idempotent -- second pass "
        "changed the source. Likely the lookahead `(?!\\s*\\.to(...)`"
        " was lost.\n--- once ---\n"
        f"{once}\n--- twice ---\n{twice}"
    )


def test_higher_precision_softmax_does_not_double_cast():
    """An already-rewritten call must not gain a second `.to(x.dtype)`."""
    from unsloth_zoo.compiler import higher_precision_softmax

    already_rewritten = (
        "attn_weights = nn.functional.softmax(attn_weights, dim=-1, "
        "dtype = torch.float32).to(attn_weights.dtype)\n"
    )
    out = higher_precision_softmax(already_rewritten)
    # No double `.to(...)` chain produced.
    assert ".dtype).to(" not in out.replace(
        # Tolerate ONE `.to(<var>.dtype)` per call -- bug emits TWO.
        ").to(attn_weights.dtype)", "", 1,
    ), (
        f"higher_precision_softmax appended a duplicate .to(..) cast:\n{out}"
    )


# ---------------------------------------------------------------------------
# Regression: backend device helpers must guard against partial torch builds
# (e.g. `torch.xpu` exists but `torch.xpu.synchronize` raises). Calling
# device_synchronize / device_empty_cache must not raise on a partial backend.
# Source: e08c1df (Guard XPU synchronize), 35dc451 (Guard XPU empty_cache).
# ---------------------------------------------------------------------------


def test_device_synchronize_tolerates_partial_backend():
    """`device_synchronize()` must not raise on a minimal stub backend."""
    from unsloth_zoo.device_type import device_synchronize

    # On the GPU-free harness this resolves to a stub; the partial-backend
    # guards live inside the implementation.
    device_synchronize()


def test_device_type_module_has_expected_helpers():
    """Pin the public API downstream zoo/unsloth/Studio code calls; a rename
    or removal here breaks consumers silently (AttributeError at train time)."""
    import unsloth_zoo.device_type as dt

    expected = [
        "DEVICE_TYPE",
        "device_synchronize",
    ]
    missing = [name for name in expected if not hasattr(dt, name)]
    assert not missing, (
        f"unsloth_zoo.device_type missing expected helpers: {missing}"
    )


# ---------------------------------------------------------------------------
# Regression: RL_REPLACEMENTS dict integrity vs the GRPO refactor wave
# (commits 466334c, 9829ade, 035f...). The dict is rebuilt per definition; a
# missing `RL_REPLACEMENTS[name] = fn` after a refactor fails silently. Pins
# the public-API keys (test_rl_replacements_cpu.py covers the IO contract).
# ---------------------------------------------------------------------------


def test_rl_replacements_registration_survived_grpo_refactor():
    from unsloth_zoo import rl_replacements as rr

    expected_min = {
        "calculate_pad_tokens_in_prompt",
        "create_completion_attention_mask",
        "left_pack_padding",
        "sanitize_logprob",
    }
    missing = expected_min - set(rr.RL_REPLACEMENTS.keys())
    assert not missing, (
        f"RL_REPLACEMENTS dict lost public-API key(s) after a refactor: "
        f"{sorted(missing)}. Recheck the `RL_REPLACEMENTS[name] = fn` "
        f"lines below each definition in rl_replacements.py."
    )


# ---------------------------------------------------------------------------
# Regression: gpt-oss inference leaks a flex BlockMask into the eager attention
# forward. When config._attn_implementation == "flex_attention" the inference
# path receives a BlockMask and inplace_eager_attention_forward crashes with
#   TypeError: unsupported operand type(s) for +=: 'Tensor' and 'BlockMask'
# Two AST guards (one over wrap.return_attention_mask, one over
# patch_GptOssModel) catch silent deletion of the flex_attention literal.
# Source: PR #690; latent since commit ef819214 ("Fix Flex").
# ---------------------------------------------------------------------------


def test_gpt_oss_wrap_has_flex_attention_inference_guard():
    import ast
    import inspect
    from unsloth_zoo.temporary_patches import gpt_oss as _M

    src = inspect.getsource(_M.patch_GptOssModel)
    tree = ast.parse(src)

    wrap_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "wrap":
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.FunctionDef)
                    and inner.name == "return_attention_mask"
                ):
                    wrap_fn = inner
                    break
        if wrap_fn is not None:
            break

    assert wrap_fn is not None, "wrap.return_attention_mask not found"

    body_src = ast.unparse(wrap_fn)
    assert (
        '"flex_attention"' in body_src or "'flex_attention'" in body_src
    ) and "_attn_implementation" in body_src, (
        "wrap(f) for create_causal_mask must read config._attn_implementation "
        "and short-circuit the flex_attention case on the inference branch. "
        "Without this the eager forward receives a BlockMask and crashes "
        "with TypeError: unsupported operand type(s) for +=: 'Tensor' "
        "and 'BlockMask'. See PR #690."
    )


def test_gpt_oss_patched_model_forward_has_flex_attention_guard():
    """The patched GptOssModel.forward body must carry the
    _attn_implementation == 'flex_attention' guard, independent of wrap's own
    guard (so removing the forward-level swap can't satisfy this test)."""
    import ast
    import inspect
    from unsloth_zoo.temporary_patches import gpt_oss as _M

    src = inspect.getsource(_M.patch_GptOssModel)
    tree = ast.parse(src)

    forward_fn = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "forward"),
        None,
    )
    assert forward_fn is not None, "patched GptOssModel.forward not found"
    body = ast.unparse(forward_fn)

    assert (
        ('"flex_attention"' in body or "'flex_attention'" in body)
        and "_attn_implementation" in body
    ), (
        "patched GptOssModel.forward must guard against "
        "_attn_implementation == 'flex_attention' independently of "
        "wrap's own guard. Without this, the forward's "
        "causal_mask_mapping construction leaks a BlockMask to the "
        "eager attention path. See PR #690."
    )


# ---------------------------------------------------------------------------
# Regression: eager attention shape mismatch when the KV cache is longer than
# the attention mask's kv dim. transformers 5.x cache pre-allocation hands a
# full-attention layer more positions than the causal mask covers (e.g. k=161
# vs a 128-wide mask), so `attn_weights += attention_mask[...]` raises
# RuntimeError on dim 3. The _align_kv_to_mask helper trims to the shorter
# length; static check: it exists AND runs in BOTH eager forwards.
# Source: PR #691.
# ---------------------------------------------------------------------------


def test_gpt_oss_eager_attention_aligns_kv_to_mask():
    """Both eager forwards must invoke the KV alignment helper. Per-function
    source-slice check (not a global callsite count) so consolidation into a
    shared closure still passes if alignment runs on both routes.
    """
    import ast
    import inspect
    from unsloth_zoo.temporary_patches import gpt_oss as _M

    src = inspect.getsource(_M.patch_GptOssAttention)

    assert "_align_kv_to_mask" in src, (
        "patch_GptOssAttention must define _align_kv_to_mask (PR #691) "
        "so eager attention can survive KV-vs-mask length mismatches "
        "from pre-allocated caches on transformers 5.x + torch < 2.11."
    )

    tree = ast.parse(src)
    forwards = {
        node.name: ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name in ("eager_attention_forward", "inplace_eager_attention_forward")
    }
    expected = {"eager_attention_forward", "inplace_eager_attention_forward"}
    missing_defs = expected - set(forwards)
    assert not missing_defs, (
        f"patch_GptOssAttention must define {sorted(expected)}; "
        f"missing: {sorted(missing_defs)}."
    )

    missing_calls = [
        name for name, body in forwards.items()
        if "_align_kv_to_mask(" not in body
    ]
    assert not missing_calls, (
        f"Eager forward(s) missing _align_kv_to_mask call: {missing_calls}. "
        "Without the alignment these paths crash with 'tensor a (N) vs "
        "tensor b (M)' on transformers 5.x + torch < 2.11."
    )


# ---------------------------------------------------------------------------
# Regression: patched GptOssModel.forward hard-coded the 4.x mask kwargs
# ("input_embeds", "cache_position") and blew up on 5.x (renamed to
# "inputs_embeds", dropped "cache_position"). PR #690 added inspect-driven
# kwarg filtering to support both; static check that the helper is present.
# ---------------------------------------------------------------------------


def test_gpt_oss_model_forward_uses_inspect_filtered_mask_kwargs():
    import inspect
    from unsloth_zoo.temporary_patches import gpt_oss as _M

    src = inspect.getsource(_M.patch_GptOssModel)
    # The inspect-driven helper, or a guard testing inputs_embeds availability.
    has_inspect_filter = (
        "_build_mask_kwargs" in src
        or "_ccm_params" in src
        or "inspect.signature(create_causal_mask).parameters" in src
    )
    assert has_inspect_filter, (
        "patch_GptOssModel must filter mask kwargs by the actual factory "
        "signature so the patch works on both transformers 4.57.6 "
        "(input_embeds + cache_position) and transformers 5.x "
        "(inputs_embeds, no cache_position). See PR #690."
    )


# ---------------------------------------------------------------------------
# Regression: patch_compiling_bitsandbytes used exec(f"import {x}",
# globals(), locals()) + eval(x) inside a function. Under PEP 667
# (Python 3.13+) the locals() snapshot does not persist, so eval raised
# NameError: name 'peft' is not defined on every model load with
# bitsandbytes < 0.46.0. Fixed by switching to importlib.import_module +
# getattr. Source: #710 (issue #311).
# These tests stub bitsandbytes/peft in sys.modules so they are hermetic:
# no GPU, no real bitsandbytes install needed.
# ---------------------------------------------------------------------------


def _install_fake_bnb_and_peft(monkeypatch, bnb_version, with_peft_leaf=True):
    import sys
    import types

    # The function under test sets UNSLOTH_PATCHED=1; setenv registers the key
    # with monkeypatch so the pre-test state (even absence) is restored.
    import os
    monkeypatch.setenv("UNSLOTH_PATCHED", os.environ.get("UNSLOTH_PATCHED", "0"))

    def _fake_layer_class():
        class _Layer:
            def forward(self):
                return "ok"
        return _Layer

    bnb = types.ModuleType("bitsandbytes")
    bnb.__version__ = bnb_version
    bnb_nn = types.ModuleType("bitsandbytes.nn")
    bnb_modules = types.ModuleType("bitsandbytes.nn.modules")
    bnb_modules.Linear4bit = _fake_layer_class()
    bnb.nn = bnb_nn
    bnb_nn.modules = bnb_modules

    for name, mod in [
        ("bitsandbytes", bnb),
        ("bitsandbytes.nn", bnb_nn),
        ("bitsandbytes.nn.modules", bnb_modules),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)

    peft_leaf = None
    if with_peft_leaf:
        peft = types.ModuleType("peft")
        peft_tuners = types.ModuleType("peft.tuners")
        peft_lora = types.ModuleType("peft.tuners.lora")
        peft_leaf = types.ModuleType("peft.tuners.lora.bnb")
        peft_leaf.Linear4bit = _fake_layer_class()
        for name, mod in [
            ("peft", peft),
            ("peft.tuners", peft_tuners),
            ("peft.tuners.lora", peft_lora),
            ("peft.tuners.lora.bnb", peft_leaf),
        ]:
            monkeypatch.setitem(sys.modules, name, mod)
    else:
        # A peft package whose submodules cannot be imported: no __path__.
        peft = types.ModuleType("peft")
        monkeypatch.setitem(sys.modules, "peft", peft)
        for name in ["peft.tuners", "peft.tuners.lora", "peft.tuners.lora.bnb"]:
            monkeypatch.delitem(sys.modules, name, raising=False)

    return bnb_modules, peft_leaf


def test_patch_compiling_bitsandbytes_old_bnb_wraps_without_nameerror(monkeypatch):
    """The #311 bug class: with bitsandbytes < 0.46.0 the patch must run to
    completion (no NameError from exec/eval namespace loss on Python 3.13+)
    and dynamo-disable every forward on both target modules."""
    from unsloth_zoo.patching_utils import patch_compiling_bitsandbytes

    bnb_modules, peft_leaf = _install_fake_bnb_and_peft(monkeypatch, "0.45.5")
    patch_compiling_bitsandbytes()

    assert hasattr(bnb_modules.Linear4bit.forward, "__wrapped__"), (
        "bitsandbytes.nn.modules forwards were not dynamo-disabled on the "
        "< 0.46.0 branch. If this raises NameError instead, the exec/eval "
        "import pattern regressed (PEP 667, fixed in #710)."
    )
    assert hasattr(peft_leaf.Linear4bit.forward, "__wrapped__"), (
        "peft.tuners.lora.bnb forwards were not dynamo-disabled on the "
        "< 0.46.0 branch (the exact failure mode of issue #311)."
    )


def test_patch_compiling_bitsandbytes_idempotent(monkeypatch):
    """A second call must not re-wrap already wrapped forwards."""
    from unsloth_zoo.patching_utils import patch_compiling_bitsandbytes

    bnb_modules, _ = _install_fake_bnb_and_peft(monkeypatch, "0.45.5")
    patch_compiling_bitsandbytes()
    first = bnb_modules.Linear4bit.forward
    patch_compiling_bitsandbytes()
    second = bnb_modules.Linear4bit.forward

    assert first is second, "forward was re-wrapped on a second call"
    assert not hasattr(first.__wrapped__, "__wrapped__"), "double wrap detected"


def test_patch_compiling_bitsandbytes_new_bnb_is_noop(monkeypatch):
    """bitsandbytes >= 0.46.0 supports torch.compile: nothing gets wrapped,
    checked at the exact 0.46.0 boundary."""
    from unsloth_zoo.patching_utils import patch_compiling_bitsandbytes

    bnb_modules, peft_leaf = _install_fake_bnb_and_peft(monkeypatch, "0.46.0")
    patch_compiling_bitsandbytes()

    assert not hasattr(bnb_modules.Linear4bit.forward, "__wrapped__")
    assert not hasattr(peft_leaf.Linear4bit.forward, "__wrapped__")


def test_patch_compiling_bitsandbytes_missing_peft_helpful_error(monkeypatch):
    """A missing peft leaf must raise the actionable install message with the
    original ImportError chained, not a bare NameError/ImportError."""
    import pytest
    from unsloth_zoo.patching_utils import patch_compiling_bitsandbytes

    _install_fake_bnb_and_peft(monkeypatch, "0.45.5", with_peft_leaf=False)
    with pytest.raises(ImportError, match="pip install peft") as excinfo:
        patch_compiling_bitsandbytes()
    assert excinfo.value.__cause__ is not None, (
        "the original ImportError must be chained (raise ... from e) so an "
        "installed-but-broken peft stays debuggable"
    )


# ---------------------------------------------------------------------------
# Gemma-4 E-series KV sharing under use_cache=False / gradient checkpointing
# (transformers#45242). These lock the zoo fix in and check its invariants on CPU.
# ---------------------------------------------------------------------------
def test_gemma4_kv_shared_patches_registered():
    """Both patch functions stay registered AND wire the carrier (not bare return stubs)."""
    import inspect
    from unsloth_zoo.temporary_patches import gemma4 as _M
    from unsloth_zoo.temporary_patches.common import TEMPORARY_PATCHES

    names = {getattr(p, "__name__", "") for p in TEMPORARY_PATCHES}
    assert "patch_Gemma4TextModel_forward_kv_shared_no_cache" in names
    assert "patch_Gemma4Model_forward_kv_shared_no_cache" in names
    for fn in (_M.patch_Gemma4TextModel_forward_kv_shared_no_cache,
               _M.patch_Gemma4Model_forward_kv_shared_no_cache):
        src = inspect.getsource(fn)
        assert "_patch_forward_for_kv_shared_no_cache" in src and "_patch_gemma4_attention_carrier" in src, (
            "the shared-KV fix must install both the model carrier-attacher and the "
            "attention carrier patch; do not reduce it to a no-op"
        )


def test_gemma4_shared_kv_carrier_semantics():
    """Carrier exposes only shared_layers + a no-op update, so non-shared layers stay cache-free."""
    from unsloth_zoo.temporary_patches.gemma4 import _Gemma4SharedKVCarrier
    c = _Gemma4SharedKVCarrier()
    assert c.shared_layers == {}
    k, v = object(), object()
    assert c.update(k, v, 0) == (k, v)
    assert c.get_seq_length() == 0
    c.shared_layers[3] = (k, v)
    assert c.shared_layers[3] == (k, v)


def test_gemma4_capability_gate(monkeypatch):
    """needs_cache() is True when the attention forward lacks shared_kv_states, False once it
    has it. Uses a synthetic module, independent of the installed transformers."""
    import sys, types, inspect
    from unsloth_zoo.temporary_patches import gemma4 as _M

    class _Buggy:
        def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kw):
            return None
    class _Fixed:
        def forward(self, hidden_states, position_embeddings, attention_mask, shared_kv_states, past_key_values=None, **kw):
            return None
    # The discriminator is the `shared_kv_states` parameter on the attention forward.
    assert "shared_kv_states" not in inspect.signature(_Buggy.forward).parameters
    assert "shared_kv_states" in inspect.signature(_Fixed.forward).parameters

    import transformers.models as tm
    pkg = types.ModuleType("transformers.models.gemma4")
    leaf = types.ModuleType("transformers.models.gemma4.modeling_gemma4")
    pkg.modeling_gemma4 = leaf
    monkeypatch.setitem(sys.modules, "transformers.models.gemma4", pkg)
    monkeypatch.setitem(sys.modules, "transformers.models.gemma4.modeling_gemma4", leaf)
    monkeypatch.setattr(tm, "gemma4", pkg, raising=False)

    leaf.Gemma4TextAttention = _Buggy
    assert _M._gemma4_kv_sharing_needs_cache() is True
    leaf.Gemma4TextAttention = _Fixed
    assert _M._gemma4_kv_sharing_needs_cache() is False


def test_gemma4_attention_carrier_substitution():
    """The wrapper supplies the carrier only when past_key_values is None; a real cache and
    the no-carrier case pass through unchanged."""
    from unsloth_zoo.temporary_patches.gemma4 import (
        _make_gemma4_attention_carrier_forward, _Gemma4SharedKVCarrier,
    )
    seen = {}
    def orig(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kw):
        seen["pkv"] = past_key_values
        return "ok"
    wrapped = _make_gemma4_attention_carrier_forward(orig)

    class M:
        pass
    m = M(); carrier = _Gemma4SharedKVCarrier(); m._unsloth_shared_kv_carrier = carrier
    wrapped(m, "h", "pe", "am", past_key_values=None)
    assert seen["pkv"] is carrier, "None past_key_values must be replaced by the carrier"
    real = object()
    wrapped(m, "h", "pe", "am", past_key_values=real)
    assert seen["pkv"] is real, "a real cache must NOT be replaced"
    m2 = M()  # no carrier attached
    wrapped(m2, "h", "pe", "am", past_key_values=None)
    assert seen["pkv"] is None, "without a carrier, pass through unchanged"


def test_gemma4_clear_shared_kv_carrier_releases_kv():
    """clear drops the pinned K/V and removes the module attr, so a no_grad/eval forward does
    not hold the carrier alive until the next forward. No-op when there is no carrier."""
    from unsloth_zoo.temporary_patches.gemma4 import (
        _Gemma4SharedKVCarrier, _gemma4_clear_shared_kv_carrier,
    )

    class _Attn:
        pass

    a0, a1 = _Attn(), _Attn()
    carrier = _Gemma4SharedKVCarrier()
    carrier.shared_layers[0] = ("k", "v")
    a0._unsloth_shared_kv_carrier = carrier
    a1._unsloth_shared_kv_carrier = carrier

    class _Model:
        pass

    model = _Model()
    model._unsloth_gemma4_attns = [a0, a1]

    _gemma4_clear_shared_kv_carrier(model)
    assert carrier.shared_layers == {}, "producer K/V must be released"
    assert not hasattr(a0, "_unsloth_shared_kv_carrier"), "carrier attr must be removed"
    assert not hasattr(a1, "_unsloth_shared_kv_carrier")

    # Idempotent / no-op when nothing is attached.
    model2 = _Model()
    _gemma4_clear_shared_kv_carrier(model2)  # no _unsloth_gemma4_attns -> no error
    model3 = _Model(); model3._unsloth_gemma4_attns = []
    _gemma4_clear_shared_kv_carrier(model3)  # empty list -> no error


def test_gemma4_force_nonreentrant_checkpointing(monkeypatch):
    """Overrides _gradient_checkpointing_func on checkpointed gemma-4 layers with a NON-reentrant
    checkpoint, resolving the pristine one past Unsloth's smart-GC shim. Non-checkpointed and
    non-gemma-4 layers are left untouched."""
    import functools
    import torch.utils.checkpoint as _ckpt
    from unsloth_zoo.temporary_patches import gemma4 as g4
    from unsloth_zoo.temporary_patches.gemma4 import _gemma4_force_nonreentrant_checkpointing

    def _pristine(function, *args, use_reentrant=None, **kw):
        return ("pristine", use_reentrant)
    def unsloth_checkpoint(function, *args, **kw):
        return "shim"
    unsloth_checkpoint.__name__ = "unsloth_checkpoint"  # reentrant-forcing shim
    def unsloth_offloaded_gradient_checkpoint(function, *args, **kw):
        return "shim2"
    unsloth_offloaded_gradient_checkpoint.__name__ = "unsloth_offloaded_gradient_checkpoint"
    # Simulate STACKED Unsloth shims: checkpoint and _old_checkpoint are BOTH shims, so
    # the import-captured pristine is what must be used.
    monkeypatch.setattr(_ckpt, "checkpoint", unsloth_checkpoint, raising=False)
    monkeypatch.setattr(_ckpt, "_old_checkpoint", unsloth_offloaded_gradient_checkpoint, raising=False)
    monkeypatch.setattr(g4, "_PRISTINE_TORCH_CHECKPOINT", _pristine, raising=False)
    # Real GC patching may have stashed a set-once pristine on the live module; clear it so the
    # import-captured and _old_checkpoint-unwrap paths below are what gets exercised.
    monkeypatch.delattr(_ckpt, "_unsloth_pristine_checkpoint", raising=False)

    Gemma4TextAttention = type("Gemma4TextAttention", (), {})
    OtherAttention = type("OtherAttention", (), {})

    class _Layer:
        def __init__(self, attn_cls, gc, is_shared=False, existing="ORIG"):
            self.self_attn = attn_cls()
            self.self_attn.is_kv_shared_layer = is_shared
            self.gradient_checkpointing = gc
            self._gradient_checkpointing_func = existing

    # `on` carries a user checkpoint kwarg (preserve_rng_state=False) that must survive.
    on = _Layer(Gemma4TextAttention, True, is_shared=True,
                existing=functools.partial(unsloth_checkpoint, use_reentrant=True, preserve_rng_state=False))
    off = _Layer(Gemma4TextAttention, False)  # not checkpointed -> untouched
    other = _Layer(OtherAttention, True)      # non-gemma-4 -> untouched

    class _Model:
        # modules() must expose the attentions so the fix can detect KV sharing.
        def modules(self):
            return [self, on, on.self_attn, off, off.self_attn, other, other.self_attn]
    m = _Model()

    _gemma4_force_nonreentrant_checkpointing(m)

    # Non-E-series gemma-4 (no shared layer) must be a strict no-op.
    plain = _Layer(Gemma4TextAttention, True)  # gemma-4 but NOT a shared layer
    class _ModelNoShare:
        def modules(self):
            return [self, plain, plain.self_attn]
    mns = _ModelNoShare()
    _gemma4_force_nonreentrant_checkpointing(mns)
    assert plain._gradient_checkpointing_func == "ORIG", "no-KV-sharing model must be untouched"

    f = on._gradient_checkpointing_func
    assert isinstance(f, functools.partial), "checkpointed gemma-4 layer must be overridden"
    assert f.func is _pristine, "must resolve the PRISTINE checkpoint past stacked shims"
    assert f.keywords.get("use_reentrant") is False, "must force non-reentrant"
    assert f.keywords.get("preserve_rng_state") is False, "must preserve user checkpoint kwargs"
    assert off._gradient_checkpointing_func == "ORIG", "non-checkpointed layer left alone"
    assert other._gradient_checkpointing_func == "ORIG", "non-gemma-4 layer left alone"

    # Fallback: when no import-captured pristine, unwrap _old_checkpoint past the shim.
    monkeypatch.setattr(g4, "_PRISTINE_TORCH_CHECKPOINT", None, raising=False)
    monkeypatch.setattr(_ckpt, "_old_checkpoint", _pristine, raising=False)
    on2 = _Layer(Gemma4TextAttention, True, is_shared=True)
    class _Model2:
        def modules(self):
            return [self, on2, on2.self_attn]
    _gemma4_force_nonreentrant_checkpointing(_Model2())
    assert on2._gradient_checkpointing_func.func is _pristine, "fallback must unwrap _old_checkpoint"


def test_gemma4_pristine_checkpoint_recovered_via_set_once_capture(monkeypatch):
    """When gemma4 imports after smart GC stacked on an offloaded shim, neither shim carries a
    function-level _old_checkpoint and the module-level one points at the first shim. The set-once
    capture the GC patcher stashes before any shim must still recover the pristine fn."""
    import torch.utils.checkpoint as _ckpt
    from unsloth_zoo import gradient_checkpointing as gc
    from unsloth_zoo.temporary_patches import gemma4 as g4

    real = _ckpt.checkpoint  # genuine torch fn in the test process (no shim installed)

    # Force the "late import" path: no import-captured pristine available.
    monkeypatch.setattr(g4, "_PRISTINE_TORCH_CHECKPOINT", None, raising=False)
    monkeypatch.delattr(_ckpt, "_unsloth_pristine_checkpoint", raising=False)

    # First GC patch captures the pristine before any shim stacks.
    gc._capture_pristine_checkpoint_once()
    assert getattr(_ckpt, "_unsloth_pristine_checkpoint", None) is real

    # Now stack two shims: module-level _old_checkpoint ends up at the first shim, so unwrapping
    # it alone would only ever reach a shim.
    def unsloth_offloaded_gradient_checkpoint(*a, **k):
        raise AssertionError("a shim must never be returned as the pristine checkpoint")
    unsloth_offloaded_gradient_checkpoint.__name__ = "unsloth_offloaded_gradient_checkpoint"
    def unsloth_checkpoint(*a, **k):
        raise AssertionError("a shim must never be returned as the pristine checkpoint")
    unsloth_checkpoint.__name__ = "unsloth_checkpoint"
    monkeypatch.setattr(_ckpt, "_old_checkpoint", unsloth_offloaded_gradient_checkpoint, raising=False)
    monkeypatch.setattr(_ckpt, "checkpoint", unsloth_checkpoint, raising=False)

    assert g4._resolve_pristine_checkpoint(_ckpt) is real, (
        "set-once capture must recover the pristine checkpoint despite stacked shims"
    )
    # cleanup the attr we created (monkeypatch.delattr above only records absence)
    if hasattr(_ckpt, "_unsloth_pristine_checkpoint"):
        del _ckpt._unsloth_pristine_checkpoint


def _make_gemma4_fake_model(torch, *, kv_shared, checkpointed):
    """A stand-in Gemma-4 model: one decoder layer whose self_attn class is named
    Gemma4TextAttention, with configurable KV sharing and gradient checkpointing flags."""
    Gemma4TextAttention = type("Gemma4TextAttention", (), {})

    class _Layer:
        def __init__(self):
            self.self_attn = Gemma4TextAttention()
            self.self_attn.is_kv_shared_layer = kv_shared
            self.gradient_checkpointing = checkpointed

    layer = _Layer()

    class _Model:
        def modules(self):
            return [self, layer, layer.self_attn]

    return _Model()


def test_gemma4_carrier_rejects_two_forwards_only_when_unsafe():
    """The module-scoped 5.5.0/5.5.1 carrier cannot serve two live checkpointed graphs, so a
    second grad-enabled forward before the first's backward must raise -- but ONLY when KV sharing
    AND gradient checkpointing are both active (else no carrier is re-read in backward). Models
    without sharing (31B/26B-A4B) or with checkpointing off must keep working with DPO/contrastive."""
    import torch
    import pytest
    from unsloth_zoo.temporary_patches.gemma4 import _make_kv_shared_use_cache_false_safe_forward

    w = torch.nn.Parameter(torch.randn(4, 4))

    def _orig(self, x):
        return x @ w  # stands in for last_hidden_state; requires grad

    wrapped = _make_kv_shared_use_cache_false_safe_forward(_orig, attach_carrier=True)
    x = torch.randn(2, 4, requires_grad=True)

    # Unsafe: KV sharing + checkpointing -> arm + reject the overlapping forward.
    m = _make_gemma4_fake_model(torch, kv_shared=True, checkpointed=True)
    out1 = wrapped(m, x)
    assert getattr(m, "_unsloth_gemma4_carrier_outstanding", False) is True
    with pytest.raises(RuntimeError, match="two forward passes before a single backward"):
        wrapped(m, x)
    # after the first graph's backward the marker clears and a new forward is allowed
    out1.sum().backward()
    assert getattr(m, "_unsloth_gemma4_carrier_outstanding", False) is False
    wrapped(m, x)
    assert getattr(m, "_unsloth_gemma4_carrier_outstanding", False) is True

    # Safe (no KV sharing, e.g. 31B/26B-A4B): two forwards before one backward must NOT raise.
    m_noshare = _make_gemma4_fake_model(torch, kv_shared=False, checkpointed=True)
    wrapped(m_noshare, x)
    wrapped(m_noshare, x)  # would raise if the marker were armed
    assert getattr(m_noshare, "_unsloth_gemma4_carrier_outstanding", False) is False

    # Safe (KV sharing but checkpointing off): backward uses saved activations, never re-reads
    # the carrier, so overlapping forwards must NOT raise.
    m_nockpt = _make_gemma4_fake_model(torch, kv_shared=True, checkpointed=False)
    wrapped(m_nockpt, x)
    wrapped(m_nockpt, x)
    assert getattr(m_nockpt, "_unsloth_gemma4_carrier_outstanding", False) is False

    # eval / no_grad never arms the marker (no backward to guard)
    m_eval = _make_gemma4_fake_model(torch, kv_shared=True, checkpointed=True)
    with torch.no_grad():
        wrapped(m_eval, x)
    assert getattr(m_eval, "_unsloth_gemma4_carrier_outstanding", False) is False


# ---------------------------------------------------------------------------
# Regression: gpt-oss bnb-4bit <-> 16bit compiled-cache mode switch.
# The compiled gpt_oss module hardcodes the BnB or stock router/experts layout. Reusing a
# stale BnB-built module for a later 16bit load re-installs the BnB router -> "weights not
# initialized". patch_gpt_oss_bnb4bit_auto must restore stock classes + invalidate the module
# when the load is not bnb-4bit, and drop a stale stock module when switching into bnb.
# ---------------------------------------------------------------------------


def test_invalidate_gpt_oss_compiled_module_drops_sys_modules_and_file(tmp_path, monkeypatch):
    """_invalidate_gpt_oss_compiled_module removes the sys.modules entry and the on-disk .py so
    the next compile regenerates against the active classes."""
    import sys
    import types
    from unsloth_zoo.temporary_patches import gpt_oss as _M

    # fake an already-imported compiled module + its on-disk cache file
    sys.modules["unsloth_compiled_module_gpt_oss"] = types.ModuleType(
        "unsloth_compiled_module_gpt_oss"
    )
    cache_dir = tmp_path / "unsloth_compiled_cache"
    cache_dir.mkdir()
    cache_file = cache_dir / "unsloth_compiled_module_gpt_oss.py"
    cache_file.write_text("# stale compiled module\n")
    monkeypatch.setenv("UNSLOTH_COMPILE_LOCATION", str(cache_dir))

    try:
        _M._invalidate_gpt_oss_compiled_module()
        assert "unsloth_compiled_module_gpt_oss" not in sys.modules, (
            "stale compiled module must be evicted from sys.modules"
        )
        assert not cache_file.exists(), (
            "stale on-disk compiled module .py must be deleted"
        )
        # idempotent: a second call with nothing to clean must not raise
        _M._invalidate_gpt_oss_compiled_module()
    finally:
        sys.modules.pop("unsloth_compiled_module_gpt_oss", None)


def test_gpt_oss_bnb4bit_auto_restores_and_invalidates_on_non_4bit_load():
    """AST contract: patch_gpt_oss_bnb4bit_auto's non-bnb branch restores stock classes and
    clears UNSLOTH_GPT_OSS_BNB4BIT_PATCHED, and the function syncs the on-disk module to the
    current flavor (the sole cache invalidator). Asserted on the AST so it can't be dropped."""
    import ast
    import inspect
    from unsloth_zoo.temporary_patches import gpt_oss as _M

    src = inspect.getsource(_M.patch_gpt_oss_bnb4bit_auto)
    body = ast.unparse(ast.parse(src))

    assert "restore_gpt_oss_original" in body, (
        "non-bnb load must call restore_gpt_oss_original() so a stale BnB router "
        "swap from an earlier 4bit load does not break a later 16bit reload"
    )
    assert "_sync_gpt_oss_compiled_flavor" in body, (
        "must sync the compiled gpt_oss module to the current flavor on a "
        "bnb<->16bit mode switch (otherwise the wrong router/experts layout is "
        "re-installed); this is the sole compiled-module invalidator"
    )
    assert "UNSLOTH_GPT_OSS_BNB4BIT_PATCHED" in body, (
        "must track/clear the BNB4BIT_PATCHED flag to detect the mode switch"
    )


def test_sync_gpt_oss_compiled_flavor_drops_cross_process_stale_module(tmp_path, monkeypatch):
    """In a fresh process the in-process flag is unset, so _sync_gpt_oss_compiled_flavor must
    invalidate a stale module via the on-disk marker, and be a no-op when the flavor matches."""
    from unsloth_zoo.temporary_patches import gpt_oss as _M

    cache_dir = tmp_path / "unsloth_compiled_cache"
    cache_dir.mkdir()
    monkeypatch.setenv("UNSLOTH_COMPILE_LOCATION", str(cache_dir))
    module = cache_dir / "unsloth_compiled_module_gpt_oss.py"
    marker = cache_dir / ".unsloth_gpt_oss_compiled_flavor"

    # Prior 4bit process left a BnB-built module; a fresh 16bit (stock) load must drop it.
    module.write_text("# bnb-built\n")
    marker.write_text("bnb4bit")
    _M._sync_gpt_oss_compiled_flavor("stock")
    assert not module.exists(), "stale BnB module not invalidated for a fresh stock load"
    assert marker.read_text().strip() == "stock"

    # Matching flavor is a strict no-op (no needless recompile).
    module.write_text("# stock-built\n")
    marker.write_text("stock")
    before = module.read_text()
    _M._sync_gpt_oss_compiled_flavor("stock")
    assert module.exists() and module.read_text() == before, "matching flavor wrongly invalidated"

    # An older build with no marker is treated as unknown -> conservatively rebuilt.
    module.write_text("# unknown\n")
    if marker.exists():
        marker.unlink()
    _M._sync_gpt_oss_compiled_flavor("stock")
    assert not module.exists(), "unmarked module not conservatively invalidated"


def test_patch_gpt_oss_auto_does_not_touch_cache_for_non_gpt_loads(tmp_path, monkeypatch):
    """patch_gpt_oss_bnb4bit_auto runs for every load; loading an unrelated model must not delete
    a valid gpt-oss cache (the flavor sync is gated on the model being gpt-oss)."""
    from unsloth_zoo.temporary_patches import gpt_oss as _M

    cache_dir = tmp_path / "unsloth_compiled_cache"
    cache_dir.mkdir()
    monkeypatch.setenv("UNSLOTH_COMPILE_LOCATION", str(cache_dir))
    module = cache_dir / "unsloth_compiled_module_gpt_oss.py"
    marker = cache_dir / ".unsloth_gpt_oss_compiled_flavor"
    module.write_text("# valid bnb4bit gpt-oss compiled module\n")
    marker.write_text("bnb4bit")
    before = module.read_text()

    # Loading some other model (name has no "gpt_oss") must leave the cache intact.
    monkeypatch.setenv("UNSLOTH_MODEL_NAME", "meta-llama/Llama-3.2-1B")
    monkeypatch.setenv("UNSLOTH_GPT_OSS_BNB4BIT_PATCHED", "0")
    _M.patch_gpt_oss_bnb4bit_auto()
    assert module.exists() and module.read_text() == before, (
        "loading a non-GPT model deleted the GPT-OSS compiled cache"
    )
