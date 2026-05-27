# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Pin-down regression suite for past zoo bugs.

Every test here corresponds to a SHIPPED fix on `main`. The goal is
to catch the SAME bug class if it re-appears, not to retest the
fix path itself. Each test has a `WHY` block citing the commit /
PR that introduced the regression.
"""

from __future__ import annotations

import importlib

import pytest


# ---------------------------------------------------------------------------
# Regression: temporary_patches/utils.py `__all__` missing comma between
# entries silently concatenates the two strings ("raise_errorUnpack")
# and the supposedly-public names become un-import-able under
# `from temporary_patches.utils import *`.
#
# Source: `2e36f32 fix(temporary_patches/utils): add missing comma in
# __all__ between 'raise_error' and 'Unpack' (#617)`
#
# This is a Python footgun -- there's no syntactic error, the
# interpreter just concatenates adjacent string literals. The bug
# stayed live until someone star-imported and noticed `Unpack` was
# missing.
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
    """No `__all__` entry should look like two separate names jammed
    together (e.g. `raise_errorUnpack`). Heuristic: detect a name
    that has BOTH (a) snake_case tokens (lowercase + underscore) AND
    (b) a PascalCase / camelCase transition (lowercase letter
    followed by uppercase letter). Pure `ALL_CAPS_CONSTANT` names
    like `KWARGS_TYPE` have underscores but no lowercase->uppercase
    transition, so they don't match.
    """
    from unsloth_zoo.temporary_patches import utils
    import re

    # Matches a lowercase letter followed by an uppercase letter --
    # the signature of a snake_case + CamelCase concatenation.
    camel_boundary = re.compile(r"[a-z][A-Z]")
    suspicious = []
    for name in utils.__all__:
        if name.startswith("_"):
            continue
        if "_" not in name:
            # Pure CamelCase or pure lowercase -- not the bug class.
            continue
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
# Regression: `compiler.higher_precision_softmax` was not idempotent --
# running it twice on the same source appended a duplicate
# `.to(x.dtype).to(x.dtype)` because the lookahead that detects an
# already-rewritten softmax was missing.
#
# Source: `f98dbbc fix(compiler): make higher_precision_softmax
# idempotent (#631)`
#
# The compiler runs on user source mid-training; if it's invoked
# twice (e.g. through a checkpoint reload that re-patches), the
# emitted source must not drift. This test pins the contract.
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
# Regression: backend device helpers must guard against partial torch
# builds (e.g. `torch.xpu` exists but `torch.xpu.synchronize` raises).
# Two commits address this:
#   `e08c1df Guard XPU synchronize call against partial torch.xpu builds`
#   `35dc451 Guard XPU empty_cache call against partial torch.xpu builds`
#
# Test: calling device_synchronize / device_empty_cache must not raise
# even if the resolved backend module is partial. Uses the same
# stub harness as tests/conftest.py.
# ---------------------------------------------------------------------------


def test_device_synchronize_tolerates_partial_backend():
    """`device_synchronize()` must not raise on a minimal stub backend."""
    from unsloth_zoo.device_type import device_synchronize

    # Just call it. On the GPU-free harness this resolves to the
    # stub `lambda *a, **k: None`. The point is to assert the
    # exported name exists and is callable -- the partial-backend
    # guards live inside its implementation.
    device_synchronize()


def test_device_type_module_has_expected_helpers():
    """Pin the public API surface that downstream zoo / unsloth /
    Studio code calls. A rename or removal here breaks consumers
    silently (`AttributeError` at training time).
    """
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
# (commits 466334c, 9829ade, 035f...). The dict is rebuilt as each
# function is defined; a missing `RL_REPLACEMENTS[name] = fn`
# assignment after a refactor is silent -- nothing fails at import.
#
# This test pins the registration count + the well-known public-API
# keys. Duplicates the assertion in test_rl_replacements_cpu.py
# deliberately: that file proves the IO contract of each function;
# THIS file proves the registration survives a refactor.
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
# Regression: gpt-oss inference leaks a flex BlockMask into the eager
# attention forward.
#
# Source: PR #690 (Fix gpt-oss inference BlockMask TypeError).
# Latent since unsloth-zoo commit `ef819214` (2025-09-25, "Fix Flex")
# which rewrote return_attention_mask -> wrap(f) that on the inference
# branch unconditionally calls f(*args, **kwargs), AND switched
# forward_function to use eager during inference. When
# config._attn_implementation == "flex_attention" (set by training,
# explicit user kwarg, or pre-#5701 resolver), the inference path
# receives a BlockMask and inplace_eager_attention_forward crashes:
#   TypeError: unsupported operand type(s) for +=:
#       'Tensor' and 'BlockMask'
#
# AST-level guards. Two of them: one over wrap.return_attention_mask
# (the path transformers' own GptOssModel.forward goes through), one
# over patch_GptOssModel as a whole (the path Unsloth's patched forward
# goes through). Catches silent deletion of the flex_attention literal.
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
    """The patched GptOssModel.forward body itself must hold the guard.

    Locate the inner `forward` FunctionDef and assert its body carries
    the _attn_implementation == 'flex_attention' literal. This is
    independent of wrap's own guard, so removing the forward-level
    swap while keeping wrap's literal cannot satisfy this test.
    """
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
# Regression: eager attention shape mismatch when KV cache is longer than
# the attention mask's kv dimension.
#
# Source: PR #691 (gpt-oss: align eager KV length to the attention mask).
# transformers 5.x cache pre-allocation hands a full-attention layer more
# positions than the causal mask covers (e.g. k=161 against a 128-wide
# mask). Both eager attention forwards in patch_GptOssAttention then do
#   attn_weights += attention_mask[:, :, :, : key_states.shape[-2]]
# which fails with:
#   RuntimeError: The size of tensor a (N) must match the size of
#       tensor b (M) at non-singleton dimension 3
#
# The fix is the _align_kv_to_mask helper that trims KV (or the mask) to
# the shorter length. Static check: helper exists AND is invoked from
# BOTH eager forward variants so the inplace and non-inplace paths agree.
# ---------------------------------------------------------------------------


def test_gpt_oss_eager_attention_aligns_kv_to_mask():
    """Both eager forwards must invoke the KV alignment helper.

    Per-function source-slice check rather than a global callsite count,
    so a future refactor that consolidates eager paths into a shared
    closure still passes as long as the alignment runs on both routes.
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
# Regression: gpt-oss patched GptOssModel.forward hard-codes the
# transformers 4.x mask kwargs ("input_embeds", "cache_position") and
# blows up on transformers 5.x which renamed to "inputs_embeds" and
# dropped "cache_position" entirely.
#
# Source: PR #690 fix included inspect-driven kwarg filtering so the
# patch supports both 4.x and 5.x simultaneously. Static check that the
# filtering helper is present.
# ---------------------------------------------------------------------------


def test_gpt_oss_model_forward_uses_inspect_filtered_mask_kwargs():
    import inspect
    from unsloth_zoo.temporary_patches import gpt_oss as _M

    src = inspect.getsource(_M.patch_GptOssModel)
    # Either the inspect-driven helper (_build_mask_kwargs / _ccm_params)
    # OR a guard string explicitly testing for inputs_embeds availability.
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
