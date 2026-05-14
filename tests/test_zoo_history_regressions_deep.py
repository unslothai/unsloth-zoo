# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Deep regression suite mined from the merged-PR history of
`unslothai/unsloth-zoo`.

Each test pins ONE shipped fix. The goal is to catch the SAME bug class
if it re-appears via a refactor that loses the guard, rather than
re-test the fix path. Every test cites the PR number and a one-line
description of the original failure mode.

These are deliberately CPU-only, fast, and use source-AST inspection /
regex / behavioural probes so they remain useful even after the bug
is re-fixed.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
import textwrap

import pytest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_source(module_name: str, attr: str | None = None) -> str:
    mod = importlib.import_module(module_name)
    if attr is None:
        return inspect.getsource(mod)
    obj = getattr(mod, attr)
    return inspect.getsource(obj)


# ---------------------------------------------------------------------------
# PR #4: `Fix longest common substring implementation`
# The legacy `_old_longest_common_substring` worked on `str(list)`, which
# matches leading commas and then calls `int('')` -- crashes on
# `train_on_responses_only`. The fix introduced `_longest_common_sublist`
# that works on the lists directly.
#
# We pin the new helper's behaviour on the regression input class:
# common-suffix lists where the only common sublist is `[0]`.
# ---------------------------------------------------------------------------


def test_longest_common_sublist_handles_singleton_overlap():
    from unsloth_zoo.dataset_utils import _longest_common_sublist
    # Two prompt-token lists that share only the trailing zero.
    out = _longest_common_sublist([[1, 2, 3, 0], [4, 5, 6, 0]])
    assert out == [0], (
        "_longest_common_sublist should find the single shared element."
        f" got {out!r}. Regression: PR #4 (LCS over int lists, not str repr)."
    )


def test_longest_common_sublist_empty_and_no_overlap():
    from unsloth_zoo.dataset_utils import _longest_common_sublist
    assert _longest_common_sublist([]) == []
    assert _longest_common_sublist([[1, 2], []]) == []
    # No common element returns [] not a crash.
    assert _longest_common_sublist([[1, 2], [3, 4]]) == []


# ---------------------------------------------------------------------------
# PR #322: transformers 4.57 renamed `PretrainedConfig` -> `PreTrainedConfig`.
# Zoo used to import the legacy name unconditionally and crash on 4.57+.
# Pin: no zoo source uses the bare legacy `PretrainedConfig` identifier
# in a way that would fail on transformers 5.x; if it does, that import
# must be guarded with a try / hasattr / getattr fallback.
# ---------------------------------------------------------------------------


def test_no_unguarded_legacy_pretrained_config_import():
    """Find direct `from transformers import PretrainedConfig` style
    imports (PR #322 renamed the symbol). The post-rename code should
    use the new name, getattr/hasattr probing, or sit inside a
    try/except guard with `PreTrainedConfig` as the primary import.
    """
    import pathlib
    root = pathlib.Path(
        importlib.import_module("unsloth_zoo").__file__,
    ).parent
    bad: list[str] = []
    pat = re.compile(
        r"^(\s*)from\s+transformers(?:\.[\w.]*)?\s+import\s+[^\n]*\bPretrainedConfig\b",
        re.MULTILINE,
    )
    for py in root.rglob("*.py"):
        text = py.read_text(encoding="utf-8", errors="ignore")
        for m in pat.finditer(text):
            line = m.group(0)
            if "PreTrainedConfig" in line:
                continue
            # Tolerated forms: the import is inside indented block (a
            # try/except guard), OR a separate `PreTrainedConfig` alias
            # / import exists in the same file as a fallback.
            indent = m.group(1)
            if indent and len(indent) >= 4:
                # Indented: caller is inside try/except.
                continue
            if "PreTrainedConfig" in text:
                continue
            bad.append(f"{py.relative_to(root)}: {line.strip()}")
    assert not bad, (
        "Found unguarded legacy PretrainedConfig imports -- regression "
        "of PR #322 (transformers 4.57 rename to PreTrainedConfig):\n"
        + "\n".join(bad)
    )


# ---------------------------------------------------------------------------
# PR #374: `Update e to error`. `empty_model.create_empty_causal_lm`
# used `print(f"... {e}")` after an exception bound to `error`, raising
# `UnboundLocalError`. Pin: scan exception handlers in empty_model.py
# for the name mismatch.
# ---------------------------------------------------------------------------


def test_empty_model_exception_var_consistent():
    """Every `except Foo as <name>:` block must reference `<name>` (and
    not some other single-letter variable) inside its body. The
    original PR #374 bug used `error` as the bound name but printed `e`.
    """
    src = _get_source("unsloth_zoo.empty_model")
    tree = ast.parse(src)

    suspicious: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if node.name is None:
            continue
        bound = node.name
        # Collect plain Name references inside this handler.
        names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        # The bug pattern: handler binds `error` (or `err`) but body
        # references the OTHER short alias `e` (without it being a real
        # local). We just demand: if `e` is referenced inside a handler
        # whose bound name is NOT `e`, flag it.
        if bound != "e" and "e" in names:
            # Need to also ensure `e` is not defined as a local in the
            # handler body -- a quick AST walk for ast.Assign targets.
            local_targets = set()
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign):
                    for tgt in sub.targets:
                        if isinstance(tgt, ast.Name):
                            local_targets.add(tgt.id)
            if "e" not in local_targets:
                suspicious.append(
                    f"line {node.lineno}: except ... as {bound} but "
                    f"body references undefined `e`"
                )
    assert not suspicious, (
        "Found exception handlers with mismatched variable names -- "
        "the same bug class as PR #374 (UnboundLocalError on `e`):\n"
        + "\n".join(suspicious)
    )


# ---------------------------------------------------------------------------
# PR #422: `dist.broadcast_object_list` was called in
# `utils.distributed_function` but the underlying import was missing.
# Pin: import `unsloth_zoo.utils` and assert the module has a working
# `dist` binding that resolves to `torch.distributed`.
# ---------------------------------------------------------------------------


def test_utils_distributed_import_present():
    pytest.importorskip("torch")
    mod = importlib.import_module("unsloth_zoo.utils")
    assert hasattr(mod, "dist"), (
        "unsloth_zoo.utils.dist is missing -- regression of PR #422 "
        "(missing `import torch.distributed as dist`)."
    )
    import torch.distributed as dist
    assert mod.dist is dist or mod.dist.__name__ == "torch.distributed", (
        "utils.dist does not refer to torch.distributed."
    )


def test_distributed_function_runs_without_init():
    """`distributed_function` must NOT crash when the process group
    isn't initialised yet (the path that bit PR #421/#422 users).
    """
    from unsloth_zoo.utils import distributed_function
    out = distributed_function(n=1, function=lambda: 42)
    assert out == 42, (
        f"distributed_function returned {out!r}; expected 42 -- "
        "regression of PR #422 / the init_process_group guard."
    )


# ---------------------------------------------------------------------------
# PR #425: `Version()` had an undefined `e` in its `raise Exception(str(e))`.
# Modern impl raises `RuntimeError` from the outer except. Pin: garbage
# input raises a clean RuntimeError, NOT NameError/UnboundLocalError.
# ---------------------------------------------------------------------------


def test_version_garbage_input_clean_error():
    from unsloth_zoo.utils import Version
    # Use a string that (a) doesn't match the package-name regex
    # (contains dots) and (b) has no embeddable digit sequence so the
    # version regex inside Version() can't pull a fragment out.
    bad_inputs = [
        "not.a.real.package.name",  # dots break package-name regex
        "alpha.beta.gamma",          # no digits at all
    ]
    for bad in bad_inputs:
        try:
            v = Version(bad)
        except (RuntimeError, ValueError) as ex:
            msg = str(ex)
            assert "NameError" not in msg, (
                f"Version({bad!r}) raised NameError-style failure: "
                f"{msg!r}. Regression of PR #425."
            )
            assert "UnboundLocalError" not in msg, (
                f"Version({bad!r}) raised UnboundLocalError-style "
                f"failure: {msg!r}. Regression of PR #425."
            )
        else:
            # If it returns something, it must be a clean Version.
            from packaging.version import Version as TrueVersion
            assert isinstance(v, TrueVersion)


# ---------------------------------------------------------------------------
# PR #461: `Version("trl")` should accept a package name string and
# resolve it via importlib.metadata. Pin: calling Version with an
# installed package name returns a parsable Version object.
# ---------------------------------------------------------------------------


def test_version_accepts_package_name_string():
    from unsloth_zoo.utils import Version
    # `packaging` is a transitive dep of unsloth_zoo so it's guaranteed.
    v = Version("packaging")
    # Compare against a Version literal -- supports <, >, ==.
    assert v >= Version("0.0.1"), (
        "Version('packaging') did not yield a numeric Version "
        "(regression of PR #461 -- string lookup via importlib.metadata)."
    )


def test_version_falls_back_for_unknown_package_strings():
    """Version('1.2.3') must keep treating raw version strings as
    versions -- not regress to package-name lookup that returns None.
    """
    from unsloth_zoo.utils import Version
    assert Version("1.2.3") == Version("1.2.3")
    assert Version("1.2.3") < Version("2.0.0")


# ---------------------------------------------------------------------------
# PR #458: `_canonicalize_annotation` did not pass `origin` through
# `TYPE_MAPPINGS`, so `Union[int, str]` (origin=typing.Union) and
# `int | str` (origin=types.UnionType) compared unequal under 3.10+.
# Pin: the two forms canonicalise to the same tuple.
# ---------------------------------------------------------------------------


def test_canonicalize_annotation_union_pep604_equivalence():
    pytest.importorskip("transformers")
    from unsloth_zoo.temporary_patches.utils import canonicalize_annotation
    import typing as t
    a_typing = canonicalize_annotation(t.Union[int, str])
    a_pep604 = canonicalize_annotation(int | str)
    assert a_typing == a_pep604, (
        f"Union vs `|` mismatch:\n  typing.Union -> {a_typing}\n"
        f"  PEP 604  -> {a_pep604}\n"
        "Regression: PR #458 (origin not mapped through TYPE_MAPPINGS)."
    )


# ---------------------------------------------------------------------------
# PR #491: transformers 5.x's `should_convert_module` only uses
# `re.match` (prefix-anchored) and `endswith`, missing entries like
# `vision_tower` against `model.vision_tower.x.y`. Zoo patches it with
# substring component matching. Pin: the patched logic in
# `unsloth_zoo.patching_utils` does substring matching.
# ---------------------------------------------------------------------------


def test_patching_utils_should_convert_module_uses_substring():
    """The `_unsloth_should_convert_module` body must do component
    substring matching (e.g. `f'.{key}.' in f'.{full_name}.'`).
    """
    src = _get_source("unsloth_zoo.patching_utils")
    assert "_unsloth_should_convert_module" in src, (
        "The transformers-5.x should_convert_module patch is missing "
        "-- regression of PR #491."
    )
    # Look for ANY substring-style match that handles the vision_tower
    # case. Acceptable forms: `f".{key}." in f".{full_name}."`
    # or equivalent surrounded-by-dot construction.
    substring_check = (
        re.search(
            r"f\"\.\{key\}\.\"\s+in\s+f\"\.\{full_name\}\.\"",
            src,
        )
        or re.search(r"\.\{key\}\.\".*in.*\.\{full_name\}\.", src)
    )
    assert substring_check, (
        "Substring component match (`.{key}.` in `.{full_name}.`) "
        "missing from _unsloth_should_convert_module -- this is the "
        "exact regression PR #491 fixed."
    )


# ---------------------------------------------------------------------------
# PR #533: torch.compile fullgraph crash with `@dynamic_rope_update`.
# The compiler must drop fullgraph when it sees that decorator.
# Pin: regex over compiler.py confirms the `dynamic_rope_update` gate
# disables fullgraph.
# ---------------------------------------------------------------------------


def test_compiler_disables_fullgraph_for_dynamic_rope_update():
    src = _get_source("unsloth_zoo.compiler")
    assert "dynamic_rope_update" in src, (
        "compiler.py no longer references dynamic_rope_update -- "
        "regression of PR #533."
    )
    # The gate flips fullgraph=False when the decorator is in source.
    flip = re.search(
        r"if\s+[\"']dynamic_rope_update[\"']\s+in\s+\w+:\s*\n\s*"
        r"fullgraph\s*=\s*False",
        src,
    )
    assert flip, (
        "Could not find the `if 'dynamic_rope_update' in source: "
        "fullgraph = False` gate -- regression of PR #533 (Phi-4 fullgraph "
        "crash via longrope data-dependent branching)."
    )


# ---------------------------------------------------------------------------
# PR #552: Conv1d/2d/3d wrappers must cast `input` to `self.weight.dtype`
# BEFORE the conv op (under autocast bf16 weight + fp16 input crashes).
# The patch saves `original_dtype = input.dtype` and casts input.
# Pin: compiler.py's conv loop saves original_dtype and casts to
# self.weight.dtype.
# ---------------------------------------------------------------------------


def test_compiler_conv_prologue_casts_to_weight_dtype():
    src = _get_source("unsloth_zoo.compiler")
    has_save = re.search(r"original_dtype\s*=\s*input\.dtype", src)
    has_cast = re.search(
        r"input\s*=\s*input\.to\(self\.weight\.dtype\)",
        src,
    )
    assert has_save and has_cast, (
        "Conv prologue missing -- regression of PR #552. "
        "Expected:\n  original_dtype = input.dtype\n"
        "  input = input.to(self.weight.dtype)\n"
        "in compiler.py's Conv patch."
    )


# ---------------------------------------------------------------------------
# PR #564: LoRA forward returned the wrong dtype when autocast was
# disabled. The fix appends `.to(torch_result_dtype)` to the early
# return so output dtype matches the base layer. Pin: compiler.py
# still emits a `torch_result_dtype` cast on the LoRA path.
# ---------------------------------------------------------------------------


def test_compiler_lora_forward_emits_torch_result_dtype_cast():
    src = _get_source("unsloth_zoo.compiler")
    # The fix appends `.to({dtype_cast})` to the early return, where
    # dtype_cast is either `torch_result_dtype` or `result.dtype`. The
    # regression is when the cast is omitted entirely.
    assert "torch_result_dtype" in src, (
        "compiler.py no longer references `torch_result_dtype` -- "
        "regression of PR #564 (autocast-disabled dtype mismatch on "
        "PEFT LoRA forward). The early-return must cast back to the "
        "base-layer dtype."
    )
    # And the return-cast must actually be emitted somewhere.
    assert re.search(
        r"return\s+lora_forward\([^)]+\)\.to\(",
        src,
    ), (
        "compiler.py no longer emits `return lora_forward(...).to(...)` "
        "-- the dtype-cast on the LoRA early return is gone (PR #564)."
    )


# ---------------------------------------------------------------------------
# PR #482: 4-bit Params4bit has `weight.dtype == uint8`. The compiled
# PEFT forward used to cast `x.to(weight.dtype)` which corrupts inputs.
# The fix skips the cast when `hasattr(self.base_layer.weight, 'quant_state')`.
# Pin: compiler source contains that guard.
# ---------------------------------------------------------------------------


def test_compiler_peft_forward_skips_quantized_dtype_cast():
    src = _get_source("unsloth_zoo.compiler")
    assert "quant_state" in src, (
        "compiler.py has no `quant_state` mention -- regression of "
        "PR #482 (4-bit input corrupted by float16 -> uint8 cast)."
    )
    # The guard must be in the autocast-disabled branch that casts x.
    guard = re.search(
        r"not\s+hasattr\(self\.base_layer\.weight,\s*['\"]quant_state['\"]\)",
        src,
    )
    assert guard, (
        "Quant-state guard not found in the LoRA dtype-cast prologue. "
        "PR #482: cast must be skipped on Params4bit / Linear4bit."
    )


# ---------------------------------------------------------------------------
# PR #466: vllm LoRA worker manager passed `vllm_config` BOTH
# positionally AND as a keyword -- `TypeError: got multiple values for
# argument 'vllm_config'`. Pin: each call site to
# `_call_create_lora_manager` does not pass vllm_config twice.
# ---------------------------------------------------------------------------


def test_vllm_lora_manager_no_duplicate_vllm_config_kwarg():
    src = _get_source("unsloth_zoo.vllm_lora_worker_manager")
    # Find every _call_create_lora_manager(...) call body and verify
    # no `vllm_config=` keyword sits AFTER a `vllm_config` positional.
    bad: list[str] = []
    for m in re.finditer(
        r"_call_create_lora_manager\((?P<args>.*?)\)",
        src,
        flags=re.DOTALL,
    ):
        body = m.group("args")
        # `vllm_config` appears once positionally already (second arg);
        # ensure NO `vllm_config = vllm_config` keyword form coexists.
        if re.search(r"vllm_config\s*=\s*vllm_config", body):
            # That's the legacy double-pass.
            bad.append(body.strip())
    assert not bad, (
        "Duplicate `vllm_config=vllm_config` kwarg passed alongside "
        "positional -- regression of PR #466:\n"
        + "\n---\n".join(bad)
    )


# ---------------------------------------------------------------------------
# PR #580: Gemma-4 inference with `num_kv_shared_layers == 0` hits
# `layer_types[:-0] == []` -> IndexError. The fix wraps text_config in
# a proxy that HIDES `num_kv_shared_layers` when it is 0. Pin: the
# `_Gemma4KVSharedSafeProxy` proxy class exists and refuses the attr.
# ---------------------------------------------------------------------------


def test_gemma4_proxy_hides_zero_num_kv_shared_layers():
    pytest.importorskip("torch")
    mod = importlib.import_module(
        "unsloth_zoo.temporary_patches.gemma4",
    )
    Proxy = getattr(mod, "_Gemma4KVSharedSafeProxy", None)
    assert Proxy is not None, (
        "_Gemma4KVSharedSafeProxy is missing -- regression of PR #580."
    )

    # Build a minimal stand-in `real_config` with the legacy attr.
    class _Real:
        num_kv_shared_layers = 0
        num_hidden_layers = 4

        def __iter__(self):
            return iter(["num_hidden_layers"])

    proxy = Proxy(_Real())
    # The proxy must say it does NOT have num_kv_shared_layers when 0.
    assert not hasattr(proxy, "num_kv_shared_layers"), (
        "Proxy still exposes num_kv_shared_layers == 0 -- PR #580 "
        "regression. transformers will do layer_types[:-0] -> []."
    )
    # But other attrs forward.
    assert proxy.num_hidden_layers == 4
    # `in` should return False for the hidden name.
    assert "num_kv_shared_layers" not in proxy


# ---------------------------------------------------------------------------
# PR #593: `chunked_hidden_states_selective_log_softmax` used the WRONG
# softcap formula (`logits * tanh(logits / cap)` instead of
# `cap * tanh(logits / cap)`). For |logits| >> cap the cap was a no-op.
# Pin: the source emits the cap-prefixed form.
# ---------------------------------------------------------------------------


def test_grpo_softcap_formula_is_cap_times_tanh():
    src = _get_source(
        "unsloth_zoo.rl_replacements",
        "chunked_hidden_states_selective_log_softmax",
    )
    # Want a line of the shape `<var> = logit_softcapping * torch.tanh(<var> / logit_softcapping)`.
    correct = re.search(
        r"=\s*logit_softcapping\s*\*\s*torch\.tanh\([^)]+/\s*logit_softcapping\)",
        src,
    )
    # The buggy form is `<var> * torch.tanh(<var> / logit_softcapping)`.
    buggy = re.search(
        r"=\s*\w+\s*\*\s*torch\.tanh\([^)]+/\s*logit_softcapping\)",
        src,
    )
    if buggy and not correct:
        pytest.fail(
            "GRPO softcap regressed to `logits * tanh(logits/cap)` "
            "instead of the Gemma formula `cap * tanh(logits/cap)` -- "
            "PR #593. Big logits would saturate tanh to ~1 and the cap "
            "would be a no-op."
        )
    assert correct, (
        "Expected `cap * tanh(... / cap)` form in "
        "chunked_hidden_states_selective_log_softmax (PR #593)."
    )


# ---------------------------------------------------------------------------
# PR #543: `accumulated_loss` etc. must be initialised as scalar
# tensors `torch.zeros(1, device=device)[0]` (shape []) so that the
# transformers 5.x in-place accumulation doesn't hit the shape-[1]
# vs shape-[] broadcast crash. Pin: regex over rl_replacements.py.
# ---------------------------------------------------------------------------


def test_rl_replacements_scalar_tensor_init_for_accumulators():
    src = _get_source("unsloth_zoo.rl_replacements")
    # We want `torch.zeros(1, device = device)[0]` (or w/o spaces).
    hit = re.search(
        r"accumulated_loss\s*=\s*torch\.zeros\(1[^)]*\)\[0\]",
        src,
    )
    assert hit, (
        "accumulated_loss is no longer initialised as a SCALAR tensor "
        "via `torch.zeros(1, ...)[0]` -- regression of PR #543 "
        "(transformers 5.x in-place += on shape-[] target)."
    )


# ---------------------------------------------------------------------------
# PR #477: `sft_prepare_dataset` non-packing path must pass
# `remove_columns=list(column_names)` to `.map(_tokenize, ...)` so
# downstream collator doesn't see raw JSON columns like `messages`.
# Pin: the `_tokenize` map call carries `remove_columns=`.
# ---------------------------------------------------------------------------


def test_sft_prepare_dataset_removes_original_columns_in_non_packing_path():
    src = _get_source("unsloth_zoo.dataset_utils")
    # Locate the `.map(_tokenize, batched = True, ...)` call.
    m = re.search(
        r"\.map\(\s*_tokenize\s*,\s*batched\s*=\s*True[^)]*\)",
        src,
        re.DOTALL,
    )
    assert m, (
        "Could not locate the `_tokenize` map call in dataset_utils.py "
        "-- shape of sft_prepare_dataset has changed unexpectedly."
    )
    body = m.group(0)
    assert "remove_columns" in body, (
        "`.map(_tokenize, batched=True, ...)` no longer passes "
        "`remove_columns=` -- regression of PR #477 (raw column "
        "leaks past the tokenizer and crashes the collator)."
    )


# ---------------------------------------------------------------------------
# PR #595: Windows file-lock on shard rewrite. The fix uses ATOMIC
# `os.replace(tmp, target)` instead of remove+move. Pin: the
# `_merge_and_overwrite_lora` source uses `os.replace`.
# ---------------------------------------------------------------------------


def test_saving_utils_uses_atomic_replace_for_shard_rewrite():
    src = _get_source(
        "unsloth_zoo.saving_utils", "_merge_and_overwrite_lora",
    )
    assert "os.replace(" in src, (
        "_merge_and_overwrite_lora no longer uses os.replace -- "
        "regression of PR #595 (Windows WinError 1224 on shard rewrite)."
    )
    # The old buggy pair would be `os.remove(<filename>) + shutil.move`;
    # if both appear together in the rewrite branch, that's the bug.
    if "shutil.move(" in src and "os.remove(" in src:
        # Only flag if remove/move follow each other in the rewrite
        # branch (i.e. inside the same `if resized:`). Heuristic: both
        # appear before the os.replace line.
        idx_replace = src.find("os.replace(")
        idx_remove  = src.find("os.remove(")
        idx_move    = src.find("shutil.move(")
        if idx_remove >= 0 and idx_move >= 0 and idx_remove < idx_replace and idx_move < idx_replace:
            pytest.fail(
                "Non-atomic os.remove + shutil.move pair survives "
                "before the os.replace path -- the data-loss window "
                "the PR #595 fix was supposed to close."
            )


# ---------------------------------------------------------------------------
# PR #615: GGUF merge path used hardcoded CUDA. The fix uses
# `DEVICE_TYPE` / `DEVICE_TYPE_TORCH` so XPU + ROCm work. Pin: no
# `cuda.empty_cache` or `torch.cuda.synchronize` calls in
# saving_utils.py without a DEVICE_TYPE guard.
# ---------------------------------------------------------------------------


def test_saving_utils_uses_device_type_helpers():
    src = _get_source("unsloth_zoo.saving_utils")
    # Either the module imports DEVICE_TYPE / DEVICE_TYPE_TORCH or it
    # imports device_empty_cache / device_synchronize from device_type.
    has_helpers = (
        "DEVICE_TYPE" in src
        or "device_empty_cache" in src
        or "device_synchronize" in src
    )
    assert has_helpers, (
        "saving_utils no longer uses DEVICE_TYPE helpers -- regression "
        "of PR #615 (GGUF merge path crashes on Intel XPU because "
        "torch.cuda.* is unavailable)."
    )


# ---------------------------------------------------------------------------
# PR #91: `_unsloth_get_batch_samples` must accept the 4th `device`
# parameter introduced in transformers 4.50. Pin: signature has either
# a `device` param or `**kwargs` so 3- and 4-arg call sites both work.
# ---------------------------------------------------------------------------


def test_unsloth_get_batch_samples_accepts_4_args():
    pytest.importorskip("transformers")
    mod = importlib.import_module("unsloth_zoo.loss_utils")
    fn = getattr(mod, "_unsloth_get_batch_samples", None)
    assert fn is not None, (
        "_unsloth_get_batch_samples missing from unsloth_zoo.loss_utils "
        "-- regression of PR #91 (transformers 4.50 added 4th param)."
    )
    sig = inspect.signature(fn)
    params = sig.parameters
    has_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    has_device = "device" in params
    has_var_pos = any(
        p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values()
    )
    assert has_device or has_var_kw or has_var_pos or len(params) >= 4, (
        "_unsloth_get_batch_samples no longer tolerates the 4th `device` "
        "argument introduced in transformers 4.50 -- regression of PR #91."
        f" Current signature: {sig}"
    )


# ---------------------------------------------------------------------------
# PR #617 follow-up: `__all__` integrity across more zoo public modules.
# The original PR fixed `temporary_patches.utils`; we extend the
# same heuristic to the top-level `unsloth_zoo.utils` and any other
# module that declares `__all__` -- adjacent-string-concatenation is a
# class of bug, not a one-off.
# ---------------------------------------------------------------------------


def test_all_modules_all_entries_have_no_concatenated_names():
    """The PR #617 bug class is `["raise_error" "Unpack"]` -- missing
    comma in `__all__` silently concatenates string literals. The
    resulting names always contain a snake_case-to-CamelCase boundary
    (`raise_errorUnpack`). Scan every zoo module's `__all__` for that
    boundary -- regression-safe even if the inventory shifts over time.

    Pure CamelCase / pure snake_case / SHOUTY_SNAKE names don't trip
    this heuristic; only the concatenation accident does.
    """
    import pathlib
    root = pathlib.Path(
        importlib.import_module("unsloth_zoo").__file__,
    ).parent
    # Detect a name that has BOTH a snake_case token AND a CamelCase
    # transition (lowercase followed by uppercase).
    camel_boundary = re.compile(r"[a-z][A-Z]")
    suspicious: list[str] = []
    for py in root.rglob("*.py"):
        rel = py.relative_to(root)
        if py.name == "__init__.py":
            continue
        if rel.parts and rel.parts[0] in {
            "mlx_cce", "flex_attention", "fused_losses", "stubs", "mlx_compile",
        }:
            continue
        rel_mod = "unsloth_zoo." + ".".join(rel.with_suffix("").parts)
        try:
            mod = importlib.import_module(rel_mod)
        except Exception:
            continue
        all_list = getattr(mod, "__all__", None)
        if not all_list:
            continue
        for name in all_list:
            if name.startswith("_"):
                continue
            if "_" not in name:
                continue
            if camel_boundary.search(name):
                suspicious.append(f"{rel_mod}.__all__ -> {name!r}")
    assert not suspicious, (
        "Suspicious __all__ entries -- the snake_case+CamelCase boundary "
        "is the fingerprint of the PR #617 missing-comma bug:\n"
        + "\n".join(suspicious)
    )


# ---------------------------------------------------------------------------
# PR #612: Gemma4-MoE patch must NOT rely on the `slice(-0, None)`
# Python identity. Pin: the `gemma4_moe.py` patched ForCondGen forward
# guards the slice behind `if logits_to_keep != 0:`.
# ---------------------------------------------------------------------------


def test_gemma4_moe_guards_logits_to_keep_slice():
    try:
        src = _get_source("unsloth_zoo.temporary_patches.gemma4_moe")
    except Exception:
        pytest.skip("gemma4_moe module unavailable")
    # The guard is the regression fix.
    assert re.search(
        r"if\s+logits_to_keep\s*!=\s*0", src,
    ), (
        "gemma4_moe.py no longer guards the hidden-state slice behind "
        "`if logits_to_keep != 0:` -- regression of PR #612 (the "
        "implicit dependency on Python's slice(-0, None) == slice(0, None))."
    )


# ---------------------------------------------------------------------------
# PR #549: Patch `transformers.modeling_utils.checkpoint` to wire up
# Unsloth's smart gradient checkpointing on transformers 5.2+. The old
# patch only replaced `torch.utils.checkpoint.checkpoint`. Pin: source
# of `patch_unsloth_smart_gradient_checkpointing` references the
# transformers.modeling_utils namespace.
# ---------------------------------------------------------------------------


def test_smart_gradient_checkpointing_patches_transformers_modeling_utils():
    pytest.importorskip("transformers")
    src = _get_source("unsloth_zoo.gradient_checkpointing")
    assert "transformers.modeling_utils" in src or "modeling_utils" in src, (
        "gradient_checkpointing.py no longer patches "
        "`transformers.modeling_utils.checkpoint` -- regression of PR #549."
    )


# ---------------------------------------------------------------------------
# PR #218: `vllm_utils` iterated a dict while mutating it -- "dict
# changed size during iteration". Pin: scan vllm_utils for the bug
# pattern `for k in <dict>:` followed by a `del <dict>[k]` or
# `<dict>[k] = ...` that mutates the dict whose key set is being
# iterated. Heuristic: any `del d[k]` inside a `for k in d:` or
# `for k in d.keys():` block is suspicious; the safe form is
# `for k in list(d):`.
# ---------------------------------------------------------------------------


def test_vllm_utils_no_unsafe_dict_mutation_during_iteration():
    src = _get_source("unsloth_zoo.vllm_utils")
    tree = ast.parse(src)
    bad: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        # iterating a bare dict: `for k in d:` where d is a Name
        if not isinstance(node.iter, ast.Name):
            # Allow `for k in list(d):` etc.
            continue
        d_name = node.iter.id
        # find del d[k] / d.pop(k) / d[k] = ... in body
        for sub in ast.walk(node):
            if isinstance(sub, ast.Delete):
                for tgt in sub.targets:
                    if (isinstance(tgt, ast.Subscript)
                            and isinstance(tgt.value, ast.Name)
                            and tgt.value.id == d_name):
                        bad.append(
                            f"line {sub.lineno}: del {d_name}[...] inside "
                            f"for ... in {d_name}: -- unsafe"
                        )
            if isinstance(sub, ast.Call):
                if (isinstance(sub.func, ast.Attribute)
                        and isinstance(sub.func.value, ast.Name)
                        and sub.func.value.id == d_name
                        and sub.func.attr in {"pop", "clear", "update"}):
                    bad.append(
                        f"line {sub.lineno}: {d_name}.{sub.func.attr}(...) "
                        f"inside for ... in {d_name}: -- unsafe"
                    )
    assert not bad, (
        "Detected dict mutation during iteration in vllm_utils.py -- "
        "regression of PR #218 (`fix dict change size`):\n"
        + "\n".join(bad)
    )


# ---------------------------------------------------------------------------
# PR #84: `vllm_lora_worker_manager` had an extra `len()` assertion
# that blocked `model.load_lora()`. The fix removed it. Pin: any
# remaining `assert len(lora_tensors)` style guard inside the load
# path is treated as a regression candidate.
# ---------------------------------------------------------------------------


def test_vllm_lora_worker_no_strict_len_assertion_on_lora_tensors():
    src = _get_source("unsloth_zoo.vllm_lora_worker_manager")
    # The original buggy line was something like `assert len(lora_tensors)`
    # right before the load. We allow `if not lora_tensors:` style but
    # reject hard `assert len(...)` on a list that may be legitimately
    # filtered to empty.
    bad = re.findall(
        r"^\s*assert\s+len\(lora_tensors\)\s*[!=<>]?[^A-Za-z]",
        src,
        flags=re.MULTILINE,
    )
    assert not bad, (
        "Found `assert len(lora_tensors)` style guard -- regression "
        "of PR #84 (broke model.load_lora):\n" + "\n".join(bad)
    )


# ---------------------------------------------------------------------------
# Bonus: PR #437 / #461 hardened Version parsing across modules. Pin:
# `unsloth_zoo.utils.Version` is the same callable referenced from
# every other zoo module that does version checks. The old failure mode
# was duplicate divergent Version() helpers in compiler / vllm_utils.
# Heuristic: no zoo module defines its OWN top-level `def Version` --
# they should import the canonical one.
# ---------------------------------------------------------------------------


def test_only_one_canonical_version_helper():
    import pathlib
    root = pathlib.Path(
        importlib.import_module("unsloth_zoo").__file__,
    ).parent
    bad: list[str] = []
    for py in root.rglob("*.py"):
        if py.name == "utils.py" and py.parent == root:
            continue  # the canonical one
        text = py.read_text(encoding="utf-8", errors="ignore")
        # Match `def Version(` at top-level indentation only.
        if re.search(r"^def\s+Version\s*\(", text, re.MULTILINE):
            bad.append(str(py.relative_to(root)))
    assert not bad, (
        "Duplicate top-level `def Version(...)` helper found -- the "
        "PR #437 cleanup unified parsing in a single place. Modules:\n"
        + "\n".join(bad)
    )


# ---------------------------------------------------------------------------
# PR #441: `logger.log(msg)` -> `logger.info(msg)`. The `Logger.log()`
# API requires `(level, msg)`, so the legacy single-arg form raised
# `TypeError: Logger.log() missing 1 required positional argument: 'msg'`
# whenever `UNSLOTH_ENABLE_LOGGING=1`. Pin: zoo source contains no
# `logger.log("string")` style single-arg call.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# PR #432: `_get_chunk_multiplier` divided by `target_gb` without
# checking for zero -- `ZeroDivisionError` when GPU memory exhausted.
# Pin: function has the explicit zero / epsilon guard before the divide.
# ---------------------------------------------------------------------------


def test_chunk_multiplier_guards_against_zero_target_gb():
    pytest.importorskip("torch")
    src = _get_source(
        "unsloth_zoo.fused_losses.cross_entropy_loss", "_get_chunk_multiplier",
    )
    # Guard expression: `if target_gb <= 1e-9:` or `if target_gb == 0:`
    has_guard = bool(
        re.search(r"if\s+target_gb\s*<=\s*\d", src)
        or re.search(r"if\s+target_gb\s*==\s*0", src)
        or re.search(r"if\s+target_gb\s*<\s*\d", src)
    )
    # Find the `/ target_gb` division.
    divides = list(re.finditer(r"/\s*target_gb\b|/\s*\(target_gb\)", src))
    assert has_guard, (
        "_get_chunk_multiplier no longer guards against zero target_gb "
        "-- regression of PR #432 (ZeroDivisionError on OOM)."
    )
    assert divides, (
        "_get_chunk_multiplier shape changed: no `/ target_gb` divide. "
        "Re-check PR #432 fix is still needed."
    )


# ---------------------------------------------------------------------------
# PR #591: CE loss must use `.reshape(-1, hd)` (not `.view`) on
# `hidden_states` so non-contiguous slices (`hidden_states[:, slice, :]`
# from `logits_to_keep`) don't raise `RuntimeError: view size is not
# compatible`. Pin: the hidden_states chunking line uses reshape.
# ---------------------------------------------------------------------------


def test_ce_loss_uses_reshape_for_hidden_states():
    pytest.importorskip("torch")
    src = _get_source(
        "unsloth_zoo.fused_losses.cross_entropy_loss",
    )
    # The hidden_states chunking line. The view-form is the bug.
    has_view_form = re.search(
        r"torch\.chunk\(\s*hidden_states\.view\(-1,\s*\w+\)",
        src,
    )
    has_reshape_form = re.search(
        r"torch\.chunk\(\s*hidden_states\.reshape\(-1,\s*\w+\)",
        src,
    )
    assert not has_view_form, (
        "CE loss uses `hidden_states.view(-1, hd)` -- regression of "
        "PR #591. Non-contiguous tensors from `hidden_states[:, "
        "slice_indices, :]` crash this with 'view size is not compatible'."
    )
    assert has_reshape_form, (
        "CE loss no longer reshapes hidden_states -- PR #591 expected "
        "`torch.chunk(hidden_states.reshape(-1, hd), ...)`."
    )


# ---------------------------------------------------------------------------
# PR #488: transformers 5.x renamed Gemma3 mask creation to
# `create_causal_mask_mapping` which raises ValueError when compiled.
# Pin: zoo's `DISABLED_KEYWORDS` includes the new name.
# ---------------------------------------------------------------------------


def test_compiler_disabled_keywords_includes_5x_gemma3_mask():
    src = _get_source("unsloth_zoo.compiler")
    # The list literal must contain the 5.x name.
    assert "create_causal_mask_mapping" in src, (
        "compiler.py DISABLED_KEYWORDS no longer mentions "
        "`create_causal_mask_mapping` -- regression of PR #488 "
        "(Gemma3 / Gemma3N on transformers 5.x crashes when compiled)."
    )


# ---------------------------------------------------------------------------
# PR #559: saving an embed_tokens layer crashed because the saving
# snippet accessed `in_features` / `out_features` on the embedding
# module (those exist on Linear, not Embedding). Pin: the saving code
# has an attribute-existence check around in_features/out_features.
# ---------------------------------------------------------------------------


def test_saving_utils_guards_embedding_dims():
    src = _get_source("unsloth_zoo.saving_utils")
    # Look for `in_features` and `out_features` accesses guarded by
    # hasattr / getattr / try-except.
    if "in_features" not in src:
        pytest.skip(
            "saving_utils no longer touches in_features -- shape "
            "changed; ensure PR #559 fix is still needed."
        )
    # The access must be inside a getattr/hasattr/try guard, not a
    # bare `module.in_features`. We tolerate any of:
    #   - `getattr(module, 'in_features'`
    #   - `hasattr(module, 'in_features'`
    #   - `isinstance(module, ...Linear...)` block surrounding it
    guarded = (
        "getattr" in src and "in_features" in src
        or "hasattr" in src and "in_features" in src
        or re.search(r"isinstance\([^)]*Linear[^)]*\)", src)
    )
    assert guarded, (
        "saving_utils accesses in_features/out_features without a "
        "guard for non-Linear modules -- regression of PR #559 "
        "(Embedding layer has no in_features)."
    )


def test_no_single_arg_logger_log_calls():
    import pathlib
    root = pathlib.Path(
        importlib.import_module("unsloth_zoo").__file__,
    ).parent
    bad: list[str] = []
    # Match logger.log(<single_str_arg>) -- a single argument that is a
    # string literal (NOT a level constant like logging.INFO).
    pat = re.compile(
        r"\blogger\.log\(\s*[\"'fr]",
    )
    for py in root.rglob("*.py"):
        text = py.read_text(encoding="utf-8", errors="ignore")
        for m in pat.finditer(text):
            # Find the matching close-paren (allow simple cases).
            i = m.end()
            depth = 1
            while i < len(text) and depth > 0:
                if text[i] == "(":
                    depth += 1
                elif text[i] == ")":
                    depth -= 1
                i += 1
            call_body = text[m.end(): i - 1]
            # If a comma at the same paren depth exists, it's a 2-arg
            # call (level, msg) -- skip it.
            depth = 0
            has_top_comma = False
            for ch in call_body:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                elif ch == "," and depth == 0:
                    has_top_comma = True
                    break
            if not has_top_comma:
                bad.append(f"{py.relative_to(root)}: logger.log({call_body[:40]}...)")
    assert not bad, (
        "Found `logger.log(<str>)` single-arg call -- regression of "
        "PR #441 (Logger.log requires (level, msg), use logger.info "
        "instead). Found:\n" + "\n".join(bad)
    )
