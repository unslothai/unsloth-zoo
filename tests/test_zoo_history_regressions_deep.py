# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Regressions mined from merged-PR history of unslothai/unsloth-zoo."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import pathlib
import re
import textwrap

import pytest


def _module_source_path(module_name: str) -> pathlib.Path:
    # find_spec is metadata-only so we avoid running compiler.py:87's
    # torch.cuda.get_device_capability() on CPU-only CI.
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin in (None, "built-in"):
        raise ImportError(f"could not locate source for {module_name!r}")
    return pathlib.Path(spec.origin)


def _get_source(module_name: str, attr: str | None = None) -> str:
    if attr is None:
        return _module_source_path(module_name).read_text(encoding="utf-8")
    mod = importlib.import_module(module_name)
    obj = getattr(mod, attr)
    return inspect.getsource(obj)


def test_longest_common_sublist_handles_singleton_overlap():
    """PR #4: _longest_common_sublist works on lists (not str(list))."""
    from unsloth_zoo.dataset_utils import _longest_common_sublist
    out = _longest_common_sublist([[1, 2, 3, 0], [4, 5, 6, 0]])
    assert out == [0], (
        "_longest_common_sublist should find the single shared element."
        f" got {out!r}. Regression: PR #4 (LCS over int lists, not str repr)."
    )


def test_longest_common_sublist_empty_and_no_overlap():
    from unsloth_zoo.dataset_utils import _longest_common_sublist
    assert _longest_common_sublist([]) == []
    assert _longest_common_sublist([[1, 2], []]) == []
    assert _longest_common_sublist([[1, 2], [3, 4]]) == []


def test_no_unguarded_legacy_pretrained_config_import():
    """PR #322: transformers 4.57 renamed PretrainedConfig -> PreTrainedConfig.
    Reject unguarded legacy imports."""
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
            indent = m.group(1)
            if indent and len(indent) >= 4:
                # Indented: inside try/except guard.
                continue
            if "PreTrainedConfig" in text:
                continue
            bad.append(f"{py.relative_to(root)}: {line.strip()}")
    assert not bad, (
        "Found unguarded legacy PretrainedConfig imports -- regression "
        "of PR #322 (transformers 4.57 rename to PreTrainedConfig):\n"
        + "\n".join(bad)
    )


def test_empty_model_exception_var_consistent():
    """PR #374: bound exception variable matches body references
    (UnboundLocalError when `as error` body uses `e`)."""
    src = _get_source("unsloth_zoo.empty_model")
    tree = ast.parse(src)

    suspicious: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if node.name is None:
            continue
        bound = node.name
        names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        if bound != "e" and "e" in names:
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


def test_utils_distributed_import_present():
    """PR #422: utils must bind `dist` to torch.distributed."""
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
    """PR #422: distributed_function tolerates uninitialised process group."""
    from unsloth_zoo.utils import distributed_function
    out = distributed_function(n=1, function=lambda: 42)
    assert out == 42, (
        f"distributed_function returned {out!r}; expected 42 -- "
        "regression of PR #422 / the init_process_group guard."
    )


def test_version_garbage_input_clean_error():
    """PR #425: Version() raises clean RuntimeError on garbage input,
    not NameError/UnboundLocalError on undefined `e`."""
    from unsloth_zoo.utils import Version
    bad_inputs = [
        "not.a.real.package.name",
        "alpha.beta.gamma",
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
            from packaging.version import Version as TrueVersion
            assert isinstance(v, TrueVersion)


def test_version_accepts_package_name_string():
    """PR #461: Version(<pkg-name>) resolves via importlib.metadata."""
    from unsloth_zoo.utils import Version
    v = Version("packaging")
    assert v >= Version("0.0.1"), (
        "Version('packaging') did not yield a numeric Version "
        "(regression of PR #461 -- string lookup via importlib.metadata)."
    )


def test_version_falls_back_for_unknown_package_strings():
    """PR #461: raw version strings still parse as versions."""
    from unsloth_zoo.utils import Version
    assert Version("1.2.3") == Version("1.2.3")
    assert Version("1.2.3") < Version("2.0.0")


def test_canonicalize_annotation_union_pep604_equivalence():
    """PR #458: Union[int, str] and int|str canonicalise identically
    (origin must flow through TYPE_MAPPINGS)."""
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


def test_patching_utils_should_convert_module_uses_substring():
    """PR #491: _unsloth_should_convert_module uses substring component
    matching (transformers 5.x prefix-anchored re.match misses vision_tower)."""
    src = _get_source("unsloth_zoo.patching_utils")
    assert "_unsloth_should_convert_module" in src, (
        "The transformers-5.x should_convert_module patch is missing "
        "-- regression of PR #491."
    )
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


def test_compiler_disables_fullgraph_for_dynamic_rope_update():
    """PR #533: compiler flips fullgraph=False when dynamic_rope_update
    decorator is present (Phi-4 longrope data-dependent branching)."""
    src = _get_source("unsloth_zoo.compiler")
    assert "dynamic_rope_update" in src, (
        "compiler.py no longer references dynamic_rope_update -- "
        "regression of PR #533."
    )
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


def test_compiler_conv_prologue_casts_to_weight_dtype():
    """PR #552: Conv wrappers cast input to weight dtype BEFORE the op
    (autocast bf16 weight + fp16 input crashes)."""
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


def test_compiler_lora_forward_emits_torch_result_dtype_cast():
    """PR #564: LoRA forward early-return casts back to base-layer dtype
    when autocast is disabled."""
    src = _get_source("unsloth_zoo.compiler")
    assert "torch_result_dtype" in src, (
        "compiler.py no longer references `torch_result_dtype` -- "
        "regression of PR #564 (autocast-disabled dtype mismatch on "
        "PEFT LoRA forward). The early-return must cast back to the "
        "base-layer dtype."
    )
    assert re.search(
        r"return\s+lora_forward\([^)]+\)\.to\(",
        src,
    ), (
        "compiler.py no longer emits `return lora_forward(...).to(...)` "
        "-- the dtype-cast on the LoRA early return is gone (PR #564)."
    )


def test_compiler_peft_forward_skips_quantized_dtype_cast():
    """PR #482: skip x.to(weight.dtype) on Params4bit (weight.dtype==uint8
    corrupts inputs); gated by hasattr(weight, 'quant_state')."""
    src = _get_source("unsloth_zoo.compiler")
    assert "quant_state" in src, (
        "compiler.py has no `quant_state` mention -- regression of "
        "PR #482 (4-bit input corrupted by float16 -> uint8 cast)."
    )
    guard = re.search(
        r"not\s+hasattr\(self\.base_layer\.weight,\s*['\"]quant_state['\"]\)",
        src,
    )
    assert guard, (
        "Quant-state guard not found in the LoRA dtype-cast prologue. "
        "PR #482: cast must be skipped on Params4bit / Linear4bit."
    )


def test_vllm_lora_manager_no_duplicate_vllm_config_kwarg():
    """PR #466: _call_create_lora_manager must not pass vllm_config
    both positionally and as keyword."""
    src = _get_source("unsloth_zoo.vllm_lora_worker_manager")
    bad: list[str] = []
    for m in re.finditer(
        r"_call_create_lora_manager\((?P<args>.*?)\)",
        src,
        flags=re.DOTALL,
    ):
        body = m.group("args")
        if re.search(r"vllm_config\s*=\s*vllm_config", body):
            bad.append(body.strip())
    assert not bad, (
        "Duplicate `vllm_config=vllm_config` kwarg passed alongside "
        "positional -- regression of PR #466:\n"
        + "\n---\n".join(bad)
    )


def test_gemma4_proxy_hides_zero_num_kv_shared_layers():
    """PR #580: _Gemma4KVSharedSafeProxy hides num_kv_shared_layers==0
    so transformers' layer_types[:-0] doesn't return []."""
    pytest.importorskip("torch")
    mod = importlib.import_module(
        "unsloth_zoo.temporary_patches.gemma4",
    )
    Proxy = getattr(mod, "_Gemma4KVSharedSafeProxy", None)
    assert Proxy is not None, (
        "_Gemma4KVSharedSafeProxy is missing -- regression of PR #580."
    )

    class _Real:
        num_kv_shared_layers = 0
        num_hidden_layers = 4

        def __iter__(self):
            return iter(["num_hidden_layers"])

    proxy = Proxy(_Real())
    assert not hasattr(proxy, "num_kv_shared_layers"), (
        "Proxy still exposes num_kv_shared_layers == 0 -- PR #580 "
        "regression. transformers will do layer_types[:-0] -> []."
    )
    assert proxy.num_hidden_layers == 4
    assert "num_kv_shared_layers" not in proxy


def test_grpo_softcap_formula_is_cap_times_tanh():
    """PR #593: softcap is `cap * tanh(logits/cap)`, not
    `logits * tanh(logits/cap)` (else saturates to no-op)."""
    src = _get_source(
        "unsloth_zoo.rl_replacements",
        "chunked_hidden_states_selective_log_softmax",
    )
    correct = re.search(
        r"=\s*logit_softcapping\s*\*\s*torch\.tanh\([^)]+/\s*logit_softcapping\)",
        src,
    )
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


def test_rl_replacements_scalar_tensor_init_for_accumulators():
    """PR #543: accumulated_loss = torch.zeros(1, ...)[0] (shape [])
    so transformers 5.x in-place += doesn't broadcast-crash."""
    src = _get_source("unsloth_zoo.rl_replacements")
    hit = re.search(
        r"accumulated_loss\s*=\s*torch\.zeros\(1[^)]*\)\[0\]",
        src,
    )
    assert hit, (
        "accumulated_loss is no longer initialised as a SCALAR tensor "
        "via `torch.zeros(1, ...)[0]` -- regression of PR #543 "
        "(transformers 5.x in-place += on shape-[] target)."
    )


def test_sft_prepare_dataset_removes_original_columns_in_non_packing_path():
    """PR #477: .map(_tokenize, ...) passes remove_columns= so raw
    JSON columns don't leak to the collator."""
    src = _get_source("unsloth_zoo.dataset_utils")
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


def test_saving_utils_uses_atomic_replace_for_shard_rewrite():
    """PR #595: the resized-shard rewrite uses atomic os.replace
    (not remove+move; Windows WinError 1224 on shard rewrite). The rewrite +
    replace now lives in _stream_rewrite_resized_shard_and_replace."""
    src = _get_source(
        "unsloth_zoo.saving_utils", "_stream_rewrite_resized_shard_and_replace",
    )
    assert "os.replace(" in src, (
        "_stream_rewrite_resized_shard_and_replace no longer uses os.replace -- "
        "regression of PR #595 (Windows WinError 1224 on shard rewrite)."
    )
    if "shutil.move(" in src and "os.remove(" in src:
        idx_replace = src.find("os.replace(")
        idx_remove  = src.find("os.remove(")
        idx_move    = src.find("shutil.move(")
        if idx_remove >= 0 and idx_move >= 0 and idx_remove < idx_replace and idx_move < idx_replace:
            pytest.fail(
                "Non-atomic os.remove + shutil.move pair survives "
                "before the os.replace path -- the data-loss window "
                "the PR #595 fix was supposed to close."
            )


def test_saving_utils_uses_device_type_helpers():
    """PR #615: GGUF merge path uses DEVICE_TYPE helpers (XPU + ROCm
    have no torch.cuda namespace)."""
    src = _get_source("unsloth_zoo.saving_utils")
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


def test_unsloth_get_batch_samples_accepts_4_args():
    """PR #91: _unsloth_get_batch_samples accepts the 4th `device`
    parameter transformers 4.50 added."""
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


def test_all_modules_all_entries_have_no_concatenated_names():
    """PR #617: __all__ entries with snake_case+CamelCase boundary
    (e.g. `raise_errorUnpack`) are the missing-comma bug fingerprint."""
    import pathlib
    root = pathlib.Path(
        importlib.import_module("unsloth_zoo").__file__,
    ).parent
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


def test_gemma4_moe_guards_logits_to_keep_slice():
    """PR #612: gemma4_moe guards the slice behind `if logits_to_keep != 0`
    (Python's slice(-0, None) == slice(0, None) trap)."""
    try:
        src = _get_source("unsloth_zoo.temporary_patches.gemma4_moe")
    except Exception:
        pytest.skip("gemma4_moe module unavailable")
    assert re.search(
        r"if\s+logits_to_keep\s*!=\s*0", src,
    ), (
        "gemma4_moe.py no longer guards the hidden-state slice behind "
        "`if logits_to_keep != 0:` -- regression of PR #612 (the "
        "implicit dependency on Python's slice(-0, None) == slice(0, None))."
    )


def test_smart_gradient_checkpointing_patches_transformers_modeling_utils():
    """PR #549: gradient_checkpointing patches transformers.modeling_utils
    (transformers 5.2+ moved the checkpoint reference)."""
    pytest.importorskip("transformers")
    src = _get_source("unsloth_zoo.gradient_checkpointing")
    assert "transformers.modeling_utils" in src or "modeling_utils" in src, (
        "gradient_checkpointing.py no longer patches "
        "`transformers.modeling_utils.checkpoint` -- regression of PR #549."
    )


def test_vllm_utils_no_unsafe_dict_mutation_during_iteration():
    """PR #218: no `del d[k]` / `d.pop(k)` inside `for k in d:` loops
    in vllm_utils (dict-changed-size-during-iteration)."""
    src = _get_source("unsloth_zoo.vllm_utils")
    tree = ast.parse(src)
    bad: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        if not isinstance(node.iter, ast.Name):
            continue
        d_name = node.iter.id
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


def test_vllm_lora_worker_no_strict_len_assertion_on_lora_tensors():
    """PR #84: no `assert len(lora_tensors)` guard (blocked load_lora
    when filtering yields legitimately empty list)."""
    src = _get_source("unsloth_zoo.vllm_lora_worker_manager")
    bad = re.findall(
        r"^\s*assert\s+len\(lora_tensors\)\s*[!=<>]?[^A-Za-z]",
        src,
        flags=re.MULTILINE,
    )
    assert not bad, (
        "Found `assert len(lora_tensors)` style guard -- regression "
        "of PR #84 (broke model.load_lora):\n" + "\n".join(bad)
    )


def test_only_one_canonical_version_helper():
    """PR #437 / #461: only one top-level `def Version(...)` across zoo
    (no duplicate divergent helpers)."""
    import pathlib
    root = pathlib.Path(
        importlib.import_module("unsloth_zoo").__file__,
    ).parent
    bad: list[str] = []
    for py in root.rglob("*.py"):
        if py.name == "utils.py" and py.parent == root:
            continue  # the canonical one
        text = py.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"^def\s+Version\s*\(", text, re.MULTILINE):
            bad.append(str(py.relative_to(root)))
    assert not bad, (
        "Duplicate top-level `def Version(...)` helper found -- the "
        "PR #437 cleanup unified parsing in a single place. Modules:\n"
        + "\n".join(bad)
    )


def test_chunk_multiplier_guards_against_zero_target_gb():
    """PR #432: _get_chunk_multiplier guards against zero target_gb
    (ZeroDivisionError on GPU OOM)."""
    pytest.importorskip("torch")
    src = _get_source(
        "unsloth_zoo.fused_losses.cross_entropy_loss", "_get_chunk_multiplier",
    )
    has_guard = bool(
        re.search(r"if\s+target_gb\s*<=\s*\d", src)
        or re.search(r"if\s+target_gb\s*==\s*0", src)
        or re.search(r"if\s+target_gb\s*<\s*\d", src)
    )
    divides = list(re.finditer(r"/\s*target_gb\b|/\s*\(target_gb\)", src))
    assert has_guard, (
        "_get_chunk_multiplier no longer guards against zero target_gb "
        "-- regression of PR #432 (ZeroDivisionError on OOM)."
    )
    assert divides, (
        "_get_chunk_multiplier shape changed: no `/ target_gb` divide. "
        "Re-check PR #432 fix is still needed."
    )


def test_ce_loss_uses_reshape_for_hidden_states():
    """PR #591: CE loss uses .reshape (not .view) on hidden_states slices
    (non-contiguous from logits_to_keep slicing)."""
    pytest.importorskip("torch")
    src = _get_source(
        "unsloth_zoo.fused_losses.cross_entropy_loss",
    )
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


def test_compiler_disabled_keywords_includes_5x_gemma3_mask():
    """PR #488: DISABLED_KEYWORDS includes create_causal_mask_mapping
    (Gemma3/3N transformers 5.x compile crash)."""
    src = _get_source("unsloth_zoo.compiler")
    assert "create_causal_mask_mapping" in src, (
        "compiler.py DISABLED_KEYWORDS no longer mentions "
        "`create_causal_mask_mapping` -- regression of PR #488 "
        "(Gemma3 / Gemma3N on transformers 5.x crashes when compiled)."
    )


def test_saving_utils_guards_embedding_dims():
    """PR #559: saving guards in_features/out_features access for
    Embedding modules (which lack those attrs)."""
    src = _get_source("unsloth_zoo.saving_utils")
    if "in_features" not in src:
        pytest.skip(
            "saving_utils no longer touches in_features -- shape "
            "changed; ensure PR #559 fix is still needed."
        )
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
    """PR #441: no `logger.log(<str>)` single-arg calls (Logger.log
    requires (level, msg); UNSLOTH_ENABLE_LOGGING=1 would TypeError)."""
    import pathlib
    root = pathlib.Path(
        importlib.import_module("unsloth_zoo").__file__,
    ).parent
    bad: list[str] = []
    pat = re.compile(
        r"\blogger\.log\(\s*[\"'fr]",
    )
    for py in root.rglob("*.py"):
        text = py.read_text(encoding="utf-8", errors="ignore")
        for m in pat.finditer(text):
            i = m.end()
            depth = 1
            while i < len(text) and depth > 0:
                if text[i] == "(":
                    depth += 1
                elif text[i] == ")":
                    depth -= 1
                i += 1
            call_body = text[m.end(): i - 1]
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
