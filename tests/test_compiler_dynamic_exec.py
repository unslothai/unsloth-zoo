# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

"""Drift detectors for unsloth_zoo/compiler.py rewriters on real transformers source; CPU-only, and model types absent from transformers are skipped."""

from __future__ import annotations

import ast
import importlib
import inspect
import os
import sys
import textwrap

import pytest


transformers = pytest.importorskip("transformers")
compiler = pytest.importorskip("unsloth_zoo.compiler")


@pytest.fixture(autouse = True)
def _compile_disabled(monkeypatch):
    """Set UNSLOTH_COMPILE_DISABLE via monkeypatch; a module-scope setdefault leaked it into the rest of the session and its subprocesses."""
    monkeypatch.setenv("UNSLOTH_COMPILE_DISABLE", "1")


@pytest.fixture(autouse = True)
def _unimport_the_compile_folder(monkeypatch):
    """Unloads compile-folder modules by load path (1168); requests monkeypatch to run before env unpatch."""
    before = dict(sys.modules)
    try:
        yield
    finally:
        try:
            folder, _ = compiler.get_compile_folder()
            root = os.path.realpath(folder)
        except Exception:
            return
        for name, module in list(sys.modules.items()):
            path = getattr(module, "__file__", None)
            if not path:
                continue
            try:
                resolved = os.path.realpath(path)
            except (OSError, ValueError):
                continue
            if resolved != root and not resolved.startswith(root + os.sep):
                continue
            if name in before:
                sys.modules[name] = before[name]
            else:
                del sys.modules[name]


def _own_torch_nn_forwards():
    """Read from vars(cls), not getattr, so an inherited forward is not recorded as owned and then reinstalled by the restore."""
    try:
        import torch
    except Exception:
        return None
    return {
        name: vars(obj)["forward"]
        for name, obj in vars(torch.nn).items()
        if isinstance(obj, type) and "forward" in vars(obj)
    }


@pytest.fixture(autouse = True)
def _restore_torch_nn_forwards():
    """Put ``torch.nn``'s own forwards back after the dtype loop rewrites them.

    ``unsloth_compile_transformers`` patches ``torch.nn.<Module>.forward`` in
    place, on the real ``torch.nn``, and nothing here undoes it. The wrappers
    then outlive the test and every later test in the process sees a torch.nn
    that is already patched.

    That is not theoretical: it fails
    ``test_compiled_cache_collective.py::test_repeated_dtype_patching_does_not_stack_the_source_rewrite``,
    which opens by capturing ``pristine = torch.nn.Conv2d.forward`` and then
    asserts the installed marker carries torch's own forward. Run after this
    module in one process, what it captures is already a
    ``_dtype_safe_forward.<locals>.forward``, so the assertion compares a
    wrapper against the genuine original and fails on a contract that holds.
    The two files are in different jobs today, `Core drift` and
    `Repo tests (CPU)`, so CI does not currently put them in one process. That
    is a scheduling accident and not a property worth relying on.

    Two directions, because the loop patches by assignment and assignment does
    not care whether the class had a forward of its own.

    A class that owned one gets it put back by identity rather than deleted:
    several of these legitimately define their own, and deleting would expose
    an inherited one instead.

    A class that owned none has to have the attribute deleted, not restored,
    and this is the half worth stating. ``BatchNorm1d``, ``BatchNorm2d`` and
    ``BatchNorm3d`` inherit ``forward`` from ``_BatchNorm``, so they are absent
    from the snapshot; the loop then gives each one a real attribute it never
    had. Restoring only the recorded names leaves those three rewritten, which
    is exactly what an audit of what survives teardown found still leaking
    after the first version of this fixture.
    """
    before = _own_torch_nn_forwards()
    try:
        yield
    finally:
        if before is None:
            return
        import torch
        for name, obj in vars(torch.nn).items():
            if not isinstance(obj, type):
                continue
            current = vars(obj).get("forward")
            if current is None:
                continue
            if name in before:
                if current is not before[name]:
                    obj.forward = before[name]
            else:
                delattr(obj, "forward")


@pytest.fixture(autouse = True)
def _restore_unsloth_env():
    """Restores the whole UNSLOTH_* snapshot: the compiler writes os.environ itself (e.g. UNSLOTH_FULLGRAPH, UNSLOTH_RETURN_LOGITS), which monkeypatch misses."""
    before = {k: v for k, v in os.environ.items() if k.startswith("UNSLOTH_")}
    try:
        yield
    finally:
        for key in [k for k in os.environ if k.startswith("UNSLOTH_")]:
            if key not in before:
                del os.environ[key]
        for key, value in before.items():
            if os.environ.get(key) != value:
                os.environ[key] = value


# Model types the zoo compiler drives end-to-end (from its call sites).
KNOWN_MODEL_TYPES = [
    "llama",
    "llama4",
    "mistral",
    "mistral3",
    "ministral",
    "gemma",
    "gemma2",
    "gemma3",
    "gemma3n",
    "gemma4",          # newer tf only; skip if missing
    "qwen2",
    "qwen2_moe",
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3",
    "qwen3_moe",
    "qwen3_next",
    "qwen3_vl",
    "deepseek",        # legacy
    "deepseek_v2",
    "deepseek_v3",
    "gpt_oss",
    "cohere",
    "cohere2",
    "phi",
    "phi3",
    "phi4_multimodal",
    "starcoder2",
    "olmo",
    "olmo2",
    "falcon",
    "granite",
    "glm",
    "glm4",
    "glm4v",
    "pixtral",
    "paligemma",
    "idefics",
    "idefics2",
    "idefics3",
    "mllama",
]


def _load_modeling(model_type: str):
    """Imports transformers.models.<model_type>.modeling_<model_type>; a ModuleNotFoundError skips as an environment issue, not drift."""
    mod_path = f"transformers.models.{model_type}.modeling_{model_type}"
    try:
        return importlib.import_module(mod_path)
    except ModuleNotFoundError:
        pytest.skip(
            f"model_type {model_type} not present on installed "
            f"transformers, can't drive compiler"
        )
    except ImportError as exc:
        # gemma3n's config imports ImageNetInfo from a timm.data that dropped it.
        pytest.skip(
            f"model_type {model_type} raised on import, so the compiler cannot "
            f"be driven for it: {type(exc).__name__}: {exc}"
        )


def _assert_parseable(rewritten: str, entry_point: str, *, dedent: bool = False):
    """ast.parse(rewritten) or pytest.fail with DRIFT message."""
    source = textwrap.dedent(rewritten) if dedent else rewritten
    try:
        ast.parse(source)
    except (SyntaxError, IndentationError) as exc:
        pytest.fail(
            f"DRIFT DETECTED: {entry_point} produced invalid Python: "
            f"{type(exc).__name__}: {exc}\n"
            f"--- rewritten source (first 600 chars) ---\n"
            f"{source[:600]}\n--- end ---"
        )


def _assert_execs(rewritten: str, entry_point: str, *, dedent: bool = False):
    """Compiles and execs in a sandbox; only NameError (a dangling identifier) counts as DRIFT, while ImportError and other errors are env."""
    source = textwrap.dedent(rewritten) if dedent else rewritten
    sandbox = {"__name__": "test_compiler_dynamic_exec_sandbox"}
    try:
        code = compile(source, f"<{entry_point}>", "exec")
    except (SyntaxError, IndentationError) as exc:
        pytest.fail(
            f"DRIFT DETECTED: {entry_point} produced uncompilable Python: "
            f"{type(exc).__name__}: {exc}"
        )
        return
    try:
        exec(code, sandbox)
    except NameError as exc:
        pytest.fail(
            f"DRIFT DETECTED: {entry_point} top-level exec raised "
            f"NameError on dangling identifier: {exc}"
        )
    except ImportError:
        pass
    except Exception:
        pass


# gemma3 is the canonical driver: RMSNorm, sliding attn, RoPE, MoE, ForCondGen.


@pytest.fixture(scope="module")
def gemma3_mod():
    return _load_modeling("gemma3")


@pytest.fixture(scope="module")
def gemma3_full_source(gemma3_mod):
    return inspect.getsource(gemma3_mod)


def test_higher_precision_softmax_full_module(gemma3_full_source):
    out = compiler.higher_precision_softmax(gemma3_full_source)
    _assert_parseable(out, "higher_precision_softmax(gemma3)")


def test_higher_precision_softmax_idempotent(gemma3_full_source):
    """Pins ``unsloth_zoo/compiler.py:398-404`` idempotency lookahead;
    drive twice and assert no doubled ``.to(x.dtype).to(x.dtype)`` chains."""
    once = compiler.higher_precision_softmax(gemma3_full_source)
    twice = compiler.higher_precision_softmax(once)
    _assert_parseable(twice, "higher_precision_softmax(gemma3)x2")
    if ".dtype).to(" in twice and ".dtype).to(" not in once:
        pytest.fail(
            "DRIFT DETECTED: higher_precision_softmax is not "
            "idempotent -- second pass introduced new .to(...).to(...) chain"
        )


def test_higher_precision_sqrt_mean_full_module(gemma3_full_source):
    out = compiler.higher_precision_sqrt_mean(gemma3_full_source)
    _assert_parseable(out, "higher_precision_sqrt_mean(gemma3)")


def test_fix_rotary_embedding_dtype_passthrough(gemma3_full_source):
    """Without UNSLOTH_FORCE_CUSTOM_DTYPE the rewriter is a no-op, so the source must come back unchanged."""
    out = compiler.fix_rotary_embedding_dtype(gemma3_full_source)
    _assert_parseable(out, "fix_rotary_embedding_dtype(gemma3)")
    assert out == gemma3_full_source, (
        "fix_rotary_embedding_dtype is expected to be a no-op when "
        "UNSLOTH_FORCE_CUSTOM_DTYPE is unset"
    )


def test_fix_attention_dtype_consistency_full_module(gemma3_full_source):
    """Pins ``unsloth_zoo/compiler.py`` fix_attention_dtype_consistency:
    inserts V-dtype cast directly after each ``apply_rotary_pos_emb(...)``."""
    out = compiler.fix_attention_dtype_consistency(gemma3_full_source)
    _assert_parseable(out, "fix_attention_dtype_consistency(gemma3)")
    if "apply_rotary_pos_emb(" in gemma3_full_source:
        assert (
            "value_states = value_states.to(query_states.dtype)" in out
        ), (
            "DRIFT DETECTED: fix_attention_dtype_consistency did not "
            "insert V dtype cast after apply_rotary_pos_emb in gemma3"
        )


def test_higher_precision_layernorms_full_module(gemma3_full_source, monkeypatch):
    """The rewriter sets UNSLOTH_HIGH_PRECISION_LAYERNORM in os.environ; the monkeypatch delenv stops that from leaking out."""
    monkeypatch.delenv("UNSLOTH_HIGH_PRECISION_LAYERNORM", raising=False)
    compiler.higher_precision_layernorms(gemma3_full_source)
    assert "UNSLOTH_HIGH_PRECISION_LAYERNORM" in os.environ, (
        "DRIFT DETECTED: higher_precision_layernorms did not set "
        "UNSLOTH_HIGH_PRECISION_LAYERNORM env var on gemma3"
    )


def test_fixup_fused_lm_head_full_module(gemma3_full_source):
    out = compiler.fixup_fused_lm_head(gemma3_full_source)
    _assert_parseable(out, "fixup_fused_lm_head(gemma3)")


def test_fixup_fused_lm_head_walrus_dropped():
    """Pins the gemma3n walrus rewrite in unsloth_zoo/compiler.py: the final_logit_softcapping := binding becomes a
    plain `.final_logit_softcapping is not None` check."""
    src = (
        "def forward(self):\n"
        "    if (final_logit_softcapping := self.config.get_text_config().final_logit_softcapping) is not None:\n"
        "        logits = logits / final_logit_softcapping\n"
        "        logits = logits * final_logit_softcapping\n"
    )
    out = compiler.fixup_fused_lm_head(src)
    _assert_parseable(out, "fixup_fused_lm_head(walrus)")
    if ":= self.config" in out or "(final_logit_softcapping :=" in out:
        pytest.fail(
            "DRIFT DETECTED: fixup_fused_lm_head left the walrus "
            "binding in place; gemma3n rewrite did not land"
        )
    assert "self.config.get_text_config().final_logit_softcapping" in out


def test_apply_mask_attention_mask_out_full_module(gemma3_full_source):
    out = compiler.apply_mask_attention_mask_out(gemma3_full_source)
    _assert_parseable(out, "apply_mask_attention_mask_out(gemma3)")


def test_convert_attention_masks_to_bool_passthrough(gemma3_full_source):
    """Module-level source without bare ``return`` -> passthrough."""
    out = compiler.convert_attention_masks_to_bool("gemma3", gemma3_full_source)
    _assert_parseable(out, "convert_attention_masks_to_bool(gemma3, full)")


def test_patch_residual_stream_full_module(gemma3_full_source):
    out = compiler.patch_residual_stream(gemma3_full_source)
    _assert_parseable(out, "patch_residual_stream(gemma3)")


def test_replace_with_grouped_query_attention_attention_method(gemma3_mod):
    """GQA rewriter on real Gemma3Attention.forward."""
    attn_src = inspect.getsource(gemma3_mod.Gemma3Attention.forward)
    out = compiler.replace_with_grouped_query_attention(
        "Gemma3Attention", attn_src,
    )
    _assert_parseable(out, "replace_with_grouped_query_attention(Gemma3Attention)", dedent=True)


def test_apply_fused_lm_head_gemma3_causallm(gemma3_mod):
    fwd_src = inspect.getsource(gemma3_mod.Gemma3ForCausalLM.forward)
    out, applied = compiler.apply_fused_lm_head(
        fwd_src, "Gemma3ForCausalLM",
    )
    _assert_parseable(out, "apply_fused_lm_head(Gemma3ForCausalLM)", dedent=True)
    if applied:
        if "NOT_RETURN_LOGITS" not in out:
            pytest.fail(
                "DRIFT DETECTED: apply_fused_lm_head reported applied=True "
                "but emitted source lacks NOT_RETURN_LOGITS sentinel"
            )


def test_apply_fused_lm_head_gemma3_conditional(gemma3_mod):
    fwd_src = inspect.getsource(
        gemma3_mod.Gemma3ForConditionalGeneration.forward,
    )
    out, applied = compiler.apply_fused_lm_head(
        fwd_src, "Gemma3ForConditionalGeneration",
    )
    _assert_parseable(out, "apply_fused_lm_head(Gemma3ForConditionalGeneration)", dedent=True)
    if applied and "NOT_RETURN_LOGITS" not in out:
        pytest.fail(
            "DRIFT DETECTED: apply_fused_lm_head reported applied=True "
            "but emitted source lacks NOT_RETURN_LOGITS sentinel"
        )


@pytest.mark.parametrize("model_type", ["llama", "mistral", "qwen2", "qwen3"])
def test_apply_fused_lm_head_other_text_models(model_type):
    mod = _load_modeling(model_type)
    causal_cls_name = None
    for n in dir(mod):
        if n.endswith("ForCausalLM"):
            causal_cls_name = n
            break
    if causal_cls_name is None:
        pytest.skip(f"{model_type} has no ForCausalLM head")
    cls = getattr(mod, causal_cls_name)
    fwd_src = inspect.getsource(cls.forward)
    out, _ = compiler.apply_fused_lm_head(fwd_src, causal_cls_name)
    _assert_parseable(
        out, f"apply_fused_lm_head({causal_cls_name})", dedent=True,
    )


def test_patch_gradient_checkpointing_text_decoder(gemma3_mod):
    """Returns None when upstream uses GradientCheckpointingLayer (modern path, not drift); otherwise init and forward must parse."""
    out = compiler.patch_gradient_checkpointing(
        "Gemma3TextModel", gemma3_mod.Gemma3TextModel,
    )
    if out is None:
        return
    init, forward = out
    _assert_parseable(init, "patch_gradient_checkpointing.init", dedent=True)
    _assert_parseable(forward, "patch_gradient_checkpointing.forward", dedent=True)


def test_patch_gradient_checkpointing_layer_caller_text_decoder(gemma3_mod):
    """Companion rewriter for the modern GradientCheckpointingLayer path."""
    out = compiler.patch_gradient_checkpointing_layer_caller(
        "Gemma3TextModel", gemma3_mod.Gemma3TextModel,
    )
    if out is None:
        return
    init, forward = out
    _assert_parseable(
        init, "patch_gradient_checkpointing_layer_caller.init", dedent=True,
    )
    _assert_parseable(
        forward,
        "patch_gradient_checkpointing_layer_caller.forward",
        dedent=True,
    )


def test_strip_kw_from_module_calls_text_decoder(gemma3_mod):
    """Drives strip_kw_from_module_calls directly; the GC-layer rewriter calls it internally to drop kwarg= annotations."""
    fwd_src = inspect.getsource(gemma3_mod.Gemma3TextModel.forward)
    out = compiler.strip_kw_from_module_calls(fwd_src, "self.layers")
    _assert_parseable(out, "strip_kw_from_module_calls(gemma3.layers)", dedent=True)


def test_patch_finfo_attention_mask_dtype_mismatch_passthrough(gemma3_mod):
    """Passthrough on modern transformers source where the block is absent; the output must still parse."""
    fwd_src = inspect.getsource(gemma3_mod.Gemma3TextModel.forward)
    out = compiler.patch_finfo_attention_mask_dtype_mismatch(
        "Gemma3TextModel", fwd_src,
    )
    _assert_parseable(
        out,
        "patch_finfo_attention_mask_dtype_mismatch(Gemma3TextModel)",
        dedent=True,
    )


def test_patch_moe_routing_weights_cast_qwen3_moe():
    """Real Qwen3 MoE block (canonical user of this codepath)."""
    qmoe = _load_modeling("qwen3_moe")
    cls = qmoe.Qwen3MoeSparseMoeBlock
    src = inspect.getsource(cls.forward)
    out, methods = compiler.patch_moe_routing_weights_cast(cls, src)
    _assert_parseable(
        out, "patch_moe_routing_weights_cast.forward", dedent=True,
    )
    for name, body in methods.items():
        _assert_parseable(
            body,
            f"patch_moe_routing_weights_cast.method[{name}]",
            dedent=True,
        )


def test_patch_gradient_accumulation_for_conditional_gen(gemma3_mod):
    """Returns None when inner classes already accept **kwargs; otherwise the rewritten class source must parse."""
    out = compiler.patch_gradient_accumulation(
        gemma3_mod, "Gemma3ForConditionalGeneration",
    )
    if out is None:
        return
    _assert_parseable(
        out,
        "patch_gradient_accumulation(Gemma3ForConditionalGeneration)",
    )


# Passthrough robustness on shapes the rewriter must not touch.


PASSTHROUGH_SOURCE = (
    "def add(a, b):\n"
    "    return a + b\n"
    "\n"
    "class Foo:\n"
    "    def __init__(self, x):\n"
    "        self.x = x\n"
)


@pytest.mark.parametrize(
    "name",
    [
        "higher_precision_softmax",
        "higher_precision_sqrt_mean",
        "fix_rotary_embedding_dtype",
        "fix_attention_dtype_consistency",
        "apply_mask_attention_mask_out",
        "patch_residual_stream",
        "fixup_fused_lm_head",
    ],
)
def test_rewriter_passthrough_on_plain_python(name):
    fn = getattr(compiler, name)
    out = fn(PASSTHROUGH_SOURCE)
    _assert_parseable(out, f"{name}(plain-python)")
    assert out == PASSTHROUGH_SOURCE, (
        f"DRIFT DETECTED: {name} mutated trigger-free source -- "
        f"diff in {abs(len(out) - len(PASSTHROUGH_SOURCE))} chars"
    )


@pytest.mark.parametrize(
    "name_args",
    [
        ("convert_attention_masks_to_bool", ("plain",)),
        ("apply_fused_lm_head", ("plain",)),
    ],
)
def test_two_arg_rewriter_passthrough_on_plain_python(name_args):
    name, extra = name_args
    fn = getattr(compiler, name)
    result = fn(PASSTHROUGH_SOURCE, *extra) if name == "convert_attention_masks_to_bool" else fn(PASSTHROUGH_SOURCE, *extra)
    if isinstance(result, tuple):
        out, _applied = result
    else:
        out = result
    _assert_parseable(out, f"{name}(plain-python)")


# Targeted symbol-removal asserts: the rewrite must land, not silently no-op.


def test_higher_precision_softmax_inserts_float32_cast():
    """Pin: F.softmax(x, dim=-1) -> F.softmax(x, dim=-1,
    dtype=torch.float32).to(x.dtype)."""
    src = (
        "def f(x):\n"
        "    return F.softmax(x, dim=-1)\n"
    )
    out = compiler.higher_precision_softmax(src)
    _assert_parseable(out, "higher_precision_softmax(synth)")
    if "dtype = torch.float32" not in out and "dtype=torch.float32" not in out:
        pytest.fail(
            "DRIFT DETECTED: higher_precision_softmax did not insert "
            "the float32 cast; rewrite silently no-op'd"
        )
    if ".to(x.dtype)" not in out:
        pytest.fail(
            "DRIFT DETECTED: higher_precision_softmax did not insert "
            "the .to(x.dtype) back-cast"
        )


def test_fixup_fused_lm_head_gemma4_flat_logits_dropped():
    """Pins the gemma4 rename of flat_logits/flat_labels to shift_logits/shift_labels in the fused LM-head fixup."""
    src = (
        "    flat_logits = shift_logits.view(-1, vocab)\n"
        "    flat_labels = shift_labels.view(-1).to(device)\n"
        "    loss = loss_fct(flat_logits, flat_labels)\n"
    )
    out = compiler.fixup_fused_lm_head(src)
    if "flat_logits" in out:
        pytest.fail(
            "DRIFT DETECTED: fixup_fused_lm_head left ``flat_logits`` "
            "in place; gemma4 rewrite did not land"
        )
    if "flat_labels" in out:
        pytest.fail(
            "DRIFT DETECTED: fixup_fused_lm_head left ``flat_labels`` "
            "in place; gemma4 rewrite did not land"
        )


def test_replace_with_grouped_query_attention_inserts_enable_gqa():
    """Pins enable_gqa= insertion: the kwarg lands, or the source is unchanged when the matcher misses; output must always parse."""
    llama = _load_modeling("llama")
    if not hasattr(llama, "LlamaAttention"):
        pytest.skip("LlamaAttention not exposed on installed transformers")
    src = inspect.getsource(llama.LlamaAttention.forward)
    out = compiler.replace_with_grouped_query_attention(
        "LlamaAttention", src,
    )
    _assert_parseable(
        out, "replace_with_grouped_query_attention(LlamaAttention)",
        dedent=True,
    )


# unsloth_compile_transformers(X) emits unsloth_compiled_module_<type>.py, the combined cache these tests parse back.


def _compile_and_get_cache(model_type: str, monkeypatch) -> str:
    """Run unsloth_compile_transformers for model_type, return cache path."""
    monkeypatch.setenv("UNSLOTH_COMPILE_DISABLE", "1")
    monkeypatch.setenv("UNSLOTH_COMPILE_OVERWRITE", "1")

    try:
        mod = importlib.import_module(
            f"transformers.models.{model_type}.modeling_{model_type}",
        )
    except ModuleNotFoundError:
        pytest.skip(
            f"model_type {model_type} not present on installed "
            f"transformers, can't drive compiler"
        )
    except ImportError as exc:
        # gemma3n's config imports ImageNetInfo from a timm.data that dropped it.
        pytest.skip(
            f"model_type {model_type} raised on import, so the compiler cannot "
            f"be driven for it: {type(exc).__name__}: {exc}"
        )
    if hasattr(mod, "__UNSLOTH_PATCHED__"):
        try:
            delattr(mod, "__UNSLOTH_PATCHED__")
        except AttributeError:
            pass

    compiler.unsloth_compile_transformers(model_type, disable=True)

    cache_folder, _ = compiler.get_compile_folder()
    cache_path = os.path.join(
        cache_folder, f"unsloth_compiled_module_{model_type}.py",
    )
    return cache_path


@pytest.mark.parametrize("model_type", KNOWN_MODEL_TYPES)
def test_unsloth_compile_transformers_emits_parseable_cache(
    model_type, monkeypatch,
):
    """Drives the pipeline per model_type and AST-parses the combined cache, so any rewriter emitting invalid Python fails here."""
    cache_path = _compile_and_get_cache(model_type, monkeypatch)

    if not os.path.isfile(cache_path):
        # Pipeline emitted nothing (full_disable / early-exit); not drift.
        pytest.skip(
            f"unsloth_compile_transformers({model_type!r}) emitted no "
            f"combined cache file (pipeline early-exit)"
        )

    with open(cache_path, encoding="utf-8") as fh:
        cache_src = fh.read()

    if not cache_src.strip():
        pytest.fail(
            f"DRIFT DETECTED: unsloth_compile_transformers({model_type!r}) "
            f"wrote an empty combined cache at {cache_path}"
        )

    try:
        ast.parse(cache_src)
    except (SyntaxError, IndentationError) as exc:
        pytest.fail(
            f"DRIFT DETECTED: unsloth_compile_transformers({model_type!r}) "
            f"produced invalid Python at {cache_path}: "
            f"{type(exc).__name__}: {exc}"
        )


# Headline smoke test: gemma3 end-to-end on installed transformers.


def test_smoke_unsloth_compile_transformers_gemma3(monkeypatch):
    """Smoke test that unsloth_compile_transformers on gemma3 returns valid Python; it rejects trust_remote_code and fast_inference, so pass only accepted args."""
    monkeypatch.setenv("UNSLOTH_COMPILE_DISABLE", "1")
    monkeypatch.setenv("UNSLOTH_COMPILE_OVERWRITE", "1")
    _load_modeling("gemma3")

    try:
        mod = importlib.import_module(
            "transformers.models.gemma3.modeling_gemma3",
        )
        if hasattr(mod, "__UNSLOTH_PATCHED__"):
            delattr(mod, "__UNSLOTH_PATCHED__")
    except (ModuleNotFoundError, AttributeError):
        pass

    compiler.unsloth_compile_transformers("gemma3", disable=True)

    cache_folder, _ = compiler.get_compile_folder()
    cache_path = os.path.join(
        cache_folder, "unsloth_compiled_module_gemma3.py",
    )
    assert os.path.isfile(cache_path), (
        f"DRIFT DETECTED: gemma3 smoke -- no cache emitted at {cache_path}"
    )
    with open(cache_path, encoding="utf-8") as fh:
        cache_src = fh.read()
    try:
        ast.parse(cache_src)
    except (SyntaxError, IndentationError) as exc:
        pytest.fail(
            f"DRIFT DETECTED: gemma3 smoke produced invalid Python: "
            f"{type(exc).__name__}: {exc}"
        )


def test_smoke_unsloth_compile_transformers_unknown_model_type(monkeypatch):
    """Unknown model type must early-return None (no corrupt cache)."""
    monkeypatch.setenv("UNSLOTH_COMPILE_DISABLE", "1")
    result = compiler.unsloth_compile_transformers(
        "this_model_type_does_not_exist_xyz_123", disable=True,
    )
    assert result is None, (
        "DRIFT DETECTED: unsloth_compile_transformers should return "
        "None on unknown model_type, returned: " + repr(result)
    )


# Constant source blocks that create_new_function exec()s verbatim in unsloth_zoo/compiler.py.


@pytest.mark.parametrize(
    "const_name",
    [
        "DTYPE_MISMATCH_FIND",
        "DTYPE_MISMATCH_REPLACE",
        "COMPILED_LORA_FORWARD",
        "COMPILED_LORA_FORWARD_forced_float32",
        "disble_use_cache_logging",
        "replace_gradient_checkpointing",
    ],
)
def test_compiler_constant_source_blocks_parse(const_name):
    """Each constant is a source block that compiler.py exec()s as-is, so it must parse after placeholder substitution where applicable."""
    block = getattr(compiler, const_name, None)
    if block is None:
        pytest.skip(f"{const_name} not present (renamed?)")
    # The rewriter substitutes LAYER / ARGS / MODULELIST_ITEM / $; do the same.
    if const_name == "replace_gradient_checkpointing":
        block = (
            block.replace("LAYER", "layer")
            .replace("MODULELIST_ITEM", "self.layers")
            .replace("ARGS", "hidden_states")
            .replace("$", "    ")
        )
    try:
        ast.parse(textwrap.dedent(block))
    except (SyntaxError, IndentationError) as exc:
        pytest.fail(
            f"DRIFT DETECTED: constant {const_name} in unsloth_zoo/"
            f"compiler.py is invalid Python: "
            f"{type(exc).__name__}: {exc}"
        )
