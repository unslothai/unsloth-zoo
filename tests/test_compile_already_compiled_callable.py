# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""`torch_compile` on a callable that has already been compiled.

GPT-OSS inference dies at load on torch 2.11 with a bare, message-less

    AssertionError:

out of `patch_GptOssAttention` -> `_compile_or_fall_back` -> `torch.compile`.
The assert is torch's own:

    torch/_dynamo/eval_frame.py:
        assert not hasattr(compile_wrapper, "get_compiler_config")

`compile_wrapper` is built with `functools.wraps(fn)`, which copies
`fn.__dict__`, so the assert says "the function you handed me is already a
compiled one". Dynamo unwraps its own wrappers first (`innermost_fn`), so this
can only be reached through a `functools.wraps` copy of a compiled function
that forwards `get_compiler_config` -- which is exactly what
`_fall_back_to_eager_on_recompile_limit` returns, on purpose, so that
"is this compiled?" checks keep answering yes after the fallback is installed.

The chain under GPT-OSS:

  * `pre_compile` writes `unsloth_compiled_cache/unsloth_compiled_module_gpt_oss.py`,
    whose `apply_rotary_pos_emb` carries
    `@torch_compile_with_fallback(fullgraph = True, ...)`, and installs it on
    `transformers.models.gpt_oss.modeling_gpt_oss`.
  * `post_compile` re-runs `patch_GptOssAttention`, which imports that name and
    calls `torch_compile(apply_rotary_pos_emb)` -- no `fullgraph`.

Up to torch 2.10 `innermost_fn` followed `_torchdynamo_orig_callable`
unconditionally, unwrapped the copy to the eager original and recompiled it, so
none of this showed. torch 2.11 also requires
`_torchdynamo_wrapper_id == id(fn)`, which a `functools.wraps` copy cannot
satisfy, and the assert became reachable.

`patch_function` has always hand-unwrapped `get_compiler_config` carriers via
`__wrapped__` before compiling. The bare `torch_compile` / `_torch_compile`
decorators never did.
"""

import functools
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

ZOO = Path(__file__).resolve().parents[1] / "unsloth_zoo"

TORCH_VERSION = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])


def _run_with_compile_enabled(body):
    """Run `body` in a fresh interpreter with UNSLOTH_COMPILE_DISABLE cleared.

    `torch_compile` and `_torch_compile` are bound at import under a module-level
    `if UNSLOTH_COMPILE_DISABLE:`, so they are already `noop` by the time a test
    runs in unsloth's consolidated CI job, which pins the variable to 1 for the
    whole suite. Rebinding the module attribute afterwards cannot undo that.
    """
    env = dict(os.environ)
    env.pop("UNSLOTH_COMPILE_DISABLE", None)
    done = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output = True, text = True, env = env,
        cwd = str(ZOO.parent),
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout


# ---- the unwrap itself ----------------------------------------------------

def test_unwrap_already_compiled_returns_the_eager_original():
    from unsloth_zoo.temporary_patches.common import unwrap_already_compiled

    def eager(x): return x + 1

    compiled = torch.compile(eager)
    assert hasattr(compiled, "get_compiler_config"), \
        "torch stopped marking its wrappers; this test's premise is gone"

    @functools.wraps(compiled)
    def fallback_wrapper(*args, **kwargs): return compiled(*args, **kwargs)
    fallback_wrapper.get_compiler_config = compiled.get_compiler_config

    assert unwrap_already_compiled(fallback_wrapper) is eager
    assert unwrap_already_compiled(compiled) is eager
    # A plain function is handed straight back; nothing to unwrap.
    assert unwrap_already_compiled(eager) is eager


def test_unwrap_already_compiled_survives_a_carrier_with_no_wrapped():
    """An `OptimizedModule`, say. Bail out rather than raise AttributeError.

    `patch_function`'s hand-rolled version does `new_func.__wrapped__`
    unguarded; the shared helper cannot, because it sees everything.
    """
    from unsloth_zoo.temporary_patches.common import unwrap_already_compiled

    def opaque(x): return x
    opaque.get_compiler_config = lambda: {}
    assert unwrap_already_compiled(opaque) is opaque

    # And a `__wrapped__` cycle terminates instead of spinning.
    def a(x): return x
    def b(x): return x
    a.get_compiler_config = lambda: {}
    b.get_compiler_config = lambda: {}
    a.__wrapped__ = b
    b.__wrapped__ = a
    unwrap_already_compiled(a)


def test_unwrap_already_compiled_keeps_a_bound_method_bound():
    """`__wrapped__` on a bound method is the UNBOUND original.

    A bound method forwards attribute lookups to `__func__`, so following
    `__wrapped__` would hand back a function that still wants `self` and turn the
    receiver into the first user argument. torch's own `innermost_fn` stops on a
    bound method for the same reason, so stop here too and let the caller's guard
    take it from there.
    """
    from unsloth_zoo.temporary_patches.common import unwrap_already_compiled

    class Model:
        def forward(self, x): return x + 1

    model = Model()
    compiled_forward = torch.compile(Model.forward)
    Model.forward = compiled_forward
    bound = model.forward

    assert hasattr(bound, "get_compiler_config"), \
        "the bound method stopped forwarding the compiled marker"
    assert bound.__wrapped__ is not bound
    assert unwrap_already_compiled(bound) is bound


# ---- the funnel -----------------------------------------------------------

def test_torch_compile_hands_torch_the_eager_function():
    """Version independent: observe WHICH callable reaches `torch.compile`.

    Before the fix the wrapper itself was passed, which torch 2.11 refuses and
    torch <= 2.10 silently unwrapped for us. Asserting on the argument fails on
    every torch, not only the one where the refusal is fatal.
    """
    _run_with_compile_enabled("""
        import functools
        import torch

        from unsloth_zoo.temporary_patches import common as C

        seen = []
        real_compile = torch.compile

        def _recording_compile(model = None, **kwargs):
            if model is not None: seen.append(model)
            return real_compile(model, **kwargs) if model is not None else real_compile(**kwargs)

        def eager(x): return x + 1
        compiled = real_compile(eager)

        @functools.wraps(compiled)
        def fallback_wrapper(*args, **kwargs): return compiled(*args, **kwargs)
        fallback_wrapper.get_compiler_config = compiled.get_compiler_config

        torch.compile = _recording_compile
        try:
            for name in ("torch_compile", "_torch_compile"):
                seen.clear()
                out = getattr(C, name)(fallback_wrapper)
                assert seen, f"{name} never reached torch.compile"
                assert seen[0] is eager, (
                    f"{name} handed torch the already-compiled wrapper "
                    f"({seen[0]!r}), not the eager original"
                )
                assert out is not fallback_wrapper
        finally:
            torch.compile = real_compile
    """)


@pytest.mark.skipif(
    TORCH_VERSION < (2, 11),
    reason = "torch < 2.11 unwraps a functools.wraps copy itself, so the "
             "assert this reproduces is unreachable",
)
def test_the_gpt_oss_load_sequence_does_not_assert():
    """The real shape of the failure, end to end, on the torch that has it.

    `torch_compile_with_fallback(fullgraph = True)` first, standing in for the
    decorator in the generated module, then a bare `torch_compile` on its
    result, standing in for `patch_GptOssAttention`. Unfixed, torch 2.11 raises
    `AssertionError` with an empty message here.
    """
    _run_with_compile_enabled("""
        import torch
        from unsloth_zoo.temporary_patches.utils import torch_compile_with_fallback
        from unsloth_zoo.temporary_patches import common as C

        def apply_rotary_pos_emb(q, k):
            return q * 2, k * 2

        installed = torch_compile_with_fallback(
            fullgraph = True, dynamic = True, options = C.torch_compile_options,
        )(apply_rotary_pos_emb)
        assert hasattr(installed, "get_compiler_config"), \\
            "the fallback wrapper stopped advertising itself as compiled"

        # Decoration is the whole failure: unfixed, this line raises before any
        # tensor exists. Nothing is executed here on purpose, so the test stays
        # runnable on a CPU-only box with no working inductor backend.
        recompiled = C.torch_compile(installed)
        assert callable(recompiled) and recompiled is not installed
    """)


def test_a_refused_decoration_falls_back_to_eager_without_fullgraph():
    """The non-fullgraph branch used to reach `torch.compile` bare, so anything
    the compiler refused to decorate took the whole model load with it. It is as
    protected as the fullgraph branch now: eager, warned about, not raised.
    """
    out = _run_with_compile_enabled("""
        import logging
        import torch

        logging.basicConfig(level = logging.WARNING)

        # Imported before torch.compile is replaced: unsloth_zoo compiles a few
        # module-level kernels at import (compile_with_eager_fallback), and a
        # refusing compiler there is a different failure from the one under test.
        from unsloth_zoo.temporary_patches import common as C

        real_compile = torch.compile
        def _refuse(model = None, **kwargs):
            raise AssertionError

        def eager(x): return x + 1

        torch.compile = _refuse
        try:
            for name in ("torch_compile", "_torch_compile"):
                assert getattr(C, name)(eager) is eager, \\
                    f"{name} did not fall back to the eager function"
                assert getattr(C, name)(dynamic = True)(eager) is eager, \\
                    f"{name} as a decorator factory did not fall back"
        finally:
            torch.compile = real_compile
        print("FELL_BACK")
    """)
    assert "FELL_BACK" in out


def test_the_model_keyword_spelling_compiles_the_function():
    """`torch.compile(model = fn)` is the signature's own spelling of `(fn)`.

    Left in `kwargs` it is not the positional function, so the funnel handed back
    the undecorated `decorate` and then, when the caller called that, compiled its
    first ARGUMENT: `torch_compile(model = fn)(tensor)` returned the tensor.
    """
    _run_with_compile_enabled("""
        import torch
        from unsloth_zoo.temporary_patches import common as C

        def eager(x): return x + 1

        for name in ("torch_compile", "_torch_compile"):
            for kwargs in ({}, {"fullgraph": True, "dynamic": True}):
                out = getattr(C, name)(model = eager, **kwargs)
                assert out is not eager, f"{name} {kwargs} returned the eager function"
                assert getattr(out, "__wrapped__", None) is eager \\
                    or getattr(out, "_torchdynamo_orig_callable", None) is eager, (
                    f"{name} {kwargs} did not compile the model keyword "
                    f"(got {out!r})"
                )
        print("MODEL_KW_OK")
    """)


def test_the_explicit_model_none_decorator_factory_still_compiles():
    """`torch_compile(model = None, ...)` is the factory spelling of the same
    signature. Left in `kwargs`, `model = None` collides with the function passed
    positionally to `torch.compile`, and the eager guard swallows the resulting
    `TypeError` as "the compiler refused", turning compilation off for good.
    """
    _run_with_compile_enabled("""
        import torch
        from unsloth_zoo.temporary_patches import common as C

        def eager(x): return x + 1

        for name in ("torch_compile", "_torch_compile"):
            for kwargs in ({}, {"fullgraph": True, "dynamic": True}):
                out = getattr(C, name)(model = None, **kwargs)(eager)
                assert out is not eager, (
                    f"{name} {kwargs} with an explicit model = None fell back to eager"
                )
    """)


def test_a_working_compile_is_still_compiled():
    """The fallback must engage only on failure. Silently running eager where
    compile succeeds is a performance regression, not a fix.
    """
    _run_with_compile_enabled("""
        import torch
        from unsloth_zoo.temporary_patches import common as C

        def eager(x): return x + 1

        for name in ("torch_compile", "_torch_compile"):
            out = getattr(C, name)(eager)
            assert out is not eager, f"{name} returned the eager function"
            assert hasattr(out, "get_compiler_config"), \\
                f"{name} returned something torch does not consider compiled"

            out = getattr(C, name)(dynamic = True, fullgraph = True)(eager)
            assert out is not eager, f"{name} (fullgraph) returned the eager function"
    """)
