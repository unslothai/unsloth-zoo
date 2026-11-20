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
    """Runs body in a fresh interpreter with UNSLOTH_COMPILE_DISABLE cleared, since torch_compile is bound to noop at import and CI pins the variable to 1."""
    env = dict(os.environ)
    env.pop("UNSLOTH_COMPILE_DISABLE", None)
    done = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output = True, text = True, env = env,
        cwd = str(ZOO.parent),
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout


# ---- the unwrap itself ----

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
    assert unwrap_already_compiled(eager) is eager


def test_unwrap_already_compiled_survives_a_carrier_with_no_wrapped():
    """A carrier without __wrapped__ (an OptimizedModule, say) must bail out, not raise AttributeError; patch_function's hand-rolled unwrap is unguarded."""
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
    """__wrapped__ on a bound method is the unbound original, so unwrapping would make self a user argument; stop there as torch's innermost_fn does."""
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


# ---- the funnel ----

def test_torch_compile_hands_torch_the_eager_function():
    """Checks which callable reaches torch.compile: pre-fix the wrapper was passed (refused by torch 2.11, silently unwrapped by 2.10), so it fails on every torch."""
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
    """The GPT-OSS load sequence (fullgraph fallback decorator, then bare torch_compile on its result) raised an empty AssertionError on torch 2.11 before the fix."""
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
    """On the non-fullgraph branch a refused torch.compile decoration falls back to eager with a warning, as the
    fullgraph branch does, rather than failing the model load."""
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
    """torch.compile(model = fn) is the same as (fn); left in kwargs the funnel returned an undecorated decorate that compiled its first argument."""
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
    """The model = None factory spelling must not stay in kwargs: it collides with the positional function, and the eager guard then disables compile for good."""
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
    """The eager fallback must engage only on failure; running eager where compile succeeds is a performance regression, not a fix."""
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
