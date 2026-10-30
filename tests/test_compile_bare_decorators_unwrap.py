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
"""Funnel _compile_or_fall_back unwraps already-compiled input; torch_compile_with_fallback, _maybe_compile and
patch_function bypass it; torch 2.11+ asserts on it (functools.wraps copies get_compiler_config); most tests check
which callable reaches torch.compile; that holds on torch 2.6+."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

ZOO = Path(__file__).resolve().parents[1] / "unsloth_zoo"

TORCH_VERSION = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])


def _run(*bodies, **extra_env):
    """Runs bodies in a fresh interpreter with UNSLOTH_COMPILE_DISABLE cleared, since the compile wrappers read it at import; each body is dedented separately."""
    source = "".join(textwrap.dedent(body) for body in bodies)
    env = dict(os.environ)
    env.pop("UNSLOTH_COMPILE_DISABLE", None)
    env.update(extra_env)
    done = subprocess.run(
        [sys.executable, "-c", source],
        capture_output = True, text = True, env = env,
        cwd = str(ZOO.parent),
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout


# The carrier every test starts from: a `functools.wraps` copy of a compiled function.
_CARRIER = """
    from unsloth_zoo.temporary_patches import common as C
    from unsloth_zoo.temporary_patches.utils import torch_compile_with_fallback

    def apply_rotary_pos_emb(q, k):
        return q * 2, k * 2

    carrier = torch_compile_with_fallback(
        fullgraph = True, dynamic = True, options = C.torch_compile_options,
    )(apply_rotary_pos_emb)
    assert hasattr(carrier, "get_compiler_config"), \\
        "the fallback wrapper stopped advertising itself as compiled"
"""


# ---- what reaches torch.compile -------------------------------------------

def test_torch_compile_with_fallback_hands_torch_the_eager_function():
    """The decorator generated cache modules carry: unfixed, it passes the wrapper straight through, which torch 2.11+ refuses."""
    _run(_CARRIER, """
        import torch
        seen = []
        real_compile = torch.compile

        def _recording_compile(model = None, **kwargs):
            seen.append(model)
            return real_compile(model, **kwargs)

        torch.compile = _recording_compile
        try:
            for fullgraph in (True, False):
                seen.clear()
                out = torch_compile_with_fallback(
                    fullgraph = fullgraph, dynamic = True,
                )(carrier)
                assert seen, f"fullgraph = {fullgraph} never reached torch.compile"
                assert seen[0] is apply_rotary_pos_emb, (
                    f"torch_compile_with_fallback(fullgraph = {fullgraph}) handed "
                    f"torch the already-compiled wrapper ({seen[0]!r}), not the "
                    f"eager original"
                )
                assert out is not carrier
        finally:
            torch.compile = real_compile
    """)


def test_maybe_compile_hands_torch_the_eager_function():
    """_maybe_compile is emitted into trainer modules; its non-fullgraph branch called torch.compile bare, so both branches are checked to keep the unwrap."""
    _run(_CARRIER, """
        import torch
        seen = []
        real_compile = torch.compile

        def _recording_compile(model = None, **kwargs):
            seen.append(model)
            return real_compile(model, **kwargs)

        torch.compile = _recording_compile
        try:
            for kwargs in ({"dynamic": True}, {"dynamic": True, "fullgraph": True}):
                seen.clear()
                out = C._maybe_compile(**kwargs)(carrier)
                assert seen, f"_maybe_compile({kwargs}) never reached torch.compile"
                assert seen[0] is apply_rotary_pos_emb, (
                    f"_maybe_compile({kwargs}) handed torch the already-compiled "
                    f"wrapper ({seen[0]!r}), not the eager original"
                )
                assert out is not carrier
        finally:
            torch.compile = real_compile
    """)


def test_patch_function_unwraps_all_the_way_down():
    """patch_function must unwrap the whole __wrapped__ chain: one hop off a doubly wrapped callable lands on another carrier, which torch 2.11+ still refuses."""
    _run(_CARRIER, """
        import functools
        import torch
        from unsloth_zoo.temporary_patches.utils import patch_function

        # Two layers, which is what a re-patched module attribute looks like
        # after `pre_compile` and `post_compile` have both run.
        @functools.wraps(carrier)
        def outer(*args, **kwargs): return carrier(*args, **kwargs)

        class Attention:
            def forward(self, q, k): return q, k

        seen = []
        real_compile = torch.compile

        def _recording_compile(model = None, **kwargs):
            seen.append(model)
            return real_compile(model, **kwargs)

        torch.compile = _recording_compile
        try:
            patch_function(
                Attention, "forward", outer,
                fullgraph = True, match_level = "relaxed",
            )
        finally:
            torch.compile = real_compile

        assert seen, "patch_function never reached torch.compile"
        assert seen[0] is apply_rotary_pos_emb, (
            f"patch_function handed torch {seen[0]!r}, not the eager original"
        )
    """)


def test_patch_function_keeps_a_bound_method_bound():
    """Unwrap must stop at a bound method, whose __wrapped__ is the unbound original; torch 2.11+ refuses to compile it,
    so the eager guard keeps the load alive; can_safely_patch declines it, since installing it routes every instance
    through the original receiver"""
    _run("""
        import torch
        from unsloth_zoo.temporary_patches.utils import patch_function

        class Helper:
            def __init__(self): self.calls = []
            def forward(self, x):
                self.calls.append(x)
                return x + 1

        Helper.forward = torch.compile(Helper.forward)
        helper = Helper()
        bound = helper.forward
        assert hasattr(bound, "get_compiler_config"), \\
            "the bound method stopped forwarding the compiled marker"

        class Target:
            def forward(self, x): return x

        seen = []
        real_compile = torch.compile

        def _recording_compile(model = None, **kwargs):
            seen.append(model)
            return real_compile(model, **kwargs)

        torch.compile = _recording_compile
        try:
            patch_function(
                Target, "forward", bound,
                fullgraph = False, match_level = "relaxed",
            )
        finally:
            torch.compile = real_compile

        assert seen, "patch_function never reached torch.compile"
        assert seen[0] is bound, (
            f"patch_function unwrapped the bound method to {seen[0]!r} and lost "
            f"the receiver"
        )
        # And what it handed over is still callable AS a bound method: 3 is x,
        # not self.
        assert seen[0](3) == 4 and helper.calls == [3], (
            f"the callable handed to torch lost its receiver "
            f"(helper.calls = {helper.calls})"
        )
        print("BOUND_KEPT")
    """)


def test_patch_function_survives_a_compiler_that_refuses():
    """A carrier with no __wrapped__ is handed to torch untouched instead of raising AttributeError; the load survives
    whatever torch does, as _compile_or_fall_back guarantees"""
    _run("""
        import torch
        from unsloth_zoo.temporary_patches.utils import patch_function

        class Target:
            def forward(self, x): return x

        def replacement(self, x): return x + 1
        # A `get_compiler_config` carrier that is not a Dynamo wrapper and has
        # nothing to unwrap to; an `OptimizedModule` is the real-world shape.
        replacement.get_compiler_config = lambda: {}

        real_compile = torch.compile
        def _refuse(model = None, **kwargs):
            raise AssertionError

        torch.compile = _refuse
        try:
            assert patch_function(
                Target, "forward", replacement,
                fullgraph = True, match_level = "relaxed",
            )
        finally:
            torch.compile = real_compile

        assert Target().forward(1) == 2, "the eager replacement was not installed"
        print("REFUSAL_SURVIVED")
    """)


# ---- the raise itself, on the torch that has it ---------------------------

@pytest.mark.skipif(
    TORCH_VERSION < (2, 11),
    reason = "torch < 2.11 unwraps a functools.wraps copy itself, so the "
             "assert this reproduces is unreachable",
)
def test_the_bare_decorators_do_not_assert():
    """Each entry point decorating a carrier on torch 2.11+ must not raise AssertionError; nothing executes, so it stays
    runnable on a CPU-only box with no working inductor backend"""
    _run(_CARRIER, """
        import torch
        for fullgraph in (True, False):
            out = torch_compile_with_fallback(
                fullgraph = fullgraph, dynamic = True,
            )(carrier)
            assert callable(out) and out is not carrier

        for kwargs in ({"dynamic": True}, {"dynamic": True, "fullgraph": True}):
            out = C._maybe_compile(**kwargs)(carrier)
            assert callable(out) and out is not carrier
    """)


# ---- the cases that must NOT change ---------------------------------------

def test_a_plain_function_is_still_compiled():
    """A plain function with nothing to unwrap must stay compiled: the unwrap is a no-op, and silently running eager where compile works is a performance regression"""
    _run("""
        import torch
        from unsloth_zoo.temporary_patches import common as C
        from unsloth_zoo.temporary_patches.utils import torch_compile_with_fallback

        def eager(x): return x + 1

        for fullgraph in (True, False):
            out = torch_compile_with_fallback(fullgraph = fullgraph, dynamic = True)(eager)
            assert out is not eager, \\
                f"torch_compile_with_fallback(fullgraph = {fullgraph}) returned the eager function"

        for kwargs in ({"dynamic": True}, {"dynamic": True, "fullgraph": True}):
            out = C._maybe_compile(**kwargs)(eager)
            assert out is not eager, f"_maybe_compile({kwargs}) returned the eager function"

        # The non-fullgraph result is torch's own wrapper, unwrapped back to the
        # function we passed in.
        out = C._maybe_compile(dynamic = True)(eager)
        assert getattr(out, "_torchdynamo_orig_callable", None) is eager, \\
            f"_maybe_compile compiled something other than the function given ({out!r})"
    """)


def test_an_nn_module_survives_the_unwrap():
    """OptimizedModule carries get_compiler_config and has no __wrapped__, so unwrap_already_compiled must return it untouched instead of raising AttributeError"""
    _run("""
        import torch
        import torch.nn as nn
        from unsloth_zoo.temporary_patches import common as C
        from unsloth_zoo.temporary_patches.utils import torch_compile_with_fallback

        class Model(nn.Module):
            def forward(self, x): return x + 1

        optimized = torch.compile(Model())
        assert hasattr(optimized, "get_compiler_config"), \\
            "OptimizedModule stopped advertising itself as compiled"
        assert not hasattr(optimized, "__wrapped__"), \\
            "OptimizedModule grew a __wrapped__; this test's premise is gone"

        assert C.unwrap_already_compiled(optimized) is optimized

        # And every entry point still accepts it.
        assert callable(torch_compile_with_fallback(dynamic = True)(optimized))
        assert callable(C._maybe_compile(dynamic = True)(optimized))
        assert callable(C.torch_compile(optimized))
    """)


def test_compile_disable_keeps_everything_eager():
    """UNSLOTH_COMPILE_DISABLE = 1 must keep nothing reaching torch.compile; the unwrap runs before the disable check in one branch and after it in another"""
    _run("""
        import torch
        from unsloth_zoo.temporary_patches import common as C

        assert C.UNSLOTH_COMPILE_DISABLE, "the environment variable did not take"

        def eager(x): return x + 1

        # `_maybe_compile` short-circuits to identity.
        assert C._maybe_compile(dynamic = True)(eager) is eager
        assert C._maybe_compile(dynamic = True, fullgraph = True)(eager) is eager

        def _refuse(model = None, **kwargs):
            raise AssertionError("torch.compile must not be reached when disabled")

        real_compile = torch.compile
        torch.compile = _refuse
        try:
            # The aliases are `noop`, i.e. `torch.compiler.disable`.
            for name in ("torch_compile", "_torch_compile"):
                out = getattr(C, name)(eager)
                assert getattr(out, "_torchdynamo_disable", False), \\
                    f"{name} did not disable {eager!r} under UNSLOTH_COMPILE_DISABLE"
            assert C._maybe_compile(dynamic = True)(eager) is eager
        finally:
            torch.compile = real_compile
        print("DISABLED_OK")
    """, UNSLOTH_COMPILE_DISABLE = "1")
