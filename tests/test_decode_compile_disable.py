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


"""`torch_compiler_disable_unless_decode` is a hard disable except inside a compiled decode step."""

import os

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

import pytest

torch = pytest.importorskip("torch")

from unsloth_zoo.temporary_patches.utils import (
    UNSLOTH_DECODE_COMPILE,
    torch_compiler_disable_unless_decode,
    unsloth_decode_compile,
)


@torch_compiler_disable_unless_decode
def _inner(x):
    return x.sin() * 2


def _outer(x):
    return _inner(x + 1) - 1


@pytest.fixture(autouse = True)
def _reset():
    torch._dynamo.reset()
    UNSLOTH_DECODE_COMPILE[0] = False
    yield
    UNSLOTH_DECODE_COMPILE[0] = False
    torch._dynamo.reset()


def test_disabled_outside_decode():
    x = torch.randn(8)
    with pytest.raises(Exception, match = "(?i)disable"):
        torch.compile(_outer, fullgraph = True, backend = "eager")(x)
    explain = torch._dynamo.explain(_outer)(x)
    assert explain.graph_break_count >= 1


def test_traced_during_decode():
    x = torch.randn(8)
    UNSLOTH_DECODE_COMPILE[0] = True
    out = torch.compile(_outer, fullgraph = True, backend = "eager")(x)
    torch.testing.assert_close(out, _outer(x))


def test_eager_result_unchanged_either_way():
    x = torch.randn(8)
    expected = (x + 1).sin() * 2 - 1
    torch.testing.assert_close(_outer(x), expected)
    UNSLOTH_DECODE_COMPILE[0] = True
    torch.testing.assert_close(_outer(x), expected)


def test_flag_flip_retraces():
    x = torch.randn(8)
    compiled = torch.compile(_outer, backend = "eager")
    compiled(x)
    UNSLOTH_DECODE_COMPILE[0] = True
    torch.testing.assert_close(compiled(x), _outer(x))
    UNSLOTH_DECODE_COMPILE[0] = False
    torch.testing.assert_close(compiled(x), _outer(x))


def test_rewrapping_does_not_nest():
    again = torch_compiler_disable_unless_decode(_inner)
    assert again._unsloth_undisabled is _inner._unsloth_undisabled


def _stance_block(calls):
    from unsloth_zoo import compiler

    source = "def run():\n" + compiler.__DYNAMO__RECOMPILING__ + "    return INFERENCE_RUNS\n"
    stance = type("S", (), {"stance": "default", "skip_guard_eval_unsafe": False})()
    namespace = {
        "torch": torch,
        "INFERENCE_RUNS": 1,
        "UNSLOTH_DECODE_COMPILE": UNSLOTH_DECODE_COMPILE,
        "UNSLOTH_EAGER_STANCE_OWNED": [],
        "unsloth_claim_eager_stance": lambda stance: None,
        "unsloth_owns_stance": lambda current: False,
        "UNSLOTH_ENABLE_LOGGING": False,
        "torch_dynamo_eval_frame": type("E", (), {"_stance": stance})(),
        "torch_compiler_set_stance": lambda **kw: calls.append(kw["stance"]),
    }
    exec(source, namespace)
    return namespace["run"]


def test_stance_block_runs_for_eager_inference():
    calls = []
    assert _stance_block(calls)() == 2
    assert calls == ["eager_on_recompile"]


def test_stance_block_skipped_around_compiled_decode():
    calls = []
    UNSLOTH_DECODE_COMPILE[0] = True
    assert _stance_block(calls)() == 1
    assert calls == []


def _stance():
    import torch._dynamo.eval_frame as eval_frame
    return eval_frame._stance.stance


def test_scope_is_reference_counted_across_overlapping_calls():
    first, second = unsloth_decode_compile(), unsloth_decode_compile()
    first.__enter__()
    second.__enter__()
    first.__exit__(None, None, None)  # the earlier call finishes while the later one decodes
    assert UNSLOTH_DECODE_COMPILE[0] is True
    second.__exit__(None, None, None)
    assert UNSLOTH_DECODE_COMPILE[0] is False


def test_scope_clears_on_error():
    with pytest.raises(RuntimeError):
        with unsloth_decode_compile():
            raise RuntimeError
    assert UNSLOTH_DECODE_COMPILE[0] is False


@pytest.mark.skipif(not hasattr(torch.compiler, "set_stance"), reason = "torch without set_stance")
def test_scope_compiles_under_eager_on_recompile_and_restores_it():
    calls = []
    def backend(gm, example_inputs):
        calls.append(1)
        return gm.forward
    torch.compiler.set_stance("eager_on_recompile")
    try:
        with unsloth_decode_compile():
            assert _stance() == "default"
            torch.compile(lambda x: x.sin() + 1, backend = backend)(torch.randn(4))
        assert _stance() == "eager_on_recompile"
    finally:
        torch.compiler.set_stance("default")
    assert calls, "eager_on_recompile never compiles a new frame, so decode stayed eager"


def _nested_frames_compiled(recursive):
    frames = []
    def backend(gm, example_inputs):
        frames.append(gm)
        return gm.forward

    def inner(x):
        return x.cos() * 3

    @torch_compiler_disable_unless_decode(recursive = recursive)
    def block(x):
        return inner(x) + 1

    torch._dynamo.reset()
    torch.compile(lambda x: block(x * 2) - 1, backend = backend)(torch.randn(4))
    return len(frames)


def test_recursive_disable_keeps_the_subtree_eager_outside_decode():
    # The Qwen MoE block relied on the default recursive disable: nothing it calls may be
    # captured by an outer training compile.
    assert _nested_frames_compiled(recursive = True) < _nested_frames_compiled(recursive = False)
