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

"""A failed in-process vLLM engine must be freed before load_vllm retries.

vLLM's TorchCompileWithNoGuardsWrapper registers the compiled model's bound
`bytecode_hook` in torch._dynamo.convert_frame._bytecode_hooks, and torch.compile
stores its backend (VllmBackend, holding the traced graphs) in
torch._dynamo.eval_frame.cached_backends. Neither is removed when engine startup
fails, so the retry used to stack a second copy of the weights on the first.
"""

from __future__ import annotations

import ast
import gc
import importlib
import inspect
import textwrap
import weakref

import pytest


vllm_utils = importlib.import_module("unsloth_zoo.vllm_utils")
convert_frame = pytest.importorskip("torch._dynamo.convert_frame")
eval_frame = pytest.importorskip("torch._dynamo.eval_frame")


class _FakeCompiledModel:
    """Stands in for a vLLM model: registers itself the way vLLM does, then fails."""

    def __init__(self):
        self.weights = bytearray(1 << 20)
        self.self_ref = self  # vLLM models are full of reference cycles
        self.hook_handle = convert_frame.register_bytecode_hook(self.bytecode_hook)
        eval_frame.cached_backends.setdefault(id(self), self)

    def bytecode_hook(self, old_code, new_code):
        return None

    def __call__(self, *args, **kwargs):
        return None


def _failing_engine_build(created):
    model = _FakeCompiledModel()
    created.append(weakref.ref(model))
    raise ValueError("No available memory for the cache blocks.")


def _failed_attempt(release):
    created = []
    snapshot = vllm_utils._snapshot_dynamo_engine_registries()
    try:
        _failing_engine_build(created)
    except Exception as error:
        error = str(error)
    if release:
        vllm_utils._release_failed_vllm_engine(snapshot)
    else:
        gc.collect()
    return created[0]


@pytest.fixture(autouse = True)
def _restore_registries(monkeypatch):
    monkeypatch.setattr(vllm_utils, "_device_empty_cache", lambda: None)
    hooks = dict(convert_frame._bytecode_hooks)
    backends = dict(eval_frame.cached_backends)
    yield
    convert_frame._bytecode_hooks.clear(); convert_frame._bytecode_hooks.update(hooks)
    eval_frame.cached_backends.clear(); eval_frame.cached_backends.update(backends)


def test_failed_engine_leaks_without_the_release():
    # The leak this guards against: gc.collect alone cannot free it.
    model_ref = _failed_attempt(release = False)
    assert model_ref() is not None


def test_failed_engine_is_freed_by_the_release():
    model_ref = _failed_attempt(release = True)
    assert model_ref() is None


def test_release_keeps_entries_that_existed_before_the_attempt():
    keep = _FakeCompiledModel()
    hook_keys = set(convert_frame._bytecode_hooks.keys())
    backend_keys = set(eval_frame.cached_backends.keys())
    model_ref = _failed_attempt(release = True)
    assert model_ref() is None
    assert set(convert_frame._bytecode_hooks.keys()) == hook_keys
    assert set(eval_frame.cached_backends.keys()) == backend_keys
    assert eval_frame.cached_backends[id(keep)] is keep


def test_load_vllm_releases_before_each_retry_outside_the_except_block():
    # Inside the except block the traceback still holds the failed engine's frames,
    # so the release has to run at the top of the next loop iteration.
    tree = ast.parse(textwrap.dedent(inspect.getsource(vllm_utils.load_vllm)))
    loops = [node for node in ast.walk(tree) if isinstance(node, ast.While)
             and any(isinstance(n, ast.Try) for n in node.body)]
    assert len(loops) == 1
    body = loops[0].body
    try_at = next(i for i, n in enumerate(body) if isinstance(n, ast.Try))
    before_try = ast.unparse(ast.Module(body = body[:try_at], type_ignores = []))
    assert "_release_failed_vllm_engine(" in before_try
    assert "_snapshot_dynamo_engine_registries()" in before_try
    assert before_try.index("_release_failed_vllm_engine(") < \
        before_try.index("_snapshot_dynamo_engine_registries()")
    for handler in body[try_at].handlers:
        assert "_release_failed_vllm_engine" not in ast.unparse(handler)
