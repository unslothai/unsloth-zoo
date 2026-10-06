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

"""A failed in-process vLLM engine must be freed before load_vllm retries."""

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
    __module__ = "vllm.model_executor.fake"

    def __init__(self):
        self.weights = bytearray(1 << 20)
        self.self_ref = self  # vLLM models are full of reference cycles
        self.hook_handle = convert_frame.register_bytecode_hook(self.bytecode_hook)
        eval_frame.cached_backends.setdefault(id(self), self)

    def bytecode_hook(self, old_code, new_code):
        return None

    def __call__(self, *args, **kwargs):
        return None


class _OtherCompiledModel(_FakeCompiledModel):
    """A torch.compile from another thread while the engine starts."""
    __module__ = "user_code"


def _failing_engine_build(created, concurrent = None):
    model = _FakeCompiledModel()
    created.append(weakref.ref(model))
    if concurrent is not None: concurrent.append(_OtherCompiledModel())
    raise ValueError("No available memory for the cache blocks.")


def _failed_attempt(release, concurrent = None):
    created = []
    snapshot = vllm_utils._snapshot_dynamo_engine_registries()
    try:
        _failing_engine_build(created, concurrent)
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


def test_release_keeps_entries_another_compilation_added_meanwhile():
    concurrent = []
    model_ref = _failed_attempt(release = True, concurrent = concurrent)
    assert model_ref() is None
    other = concurrent[0]
    assert eval_frame.cached_backends.get(id(other)) is other
    assert any(getattr(hook, "__self__", None) is other for hook in convert_frame._bytecode_hooks.values())


class _FakeVllmBackend:
    __module__ = "vllm.compilation.backends"

    def __call__(self, gm, example_inputs):
        return gm.forward


def test_release_unwraps_the_torch_compile_backend_wrapper():
    torch = pytest.importorskip("torch")
    snapshot = vllm_utils._snapshot_dynamo_engine_registries()
    compiled = torch.compile(lambda x: x * 2, backend = _FakeVllmBackend())
    compiled(torch.ones(2))
    added = set(eval_frame.cached_backends) - snapshot[0][1]
    assert added, "torch.compile did not register its backend"
    vllm_utils._release_failed_vllm_engine(snapshot)
    assert not (added & set(eval_frame.cached_backends))


def _load_vllm_with_failing_engine(monkeypatch, errors):
    torch = pytest.importorskip("torch")
    vllm = pytest.importorskip("vllm")
    transformers = pytest.importorskip("transformers")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 0))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)
    monkeypatch.setattr(vllm_utils, "get_mem_info", lambda: (79 * 1024**3, 80 * 1024**3))
    created, errors = [], list(errors)

    def failing_llm(**kwargs):
        model = _FakeCompiledModel()
        created.append(weakref.ref(model))
        raise ValueError(errors.pop(0))

    monkeypatch.setattr(vllm, "LLM", failing_llm)
    config = transformers.Qwen2Config(
        hidden_size = 256, intermediate_size = 512, num_hidden_layers = 2,
        num_attention_heads = 4, num_key_value_heads = 2, vocab_size = 1000,
    )
    with pytest.raises(RuntimeError):
        vllm_utils.load_vllm(
            model_name = "Qwen/Qwen2-0.5B", config = config, max_seq_length = 512,
            gpu_memory_utilization = 0.5, use_bitsandbytes = False, dtype = torch.bfloat16,
        )
    gc.collect()
    return created


@pytest.mark.parametrize("errors", [
    ["engine startup failed"],  # not retried: raised straight away
    ["No available memory for the cache blocks.", "No available memory for the cache blocks."],
])
def test_load_vllm_frees_every_failed_engine_including_the_last(monkeypatch, errors):
    created = _load_vllm_with_failing_engine(monkeypatch, errors)
    assert len(created) == len(errors)
    assert all(ref() is None for ref in created)
