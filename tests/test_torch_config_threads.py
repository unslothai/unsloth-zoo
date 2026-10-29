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

"""torch >= 2.12 keeps config assignments per thread (a ContextVar per entry). A reentrant checkpoint
recomputes on autograd's device thread, so the grad graph it compiles there, and that graph's lazily
compiled backward, never saw a value assigned on the main thread. set_torch_config publishes it."""

import threading
import uuid
from contextvars import ContextVar

import pytest
import torch
import torch._dynamo
import torch._dynamo.config as dynamo_config
import torch._functorch.config as functorch_config
import torch._inductor.config as inductor_config
import torch.utils.checkpoint as checkpoint_module

from unsloth_zoo.torch_config import (
    publish_torch_config,
    set_torch_config,
    torch_config_is_per_thread,
)

_UNSET = object()
_LIMIT = "recompile_limit" if hasattr(dynamo_config, "recompile_limit") else "cache_size_limit"
_COMPILER_CONFIG = getattr(torch.compiler, "config", None)
_HAS_TAG = hasattr(_COMPILER_CONFIG, "cache_key_tag")
_KEYS = [
    (inductor_config, "memory_planning"),
    (inductor_config, "post_grad_custom_post_pass"),
    (dynamo_config, _LIMIT),
    (functorch_config, "activation_memory_budget"),
] + ([(_COMPILER_CONFIG, "cache_key_tag")] if _HAS_TAG else [])


def _invalidate(module):
    if hasattr(module, "_is_dirty"):
        module._is_dirty = True
    for attr, default in (
        ("_hash_dirty_var", True),
        ("_hash_cache_var", None),
        ("_get_dict_dirty_keys_var", None),
        ("_get_dict_cache_var", None),
    ):
        if isinstance(getattr(module, attr, None), ContextVar):
            setattr(module, attr, ContextVar(attr, default = default))


@pytest.fixture(autouse = True)
def _restore_torch_config():
    saved = []
    for module, name in _KEYS:
        entry = module._config[name]
        override = entry.user_override
        current = override.get() if isinstance(override, ContextVar) else override
        saved.append((module, entry, current, entry.env_value_default, entry.env_value_force))
    torch._dynamo.reset()
    yield
    for module, entry, current, env_default, env_force in saved:
        if isinstance(entry.user_override, ContextVar):
            entry.user_override.set(current)
        else:
            entry.user_override = current
        entry.env_value_default = env_default
        entry.env_value_force = env_force
        _invalidate(module)
    torch._dynamo.reset()


def _in_thread(fn, name = "worker"):
    box = {}

    def run():
        try:
            box["value"] = fn()
        except BaseException as error:  # noqa: BLE001
            box["error"] = error

    thread = threading.Thread(target = run, name = name)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


try:
    from torch._inductor.custom_graph_pass import CustomGraphPass
except ImportError:  # torch < 2.6
    CustomGraphPass = None


def _recorder():
    if CustomGraphPass is None:
        pytest.skip("torch has no CustomGraphPass")
    seen = []
    key = f"unsloth-test-thread-recorder-{uuid.uuid4()}"  # a warm FX cache would skip the pass

    class Recorder(CustomGraphPass):
        def __call__(self, graph):
            seen.append(
                (threading.current_thread().name, getattr(_COMPILER_CONFIG, "cache_key_tag", None))
            )

        def uuid(self):
            return key

    return Recorder(), seen


def test_values_reach_a_new_thread():
    set_torch_config(inductor_config, "memory_planning", True)
    set_torch_config(dynamo_config, _LIMIT, 1031)
    set_torch_config(functorch_config, "activation_memory_budget", 0.25)
    if _HAS_TAG:
        set_torch_config(_COMPILER_CONFIG, "cache_key_tag", "unsloth/test-tag")

    def read():
        return (
            inductor_config.memory_planning,
            getattr(dynamo_config, _LIMIT),
            functorch_config.activation_memory_budget,
            getattr(_COMPILER_CONFIG, "cache_key_tag", "unsloth/test-tag"),
        )

    expected = (True, 1031, 0.25, "unsloth/test-tag")
    assert read() == expected
    assert _in_thread(read) == expected


def test_compile_on_another_thread_uses_the_values():
    recorder, seen = _recorder()
    set_torch_config(inductor_config, "post_grad_custom_post_pass", recorder)
    if _HAS_TAG:
        set_torch_config(_COMPILER_CONFIG, "cache_key_tag", "unsloth/test-tag")
    compiled = torch.compile(lambda x: torch.nn.functional.gelu(x * 2).sum(-1))
    _in_thread(lambda: compiled(torch.randn(4, 8)))
    assert [name for name, _ in seen] == ["worker"]
    if _HAS_TAG:
        assert seen[0][1] == "unsloth/test-tag"


def test_checkpoint_recompute_on_device_thread_compiles_with_the_values(monkeypatch):
    # On CUDA, CheckpointFunction.backward runs on autograd's device thread, in a fresh context:
    # torch enters the caller's context only for AOT compiled backwards, never for this PyNode.
    # On CPU backward runs on the caller, so run that body on a fresh thread to get the same view.
    original = checkpoint_module.CheckpointFunction.backward

    def device_thread_backward(ctx, *grads):
        return _in_thread(lambda: original(ctx, *grads), name = "device")

    monkeypatch.setattr(
        checkpoint_module.CheckpointFunction, "backward", staticmethod(device_thread_backward)
    )
    recorder, seen = _recorder()
    set_torch_config(inductor_config, "post_grad_custom_post_pass", recorder)
    if _HAS_TAG:
        set_torch_config(_COMPILER_CONFIG, "cache_key_tag", "unsloth/test-tag")

    compiled = torch.compile(lambda x, w: torch.nn.functional.gelu(x @ w).sum(-1))
    x = torch.randn(4, 8, requires_grad = True)
    w = torch.randn(8, 8, requires_grad = True)
    loss = checkpoint_module.checkpoint(compiled, x, w, use_reentrant = True).sum()
    assert [name for name, _ in seen] == ["MainThread"]  # no_grad graph
    loss.backward()  # the recompute misses the grad-mode guard: grad graph + its backward
    on_device = [tag for name, tag in seen if name == "device"]
    assert len(on_device) == 2, seen
    if _HAS_TAG:
        assert on_device == ["unsloth/test-tag"] * 2
    reference = torch.autograd.grad(torch.nn.functional.gelu(x @ w).sum(), (x, w))
    torch.testing.assert_close(x.grad, reference[0])
    torch.testing.assert_close(w.grad, reference[1])


def test_recompile_limit_holds_on_another_thread():
    set_torch_config(dynamo_config, _LIMIT, 1024)
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(threading.current_thread().name)
        return gm.forward

    compiled = torch.compile(lambda x: x * 2 + 1, backend = backend, dynamic = False)
    _in_thread(lambda: [compiled(torch.randn(n)) for n in range(1, 13)])
    assert graphs == ["worker"] * 12  # the default limit (8) falls back to eager after 8


def test_patch_still_wins_and_restores():
    set_torch_config(inductor_config, "memory_planning", True)
    with inductor_config.patch(memory_planning = False):
        assert inductor_config.memory_planning is False
        if torch_config_is_per_thread(inductor_config):  # torch < 2.12: patch is process-wide
            assert _in_thread(lambda: inductor_config.memory_planning) is True
    assert inductor_config.memory_planning is True


def test_force_wins_over_thread_assignments():
    set_torch_config(dynamo_config, _LIMIT, 1024)
    _in_thread(lambda: setattr(dynamo_config, _LIMIT, 64))
    if not torch_config_is_per_thread(dynamo_config):
        pytest.skip("torch config is process-wide already")
    set_torch_config(dynamo_config, _LIMIT, 2048, force = True)

    def assign_then_read():
        setattr(dynamo_config, _LIMIT, 64)
        return getattr(dynamo_config, _LIMIT)

    assert _in_thread(assign_then_read) == 2048


def test_publish_existing_value_and_unknown_keys():
    inductor_config.memory_planning = True
    published = publish_torch_config(inductor_config, "memory_planning")
    assert published is torch_config_is_per_thread(inductor_config, "memory_planning")
    assert _in_thread(lambda: inductor_config.memory_planning) is True
    assert publish_torch_config(inductor_config, "no_such_key_unsloth") is False
    with pytest.raises(AttributeError):
        set_torch_config(inductor_config, "no_such_key_unsloth", 1)


def test_config_hash_matches_across_threads():
    # The FX graph cache keys on save_config_portable(): both threads must export the same values.
    # (get_hash() is not compared: it reprs object-valued entries by address, per thread.)
    set_torch_config(inductor_config, "memory_planning", True)
    set_torch_config(dynamo_config, _LIMIT, 1031)
    main = inductor_config.save_config_portable() if hasattr(inductor_config, "save_config_portable") else None
    if main is None:
        pytest.skip("torch has no save_config_portable")
    assert _in_thread(inductor_config.save_config_portable) == main
    assert main["memory_planning"] is True
