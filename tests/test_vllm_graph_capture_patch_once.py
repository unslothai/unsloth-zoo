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

"""patch_vllm() runs on every fast_inference load: its wrappers must not nest per load."""

import gc
import sys
import types

import pytest

from unsloth_zoo import vllm_utils


def _fake_vllm(monkeypatch, version):
    captures = []

    class GPUModelRunner:
        def capture_model(self, *args, **kwargs):
            captures.append(gc.collect)
            return "captured"

    class GPUModelRunnerBase:
        def capture_model(self, *args, **kwargs):
            captures.append(gc.collect)
            return "captured_v0"

    logger = types.SimpleNamespace(info = lambda *args, **kwargs: None)
    modules = {
        "vllm": types.SimpleNamespace(__version__ = version),
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.worker": types.ModuleType("vllm.v1.worker"),
        "vllm.v1.worker.gpu_model_runner": types.SimpleNamespace(
            GPUModelRunner = GPUModelRunner, logger = logger,
        ),
        "vllm.worker": types.ModuleType("vllm.worker"),
        "vllm.worker.model_runner": types.SimpleNamespace(
            GPUModelRunnerBase = GPUModelRunnerBase, logger = logger,
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    collects = []
    monkeypatch.setattr(vllm_utils, "collect_garbage", lambda *args, **kwargs: collects.append(kwargs))
    return GPUModelRunner, GPUModelRunnerBase, captures, collects


@pytest.mark.parametrize("version", ["0.11.2", "0.10.2"])
def test_repeated_patch_wraps_capture_once(monkeypatch, version):
    v1, v0, captures, collects = _fake_vllm(monkeypatch, version)
    runners = [v1] + ([v0] if version.startswith("0.10") else [])
    vllm_utils.patch_vllm_graph_capture()
    installed = [runner.capture_model for runner in runners]
    for _ in range(3):
        vllm_utils.patch_vllm_graph_capture()
    assert [runner.capture_model for runner in runners] == installed

    for runner in runners:
        captures.clear()
        collects.clear()
        result = runner().capture_model()
        assert result in ("captured", "captured_v0")
        # gc.collect is suppressed while the original capture runs, and restored after.
        assert captures and captures[0] is not gc.collect
        assert gc.collect.__name__ == "collect"
        # Main ran this once per patch_vllm() call: 4 here.
        assert len(collects) == 1


def test_capture_replaced_after_patch_is_wrapped_again(monkeypatch):
    v1, _, captures, collects = _fake_vllm(monkeypatch, "0.11.2")
    vllm_utils.patch_vllm_graph_capture()

    def replaced(self):
        captures.append(gc.collect)
        return "replaced"

    v1.capture_model = replaced
    vllm_utils.patch_vllm_graph_capture()
    assert v1.capture_model is not replaced
    assert v1().capture_model() == "replaced"
    assert len(collects) == 1


def test_repeated_sleep_mode_patch_wraps_generate_once(monkeypatch):
    wakes = []

    class _Engine:
        def generate(self, *args, **kwargs):
            return "generated"

        def wake_up(self):
            wakes.append(1)

    engine_config = types.SimpleNamespace(
        model_config = types.SimpleNamespace(enable_sleep_mode = True),
    )
    LLM = type("LLM", (_Engine,), {"llm_engine": types.SimpleNamespace(vllm_config = engine_config)})
    AsyncLLMEngine = type("AsyncLLMEngine", (_Engine,), {"vllm_config": engine_config})
    MemAllocator = type("CuMemAllocator", (), {})
    def stub(*args, **kwargs): return None
    monkeypatch.setitem(sys.modules, "vllm.device_allocator", types.ModuleType("vllm.device_allocator"))
    monkeypatch.setitem(sys.modules, "vllm.device_allocator.cumem", types.SimpleNamespace(
        CuMemAllocator = MemAllocator, AllocationData = object,
        unmap_and_release = stub, create_and_map = stub, libcudart = None,
    ))
    monkeypatch.setitem(sys.modules, "vllm.utils", types.SimpleNamespace(is_pin_memory_available = stub))
    monkeypatch.setattr(vllm_utils, "DEVICE_TYPE", "cuda")
    monkeypatch.setattr(
        vllm_utils, "vllm", types.SimpleNamespace(LLM = LLM, AsyncLLMEngine = AsyncLLMEngine), raising = False,
    )

    vllm_utils.patch_vllm_enable_sleep_mode()
    installed = (LLM.generate, AsyncLLMEngine.generate)
    for _ in range(3):
        vllm_utils.patch_vllm_enable_sleep_mode()
    assert (LLM.generate, AsyncLLMEngine.generate) == installed
    for cls in (LLM, AsyncLLMEngine):
        wakes.clear()
        assert cls().generate() == "generated"
        assert len(wakes) == 1
