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

"""CPU generate decodes on the SMT siblings too where that pays (AMD x86 without native bf16); training keeps its count."""

import pytest
import torch

from unsloth_zoo.cpu import runtime


@pytest.fixture
def zen3(monkeypatch):
    """A 2-core x 2-SMT AMD x86 host without native bf16, threads fitted by default to 2."""
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    monkeypatch.delenv("UNSLOTH_CPU_ROLLOUT_THREADS", raising = False)
    for name in ("LOCAL_WORLD_SIZE", "MPI_LOCALNRANKS", "OMPI_COMM_WORLD_LOCAL_SIZE", "MV2_COMM_WORLD_LOCAL_SIZE"):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(runtime, "_THREADS_FITTED", (2, "default"))
    monkeypatch.setattr(runtime.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(runtime, "_cpu_vendor", lambda: "authenticamd")
    monkeypatch.setattr(runtime, "cpu_bf16_supported", lambda: False)
    monkeypatch.setattr(runtime, "_logical_cpus", lambda: 4)
    yield monkeypatch
    torch.set_num_threads(old)


def test_amd_without_bf16_widens_to_the_siblings(zen3):
    assert runtime.rollout_threads() == 4


@pytest.mark.parametrize("case", ["intel", "native_bf16", "arm", "user_threads", "quota", "ranks", "no_smt",
                                  "unfitted", "threads_changed"])
def test_other_hosts_keep_the_training_count(zen3, case):
    if case == "intel":
        zen3.setattr(runtime, "_cpu_vendor", lambda: "genuineintel")
    elif case == "native_bf16":
        zen3.setattr(runtime, "cpu_bf16_supported", lambda: True)
    elif case == "arm":
        zen3.setattr(runtime.platform, "machine", lambda: "arm64")
    elif case == "user_threads":
        zen3.setattr(runtime, "_THREADS_FITTED", (2, "UNSLOTH_CPU_THREADS"))
    elif case == "quota":
        zen3.setattr(runtime, "_THREADS_FITTED", (2, "cgroup CPU quota"))
    elif case == "ranks":
        zen3.setenv("LOCAL_WORLD_SIZE", "2")
    elif case == "no_smt":
        zen3.setattr(runtime, "_logical_cpus", lambda: 2)
    elif case == "unfitted":
        zen3.setattr(runtime, "_THREADS_FITTED", None)
    elif case == "threads_changed":
        torch.set_num_threads(1)
    assert runtime.rollout_threads() is None


def test_count_is_capped(zen3):
    torch.set_num_threads(12)
    zen3.setattr(runtime, "_THREADS_FITTED", (12, "physical cores"))
    zen3.setattr(runtime, "_logical_cpus", lambda: 48)
    assert runtime.rollout_threads() == runtime._DEFAULT_THREAD_CAP


@pytest.mark.parametrize("value,expected", [("0", None), ("3", 3), ("2", None), ("x", None)])
def test_env_override(zen3, value, expected):
    zen3.setattr(runtime, "_cpu_vendor", lambda: "genuineintel")  # the override applies on any host
    zen3.setenv("UNSLOTH_CPU_ROLLOUT_THREADS", value)
    assert runtime.rollout_threads() == expected


def test_env_zero_turns_it_off_on_amd(zen3):
    zen3.setenv("UNSLOTH_CPU_ROLLOUT_THREADS", "0")
    assert runtime.rollout_threads() is None


def test_vendor_from_windows_processor_string(monkeypatch):
    runtime._cpu_vendor.cache_clear()
    try:
        def no_cpuinfo(*a, **k):
            raise OSError
        monkeypatch.setattr("builtins.open", no_cpuinfo)
        monkeypatch.setattr(runtime.platform, "processor", lambda: "AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD")
        assert "authenticamd" in runtime._cpu_vendor()
    finally:
        monkeypatch.undo()
        runtime._cpu_vendor.cache_clear()


def _tiny_attached():
    transformers = pytest.importorskip("transformers")
    from unsloth_zoo.cpu.loader import FastCPUModel
    torch.manual_seed(0)
    config = transformers.LlamaConfig(vocab_size = 256, hidden_size = 32, intermediate_size = 64, num_hidden_layers = 1,
                                      num_attention_heads = 2, num_key_value_heads = 1, max_position_embeddings = 64)
    model = transformers.LlamaForCausalLM(config).eval()
    model.generation_config.pad_token_id = 0
    model.generation_config.eos_token_id = None
    FastCPUModel._attach(model)
    return model


def test_generate_decodes_on_rollout_threads_and_restores(zen3):
    model = _tiny_attached()
    seen = []
    model.model.layers[0].register_forward_pre_hook(lambda *a: seen.append(torch.get_num_threads()))
    ids = torch.randint(3, 256, (2, 5))
    with torch.no_grad():
        model.generate(ids, attention_mask = torch.ones_like(ids), max_new_tokens = 4, do_sample = False)
    assert seen and set(seen) == {4}
    assert torch.get_num_threads() == 2


def test_generate_restores_threads_on_error(zen3):
    model = _tiny_attached()

    def boom(*a, **k):
        assert torch.get_num_threads() == 4
        raise RuntimeError("boom")
    model.model.layers[0].register_forward_pre_hook(boom)
    ids = torch.randint(3, 256, (1, 5))
    with pytest.raises(RuntimeError, match = "boom"), torch.no_grad():
        model.generate(ids, attention_mask = torch.ones_like(ids), max_new_tokens = 2, do_sample = False)
    assert torch.get_num_threads() == 2


def test_generate_keeps_threads_elsewhere(zen3):
    zen3.setattr(runtime, "_cpu_vendor", lambda: "genuineintel")
    model = _tiny_attached()
    seen = []
    model.model.layers[0].register_forward_pre_hook(lambda *a: seen.append(torch.get_num_threads()))
    ids = torch.randint(3, 256, (1, 5))
    with torch.no_grad():
        model.generate(ids, attention_mask = torch.ones_like(ids), max_new_tokens = 2, do_sample = False)
    assert set(seen) == {2}
