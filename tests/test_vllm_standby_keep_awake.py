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

"""Standby keep-awake: the GRPO step's sleep / wake_up is skipped once a measured training step fits beside the
mapped KV, against a fake vllm.LLM and a fake CUDA memory state."""

import types

import pytest

import unsloth_zoo.vllm_utils as vu

GB = 1 << 30


class Mem:
    """Device memory as torch reports it: reserved / allocated / all-time peak, and free device memory."""

    def __init__(self, total = 24 * GB, pool = 18 * GB):
        self.total, self.reserved, self.allocated, self.peak = total, pool, pool, pool

    @property
    def free(self):
        return self.total - self.reserved

    def grow(self, nbytes):
        # a training step: torch caches what it allocated, so reserved stays up after the step frees it
        self.reserved += nbytes
        self.peak = max(self.peak, self.reserved)

    def empty_cache(self):
        self.reserved = self.allocated

    def stats(self, *a, **k):
        return {"reserved_bytes.all.current": self.reserved, "reserved_bytes.all.peak": self.peak,
                "allocated_bytes.all.current": self.allocated}

    def info(self, *a, **k):
        return self.free, self.total


@pytest.fixture
def env(monkeypatch):
    def setup(device = "cuda", **mem_kw):
        mem = Mem(**mem_kw)
        calls = []

        class Completion:
            def __init__(self, n):
                self.token_ids = [0] * n

        class Out:
            def __init__(self, prompt, lens):
                self.prompt_token_ids = [0] * prompt
                self.outputs = [Completion(n) for n in lens]

        class LLM:
            def sleep(self, level = 1, **kw):
                calls.append(("sleep", level))
                mem.empty_cache()

            def wake_up(self, tags = None):
                calls.append(("wake_up", tags))
                return True

            def generate(self, prompts, rows = 8, length = 100):
                calls.append(("generate",))
                return [Out(20, [length] * rows)]

            def chat(self, messages, rows = 8, length = 100):
                calls.append(("generate",))
                return [Out(20, [length] * rows)]

        monkeypatch.setattr(vu, "vllm", types.SimpleNamespace(LLM = LLM), raising = False)
        monkeypatch.setattr(vu, "DEVICE_TYPE", device, raising = False)
        monkeypatch.setattr(vu.torch.cuda, "memory_stats", mem.stats)
        monkeypatch.setattr(vu.torch.cuda, "mem_get_info", mem.info)
        monkeypatch.delenv("UNSLOTH_VLLM_STANDBY_KEEP_AWAKE", raising = False)
        monkeypatch.setattr(vu, "_STANDBY_KEEP_AWAKE", {
            "kept_awake": 0, "slept": 0, "measured": [], "pending": None, "rollout_tokens": 0, "last": None,
            "announced": False,
        })
        vu.patch_vllm_standby_keep_awake()
        return LLM(), mem, calls
    return setup


def trl_step(llm, mem, calls, train_bytes, length = 100):
    """One TRL >= 0.23 colocate step: wake weights, wake kv_cache, generate, sleep(level = 2), train."""
    del calls[:]
    llm.wake_up(tags = ["weights"])
    llm.wake_up(tags = ["kv_cache"])
    llm.generate(["p"], length = length)
    llm.sleep(level = 2)
    mem.grow(train_bytes)
    return [c[0] for c in calls if c[0] != "generate"]


def test_first_step_sleeps_then_a_fitting_step_stays_awake(env):
    llm, mem, calls = env()
    llm.sleep(level = 2)  # the trainer's sleep at init: no rollout yet, measures nothing
    assert trl_step(llm, mem, calls, 2 * GB) == ["wake_up", "wake_up", "sleep"]
    # 2 GB measured; 6 GB free beside the mapped KV clears 1.2 x 2 GB + a 1.2 GB margin. The step after a sleep
    # still wakes, then stays awake; its training step reuses the 2 GB torch kept cached
    assert trl_step(llm, mem, calls, 0) == ["wake_up", "wake_up"]
    for _ in range(3):
        # awake: neither wake_up nor sleep reaches vLLM
        assert trl_step(llm, mem, calls, 0) == []
    stats = vu.standby_keep_awake_stats()
    assert stats["kept_awake"] == 4 and stats["slept"] == 1 and stats["last"]["kept_awake"]


def test_reused_cache_keeps_the_working_set_in_the_estimate(env):
    # many kept-awake steps that only reuse cached blocks must not decay the estimate: a 4x longer rollout still
    # scales the 2 GB working set to 8 GB, which does not fit, so vLLM sleeps
    llm, mem, calls = env()
    trl_step(llm, mem, calls, 2 * GB)
    for _ in range(vu._STANDBY_KEEP_AWAKE_HISTORY + 4):
        trl_step(llm, mem, calls, 0)
    assert calls == [("generate",)]
    assert trl_step(llm, mem, calls, 0, length = 4 * 120 - 20) == ["sleep"]


def test_a_training_step_that_does_not_fit_keeps_sleeping(env):
    llm, mem, calls = env()
    for _ in range(4):
        # 4.5 GB measured: 1.2 x 4.5 + 1.2 GB margin > 6 GB free
        assert trl_step(llm, mem, calls, int(4.5 * GB)) == ["wake_up", "wake_up", "sleep"]
    assert vu.standby_keep_awake_stats()["kept_awake"] == 0


def test_a_longer_rollout_scales_the_need_and_sleeps(env):
    llm, mem, calls = env()
    trl_step(llm, mem, calls, 2 * GB, length = 100)
    assert trl_step(llm, mem, calls, 0, length = 100) == ["wake_up", "wake_up"]
    # 4x the padded rollout: the 2 GB step scales to 8 GB, so this step sleeps and frees the KV again
    assert trl_step(llm, mem, calls, 0, length = 4 * 120 - 20) == ["sleep"]
    assert trl_step(llm, mem, calls, 0, length = 100) == ["wake_up", "wake_up"]


@pytest.mark.parametrize("case", ["env_off", "not_cuda"])
def test_kill_switch_and_other_devices_always_sleep(env, monkeypatch, case):
    if case == "env_off":
        llm, mem, calls = env()
        monkeypatch.setenv("UNSLOTH_VLLM_STANDBY_KEEP_AWAKE", "0")
    else:
        llm, mem, calls = env(device = "xpu")
    for _ in range(4):
        assert trl_step(llm, mem, calls, 1 * GB) == ["wake_up", "wake_up", "sleep"]


def test_a_sleep_without_a_rollout_always_sleeps(env):
    # a resume or a user freeing the GPU after training: no rollout since the last wake, so vLLM really sleeps
    llm, mem, calls = env()
    trl_step(llm, mem, calls, 1 * GB)
    assert trl_step(llm, mem, calls, 0) == ["wake_up", "wake_up"]
    del calls[:]
    llm.sleep(level = 1)
    assert calls == [("sleep", 1)]
    llm.wake_up()
    assert calls[-1] == ("wake_up", None)


def test_chat_rollouts_count_too(env):
    llm, mem, calls = env()
    for _ in range(2):
        llm.wake_up()
        llm.chat([{"role": "user", "content": "x"}])
        llm.sleep(level = 1)
        mem.grow(1 * GB)
    del calls[:]
    llm.chat([{"role": "user", "content": "x"}])
    llm.sleep(level = 1)
    assert calls == [("generate",)]


def test_level_zero_pause_is_always_honoured(env):
    llm, mem, calls = env()
    trl_step(llm, mem, calls, 1 * GB)
    trl_step(llm, mem, calls, 0)
    del calls[:]
    llm.sleep(level = 0)
    llm.wake_up(tags = ["scheduling"])
    assert calls == [("sleep", 0), ("wake_up", ["scheduling"])]


def test_patch_is_idempotent(env):
    llm, mem, calls = env()
    wrapped = vu.vllm.LLM.sleep
    vu.patch_vllm_standby_keep_awake()
    assert vu.vllm.LLM.sleep is wrapped


def test_installed_with_standby():
    import inspect
    src = inspect.getsource(vu.patch_vllm)
    assert "patch_vllm_reset_caches_on_sleep()\n        patch_vllm_standby_keep_awake()" in src
