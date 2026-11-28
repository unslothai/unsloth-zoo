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
mapped KV, against a fake vllm.LLM and a fake CUDA memory state. Only the owner's rollout-ending sleep is skipped,
only the default pool's cache counts as free, an out of memory error while awake turns it off, a skipped sleep still
resets the prefix and encoder caches, and torch.distributed ranks agree."""

import multiprocessing
import os
import socket
import types

import pytest

import unsloth_zoo.vllm_utils as vu

GB = 1 << 30


class Mem:
    """Device memory as torch reports it. vLLM's weights + KV live in a private (cumem) pool that may also cache free
    blocks; training allocates from the default pool, whose cache empty_cache releases."""

    def __init__(self, total = 24 * GB, pool = 18 * GB, private_cached = 0):
        self.total, self.pool, self.private_cached = total, pool, private_cached
        self.reserved_default = self.allocated_default = 0
        self.peak = self.reserved
        self.num_ooms = 0

    @property
    def reserved(self):
        return self.pool + self.private_cached + self.reserved_default

    @property
    def allocated(self):
        return self.pool + self.allocated_default

    @property
    def free(self):
        return self.total - self.reserved

    def grow(self, nbytes):
        # a training step: torch caches what it allocated, so reserved stays up after the step frees it
        self.reserved_default += nbytes
        self.peak = max(self.peak, self.reserved)

    def empty_cache(self):
        # releases the default pool's cache; a private pool that is in use keeps its blocks
        self.reserved_default = self.allocated_default

    def stats(self, *a, **k):
        return {"reserved_bytes.all.current": self.reserved, "reserved_bytes.all.peak": self.peak,
                "allocated_bytes.all.current": self.allocated, "num_ooms": self.num_ooms}

    def info(self, *a, **k):
        return self.free, self.total

    def snapshot(self, *a, **k):
        return [
            {"device": 0, "segment_type": "large", "segment_pool_id": (0, 0),
             "total_size": self.reserved_default, "active_size": self.allocated_default},
            {"device": 0, "segment_type": "large", "segment_pool_id": (1, 0),
             "total_size": self.pool + self.private_cached, "active_size": self.pool},
        ]


class Trainer:
    """The engine's owner: each method calls generate and the rollout-ending sleep from one call, as TRL does."""

    def __init__(self, llm, mem, calls):
        self.llm, self.mem, self.calls = llm, mem, calls

    def init_sleep(self, level = 2):
        self.llm.sleep(level = level)  # TRL's sleep in __init__

    def rollout(self, length = 100, level = 2, during = None):
        self.llm.wake_up(tags = ["weights"])
        self.llm.wake_up(tags = ["kv_cache"])
        self.llm.generate(["p"], length = length)
        if during is not None: during()
        self.llm.sleep(level = level)

    def hooked_step(self, train_bytes):
        """The generate goes through wrapper functions (Unsloth's generation hooks) before reaching vLLM."""
        del self.calls[:]
        self.llm.wake_up()
        _generation_hook(lambda: self.llm.generate(["p"]))
        self.llm.sleep(level = 2)
        self.mem.grow(train_bytes)
        return [c[0] for c in self.calls if c[0] in ("sleep", "wake_up")]

    def chat_rollout(self):
        self.llm.wake_up()
        self.llm.chat([{"role": "user", "content": "x"}])
        self.llm.sleep(level = 1)

    def step(self, train_bytes, length = 100):
        """One TRL >= 0.23 colocate step: wake weights, wake kv_cache, generate, sleep(level = 2), train."""
        del self.calls[:]
        self.rollout(length = length)
        self.mem.grow(train_bytes)
        return [c[0] for c in self.calls if c[0] in ("sleep", "wake_up")]


def make_env(monkeypatch):
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

        class Engine:
            def reset_encoder_cache(self):
                calls.append(("reset_encoder_cache",))

        class LLM:
            prefix_reset_ok = True

            def __init__(self):
                self.llm_engine = Engine()

            def sleep(self, level = 1, **kw):
                calls.append(("sleep", level))
                mem.empty_cache()

            def wake_up(self, tags = None):
                calls.append(("wake_up", tags))
                return True

            def reset_prefix_cache(self):
                calls.append(("reset_prefix_cache",))
                return self.prefix_reset_ok

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
        monkeypatch.setattr(vu.torch.cuda, "memory_snapshot", mem.snapshot)
        monkeypatch.setattr(vu.torch.cuda, "current_device", lambda: 0)
        monkeypatch.delenv("UNSLOTH_VLLM_STANDBY_KEEP_AWAKE", raising = False)
        monkeypatch.setattr(vu, "_STANDBY_KEEP_AWAKE", {"kept_awake": 0, "slept": 0, "last": None, "announced": False,
                                                        "disabled": None, "oom_mark": None})
        vu.patch_vllm_standby_keep_awake()
        llm = LLM()
        return Trainer(llm, mem, calls), llm, mem, calls
    return setup


def _generation_hook(call):
    return call()


@pytest.fixture
def env(monkeypatch):
    return make_env(monkeypatch)


def test_first_step_sleeps_then_a_fitting_step_stays_awake(env):
    trainer, llm, mem, calls = env()
    trainer.init_sleep()  # the trainer's sleep at init: no rollout yet, measures nothing
    assert trainer.step(2 * GB) == ["wake_up", "wake_up", "sleep"]
    # 2 GB measured; 6 GB free beside the mapped KV clears 1.2 x 2 GB + a 1.2 GB margin. The step after a sleep
    # still wakes, then stays awake; its training step reuses the 2 GB torch kept cached
    assert trainer.step(0) == ["wake_up", "wake_up"]
    for _ in range(3):
        # awake: neither wake_up nor sleep reaches vLLM
        assert trainer.step(0) == []
    stats = vu.standby_keep_awake_stats()
    assert stats["kept_awake"] == 4 and stats["slept"] == 1 and stats["last"]["kept_awake"]
    assert stats["disabled"] is None


def test_reused_cache_keeps_the_working_set_in_the_estimate(env):
    # many kept-awake steps that only reuse cached blocks must not decay the estimate: a 4x longer rollout still
    # scales the 2 GB working set to 8 GB, which does not fit, so vLLM sleeps
    trainer, llm, mem, calls = env()
    trainer.step(2 * GB)
    for _ in range(vu._STANDBY_KEEP_AWAKE_HISTORY + 4):
        trainer.step(0)
    assert trainer.step(0) == []
    assert trainer.step(0, length = 4 * 120 - 20) == ["sleep"]


def test_a_training_step_that_does_not_fit_keeps_sleeping(env):
    trainer, llm, mem, calls = env()
    for _ in range(4):
        # 4.5 GB measured: 1.2 x 4.5 + 1.2 GB margin > 6 GB free
        assert trainer.step(int(4.5 * GB)) == ["wake_up", "wake_up", "sleep"]
    assert vu.standby_keep_awake_stats()["kept_awake"] == 0


def test_a_longer_rollout_scales_the_need_and_sleeps(env):
    trainer, llm, mem, calls = env()
    trainer.step(2 * GB, length = 100)
    assert trainer.step(0, length = 100) == ["wake_up", "wake_up"]
    # 4x the padded rollout: the 2 GB step scales to 8 GB, so this step sleeps and frees the KV again
    assert trainer.step(0, length = 4 * 120 - 20) == ["sleep"]
    assert trainer.step(0, length = 100) == ["wake_up", "wake_up"]


@pytest.mark.parametrize("case", ["env_off", "not_cuda"])
def test_kill_switch_and_other_devices_always_sleep(env, monkeypatch, case):
    if case == "env_off":
        trainer, llm, mem, calls = env()
        monkeypatch.setenv("UNSLOTH_VLLM_STANDBY_KEEP_AWAKE", "0")
    else:
        trainer, llm, mem, calls = env(device = "xpu")
    for _ in range(4):
        assert trainer.step(1 * GB) == ["wake_up", "wake_up", "sleep"]


def test_a_sleep_without_a_rollout_always_sleeps(env):
    # a user freeing the GPU after training: no rollout since the last wake, so vLLM really sleeps
    trainer, llm, mem, calls = env()
    trainer.step(1 * GB)
    assert trainer.step(0) == ["wake_up", "wake_up"]
    del calls[:]
    llm.sleep(level = 1)
    assert calls == [("sleep", 1)]
    llm.wake_up()
    assert calls[-1] == ("wake_up", None)


def test_chat_rollouts_count_too(env):
    trainer, llm, mem, calls = env()
    for _ in range(2):
        trainer.chat_rollout()
        mem.grow(1 * GB)
    del calls[:]
    trainer.chat_rollout()
    assert [c[0] for c in calls if c[0] in ("sleep", "wake_up", "generate")] == ["generate"]


def test_measurements_belong_to_their_engine(env):
    # a second engine (a reload, another trainer) has measured nothing yet: its first rollout-ending sleep runs
    trainer, llm, mem, calls = env()
    trainer.step(1 * GB)
    assert trainer.step(0) == ["wake_up", "wake_up"]
    other = Trainer(type(llm)(), mem, calls)
    assert other.step(0) == ["wake_up", "wake_up", "sleep"]


def test_level_zero_pause_is_always_honoured(env):
    trainer, llm, mem, calls = env()
    trainer.step(1 * GB)
    trainer.step(0)
    del calls[:]
    llm.sleep(level = 0)
    llm.wake_up(tags = ["scheduling"])
    assert calls == [("sleep", 0), ("wake_up", ["scheduling"])]


# (a) Free memory counts only the default pool's cache: blocks cached inside vLLM's cumem pool (or the cuBLAS / CUDA
# graph pools) are reserved but unallocated, and training cannot allocate from them.
def test_private_pool_cache_is_not_free_memory(env):
    trainer, llm, mem, calls = env()
    trainer.step(2 * GB)
    assert trainer.step(0) == ["wake_up", "wake_up"]
    # During the next rollout (after its wake_up closed the measurement) vLLM's pool caches 3 GB taken from the
    # device's free memory. Training can use 1 GB of device memory plus its own 2 GB of default-pool cache:
    # 3 GB < 1.2 x 2 GB + a 1.2 GB margin, so vLLM must sleep. reserved - allocated would add the 3 GB vLLM pool
    # cache and call 6 GB free.
    def grow_private_pool():
        mem.private_cached = 3 * GB
    del calls[:]
    trainer.rollout(during = grow_private_pool)
    assert [c[0] for c in calls if c[0] in ("sleep", "wake_up")] == ["sleep"]
    last = vu.standby_keep_awake_stats()["last"]
    assert not last["kept_awake"]


def test_a_static_private_pool_cache_is_not_training_need(env):
    # 3 GB cached inside vLLM's pool from the start on a 28 GB card: the 2 GB step is measured as 2 GB, not 5 GB,
    # and 5 GB device free + 2 GB default cache keeps vLLM awake
    trainer, llm, mem, calls = env(total = 28 * GB, private_cached = 3 * GB)
    trainer.step(2 * GB)
    assert trainer.step(0) == ["wake_up", "wake_up"]
    last = vu.standby_keep_awake_stats()["last"]
    assert last["kept_awake"] and last["need_gb"] == round(1.2 * 2 * GB / 1e9, 3)


def test_a_private_pool_cache_released_during_the_step_does_not_shrink_the_estimate(env):
    # vLLM's pool drops 3 GB of cache while the 2 GB training step runs: reserved stays flat, so the step must still
    # be measured as 2 GB (3 GB was cached at the start, none at the end: nothing is taken out)
    trainer, llm, mem, calls = env(total = 26 * GB, private_cached = 3 * GB)
    trainer.init_sleep()
    del calls[:]
    trainer.rollout()
    mem.private_cached = 0
    mem.grow(2 * GB)
    trainer.step(0)
    assert vu.standby_keep_awake_stats()["last"]["need_gb"] >= round(1.2 * 2 * GB / 1e9, 3)


# (b) An out of memory error while vLLM was kept awake turns keep-awake off for the rest of the process.
def test_out_of_memory_while_awake_makes_every_later_sleep_real(env):
    trainer, llm, mem, calls = env()
    trainer.step(1 * GB)
    assert trainer.step(0) == ["wake_up", "wake_up"]
    mem.num_ooms += 1  # the kept-awake training step ran out of memory (an OOM fallback caught it)
    assert trainer.step(0) == ["sleep"]
    for _ in range(3):
        assert trainer.step(0) == ["wake_up", "wake_up", "sleep"]
    stats = vu.standby_keep_awake_stats()
    assert stats["disabled"] and stats["kept_awake"] == 1


def test_out_of_memory_after_a_level_zero_pause_still_counts(env):
    trainer, llm, mem, calls = env()
    trainer.step(1 * GB)
    assert trainer.step(0) == ["wake_up", "wake_up"]
    llm.sleep(level = 0)
    llm.wake_up(tags = ["scheduling"])
    mem.num_ooms += 1
    # the pause's wake_up reached vLLM, so this step's wake_ups do too; its sleep is real
    assert trainer.step(0) == ["wake_up", "wake_up", "sleep"]
    assert vu.standby_keep_awake_stats()["disabled"]


def test_out_of_memory_while_asleep_does_not_turn_it_off(env):
    trainer, llm, mem, calls = env()
    mem.num_ooms += 1  # e.g. a probe before the first rollout, with vLLM asleep
    trainer.step(1 * GB)
    mem.num_ooms += 1  # during a training step vLLM slept through
    assert trainer.step(0) == ["wake_up", "wake_up"]
    assert vu.standby_keep_awake_stats()["disabled"] is None


# (c) Only the owner's rollout-ending sleep may be skipped; every other sleep reaches vLLM.
def _resume(trainer):
    # unsloth's resume path: a module-level helper sleeps the trainer's engine before loading the checkpoint
    trainer.llm.sleep(1)


def _user_generate(llm):
    return llm.generate(["p"])


@pytest.mark.parametrize("case", ["resume", "user_free", "init", "user_rollout"])
def test_sleeps_that_do_not_end_the_owners_rollout_reach_vllm(env, case):
    trainer, llm, mem, calls = env()
    trainer.step(1 * GB)
    assert trainer.step(0) == ["wake_up", "wake_up"]
    # an evaluation generate between steps (fast_generate), so tokens are pending at the next sleep
    _user_generate(llm)
    del calls[:]
    if case == "resume":
        _resume(trainer)
    elif case == "user_free":
        llm.sleep(level = 1)
    elif case == "init":
        Trainer(llm, mem, calls).init_sleep(level = 1)
    else:
        # a user's own wake / generate / sleep in one function, not a method of the engine's owner
        def user_rollout():
            llm.wake_up()
            llm.generate(["p"])
            llm.sleep(level = 1)
        user_rollout()
    assert ("sleep", 1) in calls


def test_wrappers_between_the_owner_and_vllm_are_looked_past(env):
    trainer, llm, mem, calls = env()
    trainer.hooked_step(1 * GB)
    assert trainer.hooked_step(0) == ["wake_up"]
    assert trainer.hooked_step(0) == []


def test_a_rollout_sleep_from_another_call_is_not_its_end(env):
    # generate in one owner call, sleep from a different call of the same owner: not the rollout's end
    trainer, llm, mem, calls = env()
    trainer.step(1 * GB)
    assert trainer.step(0) == ["wake_up", "wake_up"]
    trainer_generate = lambda self: self.llm.generate(["p"])  # noqa: E731
    Trainer.gen_only = trainer_generate
    try:
        trainer.gen_only()
        del calls[:]
        trainer.init_sleep(level = 1)
    finally:
        del Trainer.gen_only
    assert calls == [("sleep", 1)]


# (e) A skipped sleep still drops what a sleep drops: the prefix cache and the encoder cache.
def test_a_kept_awake_step_resets_the_prefix_and_encoder_caches(env):
    trainer, llm, mem, calls = env()
    trainer.step(1 * GB)
    trainer.step(0)
    assert ("reset_prefix_cache",) in calls and ("reset_encoder_cache",) in calls
    assert not any(c[0] == "sleep" for c in calls)


def test_a_failed_prefix_cache_reset_sleeps(env):
    trainer, llm, mem, calls = env()
    llm.prefix_reset_ok = False
    trainer.step(1 * GB)
    assert trainer.step(0) == ["wake_up", "wake_up", "sleep"]
    assert vu.standby_keep_awake_stats()["kept_awake"] == 0


# (d) Under torch.distributed the ranks agree: keep awake only when every rank fits.
def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _rank_main(rank, port, private_cached, out):
    import torch.distributed as dist
    mp = pytest.MonkeyPatch()
    try:
        os.environ.update(MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(port))
        dist.init_process_group("gloo", rank = rank, world_size = 2)
        trainer, llm, mem, calls = make_env(mp)(private_cached = private_cached)
        trainer.step(2 * GB)
        out.put((rank, [trainer.step(0) for _ in range(3)], vu.standby_keep_awake_stats()["last"]))
        dist.destroy_process_group()
    finally:
        mp.undo()


@pytest.mark.skipif(not hasattr(os, "fork") or not vu.torch.distributed.is_available()
                    or not vu.torch.distributed.is_gloo_available(), reason = "needs fork and gloo")
def test_ranks_agree_to_keep_vllm_awake_only_when_every_rank_fits():
    ctx = multiprocessing.get_context("fork")
    out, port = ctx.Queue(), _free_port()
    # rank 1's card has 3 GB cached inside vLLM's pool, so its training step does not fit beside the KV
    procs = [ctx.Process(target = _rank_main, args = (rank, port, rank * 3 * GB, out)) for rank in range(2)]
    for p in procs: p.start()
    results = dict((r, (steps, last)) for r, steps, last in (out.get(timeout = 60) for _ in procs))
    for p in procs:
        p.join(timeout = 60)
        assert p.exitcode == 0
    for rank in range(2):
        steps, last = results[rank]
        assert steps == [["wake_up", "wake_up", "sleep"]] * 3, (rank, steps)
        assert last["world_size"] == 2 and not last["kept_awake"]


def test_patch_is_idempotent(env):
    trainer, llm, mem, calls = env()
    wrapped = vu.vllm.LLM.sleep
    vu.patch_vllm_standby_keep_awake()
    assert vu.vllm.LLM.sleep is wrapped


def test_installed_with_standby():
    import inspect
    src = inspect.getsource(vu.patch_vllm)
    assert "patch_vllm_reset_caches_on_sleep()\n        patch_vllm_standby_keep_awake()" in src
