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

"""Pins vLLM load-failure classification in standby mode: expandable-segments config clashes and memory profiling races
are not OOMs (loose 'alloc', 'memory' and 'free memory' substring checks caught them); real OOMs still are; load_vllm
wiring is checked."""

from __future__ import annotations

import inspect
import re
from unittest import mock

import pytest

from unsloth_zoo import vllm_utils


# The message our patched CuMemAllocator.__init__ asserts with, per env var.
def _unsloth_assertion(env_var = "PYTORCH_CUDA_ALLOC_CONF"):
    return (
        "Standby mode is not supported with expandable segments.\n"
        f"Please set environment variable {env_var} without `expandable_segments:True`.\n"
    )


# Upstream vLLM's own wording (vllm/device_allocator/cumem.py).
_VLLM_ASSERTION = (
    "Expandable segments are not compatible with memory pool. "
    "Please track https://github.com/pytorch/pytorch/issues/147851 "
    "for the latest updates."
)

# Verbatim profiling-race error from a shared 8x B200; raised by the assert in vllm/v1/worker/gpu_worker.py determine_available_memory().
_PROFILING_RACE = (
    "Error in memory profiling. Initial free memory 11.87 GiB, current free memory 138.8 GiB. "
    "This happens when other processes sharing the same container release GPU memory while vLLM "
    "is profiling during initialization. To fix this, ensure consistent GPU memory allocation or "
    "isolate vLLM in its own container."
)

# XPU worker (vllm/v1/worker/xpu_worker.py) asserts with the same opening phrase but a different tail, so the marker must match only the phrase.
_PROFILING_RACE_XPU = (
    "Error in memory profiling. Initial free memory 12746588160, current free memory 149023916032. "
    "This happens when the GPU memory was not properly cleaned up before initializing the vLLM "
    "instance."
)

# vLLM's real KV cache exhaustion (vllm/v1/core/kv_cache_utils.py); unlike the profiling race it is deterministic and gpu_memory_utilization is the fix.
_KV_CACHE_OOM = (
    "To serve at least one request with the models's max seq len (8192), (18.00 GiB KV cache is "
    "needed, which is larger than the available KV cache memory (4.00 GiB). Based on the available "
    "memory, the estimated maximum model length is 1808. Try increasing `gpu_memory_utilization` "
    "or decreasing `max_model_len` when initializing the engine."
)

_REAL_OOM_ERRORS = (
    # torch 2.9.1+cu128, verbatim. It recommends expandable segments itself.
    "CUDA out of memory. Tried to allocate 37252.90 GiB. GPU 0 has a total capacity of "
    "178.35 GiB of which 177.06 GiB is free. Process 597697 has 692.00 MiB memory in use. "
    "Including non-PyTorch memory, this process has 612.00 MiB memory in use. Of the "
    "allocated memory 0 bytes is allocated by PyTorch, and 0 bytes is reserved by PyTorch "
    "but unallocated. If reserved but unallocated memory is large try setting "
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See "
    "documentation for Memory Management  "
    "(https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)",
    "CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of "
    "79.15 GiB of which 1.06 GiB is free.",
    "torch.OutOfMemoryError: CUDA out of memory.",
    # ROCm / HIP
    "HIP out of memory. Tried to allocate 512.00 MiB",
    # vLLM engine-side
    "Free memory on device (78.68/79.15 GiB) on startup is less than desired GPU memory "
    "utilization (0.9, 71.24 GiB). Decrease GPU memory utilization or reduce GPU memory used "
    "by other processes.",
    "No available memory for the cache blocks. Try increasing `gpu_memory_utilization`.",
    "To serve at least one request with the model's max seq len (8192), (18.00 GiB KV cache "
    "is needed, which is larger than the available KV cache memory (4.00 GiB).",
    _KV_CACHE_OOM,
    # C++ level
    "std::bad_alloc",
)


@pytest.mark.parametrize(
    "env_var",
    ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_ALLOC_CONF"),
)
def test_expandable_segments_assertion_is_not_an_oom(env_var):
    # The bug: bare 'alloc' (and 'memory' for upstream wording) substrings returned True, so a config clash was raised as MemoryError.
    error = _unsloth_assertion(env_var)
    assert vllm_utils._is_expandable_segments_error(error) is True
    assert vllm_utils._is_out_of_memory_error(error) is False


def test_upstream_vllm_expandable_segments_assertion_is_not_an_oom():
    assert vllm_utils._is_expandable_segments_error(_VLLM_ASSERTION) is True
    assert vllm_utils._is_out_of_memory_error(_VLLM_ASSERTION) is False


@pytest.mark.parametrize("error", _REAL_OOM_ERRORS)
def test_real_oom_is_still_classified_as_oom(error):
    assert vllm_utils._is_out_of_memory_error(error) is True
    assert vllm_utils._is_expandable_segments_error(error) is False


@pytest.mark.parametrize(
    "error",
    (
        "ValueError: Model architecture FooForCausalLM is not supported.",
        "Could not find nvcc, please install the CUDA toolkit",
        "",
    ),
)
def test_unrelated_errors_are_neither(error):
    assert vllm_utils._is_out_of_memory_error(error) is False
    assert vllm_utils._is_expandable_segments_error(error) is False


def test_torch_oom_hint_about_expandable_segments_is_not_a_config_clash():
    # torch's genuine OOM text recommends PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, so config-clash markers must be full phrases.
    torch_oom = _REAL_OOM_ERRORS[0]
    assert "expandable_segments:True" in torch_oom
    assert vllm_utils._is_out_of_memory_error(torch_oom) is True
    assert vllm_utils._is_expandable_segments_error(torch_oom) is False


def test_classifiers_accept_exception_objects_not_just_strings():
    error = AssertionError(_unsloth_assertion())
    assert vllm_utils._is_expandable_segments_error(error) is True
    assert vllm_utils._is_out_of_memory_error(error) is False


@pytest.mark.parametrize(
    "env_var",
    ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_ALLOC_CONF"),
)
def test_message_names_the_offending_env_var_on_every_platform(env_var):
    # PYTORCH_HIP_ALLOC_CONF is the ROCm/AMD env var and PYTORCH_ALLOC_CONF the unified torch >= 2.10 one; name whichever env var actually carries the setting.
    cleared = {k: "" for k in vllm_utils._ALLOC_CONF_ENV_VARS}
    cleared[env_var] = "expandable_segments:True,roundup_power2_divisions:[32:256]"
    with mock.patch.dict(vllm_utils.os.environ, cleared, clear = False):
        assert vllm_utils._expandable_segments_env_vars() == [env_var]
        message = vllm_utils._expandable_segments_standby_message(_unsloth_assertion(env_var))
    assert env_var in message
    # Never claim the GPU ran out of memory for a config clash.
    assert "ran out of memory" not in message
    assert "NOT an out of memory error" in message
    assert "expandable_segments" in message
    assert "load_in_4bit" not in message
    assert "gpu_memory_utilization=0.6" not in message
    assert "Original error:" in message
    assert "Standby mode is not supported with expandable segments." in message


def test_message_lists_all_three_env_vars_when_none_are_visible():
    cleared = {k: "" for k in vllm_utils._ALLOC_CONF_ENV_VARS}
    with mock.patch.dict(vllm_utils.os.environ, cleared, clear = False):
        assert vllm_utils._expandable_segments_env_vars() == []
        message = vllm_utils._expandable_segments_standby_message(_VLLM_ASSERTION)
    for env_var in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        assert env_var in message
    assert "Original error:" in message


@pytest.mark.parametrize("error", (_PROFILING_RACE, _PROFILING_RACE_XPU))
def test_profiling_race_is_not_an_oom(error):
    # The bug: 'free memory' appears twice here, so the OOM markers matched and a 178 GB GPU was reported out of memory with 138.8 GiB free.
    assert "free memory" in error.lower()
    assert vllm_utils._is_memory_profiling_race_error(error) is True
    assert vllm_utils._is_out_of_memory_error(error) is False
    assert vllm_utils._is_expandable_segments_error(error) is False


def test_profiling_race_and_kv_cache_oom_classify_differently():
    # Both talk about KV cache memory in GiB. Only one of them is a shortage.
    assert vllm_utils._is_memory_profiling_race_error(_PROFILING_RACE) is True
    assert vllm_utils._is_out_of_memory_error(_PROFILING_RACE) is False

    assert vllm_utils._is_memory_profiling_race_error(_KV_CACHE_OOM) is False
    assert vllm_utils._is_out_of_memory_error(_KV_CACHE_OOM) is True


def test_free_memory_marker_still_covers_the_real_startup_shortfall():
    # Dropping the 'free memory' marker would silently lose the real startup shortfall; its text never says 'out of
    # memory' (vllm/v1/worker/utils.py request_memory).
    shortfall = (
        "Free memory on device (78.68/79.15 GiB) on startup is less than desired GPU memory "
        "utilization (0.9, 71.24 GiB). Decrease GPU memory utilization or reduce GPU memory "
        "used by other processes."
    )
    assert "out of memory" not in shortfall.lower()
    assert "free memory" in vllm_utils._OUT_OF_MEMORY_MARKERS
    assert vllm_utils._is_out_of_memory_error(shortfall) is True
    assert vllm_utils._is_memory_profiling_race_error(shortfall) is False


def test_profiling_race_classifier_accepts_exception_objects():
    assert vllm_utils._is_memory_profiling_race_error(AssertionError(_PROFILING_RACE)) is True


@pytest.mark.parametrize("standby", (False, True))
def test_profiling_race_message_never_offers_the_oom_remedies(standby):
    message = vllm_utils._memory_profiling_race_message(
        _PROFILING_RACE, trials = 3, unsloth_vllm_standby = standby,
    )
    assert "transient" in message
    assert "sharing this GPU" in message
    assert "NOT an out of memory error" in message
    assert "ran out of memory" not in message
    assert "Load the model again" in message
    assert "Unsloth already tried loading 3 times." in message
    # None of the three OOM remedies may be *offered*. Scope this to the list of
    # fixes, since the prose above it names them precisely to rule them out.
    remedies = message[message.index("Try one of these fixes:"):message.index("Original error:")]
    for useless in ("load_in_4bit", "smaller model", "4bit", "Lower gpu_memory_utilization"):
        assert useless not in remedies
    assert "gpu_memory_utilization" not in remedies
    assert "Original error:" in message
    assert _PROFILING_RACE in message


def test_profiling_race_message_explains_why_standby_cannot_retry_in_process():
    # CuMemAllocator is a process-wide singleton and Unsloth runs the engine in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0), so a retry needs a new process.
    standby = vllm_utils._memory_profiling_race_message(
        _PROFILING_RACE, trials = 1, unsloth_vllm_standby = True,
    )
    plain = vllm_utils._memory_profiling_race_message(
        _PROFILING_RACE, trials = 1, unsloth_vllm_standby = False,
    )
    assert "CuMemAllocator" in standby
    assert "restart the notebook kernel" in standby
    assert "CuMemAllocator" not in plain
    # A single attempt is not a retry, so do not claim it was.
    assert "already tried loading" not in standby


def test_load_vllm_retries_a_profiling_race_without_shrinking_anything():
    # The generic branch reacts to 'memory' by scaling gpu_memory_utilization by 0.85 and max_num_seqs by 0.75, neither of which caused a profiling race.
    source = inspect.getsource(vllm_utils.load_vllm)
    race_at = source.index("_is_memory_profiling_race_error(error)")
    standby_at = source.index("if trials >= 2 or unsloth_vllm_standby:")
    shrink_at = source.index('engine_args["gpu_memory_utilization"] *= 0.85')
    # Checked before the standby short-circuit, which gives up after a single attempt on a transient fault.
    assert race_at < standby_at < shrink_at
    assert "continue" in source[race_at:standby_at]
    assert "_MEMORY_PROFILING_RACE_MAX_TRIALS" in source[race_at:standby_at]
    assert vllm_utils._MEMORY_PROFILING_RACE_MAX_TRIALS >= 2


def test_profiling_race_retries_do_not_spend_the_generic_retry_budget():
    # A race is counted on its own race_trials; sharing trials would let a race, then a real OOM, hit trials >= 2 and raise, skipping the shrink-and-retry.
    source = inspect.getsource(vllm_utils.load_vllm)
    race_at = source.index("_is_memory_profiling_race_error(error)")
    bump_at = source.index("            trials += 1")
    standby_at = source.index("if trials >= 2 or unsloth_vllm_standby:")
    # The generic counter is bumped only after the race branch has passed on it.
    assert race_at < bump_at < standby_at
    assert source.count("            trials += 1") == 1
    race_branch = source[race_at:bump_at]
    assert "race_trials += 1" in race_branch
    assert "trials <" not in race_branch.replace("race_trials <", "")


def test_load_vllm_retry_handler_is_wired_to_the_precise_classifiers():
    # Guards the raise site: the loose substring test must be gone, the config clash is checked before MemoryError, and the OOM branch keeps 'Original error:'.
    source = inspect.getsource(vllm_utils.load_vllm)
    assert '"alloc" in error.lower()' not in source
    assert not re.search(r'\(\s*"memory" in error\.lower\(\)', source)

    config_at = source.index("_is_expandable_segments_error(error)")
    oom_at = source.index("_is_out_of_memory_error(error)")
    memory_error_at = source.index("raise MemoryError(")
    assert config_at < memory_error_at
    assert oom_at < memory_error_at

    oom_block = source[memory_error_at:]
    assert "Your GPU ran out of memory loading vLLM with standby mode enabled" in oom_block
    assert "Original error:" in oom_block
