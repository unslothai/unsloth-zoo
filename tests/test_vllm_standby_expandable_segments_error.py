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

"""Regression tests for vLLM load-failure classification with standby mode.

`load_vllm` used to classify a failed vLLM load as an out of memory condition
with `("memory" in error.lower() or "alloc" in error.lower())`. The bare
`"alloc"` substring also matches the deterministic *configuration* assertion
raised by `CuMemAllocator.__init__`:

    Standby mode is not supported with expandable segments.
    Please set environment variable PYTORCH_CUDA_ALLOC_CONF without
    `expandable_segments:True`.

because that text contains `PYTORCH_CUDA_ALLOC_CONF`. Upstream vLLM's own
wording ("Expandable segments are not compatible with memory pool.") collides
with the `"memory"` half instead. Either way the user was told their GPU ran
out of memory and offered three fixes, two of which cannot possibly help. This
was reproduced on a 183 GB B200 with roughly 180 GB free.

A third failure is mislabelled the same way. vLLM asserts that free VRAM went
*down* across its profiling forward pass, and raises

    Error in memory profiling. Initial free memory 11.87 GiB, current free
    memory 138.8 GiB. This happens when other processes sharing the same
    container release GPU memory while vLLM is profiling during initialization.

when another tenant on a shared GPU frees memory mid-profile. Free memory went
*up*, so nothing ran out, but "free memory" is itself a genuine OOM marker
(vLLM's real shortfall reads "Free memory on device ... is less than desired
GPU memory utilization"), so this was reported as an OOM too. It is transient
and clears on a plain retry. Seen twice on a shared 8x B200, 178.4 GB per GPU.

These tests pin the classification (config clash vs profiling race vs real
OOM), the wording of the new messages, and the wiring inside `load_vllm`.
"""

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

# Verbatim from a shared 8x B200, 178.4 GB per GPU, GRPO notebook. Raised by
# the assert in vllm/v1/worker/gpu_worker.py determine_available_memory().
_PROFILING_RACE = (
    "Error in memory profiling. Initial free memory 11.87 GiB, current free memory 138.8 GiB. "
    "This happens when other processes sharing the same container release GPU memory while vLLM "
    "is profiling during initialization. To fix this, ensure consistent GPU memory allocation or "
    "isolate vLLM in its own container."
)

# The XPU worker asserts with the same opening phrase and a different tail
# (vllm/v1/worker/xpu_worker.py), so the marker must not depend on the tail.
_PROFILING_RACE_XPU = (
    "Error in memory profiling. Initial free memory 12746588160, current free memory 149023916032. "
    "This happens when the GPU memory was not properly cleaned up before initializing the vLLM "
    "instance."
)

# vLLM's real KV cache exhaustion (vllm/v1/core/kv_cache_utils.py). Unlike the
# profiling race this is deterministic and gpu_memory_utilization is the fix.
_KV_CACHE_OOM = (
    "To serve at least one request with the models's max seq len (8192), (18.00 GiB KV cache is "
    "needed, which is larger than the available KV cache memory (4.00 GiB). Based on the available "
    "memory, the estimated maximum model length is 1808. Try increasing `gpu_memory_utilization` "
    "or decreasing `max_model_len` when initializing the engine."
)

_REAL_OOM_ERRORS = (
    # torch. Captured verbatim from torch 2.9.1+cu128. Note it recommends
    # expandable segments, so classification must not key off that phrase alone.
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
    # The bug: this returned True via the bare "alloc" substring (and via
    # "memory" for the upstream wording), so a config clash was raised as
    # MemoryError("Your GPU ran out of memory").
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
    # torch's genuine OOM text ends with "try setting
    # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation",
    # so the config-clash markers must be full phrases, not a bare substring,
    # or a real OOM would be routed to the configuration message.
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
    # PYTORCH_HIP_ALLOC_CONF is the ROCm/AMD one, PYTORCH_ALLOC_CONF the unified
    # torch >= 2.10 one. Whichever actually carries the setting must be named.
    cleared = {k: "" for k in vllm_utils._ALLOC_CONF_ENV_VARS}
    cleared[env_var] = "expandable_segments:True,roundup_power2_divisions:[32:256]"
    with mock.patch.dict(vllm_utils.os.environ, cleared, clear = False):
        assert vllm_utils._expandable_segments_env_vars() == [env_var]
        message = vllm_utils._expandable_segments_standby_message(_unsloth_assertion(env_var))
    assert env_var in message
    # Never claim the GPU ran out of memory for a config clash.
    assert "ran out of memory" not in message
    assert "NOT an out of memory error" in message
    # The remedy that actually works must be present, the useless ones absent.
    assert "expandable_segments" in message
    assert "load_in_4bit" not in message
    assert "gpu_memory_utilization=0.6" not in message
    # Preserve the underlying error for debugging.
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
    # The bug: "free memory" appears twice here, so the OOM markers matched and
    # a 178 GB GPU was reported out of memory with 138.8 GiB of it free.
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
    # Dropping the "free memory" marker is the easy way to stop matching the
    # profiling race, and it would silently lose this genuine shortfall, whose
    # text never says "out of memory" (vllm/v1/worker/utils.py request_memory).
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
    # Name the actual cause: a shared GPU, and a transient one.
    assert "transient" in message
    assert "sharing this GPU" in message
    assert "NOT an out of memory error" in message
    assert "ran out of memory" not in message
    # Say it can simply be retried.
    assert "Load the model again" in message
    assert "Unsloth already tried loading 3 times." in message
    # None of the three OOM remedies may be *offered*. Scope this to the list of
    # fixes, since the prose above it names them precisely to rule them out.
    remedies = message[message.index("Try one of these fixes:"):message.index("Original error:")]
    for useless in ("load_in_4bit", "smaller model", "4bit", "Lower gpu_memory_utilization"):
        assert useless not in remedies
    assert "gpu_memory_utilization" not in remedies
    # Preserve the underlying error for debugging.
    assert "Original error:" in message
    assert _PROFILING_RACE in message


def test_profiling_race_message_explains_why_standby_cannot_retry_in_process():
    # Standby keeps weights in CuMemAllocator, a process wide singleton, and
    # Unsloth runs the engine in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0), so
    # the retry has to be a fresh process. Say so instead of staying silent.
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
    # The generic branch below reacts to "memory" by scaling gpu_memory_utilization
    # by 0.85 and max_num_seqs by 0.75 for the rest of the run. Nothing about
    # either caused a profiling race, so the retry must be of the identical load.
    source = inspect.getsource(vllm_utils.load_vllm)
    race_at = source.index("_is_memory_profiling_race_error(error)")
    standby_at = source.index("if trials >= 2 or unsloth_vllm_standby:")
    shrink_at = source.index('engine_args["gpu_memory_utilization"] *= 0.85')
    # Checked before the standby short-circuit, which would otherwise give up
    # after a single attempt on a transient fault.
    assert race_at < standby_at < shrink_at
    # And it retries rather than falling through to the shrink.
    assert "continue" in source[race_at:standby_at]
    assert "_MEMORY_PROFILING_RACE_MAX_TRIALS" in source[race_at:standby_at]
    assert vllm_utils._MEMORY_PROFILING_RACE_MAX_TRIALS >= 2


def test_profiling_race_retries_do_not_spend_the_generic_retry_budget():
    # A race is counted on its own `race_trials`. If it shared `trials` with the
    # generic handler, a race on the first attempt followed by a genuine OOM on
    # the second would hit `trials >= 2` and raise, skipping the shrink-and-retry
    # that a first-attempt OOM gets today.
    source = inspect.getsource(vllm_utils.load_vllm)
    race_at = source.index("_is_memory_profiling_race_error(error)")
    bump_at = source.index("            trials += 1")
    standby_at = source.index("if trials >= 2 or unsloth_vllm_standby:")
    # The generic counter is bumped only after the race branch has passed on it.
    assert race_at < bump_at < standby_at
    assert source.count("            trials += 1") == 1
    # The race branch uses its own counter, never the generic one.
    race_branch = source[race_at:bump_at]
    assert "race_trials += 1" in race_branch
    assert "trials <" not in race_branch.replace("race_trials <", "")


def test_load_vllm_retry_handler_is_wired_to_the_precise_classifiers():
    # Guards the raise site itself: the loose substring test must be gone, the
    # config clash must be checked before the MemoryError, and the OOM branch
    # must keep `Original error:`.
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
