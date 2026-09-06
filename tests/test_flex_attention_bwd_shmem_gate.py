# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""The shared-memory gate in `temporary_patches.flex_attention_bwd`.

The patch appends smaller backward configs when the ones Inductor picked cannot fit
in shared memory. Everything here turns on the estimate that decides WHETHER it
fires, and the failure mode is one-sided: an over-estimate offers a few candidates
the autotuner discards, while an under-estimate means the gate stays shut, the
offending config is handed back untouched, and the compile dies with "No valid
triton configs".

It stayed shut on exactly the GPUs this patch was written for. The original formula
scales with num_stages, and it was fitted on an sm8x part whose head_dim >= 128
entry uses num_stages=2. sm10x, sm11x and sm12x all use num_stages=1 there, which
halves the estimate: on GB10, head_dim 256 bf16 block 64 was scored 75,776 against a
102,400 limit and passed, while Triton reported `Required: 114688 Hardware limit:
101376` and refused it.

Measured, by forcing single configs on GB10 (sm_121, torch 2.14, Triton 3.8) and
reading Triton's own "Required:": block 64 -> 114,688, block 128 -> 229,376, and
IDENTICAL for num_stages 1 and 2.

GPU-free: the arithmetic is exercised directly.
"""

from __future__ import annotations

import pytest


flex = pytest.importorskip("unsloth_zoo.temporary_patches.flex_attention_bwd")
torch = pytest.importorskip("torch")


def _staged(block, stages, head_dim, dtype_size):
    """The formula this module shipped with, kept here as the floor it must not go under."""
    return stages * block * head_dim * dtype_size * 2 + flex._SHMEM_OVERHEAD


class TestTheEstimateIsNeverLowerThanItWas:
    """Whatever else changes, no GPU may stop being covered."""

    @pytest.mark.parametrize("block", [16, 32, 64, 128])
    @pytest.mark.parametrize("stages", [1, 2, 3])
    @pytest.mark.parametrize("head_dim", [64, 128, 192, 256])
    def test_it_never_drops_below_the_original_formula(self, block, stages, head_dim):
        got = flex._estimate_shmem(block, stages, head_dim, torch.bfloat16)
        assert got >= _staged(block, stages, head_dim, 2), (
            "an estimate below the original one would close the gate on a GPU where it "
            "fires today, which is a silent failure to compile"
        )


class TestTheMeasuredRequirementIsCovered:
    """The numbers Triton itself reported on GB10, head_dim 256 bf16."""

    # (block, num_stages, what Triton said it Required)
    MEASURED = [(64, 1, 114688), (64, 2, 114688), (128, 1, 229376)]

    @pytest.mark.parametrize("block, stages, required", MEASURED)
    def test_the_estimate_reaches_the_real_requirement(self, block, stages, required):
        got = flex._estimate_shmem(block, stages, 256, torch.bfloat16)
        assert got >= required, (
            f"block {block}/stages {stages} really needs {required}; estimating {got} "
            "leaves the gate shut and the config is returned as-is"
        )

    def test_the_stage_one_case_that_slipped_through(self):
        """The exact config GB10 was handed, against the limit Triton enforces."""
        assert flex._estimate_shmem(64, 1, 256, torch.bfloat16) > 101376
        # And the old formula, which is why it slipped: 1 * 64 * 256 * 2 * 2 + 10240.
        assert _staged(64, 1, 256, 2) == 75776

    def test_a_head_dim_that_fits_still_fits(self):
        """head_dim 128 compiles on GB10 today, so the gate must stay shut for it."""
        assert flex._estimate_shmem(64, 1, 128, torch.bfloat16) <= 101376

    def test_a_large_shared_memory_gpu_is_untouched(self):
        """A100 opt-in is 166,912, so nothing here fires and autotune time is unchanged."""
        assert flex._estimate_shmem(64, 2, 256, torch.bfloat16) <= 166912


class TestTheLimitIsTheOneTritonEnforces:
    def test_the_smaller_of_the_two_is_taken(self, monkeypatch):
        class Props:
            shared_memory_per_block_optin = 101376
            shared_memory_per_multiprocessor = 102400

        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _d: Props())
        assert flex._shmem_limit(0) == 101376, (
            "Triton refuses on the per-block opt-in maximum; the 1KB above it is enough "
            "to bless a config it then rejects"
        )

    def test_an_older_torch_without_the_optin_attribute_still_answers(self, monkeypatch):
        class Props:
            shared_memory_per_multiprocessor = 102400

        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _d: Props())
        assert flex._shmem_limit(0) == 102400

    def test_no_usable_attribute_reports_zero_rather_than_guessing(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _d: object())
        assert flex._shmem_limit(0) == 0


class TestTheFallbackLadder:
    """The estimate decides whether to intervene; it does not also decide what survives.

    Inductor catches OutOfResources per candidate and keeps the ones that compile, so
    offering a config that does not fit costs one discarded candidate.
    """

    class FakeConfig:
        def __init__(self, bm1, bn1, bm2, bn2, num_stages, num_warps):
            self.block_m1, self.block_n1 = bm1, bn1
            self.block_m2, self.block_n2 = bm2, bn2
            self.num_stages, self.num_warps = num_stages, num_warps

        def __eq__(self, other):
            return vars(self) == vars(other)

    def _blocks(self, configs):
        return sorted({c.block_m1 for c in configs}, reverse = True)

    def test_every_block_below_the_offending_one_is_offered(self):
        configs = flex._generate_safe_configs(
            self.FakeConfig, 101376, 256, torch.bfloat16, worst_block = 64,
        )
        assert self._blocks(configs) == [32, 16], (
            "the offending block was 64, so 32 and 16 are the candidates that remain"
        )

    def test_a_larger_offending_block_yields_a_longer_ladder(self):
        configs = flex._generate_safe_configs(
            self.FakeConfig, 101376, 256, torch.bfloat16, worst_block = 128,
        )
        assert self._blocks(configs) == [64, 32, 16]

    def test_the_smallest_fallback_is_always_present(self):
        for worst in (0, 32, 64, 128):
            configs = flex._generate_safe_configs(
                self.FakeConfig, 101376, 256, torch.bfloat16, worst_block = worst,
            )
            assert any(c.block_m1 == 16 and c.num_stages == 1 for c in configs), (
                f"worst_block={worst} produced no 16/1 fallback"
            )


class TestThePatchIsSafeToApplyTwice:
    def test_a_second_call_does_not_wrap_the_wrapper(self):
        if not torch.cuda.is_available():
            pytest.skip("the patch returns early without CUDA")
        from torch._inductor.template_heuristics import triton as th  # noqa: PLC0415

        flex.patch_flex_attention_bwd_configs()
        first = th.CUDAConfigHeuristic().get_flex_attn_bwd_configs(256, torch.bfloat16)
        flex.patch_flex_attention_bwd_configs()
        second = th.CUDAConfigHeuristic().get_flex_attn_bwd_configs(256, torch.bfloat16)
        assert len(first) == len(second), (
            "the second call wrapped the wrapper, so the ladder is being appended twice"
        )
