# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Regression for unslothai/unsloth#5441.

PreTrainedModel.__init__ resolves loss_type from the class name. Anything
whose name doesn't appear as a literal LOSS_MAPPING key falls back to a
regex match, which means `Qwen3_5ForConditionalGeneration` lands on
`LOSS_MAPPING["ForConditionalGeneration"]`. That entry, plus
`CsmForConditionalGeneration`, is aliased to the stock `ForCausalLMLoss`
in transformers. `patch_loss_functions()` only rewrote
`LOSS_MAPPING["ForCausalLM"]`, leaving those aliases pointing at the
un-patched loss which does `logits.float()` and OOMs on <= 24 GB GPUs at
large vocab sizes.

This suite pins:
  - Every key originally aliased to ForCausalLMLoss is replaced with
    the Unsloth kernel.
  - Keys aliased to other loss types (ForMaskedLMLoss, segmentation,
    detection, etc.) are not overwritten.
  - The patch is idempotent.
"""

from __future__ import annotations

import pytest


def _restore(mapping, saved):
    mapping.clear()
    mapping.update(saved)


def test_loss_mapping_for_conditional_generation_patched():
    lu = pytest.importorskip("transformers.loss.loss_utils")
    from unsloth_zoo import loss_utils as zoo_loss
    from unsloth_zoo.fused_losses import unsloth_fused_ce_loss  # noqa: F401

    saved = dict(lu.LOSS_MAPPING)
    try:
        # A naive cross_entropy stub keeps torch.compile out of the picture and
        # makes the regression test pure-Python.
        def _fast_ce(logits, labels, n_items=None, **kw):
            import torch
            return torch.nn.functional.cross_entropy(
                logits.float(), labels, ignore_index=-100,
            )
        zoo_loss.patch_loss_functions(_fast_ce, torch_compile=False)

        forcausal = lu.LOSS_MAPPING.get("ForCausalLM")
        assert forcausal is not None
        assert getattr(forcausal, "__name__", "") != "ForCausalLMLoss", (
            f"LOSS_MAPPING['ForCausalLM'] was not replaced: {forcausal}"
        )

        cg = lu.LOSS_MAPPING.get("ForConditionalGeneration")
        assert cg is forcausal, (
            f"LOSS_MAPPING['ForConditionalGeneration'] still aliases the stock "
            f"ForCausalLMLoss; Qwen3_5ForConditionalGeneration would OOM via "
            f"logits.float(). got: {cg}"
        )
    finally:
        _restore(lu.LOSS_MAPPING, saved)


def test_loss_mapping_other_losses_left_alone():
    lu = pytest.importorskip("transformers.loss.loss_utils")
    from unsloth_zoo import loss_utils as zoo_loss

    # Keys not currently aliased to ForCausalLMLoss must survive the sweep.
    non_causal = {
        k: v for k, v in lu.LOSS_MAPPING.items()
        if getattr(v, "__name__", "") != "ForCausalLMLoss"
    }
    saved = dict(lu.LOSS_MAPPING)
    try:
        def _fast_ce(logits, labels, n_items=None, **kw):
            import torch
            return torch.nn.functional.cross_entropy(logits.float(), labels, ignore_index=-100)
        zoo_loss.patch_loss_functions(_fast_ce, torch_compile=False)

        unsloth_loss = lu.LOSS_MAPPING["ForCausalLM"]
        for key, original_fn in non_causal.items():
            assert lu.LOSS_MAPPING[key] is original_fn, (
                f"LOSS_MAPPING['{key}'] was overwritten by the sweep; "
                f"expected {original_fn}, got {lu.LOSS_MAPPING[key]}"
            )
            assert lu.LOSS_MAPPING[key] is not unsloth_loss, (
                f"LOSS_MAPPING['{key}'] incorrectly replaced with the Unsloth kernel."
            )
    finally:
        _restore(lu.LOSS_MAPPING, saved)


def test_loss_mapping_sweep_idempotent():
    lu = pytest.importorskip("transformers.loss.loss_utils")
    from unsloth_zoo import loss_utils as zoo_loss

    saved = dict(lu.LOSS_MAPPING)
    try:
        def _fast_ce(logits, labels, n_items=None, **kw):
            import torch
            return torch.nn.functional.cross_entropy(logits.float(), labels, ignore_index=-100)
        zoo_loss.patch_loss_functions(_fast_ce, torch_compile=False)
        first = dict(lu.LOSS_MAPPING)
        zoo_loss.patch_loss_functions(_fast_ce, torch_compile=False)
        second = dict(lu.LOSS_MAPPING)
        for k in first:
            assert first[k] is second[k], (
                f"LOSS_MAPPING['{k}'] mutated on second patch_loss_functions call."
            )
    finally:
        _restore(lu.LOSS_MAPPING, saved)
