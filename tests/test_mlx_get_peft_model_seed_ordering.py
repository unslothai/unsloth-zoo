# Unsloth Zoo - Utilities for Unsloth
# Pin the seed-immediately-before-linear_to_lora_layers ordering in
# FastMLXModel.get_peft_model so we match mlx-lm CLI's basin family.
#
# The actual numerical test (lora_a values bit-identical to
# `mlx_lm.tuner.utils.linear_to_lora_layers` after `mx.random.seed`)
# needs real MLX runtime; that's covered by probe 39 on
# danielhanchen/unsloth-staging-2. Locally we guard against a future
# refactor reverting to a single `_seed_mlx_random_state` call far
# above `linear_to_lora_layers` again.

from __future__ import annotations

import inspect
import re

import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_mlx_shim():
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()


def _get_peft_model_source():
    from unsloth_zoo.mlx.loader import FastMLXModel
    return inspect.getsource(FastMLXModel.get_peft_model)


def test_seed_immediately_precedes_each_linear_to_lora_layers_call():
    """Every `linear_to_lora_layers(...)` call inside get_peft_model
    must be preceded -- within ~20 source lines, with no other MLX op
    or model-tree walk in between -- by a `_seed_mlx_random_state` call.

    Background: empirically (Round BQ probe 39 on Apple Silicon), having
    the seed call >100 lines above the LoRA construction allows lazy MLX
    state advances to slip in between seeding and lora_a init, producing
    lora_a matrices different from mlx-lm CLI's despite both paths
    re-seeding to the same int. The fix is to seed immediately above
    each `linear_to_lora_layers` call.
    """
    src = _get_peft_model_source()
    lines = src.splitlines()

    call_lines = [
        i for i, line in enumerate(lines)
        if "linear_to_lora_layers(" in line and not line.strip().startswith("#")
    ]
    assert call_lines, "expected at least one linear_to_lora_layers call in get_peft_model"

    for call_idx in call_lines:
        window_start = max(0, call_idx - 20)
        window = "\n".join(lines[window_start:call_idx])
        assert "_seed_mlx_random_state" in window, (
            f"linear_to_lora_layers at line {call_idx+1} of get_peft_model "
            f"is not preceded by `_seed_mlx_random_state` within the prior "
            f"20 lines. Move the seed call closer to the LoRA construction "
            f"to keep mlx-lm CLI parity."
        )


def test_get_peft_model_still_accepts_random_state_arg():
    """Ensure the API surface didn't change while moving the seed."""
    import inspect
    from unsloth_zoo.mlx.loader import FastMLXModel
    sig = inspect.signature(FastMLXModel.get_peft_model)
    assert "random_state" in sig.parameters
    assert sig.parameters["random_state"].default == 3407


def test_no_other_mlx_op_between_seed_and_linear_to_lora_layers():
    """Between the seed-just-above-LoRA call and the linear_to_lora_layers
    call itself, the only allowed code is the LoRA call. Anything else --
    even pure Python -- is a tripwire because it makes it easy to slip
    in an mx.eval/random call later. Enforced as a regex tripwire."""
    src = _get_peft_model_source()
    pattern = re.compile(
        r"_seed_mlx_random_state\(random_state\)\s*"
        r"(?:\n\s*#[^\n]*)*"
        r"\s*\n\s*linear_to_lora_layers\(",
    )
    matches = pattern.findall(src)
    assert len(matches) >= 2, (
        "Expected at least two seed+LoRA-call tight pairings in "
        "get_peft_model (one for VLM language LoRA, one for text). "
        f"Found {len(matches)}. The seed call must immediately precede "
        "each linear_to_lora_layers invocation with only comment lines "
        "in between."
    )
