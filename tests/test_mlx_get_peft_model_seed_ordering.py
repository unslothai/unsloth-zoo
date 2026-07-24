# Unsloth Zoo - Utilities for Unsloth
# Pin the seed-immediately-before-linear_to_lora_layers ordering in
# FastMLXModel.get_peft_model so we match mlx-lm CLI's basin family.
# The numerical test (bit-identical lora_a) needs real MLX and is
# covered by probe 39 on danielhanchen/unsloth-staging-2; locally we
# guard against a refactor moving the seed call far from the LoRA init.

from __future__ import annotations

import inspect
import re

import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_mlx_shim():
    try:
        import mlx  # noqa: F401
    except ImportError:
        from mlx_simulation import simulate_mlx_on_torch
        simulate_mlx_on_torch()


def _get_peft_model_source():
    from unsloth_zoo.mlx.loader import FastMLXModel
    return inspect.getsource(FastMLXModel.get_peft_model)


def test_seed_immediately_precedes_each_linear_to_lora_layers_call():
    """Every `linear_to_lora_layers(...)` in get_peft_model must be
    preceded within ~20 lines by `_seed_mlx_random_state`.

    Empirically (probe 39 on Apple Silicon), a far-away seed lets lazy
    MLX state advances slip in before lora_a init, diverging from mlx-lm
    CLI; fix is to seed immediately above each LoRA construction.
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
    """Only comment lines may sit between the seed call and the
    linear_to_lora_layers call; anything else risks an mx.eval/random op
    slipping in later. Enforced as a regex tripwire."""
    src = _get_peft_model_source()
    pattern = re.compile(
        r"_seed_mlx_random_state\(random_state\)\s*"
        r"(?:\n\s*#[^\n]*)*"
        r"\s*\n\s*(?:[A-Za-z_][A-Za-z0-9_]*(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]*)*\s*=\s*)?"
        r"linear_to_lora_layers\(",
    )
    matches = pattern.findall(src)
    assert len(matches) >= 2, (
        "Expected at least two seed+LoRA-call tight pairings in "
        "get_peft_model (one for VLM language LoRA, one for text). "
        f"Found {len(matches)}. The seed call must immediately precede "
        "each linear_to_lora_layers invocation with only comment lines "
        "in between."
    )
