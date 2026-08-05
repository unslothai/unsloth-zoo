from __future__ import annotations

import importlib.util
import pathlib

import pytest

# The probe is a runnable script rather than a package module, so load it by
# path. It imports only the standard library at module level, which is why
# this costs nothing and needs neither mlx nor a checkpoint.
_PROBE = pathlib.Path(__file__).with_name("gemma4_audio_version_probe.py")
_spec = importlib.util.spec_from_file_location("gemma4_audio_version_probe", _PROBE)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


def test_importing_the_probe_does_not_rewrite_the_environment(monkeypatch):
    """Loading it above must not have set UNSLOTH_* for the whole session."""
    monkeypatch.delenv("UNSLOTH_ALLOW_CPU", raising=False)
    importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(importlib.util.module_from_spec(_spec))
    import os
    assert "UNSLOTH_ALLOW_CPU" not in os.environ


@pytest.mark.parametrize(
    "name, repeats, other, expected",
    [
        # A machine where the model is reproducible: the repeats agree, so the
        # noise floor is zero and the gap only has to be non-zero. This is the
        # behaviour that existed before and must not change.
        ("reproducible_and_connected", [19.3557] * 3, 19.6828, "pass"),
        ("reproducible_and_disconnected", [19.3557] * 3, 19.3557, "same_loss"),
        # Apple Silicon with Gemma 4: identical forwards disagree, so a bare
        # inequality is satisfied by the drift alone. Here the audio is
        # disconnected and the gap is smaller than the model's own jitter.
        ("drifting_and_disconnected", [19.30, 19.55, 19.81], 19.62,
         "below_noise"),
        ("drifting_and_connected", [19.30, 19.55, 19.81], 27.4, "pass"),
        # Two repeats can agree by luck on a drifting machine, which would put
        # the floor back at zero and pass the disconnected case; the third is
        # what stops that.
        ("two_repeats_agree_by_luck", [19.30, 19.30, 19.81], 19.62,
         "below_noise"),
        # Exactly at the margin counts as not clear of it.
        ("exactly_at_the_margin", [19.0, 19.0, 19.5], 21.0, "below_noise"),
    ],
)
def test_two_tone_verdict(name, repeats, other, expected):
    verdict, detail = probe.two_tone_verdict(repeats, other)
    assert verdict == expected, f"{name}: {detail}"
    assert "signal=" in detail and "noise=" in detail


def test_every_failure_verdict_has_a_message():
    """A verdict with no entry would raise KeyError instead of reporting."""
    for verdict in ("same_loss", "below_noise"):
        assert probe.TWO_TONE_REASONS[verdict]
