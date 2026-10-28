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

from __future__ import annotations

import importlib.util
import os
import pathlib

import pytest

# Probe is a runnable script, not a package module: load by path; it imports only the stdlib
_PROBE = pathlib.Path(__file__).with_name("gemma4_audio_version_probe.py")
_spec = importlib.util.spec_from_file_location("gemma4_audio_version_probe", _PROBE)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


@pytest.mark.parametrize("var", ["UNSLOTH_ALLOW_CPU", "UNSLOTH_IS_PRESENT"])
def test_importing_the_probe_does_not_rewrite_the_environment(monkeypatch, var):
    """Probe import must not set env vars: conftest sets UNSLOTH_ALLOW_CPU and unsloth/__init__.py sets
    UNSLOTH_IS_PRESENT; os.environ.setdefault writes through to the real process env, so clear them and re-execute"""
    monkeypatch.delenv(var, raising=False)
    _spec.loader.exec_module(importlib.util.module_from_spec(_spec))
    assert var not in os.environ


@pytest.mark.parametrize(
    "name, repeats, other, expected",
    [
        # Reproducible machine: repeats agree, so the noise floor is zero.
        ("reproducible_and_connected", [19.3557] * 3, 19.6828, "pass"),
        ("reproducible_and_disconnected", [19.3557] * 3, 19.3557, "same_loss"),
        # Apple Silicon (macos-14, mlx-vlm 0.6.3): identical 440 Hz repeats spread 0.827; 1760 Hz gap 1.30
        ("measured_on_metal", [24.8986, 25.4577, 25.7256], 26.1988,
         "below_noise"),
        # Same machine, audio connected: a gap dwarfing the spread still decides.
        ("drifting_but_signal_dwarfs_it", [24.8986, 25.4577, 25.7256], 40.0,
         "pass"),
        # Two repeats can agree by luck on a drifting machine, zeroing the noise floor; a third stops that.
        ("two_repeats_agree_by_luck", [19.30, 19.30, 19.81], 19.62,
         "below_noise"),
    ],
)
def test_two_tone_verdict(name, repeats, other, expected):
    verdict, detail = probe.two_tone_verdict(repeats, other)
    assert verdict == expected, f"{name}: {detail}"
    # Every non-pass verdict needs a TWO_TONE_REASONS message, else the caller raises KeyError
    if verdict != "pass":
        assert probe.TWO_TONE_REASONS[verdict]


def test_a_gap_exactly_at_the_margin_counts_as_not_clear_of_it():
    """Boundary built from the gate's arithmetic so it lands on <=; hardcoding it tests float rounding"""
    repeats = [19.0, 19.0, 19.5]
    at_margin = (sum(repeats) / len(repeats)
                 + probe.TWO_TONE_MARGIN * (max(repeats) - min(repeats)))
    assert probe.two_tone_verdict(repeats, at_margin)[0] == "below_noise"


def test_detail_records_the_repeats_not_just_their_spread():
    """A spread cannot tell jitter from a drift accumulating across forwards."""
    _, detail = probe.two_tone_verdict([19.30, 19.55, 19.81], 27.4)
    assert "repeats=[19.3000, 19.5500, 19.8100]" in detail
    assert "signal=" in detail and "noise=" in detail


def test_the_reported_tone_loss_is_the_mean_of_the_repeats():
    """Three forwards are paid for, so the figure printed uses all three."""
    _, detail = probe.two_tone_verdict([19.0, 20.0, 21.0], 30.0)
    assert "440Hz=20.0000" in detail


def test_an_unmeasurable_stage_does_not_fail_the_version():
    """Stage 4 is inconclusive on Metal: repeat spread exceeds the tone gap; gating fails all versions"""
    stages = {
        "0_mlx_vlm_alone_loads": {"ok": False},
        "1_zoo_loads_it": {"ok": True},
        "4_audio_reaches_the_loss": {"ok": False, "inconclusive": True},
    }
    gating = probe.gating_stages(stages)
    assert set(gating) == {"1_zoo_loads_it"}
    assert all(s["ok"] for s in gating.values())


def test_a_real_stage_failure_still_fails_the_version():
    stages = {
        "1_zoo_loads_it": {"ok": True},
        "3_placeholders_match_the_audio_tower": {"ok": False},
    }
    gating = probe.gating_stages(stages)
    assert not all(s["ok"] for s in gating.values())
