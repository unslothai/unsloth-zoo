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

"""Tests for the per-machine Xet verdict.

No test here touches the network: ``probe = False`` exercises the local decision path, and the one
test that needs a probe result stubs ``_probe_cas_reachable``.
"""

from __future__ import annotations

import dataclasses
import json
import time

import pytest

from unsloth_zoo import hf_xet_health as health
from unsloth_zoo import hf_xet_tuning as tuning

GB = 1_000_000_000

# Grabbed at import, before the autouse fixture stubs it out.
_UNSTUBBED_PROBE = health._probe_cas_reachable


@pytest.fixture(autouse = True)
def _clean(monkeypatch, tmp_path):
    """Every test starts with no verdict and a reachable-looking probe."""
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    for var in ("UNSLOTH_DISABLE_XET", "UNSLOTH_STABLE_DOWNLOADS", "HF_HUB_DISABLE_XET",
                "UNSLOTH_FORCE_XET", "HF_ENDPOINT"):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setattr(health, "_hf_xet_version", lambda: "9.9.9-test")
    monkeypatch.setattr(health, "_probe_cas_reachable", lambda: (True, "probe ok"))
    health.clear_xet_health()
    yield
    health.clear_xet_health()


def _big_machine(monkeypatch, ram_gb = 64):
    monkeypatch.setattr(
        health, "system_profile",
        lambda: tuning.SystemProfile(int(ram_gb * GB), int(ram_gb * GB), 8, "test", "test"),
    )


def test_defaults_to_xet_with_no_verdict(monkeypatch):
    _big_machine(monkeypatch)
    result = health.xet_health(probe = False)
    assert result.use_xet is True
    assert result.source == "default"


def test_missing_hf_xet_means_http(monkeypatch):
    _big_machine(monkeypatch)
    monkeypatch.setattr(health, "_hf_xet_version", lambda: None)
    result = health.xet_health(probe = False)
    assert result.use_xet is False
    assert result.source == "unavailable"


def test_low_ram_machine_skips_xet(monkeypatch):
    """Below the floor, Xet's smallest working set is still the wrong trade."""
    _big_machine(monkeypatch, ram_gb = 2)
    result = health.xet_health(probe = False)
    assert result.use_xet is False
    assert result.source == "low-ram"
    assert "2.0GB" in result.reason


def test_env_override_forces_http(monkeypatch):
    _big_machine(monkeypatch)
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "1")
    assert health.xet_health().use_xet is False


def test_env_override_forces_xet_even_on_a_demoted_machine(monkeypatch):
    """An explicit UNSLOTH_FORCE_XET must beat a remembered demotion, or a user cannot re-test."""
    _big_machine(monkeypatch)
    health.record_xet_outcome(False, "stall")
    health.record_xet_outcome(False, "stall")
    assert health.xet_health(probe = False).use_xet is False
    monkeypatch.setenv("UNSLOTH_FORCE_XET", "1")
    assert health.xet_health().use_xet is True


def test_one_failure_does_not_demote(monkeypatch):
    """A single dropped connection fails HTTP too; only a pattern should switch transports."""
    _big_machine(monkeypatch)
    health.record_xet_outcome(False, "one bad download")
    assert health.xet_health(probe = False).use_xet is True


def test_two_failures_demote_and_a_success_recovers(monkeypatch):
    _big_machine(monkeypatch)
    health.record_xet_outcome(False, "Xet stalled")
    health.record_xet_outcome(False, "Xet stalled")
    demoted = health.xet_health(probe = False)
    assert demoted.use_xet is False
    assert demoted.source == "cached"
    assert "stalled" in demoted.reason

    health.record_xet_outcome(True, "recovered")
    assert health.xet_health(probe = False).use_xet is True


def test_verdict_expires(monkeypatch):
    _big_machine(monkeypatch)
    health.record_xet_outcome(False, "x")
    health.record_xet_outcome(False, "x")
    assert health.xet_health(probe = False).use_xet is False

    state = json.loads(health.health_state_path().read_text())
    state["ts"] = time.time() - (health.DEMOTED_TTL_SECONDS + 60)
    health.health_state_path().write_text(json.dumps(state))
    health._CACHED = None
    # Expired: the machine gets another chance rather than being demoted forever.
    assert health.xet_health(probe = False).source == "default"


@pytest.mark.parametrize("field, value", [("hf_xet_version", "0.0.1"), ("endpoint", "https://x")])
def test_verdict_is_scoped_to_version_and_endpoint(monkeypatch, field, value):
    """A verdict measured against a different hf_xet build or Hub endpoint says nothing here."""
    _big_machine(monkeypatch)
    health.record_xet_outcome(False, "x")
    health.record_xet_outcome(False, "x")
    assert health.xet_health(probe = False).use_xet is False

    state = json.loads(health.health_state_path().read_text())
    state[field] = value
    health.health_state_path().write_text(json.dumps(state))
    health._CACHED = None
    assert health.xet_health(probe = False).source == "default"


def test_failure_streak_resets_across_hf_xet_versions(monkeypatch):
    """Two failures on an old build must not demote a freshly upgraded one."""
    _big_machine(monkeypatch)
    health.record_xet_outcome(False, "old build")
    monkeypatch.setattr(health, "_hf_xet_version", lambda: "10.0.0-new")
    health.record_xet_outcome(False, "new build")
    state = json.loads(health.health_state_path().read_text())
    assert state["consecutive_failures"] == 1
    assert state["verdict"] == "xet"


def test_probe_result_is_persisted(monkeypatch):
    _big_machine(monkeypatch)
    monkeypatch.setattr(health, "_probe_cas_reachable", lambda: (False, "Xet CAS unreachable"))
    result = health.xet_health()
    assert result.use_xet is False
    assert result.source == "probe"
    assert json.loads(health.health_state_path().read_text())["verdict"] == "http"


def test_unwritable_state_dir_still_answers(monkeypatch):
    """A read-only HF_HOME must not stop downloads; it only costs the memory of the verdict."""
    _big_machine(monkeypatch)
    monkeypatch.setattr(health, "health_state_path", lambda: None)
    health._CACHED = None
    assert health.xet_health(probe = False).use_xet is True
    health.record_xet_outcome(False, "x")  # must not raise


def test_corrupt_state_file_is_ignored(monkeypatch):
    _big_machine(monkeypatch)
    health.health_state_path().write_text("{not json")
    health._CACHED = None
    assert health.xet_health(probe = False).use_xet is True


def test_result_is_truthy_like_a_bool(monkeypatch):
    _big_machine(monkeypatch)
    assert bool(health.XetHealth(True, "", "test")) is True
    assert bool(health.XetHealth(False, "", "test")) is False


def test_probe_404_is_inconclusive_not_a_demotion(monkeypatch):
    """An HF_ENDPOINT mirror that does not host the probe repo answers 404.

    That is proof the endpoint is REACHABLE, which is the only thing this probe measures. Treating
    it as "Xet unreachable" pinned every on-prem or mirror user to HTTP for 24h.
    """
    import urllib.error
    import urllib.request

    _big_machine(monkeypatch)
    monkeypatch.setattr(health, "_probe_cas_reachable", _UNSTUBBED_PROBE)

    def _raise(*args, **kwargs):
        raise urllib.error.HTTPError("http://mirror/x", 404, "Not Found", None, None)

    monkeypatch.setattr(urllib.request, "urlopen", _raise)

    ok, reason = health._probe_cas_reachable()
    assert ok is None, reason

    verdict = health.xet_health(force = True, probe = True)
    assert verdict.use_xet is True
    assert verdict.source == "default"
    assert not health.health_state_path().exists(), "an inconclusive probe must persist nothing"


def test_probe_403_still_demotes(monkeypatch):
    """A blocking corporate proxy legitimately answers 403, and that machine should use HTTP."""
    import urllib.error
    import urllib.request

    _big_machine(monkeypatch)
    monkeypatch.setattr(health, "_probe_cas_reachable", _UNSTUBBED_PROBE)

    def _raise(*args, **kwargs):
        raise urllib.error.HTTPError("http://hf/x", 403, "Forbidden", None, None)

    monkeypatch.setattr(urllib.request, "urlopen", _raise)

    ok, reason = health._probe_cas_reachable()
    assert ok is False
    assert "403" in reason
