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


def test_a_success_racing_a_failure_cannot_fabricate_a_demotion(monkeypatch):
    """A success RESETS the streak while a failure INCREMENTS it, so from a streak of 1 no serial
    order reaches the threshold: (fail, ok) ends at 0, (ok, fail) ends at 1. A demotion therefore
    proves a dropped update, costing a full day on HTTP on a machine where Xet works. The
    interleaving is forced rather than raced, so this cannot pass by luck of timing.
    """
    import threading

    health.record_xet_outcome(False, "seed")           # streak = 1
    assert json.loads(health.health_state_path().read_text())["consecutive_failures"] == 1

    real_read = health._read_state
    failure_thread = {"id": None}
    failure_has_read = threading.Event()
    success_done = threading.Event()

    def _instrumented_read():
        state = real_read()
        if threading.get_ident() == failure_thread["id"]:
            failure_has_read.set()
            # Hand the success every chance to slip inside our read-modify-write; under the guard it
            # cannot, so this simply times out.
            success_done.wait(1.0)
        return state

    monkeypatch.setattr(health, "_read_state", _instrumented_read)

    def _failure():
        failure_thread["id"] = threading.get_ident()
        health.record_xet_outcome(False, "concurrent failure")

    def _success():
        assert failure_has_read.wait(5.0)
        health.record_xet_outcome(True, "concurrent success")
        success_done.set()

    threads = [threading.Thread(target = _failure), threading.Thread(target = _success)]
    failure_thread["id"] = None
    threads[0].start()
    threads[1].start()
    for thread in threads:
        thread.join(15.0)
        assert not thread.is_alive()

    final = json.loads(health.health_state_path().read_text())
    assert final["verdict"] == "xet", f"a lost update demoted a healthy machine: {final}"
    assert final["consecutive_failures"] < health.DEMOTION_THRESHOLD, final


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
    """An HF_ENDPOINT mirror that does not host the probe repo answers 404, which proves the endpoint
    is REACHABLE. Treating it as "Xet unreachable" pinned every mirror user to HTTP for 24h."""
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


def test_an_unprobed_memo_does_not_satisfy_an_explicit_probe(monkeypatch):
    """The download path calls xet_health(probe=False) every time; memoizing that optimistic default
    for all callers disarmed the explicit preflight for a minute, on exactly the CAS-blocked machine
    the probe exists to catch."""
    _big_machine(monkeypatch)
    probes: list[bool] = []

    def _blocked():
        probes.append(True)
        return (False, "CAS blocked by corporate proxy")

    monkeypatch.setattr(health, "_probe_cas_reachable", _blocked)

    cheap = health.xet_health(probe = False)
    assert cheap.use_xet is True and cheap.source == "default"
    assert probes == [], "the cheap path must not probe"

    preflight = health.xet_health()          # probe defaults to True
    assert probes == [True], "the explicit preflight was answered from an unprobed memo"
    assert preflight.use_xet is False


def test_a_real_verdict_still_short_circuits_every_caller(monkeypatch):
    """The memo must keep working for real verdicts, or a snapshot pays a probe per file."""
    _big_machine(monkeypatch)
    probes: list[bool] = []

    def _reachable():
        probes.append(True)
        return (True, "Xet CAS reachable")

    monkeypatch.setattr(health, "_probe_cas_reachable", _reachable)

    health.xet_health(force = True)
    assert probes == [True]
    for _ in range(5):
        health.xet_health()
    assert probes == [True], "a real verdict should not be re-probed within the memo window"


def test_a_foreign_nodes_verdict_is_ignored(monkeypatch, tmp_path):
    """HF_HOME is routinely shared across a cluster: without machine scoping, one node with blocked
    CAS demotes every node for 24h, and no node then starts on Xet to record the clearing success."""
    _big_machine(monkeypatch)
    monkeypatch.setattr(health, "_probe_cas_reachable", lambda: (True, "probe ok"))

    health.record_xet_outcome(False, "stall")
    health.record_xet_outcome(False, "stall")
    demoted = health.xet_health(probe = False)
    assert demoted.use_xet is False, "this node should be demoted by its own failures"

    monkeypatch.setattr(health, "_machine_id", lambda: "some-other-node")
    health._CACHED = None
    assert health.xet_health(probe = False).use_xet is True, (
        "a peer node's verdict on a shared cache must not demote this one"
    )


def test_the_probe_is_bounded_by_a_wall_clock(monkeypatch):
    """urlopen's timeout is per operation and does not cover DNS, so the probe needs its own bound."""
    import time as _time

    monkeypatch.setattr(
        health, "_probe_cas_reachable_inner", lambda: (_time.sleep(30), (True, "never"))[1]
    )
    monkeypatch.setattr(health, "PROBE_TIMEOUT_SECONDS", 0.2)

    started = _time.monotonic()
    ok, reason = _UNSTUBBED_PROBE()   # the autouse fixture stubs the module attribute
    elapsed = _time.monotonic() - started

    assert elapsed < 5.0, f"probe ran for {elapsed:.1f}s despite its budget"
    assert ok is None, "an unbounded probe measured nothing, so it must not demote"
    assert "budget" in reason


def test_a_peer_nodes_failures_do_not_demote_this_one(monkeypatch, tmp_path):
    """Shared HF_HOME: scoping only the READ left writes merging streaks across nodes, so a peer's
    single failure demoted a healthy node on its own first failure."""
    _big_machine(monkeypatch)
    monkeypatch.setattr(health, "_probe_cas_reachable", lambda: (True, "probe ok"))

    monkeypatch.setattr(health, "_machine_id", lambda: "node-a")
    health.record_xet_outcome(False, "stall on A")

    monkeypatch.setattr(health, "_machine_id", lambda: "node-b")
    health._CACHED = None
    health.record_xet_outcome(False, "first ever stall on B")
    assert health.xet_health(probe = False).use_xet is True, (
        "node B was demoted on its FIRST failure by inheriting node A's streak"
    )


def test_a_peer_nodes_successes_do_not_rescue_a_broken_node(monkeypatch, tmp_path):
    """The mirror failure: a healthy peer's success kept zeroing a broken node's streak."""
    _big_machine(monkeypatch)
    monkeypatch.setattr(health, "_probe_cas_reachable", lambda: (True, "probe ok"))

    for _ in range(2):
        monkeypatch.setattr(health, "_machine_id", lambda: "node-b")
        health._CACHED = None
        health.record_xet_outcome(False, "stall on B")
        monkeypatch.setattr(health, "_machine_id", lambda: "node-a")
        health._CACHED = None
        health.record_xet_outcome(True, "success on A")

    monkeypatch.setattr(health, "_machine_id", lambda: "node-b")
    health._CACHED = None
    assert health.xet_health(probe = False).use_xet is False, (
        "a genuinely broken node never demoted because peers kept clearing its streak"
    )
