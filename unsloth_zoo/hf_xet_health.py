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

"""Decide, per machine, whether to START a download on Xet or go straight to HTTP.

``hf_xet_fallback`` already recovers from a bad Xet attempt, but recovery costs the user the whole
stalled attempt EVERY time. On a machine where Xet reliably fails -- a blocked CAS endpoint, a
proxy that mangles range requests, too little RAM -- that toll repeats forever. This module
remembers outcomes so such a machine stops paying it, and re-probes later in case the cause was
temporary.

Design rules:
  * Never block a download. Every probe is time-boxed and every failure path answers "use Xet",
    because a wrong "healthy" costs one fallback while a wrong "unhealthy" would permanently
    downgrade a machine on which Xet works fine.
  * Cheapest checks first: an explicit override, then whether hf_xet exists at all, then RAM, then
    the remembered verdict, and only then the network.
  * A verdict is scoped to what could invalidate it (hf_xet version, endpoint), and expires.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .hf_xet_tuning import MIN_XET_RAM_BYTES, system_profile

logger = logging.getLogger(__name__)

__all__ = [
    "XetHealth",
    "xet_health",
    "record_xet_outcome",
    "clear_xet_health",
    "health_state_path",
]

STATE_FILENAME = "unsloth_xet_health.json"

# A healthy verdict is re-probed weekly; a demotion is retried the next day, since the causes of a
# demotion (a flaky link, a transient CAS outage, a temporarily loaded box) are usually short-lived.
HEALTHY_TTL_SECONDS = 7 * 24 * 3600
DEMOTED_TTL_SECONDS = 24 * 3600

# One bad download is noise (a dropped wifi packet fails HTTP too); two in a row is a pattern.
DEMOTION_THRESHOLD = 2

# The probe runs on the request path, so it gets a strict budget.
PROBE_TIMEOUT_SECONDS = 3.0
_PROBE_REPO = "unsloth/Qwen3-30B-A3B-Instruct-2507"

_TRUTHY = {"1", "true", "yes", "on"}
_LOCK = threading.Lock()
# (timestamp, verdict, was_probed). The probe flag is part of the key because the download path
# calls this with probe = False on every download: without it, that cheap lookup memoizes the
# optimistic "no cached verdict; defaulting to Xet" answer and silently disarms an explicit
# preflight -- e.g. Studio's transport picker -- for the next minute, on exactly the CAS-blocked
# machine the probe exists to catch.
_CACHED: "Optional[tuple[float, XetHealth, bool]]" = None
_MEMO_SECONDS = 60.0  # a snapshot download asks repeatedly; don't re-probe per file


def _is_true(value: Optional[str]) -> bool:
    return value is not None and str(value).strip().lower() in _TRUTHY


@dataclass(frozen = True)
class XetHealth:
    use_xet: bool
    reason: str
    source: str

    def __bool__(self) -> bool:  # `if xet_health():` reads as "is Xet usable"
        return self.use_xet


def _endpoint() -> str:
    return os.environ.get("HF_ENDPOINT") or "https://huggingface.co"


def _hf_xet_version() -> Optional[str]:
    try:
        import hf_xet  # type: ignore
    except Exception:
        return None
    version = getattr(hf_xet, "__version__", None)
    if version:
        return str(version)
    try:
        from importlib.metadata import version as _version

        return str(_version("hf_xet"))
    except Exception:
        return "unknown"


def health_state_path() -> Optional[Path]:
    """Where the verdict lives: beside the HF cache, so it follows a relocated ``HF_HOME``."""
    try:
        from .hf_cache import _active_caches

        from .hf_cache import _is_writable

        hf_home, hub_cache, _ = _active_caches()
        # HF_HOME resolves to a default even when it is read-only, so preferring it unconditionally
        # sent the verdict to an unwritable path whenever a user relocated only HF_HUB_CACHE. Every
        # write then failed silently, and since the failure streak is rebuilt from this file alone,
        # such a machine could never be demoted no matter how often Xet stalled.
        candidates = [hf_home, hub_cache.parent if hub_cache is not None else None]
        for base in candidates:
            if base is not None and _is_writable(base):
                return base / STATE_FILENAME
        base = next((c for c in candidates if c is not None), None)
        return base / STATE_FILENAME if base is not None else None
    except Exception:
        return None


def _read_state() -> dict:
    path = health_state_path()
    if path is None:
        return {}
    try:
        with open(path, "r") as f:
            state = json.load(f)
        return state if isinstance(state, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_state(state: dict) -> None:
    """Atomic replace so a concurrent reader never sees a half-written verdict."""
    path = health_state_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents = True, exist_ok = True)
        fd, tmp = tempfile.mkstemp(dir = str(path.parent), prefix = ".xet_health_")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(state, f)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except OSError as e:
        logger.debug("Could not persist Xet health state: %s", e)


def _state_is_current(state: dict) -> bool:
    """A verdict only speaks for the version and endpoint it was measured against."""
    if not state or "verdict" not in state:
        return False
    if state.get("hf_xet_version") != _hf_xet_version():
        return False
    if state.get("endpoint") != _endpoint():
        return False
    ttl = HEALTHY_TTL_SECONDS if state.get("verdict") == "xet" else DEMOTED_TTL_SECONDS
    try:
        return (time.time() - float(state.get("ts", 0))) < ttl
    except (TypeError, ValueError):
        return False


def _probe_cas_reachable() -> "tuple[Optional[bool], str]":
    """Can we get a Xet read token and reach the CAS endpoint it names?

    A token is NOT required: the endpoint answers anonymously, so this measures reachability
    (corporate proxy, firewalled CAS domain, offline), not authentication.
    """
    try:
        import urllib.error
        import urllib.request

        url = f"{_endpoint()}/api/models/{_PROBE_REPO}/xet-read-token/main"
        request = urllib.request.Request(url, headers = {"User-Agent": "unsloth-xet-probe"})
        token = os.environ.get("HF_TOKEN")
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        deadline = time.monotonic() + PROBE_TIMEOUT_SECONDS
        with urllib.request.urlopen(request, timeout = PROBE_TIMEOUT_SECONDS) as response:
            if response.status != 200:
                return (False, f"Xet token endpoint returned HTTP {response.status}")
            payload = json.loads(response.read(64 * 1024).decode("utf-8", "replace"))

        cas = payload.get("casUrl") or payload.get("cas_url")
        if not cas:
            return (True, "Xet token issued")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return (True, "Xet token issued (CAS check skipped, out of budget)")
        # A HEAD to the CAS host root: we only care that the connection completes, so any HTTP
        # status counts as reachable. Only a transport error means Xet cannot work here.
        head = urllib.request.Request(cas, method = "HEAD", headers = {"User-Agent": "unsloth-xet-probe"})
        try:
            urllib.request.urlopen(head, timeout = remaining)
        except urllib.error.HTTPError:
            pass
        return (True, "Xet CAS reachable")
    except urllib.error.HTTPError as e:
        # The endpoint ANSWERED, which is the only thing this probe measures. A 404 just means the
        # probe repo is not hosted here -- an HF_ENDPOINT mirror or on-prem deployment -- and must
        # not pin the machine to HTTP for 24h. A 401/403 still demotes: a blocking corporate proxy
        # legitimately answers that way.
        if e.code == 404:
            return (None, "Xet probe repo absent on this endpoint; assuming Xet")
        return (False, f"Xet token endpoint returned HTTP {e.code}")
    except Exception as e:
        return (False, f"Xet CAS unreachable: {type(e).__name__}")


def xet_health(*, force: bool = False, probe: bool = True) -> XetHealth:
    """Whether a download should START on Xet. Cheap and safe to call per download."""
    global _CACHED

    if _is_true(os.environ.get("UNSLOTH_DISABLE_XET")) or _is_true(
        os.environ.get("UNSLOTH_STABLE_DOWNLOADS")
    ) or _is_true(os.environ.get("HF_HUB_DISABLE_XET")):
        return XetHealth(False, "Xet disabled by environment", "forced")
    if _is_true(os.environ.get("UNSLOTH_FORCE_XET")):
        return XetHealth(True, "Xet forced by environment", "forced")

    with _LOCK:
        if (
            not force
            and _CACHED is not None
            and (time.monotonic() - _CACHED[0]) < _MEMO_SECONDS
            # An unprobed optimistic default only satisfies a caller that did not ask to probe. Any
            # real verdict (cached / forced / probe) still short-circuits every caller.
            and (_CACHED[2] or not probe or _CACHED[1].source != "default")
        ):
            return _CACHED[1]

    result = _evaluate(force = force, probe = probe)
    with _LOCK:
        _CACHED = (time.monotonic(), result, probe)
    return result


def _evaluate(*, force: bool, probe: bool) -> XetHealth:
    if _hf_xet_version() is None:
        return XetHealth(False, "hf_xet is not installed", "unavailable")

    profile = system_profile()
    if profile.total_ram_bytes and profile.total_ram_bytes < MIN_XET_RAM_BYTES:
        gb = profile.total_ram_bytes / 1e9
        return XetHealth(
            False,
            f"only {gb:.1f}GB RAM available to this process; HTTP uses far less",
            "low-ram",
        )

    state = _read_state()
    if not force and _state_is_current(state):
        verdict = state.get("verdict") == "xet"
        return XetHealth(verdict, str(state.get("reason") or "remembered verdict"), "cached")

    if not probe:
        # No fresh verdict and probing not allowed: Xet is the better default (the ladder still
        # covers a bad guess).
        return XetHealth(True, "no cached verdict; defaulting to Xet", "default")

    ok, reason = _probe_cas_reachable()
    if ok is None:
        # Inconclusive. Persist nothing: a wrong "unhealthy" downgrades a working machine for a
        # day, a wrong "healthy" costs one fallback.
        return XetHealth(True, reason, "default")
    _write_state({
        "verdict": "xet" if ok else "http",
        "reason": reason,
        "ts": time.time(),
        "hf_xet_version": _hf_xet_version(),
        "endpoint": _endpoint(),
        # A fresh probe supersedes the old streak; failures are counted from here.
        "consecutive_failures": 0,
    })
    return XetHealth(ok, reason, "probe")


def record_xet_outcome(ok: bool, reason: str = "") -> None:
    """Feed a finished Xet attempt back in. Two consecutive failures demote the machine to HTTP.

    Called from the download ladder, so it must never raise and never slow the caller down.
    """
    global _CACHED
    try:
        state = _read_state()
        # A streak recorded against a different hf_xet build says nothing about this one.
        if state.get("hf_xet_version") != _hf_xet_version() or state.get("endpoint") != _endpoint():
            state = {}
        failures = 0 if ok else int(state.get("consecutive_failures") or 0) + 1

        if ok:
            verdict, note = "xet", reason or "Xet download succeeded"
        elif failures >= DEMOTION_THRESHOLD:
            verdict = "http"
            note = reason or f"Xet failed {failures} times in a row on this machine"
            logger.warning(
                "Xet has now failed %d times in a row; new downloads will use HTTP for the next "
                "%dh (set UNSLOTH_FORCE_XET=1 to override).",
                failures, DEMOTED_TTL_SECONDS // 3600,
            )
        else:
            # Below the threshold, keep whatever verdict stands rather than promoting on a failure.
            verdict = str(state.get("verdict") or "xet")
            note = str(state.get("reason") or "")

        _write_state({
            "verdict": verdict,
            "reason": note,
            "ts": time.time(),
            "hf_xet_version": _hf_xet_version(),
            "endpoint": _endpoint(),
            "consecutive_failures": failures,
        })
        with _LOCK:
            _CACHED = None  # the next call must see the new verdict
    except Exception as e:
        logger.debug("Could not record Xet outcome: %s", e)


def clear_xet_health() -> None:
    """Forget the verdict (used by tests and by an explicit user reset)."""
    global _CACHED
    with _LOCK:
        _CACHED = None
    path = health_state_path()
    if path is not None:
        try:
            path.unlink()
        except OSError:
            pass
