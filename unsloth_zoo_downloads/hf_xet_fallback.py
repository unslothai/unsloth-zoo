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

"""Xet-primary HF downloads with a Xet retry, then an automatic HTTP fallback, on a stall or a fault.

Xet (``hf_xet``) is fast but can hang with no progress, no exception, and an un-killable native thread.
``HF_HUB_DISABLE_XET`` is read at import time, so the fallback runs in a fresh ``spawn`` child (not a
thread) that sets the env before importing ``huggingface_hub``. Cached files short-circuit with no
child; deterministic errors (401/403/404/disk-full) and cancellation propagate without a fallback.

The transport also changes on a transient fault, a crash and an incomplete snapshot, not only on a
stall, and on Xet an UNRECOGNIZED error counts as transient: hf_xet reports a CAS fault as a bare
``RuntimeError`` whose Rust error chain this package does not own, and treating that as deterministic
skipped the HTTP rung entirely. HTTP is the last rung and gets ``UNSLOTH_HTTP_ATTEMPTS`` children
(default 2, ``UNSLOTH_HTTP_RETRY_BACKOFF`` seconds apart) of its own, because the Xet bridge CDN serves
Xet-backed blobs over plain HTTP too and one degraded CDN therefore fails both rungs. Exhausting them
raises ``DownloadTransportError``, a ``DownloadStallError``, so a caller's guard around the
supervised download cannot be bypassed by a retryable CDN error and fall through to the unguarded
in-process load.

A DATA-phase stall spends one more Xet child (``UNSLOTH_XET_ATTEMPTS``, default 2) before the
transport changes, since that hang is usually a wedged CAS stream that clears on a fresh process.
The retry RE-RUNS the transfer rather than resuming it: ``xet_get`` never passes the ``resume_size``
the HTTP branch uses, so the file is rebuilt from offset zero. It keeps everything already finalized
into a blob (for a snapshot, every completed shard), so the cost is replaying the in-flight file.
``snapshot_download_with_xet_fallback`` warms a whole repo in a killable child before Unsloth's
in-process load; ``hf_hub_download_with_xet_fallback`` does a single file. Unsloth cache / secret /
process helpers are used best-effort (imported only if present) or injected. The child imports only
this package; it still inherits ``UNSLOTH_ZOO_DISABLE_GPU_INIT=1`` so any ``unsloth_zoo`` import there
takes the lightweight path (no torch / transformers).
"""

from __future__ import annotations

import builtins
import errno
import importlib.util
import math
import multiprocessing as mp
import logging
import os
import stat
import queue
import re
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from unsloth_zoo_downloads.hf_cache_state import (
    INCOMPLETE_SUFFIX,
    _ROOT_MODEL_SHARD_INDEX_RE,
    _ROOT_MODEL_VARIANT_WEIGHT_RE,
    _as_pattern_list,
    _component_index_weight_probe,
    _diffusers_component_shards_incomplete,
    _diffusers_declared_component_specs,
    _filter_paths,
    _has_glob,
    _has_incomplete_canonical_root_shards,
    _has_incomplete_variant_root_shards,
    _index_variant_token,
    _index_weight_probe,
    _is_canonical_component_shard_index,
    _is_canonical_weight_shard_index,
    _is_loadable_weight_file,
    _read_format_kept,
    _selected_shard_index_incomplete,
    _sentence_transformers_subfolder_incomplete,
    _iter_snapshot_dirs,
    _weight_shard_index_complete,
    blob_bytes_present,
    repo_cache_dir_name,
    has_active_incomplete_blobs,
    hf_cache_root,
    iter_active_repo_cache_dirs,
    request_can_include_weights,
    requested_named_files_present,
    snapshot_dir_is_complete,
    snapshot_has_requested_broken_symlinks,
)

logger = logging.getLogger(__name__)

# Explicit list keeps stdlib imports out of Unsloth's `import *` re-export shim.
__all__ = [
    "DownloadStallError",
    "DownloadTransportError",
    "hf_hub_download_with_xet_fallback",
    "snapshot_download_with_xet_fallback",
    "start_watchdog",
    "get_hf_download_state",
    "is_hf_xet_available",
    "xet_force_disabled",
    "child_should_disable_xet",
    "is_data_phase_stall",
    "xet_attempts",
    "http_attempts",
    "DEFAULT_STALL_TIMEOUT",
    "DEFAULT_CONNECT_TIMEOUT",
    "DEFAULT_HTTP_STALL_TIMEOUT",
    "DEFAULT_XET_ATTEMPTS",
    "DEFAULT_HTTP_ATTEMPTS",
    "DEFAULT_HTTP_RETRY_BACKOFF",
]

_CTX = mp.get_context("spawn")

# Stall thresholds by phase: once a partial exists, 30s of frozen bytes means a hung Xet transfer since Xet writes
# continuously; before any partial, DNS, TLS and metadata get a longer connect window; HTTP keeps the patient threshold
# as the last resort.
DEFAULT_STALL_TIMEOUT = 30.0
DEFAULT_CONNECT_TIMEOUT = 90.0
DEFAULT_HTTP_STALL_TIMEOUT = 180.0
# Minimum partial age before clearing it while one of our own processes is uninspectable (non-dumpable); far above any
# live Xet writer's no-growth window, and crash leftovers are older.
_OPAQUE_WRITER_GRACE = 1800.0
# Watchdog measure interval; detection latency is up to one interval beyond the timeout, so it must stay well under DEFAULT_STALL_TIMEOUT.
DEFAULT_POLL_INTERVAL = 5.0

# Xet attempts before falling back to HTTP; a wedged transfer often clears on a fresh child, and only DATA-phase stalls
# are retried, bounding the extra wait to one stall_timeout; 1 restores straight-to-HTTP.
DEFAULT_XET_ATTEMPTS = 2
# Past this the ladder just burns time before the transport that would have worked.
_MAX_XET_ATTEMPTS = 8

# HTTP attempts before giving up; the Xet bridge CDN serves Xet-backed blobs over plain HTTP too, so a degraded CDN
# fails both rungs and one child would turn a blip into a hard failure; 1 restores single-attempt.
DEFAULT_HTTP_ATTEMPTS = 2
_MAX_HTTP_ATTEMPTS = 8
# Wait between HTTP children so a shedding CDN is not hit again at once; 0 is valid and means retry immediately; UNSLOTH_HTTP_RETRY_BACKOFF overrides.
DEFAULT_HTTP_RETRY_BACKOFF = 5.0
# Caps the HTTP retry wait: longer waits hold the caller hostage, and past PY_TIMEOUT_MAX time.sleep and Event.wait
# raise OverflowError, which escapes the DownloadStallError guard into the unguarded load.
_MAX_HTTP_RETRY_BACKOFF = 300.0
# hub's short-download error: a bare EnvironmentError, so nothing in its TYPE says "network".
_HUB_CONSISTENCY_ERROR_RE = re.compile(
    r"consistency check failed: file should be of size", re.IGNORECASE
)

# Disk-full errnos read defensively via getattr: which errno names exist is platform-specific (CPython builds ifdef each
# one), so a bare errno.EDQUOT would raise AttributeError.
_DISK_FULL_ERRNOS = frozenset(
    code for code in (getattr(errno, name, None) for name in ("ENOSPC", "EDQUOT"))
    if code is not None
)

# RSS growth from a receiving child tracks the wire rate; this epsilon sits above allocator noise but below a thin
# link's per-poll volume, so a genuinely hung child cannot drift past it.
_RSS_PROGRESS_EPSILON = 4 * 1024 * 1024
# Growth above _RSS_SLOW_LINK_EPSILON but below _RSS_PROGRESS_EPSILON is a slow link still receiving (xet touches the
# file only at head-of-line boundaries); the window moves rather than the epsilon, which is frozen-child noise margin.
_RSS_SLOW_LINK_EPSILON = 256 * 1024

# Absolute cap on peer liveness resetting the connect clock: the hub lock is per blob so one blob's wait is bounded, but
# unrelated same-repo downloads would otherwise suppress the clock forever and hide a hung child.
PEER_SUPPRESSION_CEILING = 3600.0
# How often a "still downloading" status is pushed to the UI (independent of the measure rate).
DEFAULT_HEARTBEAT_INTERVAL = 30.0
DEFAULT_GRACE_PERIOD = 10.0
_POLL_INTERVAL = 0.5


def _env_seconds(name: str, default: float) -> float:
    """Positive finite float from the env, junk ignored; nan and inf are rejected since they pass a <= 0 test but break
    deadlines (nan never fires, inf raises OverflowError)."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("Ignoring non-numeric %s=%r", name, raw)
        return default
    if not math.isfinite(value) or value <= 0:
        logger.warning("Ignoring non-positive or non-finite %s=%r", name, raw)
        return default
    return value


def xet_attempts() -> int:
    """Xet attempts from UNSLOTH_XET_ATTEMPTS; junk or non-positive uses the default and the value is clamped so a typo
    cannot strand a user on a dead transport; 0 does not disable Xet (see UNSLOTH_DISABLE_XET)."""
    raw = os.environ.get("UNSLOTH_XET_ATTEMPTS")
    if not raw:
        return DEFAULT_XET_ATTEMPTS
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning("Ignoring non-integer UNSLOTH_XET_ATTEMPTS=%r", raw)
        return DEFAULT_XET_ATTEMPTS
    if value <= 0:
        logger.warning("Ignoring non-positive UNSLOTH_XET_ATTEMPTS=%r", raw)
        return DEFAULT_XET_ATTEMPTS
    return min(value, _MAX_XET_ATTEMPTS)


def http_attempts() -> int:
    """HTTP attempts from UNSLOTH_HTTP_ATTEMPTS, with xet_attempts' junk and clamp rules; retried only on transient
    transport failure or crash, never on a stall, which already waited out the patient threshold."""
    raw = os.environ.get("UNSLOTH_HTTP_ATTEMPTS")
    if not raw:
        return DEFAULT_HTTP_ATTEMPTS
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning("Ignoring non-integer UNSLOTH_HTTP_ATTEMPTS=%r", raw)
        return DEFAULT_HTTP_ATTEMPTS
    if value <= 0:
        logger.warning("Ignoring non-positive UNSLOTH_HTTP_ATTEMPTS=%r", raw)
        return DEFAULT_HTTP_ATTEMPTS
    return min(value, _MAX_HTTP_ATTEMPTS)


def _http_retry_backoff() -> float:
    """Seconds between HTTP children from UNSLOTH_HTTP_RETRY_BACKOFF, clamped to _MAX_HTTP_RETRY_BACKOFF; 0 means retry
    at once, not junk; negative or unparseable falls back; nan and inf are rejected by name, since an error escaping
    time.sleep would be swallowed by the caller."""
    raw = os.environ.get("UNSLOTH_HTTP_RETRY_BACKOFF")
    if not raw:
        return DEFAULT_HTTP_RETRY_BACKOFF
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning("Ignoring non-numeric UNSLOTH_HTTP_RETRY_BACKOFF=%r", raw)
        return DEFAULT_HTTP_RETRY_BACKOFF
    if not math.isfinite(value) or value < 0:
        logger.warning("Ignoring negative or non-finite UNSLOTH_HTTP_RETRY_BACKOFF=%r", raw)
        return DEFAULT_HTTP_RETRY_BACKOFF
    return min(value, _MAX_HTTP_RETRY_BACKOFF)


def is_data_phase_stall(message: str) -> bool:
    """True if a watchdog verdict fired after bytes flowed; the pre-first-byte 'did not start' wording is excluded, so a
    reworded verdict fails cheaply (no retry); the health tracker shares this rule."""
    return "did not start" not in (message or "")


# Serializes the parent env and __main__.__file__ mutation around a child spawn so concurrent downloads cannot see each other's transport env.
_SPAWN_ENV_LOCK = threading.Lock()

# Sentinel: "__main__.__file__ untouched for this spawn" (distinct from a saved None).
_UNSET = object()

# HF boolean env convention, case-insensitive.
_TRUTHY = {"1", "true", "yes", "on"}


def _is_true(value: Optional[str]) -> bool:
    return value is not None and str(value).strip().lower() in _TRUTHY


def _safe_status(callback: Optional[Callable[[str], None]], message: str) -> None:
    """Calls the status callback, swallowing its exceptions so a disconnected client cannot kill the watchdog thread and stop stall detection."""
    if callback is None:
        return
    try:
        callback(message)
    except Exception as e:
        logger.debug("watchdog status callback raised (ignored): %s", e)


def _xet_health_or_none():
    """Xet verdict or None; probe=False reads only cheap local signals (RAM floor, remembered verdict) so downloads add
    no latency, and any failure means no opinion, so the Xet-first ladder runs."""
    try:
        from unsloth_zoo_downloads.hf_xet_health import xet_health

        return xet_health(probe = False)
    except Exception as e:
        logger.debug("Xet health check unavailable: %s", e)
        return None


def _xet_log_dir() -> Optional[str]:
    """Where hf_xet writes logs: HF_XET_LOG_DEST if it is a directory, else the logs/ subdir of the active Xet cache (hf_xet's default)."""
    dest = os.environ.get("HF_XET_LOG_DEST")
    if dest:
        # An empty value means "log to console"; a file path is not a directory we can scan.
        return dest if os.path.isdir(dest) else None
    try:
        from unsloth_zoo_downloads.hf_cache import _active_caches

        _, _, xet_cache = _active_caches()
        return str(xet_cache / "logs") if xet_cache is not None else None
    except Exception:
        return None


def _xet_failure_reason(summary: str) -> str:
    """Attach hf_xet's own ERROR/WARN lines to *summary* so the recorded verdict says WHY."""
    try:
        from unsloth_zoo_downloads.hf_xet_tuning import scan_xet_log

        # Scrubbed like other child errors: a reqwest failure displays the full URL, and CAS/S3 presigned URLs carry an
        # X-Amz-Signature; this reason is persisted to health state and shown in the UI for 24h.
        messages = [
            _default_scrub_secrets(m)
            for m in scan_xet_log(_xet_log_dir(), max_messages = 2)
        ]
    except Exception:
        messages = []
    return f"{summary}: {' | '.join(messages)}" if messages else summary


def _record_xet_outcome(ok: bool, reason: str = "") -> None:
    """Report a finished Xet attempt to the health tracker; never raises."""
    try:
        from unsloth_zoo_downloads.hf_xet_health import record_xet_outcome

        record_xet_outcome(ok, reason)
    except Exception as e:
        logger.debug("Could not record Xet outcome: %s", e)


class DownloadStallError(RuntimeError):
    """Raised when no download progress is observed for too long. Unsloth re-imports this canonical type."""


class DownloadTransportError(DownloadStallError):
    """Every transport failed with a transient error (CDN 5xx, CAS fault, child crash), not a hang; subclasses
    DownloadStallError so a caller's guard cannot let it fall through to the unwatched load; still a RuntimeError."""


def is_hf_xet_available() -> bool:
    """True iff the ``hf_xet`` extra is importable (Hub uses it automatically)."""
    try:
        return importlib.util.find_spec("hf_xet") is not None
    except Exception:
        return False


def xet_force_disabled() -> bool:
    """True when UNSLOTH_DISABLE_XET, UNSLOTH_STABLE_DOWNLOADS or HF_HUB_DISABLE_XET forces HTTP up front."""
    return (
        _is_true(os.environ.get("UNSLOTH_DISABLE_XET"))
        or _is_true(os.environ.get("UNSLOTH_STABLE_DOWNLOADS"))
        or _is_true(os.environ.get("HF_HUB_DISABLE_XET"))
    )


def child_should_disable_xet(config: dict) -> bool:
    """Single source of truth for the per-worker Xet env flip."""
    return bool(config.get("disable_xet"))


def _default_scrub_secrets(text: str, hf_token: Optional[str] = None) -> str:
    """Best-effort redaction of a token / bearer credential from an error string."""
    if not text:
        return text
    out = text
    # token=True ("use cached token") is common; only a real string token can be substring-redacted.
    if isinstance(hf_token, str) and hf_token:
        out = out.replace(hf_token, "***")
    out = re.sub(r"hf_[A-Za-z0-9]{8,}", "***", out)
    out = re.sub(r"([Bb]earer\s+)[A-Za-z0-9._\-]+", r"\1***", out)
    # Redact the query of presigned S3/CAS blob URLs (temporary creds in the query string); keep unsigned URLs such as ?download=true.
    def _redact_signed_query(match: "re.Match") -> str:
        base, query = match.group(1), match.group(2)
        if re.search(
            r"(X-Amz-|[Ss]ignature|(?:^|&)(?:sig|token|key|Expires|Policy|Key-Pair-Id)=)",
            query,
        ):
            return f"{base}?***"
        return match.group(0)

    # Query match stops at whitespace or a structural delimiter (quote, bracket, brace, paren, angle, pipe), since a URL
    # in JSON or a dict repr would otherwise swallow the closing brace; signed values are percent-encoded so presigned
    # URLs stay fully redacted.
    out = re.sub(
        r"(https?://[^\s?]+)\?([^\s\"'()<>{}|[\]]*)", _redact_signed_query, out
    )
    return out


def _partial_partners_of(target: Path) -> "Optional[list]":
    """Matches both partial spellings: <etag>.incomplete and the process-unique {stem}.<hex8>.incomplete hub writes
    since huggingface_hub PR 4228 (a shared name corrupts caches where flock silently succeeds); an exact-name check
    misses live siblings."""
    partners = []
    exact = target.with_name(target.name + INCOMPLETE_SUFFIX)
    try:
        if stat.S_ISREG(os.stat(exact).st_mode):
            partners.append(exact)
    except (FileNotFoundError, NotADirectoryError):
        pass
    except OSError:
        # An OSError here means unknown, not absence: reading a permission or FUSE flap as no partner would drop a
        # sibling's in-progress blob; unknown must propagate, as an older revision's link is not recreated.
        return None
    # Uses os.scandir, not Path.glob: glob hides a read error as an empty list, reading as no partner and unlinking a
    # link while a sibling writes; a blobs dir with execute but no read fails only in the listing.
    prefix = target.name + "."
    try:
        with os.scandir(target.parent) as entries:
            for entry in entries:
                if not (entry.name.startswith(prefix)
                        and entry.name.endswith(INCOMPLETE_SUFFIX)):
                    continue
                try:
                    if entry.is_file():
                        partners.append(Path(entry.path))
                except OSError:
                    return None
    except (FileNotFoundError, NotADirectoryError):
        return partners
    except OSError:
        return None
    return partners


def _broken_link_has_active_partner(link: Path, *, active_grace: float) -> bool:
    """Spares a dangling snapshot link only while its target has a FRESH .incomplete partner, not judged by link mtime:
    a killed child's partner goes static and is purged, a live sibling's grows; no partner clears the link."""
    try:
        target = Path(os.readlink(link))
        if not target.is_absolute():
            target = link.parent / target
        # Considers both partial spellings and takes the freshest: current hub siblings write <etag>.<nonce>.incomplete, which the exact name alone never sees.
        partners = _partial_partners_of(target)
        if partners is None:
            # Could not look. "No partner is writing this" is the answer that clears the link.
            return True
        for partner in partners:
            try:
                if time.time() - partner.stat().st_mtime < active_grace:
                    return True
            except OSError:
                continue
    except OSError:
        return False
    return False


def _link_incomplete_partner_name(link: Path) -> Optional[str]:
    """The ``<hash>.incomplete`` basename for a dangling symlink's target blob, or None."""
    try:
        target = Path(os.readlink(link))
        return target.name + INCOMPLETE_SUFFIX
    except OSError:
        return None


def _partial_name_is_process_unique(name: str) -> bool:
    """True for the process-unique {stem}.<nonce>.incomplete spelling, which no sibling can recreate; the deterministic
    <etag>.incomplete can be reopened after a purge. Blob names are hex, so an inner dot is the nonce."""
    if not name.endswith(INCOMPLETE_SUFFIX):
        return False
    return "." in name[: -len(INCOMPLETE_SUFFIX)]


def _link_partner_still_on_disk(link: Path, removed_partials: set) -> bool:
    """True if the link's target partial still exists and this call did not remove it; not a freshness test, since an opened-but-unwritten partial has a stale mtime."""
    try:
        target = Path(os.readlink(link))
        if not target.is_absolute():
            target = link.parent / target
    except OSError:
        return True                      # cannot tell, so do not remove the record
    partners = _partial_partners_of(target)
    if partners is None:
        return True
    return any(partner.name not in removed_partials for partner in partners)


def _link_partner_is_owned(link: Path, owned_incomplete_blobs: set) -> bool:
    """Whether a dangling link's target blob is one of our partials, matched by the blob name it points at, since both
    <etag>.incomplete and <etag>.<nonce>.incomplete occur."""
    try:
        target_name = Path(os.readlink(link)).name
    except OSError:
        return False
    prefix = target_name + "."
    return any(
        name == target_name + INCOMPLETE_SUFFIX
        or (name.startswith(prefix) and name.endswith(INCOMPLETE_SUFFIX))
        for name in owned_incomplete_blobs
    )


def _clear_partials(
    repo_type: str,
    repo_id: str,
    *,
    cache_dir: Optional[str] = None,
    active_grace: float = DEFAULT_STALL_TIMEOUT,
    owned_incomplete_blobs: Optional[set] = None,
    ownership_is_an_earlier_scan: bool = False,
    owned_names_may_be_reopened: bool = False,
) -> None:
    """Purges the repo's active *.incomplete blobs and broken snapshot links, scoped to owned_incomplete_blobs; an
    earlier-scan whitelist gets fresh age and live-writer checks at unlink; a reopened-name one skips the age check
    and is spared by a live writer (issue 9094)."""
    # Walks the process table once per call rather than per blob, since process_iter scans the whole host; taken at deletion time, not at the eligibility scan.
    rescanned_writers = None
    if ownership_is_an_earlier_scan or owned_names_may_be_reopened:
        _reset_live_writer_walk_record()
        rescanned_writers = _partial_paths_with_a_live_writer()
    if ownership_is_an_earlier_scan:
        # Same gate as the eligibility scan: a walk that could not inspect every process is only a lower bound, so
        # finding no writer is not proof; on a cache another UID can write, an unreadable process is a likely writer.
        # Without the gate the rescan would overturn the scan's decision.
        if rescanned_writers is not None and not _process_walk_sees_every_writer(cache_dir):
            rescanned_writers = None
        if rescanned_writers is not None and not _live_writer_walk_was_complete():
            if not _cache_is_private_to_this_user(
                cache_dir, repo_type = repo_type, repo_id = repo_id,
            ):
                rescanned_writers = None
            elif _live_writer_walk_missed_our_own_uid():
                # Mirrors the eligibility scan: an unreadable process of our own is not excused by privacy (declining
                # would spare every blob on a systemd login), so the age bar is raised the same way.
                active_grace = max(active_grace * 10.0, _OPAQUE_WRITER_GRACE)
    # The same question for the reopened-name pass, asked the other way round. There the walk is
    # not proof of ownership but proof that nobody ELSE holds a name we once owned, so a lower
    # bound reads as "free to unlink". PRIVACY decides it first: where no other UID can write
    # into the cache, no sibling can have reopened the name and an unreadable writer table costs
    # nothing -- that is every single-user machine, #9094's included, and declining there would
    # hand them all the repo-wide force this exists to avoid. Only on a cache somebody else can
    # write into does the walk have to carry the claim, and there a walk that could not read
    # every process is not proof of absence.
    # Privacy cannot cover a process running as US that the walk could not read, so that case
    # declines here too; the deterministic name is the one a same-uid sibling could hold.
    reopened_scan_is_evidence = True
    if owned_names_may_be_reopened:
        reopened_scan_is_evidence = (
            _cache_is_private_to_this_user(
                cache_dir, repo_type = repo_type, repo_id = repo_id,
            )
            # Privacy is about UIDs, but a sibling container shares the UID in its own PID namespace, so where the walk
            # cannot see every writer, unlinking the deterministic name could truncate a live download.
            and _process_walk_sees_every_writer(cache_dir)
            # Privacy does not excuse our own unreadable processes, but only the deterministic <etag>.incomplete can be
            # reopened, so on current hub the guard is skipped per blob.
            and not _live_writer_walk_missed_our_own_uid()
        ) or (
            rescanned_writers is not None
            and _process_walk_sees_every_writer(cache_dir)
            and _live_writer_walk_was_complete()
        )
    # Records what this call unlinked, so the snapshot sweep below can tell a link whose blob is gone from one whose partial a guard kept.
    removed_partials: set = set()
    try:
        for entry in iter_active_repo_cache_dirs(repo_type, repo_id, cache_dir = cache_dir):
            blobs_dir = entry / "blobs"
            if blobs_dir.is_dir():
                for blob in blobs_dir.iterdir():
                    if blob.is_file() and blob.name.endswith(INCOMPLETE_SUFFIX):
                        # Scope to our own partials when known: never delete a sibling's blob.
                        if owned_incomplete_blobs is not None and blob.name not in owned_incomplete_blobs:
                            continue
                        try:
                            # The mtime grace only guesses at sibling ownership and is wrong when ownership is known: it
                            # spares the killed child's partial, so has_active_incomplete_blobs() forces a full
                            # re-fetch.
                            owned = (
                                owned_incomplete_blobs is not None
                                and blob.name in owned_incomplete_blobs
                                and not ownership_is_an_earlier_scan
                            )
                            if not owned and time.time() - blob.stat().st_mtime < active_grace:
                                continue
                            if ownership_is_an_earlier_scan and (
                                rescanned_writers is None
                                or _normalized_partial_key(blob) in rescanned_writers
                            ):
                                # A writer appeared since the scan, or the writer table could not be read; either way it
                                # is no longer a partial nobody is writing, which is the whole claim.
                                continue
                            if (
                                owned
                                and owned_names_may_be_reopened
                                and not _partial_name_is_process_unique(blob.name)
                                and (
                                    not reopened_scan_is_evidence
                                    or (
                                        rescanned_writers is not None
                                        and _normalized_partial_key(blob) in rescanned_writers
                                    )
                                )
                            ):
                                # Our child is dead, so a current writer is a sibling that recreated the deterministic
                                # partial after the purge; an unanswered walk does not prove there is none.
                                continue
                            blob.unlink()
                            removed_partials.add(blob.name)
                        except OSError:
                            continue  # a locked / denied blob must not abort the rest
            # Clears broken snapshot symlinks, which also count as active incomplete state; sweeps every snapshot, since
            # a dangling link under an older revision keeps the repo incomplete and re-trips.
            snapshots_dir = entry / "snapshots"
            try:
                snapshot_dirs = [s for s in snapshots_dir.iterdir() if s.is_dir()]
            except OSError:
                snapshot_dirs = []
            for snapshot in snapshot_dirs:
                try:
                    for link in snapshot.rglob("*"):
                        if link.is_symlink() and not link.exists():
                            # Scope to our own partials when known; a link to a sibling's blob is theirs.
                            if owned_incomplete_blobs is not None and not (
                                _link_partner_is_owned(link, owned_incomplete_blobs)
                            ):
                                continue
                            # Spare a sibling's active link (target still has a fresh .incomplete).
                            if _broken_link_has_active_partner(link, active_grace = active_grace):
                                continue
                            # Also spare a link whose partial a check above kept: a writer reappearing after the scan
                            # may not have written yet, so mtime reads it as stale; unlinking would strand a blob this
                            # retry will not relink.
                            if _link_partner_still_on_disk(link, removed_partials):
                                continue
                            try:
                                link.unlink()
                            except OSError:
                                continue
                except OSError:
                    continue
    except Exception as e:
        logger.debug("partial purge failed for %s: %s", repo_id, e)


def _default_prepare_for_http(
    repo_type: str,
    repo_id: str,
    *,
    cache_dir: Optional[str] = None,
    active_grace: float = DEFAULT_STALL_TIMEOUT,
    owned_incomplete_blobs: Optional[set] = None,
) -> None:
    """Clears the partial before an HTTP resume, since resuming over a sparse Xet or hf_transfer partial silently
    corrupts the blob; Unsloth injects a marker-aware version instead."""
    _clear_partials(
        repo_type, repo_id, cache_dir = cache_dir, active_grace = active_grace,
        owned_incomplete_blobs = owned_incomplete_blobs,
    )


def _purge_owned_partials_for_xet_retry(
    repo_type: str,
    repo_id: str,
    *,
    cache_dir: Optional[str] = None,
    owned_incomplete_blobs: Optional[set] = None,
) -> None:
    """Clear child partial before Xet retry (hf_xet restarts at zero); patient HTTP grace; own blobs exempt"""
    _clear_partials(
        repo_type, repo_id, cache_dir = cache_dir,
        active_grace = DEFAULT_HTTP_STALL_TIMEOUT,
        owned_incomplete_blobs = owned_incomplete_blobs,
    )


def _active_incomplete_blob_sizes(
    repo_type: Optional[str], repo_id: str, cache_dir: Optional[str] = None
) -> dict[str, int]:
    """Maps {blob_filename: bytes present} (sparse-aware) for the repo's *.incomplete partials, so a sibling download of another file cannot mask this file's stall."""
    sizes: dict[str, int] = {}
    try:
        for entry in iter_active_repo_cache_dirs(repo_type, repo_id, cache_dir = cache_dir):
            blobs_dir = entry / "blobs"
            if not blobs_dir.is_dir():
                continue
            for blob in blobs_dir.iterdir():
                try:
                    if blob.is_file() and blob.name.endswith(INCOMPLETE_SUFFIX):
                        sizes[blob.name] = blob_bytes_present(blob)
                except OSError:
                    pass
    except Exception:
        pass
    return sizes


def _cache_root_is_merely_absent(cache_dir: Optional[str] = None) -> bool:
    """True only if the cache root does not exist (os.stat raises FileNotFoundError), not if it is unreadable;
    hf_cache_root gives None for both, so a baseline needs this split."""
    if cache_dir is None:
        try:
            from huggingface_hub import constants as hf_constants
        except ImportError:
            return False
        candidate = Path(hf_constants.HF_HUB_CACHE)
    else:
        try:
            candidate = Path(cache_dir).expanduser()
        except (RuntimeError, OSError):
            candidate = Path(cache_dir)
    try:
        stat_result = os.stat(candidate)
    except (FileNotFoundError, NotADirectoryError):
        return True
    except OSError:
        return False
    # There, and not a directory: nothing was ever cached under it either.
    return not stat.S_ISDIR(stat_result.st_mode)


def _strict_repo_cache_dirs(
    repo_type: Optional[str],
    repo_id: str,
    cache_dir: Optional[str] = None,
    *,
    absent_root_is_empty: bool = False,
) -> list:
    """Same case rule as _case_safe_repo_cache_dirs, but read errors are raised rather than reported as no directories,
    so 'cannot tell' stays distinct from 'nothing there'; callers map a raise to None."""
    root = hf_cache_root(cache_dir = cache_dir)
    if root is None:
        # Only a baseline may treat a missing root as empty: before the first download it legitimately does not exist,
        # and calling that unknown would leave our child's partial protected by the sibling grace (issue 9094).
        if absent_root_is_empty and _cache_root_is_merely_absent(cache_dir):
            return []
        raise OSError(f"no HF cache root for {cache_dir!r}")
    target = repo_cache_dir_name(repo_type, repo_id)
    folded_target = target.lower()
    entries = [entry for entry in root.iterdir() if entry.name.lower() == folded_target]
    exact = [entry for entry in entries if entry.name == target]
    if exact:
        return exact
    if len(entries) == 1 and (root / target).exists():
        return entries
    return []


def _blobs_dir_is_absent(blobs_dir) -> bool:
    """True only for ENOENT or ENOTDIR on blobs; other OSErrors raise, since Path.is_dir() returns False on a permission
    flap and that would release the partial guard."""
    try:
        return not stat.S_ISDIR(os.stat(blobs_dir).st_mode)
    except (FileNotFoundError, NotADirectoryError):
        return True


def _no_incomplete_blobs_confirmed(
    repo_type: Optional[str], repo_id: str, cache_dir: Optional[str] = None
) -> bool:
    """True only if the cache was read and holds no partials or broken active links; unreadable must not look clean, as
    an HTTP resume would finalize a corrupt blob. Asked only once partials were seen."""
    names = _incomplete_partial_names(repo_type, repo_id, cache_dir)
    if names is None or names:
        return False
    return not has_active_incomplete_blobs(repo_type, repo_id, cache_dir = cache_dir)


def _baseline_incomplete_blob_names(
    repo_type: Optional[str], repo_id: str, cache_dir: Optional[str] = None
) -> Optional[set]:
    """Partial names present before spawn, or None if unreadable; an empty baseline from a failed scan would claim a live sibling's partial as ours and delete it."""
    try:
        names = set()
        # Enumerated strictly, not via iter_active_repo_cache_dirs: that swallows the root's OSError and would return an
        # empty baseline instead of None, so sibling partials looked child-owned and were deleted mid-write.
        for entry in _strict_repo_cache_dirs(
            repo_type, repo_id, cache_dir, absent_root_is_empty = True,
        ):
            blobs_dir = entry / "blobs"
            if _blobs_dir_is_absent(blobs_dir):
                continue
            # Not per-blob, unlike the sizes scan: any uninspectable entry makes the whole baseline None, since a name
            # this scan misses would be credited to our child by the current-minus-baseline subtraction.
            for blob in blobs_dir.iterdir():
                if blob.is_file() and blob.name.endswith(INCOMPLETE_SUFFIX):
                    names.add(blob.name)
        return names
    except Exception:
        return None


def _incomplete_partial_names(
    repo_type: Optional[str], repo_id: str, cache_dir: Optional[str] = None
) -> Optional[set]:
    """Names of the *.incomplete partials, or None if the cache is unreadable; reports absence only, since allocation
    metadata cannot show holes and HTTP resume verifies size, not content."""
    names: set = set()
    try:
        # Selects from one root listing, not iter_active_repo_cache_dirs, which swallows OSError and would report a
        # permission flap or remount as no partials, releasing the guard; here every error surfaces.
        for entry in _strict_repo_cache_dirs(repo_type, repo_id, cache_dir):
            blobs_dir = entry / "blobs"
            if _blobs_dir_is_absent(blobs_dir):
                continue
            for blob in blobs_dir.iterdir():
                if blob.name.endswith(INCOMPLETE_SUFFIX):
                    names.add(blob.name)
    except Exception:
        return None
    return names


# Whether the last process walk could read every process; psutil.AccessDenied hides other-UID writers, so an incomplete
# walk yields only a lower bound; per thread, since a module global let one thread overwrite another's verdict.
_LIVE_WRITER_WALK = threading.local()


def _live_writer_walk_was_complete() -> bool:
    """Whether this thread's last live-writer walk read every process it listed; True by default, the answer for a
    caller that never walked (e.g. a monkeypatched probe)."""
    return bool(getattr(_LIVE_WRITER_WALK, "complete", True))


def _reset_live_writer_walk_record() -> None:
    """Clears this thread's last live-writer walk record before a new walk, so a walk that never runs cannot be judged by an earlier walk's readings."""
    _LIVE_WRITER_WALK.complete = True
    _LIVE_WRITER_WALK.missed_our_uid = False


def _live_writer_walk_missed_our_own_uid() -> bool:
    """Whether this thread's last walk failed to read a process that may run as our own UID, which cache privacy cannot rule out; False by default."""
    return bool(getattr(_LIVE_WRITER_WALK, "missed_our_uid", False))


def _fd_table_is_available() -> bool:
    """True where /proc/<pid>/fd gives open files independently of psutil, which answers for our own processes when
    psutil refuses them; macOS and Windows have no equivalent."""
    return os.name != "nt" and os.path.isdir("/proc")


def _proc_fd_partial_paths(pid: int) -> Optional[set]:
    """Partial paths pid holds open, via readlink on /proc/<pid>/fd; None if unreadable. Not psutil open_files(), which
    turns one unstattable file into AccessDenied (psutil issue 571)."""
    fd_dir = f"/proc/{pid}/fd"
    try:
        entries = os.listdir(fd_dir)
    except OSError:
        return None
    held: set = set()
    for fd in entries:
        try:
            target = os.readlink(os.path.join(fd_dir, fd))
        except OSError:
            continue
        if target.endswith(INCOMPLETE_SUFFIX):
            held.add(_normalized_partial_key(target))
    return held


def _process_may_be_this_user(proc, psutil) -> bool:
    """Whether *proc* might be running under our own effective uid. Unknown counts as yes."""
    if getattr(os, "geteuid", None) is None:
        return True                      # Windows: no uid to compare, so never excluded
    try:
        return proc.uids().effective == os.geteuid()
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        # A process that exited mid-walk holds nothing and is nobody's writer; counting it as unreadable would decline the purge on any busy machine.
        return False
    except Exception:
        return True


def _read_proc_text(path: str) -> Optional[str]:
    """Reads one /proc file as text, or None if unreadable; a seam so tests need not patch open() for the whole interpreter."""
    try:
        with open(path, "r", encoding = "utf-8", errors = "replace") as handle:
            return handle.read()
    except OSError:
        return None


def _device_of(path) -> Optional[int]:
    """st_dev of path, or None if it cannot be stat'ed; a seam so tests do not depend on the mount topology of the machine running them."""
    try:
        return os.stat(path).st_dev
    except OSError:
        return None


def _process_walk_sees_every_writer(cache_dir: Optional[str] = None) -> bool:
    """False in a container when the cache is on a separate mount, as the PID walk cannot see sibling containers sharing
    it; on the container's own root FS it is True; unreadable state is False too."""
    try:
        if os.name == "nt" or not os.path.isdir("/proc"):
            return True                      # no PID namespaces to hide a writer in
        if not _running_in_a_container():
            return True
        root = hf_cache_root(cache_dir = cache_dir)
        if root is None:
            return False
        # Container-privacy is judged by mount topology, not the device id: a host bind mount into an LXC or nspawn
        # container has the same st_dev, while sibling containers write into it unseen by process_iter.
        return _path_is_on_the_root_mount(root)
    except Exception:
        return False


# Init names for PID 1 on a real host, not a container: a namespaced process walk is what is being detected, and an init
# system as PID 1 is the strongest evidence, since a container's PID 1 is its entrypoint.
_HOST_INIT_NAMES = frozenset({
    "systemd", "init", "launchd", "runit", "openrc-init", "s6-svscan", "upstart", "sysvinit",
})


def _path_is_on_the_root_mount(path) -> bool:
    """True if path is carried by the root mount with no mount beneath it, per /proc/self/mountinfo; unreadable mountinfo answers False, which only costs a purge."""
    text = _read_proc_text("/proc/self/mountinfo")
    if not text:
        return False
    try:
        target = os.path.realpath(str(path))
    except Exception:
        return False
    carrier = ""
    for line in text.splitlines():
        fields = line.split(" ")
        if len(fields) < 5:
            continue
        # Mount points are octal-escaped for space, tab, newline and backslash.
        mount_point = (
            fields[4].replace("\\040", " ").replace("\\011", "\t")
            .replace("\\012", "\n").replace("\\134", "\\")
        )
        if target == mount_point or target.startswith(mount_point.rstrip("/") + os.sep):
            if len(mount_point) > len(carrier):
                carrier = mount_point
    return carrier == "/"


def _running_in_a_container() -> bool:
    """Whether this may be in its own PID namespace; uncertain answers yes, since a wrong yes only declines a purge;
    checks markers, a bare cgroup v2 '0::/', then PID 1's name (init on host, entrypoint in container)."""
    # The markers systemd itself writes, which is what an init-based container looks like:
    # nspawn and its relatives run systemd as PID 1 with a perfectly ordinary `/init.scope`
    # cgroup and no runtime marker anywhere, so PID 1's NAME would answer "host" and the mount
    # check below would never be consulted. `/run/systemd/container` names the technology;
    # `/run/host` is the host mount nspawn exposes; `/proc/vz` without `/proc/bc` is OpenVZ.
    for marker in (
        "/.dockerenv", "/run/.containerenv", "/run/systemd/container", "/run/host",
    ):
        if os.path.exists(marker):
            return True
    if os.path.isdir("/proc/vz") and not os.path.isdir("/proc/bc"):
        return True
    cgroup = _read_proc_text("/proc/1/cgroup")
    if cgroup is None:
        return True                      # cannot tell, so do not claim the host
    if any(
        marker in cgroup
        for marker in ("docker", "kubepods", "containerd", "lxc", "podman", "libpod", "crio")
    ):
        return True
    lines = [line for line in cgroup.splitlines() if line.strip()]
    pid_one = _read_proc_text("/proc/1/comm")
    if lines == ["0::/"]:
        # A root cgroup alone is not container-specific (OpenRC, runit or busybox-init hosts match too), so the shape is
        # believed only if PID 1 is a supported init and the cgroup namespace is the initial one (4026531835)
        if pid_one is None or pid_one.strip() not in _HOST_INIT_NAMES:
            return True
        try:
            if os.readlink("/proc/1/ns/cgroup") != "cgroup:[4026531835]":
                return True
        except OSError:
            return True                  # cannot tell, so do not claim the host
        return False
    if pid_one is None:
        return True
    return pid_one.strip() not in _HOST_INIT_NAMES


def _group_is_private_to_this_user(gid: int) -> bool:
    """True if only this user is in the gid (gr_mem omits primary members); unreadable is False"""
    if gid != os.getegid():
        return False
    try:
        import grp  # noqa: PLC0415 -- POSIX only, and only on this path
        import pwd  # noqa: PLC0415

        if grp.getgrgid(gid).gr_mem:
            return False
        me = os.geteuid()
        return all(user.pw_gid != gid or user.pw_uid == me for user in pwd.getpwall())
    except Exception:
        return False


def _posix_directory_is_private(path) -> bool:
    """Whether only this UID can create or replace entries in *path*, on POSIX.

    Ownership and the write bits, plus the absence of a POSIX ACL: an ACL can grant another
    user write access with the mode bits reading 0755, and a directory that carries one is
    treated as shared because this cannot evaluate who it names. Where extended attributes
    cannot be listed at all the ACL question is unanswerable, and an unanswerable question is
    not evidence of privacy.
    """
    info = os.stat(path)
    if info.st_uid != os.geteuid():
        return False
    if info.st_mode & stat.S_IWOTH:
        return False
    if info.st_mode & stat.S_IWGRP and not _group_is_private_to_this_user(info.st_gid):
        return False
    listxattr = getattr(os, "listxattr", None)
    if listxattr is None:
        # No os.listxattr on this platform build, so an ACL cannot be ruled out; the mode bits, already checked, are all there is
        return True
    try:
        names = listxattr(path)
    except OSError:
        return False
    return not any(
        name in ("system.posix_acl_access", "system.posix_acl_default")
        or name.startswith("system.nfs4_acl")
        for name in names
    )


def _cache_is_private_to_this_user(
    cache_dir: Optional[str] = None,
    *,
    repo_type: Optional[str] = None,
    repo_id: Optional[str] = None,
) -> bool:
    """True only if no other UID can write partials: on POSIX, owner, write bits and ACL of the root, the repo dir and
    its blobs; on Windows, containment in the home dir; anything else is False."""
    try:
        root = hf_cache_root(cache_dir = cache_dir)
        if root is None:
            return False
        if getattr(os, "geteuid", None) is None:
            home = Path.home().resolve()
            return home in Path(root).resolve().parents or Path(root).resolve() == home
        if not _posix_directory_is_private(root):
            return False
        if repo_id is None:
            return True
        # A repo that does not exist yet is not a reason to call the cache shared; a directory that exists and is writable by another user must not be missed
        for entry in _strict_repo_cache_dirs(
            repo_type, repo_id, cache_dir, absent_root_is_empty = True,
        ):
            if not _posix_directory_is_private(entry):
                return False
            blobs_dir = entry / "blobs"
            if _blobs_dir_is_absent(blobs_dir):
                continue
            if not _posix_directory_is_private(blobs_dir):
                return False
        return True
    except Exception:
        return False


def _normalized_partial_key(path) -> str:
    """Comparable spelling of a partial's location (realpath, case folded where the filesystem folds case), so a live writer is matched by where it writes."""
    text = str(path)
    try:
        text = os.path.realpath(text)
    except Exception:
        pass
    return os.path.normcase(text)


def _live_writer_names_for_repo(
    live_paths: set, repo_type: str, repo_id: str, cache_dir: Optional[str],
) -> set:
    """Basenames of live writer paths inside this repo's blobs dirs, so a partial held open in another repo cannot
    count; if the cache cannot be enumerated, the whole set is used."""
    roots = []
    try:
        for entry in iter_active_repo_cache_dirs(repo_type, repo_id, cache_dir = cache_dir):
            roots.append(_normalized_partial_key(entry / "blobs") + os.sep)
    except Exception:
        return {os.path.basename(path) for path in live_paths}
    if not roots:
        return {os.path.basename(path) for path in live_paths}
    return {
        os.path.basename(path)
        for path in live_paths
        if any(path.startswith(root) for root in roots)
    }


def _partial_paths_with_a_live_writer() -> Optional[set]:
    """Normalized paths, not basenames, since identical files share etags across repos; None when unanswerable, so the
    caller declines; skipped processes make it a lower bound."""
    _LIVE_WRITER_WALK.complete = True
    _LIVE_WRITER_WALK.missed_our_uid = False
    try:
        import psutil  # type: ignore
    except ImportError:
        return None
    open_blobs: set = set()
    try:
        for proc in psutil.process_iter():
            try:
                for handle in proc.open_files():
                    if handle.path.endswith(INCOMPLETE_SUFFIX):
                        open_blobs.add(_normalized_partial_key(handle.path))
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                # Gone between the listing and the read, so it holds nothing: not a gap.
                continue
            except Exception:
                # psutil denies a whole process over one unstattable target, so ask the fd table which partials it
                # holds; that answer is complete for the process
                pid = getattr(proc, "pid", None)
                recovered = _proc_fd_partial_paths(pid) if pid is not None else None
                if recovered is not None:
                    open_blobs |= recovered
                    continue
                # One unreadable process does not abandon the rest of the walk, but it means the result cannot be called proof of absence
                _LIVE_WRITER_WALK.complete = False
                # WHOSE process it was decides whether cache privacy can cover for it. Privacy
                # only rules out other UIDs, so an unreadable process running as US is a writer
                # nothing here excludes -- and macOS denies `open_files()` for same-uid
                # processes under TCC and the hardened runtime, not only for other users'.
                # Recorded per process, not per platform: `/proc` existing says nothing about
                # THIS process, whose fd table the read above just failed on. A non-dumpable
                # same-uid process refuses both, and that is not only sd-pam -- an Unsloth
                # Studio backend on the same login refuses both too, and Studio downloads from
                # the Hub. What this costs is bounded below rather than by declining, since on
                # an ordinary Linux login there is always at least one such process and
                # declining outright would turn the clearance off for everyone.
                try:
                    _LIVE_WRITER_WALK.missed_our_uid = (
                        _LIVE_WRITER_WALK.missed_our_uid
                        or _process_may_be_this_user(proc, psutil)
                    )
                except Exception:
                    _LIVE_WRITER_WALK.missed_our_uid = True
                continue
    except Exception:
        return None
    return open_blobs


def _unowned_partials_safe_to_clear(
    repo_type: str,
    repo_id: str,
    cache_dir: Optional[str],
    active_grace: float,
    owned_incomplete_blobs: Optional[set],
) -> Optional[set]:
    """Partials outside the ownership set that may be removed, or None to decline; both must hold: older than
    active_grace and held open by no live process, which age alone cannot show."""
    _reset_live_writer_walk_record()
    live_writers = _partial_paths_with_a_live_writer()
    if live_writers is None:
        return None
    if not _process_walk_sees_every_writer(cache_dir):
        # A PID namespace hides a sibling container's downloader without raising, so the walk cannot record itself as
        # incomplete; decline, since nothing here is proof
        return None
    # An own-UID process that psutil and the fd table cannot describe always exists on systemd logins, so declining
    # would disable this for all Linux users; the age bar is raised tenfold to at least _OPAQUE_WRITER_GRACE, as a live
    # writer may buffer for minutes without touching its partial
    unreadable_writer_of_ours = _live_writer_walk_missed_our_own_uid()
    if unreadable_writer_of_ours:
        active_grace = max(active_grace * 10.0, _OPAQUE_WRITER_GRACE)
    if not _live_writer_walk_was_complete() and not _cache_is_private_to_this_user(
        cache_dir, repo_type = repo_type, repo_id = repo_id,
    ):
        # An unreadable process may be a writer (macOS denies open_files() for same-uid processes too): on a cache
        # another user can write, decline; on a private cache the lower bound is exact
        return None
    owned = owned_incomplete_blobs or set()
    now = time.time()
    clearable: set = set()
    try:
        for entry in iter_active_repo_cache_dirs(repo_type, repo_id, cache_dir = cache_dir):
            blobs_dir = entry / "blobs"
            if not blobs_dir.is_dir():
                continue
            for blob in blobs_dir.iterdir():
                try:
                    if not (blob.is_file() and blob.name.endswith(INCOMPLETE_SUFFIX)):
                        continue
                    # Ownership is this repo's own naming, so it compares names; the live-writer test is about the exact file on disk, so it compares paths
                    if blob.name in owned or _normalized_partial_key(blob) in live_writers:
                        continue
                    if now - blob.stat().st_mtime < active_grace:
                        continue
                except OSError:
                    continue
                clearable.add(blob.name)
    except Exception:
        return None
    return clearable


def _orphan_snapshot_links_safe_to_clear(
    repo_type: str, repo_id: str, cache_dir: Optional[str] = None
) -> Optional[list]:
    """Dangling snapshot links with no .incomplete partner (None if unreadable), so clearing them avoids the repo-wide
    force_download (issue 9094); a link with a partner may belong to a running download."""
    orphans: list = []
    try:
        for entry in _strict_repo_cache_dirs(repo_type, repo_id, cache_dir):
            for snapshot in _iter_snapshot_dirs(entry):
                for link in snapshot.rglob("*"):
                    try:
                        if not link.is_symlink() or link.exists():
                            continue
                        target = Path(os.path.realpath(link))
                        partners = _partial_partners_of(target)
                        # None is "could not read", which is not the same as "no partner" and
                        # must not license the deletion.
                        if partners is None or partners:
                            continue
                    except OSError:
                        continue
                    orphans.append(link)
    except Exception:
        return None
    return orphans


def _clear_orphan_snapshot_links(
    repo_type: str, repo_id: str, cache_dir: Optional[str] = None
) -> Optional[set]:
    """Unlinks the orphan snapshot links; returns the names that survived (locked or no permission), or None if unreadable. Survivors still need the forced download."""
    orphans = _orphan_snapshot_links_safe_to_clear(repo_type, repo_id, cache_dir)
    if orphans is None:
        return None
    survivors: set = set()
    for link in orphans:
        try:
            # Re-check at unlink: a concurrent download may finalize the blob, making this link valid, and removing it
            # would lose the only record, since a retry does not recreate old-revision links; a partial beside the
            # target means a download is in progress
            if not link.is_symlink() or link.exists():
                continue
            target = Path(os.path.realpath(link))
            partners = _partial_partners_of(target)
            if partners is None or partners:
                continue
            link.unlink()
        except OSError:
            survivors.add(link.name)
    return survivors


def _clear_unsafe_partials_for_http(
    repo_type: str,
    repo_id: str,
    *,
    cache_dir: Optional[str] = None,
    active_grace: float = DEFAULT_HTTP_STALL_TIMEOUT,
    owned_incomplete_blobs: Optional[set] = None,
) -> Optional[set]:
    """Blob-scoped clear of partials blocking an HTTP resume, avoiding the repo-wide force_download (issue 9094); age
    does not prove a writer exited, so live open files are checked; survivors still force, None if unreadable."""
    if owned_incomplete_blobs:
        _clear_partials(
            repo_type, repo_id, cache_dir = cache_dir, active_grace = active_grace,
            owned_incomplete_blobs = owned_incomplete_blobs,
            # Names captured before _default_prepare_for_http ran are no proof of ownership on older hubs, where a
            # sibling on the blob lock recreates the same .incomplete; a live writer spares it; the age guard would
            # spare our own fresh partial, which is issue 9094 itself
            owned_names_may_be_reopened = True,
        )
    clearable = _unowned_partials_safe_to_clear(
        repo_type, repo_id, cache_dir, active_grace, owned_incomplete_blobs,
    )
    if clearable:
        _clear_partials(
            repo_type, repo_id, cache_dir = cache_dir, active_grace = active_grace,
            owned_incomplete_blobs = clearable,
            # A whitelist, not a fact about our own child: re-checked at the unlink.
            ownership_is_an_earlier_scan = True,
        )
    survivors = _incomplete_partial_names(repo_type, repo_id, cache_dir)
    if survivors is None:
        return None
    # Dangling snapshot links with no partial beside them are invisible to both passes yet keep
    # has_active_incomplete_blobs true, so they are cleared too, avoiding the repo-wide force
    orphan_survivors = _clear_orphan_snapshot_links(repo_type, repo_id, cache_dir)
    if orphan_survivors is None:
        return None
    return survivors | orphan_survivors


def _child_rss(pid: int) -> Optional[int]:
    """Resident bytes of pid (None if unknown), the sensor for data received but not yet written: xet-core reconstructs
    sequentially, so bytes sit in RSS until the head-of-line xorb (up to 64 MiB) completes; rchar misses socket
    reads."""
    try:
        import psutil  # type: ignore
    except ImportError:
        psutil = None  # type: ignore
    if psutil is not None:
        try:
            return int(psutil.Process(pid).memory_info().rss)
        except Exception:
            return None
    try:
        with open(f"/proc/{pid}/status", "r") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _child_open_incomplete_blobs(pid: int) -> Optional[set]:
    """Basenames of the .incomplete blobs pid has open, i.e. this child's own partial even when resumed; None if
    undeterminable, empty if the child is not yet writing (connect phase)."""
    # psutil is cross-platform (Linux / macOS / Windows).
    try:
        import psutil  # type: ignore
    except ImportError:
        psutil = None  # type: ignore
    if psutil is not None:
        try:
            files = psutil.Process(pid).open_files()
        except Exception:
            return None
        return {os.path.basename(f.path) for f in files if f.path.endswith(INCOMPLETE_SUFFIX)}
    # Linux fallback: read open fds from /proc.
    fd_dir = f"/proc/{pid}/fd"
    try:
        entries = os.listdir(fd_dir)
    except OSError:
        return None  # no /proc (non-Linux) or process gone
    open_blobs: set = set()
    for fd in entries:
        try:
            target = os.readlink(os.path.join(fd_dir, fd))
        except OSError:
            continue
        if target.endswith(INCOMPLETE_SUFFIX):
            open_blobs.add(os.path.basename(target))
    return open_blobs


def get_hf_download_state(
    repo_ids: Optional[list[str]] = None,
    *,
    repo_type: Optional[str] = "model",
    cache_dir: Optional[str] = None,
) -> Optional[tuple[int, bool]]:
    """(total_on_disk_bytes, has_incomplete) for the HF cache, sparse-aware so a partial is not counted as full
    progress; a missing cache gives (0, False), a probe exception gives None."""
    try:
        if hf_cache_root(cache_dir = cache_dir) is None:
            return (0, False)

        total = 0
        has_incomplete = False
        for repo_id in repo_ids or []:
            # Skip local paths: HF IDs never start with / . ~, contain "\", or a drive-letter ":".
            if (
                not repo_id
                or repo_id.startswith(("/", ".", "~"))
                or "\\" in repo_id
                or ":" in repo_id
            ):
                continue
            for entry in iter_active_repo_cache_dirs(repo_type, repo_id, cache_dir = cache_dir):
                blobs_dir = entry / "blobs"
                if not blobs_dir.is_dir():
                    continue
                for blob in blobs_dir.iterdir():
                    try:
                        if blob.is_file():
                            total += blob_bytes_present(blob)
                    except OSError:
                        pass
            if has_active_incomplete_blobs(repo_type, repo_id, cache_dir = cache_dir):
                has_incomplete = True
        return (total, has_incomplete)
    except Exception as e:
        logger.debug("Failed to determine HF download state: %s", e)
        return None


def start_watchdog(
    *,
    repo_ids: list[str],
    on_stall: Callable[[str], None],
    repo_type: Optional[str] = "model",
    cache_dir: Optional[str] = None,
    interval: Optional[float] = None,
    stall_timeout: Optional[float] = None,
    connect_timeout: Optional[float] = None,
    heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
    xet_disabled: bool = False,
    on_heartbeat: Optional[Callable[[str], None]] = None,
    watch_new_partials_only: bool = False,
    baseline_incomplete_blobs: Optional[set] = None,
    child_pid: Optional[int] = None,
    watch_connect: Optional[bool] = None,
) -> threading.Event:
    """Daemon watchdog that fires on_stall once: a flat size trips stall_timeout (30s Xet) once a partial exists, else
    connect_timeout (90s); env UNSLOTH_XET_STALL_TIMEOUT / UNSLOTH_XET_CONNECT_TIMEOUT; watch_new_partials_only
    follows only the child's own partial."""
    stop = threading.Event()
    transport = "https" if xet_disabled else "xet"
    fired = False
    if stall_timeout is None:
        stall_timeout = _env_seconds(
            "UNSLOTH_XET_STALL_TIMEOUT",
            DEFAULT_HTTP_STALL_TIMEOUT if xet_disabled else DEFAULT_STALL_TIMEOUT,
        )
    if connect_timeout is None:
        connect_timeout = _env_seconds("UNSLOTH_XET_CONNECT_TIMEOUT", DEFAULT_CONNECT_TIMEOUT)
    if interval is None:
        # Never sample slower than the tighter of the two deadlines, or the timeout is meaningless.
        interval = min(DEFAULT_POLL_INTERVAL, max(1.0, min(stall_timeout, connect_timeout) / 4.0))
    baseline = set(baseline_incomplete_blobs or ())
    connect_clock = (child_pid is not None) if watch_connect is None else bool(watch_connect)
    single_repo_id = repo_ids[0] if repo_ids else ""

    def _partial_bytes(ids) -> int:
        """Bytes sitting in ACTIVE ``*.incomplete`` partials, nothing else."""
        total = 0
        for one in ids or []:
            total += sum(_active_incomplete_blob_sizes(repo_type, one, cache_dir).values())
        return total

    def _measure() -> Optional[tuple[int, bool, int]]:
        """Returns (bytes_watched, has_incomplete, bytes_in_active_partials); bytes_watched includes blobs already
        complete before this download, so only the third element shows the current transfer got a first byte."""
        if watch_new_partials_only:
            sizes = _active_incomplete_blob_sizes(repo_type, single_repo_id, cache_dir)
            open_names = _child_open_incomplete_blobs(child_pid) if child_pid else None
            if open_names is not None:
                # Only partials this child holds open: a resume reusing a baseline name still counts and siblings do
                # not; an empty set means no partial yet (connect or metadata), not a sibling's
                owned = {name: n for name, n in sizes.items() if name in open_names}
                owned_bytes = sum(owned.values())
                return (owned_bytes, len(owned) > 0, owned_bytes)
            if child_pid:
                # With no psutil and no /proc the child's open files are unknown; name filtering would forever exclude a
                # resumed partial reusing a baseline name, so use the repo-wide measure, accepting that a sibling may
                # mask a stall
                state = get_hf_download_state(
                    [single_repo_id], repo_type = repo_type, cache_dir = cache_dir
                )
                return None if state is None else (state[0], state[1], sum(sizes.values()))
            # No child pid at all: follow only newly-created (post-baseline) partials.
            owned = {name: n for name, n in sizes.items() if name not in baseline}
            owned_bytes = sum(owned.values())
            return (owned_bytes, len(owned) > 0, owned_bytes)
        state = get_hf_download_state(repo_ids, repo_type = repo_type, cache_dir = cache_dir)
        return None if state is None else (state[0], state[1], _partial_bytes(repo_ids))

    def _beat() -> None:
        nonlocal fired
        state = _measure()
        last_size = state[0] if state is not None else 0
        last_change = time.monotonic()
        last_heartbeat = 0.0
        # False until a byte is observed, not on partial existence: hub and hf_xet create .incomplete before CAS setup,
        # which would give a slow connect the data deadline; a partial vanishing after bytes flowed means finished
        seen_bytes = bool(state[2]) if state is not None else False
        # Sticky, unlike seen_bytes: it only picks the did not resume wording for the gap between files, and must not
        # reset when the partial set momentarily empties
        seen_bytes_ever = seen_bytes
        peer_wait_start: Optional[float] = None
        # peer_hold_start serves the branches that suppress on a peer after bytes were seen, since peer_wait_start is
        # cleared then; without a ceiling a stale peer partial could hold the clock forever
        peer_hold_start: Optional[float] = None
        last_rss = (_child_rss(child_pid) if child_pid else None) or 0
        peer_size: Optional[int] = None

        def _peer_progressing() -> bool:
            """Is another writer growing this repo? A waiter queued on huggingface_hub's per-file lock owns no
            .incomplete and looks hung; repo-wide growth separates queued from hung, else a long first download fails
            concurrent callers at connect_timeout."""
            nonlocal peer_size
            if not watch_new_partials_only:
                return False  # the snapshot measure is already repo-wide
            peer = get_hf_download_state(repo_ids, repo_type = repo_type, cache_dir = cache_dir)
            current = peer[0] if peer is not None else None
            if current is None or current != peer_size:
                peer_size = current
                return True
            # Flat disk is not proof the peer is dead (a healthy Xet transfer writes sequentially and can sit unchanged
            # for minutes); a fresh peer-owned partial is the liveness signal, bounded by PEER_SUPPRESSION_CEILING
            # rather than an mtime that races link speed
            fresh = PEER_SUPPRESSION_CEILING
            now_wall = time.time()
            for one in repo_ids or []:
                for entry in iter_active_repo_cache_dirs(repo_type, one, cache_dir = cache_dir):
                    blobs_dir = entry / "blobs"
                    if not blobs_dir.is_dir():
                        continue
                    try:
                        for blob in blobs_dir.iterdir():
                            try:
                                if (
                                    blob.name.endswith(INCOMPLETE_SUFFIX)
                                    and blob.is_file()
                                    and now_wall - blob.stat().st_mtime < fresh
                                ):
                                    return True
                            except OSError:
                                continue
                    except OSError:
                        continue
            return False

        def _waiting_on_a_peers_partial() -> bool:
            """Repo-wide mode: a child parked on another process's per-blob lock is not stalled, so killing it would
            record a false Xet failure; bounded by PEER_SUPPRESSION_CEILING, not partial mtime, whose window expires
            on slow healthy peers."""
            if watch_new_partials_only or not child_pid:
                return False
            owned = _child_open_incomplete_blobs(child_pid)
            if owned is None or owned:
                # Uninspectable -> keep the disk-only verdict. Non-empty -> the partial is ours.
                return False
            fresh = PEER_SUPPRESSION_CEILING
            now_wall = time.time()
            for one in repo_ids or []:
                for entry in iter_active_repo_cache_dirs(repo_type, one, cache_dir = cache_dir):
                    blobs_dir = entry / "blobs"
                    if not blobs_dir.is_dir():
                        continue
                    try:
                        for blob in blobs_dir.iterdir():
                            try:
                                if (
                                    blob.name.endswith(INCOMPLETE_SUFFIX)
                                    and blob.is_file()
                                    and now_wall - blob.stat().st_mtime < fresh
                                ):
                                    return True
                            except OSError:
                                continue
                    except OSError:
                        continue
            return False

        def _child_exited(pid: Optional[int]) -> bool:
            """True once the supervised child has exited (False if no pid or no /proc); an exited child is not hung, so no connect trip should fire for it."""
            if not pid or not os.path.isdir("/proc"):
                return False
            return not os.path.isdir(f"/proc/{pid}")

        def _peer_hold_ok(now: float) -> bool:
            """May a peer keep holding the clock off? Bounded by PEER_SUPPRESSION_CEILING."""
            nonlocal peer_hold_start
            if not _waiting_on_a_peers_partial():
                return False
            if peer_hold_start is None:
                peer_hold_start = now
                return True
            return now - peer_hold_start < PEER_SUPPRESSION_CEILING

        def _trip(message: str) -> None:
            nonlocal fired
            if not fired:
                fired = True
                on_stall(message)

        while not stop.wait(interval):
            state = _measure()
            now = time.monotonic()

            if state is None:
                # Unmeasurable this tick (transient FS error): treat as progress so the gap cannot trip a false stall once readable again
                last_change = now
                _safe_status(on_heartbeat, f"Downloading ({transport} transport)...")
                continue

            current_size, has_incomplete, partial_bytes = state
            rss = _child_rss(child_pid) if child_pid else None
            if current_size != last_size:
                last_size = current_size
                last_change = now
                # After the buffer drains, re-base last_rss on the current reading, not a lifetime peak: xet's buffer is
                # a process-wide budget reused across files, so a high-water mark would hide the next fill
                if rss is not None:
                    last_rss = rss
                peer_hold_start = None      # our own transfer moved: we are not holding on a peer
            # Phase on partial bytes open right now, re-evaluated each tick: a run-once latch charged file N+1's empty
            # .incomplete the data deadline; last_size and last_change stay repo-wide
            seen_bytes = bool(partial_bytes)
            if seen_bytes:
                seen_bytes_ever = True
                peer_wait_start = None      # a byte landed: we are not a lock waiter

            elapsed = now - last_change
            if has_incomplete and seen_bytes:
                # A frozen byte count is a hang unless the child is filling its reconstruction buffer, which touches no
                # disk until the head-of-line block lands; see _child_rss
                if rss is not None and rss < last_rss:
                    # When the buffer is handed back to the allocator, the RSS baseline is the trough since the last
                    # reset, never a peak, or growth below it reads as no progress
                    last_rss = rss
                if rss is not None and rss > last_rss + _RSS_PROGRESS_EPSILON:
                    last_rss = rss
                    last_change = now
                elif elapsed >= _deadline(rss, last_rss, stall_timeout):
                    if _peer_hold_ok(now):
                        # Someone else owns this transfer; we are a lock waiter, not a stall.
                        last_change = now
                        continue
                    _trip(
                        f"Download appears stalled ({transport} transport) "
                        f"-- no progress for {int(elapsed)}s"
                    )
                    return
            elif not has_incomplete and seen_bytes_ever:
                # After bytes flowed, no open partial means finished or between files, so the connect clock governs the
                # gap instead of resetting; it applies only with a child to kill, since a wrapped from_pretrained looks
                # finished while quantising
                if (
                    connect_clock
                    and not _child_exited(child_pid)
                    and elapsed >= connect_timeout
                    and not _peer_hold_ok(now)
                ):
                    _trip(
                        f"Download did not resume ({transport} transport) "
                        f"-- no data for {int(elapsed)}s"
                    )
                    return
            else:
                # No byte in any open partial means stuck before the first byte or queued on the hub's per-file lock;
                # the peer check accepts any fresh repo partial (filelock closes its fd between polls), bounded by
                # PEER_SUPPRESSION_CEILING
                if _peer_progressing() and (
                    peer_wait_start is None
                    or now - peer_wait_start < PEER_SUPPRESSION_CEILING
                ):
                    if peer_wait_start is None:
                        peer_wait_start = now
                    last_change = now
                elif (
                    connect_clock
                    and not _child_exited(child_pid)
                    and elapsed >= connect_timeout
                ):
                    _trip(
                        f"Download did not start ({transport} transport) "
                        f"-- no data after {int(elapsed)}s"
                    )
                    return

            if now - last_heartbeat >= heartbeat_interval:
                last_heartbeat = now
                _safe_status(on_heartbeat, f"Downloading ({transport} transport)...")

    threading.Thread(target = _beat, daemon = True, name = "hf-xet-watchdog").start()
    return stop


def _deadline(rss: Optional[int], trough: int, stall_timeout: float) -> float:
    """A flat disk is a hang only if RSS is flat too: RSS growth since the trough means the child is receiving on a slow
    link, so it gets the patient HTTP threshold; flat or unreadable RSS keeps the short one."""
    if rss is not None and rss > trough + _RSS_SLOW_LINK_EPSILON:
        return max(stall_timeout, DEFAULT_HTTP_STALL_TIMEOUT)
    return stall_timeout


def _scrub_in_child(text: str, token: Optional[str]) -> str:
    """Redact secrets from a child error string, preferring Unsloth's patterns if present."""
    try:
        from hub.utils.download_registry import scrub_secrets  # type: ignore

        return scrub_secrets(text, hf_token = token)
    except Exception:
        return _default_scrub_secrets(text, hf_token = token)


# Deterministic Hub errors recur identically over either transport, so HTTP retry is pointless; matched by class name so
# the parent need not import HF's error classes
_DETERMINISTIC_ERROR_NAMES = frozenset({
    "RepositoryNotFoundError",
    "RevisionNotFoundError",
    "EntryNotFoundError",
    "GatedRepoError",
    "DisabledRepoError",
    "LocalEntryNotFoundError",
    "LocalTokenNotFoundError",  # a missing required token fails identically either way
    "BadRequestError",
    "HFValidationError",        # a malformed repo id / revision never reaches the network
    # An OSError (via builtin ConnectionError) with no errno, so nothing below catches it and the rung
    # default would spend an HTTP child plus the destructive purge on a repo the user switched offline.
    "OfflineModeIsEnabled",
})
# Type rebuilt across the spawn but not retry-deterministic: HfHubHTTPError spans 4xx and 5xx/429, so retry stays
# status-driven, and the real type is re-raised so except HfHubHTTPError keeps working
_TYPE_PRESERVE_ONLY_NAMES = frozenset({
    "HfHubHTTPError",
})
# Substrings marking a transient transport failure (hf_xet / CAS error, timeout, reset, 5xx / 429) that an HTTP retry may recover from
_TRANSIENT_ERROR_HINTS = (
    "xet", "casclient", "cas_", "timeout", "timed out", "connection", "reset by peer",
    "temporarily", "try again", "incompleteread", "protocolerror", "remotedisconnected",
    "broken pipe", "ssl", "eof occurred", "502", "503", "504", "500 server", "429",
    "too many requests", "service unavailable", "bad gateway", "gateway time",
    "connection aborted",
)


def _resolve_exception_class(type_name: str) -> "Optional[type]":
    """Maps a child's error class name back to its class so the parent re-raises the original type, including builtin OSError subclasses; unknown names give None."""
    if type_name == "OSError":
        return OSError
    # Preserve builtin OSError subclasses (PermissionError, FileNotFoundError, ...): deterministic FS failures the child
    # cannot retry away, so caller except clauses still fire, not RuntimeError
    builtin_cls = getattr(builtins, type_name, None)
    if isinstance(builtin_cls, type) and issubclass(builtin_cls, OSError):
        return builtin_cls
    if type_name not in _DETERMINISTIC_ERROR_NAMES and type_name not in _TYPE_PRESERVE_ONLY_NAMES:
        return None
    for module_name in ("huggingface_hub.errors", "huggingface_hub.utils"):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        cls = getattr(module, type_name, None)
        if isinstance(cls, type) and issubclass(cls, BaseException):
            return cls
    return None


def _instantiate_preserving_type(exc_cls: type, message: str) -> "Optional[BaseException]":
    """Builds exc_cls(message) even when __init__ needs more args (HfHubHTTPError may require response): tries normal
    constructors, then __new__, so type and message survive; None only if __new__ fails."""
    for build in (
        lambda: exc_cls(message),
        lambda: exc_cls(message, response = None),
    ):
        try:
            return build()
        except Exception:
            continue
    try:
        exc = exc_cls.__new__(exc_cls)
        BaseException.__init__(exc, message)
        return exc
    except Exception:
        return None


def _parse_errno(message: str) -> "Optional[int]":
    """Recovers the errno from a stringified OSError ('[Errno N] ...'), so a disk-full or quota error keeps its code across the spawn boundary for exc.errno checks."""
    match = re.search(r"\[Errno (\d+)\]", message)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _is_builtin_oserror(exc: BaseException) -> bool:
    """True only for a builtin OSError (or subclass) with a real errno; requests and HF HTTP errors subclass OSError
    without one, so bracketed [Errno N] text in their message is ignored."""
    if not isinstance(exc, OSError):
        return False
    builtin = getattr(builtins, type(exc).__name__, None)
    return isinstance(builtin, type) and issubclass(builtin, OSError) and isinstance(exc, builtin)


def _raise_child_error(message: str) -> None:
    """Re-raises a child's 'ClassName: message' error as its original type so except clauses still match across the
    spawn; unknown or uninstantiable classes raise RuntimeError."""
    type_name = message.split(":", 1)[0].strip() if ":" in message else ""
    exc_cls = _resolve_exception_class(type_name)
    if exc_cls is None:
        raise RuntimeError(message)
    exc = _instantiate_preserving_type(exc_cls, message)
    if exc is None:
        raise RuntimeError(message)
    if _is_builtin_oserror(exc) and getattr(exc, "errno", None) is None:
        # Carry errno (ENOSPC, EDQUOT, ...) across the spawn for builtin OSError subclasses only, since HF HTTP errors'
        # [Errno N] is not real; set as an attribute, because LocalEntryNotFoundError's single-arg __init__ rejects a
        # second arg and a two-arg form doubles the prefix
        errno_val = _parse_errno(message)
        if errno_val is not None:
            try:
                exc.errno = errno_val
            except Exception:
                pass
    raise exc


def _is_retryable_download_error(exc: BaseException, *, on_xet: bool) -> bool:
    """True for transient transport failures (CAS, reset, timeout, 5xx, 429) worth an HTTP retry, not deterministic Hub
    errors (auth, not-found, gated, disk-full); on Xet an unrecognized error counts as transport, since one HTTP
    attempt can disprove it; on HTTP it is surfaced."""
    name = type(exc).__name__
    # Control flow, not transport: KeyboardInterrupt, SystemExit and GeneratorExit must not be retried, which would
    # spend an HTTP child and the purge; programming errors stay unlisted and get one HTTP attempt
    if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
        return False
    # LocalEntryNotFoundError wraps both a deterministic offline miss and a transient HEAD connection error or timeout,
    # so only the transient hint cases are retried
    if name == "LocalEntryNotFoundError" and any(
        hint in f"{name}: {exc}".lower() for hint in _TRANSIENT_ERROR_HINTS
    ):
        return True
    if name in _DETERMINISTIC_ERROR_NAMES:
        return False
    # Disk full or quota cannot be helped by another transport; errno is read defensively, since an AttributeError here
    # would turn the report into a crashed verdict that spends the budget
    if isinstance(exc, OSError) and getattr(exc, "errno", None) in _DISK_FULL_ERRNOS:
        return False
    # HTTP status (HfHubHTTPError carries a requests or httpx response): 5xx, 429 and 408 are transient; other 4xx such
    # as 401, 403, 404 and 416 are deterministic
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if not isinstance(status, int):
        status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status >= 500 or status in (408, 429)
    # Builtin OSError is decided by class before the text scan, since str() embeds the filename under the xet cache dir
    # and would match the xet hint; network subclasses stay retryable by type
    if _is_builtin_oserror(exc):
        # Exception: the Hub's bare EnvironmentError for a short download is retryable, matched on the Hub's own wording
        # since a filename can collide with the xet hint
        if _HUB_CONSISTENCY_ERROR_RE.search(str(exc)):
            return True
        return isinstance(exc, (ConnectionError, TimeoutError, BrokenPipeError, BlockingIOError))
    text = f"{name}: {exc}".lower()
    if any(hint in text for hint in _TRANSIENT_ERROR_HINTS):
        return True
    return on_xet


def _child_download(*, kind: str, params: dict, token: Optional[str], repo_type: str) -> str:
    """Run the actual HF download for one attempt inside the spawn child."""
    if kind == "snapshot":
        from huggingface_hub import snapshot_download

        return snapshot_download(
            repo_id = params["repo_id"],
            repo_type = repo_type,
            token = token,
            revision = params.get("revision"),
            cache_dir = params.get("cache_dir"),
            allow_patterns = params.get("allow_patterns"),
            ignore_patterns = params.get("ignore_patterns"),
            force_download = params.get("force_download", False),
        )

    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id = params["repo_id"],
        filename = params["filename"],
        subfolder = params.get("subfolder"),
        repo_type = repo_type,
        token = token,
        revision = params.get("revision"),
        cache_dir = params.get("cache_dir"),
        force_download = params.get("force_download", False),
    )


def _download_child_entry(
    *,
    kind: str,
    params: dict,
    token: Optional[str],
    repo_type: str,
    disable_xet: bool,
    result_queue: Any,
) -> None:
    """Spawn-child entry point (picklable): sets Xet env before importing huggingface_hub, makes its own process group
    so the parent can kill the whole transfer, never logs tokens or signed URLs."""
    # Die with the parent on Linux under Unsloth (best-effort; module absent standalone).
    try:
        from utils.process_lifetime import bind_current_process_to_parent_lifetime  # type: ignore

        bind_current_process_to_parent_lifetime()
    except Exception:
        pass

    if hasattr(os, "setsid"):
        try:
            os.setsid()
        except OSError:
            pass

    if disable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        # Keep the HTTP writer sequential and resumable (hf_transfer's sparse partials cannot).
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    else:
        # Defensive: the parent seeds these around the spawn, but a child launched by any other path must not inherit
        # hf_xet's unbounded multi-GB buffer defaults
        try:
            from unsloth_zoo_downloads.hf_xet_tuning import resize_for_cache_dir

            # A supervised child takes the shortened Xet timeouts so a failure reaches the ladder instead of being
            # retried for minutes; resize, not apply, since setdefault would keep import-time sizing
            resize_for_cache_dir(os.environ, params.get("cache_dir"))
        except Exception as e:
            logger.debug("Could not apply Xet tuning in child: %s", e)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    repo_id = params["repo_id"]

    # Test-only fault injection, never set in production: UNSLOTH_HF_XET_FORCE_STALL=1 stalls the Xet attempt to
    # exercise the watchdog and HTTP fallback against a real repo
    if not disable_xet and os.environ.get("UNSLOTH_HF_XET_FORCE_STALL") == "1":
        _stall_fh = None
        try:
            from huggingface_hub.constants import HF_HUB_CACHE

            # Write the fake partial under the repo_type-correct dir the watchdog scans.
            cache_root = params.get("cache_dir") or HF_HUB_CACHE
            repo_dir_name = f"{repo_type or 'model'}s--" + repo_id.replace("/", "--")
            blobs = os.path.join(cache_root, repo_dir_name, "blobs")
            os.makedirs(blobs, exist_ok = True)
            # Hold the partial open for the whole stall, since the single-file watchdog counts only partials this PID
            # holds open; bound to a local so it stays open across the sleep
            _stall_fh = open(os.path.join(blobs, "xet-force-stall.incomplete"), "wb")
            _stall_fh.write(b"\0" * 4096)
            _stall_fh.flush()
        except OSError:
            pass
        while True:
            time.sleep(3600)

    try:
        path = _child_download(kind = kind, params = params, token = token, repo_type = repo_type)
        result_queue.put({"ok": True, "path": path})
    except BaseException as e:  # noqa: BLE001 - report every failure to the parent
        # Classify here, where status, errno and type are intact; the rung decides unrecognized errors, and only this child knows which rung it ran on
        result_queue.put({
            "ok": False,
            "error": _scrub_in_child(f"{type(e).__name__}: {e}", token),
            "retryable": _is_retryable_download_error(e, on_xet = not disable_xet),
        })


def _terminate_process_group(proc: "mp.process.BaseProcess", grace_period: float) -> None:
    """Kills proc's whole process group (Xet may spawn helpers): SIGTERM, then SIGKILL after grace_period; signals the
    group only once getpgid(pid) == pid confirms the child leads it."""
    pid = proc.pid

    def _signal_group(sig: int) -> None:
        # Signal the group only once the child leads it (pgid == pid after setsid): before that its pid could name a
        # recycled group; otherwise (Windows has no killpg or getpgid) signal the process alone
        if pid is not None and hasattr(os, "killpg") and hasattr(os, "getpgid"):
            try:
                if os.getpgid(pid) == pid:
                    os.killpg(pid, sig)
                    return
            except (ProcessLookupError, PermissionError, OSError):
                pass
        try:
            proc.terminate() if sig != getattr(signal, "SIGKILL", -9) else proc.kill()
        except Exception:
            pass

    _signal_group(getattr(signal, "SIGTERM", signal.SIGINT))
    proc.join(timeout = grace_period)
    # SIGKILL only while the child is alive: after join() reaps the leader, its pid could be recycled into an unrelated
    # group that killpg would hit; hf_xet 1.5.x spawns no helpers, so nothing else needs cleanup
    if proc.is_alive():
        # The -9 sentinel takes the force-kill branch on Windows, where signal.SIGKILL is undefined; there proc.kill()
        # equals proc.terminate(), so the call site stays consistent
        _signal_group(getattr(signal, "SIGKILL", -9))
        proc.join(timeout = 5.0)


def _run_download_attempt(
    repo_id: str,
    *,
    kind: str,
    params: dict,
    token: Optional[str],
    repo_type: str,
    disable_xet: bool,
    cancel_event: Optional[threading.Event],
    stall_timeout: Optional[float],
    interval: Optional[float],
    grace_period: float,
    on_status: Optional[Callable[[str], None]],
) -> tuple[str, Optional[str]]:
    """Runs one download in a spawn child under the watchdog and returns a tagged result: ok, stall, cancelled, crashed,
    retryable_error (transient) or error (deterministic); tests patch this seam."""
    # Partials present before spawning are the baseline, so a new one is provably this child's; without that, Windows
    # has no ownership evidence and the repo-wide force re-downloads every completed shard (issue 9094); sound because a
    # fired stall means nothing in the whole repo grew
    baseline_partials: Optional[set] = _baseline_incomplete_blob_names(
        repo_type, repo_id, params.get("cache_dir")
    )
    result_queue: Any = _CTX.Queue()
    proc = _CTX.Process(
        target = _download_child_entry,
        kwargs = dict(
            kind = kind,
            params = params,
            token = token,
            repo_type = repo_type,
            disable_xet = disable_xet,
            result_queue = result_queue,
        ),
        daemon = True,
    )
    # Set the transport env in this process around the spawn: HF caches HF_HUB_DISABLE_XET at import and the child
    # re-imports huggingface_hub, so a child-side set is too late
    child_env = {
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        # Skip unsloth_zoo's heavy torch / transformers / device init in the child.
        "UNSLOTH_ZOO_DISABLE_GPU_INIT": "1",
    }
    if disable_xet:
        child_env["HF_HUB_DISABLE_XET"] = "1"
        child_env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    with _SPAWN_ENV_LOCK:
        if not disable_xet:
            # hf_xet reads its buffer caps natively at import, so they must be in the environment before the child
            # starts; setting them inside the child is too late
            try:
                from unsloth_zoo_downloads.hf_xet_tuning import resize_for_cache_dir

                # Resize, not apply, against a copy of the real env so explicit user settings such as
                # HF_XET_HIGH_PERFORMANCE stand; UNSLOTH_XET_FORCE_CAPS=1 still caps; disk clamp redone for cache_dir,
                # which overrides HF_HUB_CACHE; under the lock so a concurrent overlay is not read as user settings
                child_env.update(resize_for_cache_dir(dict(os.environ), params.get("cache_dir")))
            except Exception as e:
                logger.debug("Could not compute Xet tuning env: %s", e)
        # Import huggingface_hub.constants in the parent before the child-only HF_HUB_DISABLE_XET=1 is set, else a
        # concurrent first import caches the disabled value and routes downloads over HTTP
        try:
            import huggingface_hub.constants  # noqa: F401
        except Exception:
            pass
        saved_env = {k: os.environ.get(k) for k in child_env}
        # spawn rebuilds __main__ from __main__.__file__: pseudo-paths such as <stdin> or a notebook fail, and an
        # unguarded script would re-run its top level as __mp_main__, so __main__ is pointed at this module
        main_module = sys.modules.get("__main__")
        saved_main_file = _UNSET
        saved_main_spec = _UNSET
        if main_module is not None:
            saved_main_file = getattr(main_module, "__file__", _UNSET)
            main_module.__file__ = __file__
            # With python -m pkg, spawn re-imports by __spec__.name, so __spec__ is cleared to force the __file__ path branch
            saved_main_spec = getattr(main_module, "__spec__", _UNSET)
            main_module.__spec__ = None
        try:
            os.environ.update(child_env)
            proc.start()
        except BaseException:
            # proc.start() can raise OSError under fd or thread exhaustion before the lifecycle finally runs, so the
            # queue's pipe fds are closed here to avoid an fd leak
            try:
                result_queue.cancel_join_thread()
                result_queue.close()
            except Exception:
                pass
            raise
        finally:
            for k, v in saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            if main_module is not None:
                if saved_main_file is _UNSET:
                    # __file__ was absent before; remove the one we added.
                    try:
                        delattr(main_module, "__file__")
                    except AttributeError:
                        pass
                else:
                    main_module.__file__ = saved_main_file
                if saved_main_spec is _UNSET:
                    try:
                        delattr(main_module, "__spec__")
                    except AttributeError:
                        pass
                else:
                    main_module.__spec__ = saved_main_spec

    # Bind the child to the parent lifetime when running under Unsloth (best-effort).
    try:
        from utils.process_lifetime import adopt_pid  # type: ignore

        adopt_pid(proc.pid)
    except Exception:
        pass

    stalled = threading.Event()
    # stall_verdict records the verdict, not just a stall: the ladder retries a data-phase hang over Xet but sends a
    # pre-first-byte one straight to HTTP; written before the event is set so readers see it
    stall_verdict: list = []
    # If start_watchdog raises, the already-started child must still be reaped, so the call sits inside the try whose
    # finally reaps it; stop_watchdog stays None until it works
    stop_watchdog = None
    result: Optional[dict] = None

    def _note_stall(message: str) -> None:
        stall_verdict.append(message)
        stalled.set()

    try:
        stop_watchdog = start_watchdog(
            repo_ids = [repo_id],
            on_stall = _note_stall,
            repo_type = repo_type,
            cache_dir = params.get("cache_dir"),
            interval = interval,
            stall_timeout = stall_timeout,
            xet_disabled = disable_xet,
            on_heartbeat = on_status,
            watch_new_partials_only = (kind == "file"),
            # A snapshot must keep measuring the whole repo; only single-file narrows.
            baseline_incomplete_blobs = baseline_partials if kind == "file" else None,
            child_pid = proc.pid,
        )
        while proc.is_alive():
            if cancel_event is not None and cancel_event.is_set():
                _terminate_process_group(proc, grace_period)
                return ("cancelled", None)
            if stalled.is_set():
                # Prefer a result enqueued in the watchdog's fire window so a just-succeeded download is not killed; use
                # a short get timeout, since the feeder may not have flushed the put
                try:
                    result = result_queue.get(timeout = 1.0)
                    break
                except queue.Empty:
                    pass
                # Capture this child's partials before killing it so the HTTP purge is scoped: per-pid open fds first,
                # then post-baseline partials, else None for the coarser mtime guard
                owned = _child_open_incomplete_blobs(proc.pid) if proc.pid else None
                child_open = set(owned) if owned else set()
                needs_inference = not owned and baseline_partials is not None
                if needs_inference:
                    # Kill the child before inferring from live writers: asking first subtracted our own held partial
                    # and inferred nothing on hosts with no per-pid answer (Windows, macOS open_files denied)
                    _terminate_process_group(proc, grace_period)
                if needs_inference:
                    # None (no psutil or /proc) or an empty set (child stalled before opening any partial): try the post-baseline diff before giving up on scope
                    current = set(
                        _active_incomplete_blob_sizes(repo_type, repo_id, params.get("cache_dir"))
                    )
                    inferred = current - baseline_partials
                    # Post-baseline partials do not prove ownership (a same-repo sibling can create them in the same
                    # window); names held by live writers are dropped only when the walk saw every process or the cache
                    # is private, otherwise nothing is inferred
                    _reset_live_writer_walk_record()
                    writer_paths = _partial_paths_with_a_live_writer()
                    trustworthy = writer_paths is not None and _process_walk_sees_every_writer(
                        params.get("cache_dir")
                    ) and (
                        _live_writer_walk_was_complete()
                        or (
                            _cache_is_private_to_this_user(
                                params.get("cache_dir"),
                                repo_type = repo_type,
                                repo_id = repo_id,
                            )
                            # Cache privacy excludes other UIDs only, so an unreadable own-UID process must still void
                            # the inference; otherwise a sibling's partial is claimed as ours and a live download
                            # unlinked; declining costs bandwidth
                            and not _live_writer_walk_missed_our_own_uid()
                        )
                    )
                    if not trustworthy:
                        inferred = set()
                    else:
                        # inferred names partials in this repo's cache, so the host-wide live-writer walk is projected
                        # onto it; a writer busy in another repo says nothing about a same-etag partial here
                        writers = _live_writer_names_for_repo(
                            writer_paths, repo_type, repo_id, params.get("cache_dir"),
                        )
                        inferred -= writers - child_open
                    owned = inferred
                # An empty ownership set would scope the purge to nothing and leave stale partials for the retry to
                # re-trip on; use None (unscoped) so the mtime and partner guards still apply
                params["_owned_incomplete_blobs"] = owned or None
                if not needs_inference:
                    _terminate_process_group(proc, grace_period)
                return ("stall", stall_verdict[0] if stall_verdict else "")
            try:
                result = result_queue.get(timeout = _POLL_INTERVAL)
                break
            except queue.Empty:
                continue
        else:
            # Drain any enqueued result with a short get timeout, not get_nowait(): the child can exit before its feeder
            # flushes the pipe, which would look falsely resultless
            try:
                result = result_queue.get(timeout = 1.0)
            except queue.Empty:
                result = None
    finally:
        if stop_watchdog is not None:
            stop_watchdog.set()
        proc.join(timeout = grace_period)
        # No loop exit may leak the child; _terminate_process_group is idempotent, so a redundant call is a no-op
        if proc.is_alive():
            _terminate_process_group(proc, grace_period)
        # Release the queue's pipe fds now rather than at GC; cancel the feeder before close, since a killed child has nothing to flush
        try:
            result_queue.cancel_join_thread()
            result_queue.close()
        except Exception:
            pass

    if result is None:
        # Child exited without a result: a process-level crash (native hf_xet abort or segfault), not a captured
        # exception, so the other transport may still succeed; reported as "crashed" here.
        return (
            "crashed",
            f"download process for '{repo_id}' exited "
            f"(code={proc.exitcode}) without a result",
        )
    if result.get("ok"):
        return ("ok", result["path"])
    message = result.get("error") or "unknown download error"
    if result.get("retryable"):
        return ("retryable_error", message)
    return ("error", message)


def _intact_subset(
    snapshot_dir: Path, *, repo_type: str, allow_patterns: Any, ignore_patterns: Any,
) -> bool:
    """True when the selected files show no interrupted download: no dangling requested symlink and every exact-named
    requested file present; a dangling excluded weight does not reject."""
    return (
        not snapshot_has_requested_broken_symlinks(
            snapshot_dir, allow_patterns = allow_patterns, ignore_patterns = ignore_patterns,
            repo_type = repo_type,
        )
        and requested_named_files_present(
            snapshot_dir, allow_patterns = allow_patterns, ignore_patterns = ignore_patterns
        )
    )


def _is_default_load_weight_file(name: str) -> bool:
    """Only safetensors or bin count as a default-load weight; a cached gguf must not satisfy the check, or the missing safetensors is fetched over un-killable Xet."""
    return _is_loadable_weight_file(name) and name.endswith((".safetensors", ".bin"))


# Default root weight names only (incl. numbered shards); variants, adapters, gguf must not match
_CANONICAL_ROOT_MODEL_WEIGHT_RE = re.compile(
    r"^(?:model|pytorch_model)(?:-\d{5}-of-\d{5})?\.(?:safetensors|bin)$"
)

# Canonical component weight a plain pipeline load reads; variant-named (fp16) weights are excluded
_CANONICAL_COMPONENT_WEIGHT_RE = re.compile(
    r"^(?:diffusion_pytorch_model|model|pytorch_model)(?:-\d{5}-of-\d{5})?\.(?:safetensors|bin)$"
)

# Single-file TF/Flax root weight only; a sharded set is judged via its index, not a lone shard
_CANONICAL_ROOT_TF_FLAX_WEIGHT_RE = re.compile(r"^(?:tf_model\.h5|flax_model\.msgpack)$")

# The shard-index sidecars a sharded TF / Flax weight is enumerated through.
_TF_FLAX_WEIGHT_INDEX_NAMES = ("tf_model.h5.index.json", "flax_model.msgpack.index.json")


def _pytorch_root_weight_formats_ignored(ignore_patterns: Any) -> bool:
    """True when the ignore filter drops both model.safetensors and pytorch_model.bin (the from_tf / from_flax case), so
    the TF or Flax weight counts as readable; never true for a normal load."""
    return not _filter_paths(
        ["model.safetensors", "pytorch_model.bin"], None, ignore_patterns
    )

# Checkpoint subdirs (checkpoint-500/) are never diffusers components; must not mask missing weights
_CHECKPOINT_DIR_RE = re.compile(r"^checkpoint[-_]\d+$")


def _diffusers_active_component_dirs(specs: dict) -> set:
    """Components a pipeline loads: a [library, class] spec with both entries non-null; a [null, null] disabled
    component such as safety_checker is skipped by the load, so not required."""
    active: set = set()
    for name, spec in specs.items():
        if (
            isinstance(spec, (list, tuple)) and len(spec) >= 2
            and spec[0] is not None and spec[1] is not None
        ):
            active.add(name)
    return active


def _is_weightless_component(names: list) -> bool:
    """True for a weightless component (scheduler, tokenizer, feature extractor) with a *_config.json or tokenizer
    sidecar; a dir with no config.json, weight or sidecar is an incomplete model."""
    return any(
        n.endswith("_config.json")
        or n.startswith("tokenizer")
        or n in ("preprocessor_config.json", "special_tokens_map.json", "merges.txt", "vocab.json")
        for n in names
    )


# Weightless component classes (Scheduler, Tokenizer, Processor) matched by substring, not models
_WEIGHTLESS_COMPONENT_CLASS_RE = re.compile(
    r"Scheduler|Tokenizer|FeatureExtractor|ImageProcessor|Processor"
)


def _component_requires_weight(spec: Any, names: list) -> bool:
    """True if a component with no config.json or readable weight is incomplete, not weightless; the declared
    model_index.json class decides (scheduler/tokenizer need no weight), so a lone sidecar cannot pass; name
    heuristic only if the class is unknown"""
    cls = (
        spec[1]
        if isinstance(spec, (list, tuple)) and len(spec) >= 2 and isinstance(spec[1], str)
        else None
    )
    if cls is not None:
        return not bool(_WEIGHTLESS_COMPONENT_CLASS_RE.search(cls))
    return not _is_weightless_component(names)


def _diffusers_component_weights_complete(
    snapshot_dir: Path, *, variant: Optional[str], ignore_patterns: Any = None,
) -> bool:
    """True only if each declared active component has a readable weight of the read format (sidecar-only components
    excepted); root and subtree checkpoint weights do not count; malformed model_index.json fails open; variants
    accept canonical component weights"""
    specs = _diffusers_declared_component_specs(snapshot_dir)
    declared = set(specs) if specs else None
    active = _diffusers_active_component_dirs(specs) if specs else None
    # Variant weight = canonical base + variant token, so sidecars like adapter_model.fp16 don't count
    variant_weight_re = (
        re.compile(
            rf"^(?:diffusion_pytorch_model|model|pytorch_model)\.{re.escape(variant)}"
            rf"(?:-\d{{5}}-of-\d{{5}})?\.(?:safetensors|bin)$"
        )
        if variant else None
    )
    per_comp_canon: dict = {}
    per_comp_variant: dict = {}
    try:
        for entry in snapshot_dir.rglob("*"):
            name = entry.name
            is_index = _is_canonical_component_shard_index(name)
            if not is_index and not _is_default_load_weight_file(name):
                continue
            try:
                if not entry.is_file():
                    continue
                rel = entry.relative_to(snapshot_dir).as_posix()
            except (OSError, ValueError):
                continue
            parts = rel.split("/")
            if len(parts) != 2:
                continue  # only a weight AT the component root counts (not a root file nor a nested
                # descendant such as unet/old/diffusion_pytorch_model.safetensors, which diffusers does
                # not read and which must not mask a missing top-level unet/diffusion_pytorch_model.*)
            comp = parts[0]
            if declared is not None and comp not in declared:
                continue  # an UNDECLARED subtree the load does not read
            if _CHECKPOINT_DIR_RE.match(comp):
                continue  # a training-checkpoint subtree, not a component
            if is_index:
                # Component shard index proves a readable weight even for odd shard names; probes its own base
                tok = _index_variant_token(name)
                probe = _component_index_weight_probe(name, comp)
                if probe is None:
                    continue
                if tok is None:
                    # Canonical component index: under a variant load accept only when complete; plain loads record it
                    if variant is not None and not _weight_shard_index_complete(entry):
                        continue
                    per_comp_canon.setdefault(comp, []).append(probe)
                elif tok == variant:
                    per_comp_variant.setdefault(comp, []).append(probe)
                continue
            if variant_weight_re is not None and variant_weight_re.match(name):
                per_comp_variant.setdefault(comp, []).append(rel)
            elif _CANONICAL_COMPONENT_WEIGHT_RE.match(name):
                per_comp_canon.setdefault(comp, []).append(rel)
    except OSError:
        return False

    def _has_canon(comp: str) -> bool:
        return bool(_filter_paths(per_comp_canon.get(comp, []), None, ignore_patterns))

    def _has_variant(comp: str) -> bool:
        return bool(_filter_paths(per_comp_variant.get(comp, []), None, ignore_patterns))

    def _has_read_weight(comp: str) -> bool:
        # variant load falls back to a component's canonical weight when it ships no variant file
        return _has_variant(comp) or _has_canon(comp) if variant else _has_canon(comp)

    if active is not None:
        for comp in active:
            comp_dir = snapshot_dir / comp
            try:
                names = [e.name for e in comp_dir.iterdir()] if comp_dir.is_dir() else []
            except OSError:
                names = []
            if not names:
                return False  # a declared active component was never materialised
            if "config.json" in names:  # a model-style component MUST hold a readable weight
                if not _has_read_weight(comp):
                    return False
            elif not _has_read_weight(comp) and _component_requires_weight(specs.get(comp), names):
                # No config.json and no readable weight: an incomplete model-style component whose weight
                # the load would fetch over un-killable Xet. A genuinely weightless component (scheduler /
                # tokenizer / feature_extractor) is exempt -- decided by its DECLARED class when known, so
                # a model component holding only a stray sidecar cannot pass as weightless.
                return False
    # Floor: some component must hold a READ-format weight; rejects variant-only or config-only snapshots
    if variant:
        return any(_has_variant(c) for c in (declared or per_comp_variant))
    return any(_has_canon(c) for c in (declared or per_comp_canon))


def _has_diffusers_component_weight(snapshot_dir: Path, *, ignore_patterns: Any = None) -> bool:
    """Plain (non-variant) diffusers pipeline warm completeness check; the logic lives in _diffusers_component_weights_complete."""
    return _diffusers_component_weights_complete(
        snapshot_dir, variant = None, ignore_patterns = ignore_patterns
    )


def _root_model_has_weight(snapshot_dir: Path, *, ignore_patterns: Any = None) -> bool:
    """Canonical root weight (single or numbered shard) or diffusers component weight only; a subtree checkpoint must
    not count, else root weights get fetched over un-killable Xet; ignore filter applied."""
    try:
        is_diffusers = (snapshot_dir / "model_index.json").is_file()
    except OSError:
        is_diffusers = False
    if is_diffusers:
        return _has_diffusers_component_weight(snapshot_dir, ignore_patterns = ignore_patterns)
    rels: list = []
    tf_flax_rels: list = []
    index_probes: set = set()
    try:
        for entry in snapshot_dir.iterdir():
            name = entry.name
            try:
                if not entry.is_file():
                    continue
            except OSError:
                continue
            if _CANONICAL_ROOT_MODEL_WEIGHT_RE.match(name):
                rels.append(name)  # canonical model / pytorch_model (single or shard)
            elif _is_canonical_weight_shard_index(name):
                # Sharded model enumerated via its root index is recorded: shard names may be non-standard
                index_probes.add(_index_weight_probe(name))
            elif _CANONICAL_ROOT_TF_FLAX_WEIGHT_RE.match(name):
                tf_flax_rels.append(name)  # TF/Flax root weight (from_tf / from_flax)
    except OSError:
        return False
    if _filter_paths(rels, None, ignore_patterns):
        return True
    # Root shard index of a kept format proves a readable weight even for non-standard shard names
    if any(_read_format_kept(probe, ignore_patterns) for probe in index_probes):
        return True
    # Under from_tf/from_flax (PyTorch ignored) count a single TF/Flax weight or a complete sharded set
    if _pytorch_root_weight_formats_ignored(ignore_patterns):
        if tf_flax_rels and _filter_paths(tf_flax_rels, None, ignore_patterns):
            return True
        for index_name in _TF_FLAX_WEIGHT_INDEX_NAMES:
            index_path = snapshot_dir / index_name
            try:
                if index_path.is_file() and _weight_shard_index_complete(index_path):
                    return True
            except OSError:
                continue
    return False


def _root_has_variant_weight(
    snapshot_dir: Path, variant: str, *, ignore_patterns: Any = None
) -> bool:
    """Root variant weight (model.<variant>.safetensors or .<variant>- shards) or a root variant shard index counts, so
    non-standard shard names are not false-rejected; PEFT, non-canonical bases and non-model variants do not; ignore
    filter applied."""
    infix_dot = f".{variant}."
    infix_dash = f".{variant}-"
    rels: list = []
    index_probes: set = set()
    try:
        for entry in snapshot_dir.iterdir():
            name = entry.name
            if infix_dot not in name and infix_dash not in name:
                continue  # not the requested variant token
            try:
                if not entry.is_file():
                    continue
            except OSError:
                continue
            if _ROOT_MODEL_VARIANT_WEIGHT_RE.match(name):
                rels.append(name)  # canonical model / pytorch_model variant weight (single or shard)
            elif _ROOT_MODEL_SHARD_INDEX_RE.match(name):
                # Sharded variant enumerated via its root index is recorded: shard names may be non-standard
                index_probes.add(_index_weight_probe(name, variant))
    except OSError:
        return False
    if _filter_paths(rels, None, ignore_patterns):
        return True
    # Root variant shard index proves a readable weight even for odd shard names (completeness separate).
    return any(_read_format_kept(probe, ignore_patterns) for probe in index_probes)


def _has_diffusers_component_variant_weight(
    snapshot_dir: Path, variant: str, *, ignore_patterns: Any = None
) -> bool:
    """Diffusers variant completeness check; variant weights live in component subfolders, so a root-only check would false-reject them."""
    return _diffusers_component_weights_complete(
        snapshot_dir, variant = variant, ignore_patterns = ignore_patterns
    )


def _root_model_has_variant_weight(
    snapshot_dir: Path, variant: str, *, ignore_patterns: Any = None
) -> bool:
    """Unpatterned variant check: a root variant weight, or for diffusers a component-subfolder variant weight, since diffusers variants live in subfolders."""
    try:
        is_diffusers = (snapshot_dir / "model_index.json").is_file()
    except OSError:
        is_diffusers = False
    if is_diffusers:
        return _has_diffusers_component_variant_weight(
            snapshot_dir, variant, ignore_patterns = ignore_patterns
        )
    return _root_has_variant_weight(snapshot_dir, variant, ignore_patterns = ignore_patterns)


# Either-format pairs are one group (any one satisfies it); distinct logical weights stay separate
_EITHER_FORMAT_WEIGHT_RE = re.compile(
    r"^(model|pytorch_model|adapter_model)(?:\.([^.]+))?\.(?:safetensors|bin)$"
)


def _exact_weight_logical(base: str) -> Any:
    """Group key: model.safetensors and pytorch_model.bin share ("root_model", None), or ("root_model", variant) for a
    variant; adapter_model.* keys as ("adapter_model", ...); sharded or non-weight names map to themselves and are
    each still required."""
    m = _EITHER_FORMAT_WEIGHT_RE.match(base)
    if m is None:
        return base
    stem, variant = m.group(1), m.group(2)
    logical = "adapter_model" if stem == "adapter_model" else "root_model"
    return (logical, variant)


def _requested_exact_files_present_grouped(
    snapshot_dir: Path, *, allow_patterns: Any, ignore_patterns: Any,
) -> bool:
    """A missing exact-named requested file fails; interchangeable weights need any one of the group, distinct logical
    files each. Globs and unpatterned requests always pass."""
    allow = _as_pattern_list(allow_patterns)
    ignore = _as_pattern_list(ignore_patterns)
    if not allow or any(not isinstance(p, str) or _has_glob(p) for p in allow):
        return True
    requested = _filter_paths(allow, None, ignore)
    if not requested:
        return True  # ignore filter dropped every named file -> nothing to require
    try:
        present = {
            entry.relative_to(snapshot_dir).as_posix()
            for entry in snapshot_dir.rglob("*")
            if entry.is_file()
        }
    except OSError:
        return True  # cannot enumerate -> do not reject on an unreadable dir
    groups: "dict[tuple[str, Any], list[str]]" = {}
    for rel in requested:
        parent, base = rel.rsplit("/", 1) if "/" in rel else ("", rel)
        logical = _exact_weight_logical(base)
        groups.setdefault((parent, logical), []).append(rel)
    return all(
        any(candidate in present for candidate in candidates) for candidates in groups.values()
    )


def _has_selected_weight(
    snapshot_dir: Path, *, allow_patterns: Any, ignore_patterns: Any,
) -> bool:
    """True if a loadable weight the allow/ignore filters select exists, so a stale out-of-scope weight cannot satisfy a patterned request."""
    weights: list = []
    try:
        for entry in snapshot_dir.rglob("*"):
            if not _is_loadable_weight_file(entry.name):
                continue
            try:
                if entry.is_file():
                    weights.append(entry.relative_to(snapshot_dir).as_posix())
            except (OSError, ValueError):
                continue
    except OSError:
        return False
    return bool(_filter_paths(weights, allow_patterns, ignore_patterns))


def _has_selected_variant_weight(
    snapshot_dir: Path, *, allow_patterns: Any, ignore_patterns: Any, variant: str,
) -> bool:
    """True if a selected variant-token weight exists; a canonical-only partial is retried over HTTP instead of fetching the variant over un-killable Xet."""
    infix_dot = f".{variant}."
    infix_dash = f".{variant}-"
    weights: list = []
    try:
        for entry in snapshot_dir.rglob("*"):
            name = entry.name
            if not _is_loadable_weight_file(name):
                continue
            if infix_dot not in name and infix_dash not in name:
                continue
            try:
                if entry.is_file():
                    weights.append(entry.relative_to(snapshot_dir).as_posix())
            except (OSError, ValueError):
                continue
    except OSError:
        return False
    return bool(_filter_paths(weights, allow_patterns, ignore_patterns))


def _patterns_are_exact_names(patterns: Any) -> bool:
    """True only for a non-empty allow list of EXACT filenames (no ``None`` / glob / trailing-slash dir).
    Only such a request is locally provable complete; ``None`` / a glob needs the Hub manifest."""
    patterns = _as_pattern_list(patterns)
    if patterns is None:
        return False
    if not patterns:
        return True  # selects nothing -> nothing to fetch
    return all(isinstance(p, str) and not _has_glob(p) for p in patterns)


def _request_selects_canonical_root_shards(allow_patterns: Any, ignore_patterns: Any) -> bool:
    """False means an incomplete canonical root shard set is out of scope, so the shard gate must not reject it; else a
    complete patterned download would fail with DownloadStallError."""
    probes = ["model-00001-of-00002.safetensors", "pytorch_model-00001-of-00002.bin"]
    return bool(_filter_paths(probes, allow_patterns, ignore_patterns))


def _request_selects_root_variant_weight(
    allow_patterns: Any, ignore_patterns: Any, variant: str,
) -> bool:
    """False means a stale root variant shard set is out of scope (e.g. a subfolder request whose variant weights live
    under unet/), so the root variant-shard gate must not reject a complete in-scope download."""
    probes = [
        f"model.{variant}.safetensors", f"model.{variant}-00001-of-00002.safetensors",
        f"pytorch_model.{variant}.bin", f"pytorch_model.{variant}-00001-of-00002.bin",
    ]
    return bool(_filter_paths(probes, allow_patterns, ignore_patterns))


def _cache_can_skip_download(
    snapshot_dir: Path, *, repo_type: str, allow_patterns: Any, ignore_patterns: Any,
    variant: Optional[str] = None,
) -> bool:
    """Strict pre-download skip: a false True lets a missing weight be fetched over un-killable Xet; only
    snapshot_dir_is_complete skips weight-bearing requests, variants/diffusers/patterns/sharded-without-index spawn
    the child; weightless requests need an intact exact-named subset."""
    if repo_type in (None, "model") and request_can_include_weights(allow_patterns, ignore_patterns):
        # Variant load reads variant-named weights the canonical gate skips, so defer to the child
        if variant:
            return False
        # Sentence-transformers subfolder weight (e.g. 2_Dense) is unchecked by the root gate; defer to child
        if _sentence_transformers_subfolder_incomplete(
            snapshot_dir, prefer_safetensors = True, ignore_patterns = ignore_patterns, strict = True
        ):
            return False
        # Bin-only cache defers under prefer_safetensors, since safetensors can't be proven locally
        return snapshot_dir_is_complete(
            snapshot_dir, allow_patterns = allow_patterns, ignore_patterns = ignore_patterns,
            prefer_safetensors = True,
        )
    # Weightless / non-model: skip only for an intact exact-named subset; None / glob defers to the child.
    if not _patterns_are_exact_names(allow_patterns):
        return False
    return _intact_subset(
        snapshot_dir, repo_type = repo_type, allow_patterns = allow_patterns,
        ignore_patterns = ignore_patterns,
    )


def _has_readable_weight(
    snapshot_dir: Path, *, allow_patterns: Any, ignore_patterns: Any, variant: Optional[str],
) -> bool:
    """Invariant A: the weight the load reads must exist (ignore filter applied): unpatterned variant needs a root
    variant weight, patterned a selected variant weight; unpatterned plain a root or diffusers component weight (not
    a stray subfolder checkpoint), patterned a selected weight."""
    if variant:
        if allow_patterns is None:
            return _root_model_has_variant_weight(
                snapshot_dir, variant, ignore_patterns = ignore_patterns
            )
        return _has_selected_variant_weight(
            snapshot_dir, allow_patterns = allow_patterns,
            ignore_patterns = ignore_patterns, variant = variant,
        )
    if allow_patterns is None:
        return _root_model_has_weight(snapshot_dir, ignore_patterns = ignore_patterns)
    return _has_selected_weight(
        snapshot_dir, allow_patterns = allow_patterns, ignore_patterns = ignore_patterns
    )


def _readable_shard_set_incomplete(
    snapshot_dir: Path, *, allow_patterns: Any, ignore_patterns: Any, variant: Optional[str],
) -> bool:
    """Invariant B: the in-scope shard set the load reads must be complete (a missing index shard, or a numbered shard
    without its index, is retried); checks are scoped to the request selection; diffusers skips root shards and
    checks component shards."""
    try:
        is_diffusers = (snapshot_dir / "model_index.json").is_file()
    except OSError:
        is_diffusers = False
    if _patterns_are_exact_names(allow_patterns):
        # Exact-named requests defer to exact-file check; a lone shard is not judged by its unrequested index
        return False
    if variant:
        if not is_diffusers and (
            allow_patterns is None
            or _request_selects_root_variant_weight(allow_patterns, ignore_patterns, variant)
        ):
            # Root-shard check only for non-diffusers variant loads; diffusers variant weights are per component
            if _has_incomplete_variant_root_shards(
                snapshot_dir, variant, ignore_patterns = ignore_patterns
            ):
                return True
        if allow_patterns is not None:
            if _selected_shard_index_incomplete(
                snapshot_dir, allow_patterns = allow_patterns,
                ignore_patterns = ignore_patterns, variant = variant,
            ):
                return True
        elif is_diffusers and _diffusers_component_shards_incomplete(
            snapshot_dir, variant = variant, ignore_patterns = ignore_patterns
        ):
            # Unpatterned variant: a component variant shard index must be complete; root check covers root only
            return True
        return False
    if allow_patterns is None:
        if not is_diffusers and _has_incomplete_canonical_root_shards(
            snapshot_dir, ignore_patterns = ignore_patterns
        ):
            # Non-diffusers root model loads only: a diffusers stale root index is left to the component check
            return True
        if is_diffusers and _diffusers_component_shards_incomplete(
            snapshot_dir, variant = None, ignore_patterns = ignore_patterns
        ):
            # Unpatterned plain diffusers: incomplete component shard index is not covered by root-shard check
            return True
        return False
    if not is_diffusers and _request_selects_canonical_root_shards(
        allow_patterns, ignore_patterns
    ) and _has_incomplete_canonical_root_shards(snapshot_dir, ignore_patterns = ignore_patterns):
        # Non-diffusers only: pipelines never read root shards, so a stale root index must not reject it
        return True
    return _selected_shard_index_incomplete(
        snapshot_dir, allow_patterns = allow_patterns,
        ignore_patterns = ignore_patterns, variant = None,
    )


def _selected_readable_weight_complete(
    snapshot_dir: Path, *, allow_patterns: Any, ignore_patterns: Any, variant: Optional[str],
) -> bool:
    """Model acceptance: the readable weight must be present (Invariant A) and its in-scope shard set complete
    (Invariant B); both use the same filter and scope, so stray partials neither mask a hang nor cause a false
    DownloadStallError."""
    if not _has_readable_weight(
        snapshot_dir, allow_patterns = allow_patterns,
        ignore_patterns = ignore_patterns, variant = variant,
    ):
        return False
    if _readable_shard_set_incomplete(
        snapshot_dir, allow_patterns = allow_patterns,
        ignore_patterns = ignore_patterns, variant = variant,
    ):
        return False
    return True


def _download_result_usable(
    snapshot_dir: Path, *, repo_type: str, allow_patterns: Any, ignore_patterns: Any,
    variant: Optional[str] = None,
) -> bool:
    """Post-download check, lenient by design: snapshot_download already compares the manifest, so only positive
    breakage fails (dangling requested symlink, missing exact-named file, incomplete readable weight); a transient
    metadata error can yield a stale cache."""
    if snapshot_has_requested_broken_symlinks(
        snapshot_dir, allow_patterns = allow_patterns, ignore_patterns = ignore_patterns,
        repo_type = repo_type,
    ):
        return False
    if not _requested_exact_files_present_grouped(
        snapshot_dir, allow_patterns = allow_patterns, ignore_patterns = ignore_patterns
    ):
        return False
    if repo_type in (None, "model") and request_can_include_weights(allow_patterns, ignore_patterns):
        if not _selected_readable_weight_complete(
            snapshot_dir, allow_patterns = allow_patterns,
            ignore_patterns = ignore_patterns, variant = variant,
        ):
            return False
        # Missing known weighted subfolder module (2_Dense) rejects, retrying over HTTP; unknown modules pass
        if _sentence_transformers_subfolder_incomplete(
            snapshot_dir, prefer_safetensors = False, ignore_patterns = ignore_patterns, strict = False
        ):
            return False
    return True


def _snapshot_payload_incomplete(
    payload: Any, *, repo_type: str, allow_patterns: Any, ignore_patterns: Any,
    variant: Optional[str] = None,
) -> bool:
    """True only for a real existing snapshot dir that fails _download_result_usable; non-path payloads (test mocks) are trusted."""
    try:
        path = Path(payload)
    except (TypeError, ValueError, OSError):
        return False  # non-path payload (test sentinel) or invalid path -> trust it
    try:
        if not path.is_dir():
            return False
    except OSError:
        return False
    return not _download_result_usable(
        path, repo_type = repo_type, allow_patterns = allow_patterns,
        ignore_patterns = ignore_patterns, variant = variant,
    )


def _wait_before_http_retry(cancel_event: Optional[threading.Event]) -> None:
    """Backs off between HTTP children so a shedding CDN is not hit again at once; a cancel during the wait raises immediately instead of starting another child."""
    delay = _http_retry_backoff()
    if cancel_event is None:
        time.sleep(delay)
        return
    if cancel_event.wait(delay):
        raise RuntimeError("Cancelled")


def _download_with_xet_fallback(
    *,
    repo_id: str,
    label: str,
    kind: str,
    params: dict,
    token: Optional[str],
    repo_type: str,
    cancel_event: Optional[threading.Event],
    stall_timeout: Optional[float],
    interval: Optional[float],
    grace_period: float,
    on_status: Optional[Callable[[str], None]],
    prepare_for_http_fn: Optional[Callable[[str, str], None]],
    variant: Optional[str] = None,
) -> str:
    """Xet first, HTTP last: Xet gets xet_attempts() children, spent only on DATA-phase stalls, other Xet failures flip
    to HTTP; HTTP gets http_attempts() children on transient faults or crashes since a degraded CDN fails both rungs;
    a stall on HTTP raises at once."""
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Cancelled")

    cache_dir = params.get("cache_dir")
    # xet_force_disabled() is read under _SPAWN_ENV_LOCK so peer spawn-time HF_HUB_DISABLE_XET is ignored
    with _SPAWN_ENV_LOCK:
        disable_xet = xet_force_disabled()
    # A user disable is an env var the fallback inherits; a health demotion is internal to this module
    xet_disabled_by_user = disable_xet

    # Skip a known-doomed Xet attempt (low RAM, unreachable CAS, recent failures) to avoid a stall
    if not disable_xet:
        health = _xet_health_or_none()
        if health is not None and not health.use_xet:
            logger.info("Starting '%s' on HTTP instead of Xet: %s", label, health.reason)
            _safe_status(on_status, f"{label}: using HTTP ({health.reason})")
            disable_xet = True
    started_on_xet = not disable_xet
    # Xet children this download may spend before HTTP, and how many it has spent.
    xet_budget = xet_attempts() if started_on_xet else 0
    xet_used = 0
    # HTTP children this download may spend, and how many it has spent.
    http_budget = http_attempts()
    http_used = 0
    # The caller's own force_download is restored only once the forced partial is provably gone
    caller_force_download = params.get("force_download", False)
    forced_clean_redownload = False
    # Purge once at the Xet->HTTP transition: repeats would spare the fresh partial and force restarts
    http_prepared = False
    # Report health once per download: charging both attempts would demote after a single bad download
    pending_xet_failure: Optional[str] = None
    # A stall is charged on any exit; a fault or crash needs HTTP success to prove it was Xet
    pending_needs_http_success = False

    def _flush_pending_failure() -> None:
        """Report a deferred Xet failure on the way out of the ladder (exactly once)."""
        nonlocal pending_xet_failure
        if pending_xet_failure is not None:
            _record_xet_outcome(False, pending_xet_failure)
            pending_xet_failure = None

    def _hold_unproven_xet_failure(reason: str) -> None:
        """A held stall outranks a later fault: a stall is proven on this machine, a fault only once HTTP shows the
        network was fine; overwriting it would set pending_needs_http_success and drop the stall."""
        nonlocal pending_xet_failure, pending_needs_http_success
        if pending_xet_failure is None:
            pending_xet_failure = _xet_failure_reason(reason)
            pending_needs_http_success = True

    while True:
        if disable_xet:
            http_used += 1
        else:
            xet_used += 1

        if disable_xet and not http_prepared:
            http_prepared = True
            # Purge non-HTTP partials first (own only): HTTP resume corrupts a sparse Xet/hf_transfer partial
            owned_incomplete = params.pop("_owned_incomplete_blobs", None)
            try:
                if prepare_for_http_fn is None:
                    # Grace protects a live sibling's partial, so it stays longer than the 30s Xet stall trip
                    _default_prepare_for_http(
                        repo_type, repo_id, cache_dir = cache_dir,
                        active_grace = max(stall_timeout or 0.0, DEFAULT_HTTP_STALL_TIMEOUT),
                        owned_incomplete_blobs = owned_incomplete,
                    )
                else:
                    prepare_for_http_fn(repo_type, repo_id)
            except Exception as e:
                logger.debug("prepare_for_http failed for %s: %s", repo_id, e)
            # An unsafe partial that could not be cleared (locked / permission) would corrupt the blob
            # on an HTTP resume, and the force that avoids that is REPO-WIDE, so spend one more
            # BLOB-SCOPED attempt first: a spared partial cost every completed shard (#9094).
            #
            # The GATE is the tri-state, not `has_active_incomplete_blobs`, which walks
            # `iter_active_repo_cache_dirs` and swallows OSError: a cache that has gone briefly
            # unreadable answers False there, indistinguishable from a clean one, and False here
            # skips the clearance AND the confirmation below, so an unforced HTTP child resumes
            # whatever sparse partial is still on disk and finalizes it under its sha256 name.
            # Only a POSITIVE confirmation that the cache is clean may skip this block.
            if not _no_incomplete_blobs_confirmed(repo_type, repo_id, cache_dir):
                survivors = _clear_unsafe_partials_for_http(
                    repo_type, repo_id, cache_dir = cache_dir,
                    active_grace = max(stall_timeout or 0.0, DEFAULT_HTTP_STALL_TIMEOUT),
                    owned_incomplete_blobs = owned_incomplete,
                )
                # Only absence counts as clear: an unreadable cache (None) is not clear; dangling links read active
                cleared = (
                    survivors is not None
                    and not survivors
                    and _no_incomplete_blobs_confirmed(repo_type, repo_id, cache_dir)
                )
                if cleared:
                    logger.info(
                        "Unsafe partial for '%s' cleared; re-fetching only the affected file(s) "
                        "and keeping every completed one.", label
                    )
                else:
                    logger.warning(
                        "Unsafe partial for '%s' could not be cleared; forcing a clean "
                        "HTTP re-download instead of an unsafe resume.", label
                    )
                    params = {**params, "force_download": True}
                    forced_clean_redownload = True
        elif disable_xet:
            # Re-check before each HTTP child; the force is released only when no *.incomplete file remains
            present = _incomplete_partial_names(repo_type, repo_id, cache_dir)
            if forced_clean_redownload:
                if present is not None and not present:
                    forced_clean_redownload = False
                    params = {**params, "force_download": caller_force_download}
                else:
                    logger.debug(
                        "Keeping force_download for '%s': %s",
                        label,
                        "cache not inspectable" if present is None
                        else f"{len(present)} partial(s) still present",
                    )
            elif present:
                # An uninspectable cache leaves the guard as it was; no forced re-download without positive proof
                logger.warning(
                    "Partial for '%s' appeared after the guard was released; forcing a clean "
                    "HTTP re-download instead of an unsafe resume.", label
                )
                forced_clean_redownload = True
                params = {**params, "force_download": True}

        kind_result, payload = _run_download_attempt(
            repo_id,
            kind = kind,
            params = params,
            token = token,
            repo_type = repo_type,
            disable_xet = disable_xet,
            cancel_event = cancel_event,
            stall_timeout = stall_timeout,
            interval = interval,
            grace_period = grace_period,
            on_status = on_status,
        )

        # An incomplete snapshot rides on an ok result but is a FAILURE; classify it before the cancel check, or a
        # cancel would still run the destructive purge and spend another child.
        incomplete_snapshot = kind_result == "ok" and kind == "snapshot" and (
            _snapshot_payload_incomplete(
                payload,
                repo_type = repo_type,
                allow_patterns = params.get("allow_patterns"),
                ignore_patterns = params.get("ignore_patterns"),
                variant = variant,
            )
        )

        # Cancellation wins over every FAILURE verdict, uniformly. Per-branch precedence let a cancel
        # landing on a stall, a deterministic error or an incomplete snapshot report the download's
        # own error, and on a Xet verdict run the destructive purge and spend another child. A
        # genuine success is exempt: the bytes are on disk, so returning them beats discarding them.
        if (
            (kind_result != "ok" or incomplete_snapshot)
            and cancel_event is not None
            and cancel_event.is_set()
        ):
            # The user's decision says nothing about this machine's Xet health.
            pending_xet_failure = None
            raise RuntimeError("Cancelled")

        if kind_result == "ok":
            if incomplete_snapshot:
                # Never load an incomplete snapshot dir in-process (HF may return one offline); retry over HTTP
                if not disable_xet:
                    logger.warning(
                        "Download for '%s' returned an incomplete snapshot -- "
                        "retrying with HF_HUB_DISABLE_XET=1", label
                    )
                    _safe_status(on_status, f"{label}: incomplete snapshot, retrying over HTTP")
                    # Hold a short snapshot as unproven: a transport symptom, not evidence against this machine's Xet
                    _hold_unproven_xet_failure("Xet returned an incomplete snapshot")
                    disable_xet = True
                    continue
                # If both rungs fall short, an unproven reason is dropped; a watchdog-proven stall is still charged
                if pending_needs_http_success:
                    pending_xet_failure = None
                _flush_pending_failure()
                raise DownloadStallError(
                    f"Download for '{label}' returned an incomplete snapshot even with "
                    f"HF_HUB_DISABLE_XET=1 -- missing files, check your network connection"
                )
            if started_on_xet and not disable_xet:
                # Completed on its starting transport (Xet works here), so an earlier recovered stall is dropped.
                pending_xet_failure = None
                _record_xet_outcome(True)
            elif started_on_xet:
                # HTTP finished what Xet could not, so the held failure is now evidence. Charge it.
                _flush_pending_failure()
            return payload  # type: ignore[return-value]
        if kind_result == "cancelled":
            # The user's decision says nothing about this machine's Xet health.
            pending_xet_failure = None
            raise RuntimeError("Cancelled")
        if kind_result == "error":
            # Deterministic error fails on both transports; an earlier stall is still reported before the raise
            if pending_needs_http_success:
                pending_xet_failure = None
            _flush_pending_failure()
            # Keep unrecognized errors inside the guard (issue 1122); scope by why Xet is off, not if used
            if disable_xet and not xet_disabled_by_user:
                type_name = payload.split(":", 1)[0].strip() if ":" in (payload or "") else ""
                if _resolve_exception_class(type_name) is None:
                    raise DownloadTransportError(payload)
            _raise_child_error(payload)
        if kind_result == "retryable_error":
            # Transient transport fault: switch to HTTP, then surface as stall-class error, not bare RuntimeError
            if not disable_xet:
                logger.warning(
                    "Download for '%s' hit a transient Xet transport error -- retrying "
                    "with HF_HUB_DISABLE_XET=1: %s", label, payload
                )
                _safe_status(on_status, f"{label}: transient Xet error, retrying over HTTP")
                # Held, not recorded: a transient fault counts against this machine's Xet only if HTTP then succeeds
                _hold_unproven_xet_failure("transient Xet transport error")
                disable_xet = True
                continue
            if http_used < http_budget:
                logger.warning(
                    "Download for '%s' hit a transient error on HTTP -- retrying "
                    "(attempt %d of %d): %s", label, http_used + 1, http_budget, payload
                )
                _safe_status(
                    on_status,
                    f"{label}: transient HTTP error, retrying "
                    f"(attempt {http_used + 1} of {http_budget})",
                )
                # Same fault on HTTP is a shared incident, not Xet-specific: drop held reason; proven stalls stay
                if pending_needs_http_success:
                    pending_xet_failure = None
                    pending_needs_http_success = False
                _wait_before_http_retry(cancel_event)
                continue
            # Same fault on both rungs is no evidence against Xet: drop unproven reason; a proven stall stays
            if pending_needs_http_success:
                pending_xet_failure = None
            _flush_pending_failure()
            raise DownloadTransportError(payload)
        if kind_result == "crashed":
            # Process-level crash with no captured exception: HTTP may still succeed, so retry once.
            if not disable_xet:
                logger.warning(
                    "Download process for '%s' crashed without a result -- "
                    "retrying with HF_HUB_DISABLE_XET=1", label
                )
                _safe_status(on_status, f"{label}: download crashed, retrying over HTTP")
                # Held for the same reason as a transient error: only an HTTP rescue proves it was Xet.
                _hold_unproven_xet_failure("Xet download process crashed")
                disable_xet = True
                continue
            if http_used < http_budget:
                logger.warning(
                    "Download process for '%s' crashed on HTTP -- retrying (attempt %d of %d)",
                    label, http_used + 1, http_budget,
                )
                _safe_status(
                    on_status,
                    f"{label}: download crashed, retrying "
                    f"(attempt {http_used + 1} of {http_budget})",
                )
                # Crash on HTTP as well is not evidence against Xet; drop the held reason here, proven stalls stay
                if pending_needs_http_success:
                    pending_xet_failure = None
                    pending_needs_http_success = False
                _wait_before_http_retry(cancel_event)
                continue
            if pending_needs_http_success:
                pending_xet_failure = None
            _flush_pending_failure()
            raise DownloadTransportError(payload)
        # kind_result == "stall"
        if not disable_xet:
            # Another Xet child only for a DATA-phase stall while budget lasts; a cancel landing in the stall window
            # must not spawn one; pre-first-byte trips cost a full connect_timeout
            if (
                xet_used < xet_budget
                and is_data_phase_stall(payload or "")
                and not (cancel_event is not None and cancel_event.is_set())
            ):
                logger.warning(
                    "Download stalled for '%s' -- retrying on Xet (attempt %d of %d)",
                    label, xet_used + 1, xet_budget,
                )
                _safe_status(
                    on_status,
                    f"{label}: Xet stalled, retrying Xet "
                    f"(attempt {xet_used + 1} of {xet_budget})",
                )
                # Held, not recorded: if the next attempt succeeds this stall was noise.
                pending_xet_failure = _xet_failure_reason("Xet download stalled")
                pending_needs_http_success = False
                _purge_owned_partials_for_xet_retry(
                    repo_type, repo_id, cache_dir = cache_dir,
                    owned_incomplete_blobs = params.pop("_owned_incomplete_blobs", None),
                )
                continue
            logger.warning(
                "Download stalled for '%s' -- retrying with HF_HUB_DISABLE_XET=1", label
            )
            # _safe_status: a raising status hook must not abort the retry before disable_xet is set.
            _safe_status(on_status, f"{label}: Xet stalled, retrying over HTTP")
            # Xet phase over: report the earlier deferred stall if there was one, else this one.
            pending_xet_failure = pending_xet_failure or _xet_failure_reason("Xet download stalled")
            _flush_pending_failure()
            disable_xet = True
            continue
        # An HTTP stall says nothing about Xet; a held Xet stall still counts at the final exit
        if pending_needs_http_success:
            pending_xet_failure = None
        _flush_pending_failure()
        raise DownloadStallError(
            f"Download stalled for '{label}' even with HF_HUB_DISABLE_XET=1 "
            f"-- check your network connection"
        )


def hf_hub_download_with_xet_fallback(
    repo_id: str,
    filename: str,
    token: Optional[str] = None,
    *,
    cancel_event: Optional[threading.Event] = None,
    repo_type: Optional[str] = "model",
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    subfolder: Optional[str] = None,
    force_download: bool = False,
    local_files_only: bool = False,
    stall_timeout: Optional[float] = None,
    interval: Optional[float] = None,
    grace_period: float = DEFAULT_GRACE_PERIOD,
    on_status: Optional[Callable[[str], None]] = None,
    prepare_for_http_fn: Optional[Callable[[str, str], None]] = None,
) -> str:
    """Xet then HTTP; RuntimeError on cancel, deterministic child errors re-raised; DownloadStallError if both stall,
    DownloadTransportError (also a DownloadStallError) on transient faults; neither raised while a rung is untried;
    local_files_only=True reads cache in-process."""
    repo_type = repo_type or "model"  # HF treats None as the default model repo.
    # Expand ~ as huggingface_hub does, so the probe and the child resolve to the same location.
    if isinstance(cache_dir, (str, os.PathLike)):
        cache_dir = os.path.expanduser(os.fspath(cache_dir))
    # Honor cancellation before any probe (the short-circuits below bypass the fallback's cancel check).
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Cancelled")
    # Offline: resolve from cache. HF raises LocalEntryNotFoundError if uncached; let it propagate.
    if local_files_only:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id = repo_id,
            filename = filename,
            subfolder = subfolder,
            token = token,
            repo_type = repo_type,
            revision = revision,
            cache_dir = cache_dir,
            local_files_only = True,
        )
    # Cached finalized blob returns with no child unless force_download; subfolder files keyed by path
    if not force_download:
        try:
            from huggingface_hub import try_to_load_from_cache

            probe_filename = f"{subfolder}/{filename}" if subfolder else filename
            cached = try_to_load_from_cache(
                repo_id, probe_filename, repo_type = repo_type, revision = revision, cache_dir = cache_dir
            )
            if isinstance(cached, str) and os.path.exists(cached):
                return cached
        except Exception as e:
            logger.debug("Cached probe failed for %s/%s: %s", repo_id, filename, e)

    return _download_with_xet_fallback(
        repo_id = repo_id,
        label = f"{repo_id}/{filename}",
        kind = "file",
        params = {
            "repo_id": repo_id,
            "filename": filename,
            "subfolder": subfolder,
            "revision": revision,
            "cache_dir": cache_dir,
            "force_download": force_download,
        },
        token = token,
        repo_type = repo_type,
        cancel_event = cancel_event,
        stall_timeout = stall_timeout,
        interval = interval,
        grace_period = grace_period,
        on_status = on_status,
        prepare_for_http_fn = prepare_for_http_fn,
    )


def snapshot_download_with_xet_fallback(
    repo_id: str,
    *,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    repo_type: Optional[str] = "model",
    cache_dir: Optional[str] = None,
    allow_patterns: Optional[Any] = None,
    ignore_patterns: Optional[Any] = None,
    force_download: bool = False,
    local_files_only: bool = False,
    variant: Optional[str] = None,
    cancel_event: Optional[threading.Event] = None,
    stall_timeout: Optional[float] = None,
    interval: Optional[float] = None,
    grace_period: float = DEFAULT_GRACE_PERIOD,
    on_status: Optional[Callable[[str], None]] = None,
    prepare_for_http_fn: Optional[Callable[[str, str], None]] = None,
) -> str:
    """Snapshot Xet-then-HTTP warm-up in a killable child before from_pretrained's in-process load; transient faults
    raise DownloadTransportError, not a bare RuntimeError a caller could mistake for a skippable pre-download
    failure; variant forces the child."""
    repo_type = repo_type or "model"  # HF treats None as the default model repo.
    # Expand ~ as huggingface_hub does, so the probe and the child resolve to the same location.
    if isinstance(cache_dir, (str, os.PathLike)):
        cache_dir = os.path.expanduser(os.fspath(cache_dir))
    # Honor cancellation before any probe (the short-circuits below bypass the fallback's cancel check).
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Cancelled")
    # Offline: resolve from cache. HF raises if uncached; let it propagate.
    if local_files_only:
        from huggingface_hub import snapshot_download

        return snapshot_download(
            repo_id = repo_id,
            repo_type = repo_type,
            revision = revision,
            cache_dir = cache_dir,
            allow_patterns = allow_patterns,
            ignore_patterns = ignore_patterns,
            local_files_only = True,
        )
    # Fast path: everything on disk -> resolve in-process (no Xet, no hang). Skipped under force_download.
    if not force_download:
        try:
            from huggingface_hub import snapshot_download

            cached_dir = snapshot_download(
                repo_id = repo_id,
                repo_type = repo_type,
                revision = revision,
                cache_dir = cache_dir,
                allow_patterns = allow_patterns,
                ignore_patterns = ignore_patterns,
                local_files_only = True,
            )
            # local_files_only can return an interrupted snapshot dir; validate the exact returned revision
            if _cache_can_skip_download(
                Path(cached_dir),
                repo_type = repo_type,
                allow_patterns = allow_patterns,
                ignore_patterns = ignore_patterns,
                variant = variant,
            ):
                return cached_dir
            logger.debug("Cached snapshot for %s is incomplete; downloading.", repo_id)
        except Exception as e:
            logger.debug("Snapshot not fully cached for %s (%s); downloading.", repo_id, e)

    return _download_with_xet_fallback(
        repo_id = repo_id,
        label = repo_id,
        kind = "snapshot",
        params = {
            "repo_id": repo_id,
            "revision": revision,
            "cache_dir": cache_dir,
            "allow_patterns": allow_patterns,
            "ignore_patterns": ignore_patterns,
            "force_download": force_download,
        },
        token = token,
        repo_type = repo_type,
        cancel_event = cancel_event,
        stall_timeout = stall_timeout,
        interval = interval,
        grace_period = grace_period,
        on_status = on_status,
        prepare_for_http_fn = prepare_for_http_fn,
        variant = variant,
    )
