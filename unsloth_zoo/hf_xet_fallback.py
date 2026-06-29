# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# This file is licensed under the GNU Affero General Public License v3.0 only
# (AGPL-3.0-only), unlike the rest of unsloth_zoo which is LGPL-3.0-or-later. It
# is the single shared home for the Xet -> HTTP stall fallback used by both
# Unsloth (FastModel.from_pretrained) and Unsloth Studio, which imports it.
# See <https://www.gnu.org/licenses/agpl-3.0.html>.

"""Xet-primary HF downloads with an automatic HTTP fallback on a no-progress stall.

Xet (``hf_xet``) is the fast default but can hang with no progress and no
exception, and a blocked native thread cannot be killed. Keep Xet primary; fall
back to plain HTTP only when the parent observes a stall. ``HF_HUB_DISABLE_XET``
is read at import time, so the fallback runs in a fresh ``spawn`` child (not a
thread) that sets the env before importing ``huggingface_hub``. Cached files
short-circuit with no child; deterministic errors (401/403/404/disk-full) and
cancellation propagate without a fallback.

``hf_hub_download_with_xet_fallback`` downloads a single file; the new
``snapshot_download_with_xet_fallback`` does a whole repo (the entrypoint
Unsloth's ``from_pretrained`` uses to warm the cache in a killable child before
the in-process load). Studio-specific cache/secret/process helpers are used
best-effort (imported only if present) or injected, so the same code runs both
inside Unsloth Studio and in Unsloth itself.

Like the rest of ``unsloth_zoo``, this module is imported with ``unsloth``
installed; the package ``__init__`` runs its device init on first import. The
download spawn child does not need that and sets ``UNSLOTH_ZOO_DISABLE_GPU_INIT=1``
before it imports the package, which selects ``unsloth_zoo``'s lightweight import
path (no torch/transformers), keeping each child fast.
"""

from __future__ import annotations

import errno
import importlib.util
import multiprocessing as mp
import logging
import os
import queue
import re
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from unsloth_zoo.hf_cache_state import (
    INCOMPLETE_SUFFIX,
    _is_loadable_weight_file,
    blob_bytes_present,
    has_active_incomplete_blobs,
    _has_incomplete_canonical_root_shards,
    hf_cache_root,
    iter_active_repo_cache_dirs,
    request_can_include_weights,
    requested_named_files_present,
    snapshot_dir_is_complete,
    snapshot_has_requested_broken_symlinks,
)

logger = logging.getLogger(__name__)

# Public surface (Studio imports from this module, including a `import *` re-export shim), so
# an explicit list keeps the stdlib imports (os, re, signal, errno, ...) out of `import *`.
__all__ = [
    "DownloadStallError",
    "hf_hub_download_with_xet_fallback",
    "snapshot_download_with_xet_fallback",
    "start_watchdog",
    "get_hf_download_state",
    "is_hf_xet_available",
    "xet_force_disabled",
    "child_should_disable_xet",
]

_CTX = mp.get_context("spawn")

# Defaults match the existing Studio inference watchdog and hub shutdown deadline.
DEFAULT_HEARTBEAT_INTERVAL = 30.0
DEFAULT_STALL_TIMEOUT = 180.0
DEFAULT_GRACE_PERIOD = 10.0
_POLL_INTERVAL = 0.5

# Serializes the brief parent-env (and __main__.__file__) mutation around a child
# spawn (below) so concurrent downloads cannot observe each other's transport env.
_SPAWN_ENV_LOCK = threading.Lock()

# Sentinel: "__main__.__file__ was not touched for this spawn" (distinct from a
# real saved value of None, which means the attribute was absent).
_UNSET = object()

# Hugging Face boolean env convention: 1 / ON / YES / TRUE, case-insensitive.
_TRUTHY = {"1", "true", "yes", "on"}


def _is_true(value: Optional[str]) -> bool:
    return value is not None and str(value).strip().lower() in _TRUTHY


def _safe_status(callback: Optional[Callable[[str], None]], message: str) -> None:
    """Invoke a caller status/heartbeat callback without letting it kill the
    daemon watchdog thread. A disconnected Studio client can make on_status raise;
    if that propagated, stall detection for a genuinely hung child would stop and
    the HTTP retry would never fire."""
    if callback is None:
        return
    try:
        callback(message)
    except Exception as e:
        logger.debug("watchdog status callback raised (ignored): %s", e)


class DownloadStallError(RuntimeError):
    """Raised when no download progress is observed for too long.

    Canonical home; Studio's orchestrator re-imports it so all paths share one type.
    """


def is_hf_xet_available() -> bool:
    """True iff the ``hf_xet`` extra is importable (Hub uses it automatically)."""
    try:
        return importlib.util.find_spec("hf_xet") is not None
    except Exception:
        return False


def xet_force_disabled() -> bool:
    """Whether the user has asked us to skip Xet up front (force HTTP).

    Honors the Unsloth knobs ``UNSLOTH_DISABLE_XET`` / ``UNSLOTH_STABLE_DOWNLOADS``
    and Hugging Face's own ``HF_HUB_DISABLE_XET``.
    """
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
    # HF callers commonly pass token=True ("use the cached token"); only a real
    # string token can be substring-redacted (str.replace(True, ...) raises).
    if isinstance(hf_token, str) and hf_token:
        out = out.replace(hf_token, "***")
    out = re.sub(r"hf_[A-Za-z0-9]{8,}", "***", out)
    out = re.sub(r"([Bb]earer\s+)[A-Za-z0-9._\-]+", r"\1***", out)
    # HF download errors can embed the presigned S3/CAS blob URL, whose query
    # string carries temporary credentials (X-Amz-Signature, sig, token, ...).
    # Redact the query of any URL that looks signed so it is not echoed back to
    # the parent and logged. Non-signed URLs (e.g. ...?download=true) are kept.
    def _redact_signed_query(match: "re.Match") -> str:
        base, query = match.group(1), match.group(2)
        if re.search(
            r"(X-Amz-|[Ss]ignature|(?:^|&)(?:sig|token|key|Expires|Policy|Key-Pair-Id)=)",
            query,
        ):
            return f"{base}?***"
        return match.group(0)

    out = re.sub(r"(https?://[^\s?]+)\?([^\s]*)", _redact_signed_query, out)
    return out


def _broken_link_has_active_partner(link: Path, *, active_grace: float) -> bool:
    """True if a dangling snapshot symlink should be SPARED from the HTTP-prep purge because a
    concurrent sibling download (a different process pulling the same repo into the same cache, common
    in multi-rank training) is still writing the blob it points at.

    The reliable discriminator is a FRESH ``.incomplete`` partner of the link's target blob (mirroring
    the active-grace guard the ``.incomplete`` blob purge already uses), NOT the link's own mtime: our
    OWN killed child's link is freshly created too, but by this point its ``.incomplete`` has been
    static for the full stall timeout and is purged first, so the target has no partner and the link is
    correctly cleared -- while a sibling mid-download still has a growing ``.incomplete`` partner, so
    its link is spared."""
    try:
        target = Path(os.readlink(link))
        if not target.is_absolute():
            target = link.parent / target
        incomplete_partner = target.with_name(target.name + INCOMPLETE_SUFFIX)
        if incomplete_partner.is_file():
            return time.time() - incomplete_partner.stat().st_mtime < active_grace
    except OSError:
        return False
    return False


def _default_prepare_for_http(
    repo_type: str,
    repo_id: str,
    *,
    cache_dir: Optional[str] = None,
    active_grace: float = DEFAULT_STALL_TIMEOUT,
) -> None:
    """Generic 'make the partial safe for an HTTP resume': delete the repo's active
    ``*.incomplete`` blobs (an HTTP resume over a sparse Xet/hf_transfer partial
    silently corrupts the blob) and any broken snapshot symlinks the incomplete
    detector counts as active (else the HTTP retry inherits stale 'incomplete'
    state and trips the watchdog again). Studio injects its marker-aware version
    instead.

    ``iter_active_repo_cache_dirs`` is case-collision safe, so this destructive
    purge only touches an exact-case (or single unambiguous) repo cache dir.
    """
    try:
        for entry in iter_active_repo_cache_dirs(repo_type, repo_id, cache_dir = cache_dir):
            blobs_dir = entry / "blobs"
            if blobs_dir.is_dir():
                for blob in blobs_dir.iterdir():
                    if blob.is_file() and blob.name.endswith(INCOMPLETE_SUFFIX):
                        try:
                            # Do not unlink a partial another concurrent download is
                            # still actively writing: on POSIX that lets the sibling keep
                            # writing to an unlinked path and then fail when the Hub moves
                            # its temp file into place. Spare any partial written within
                            # active_grace (the stall timeout in use): the watchdog only
                            # declares a download stalled after that long with no growth,
                            # so a slower sibling that simply has not written recently is
                            # not stalled and must be left alone. Our own killed partial
                            # has been static for the full stall timeout, so it is purged.
                            if time.time() - blob.stat().st_mtime < active_grace:
                                continue
                            blob.unlink()
                        except OSError:
                            # A locked / permission-denied blob (common on Windows)
                            # must not abort cleanup of the rest of the partials.
                            continue
            # repo_cache_dir_has_incomplete_blobs() also flags a broken snapshot
            # symlink as active incomplete state; clear those too so the detector
            # reads clean after prep. Sweep EVERY snapshot, not just the newest:
            # the broken-symlink detector now inspects all of them, so a stale
            # dangling link under an older revision would otherwise keep the repo
            # marked incomplete after prep and re-trip the watchdog.
            snapshots_dir = entry / "snapshots"
            try:
                snapshot_dirs = [s for s in snapshots_dir.iterdir() if s.is_dir()]
            except OSError:
                snapshot_dirs = []
            for snapshot in snapshot_dirs:
                try:
                    for link in snapshot.rglob("*"):
                        if link.is_symlink() and not link.exists():
                            # Spare a concurrent sibling's active dangling link (its target blob still
                            # has a fresh .incomplete partner); only purge our own stale
                            # interrupted-download links so the HTTP retry reads clean.
                            if _broken_link_has_active_partner(link, active_grace = active_grace):
                                continue
                            try:
                                link.unlink()
                            except OSError:
                                continue
                except OSError:
                    continue
    except Exception as e:
        logger.debug("default prepare_for_http failed for %s: %s", repo_id, e)


def _active_incomplete_blob_sizes(
    repo_type: Optional[str], repo_id: str, cache_dir: Optional[str] = None
) -> dict[str, int]:
    """Map ``{blob_filename: bytes_present}`` for the repo's ``*.incomplete`` partials.

    Sparse-aware (st_blocks based). The single-file watchdog uses this to follow only the
    partials its own child created, so a concurrent sibling download of a different file in
    the same repo (its partial already present when this download began) cannot mask this
    file's stall by contributing its own progress.
    """
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


def _child_open_incomplete_blobs(pid: int) -> Optional[set]:
    """Basenames of the ``*.incomplete`` blob files the download child *pid* currently has
    open.

    This pinpoints exactly the partial THIS child is writing -- including a resumed prior
    partial that reuses the same blob-hash filename (which Hugging Face does on a retry), so
    a hung resume is still detected -- without confusing it for a concurrent sibling
    download's partial (held open by a different pid). Returns ``None`` when it cannot be
    determined (no ``psutil`` and no ``/proc``, or the process is gone), so the caller falls
    back to a coarser measure; an empty set means the child is running but not yet writing a
    partial (connect / metadata phase).
    """
    # Cross-platform (Linux / macOS / Windows) when psutil is available.
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
    # Linux fallback: read the open fds directly from /proc.
    fd_dir = f"/proc/{pid}/fd"
    try:
        entries = os.listdir(fd_dir)
    except OSError:
        return None  # no /proc (non-Linux) or the process is already gone
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
    """Return ``(total_on_disk_bytes, has_incomplete)`` for the HF cache being written.

    Scans *cache_dir* when the download targets a caller-supplied cache, else the
    active ``HF_HUB_CACHE``. Sparse-aware (st_blocks based) so a sparse Xet/
    ``hf_transfer`` ``.incomplete`` is not mistaken for full-size progress. ``None``
    means the state could not be measured, so callers skip stall logic for that tick.
    """
    try:
        if hf_cache_root(cache_dir = cache_dir) is None:
            return (0, False)

        total = 0
        has_incomplete = False
        for repo_id in repo_ids or []:
            # Skip local paths: HF IDs never start with / . ~, contain "\", or a
            # drive-letter ":" (e.g. C:/models or C:\models on Windows).
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
    interval: float = DEFAULT_HEARTBEAT_INTERVAL,
    stall_timeout: float = DEFAULT_STALL_TIMEOUT,
    xet_disabled: bool = False,
    on_heartbeat: Optional[Callable[[str], None]] = None,
    watch_new_partials_only: bool = False,
    baseline_incomplete_blobs: Optional[set] = None,
    child_pid: Optional[int] = None,
) -> threading.Event:
    """Start a daemon thread that fires ``on_stall(message)`` exactly once iff a
    ``*.incomplete`` is present AND the on-disk size is unchanged for
    *stall_timeout* seconds. The timer resets while no ``*.incomplete`` exists, so
    post-download init is never misread as a stall. Scans *cache_dir* when the
    download targets a caller-supplied cache, else the active ``HF_HUB_CACHE``.
    Returns a stop event the caller sets when the download phase ends.

    When *watch_new_partials_only* is set (single-file downloads), progress is measured only
    over the child's own partial, so a concurrent sibling download of a different file in the
    same repo cannot reset the stall timer with its progress (which would keep a hung child
    alive forever). The child's partial is identified, in order of preference, by the
    ``*.incomplete`` blobs the *child_pid* process actually has open (precise across a
    resumed download that reuses a prior blob-hash filename), else by the partials that did
    NOT already exist in *baseline_incomplete_blobs* (captured before the child started).
    Snapshot downloads keep the repo-wide measurement (every blob is part of the one pull).
    """
    stop = threading.Event()
    transport = "https" if xet_disabled else "xet"
    fired = False
    baseline = set(baseline_incomplete_blobs or ())
    single_repo_id = repo_ids[0] if repo_ids else ""

    def _measure() -> Optional[tuple[int, bool]]:
        if watch_new_partials_only:
            sizes = _active_incomplete_blob_sizes(repo_type, single_repo_id, cache_dir)
            open_names = _child_open_incomplete_blobs(child_pid) if child_pid else None
            if open_names is not None:
                # Precise: only the partials this child process holds open (handles a resumed
                # partial that reuses a baseline blob-hash name, and excludes siblings). hf_xet
                # writes in-process and holds the .incomplete fd continuously, so an EMPTY set
                # here means the child owns no partial YET (the connect / metadata phase), NOT
                # that a helper process owns one -- it must own nothing this tick, so a stalled
                # sibling's post-baseline partial cannot be misattributed and kill a connecting
                # child.
                owned = {name: n for name, n in sizes.items() if name in open_names}
            else:
                # Cannot inspect the child (no psutil / no /proc): best-effort fall back to
                # following only newly-created partials (not in the pre-spawn baseline).
                owned = {name: n for name, n in sizes.items() if name not in baseline}
            return (sum(owned.values()), len(owned) > 0)
        return get_hf_download_state(repo_ids, repo_type = repo_type, cache_dir = cache_dir)

    def _beat() -> None:
        nonlocal fired
        state = _measure()
        last_size = state[0] if state is not None else 0
        last_change = time.monotonic()

        while not stop.wait(interval):
            state = _measure()
            now = time.monotonic()

            if state is None:
                # Unmeasurable this tick (transient FS error): treat as progress
                # so a long unmeasurable gap cannot trip a false stall the instant
                # the state becomes readable again.
                last_change = now
                _safe_status(on_heartbeat, f"Downloading ({transport} transport)...")
                continue

            current_size, has_incomplete = state
            if current_size != last_size:
                last_size = current_size
                last_change = now

            # Reset unless .incomplete confirms an active download, so model init
            # and lock waits are not counted as a stall.
            if not has_incomplete:
                last_change = now
            elif now - last_change >= stall_timeout:
                if not fired:
                    fired = True
                    on_stall(
                        f"Download appears stalled ({transport} transport) "
                        f"-- no progress for {int(now - last_change)}s"
                    )
                return

            _safe_status(on_heartbeat, f"Downloading ({transport} transport)...")

    threading.Thread(target = _beat, daemon = True, name = "hf-xet-watchdog").start()
    return stop


def _scrub_in_child(text: str, token: Optional[str]) -> str:
    """Redact secrets from a child error string, preferring Studio's richer
    patterns when running inside Studio, else the generic redaction."""
    try:
        from hub.utils.download_registry import scrub_secrets  # type: ignore

        return scrub_secrets(text, hf_token = token)
    except Exception:
        return _default_scrub_secrets(text, hf_token = token)


# Deterministic Hub failures that recur identically over either transport, so switching from
# Xet to HTTP is pointless: surface them. Matched by exception class name so the parent need
# not import huggingface_hub's error classes.
_DETERMINISTIC_ERROR_NAMES = frozenset({
    "RepositoryNotFoundError",
    "RevisionNotFoundError",
    "EntryNotFoundError",
    "GatedRepoError",
    "DisabledRepoError",
    "LocalEntryNotFoundError",
    # A required token that is absent locally fails identically over either transport (it never
    # reaches the network), so surface it deterministically with its real type.
    "LocalTokenNotFoundError",
    "BadRequestError",
    # A malformed repo id / revision fails identically over either transport (it never reaches the
    # network), so surface it with its real type instead of a generic RuntimeError or a pointless
    # HTTP retry.
    "HFValidationError",
})
# Names whose TYPE should be reconstructed across the spawn boundary but which must NOT join the
# retry-deterministic shortcut above. ``HfHubHTTPError`` is the base of both the deterministic 4xx
# (401 / 403 / 404 / 416) and the transient 5xx / 429 errors, so the retry decision for it must stay
# status-code driven (``_is_retryable_download_error`` falls through to the status check). But once an
# error has been classified deterministic and surfaced as ``"HfHubHTTPError: <msg>"``, the parent
# should still re-raise the original type so a caller's ``except HfHubHTTPError`` (auth / quota /
# permission handling) keeps working instead of seeing a generic ``RuntimeError``.
_TYPE_PRESERVE_ONLY_NAMES = frozenset({
    "HfHubHTTPError",
})
# Substrings that mark a transient transport failure (hf_xet / CAS error, timeout, reset,
# HTTP 5xx / 429) that disabling Xet and retrying over HTTP may recover.
_TRANSIENT_ERROR_HINTS = (
    "xet", "casclient", "cas_", "timeout", "timed out", "connection", "reset by peer",
    "temporarily", "try again", "incompleteread", "protocolerror", "remotedisconnected",
    "broken pipe", "ssl", "eof occurred", "502", "503", "504", "500 server", "429",
    "too many requests", "service unavailable", "bad gateway", "gateway time",
    "connection aborted",
)


def _resolve_exception_class(type_name: str) -> "Optional[type]":
    """Map a deterministic Hub / OS error class NAME (as captured in the child) back to its class,
    so the parent can re-raise the original type rather than a generic RuntimeError. Best-effort: an
    unknown name returns None. Imports are local so the helper stays import-light when no error
    occurs and never hard-depends on a specific huggingface_hub layout."""
    if type_name == "OSError":
        return OSError
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
    """Build an *exc_cls* instance carrying *message*, robust to a finicky constructor. Hub error
    classes (``RepositoryNotFoundError`` ...) subclass ``HfHubHTTPError``, whose ``response`` arg is
    keyword-only -- and required on some huggingface_hub versions -- so a plain ``exc_cls(message)``
    can raise ``TypeError``. Try the normal constructors first (best fidelity: they default
    ``response`` / ``server_message``), then BYPASS ``__init__`` via ``__new__`` so the TYPE and the
    message survive even when no constructor accepts a lone string. Returns None only if even
    ``__new__`` fails, so the caller can fall back to ``RuntimeError``."""
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
    """Pull the errno out of a stringified OSError. CPython formats it as ``[Errno 28] ...``, so a
    disk-full (ENOSPC) / quota (EDQUOT) error keeps its code across the spawn boundary when the
    parent reconstructs the OSError, letting callers branch on ``exc.errno``."""
    match = re.search(r"\[Errno (\d+)\]", message)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _raise_child_error(message: str) -> None:
    """Re-raise a deterministic child download error, preserving its original exception TYPE when it
    is a known Hub / OS error, so callers that catch ``RepositoryNotFoundError`` / ``GatedRepoError``
    / ``OSError`` (auth prompts, offline handling, disk cleanup) still see those types across the
    spawn-process boundary. The child reports the failure as ``"<ClassName>: <message>"``, so the
    type is reconstructed from that prefix; anything unrecognized -- or a class that cannot be
    instantiated at all -- falls back to ``RuntimeError`` (the prior behavior)."""
    type_name = message.split(":", 1)[0].strip() if ":" in message else ""
    exc_cls = _resolve_exception_class(type_name)
    if exc_cls is None:
        raise RuntimeError(message)
    if exc_cls is OSError:
        # Preserve errno (ENOSPC / EDQUOT ...) so a caller's `except OSError` cleanup can still
        # branch on exc.errno; OSError(message) alone would leave errno = None.
        errno_val = _parse_errno(message)
        if errno_val is not None:
            raise OSError(errno_val, message)
        raise OSError(message)
    exc = _instantiate_preserving_type(exc_cls, message)
    if exc is None:
        raise RuntimeError(message)
    raise exc


def _is_retryable_download_error(exc: BaseException) -> bool:
    """True when a captured download exception looks like a transient transport failure (an
    ``hf_xet`` / CAS error, connection reset, timeout, HTTP 5xx / 429) that the OTHER transport
    may recover, vs a deterministic Hub error (auth, not-found, gated, disk-full) that would
    fail identically. Unknown errors are treated as deterministic, so a real repeatable failure
    is surfaced rather than looped between transports."""
    name = type(exc).__name__
    if name in _DETERMINISTIC_ERROR_NAMES:
        return False
    # Disk full / quota: a different transport cannot help.
    if isinstance(exc, OSError) and getattr(exc, "errno", None) in (errno.ENOSPC, errno.EDQUOT):
        return False
    # An HTTP status (HfHubHTTPError carries a requests / httpx response): 5xx and 429 are
    # transient; other 4xx (401 / 403 / 404 / 416) are deterministic.
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if not isinstance(status, int):
        status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        # 5xx server errors, 429 rate-limit, 408 request-timeout are transient; other 4xx
        # (401 / 403 / 404 / 416) are deterministic and would fail identically over HTTP.
        return status >= 500 or status in (408, 429)
    text = f"{name}: {exc}".lower()
    return any(hint in text for hint in _TRANSIENT_ERROR_HINTS)


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
    """Spawn-child entrypoint: download and report the result.

    Top-level and picklable. Sets the Xet env BEFORE importing huggingface_hub,
    forms its own process group so the parent can kill the whole transfer, and
    never logs the token or signed URLs.
    """
    # Die with the parent on Linux when running under Studio (best-effort; the
    # module is absent standalone, in which case there is nothing to bind to).
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
        # Keep the HTTP writer sequential and resumable (hf_transfer leaves sparse
        # partials a sequential resume cannot safely continue).
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    repo_id = params["repo_id"]

    # Test-only fault injection (never set in production): stall the Xet attempt
    # so the watchdog + HTTP fallback can be exercised against a real repo.
    if not disable_xet and os.environ.get("UNSLOTH_HF_XET_FORCE_STALL") == "1":
        _stall_fh = None
        try:
            from huggingface_hub.constants import HF_HUB_CACHE

            # Write the fake partial under the SAME cache the watchdog scans
            # (params["cache_dir"] when the caller set one, else HF_HUB_CACHE) and
            # under the repo_type-correct dir name, so has_active_incomplete_blobs
            # sees it and the stall/HTTP fallback actually fires in tests.
            cache_root = params.get("cache_dir") or HF_HUB_CACHE
            repo_dir_name = f"{repo_type or 'model'}s--" + repo_id.replace("/", "--")
            blobs = os.path.join(cache_root, repo_dir_name, "blobs")
            os.makedirs(blobs, exist_ok = True)
            # Hold the fake partial OPEN for the whole stall. The snapshot watchdog finds it by
            # filename (has_active_incomplete_blobs), but the single-file watchdog
            # (watch_new_partials_only) counts ONLY partials this child PID holds open via
            # _child_open_incomplete_blobs -- a closed file there is ignored and the stall never
            # trips. Keeping the fd open lets BOTH modes see it. The handle is bound to a local so
            # it stays open across the sleep below.
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
        # Classify here, where the exception object (status code, errno, type) is intact, so the
        # parent can retry a transient Xet transport failure over HTTP and still surface a
        # deterministic Hub error without a pointless second attempt.
        result_queue.put({
            "ok": False,
            "error": _scrub_in_child(f"{type(e).__name__}: {e}", token),
            "retryable": _is_retryable_download_error(e),
        })


def _terminate_process_group(proc: "mp.process.BaseProcess", grace_period: float) -> None:
    """Kill *proc* and its whole process group (Xet may spawn helper procs).

    The child calls ``os.setsid()`` so its pgid equals its pid; signal via
    ``os.killpg(pid, ...)`` -- NOT ``getpgid``, which before the child becomes a
    group leader resolves to OUR group. SIGTERM, then SIGKILL after *grace_period*.
    """
    pid = proc.pid

    def _signal_group(sig: int) -> None:
        if pid is not None and hasattr(os, "killpg"):
            try:
                os.killpg(pid, sig)
                return
            except (ProcessLookupError, PermissionError, OSError):
                pass
        # Windows or pre-setsid: best effort on the single process.
        try:
            proc.terminate() if sig != getattr(signal, "SIGKILL", -9) else proc.kill()
        except Exception:
            pass

    _signal_group(getattr(signal, "SIGTERM", signal.SIGINT))
    proc.join(timeout = grace_period)
    # Post-grace SIGKILL only while the leader is still alive, so its pid (== pgid after setsid) is
    # a live target. Once proc.join() reaps a leader that exited on SIGTERM, that pid is free and a
    # busy host can recycle it into an unrelated setsid'd group within the grace window -- a
    # killpg(pid) would then signal the WRONG group. hf_xet 1.5.x writes in-process and spawns no
    # helper procs, so a reaped leader leaves nothing in the group to clean up.
    if proc.is_alive():
        _signal_group(getattr(signal, "SIGKILL", signal.SIGTERM))
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
    stall_timeout: float,
    interval: float,
    grace_period: float,
    on_status: Optional[Callable[[str], None]],
) -> tuple[str, Optional[str]]:
    """Run one download in a spawn child supervised by the no-progress watchdog.

    Returns ``("ok", path)``, ``("stall", None)``, ``("cancelled", None)``,
    ``("crashed", message)`` (process-level crash, no captured exception),
    ``("retryable_error", message)`` (a transient Xet transport failure worth an HTTP retry),
    or ``("error", message)`` (a deterministic Hub error). This is the seam tests monkeypatch
    to avoid spawning.
    """
    # A single-file download scopes its stall detection to its own child's partials.
    # Capture the partials already on disk for this repo BEFORE spawning, so the watchdog
    # can ignore a concurrent sibling's in-flight partial (a different file in the same
    # repo) and only follow the blob(s) this child newly writes. Snapshots stay repo-wide.
    baseline_partials: Optional[set] = None
    if kind == "file":
        baseline_partials = set(
            _active_incomplete_blob_sizes(repo_type, repo_id, params.get("cache_dir"))
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
    # Set the transport env in THIS process around the spawn so the child inherits
    # it from creation. HF reads HF_HUB_DISABLE_XET into constants at import time,
    # and a spawn child re-imports the (heavy) unsloth_zoo package -- importing
    # huggingface_hub -- before the child body runs, so a child-side os.environ
    # assignment would land too late. The child still sets it too, defensively.
    child_env = {
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        # The download child is a fresh spawn interpreter that only needs
        # huggingface_hub; tell unsloth_zoo's __init__ to skip its heavy torch /
        # transformers / device init in that process (the parent keeps full init).
        "UNSLOTH_ZOO_DISABLE_GPU_INIT": "1",
    }
    if disable_xet:
        child_env["HF_HUB_DISABLE_XET"] = "1"
        child_env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    with _SPAWN_ENV_LOCK:
        # Cache huggingface_hub's transport constants in the PARENT from the REAL environment NOW,
        # before the child-only env (HF_HUB_DISABLE_XET=1) is briefly set below. Hub reads
        # HF_HUB_DISABLE_XET into a module constant at import time; without this, a concurrent thread
        # doing its FIRST `import huggingface_hub` inside the spawn window could cache the child-only
        # disabled-Xet value in the parent and silently route later in-process downloads over HTTP.
        # Once imported it is a no-op, so a concurrent import in the window then re-reads nothing.
        try:
            import huggingface_hub.constants  # noqa: F401
        except Exception:
            pass
        saved_env = {k: os.environ.get(k) for k in child_env}
        # multiprocessing 'spawn' reconstructs __main__ in the child from
        # __main__.__file__. If that is a pseudo-path ('<stdin>', a notebook) the
        # child fails to start; if it is a real but UNGUARDED caller script the
        # child re-imports it as __mp_main__ and re-runs the top-level
        # from_pretrained/download, hitting the "start a new process before
        # bootstrapping" error -> the parent then sees the child exit without a
        # result. In every case we only need the child to unpickle and run
        # _download_child_entry, so point __main__ at THIS importable, side-effect
        # -free module for the spawn (and restore it after). The child imports us
        # as __mp_main__ instead of re-executing the caller's script.
        main_module = sys.modules.get("__main__")
        saved_main_file = _UNSET
        saved_main_spec = _UNSET
        if main_module is not None:
            saved_main_file = getattr(main_module, "__file__", _UNSET)
            main_module.__file__ = __file__
            # When the caller was launched as a module (python -m pkg), spawn's
            # preparation prefers __main__.__spec__.name over __file__ and re-imports
            # the user's module BY NAME -> re-runs its top-level from_pretrained in
            # the child and hits the bootstrapping error. Clearing __spec__ forces
            # the path branch, which uses the __file__ we just repointed at this
            # side-effect-free helper module.
            saved_main_spec = getattr(main_module, "__spec__", _UNSET)
            main_module.__spec__ = None
        try:
            os.environ.update(child_env)
            proc.start()
        except BaseException:
            # proc.start() can raise (e.g. OSError "can't start new process" under fd /
            # thread exhaustion). The result_queue's OS pipe fds were allocated above, but
            # the lifecycle try/finally that closes them is only entered AFTER a successful
            # start, so on a failed spawn that cleanup never runs and the fds leak. Close
            # the queue here so a failed spawn is deterministic rather than fd-leaking.
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

    # Bind the child to the parent lifetime when running under Studio (best-effort).
    try:
        from utils.process_lifetime import adopt_pid  # type: ignore

        adopt_pid(proc.pid)
    except Exception:
        pass

    stalled = threading.Event()
    # start_watchdog creates and starts a thread; if that raises (e.g. "can't start new thread"
    # under thread/FD exhaustion), the child already started above must STILL be terminated. So it
    # runs inside the try whose finally reaps the child; stop_watchdog stays None until it succeeds.
    stop_watchdog = None
    result: Optional[dict] = None
    try:
        stop_watchdog = start_watchdog(
            repo_ids = [repo_id],
            on_stall = lambda msg: stalled.set(),
            repo_type = repo_type,
            cache_dir = params.get("cache_dir"),
            interval = interval,
            stall_timeout = stall_timeout,
            xet_disabled = disable_xet,
            on_heartbeat = on_status,
            watch_new_partials_only = (kind == "file"),
            baseline_incomplete_blobs = baseline_partials,
            child_pid = proc.pid,
        )
        while proc.is_alive():
            if cancel_event is not None and cancel_event.is_set():
                _terminate_process_group(proc, grace_period)
                return ("cancelled", None)
            if stalled.is_set():
                # Prefer a result the child enqueued in the same ~interval window the watchdog
                # fired in over a late stall, so a download that just succeeded is not killed and
                # needlessly retried over HTTP. A spawn Queue has a child-side feeder thread, so a
                # result put microseconds earlier is not yet readable by get_nowait(); use a short
                # timeout (matching the process-exit drain below) to let the pipe flush.
                try:
                    result = result_queue.get(timeout = 1.0)
                    break
                except queue.Empty:
                    pass
                _terminate_process_group(proc, grace_period)
                return ("stall", None)
            try:
                result = result_queue.get(timeout = _POLL_INTERVAL)
                break
            except queue.Empty:
                continue
        else:
            # Process exited; drain any result it enqueued. Use a short timeout,
            # not get_nowait(): the child can exit microseconds before its queue
            # feeder flushes the pipe, and a bare get_nowait() would then spuriously
            # report "exited without a result" on an otherwise successful download.
            try:
                result = result_queue.get(timeout = 1.0)
            except queue.Empty:
                result = None
    finally:
        if stop_watchdog is not None:
            stop_watchdog.set()
        proc.join(timeout = grace_period)
        # Any exit from the loop -- normal completion, cancel/stall, or an
        # unexpected exception (e.g. KeyboardInterrupt) -- must not leak the child.
        # If it is still alive after the grace join, kill its whole process group.
        # _terminate_process_group is idempotent, so a redundant call after the
        # cancel/stall branch already terminated it is a harmless no-op.
        if proc.is_alive():
            _terminate_process_group(proc, grace_period)
        # Release the queue's pipe fds deterministically rather than waiting for GC (which is
        # fragile when the child was killed mid-put). The result, if any, is already extracted,
        # and a killed child has nothing more to flush, so cancel the feeder join before close.
        try:
            result_queue.cancel_join_thread()
            result_queue.close()
        except Exception:
            pass

    if result is None:
        # The child exited without enqueuing a result: a process-level crash (e.g. a native
        # hf_xet abort / segfault), NOT a captured Hub exception. No deterministic error was
        # observed, so the other transport may still succeed -- report it as "crashed" so the
        # caller can retry over HTTP rather than surfacing a hard error.
        return (
            "crashed",
            f"download process for '{repo_id}' exited "
            f"(code={proc.exitcode}) without a result",
        )
    if result.get("ok"):
        return ("ok", result["path"])
    message = result.get("error") or "unknown download error"
    if result.get("retryable"):
        # A transient transport failure the child flagged as worth another transport.
        return ("retryable_error", message)
    return ("error", message)


def _intact_subset(
    snapshot_dir: Path, *, repo_type: str, allow_patterns: Any, ignore_patterns: Any,
) -> bool:
    """No interrupted-download evidence for the files the request SELECTS: no dangling requested
    symlink, and every EXACT-named requested file present. Used for a weightless / non-model request
    (a dataset, a tokenizer-only allow list) and as the breakage check for a finished download. A
    dangling EXCLUDED weight from an earlier interrupted pull does not reject a complete subset."""
    return (
        not snapshot_has_requested_broken_symlinks(
            snapshot_dir, allow_patterns = allow_patterns, ignore_patterns = ignore_patterns,
            repo_type = repo_type,
        )
        and requested_named_files_present(
            snapshot_dir, allow_patterns = allow_patterns, ignore_patterns = ignore_patterns
        )
    )


def _has_any_weight(snapshot_dir: Path) -> bool:
    """True if the snapshot holds at least one loadable model weight anywhere (root or a component
    subfolder). Lenient on purpose: it only distinguishes a real model warm from the config-only
    stale snapshot HF can hand back on an offline / timed-out request, without classifying layout."""
    try:
        for entry in snapshot_dir.rglob("*"):
            if _is_loadable_weight_file(entry.name):
                try:
                    if entry.is_file():
                        return True
                except OSError:
                    continue
    except OSError:
        return False
    return False


def _cache_can_skip_download(
    snapshot_dir: Path, *, repo_type: str, allow_patterns: Any, ignore_patterns: Any,
) -> bool:
    """PRE-download: whether a locally cached snapshot is complete enough that the in-process load
    will not fetch anything, so the protective child can be skipped.

    STRICT for a weight-bearing model request: only the conservative canonical fast-path
    (``snapshot_dir_is_complete``) may skip the child; anything uncertain (diffusers, variants,
    non-trivial patterns, sharded-without-index) returns False -> spawn the child. A false True here
    would let the in-process load fetch a missing weight over un-killable Xet (the hang). A weightless
    / non-model request has no weight to hang on, so an intact requested subset is enough -- this
    preserves the offline short-circuit for a tokenizer-only / dataset warm."""
    if repo_type in (None, "model") and request_can_include_weights(allow_patterns, ignore_patterns):
        return snapshot_dir_is_complete(
            snapshot_dir, allow_patterns = allow_patterns, ignore_patterns = ignore_patterns,
        )
    return _intact_subset(
        snapshot_dir, repo_type = repo_type, allow_patterns = allow_patterns,
        ignore_patterns = ignore_patterns,
    )


def _download_result_usable(
    snapshot_dir: Path, *, repo_type: str, allow_patterns: Any, ignore_patterns: Any,
) -> bool:
    """POST-download: whether the child's ``snapshot_download`` result is usable, or should be retried
    over HTTP. snapshot_download already did the authoritative manifest compare + resume, so accept
    unless there is POSITIVE evidence of a silent-Xet partial: a dangling REQUESTED symlink (a blob
    that is missing or still ``.incomplete``), or a weight-bearing model warm that came back with NO
    weight at all (HF handed back a stale config-only snapshot on an offline / timed-out request).
    LENIENT otherwise -- a finished diffusers / variant / either-format download passes, and a named
    file simply absent from the repo is not treated as missing -- so a good download is never failed
    and re-looped into a ``DownloadStallError``.

    The no-weight rejection fires whenever the request can include weights (``request_can_include_weights``):
    an unpatterned model warm, or an explicit weight request (``allow_patterns=['model.safetensors']``)
    that came back with no weight, is a stale config-only snapshot and is retried. A genuinely weightless
    request (``allow_patterns=['tokenizer*']`` / ``['*.json']``) reads weightless there, so its valid
    no-weight result is accepted rather than failed.

    It also rejects an interrupted CANONICAL sharded warm (loose ``model-00001-of-00002.safetensors``
    without its index or a sibling shard) for an unpatterned request: that layout has a loadable weight
    file but a default load still cannot read it and would fetch the rest over un-killable Xet."""
    if snapshot_has_requested_broken_symlinks(
        snapshot_dir, allow_patterns = allow_patterns, ignore_patterns = ignore_patterns,
        repo_type = repo_type,
    ):
        return False
    if repo_type in (None, "model"):
        if (
            request_can_include_weights(allow_patterns, ignore_patterns)
            and not _has_any_weight(snapshot_dir)
        ):
            return False
        if allow_patterns is None and _has_incomplete_canonical_root_shards(snapshot_dir):
            return False
    return True


def _snapshot_payload_incomplete(
    payload: Any, *, repo_type: str, allow_patterns: Any, ignore_patterns: Any
) -> bool:
    """True when a snapshot download returned a real directory that is not usable for the request
    (see ``_download_result_usable``). Guarded to an existing directory so a mocked / non-path
    payload (unit tests) or an unexpected return is trusted rather than rejected; in production the
    child always returns a real snapshot dir, where this catches HF handing back an existing partial
    snapshot on an offline / timed-out request."""
    try:
        path = Path(payload)
    except (TypeError, ValueError, OSError):
        # Non-path payload (unit-test sentinel) or, on Windows, a path with invalid characters
        # (ValueError / OSError): trust it rather than reject -- production always returns a real dir.
        return False
    try:
        if not path.is_dir():
            return False
    except OSError:
        return False
    return not _download_result_usable(
        path, repo_type = repo_type, allow_patterns = allow_patterns,
        ignore_patterns = ignore_patterns,
    )


def _download_with_xet_fallback(
    *,
    repo_id: str,
    label: str,
    kind: str,
    params: dict,
    token: Optional[str],
    repo_type: str,
    cancel_event: Optional[threading.Event],
    stall_timeout: float,
    interval: float,
    grace_period: float,
    on_status: Optional[Callable[[str], None]],
    prepare_for_http_fn: Optional[Callable[[str, str], None]],
) -> str:
    """Shared 2-attempt loop: Xet primary, HTTP on a stall. Returns the local path."""
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Cancelled")

    cache_dir = params.get("cache_dir")
    # The Unsloth/HF knobs can force HTTP from the very first attempt. xet_force_disabled() reads
    # os.environ["HF_HUB_DISABLE_XET"] live, and a CONCURRENT download briefly sets that var in the
    # parent env around its spawn (under _SPAWN_ENV_LOCK) so its child inherits it. Read under the
    # same lock so this download cannot observe the other's child-only value and wrongly force itself
    # onto HTTP from the start.
    with _SPAWN_ENV_LOCK:
        disable_xet = xet_force_disabled()

    for attempt in range(2):
        if disable_xet:
            # Purge a non-HTTP partial before resuming over HTTP: an HTTP resume
            # over a sparse Xet/hf_transfer partial silently corrupts the blob.
            # The generic purge is cache_dir-aware; an injected (Studio) hook owns
            # its own cache accounting and keeps the (repo_type, repo_id) signature.
            try:
                if prepare_for_http_fn is None:
                    _default_prepare_for_http(
                        repo_type, repo_id, cache_dir = cache_dir, active_grace = stall_timeout
                    )
                else:
                    prepare_for_http_fn(repo_type, repo_id)
            except Exception as e:
                logger.debug("prepare_for_http failed for %s: %s", repo_id, e)
            # If an unsafe partial could not be cleared (e.g. a locked file or a
            # permission error), an HTTP resume over a sparse Xet/hf_transfer
            # partial would silently corrupt the blob. Force a clean re-download
            # for this HTTP attempt instead of resuming over it.
            if has_active_incomplete_blobs(repo_type, repo_id, cache_dir = cache_dir):
                logger.warning(
                    "Unsafe partial for '%s' could not be cleared; forcing a clean "
                    "HTTP re-download instead of an unsafe resume.", label
                )
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

        if kind_result == "ok":
            if kind == "snapshot" and _snapshot_payload_incomplete(
                payload,
                repo_type = repo_type,
                allow_patterns = params.get("allow_patterns"),
                ignore_patterns = params.get("ignore_patterns"),
            ):
                # HF can return an existing, incomplete snapshot dir on an offline or
                # timed-out request instead of fetching the missing files. Never hand an
                # incomplete snapshot to the in-process load: retry over HTTP, and if it
                # still comes back incomplete, fail loudly rather than silently loading a
                # broken cache. (A patterned / non-model request is judged by its own
                # requested subset, so this never rejects a valid weightless snapshot.)
                if not disable_xet:
                    logger.warning(
                        "Download for '%s' returned an incomplete snapshot -- "
                        "retrying with HF_HUB_DISABLE_XET=1", label
                    )
                    _safe_status(on_status, f"{label}: incomplete snapshot, retrying over HTTP")
                    disable_xet = True
                    continue
                raise DownloadStallError(
                    f"Download for '{label}' returned an incomplete snapshot even with "
                    f"HF_HUB_DISABLE_XET=1 -- missing files, check your network connection"
                )
            return payload  # type: ignore[return-value]
        if kind_result == "cancelled":
            raise RuntimeError("Cancelled")
        if kind_result == "error":
            # Deterministic failure (a captured Hub exception: auth, not-found, gated, disk
            # full): the other transport would fail identically, so do not retry. Re-raise
            # preserving the original exception type (RepositoryNotFoundError / GatedRepoError /
            # OSError ...) where known, so callers' typed except clauses still match across the
            # spawn boundary; unknown errors fall back to RuntimeError.
            _raise_child_error(payload)
        if kind_result == "retryable_error":
            # A transient transport failure (hf_xet CAS timeout, 5xx, connection reset) rather
            # than a deterministic Hub error: disabling Xet and retrying over HTTP may recover,
            # so try the other transport once before surfacing it (mirrors the crash / stall
            # paths). If HTTP also failed, there is no other transport left -- raise.
            if not disable_xet:
                logger.warning(
                    "Download for '%s' hit a transient Xet transport error -- retrying "
                    "with HF_HUB_DISABLE_XET=1: %s", label, payload
                )
                _safe_status(on_status, f"{label}: transient Xet error, retrying over HTTP")
                disable_xet = True
                continue
            raise RuntimeError(payload)
        if kind_result == "crashed":
            # A process-level crash with no captured exception: HTTP may still succeed, so
            # retry over it once before surfacing a hard error (mirrors the stall path).
            if not disable_xet:
                logger.warning(
                    "Download process for '%s' crashed without a result -- "
                    "retrying with HF_HUB_DISABLE_XET=1", label
                )
                _safe_status(on_status, f"{label}: download crashed, retrying over HTTP")
                disable_xet = True
                continue
            raise RuntimeError(payload)
        # kind_result == "stall"
        if not disable_xet:
            logger.warning(
                "Download stalled for '%s' -- retrying with HF_HUB_DISABLE_XET=1", label
            )
            # _safe_status: a raising status hook (e.g. a disconnected client) must
            # not abort the retry before disable_xet is set, turning a recoverable
            # stall into a failed download.
            _safe_status(on_status, f"{label}: Xet stalled, retrying over HTTP")
            disable_xet = True
            continue
        raise DownloadStallError(
            f"Download stalled for '{label}' even with HF_HUB_DISABLE_XET=1 "
            f"-- check your network connection"
        )

    # Unreachable: the loop either returns or raises on each attempt.
    raise DownloadStallError(f"Download failed for '{label}'")


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
    stall_timeout: float = DEFAULT_STALL_TIMEOUT,
    interval: float = DEFAULT_HEARTBEAT_INTERVAL,
    grace_period: float = DEFAULT_GRACE_PERIOD,
    on_status: Optional[Callable[[str], None]] = None,
    prepare_for_http_fn: Optional[Callable[[str, str], None]] = None,
) -> str:
    """Download a single file with Xet primary and HTTP as a stall-only fallback.

    Returns the local cache path. Raises ``RuntimeError("Cancelled")`` if
    *cancel_event* is set, re-raises a deterministic child error unchanged (no
    fallback), and raises ``DownloadStallError`` only if BOTH transports stall.
    ``force_download=True`` re-fetches even if cached (skips the cache short-circuit).
    ``local_files_only=True`` resolves from cache in-process and never spawns a
    network child (matching Hugging Face offline semantics). ``subfolder`` is
    forwarded to ``hf_hub_download`` for files stored under a repo subdirectory.
    """
    repo_type = repo_type or "model"  # HF treats None as the default model repo.
    # Expand ~ as huggingface_hub does before writing, so the cache probe below and
    # the child both resolve to the same on-disk location (else a warm ~/hf-cache
    # is missed and we spawn a child for an already-cached file). Path-like cache
    # dirs are normalized too, since HF accepts pathlib.Path.
    if isinstance(cache_dir, (str, os.PathLike)):
        cache_dir = os.path.expanduser(os.fspath(cache_dir))
    # Honor an already-set cancellation before any cache probe or network work. The offline and
    # warm-cache short-circuits below return without reaching _download_with_xet_fallback (which
    # holds the only other cancel check), so a request cancelled before this point must not
    # resolve and hand back a cached file.
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Cancelled")
    # Offline: resolve purely from the local cache, never reaching the network. HF
    # raises LocalEntryNotFoundError if it is not cached; let that propagate.
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
    # Finalized blob already cached: return it with no child and no network
    # (skipped when force_download re-fetches unconditionally). The cache stores a
    # subfolder file under "<subfolder>/<filename>", which is what the probe wants.
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
    cancel_event: Optional[threading.Event] = None,
    stall_timeout: float = DEFAULT_STALL_TIMEOUT,
    interval: float = DEFAULT_HEARTBEAT_INTERVAL,
    grace_period: float = DEFAULT_GRACE_PERIOD,
    on_status: Optional[Callable[[str], None]] = None,
    prepare_for_http_fn: Optional[Callable[[str, str], None]] = None,
) -> str:
    """Download a whole repo snapshot with Xet primary and HTTP as a stall-only
    fallback, returning the local snapshot dir.

    Used by Unsloth's ``from_pretrained`` to warm the cache in a killable child
    BEFORE the in-process model load (which then hits a warm cache and cannot
    hang on a native Xet thread). A fully cached repo short-circuits in-process
    via ``local_files_only`` with no child and no network. ``force_download=True``
    re-fetches in the killable child even if cached (skips that short-circuit).
    ``local_files_only=True`` resolves from cache in-process and never spawns a
    network child (matching Hugging Face offline semantics).
    """
    repo_type = repo_type or "model"  # HF treats None as the default model repo.
    # Expand ~ as huggingface_hub does before writing, so the probe and the child
    # resolve to the same on-disk cache location.
    if isinstance(cache_dir, (str, os.PathLike)):
        cache_dir = os.path.expanduser(os.fspath(cache_dir))
    # Honor an already-set cancellation before any cache probe or network work. The offline and
    # warm-cache short-circuits below return without reaching _download_with_xet_fallback (which
    # holds the only other cancel check), so a request cancelled before this point must not
    # resolve and hand back a snapshot.
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Cancelled")
    # Offline: resolve purely from the local cache, never reaching the network. HF
    # raises if the snapshot is not cached; let that propagate.
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
    # Fast path: everything already on disk -> resolve in-process (no Xet, no
    # hang). Skipped when force_download re-fetches unconditionally.
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
            # local_files_only returns a snapshot dir whenever refs/<rev> and
            # snapshots/<sha> exist, even one left by a prior interrupted or patterned
            # download (a config-only snapshot from an AutoConfig fetch, or a partial
            # shard pull). Validate the EXACT returned revision dir against the request:
            # a full model warmup may skip the child only when its canonical weights are
            # provably complete (the conservative fast-path gate); a patterned / non-model
            # request only needs its referenced files (no dangling symlinks). Complete it in
            # the killable child otherwise, so the in-process load never proceeds with missing
            # files. Scope the check to the returned snapshot, NOT the whole repo: an
            # unrelated revision mid-download (a stale .incomplete blob or a broken older
            # snapshot elsewhere in the same repo cache) must not force a needless re-fetch.
            if _cache_can_skip_download(
                Path(cached_dir),
                repo_type = repo_type,
                allow_patterns = allow_patterns,
                ignore_patterns = ignore_patterns,
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
    )
