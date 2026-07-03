# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# This file is licensed under the GNU Affero General Public License v3.0 only
# (AGPL-3.0-only), unlike the rest of unsloth_zoo which is LGPL-3.0-or-later. It
# is the single shared home for the Xet -> HTTP stall fallback used by both
# Unsloth (FastModel.from_pretrained) and Unsloth Studio, which imports it.
# See <https://www.gnu.org/licenses/agpl-3.0.html>.

"""Xet-primary HF downloads with an automatic HTTP fallback on a no-progress stall.

Xet (``hf_xet``) is the fast default but can hang with no progress, no exception, and a native thread
that cannot be killed. Keep Xet primary and fall back to plain HTTP only when the parent observes a
stall. ``HF_HUB_DISABLE_XET`` is read at import time, so the fallback runs in a fresh ``spawn`` child
(not a thread) that sets the env before importing ``huggingface_hub``. Cached files short-circuit with
no child; deterministic errors (401/403/404/disk-full) and cancellation propagate without a fallback.

``hf_hub_download_with_xet_fallback`` does a single file; ``snapshot_download_with_xet_fallback`` does
a whole repo (the entrypoint Unsloth's ``from_pretrained`` uses to warm the cache in a killable child
before the in-process load). Studio cache / secret / process helpers are used best-effort (imported
only if present) or injected, so one body runs both inside Studio and in Unsloth.

The spawn child sets ``UNSLOTH_ZOO_DISABLE_GPU_INIT=1`` before importing the package, selecting
``unsloth_zoo``'s lightweight import path (no torch / transformers) so each child stays fast.
"""

from __future__ import annotations

import builtins
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
    _ROOT_MODEL_VARIANT_WEIGHT_RE,
    _as_pattern_list,
    _diffusers_component_shards_incomplete,
    _diffusers_declared_components,
    _filter_paths,
    _has_glob,
    _has_incomplete_canonical_root_shards,
    _has_incomplete_variant_root_shards,
    _is_loadable_weight_file,
    _selected_shard_index_incomplete,
    blob_bytes_present,
    has_active_incomplete_blobs,
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
    """Invoke a status / heartbeat callback without letting it kill the daemon watchdog thread: a
    disconnected Studio client can make on_status raise, which would stop stall detection."""
    if callback is None:
        return
    try:
        callback(message)
    except Exception as e:
        logger.debug("watchdog status callback raised (ignored): %s", e)


class DownloadStallError(RuntimeError):
    """Raised when no download progress is observed for too long. Canonical home; Studio re-imports it
    so all paths share one type."""


def is_hf_xet_available() -> bool:
    """True iff the ``hf_xet`` extra is importable (Hub uses it automatically)."""
    try:
        return importlib.util.find_spec("hf_xet") is not None
    except Exception:
        return False


def xet_force_disabled() -> bool:
    """Whether the user asked to skip Xet up front (force HTTP), via ``UNSLOTH_DISABLE_XET`` /
    ``UNSLOTH_STABLE_DOWNLOADS`` or HF's own ``HF_HUB_DISABLE_XET``."""
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

    # Match the query up to whitespace OR a structural delimiter (quote, bracket, brace, paren, angle,
    # pipe): a signed URL embedded in JSON / a dict repr / other structured text has no surrounding
    # whitespace, so a greedy [^\s]* would swallow the trailing "} / ") and replace it with ***,
    # corrupting the log line. Real signed-query values percent-encode these chars, so the redaction of
    # a genuine presigned URL is unaffected.
    out = re.sub(
        r"(https?://[^\s?]+)\?([^\s\"'()<>{}|[\]]*)", _redact_signed_query, out
    )
    return out


def _broken_link_has_active_partner(link: Path, *, active_grace: float) -> bool:
    """True if a dangling snapshot symlink should be SPARED because a concurrent sibling download is
    still writing the blob it points at. The discriminator is a FRESH ``.incomplete`` partner of the
    target blob, NOT the link's own mtime: our own killed child's ``.incomplete`` was static for the
    full stall timeout and is purged first (no partner -> link cleared), while a sibling mid-download
    still has a growing partner (link spared)."""
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


def _link_incomplete_partner_name(link: Path) -> Optional[str]:
    """The ``<hash>.incomplete`` basename for a dangling snapshot symlink's target blob, or None."""
    try:
        target = Path(os.readlink(link))
        return target.name + INCOMPLETE_SUFFIX
    except OSError:
        return None


def _default_prepare_for_http(
    repo_type: str,
    repo_id: str,
    *,
    cache_dir: Optional[str] = None,
    active_grace: float = DEFAULT_STALL_TIMEOUT,
    owned_incomplete_blobs: Optional[set] = None,
) -> None:
    """Make the partial safe for an HTTP resume: delete the repo's active ``*.incomplete`` blobs (an
    HTTP resume over a sparse Xet / hf_transfer partial silently corrupts the blob) and the broken
    snapshot symlinks the detector counts as active (else the retry inherits stale state and re-trips).
    Studio injects its marker-aware version instead. ``iter_active_repo_cache_dirs`` is case-collision
    safe, so this destructive purge only touches an unambiguous repo cache dir.

    When *owned_incomplete_blobs* is given (the basenames the stalled child held open, captured before
    it was killed), the purge is SCOPED to them, so a concurrent same-repo sibling writing a DIFFERENT
    blob is never touched even if its partial aged past *active_grace*. None -> coarser mtime guard only.
    """
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
                            # Spare a partial written within active_grace: a slower sibling that just
                            # has not written recently is not stalled. Our own killed partial has been
                            # static for the full stall timeout, so it is purged.
                            if time.time() - blob.stat().st_mtime < active_grace:
                                continue
                            blob.unlink()
                        except OSError:
                            continue  # a locked / permission-denied blob must not abort the rest
            # A broken snapshot symlink also reads as active incomplete state; clear those too. Sweep
            # EVERY snapshot (the detector inspects all), else a dangling link under an older revision
            # keeps the repo marked incomplete and re-trips the watchdog.
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
                            if owned_incomplete_blobs is not None and (
                                _link_incomplete_partner_name(link) not in owned_incomplete_blobs
                            ):
                                continue
                            # Spare a sibling's active link (target blob still has a fresh .incomplete).
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
    """Map ``{blob_filename: bytes_present}`` (sparse-aware) for the repo's ``*.incomplete`` partials.
    The single-file watchdog uses it to follow only its own child's partials, so a concurrent sibling
    download of a different file cannot mask this file's stall with its own progress."""
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
    """Basenames of the ``*.incomplete`` blobs the download child *pid* currently has open -- exactly
    the partial THIS child is writing (incl. a resumed partial that reuses a prior blob-hash name),
    not a sibling's (held by a different pid). ``None`` when undeterminable (no ``psutil`` / ``/proc``,
    or the process is gone) -> caller uses a coarser measure; an empty set means the child is not yet
    writing a partial (connect / metadata phase)."""
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
    """Return ``(total_on_disk_bytes, has_incomplete)`` for the HF cache being written (sparse-aware,
    so a partial Xet / ``hf_transfer`` blob is not read as full progress). Scans *cache_dir* or the
    active ``HF_HUB_CACHE``. A missing / empty cache reads as ``(0, False)``; ``None`` is returned only
    on a probe exception (unmeasurable -> callers skip stall logic this tick)."""
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
    interval: float = DEFAULT_HEARTBEAT_INTERVAL,
    stall_timeout: float = DEFAULT_STALL_TIMEOUT,
    xet_disabled: bool = False,
    on_heartbeat: Optional[Callable[[str], None]] = None,
    watch_new_partials_only: bool = False,
    baseline_incomplete_blobs: Optional[set] = None,
    child_pid: Optional[int] = None,
) -> threading.Event:
    """Start a daemon thread that fires ``on_stall(message)`` exactly once iff a ``*.incomplete`` is
    present AND the on-disk size is unchanged for *stall_timeout* seconds. The timer resets while no
    ``*.incomplete`` exists, so post-download init is not misread as a stall. Returns a stop event the
    caller sets when the download phase ends.

    With *watch_new_partials_only* (single-file), progress is measured only over the child's own
    partial, so a concurrent sibling pull of a different file cannot reset the timer and keep a hung
    child alive. The child's partial is identified by the blobs *child_pid* has open (precise across a
    resumed download), else by the partials not in *baseline_incomplete_blobs* (captured pre-spawn).
    Snapshots keep the repo-wide measurement (every blob is part of the one pull)."""
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
                # Only the partials this child holds open (handles a resumed partial reusing a baseline
                # name, excludes siblings). hf_xet holds the .incomplete fd continuously, so an EMPTY
                # set means the child owns no partial YET (connect / metadata phase), not a sibling's.
                owned = {name: n for name, n in sizes.items() if name in open_names}
            else:
                # No psutil / /proc: fall back to following only newly-created (post-baseline) partials.
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
                # Unmeasurable this tick (transient FS error): treat as progress so the gap cannot
                # trip a false stall once the state becomes readable again.
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
    "LocalTokenNotFoundError",  # a missing required token fails identically over either transport
    "BadRequestError",
    "HFValidationError",        # a malformed repo id / revision never reaches the network
})
# Names whose TYPE is reconstructed across the spawn boundary but which must NOT join the
# retry-deterministic set above: ``HfHubHTTPError`` is the base of both deterministic 4xx and transient
# 5xx / 429, so its retry decision stays status-code driven. Once classified deterministic and surfaced
# as ``"HfHubHTTPError: <msg>"``, the parent still re-raises the real type so ``except HfHubHTTPError``
# keeps working instead of seeing ``RuntimeError``.
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
    # Preserve builtin OSError subclasses (PermissionError, FileNotFoundError, ...): these are
    # deterministic filesystem failures (e.g. an unwritable custom cache) the child cannot retry away,
    # so a caller's `except OSError` / `except PermissionError` must still fire rather than see the
    # generic RuntimeError the resolver would otherwise fall through to.
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
    """Build an *exc_cls* instance carrying *message*, robust to a finicky constructor: Hub error
    classes subclass ``HfHubHTTPError`` whose ``response`` arg is keyword-only (required on some
    versions), so ``exc_cls(message)`` can raise ``TypeError``. Try the normal constructors first, then
    BYPASS ``__init__`` via ``__new__`` so the TYPE and message survive. None only if ``__new__`` fails."""
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


def _is_builtin_oserror(exc: BaseException) -> bool:
    """True iff *exc*'s type is a BUILTIN ``OSError`` (or subclass): a genuine OS-level error whose
    ``[Errno N]`` is a real errno. Excludes HF/requests HTTP errors, which subclass ``OSError`` via
    ``requests -> IOError`` yet carry no OS errno, so a bracketed ``[Errno N]`` in their message is not
    mistaken for one."""
    if not isinstance(exc, OSError):
        return False
    builtin = getattr(builtins, type(exc).__name__, None)
    return isinstance(builtin, type) and issubclass(builtin, OSError) and isinstance(exc, builtin)


def _raise_child_error(message: str) -> None:
    """Re-raise a deterministic child error preserving its original TYPE when it is a known Hub / OS
    error, so callers catching ``RepositoryNotFoundError`` / ``GatedRepoError`` / ``OSError`` still
    match across the spawn boundary. The child reports ``"<ClassName>: <message>"``; an unrecognized or
    uninstantiable class falls back to ``RuntimeError``."""
    type_name = message.split(":", 1)[0].strip() if ":" in message else ""
    exc_cls = _resolve_exception_class(type_name)
    if exc_cls is None:
        raise RuntimeError(message)
    exc = _instantiate_preserving_type(exc_cls, message)
    if exc is None:
        raise RuntimeError(message)
    if _is_builtin_oserror(exc) and getattr(exc, "errno", None) is None:
        # Preserve errno (ENOSPC / EDQUOT ...) across the spawn boundary so a caller's `except OSError`
        # cleanup can still branch on exc.errno -- for EVERY builtin OSError subclass (PermissionError,
        # FileNotFoundError, ...), not just exact OSError. Restricted to BUILTIN OSError types: an HF
        # HTTP error (HfHubHTTPError / RepositoryNotFoundError ...) is ALSO an OSError subclass (via
        # requests -> IOError), and a bracketed "[Errno N]" in its message must not be mistaken for a
        # real OS errno. Set it as an attribute rather than via the (errno, strerror) constructor: a
        # subclass with a single-arg __init__ (hf_hub's LocalEntryNotFoundError) rejects the two-arg
        # form, and this keeps the message clean (no doubled "[Errno N]" prefix).
        errno_val = _parse_errno(message)
        if errno_val is not None:
            try:
                exc.errno = errno_val
            except Exception:
                pass
    raise exc


def _is_retryable_download_error(exc: BaseException) -> bool:
    """True when a captured download exception looks like a transient transport failure (an
    ``hf_xet`` / CAS error, connection reset, timeout, HTTP 5xx / 429) that the OTHER transport
    may recover, vs a deterministic Hub error (auth, not-found, gated, disk-full) that would
    fail identically. Unknown errors are treated as deterministic, so a real repeatable failure
    is surfaced rather than looped between transports."""
    name = type(exc).__name__
    # huggingface_hub raises LocalEntryNotFoundError BOTH for a genuine offline / uncached miss
    # (deterministic) AND as its wrapper around a TRANSIENT HEAD connection error / timeout for an
    # uncached file ("... Please check your connection and try again"). Retry the transient sub-case
    # over the other transport; a true offline miss (no transient hint) falls through to the
    # deterministic set below and keeps its reconstructed type.
    if name == "LocalEntryNotFoundError" and any(
        hint in f"{name}: {exc}".lower() for hint in _TRANSIENT_ERROR_HINTS
    ):
        return True
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
    """Spawn-child entrypoint (top-level + picklable): set the Xet env BEFORE importing
    huggingface_hub, form its own process group so the parent can kill the whole transfer, and never
    log the token or signed URLs."""
    # Die with the parent on Linux under Studio (best-effort; the module is absent standalone).
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

    # Test-only fault injection (never set in production): stall the Xet attempt so the watchdog +
    # HTTP fallback can be exercised against a real repo.
    if not disable_xet and os.environ.get("UNSLOTH_HF_XET_FORCE_STALL") == "1":
        _stall_fh = None
        try:
            from huggingface_hub.constants import HF_HUB_CACHE

            # Write the fake partial under the cache the watchdog scans, under the repo_type-correct
            # dir, so the stall / HTTP fallback fires in tests.
            cache_root = params.get("cache_dir") or HF_HUB_CACHE
            repo_dir_name = f"{repo_type or 'model'}s--" + repo_id.replace("/", "--")
            blobs = os.path.join(cache_root, repo_dir_name, "blobs")
            os.makedirs(blobs, exist_ok = True)
            # Hold the partial OPEN for the whole stall: the snapshot watchdog finds it by filename, but
            # the single-file watchdog counts only partials this PID holds open (a closed file is
            # ignored). The handle is bound to a local so it stays open across the sleep.
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
        # Classify here, where the exception object (status, errno, type) is intact, so the parent can
        # retry a transient failure over HTTP yet surface a deterministic error without a second attempt.
        result_queue.put({
            "ok": False,
            "error": _scrub_in_child(f"{type(e).__name__}: {e}", token),
            "retryable": _is_retryable_download_error(e),
        })


def _terminate_process_group(proc: "mp.process.BaseProcess", grace_period: float) -> None:
    """Kill *proc* and its whole process group (Xet may spawn helpers). The child ``os.setsid()``s so
    its pgid equals its pid; signal via ``os.killpg(pid, ...)`` -- NOT ``getpgid``, which before the
    child is a group leader resolves to OUR group. SIGTERM, then SIGKILL after *grace_period*."""
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
    # SIGKILL only while the leader is alive, so its pid (== pgid after setsid) is a live target. Once
    # join() reaps a leader that exited on SIGTERM, that pid is free and a busy host could recycle it
    # into an unrelated group -- killpg(pid) would then signal the WRONG group. hf_xet 1.5.x spawns no
    # helpers, so a reaped leader leaves nothing to clean up.
    if proc.is_alive():
        # Match _signal_group's own SIGKILL sentinel (-9) so the force-kill branch (proc.kill()) is
        # taken on Windows, where signal.SIGKILL is undefined. Functionally moot there (multiprocessing
        # maps proc.kill() == proc.terminate() == TerminateProcess, a hard kill either way), but keeps
        # the call site and the check consistent.
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
    stall_timeout: float,
    interval: float,
    grace_period: float,
    on_status: Optional[Callable[[str], None]],
) -> tuple[str, Optional[str]]:
    """Run one download in a spawn child supervised by the no-progress watchdog. Returns ``("ok",
    path)``, ``("stall", None)``, ``("cancelled", None)``, ``("crashed", message)`` (process crash, no
    captured exception), ``("retryable_error", message)`` (transient, worth an HTTP retry), or
    ``("error", message)`` (deterministic Hub error). The seam tests monkeypatch to avoid spawning."""
    # Single-file: capture the partials on disk BEFORE spawning so the watchdog ignores a sibling's
    # in-flight partial and follows only the blob(s) this child writes. Snapshots stay repo-wide.
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
    # Set the transport env in THIS process around the spawn so the child inherits it from creation:
    # HF reads HF_HUB_DISABLE_XET into a constant at import time, and the child re-imports
    # huggingface_hub before its body runs, so a child-side assignment would land too late. The child
    # still sets it defensively.
    child_env = {
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        # Tell unsloth_zoo's __init__ to skip its heavy torch / transformers / device init in the child.
        "UNSLOTH_ZOO_DISABLE_GPU_INIT": "1",
    }
    if disable_xet:
        child_env["HF_HUB_DISABLE_XET"] = "1"
        child_env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    with _SPAWN_ENV_LOCK:
        # Cache Hub's transport constants in the PARENT from the REAL env NOW, before the child-only
        # HF_HUB_DISABLE_XET=1 is briefly set below: a concurrent thread's FIRST `import huggingface_hub`
        # in the spawn window would otherwise cache the disabled-Xet value and route later in-process
        # downloads over HTTP. Once imported this is a no-op.
        try:
            import huggingface_hub.constants  # noqa: F401
        except Exception:
            pass
        saved_env = {k: os.environ.get(k) for k in child_env}
        # 'spawn' reconstructs __main__ from __main__.__file__. A pseudo-path ('<stdin>', a notebook)
        # fails to start; a real but UNGUARDED caller script gets re-imported as __mp_main__, re-running
        # the top-level from_pretrained and hitting the "start a process before bootstrapping" error ->
        # the parent sees the child exit without a result. We only need the child to run
        # _download_child_entry, so point __main__ at THIS side-effect-free module for the spawn.
        main_module = sys.modules.get("__main__")
        saved_main_file = _UNSET
        saved_main_spec = _UNSET
        if main_module is not None:
            saved_main_file = getattr(main_module, "__file__", _UNSET)
            main_module.__file__ = __file__
            # Launched as `python -m pkg`: spawn prefers __spec__.name and re-imports the module BY
            # NAME (re-running its top-level code). Clearing __spec__ forces the __file__ path branch.
            saved_main_spec = getattr(main_module, "__spec__", _UNSET)
            main_module.__spec__ = None
        try:
            os.environ.update(child_env)
            proc.start()
        except BaseException:
            # proc.start() can raise (OSError "can't start new process" under fd / thread exhaustion).
            # The result_queue's pipe fds were allocated above but the lifecycle try/finally that
            # closes them runs only after a successful start, so close the queue here to avoid an fd leak.
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
    # If start_watchdog raises ("can't start new thread"), the already-started child must STILL be
    # reaped, so it runs inside the try whose finally reaps it; stop_watchdog stays None until it works.
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
                # Prefer a result the child enqueued in the same window the watchdog fired in, so a
                # download that just succeeded is not killed. The Queue's feeder thread may not have
                # flushed a microseconds-earlier put, so use a short timeout, not get_nowait().
                try:
                    result = result_queue.get(timeout = 1.0)
                    break
                except queue.Empty:
                    pass
                # Capture the partials THIS child owns BEFORE killing it, so HTTP prep can scope its
                # purge to them. Prefer the per-pid open-fd set; fall back to post-baseline partials
                # when the child can't be inspected. None -> prep keeps its coarser mtime guard.
                owned = _child_open_incomplete_blobs(proc.pid) if proc.pid else None
                if owned is None and baseline_partials is not None:
                    current = set(
                        _active_incomplete_blob_sizes(repo_type, repo_id, params.get("cache_dir"))
                    )
                    owned = current - baseline_partials
                params["_owned_incomplete_blobs"] = owned
                _terminate_process_group(proc, grace_period)
                return ("stall", None)
            try:
                result = result_queue.get(timeout = _POLL_INTERVAL)
                break
            except queue.Empty:
                continue
        else:
            # Process exited; drain any result it enqueued. Short timeout, not get_nowait(): the child
            # can exit just before its feeder flushes the pipe, which would spuriously look resultless.
            try:
                result = result_queue.get(timeout = 1.0)
            except queue.Empty:
                result = None
    finally:
        if stop_watchdog is not None:
            stop_watchdog.set()
        proc.join(timeout = grace_period)
        # Any loop exit (completion, cancel/stall, KeyboardInterrupt) must not leak the child.
        # _terminate_process_group is idempotent, so a redundant call here is a harmless no-op.
        if proc.is_alive():
            _terminate_process_group(proc, grace_period)
        # Release the queue's pipe fds deterministically rather than waiting for GC. The result is
        # already extracted and a killed child has nothing to flush, so cancel the feeder before close.
        try:
            result_queue.cancel_join_thread()
            result_queue.close()
        except Exception:
            pass

    if result is None:
        # The child exited without a result: a process-level crash (a native hf_xet abort / segfault),
        # not a captured exception, so the other transport may still succeed -- report "crashed".
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
    """No interrupted-download evidence for the SELECTED files: no dangling requested symlink, and
    every EXACT-named requested file present. A dangling EXCLUDED weight does not reject a complete
    subset."""
    return (
        not snapshot_has_requested_broken_symlinks(
            snapshot_dir, allow_patterns = allow_patterns, ignore_patterns = ignore_patterns,
            repo_type = repo_type,
        )
        and requested_named_files_present(
            snapshot_dir, allow_patterns = allow_patterns, ignore_patterns = ignore_patterns
        )
    )


def _has_any_weight(snapshot_dir: Path, *, ignore_patterns: Any = None) -> bool:
    """True if the snapshot holds at least one loadable weight anywhere (root or subfolder) that the
    request's ignore filter keeps. Lenient: it only tells a real model warm from a config-only stale
    snapshot, without classifying layout. The ignore filter matters for diffusers, whose component
    weights live in subfolders -- a partial holding only the ignored format (``unet/*.bin`` under
    ``ignore=['*.bin']``) is not a usable weight for a safetensors load."""
    rels: list = []
    try:
        for entry in snapshot_dir.rglob("*"):
            if not _is_loadable_weight_file(entry.name):
                continue
            try:
                if entry.is_file():
                    rels.append(entry.relative_to(snapshot_dir).as_posix())
            except (OSError, ValueError):
                continue
    except OSError:
        return False
    return bool(_filter_paths(rels, None, ignore_patterns))


def _is_default_load_weight_file(name: str) -> bool:
    """A weight in a format a DEFAULT ``from_pretrained`` reads: safetensors or bin only. Excludes gguf /
    pt / pth / onnx / msgpack / ... -- a default (non-format-specific) transformers / diffusers load does
    not read those, so a stale cache holding only e.g. ``model.Q4_K_M.gguf`` does not satisfy the load,
    which would then fetch the missing ``model.safetensors`` / ``pytorch_model.bin`` over un-killable Xet.
    Trainer / optimizer state (``optimizer.bin``, ...) is excluded by ``_is_loadable_weight_file``."""
    return _is_loadable_weight_file(name) and name.endswith((".safetensors", ".bin"))


# The CANONICAL root model weight a DEFAULT (no-variant) load reads: model.safetensors /
# pytorch_model.bin as a single file, or a numbered shard (model-00001-of-00002.safetensors -- a dash,
# not a dotted variant token). A PEFT adapter (adapter_model.*), a variant (model.fp16.safetensors), a
# gguf, and a non-canonical root weight (consolidated.safetensors, tf_model.h5) are NOT matched: a
# default from_pretrained probes only these canonical names, so a cache holding only something else does
# not satisfy the load, which would then fetch the missing canonical weight over un-killable Xet.
_CANONICAL_ROOT_MODEL_WEIGHT_RE = re.compile(
    r"^(?:model|pytorch_model)(?:-\d{5}-of-\d{5})?\.(?:safetensors|bin)$"
)

# A training-checkpoint subdir (checkpoint-500/, checkpoint_7/): its weights are never read as diffusers
# pipeline COMPONENTS, so they must not mask missing unet/vae/text-encoder weights.
_CHECKPOINT_DIR_RE = re.compile(r"^checkpoint[-_]\d+$")


def _has_diffusers_component_weight(snapshot_dir: Path, *, ignore_patterns: Any = None) -> bool:
    """True if a DECLARED diffusers pipeline COMPONENT weight (a loadable weight in a component SUBFOLDER
    the ``model_index.json`` declares: unet/, vae/, text_encoder/, ...) that the ignore filter keeps is
    present. Scoped to declared components, so a stale partial holding only an UNDECLARED leftover subtree
    (a controlnet/ dir not in ``model_index.json``) does not read as proof the pipeline is warm while the
    declared unet / vae weights are still missing -- which the in-process load would then fetch over
    un-killable Xet. Also excludes ROOT-level weights (an adapter / merged file a ``DiffusionPipeline``
    does not read as a component) and training-checkpoint subtrees (checkpoint-N/). A malformed / empty
    ``model_index.json`` fails OPEN (any component subfolder counts). Stays lenient on WHICH declared
    components are required (a pipeline's components can be optional): it only tells a real component warm
    from an undeclared-leftover / checkpoint-only / config-only stale snapshot."""
    declared = _diffusers_declared_components(snapshot_dir)
    rels: list = []
    try:
        for entry in snapshot_dir.rglob("*"):
            if not _is_default_load_weight_file(entry.name):
                continue
            try:
                if not entry.is_file():
                    continue
                rel = entry.relative_to(snapshot_dir).as_posix()
            except (OSError, ValueError):
                continue
            parts = rel.split("/")
            if len(parts) < 2:
                continue  # a ROOT-level weight is not a pipeline component
            if declared is not None and parts[0] not in declared:
                continue  # an UNDECLARED subtree the DiffusionPipeline load does not read
            if any(_CHECKPOINT_DIR_RE.match(p) for p in parts[:-1]):
                continue  # under a training-checkpoint subtree, not a component
            rels.append(rel)
    except OSError:
        return False
    return bool(_filter_paths(rels, None, ignore_patterns))


def _root_model_has_weight(snapshot_dir: Path, *, ignore_patterns: Any = None) -> bool:
    """Whether an UNPATTERNED model warm holds a weight a default load reads: a CANONICAL ROOT weight
    (``model.safetensors`` / ``pytorch_model.bin``, single or numbered shard), or -- for a diffusers
    pipeline (root ``model_index.json``) -- a component-subfolder weight. Counting any subtree weight (as
    ``_has_any_weight`` does) would accept a stale checkpoint-only snapshot and then fetch the root
    weights over un-killable Xet; diffusers is the one layout whose weights live in subfolders. Only the
    canonical names are counted (``_CANONICAL_ROOT_MODEL_WEIGHT_RE``): a VARIANT-named root weight
    (``model.fp16.safetensors``), a PEFT adapter (``adapter_model.*``), a gguf, and a NON-canonical root
    weight (``consolidated.safetensors``) are excluded, since a default from_pretrained probes only the
    canonical names, so a cache holding only something else is retried over HTTP rather than loaded (its
    canonical weight is still missing). The request's ignore filter is applied to the ROOT weights, so an
    offline-fallback partial holding only the format the load will NOT read (an ignored ``*.bin`` under a
    safetensors request) does not count as a usable weight."""
    try:
        is_diffusers = (snapshot_dir / "model_index.json").is_file()
    except OSError:
        is_diffusers = False
    if is_diffusers:
        return _has_diffusers_component_weight(snapshot_dir, ignore_patterns = ignore_patterns)
    rels: list = []
    try:
        for entry in snapshot_dir.iterdir():
            name = entry.name
            if not _CANONICAL_ROOT_MODEL_WEIGHT_RE.match(name):
                continue  # only a canonical model.safetensors / pytorch_model.bin (single or shard) is
                # read by a default load -- an adapter, variant, gguf, or consolidated.* is not
            try:
                if entry.is_file():
                    rels.append(name)
            except OSError:
                continue
    except OSError:
        return False
    return bool(_filter_paths(rels, None, ignore_patterns))


def _root_has_variant_weight(
    snapshot_dir: Path, variant: str, *, ignore_patterns: Any = None
) -> bool:
    """True if a CANONICAL ROOT model weight carrying the requested *variant* token, kept by the ignore
    filter, is present. transformers writes the variant on the model base then shards it, so the names it
    reads are ``model.<variant>.safetensors`` (single) and ``model.<variant>-00001-of-00002.safetensors``
    (a ``.<variant>-`` shard infix) -- matched by ``_ROOT_MODEL_VARIANT_WEIGHT_RE`` plus the specific
    variant infix. A non-canonical base (``consolidated.<variant>.safetensors``), a PEFT adapter, or a
    non-``model`` variant name a default variant load never reads is excluded, so a cache holding only
    those is retried over HTTP rather than loaded (its ``model.<variant>.*`` weight is still missing). The
    ignore filter is applied so a partial holding only the ignored format (``model.fp16.bin`` under
    ``ignore=['*.bin']``) does not count."""
    infix_dot = f".{variant}."
    infix_dash = f".{variant}-"
    rels: list = []
    try:
        for entry in snapshot_dir.iterdir():
            name = entry.name
            if infix_dot not in name and infix_dash not in name:
                continue  # not the requested variant token
            if not _ROOT_MODEL_VARIANT_WEIGHT_RE.match(name):
                continue  # only a canonical model / pytorch_model variant weight is read by a default
                # variant load -- an adapter, a consolidated.* sidecar, or a gguf is not
            try:
                if entry.is_file():
                    rels.append(name)
            except OSError:
                continue
    except OSError:
        return False
    return bool(_filter_paths(rels, None, ignore_patterns))


def _has_diffusers_component_variant_weight(
    snapshot_dir: Path, variant: str, *, ignore_patterns: Any = None
) -> bool:
    """Variant analog of ``_has_diffusers_component_weight``: True if a DECLARED diffusers pipeline
    COMPONENT subfolder (unet/, vae/, text_encoder/, ... that ``model_index.json`` declares) holds a
    weight carrying the requested *variant* token (``unet/diffusion_pytorch_model.fp16.safetensors``). A
    variant pipeline warm's weights are component-scoped, not root ``model.<variant>.*`` files, so a
    root-only variant check would false-reject a complete diffusers variant download into a
    ``DownloadStallError``. Scoped to declared components (as the plain component helper is), so a stale
    partial holding only an UNDECLARED leftover variant weight (a ``controlnet/`` dir not in
    ``model_index.json``) does not read as proof the pipeline is warm while the declared unet / vae
    variant weights are still missing -- which ``DiffusionPipeline.from_pretrained(..., variant=...)``
    would then fetch over un-killable Xet. A malformed / empty ``model_index.json`` fails OPEN. Excludes
    ROOT-level and training-checkpoint weights (as the plain component check does) and reads only
    safetensors / bin."""
    declared = _diffusers_declared_components(snapshot_dir)
    infix_dot = f".{variant}."
    infix_dash = f".{variant}-"
    rels: list = []
    try:
        for entry in snapshot_dir.rglob("*"):
            name = entry.name
            if not _is_default_load_weight_file(name):
                continue
            if infix_dot not in name and infix_dash not in name:
                continue
            try:
                if not entry.is_file():
                    continue
                rel = entry.relative_to(snapshot_dir).as_posix()
            except (OSError, ValueError):
                continue
            parts = rel.split("/")
            if len(parts) < 2:
                continue  # a ROOT-level variant weight is not a pipeline component
            if declared is not None and parts[0] not in declared:
                continue  # an UNDECLARED subtree the DiffusionPipeline load does not read
            if any(_CHECKPOINT_DIR_RE.match(p) for p in parts[:-1]):
                continue  # under a training-checkpoint subtree, not a component
            rels.append(rel)
    except OSError:
        return False
    return bool(_filter_paths(rels, None, ignore_patterns))


def _root_model_has_variant_weight(
    snapshot_dir: Path, variant: str, *, ignore_patterns: Any = None
) -> bool:
    """Whether an UNPATTERNED variant warm holds a variant weight a default load reads: a ROOT variant
    weight, or -- for a diffusers pipeline (root ``model_index.json``) -- a component-subfolder variant
    weight. Variant analog of ``_root_model_has_weight``: a diffusers variant's weights live in component
    subfolders, not root ``model.<variant>.*`` files, so the root-only check would false-reject them."""
    try:
        is_diffusers = (snapshot_dir / "model_index.json").is_file()
    except OSError:
        is_diffusers = False
    if is_diffusers:
        return _has_diffusers_component_variant_weight(
            snapshot_dir, variant, ignore_patterns = ignore_patterns
        )
    return _root_has_variant_weight(snapshot_dir, variant, ignore_patterns = ignore_patterns)


# Interchangeable exact weight names collapse to one equivalence group: the either-format pair
# ``["pytorch_model.bin", "model.safetensors"]`` is satisfied by ANY one -- and so is the variant pair
# ``["model.fp16.safetensors", "pytorch_model.fp16.bin"]`` (HF allow patterns are ALTERNATIVES over the
# repo, so a repo publishing only one format is complete). Distinct logical weights (base AND adapter, a
# different variant token) stay separate groups (each required).
_EITHER_FORMAT_WEIGHT_RE = re.compile(
    r"^(model|pytorch_model|adapter_model)(?:\.([^.]+))?\.(?:safetensors|bin)$"
)


def _exact_weight_logical(base: str) -> Any:
    """Equivalence key for an EXACT-named weight so the either-format alternatives share a group:
    ``model.safetensors`` / ``pytorch_model.bin`` -> ``("root_model", None)``; the same variant token in
    both formats shares ``("root_model", "<variant>")``; ``adapter_model.*`` -> ``("adapter_model", ...)``.
    A non-weight (or sharded) name maps to itself, so each distinct file is still required."""
    m = _EITHER_FORMAT_WEIGHT_RE.match(base)
    if m is None:
        return base
    stem, variant = m.group(1), m.group(2)
    logical = "adapter_model" if stem == "adapter_model" else "root_model"
    return (logical, variant)


def _requested_exact_files_present_grouped(
    snapshot_dir: Path, *, allow_patterns: Any, ignore_patterns: Any,
) -> bool:
    """True unless an EXACT-named requested file is missing. Interchangeable weights
    (``["pytorch_model.bin", "model.safetensors"]``) need any one; distinct logical files (base AND
    adapter, a tokenizer file) each. A glob / unpatterned request is trivially satisfied here."""
    allow = _as_pattern_list(allow_patterns)
    ignore = _as_pattern_list(ignore_patterns)
    if not allow or any(not isinstance(p, str) or _has_glob(p) for p in allow):
        return True
    requested = _filter_paths(allow, None, ignore)
    if not requested:
        return True  # the ignore filter dropped every named file -> nothing to require
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
    """True if a loadable weight the request SELECTS is present. Applies the allow / ignore filter (vs
    ``_has_any_weight``), so a patterned request is not satisfied by an out-of-scope weight (a stale
    ``.bin``, an unrequested checkpoint subfolder)."""
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
    """True if a SELECTED loadable weight carrying the *variant* token is present. Combines the request's
    allow / ignore scope (as ``_has_selected_weight``) with the variant infix check (as
    ``_root_has_variant_weight``): a patterned variant load (e.g. ``subfolder=`` + ``variant=``) whose
    offline-fallback partial kept only the canonical weight in scope is retried over HTTP rather than
    loaded, else the in-process load fetches ``model.<variant>.safetensors`` over un-killable Xet."""
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
    """True only for a non-empty allow list of EXACT filenames (no ``None`` / glob / trailing-slash
    dir). Only such a request is locally provable complete; ``None`` / a glob needs the Hub manifest."""
    patterns = _as_pattern_list(patterns)
    if patterns is None:
        return False
    if not patterns:
        return True  # selects nothing -> trivially satisfied, nothing to fetch
    return all(isinstance(p, str) and not _has_glob(p) for p in patterns)


def _request_selects_canonical_root_shards(allow_patterns: Any, ignore_patterns: Any) -> bool:
    """Whether the request's allow / ignore filter keeps a canonical ROOT shard name. When False, an
    incomplete canonical root shard set is OUT of the request's scope -- a co-resident leftover from a
    prior interrupted base pull that a patterned load (adapter / gguf / subfolder) never reads -- so the
    canonical-shard-completeness gate must NOT reject on it, else a genuinely complete patterned download
    is failed into a DownloadStallError."""
    probes = ["model-00001-of-00002.safetensors", "pytorch_model-00001-of-00002.bin"]
    return bool(_filter_paths(probes, allow_patterns, ignore_patterns))


def _request_selects_root_variant_weight(
    allow_patterns: Any, ignore_patterns: Any, variant: str,
) -> bool:
    """Whether the request's allow / ignore filter keeps a ROOT variant weight name. When False, a stale
    incomplete root variant shard set is OUT of the request's scope (e.g. a subfolder request
    ``allow=['unet/*']`` whose variant weights live under ``unet/``), so the ROOT variant-shard gate must
    not reject on it, else a complete in-scope variant download is failed."""
    probes = [
        f"model.{variant}.safetensors", f"model.{variant}-00001-of-00002.safetensors",
        f"pytorch_model.{variant}.bin", f"pytorch_model.{variant}-00001-of-00002.bin",
    ]
    return bool(_filter_paths(probes, allow_patterns, ignore_patterns))


def _cache_can_skip_download(
    snapshot_dir: Path, *, repo_type: str, allow_patterns: Any, ignore_patterns: Any,
    variant: Optional[str] = None,
) -> bool:
    """PRE-download: whether a cached snapshot is complete enough to skip the protective child.

    STRICT for a weight-bearing model request: only the conservative canonical gate
    (``snapshot_dir_is_complete``) skips; anything uncertain (diffusers, variants, patterns,
    sharded-without-index) spawns the child. A false True would let the load fetch a missing weight over
    un-killable Xet (the hang). A weightless model or non-model (dataset) request has no weight to hang
    on, but is locally provable complete only when it names EXACT files -- an unpatterned / glob request
    defers to the child rather than hand back a partial cache. An intact exact-named subset still
    short-circuits (offline tokenizer-only / named-file warm)."""
    if repo_type in (None, "model") and request_can_include_weights(allow_patterns, ignore_patterns):
        # A variant load reads variant-named weights (model.<variant>.safetensors) that the canonical
        # gate does not check: a cache holding only the canonical weight reads as complete, so the
        # in-process load would fetch the variant over un-killable Xet. Defer to the child (it warms
        # the variant too).
        if variant:
            return False
        # STRICT: a default load probes model.safetensors before pytorch_model.bin, so a bin-only cache
        # for a repo that also publishes safetensors (which the local cache cannot rule out) would fetch
        # the preferred safetensors in-process over Xet. prefer_safetensors defers such a cache to the
        # child; a use_safetensors=False request (safetensors ignored) still fast-paths its bin cache.
        return snapshot_dir_is_complete(
            snapshot_dir, allow_patterns = allow_patterns, ignore_patterns = ignore_patterns,
            prefer_safetensors = True,
        )
    # Weightless / non-model: skip only for an intact exact-named subset. A None / glob request cannot
    # be proven complete from local files, so defer to the child for the manifest compare + resume.
    if not _patterns_are_exact_names(allow_patterns):
        return False
    return _intact_subset(
        snapshot_dir, repo_type = repo_type, allow_patterns = allow_patterns,
        ignore_patterns = ignore_patterns,
    )


def _has_readable_weight(
    snapshot_dir: Path, *, allow_patterns: Any, ignore_patterns: Any, variant: Optional[str],
) -> bool:
    """Invariant A (presence): a weight the in-process load will READ is present on disk, with the
    request's ignore filter ALWAYS applied and the scope matched to the request:

    - variant + UNPATTERNED -> a ROOT variant weight (``model.<variant>.*``);
    - variant + PATTERNED   -> a SELECTED variant weight (within the allow scope);
    - plain  + UNPATTERNED  -> a ROOT (or diffusers-component) weight, NOT a stray subfolder checkpoint;
    - plain  + PATTERNED    -> a SELECTED weight (within the allow scope).

    A partial that kept only the ignored format (an ``*.bin`` under ``ignore=['*.bin']``) does not count,
    so the incomplete result is retried over HTTP rather than loaded in-process."""
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
    """Invariant B (shard completeness): an IN-SCOPE shard set the load reads is incomplete (an index
    present with a shard missing, or a lone numbered shard without its index) and must be retried. The
    check is ALWAYS scoped to what the request selects, so a co-resident stale shard set the load never
    reads (a leftover root checkpoint under a subfolder/adapter/gguf request) does not false-reject a
    complete download:

    - variant: the ROOT variant-shard check applies (for a NON-diffusers snapshot) for an UNPATTERNED
      request, or a PATTERNED request that selects a ROOT variant weight (a globbed ``['*.safetensors']``);
      a subfolder-scoped variant request does not root-check.
    - plain: the canonical-root-shard check applies (for a NON-diffusers snapshot) for an UNPATTERNED
      request, or a GLOBBED request that selects canonical root shards; an exact-named subset or an
      out-of-scope request does not.
    - non-root: a PATTERNED request additionally checks any SELECTED shard index the root-model checks do
      not cover (a sharded adapter under ``['adapter_model*']``, a component subfolder) via
      ``_selected_shard_index_incomplete``; an exact-named subset defers to the exact-file presence check.
    - diffusers: a pipeline (root ``model_index.json``) reads COMPONENT subfolders (unet/, vae/, ...), NOT
      root model shards, so the root-model checks above are SKIPPED for it (a stale root index must not
      reject a complete pipeline); an UNPATTERNED warm's component shard sets are checked via
      ``_diffusers_component_shards_incomplete``, and a PATTERNED one via ``_selected_shard_index_incomplete``.

    The ignore filter is threaded through so completeness is judged for the FORMAT the load reads (a
    complete safetensors set does not mask an incomplete ``.bin`` under ``ignore=['*.safetensors']``)."""
    try:
        is_diffusers = (snapshot_dir / "model_index.json").is_file()
    except OSError:
        is_diffusers = False
    if variant:
        if not is_diffusers and (
            allow_patterns is None
            or _request_selects_root_variant_weight(allow_patterns, ignore_patterns, variant)
        ):
            # A diffusers pipeline reads component-subfolder variant weights, not root model.<variant>
            # shards, so a stale root variant index must not reject a complete pipeline (handled below by
            # the component check); only a non-diffusers root variant load runs the root-shard check.
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
            # an UNPATTERNED variant diffusers warm: a component subfolder's variant shard index is
            # incomplete (the root variant check above only covers root model.<variant> shards).
            return True
        return False
    if allow_patterns is None:
        if not is_diffusers and _has_incomplete_canonical_root_shards(
            snapshot_dir, ignore_patterns = ignore_patterns
        ):
            # a non-diffusers root model load; a diffusers pipeline reads component subfolders, not root
            # model shards, so a stale root index there is handled by the component check below.
            return True
        if is_diffusers and _diffusers_component_shards_incomplete(
            snapshot_dir, variant = None, ignore_patterns = ignore_patterns
        ):
            # an UNPATTERNED plain diffusers warm reads component subfolders (unet/, vae/, ...); a
            # component shard index missing a shard is not covered by the canonical ROOT-shard check.
            return True
        return False
    if _patterns_are_exact_names(allow_patterns):
        return False  # an exact-named subset defers to the exact-file presence check
    if not is_diffusers and _request_selects_canonical_root_shards(
        allow_patterns, ignore_patterns
    ) and _has_incomplete_canonical_root_shards(snapshot_dir, ignore_patterns = ignore_patterns):
        # non-diffusers only: a diffusers pipeline never reads root model shards (its component sets are
        # checked via _selected_shard_index_incomplete below), so a stale root index must not reject it.
        return True
    return _selected_shard_index_incomplete(
        snapshot_dir, allow_patterns = allow_patterns,
        ignore_patterns = ignore_patterns, variant = None,
    )


def _selected_readable_weight_complete(
    snapshot_dir: Path, *, allow_patterns: Any, ignore_patterns: Any, variant: Optional[str],
) -> bool:
    """Single entry point for the weight-bearing MODEL acceptance check: the weight the in-process load
    will READ is present (Invariant A) AND its in-scope shard set is complete (Invariant B). Both
    invariants apply the request's ignore filter and match its scope uniformly, so a co-resident
    out-of-scope / ignored-format partial neither masks an incomplete readable weight (a silent Xet hang)
    nor false-rejects a complete download (a spurious ``DownloadStallError``)."""
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
    """POST-download: whether the child's result is usable, or should be retried over HTTP.
    snapshot_download already did the authoritative manifest compare, so accept unless there is
    POSITIVE breakage evidence; LENIENT otherwise (a finished diffusers / either-format download passes,
    an optional missing file is not treated as broken) so a good download is never looped into a
    ``DownloadStallError``. A transient connection error during the child's metadata call makes
    ``snapshot_download`` silently return an existing (stale / partial) cache instead of fetching, so
    the checks below apply the request's filters to the weight the load will actually read. Breakage:

    - A dangling REQUESTED symlink (a missing / still-``.incomplete`` blob).
    - A missing EXACT-named requested file (grouped by weight equivalence: the either-format pair needs
      one; base AND adapter, or a ``["tokenizer.json"]`` request, each). Globs stay lenient.
    - A weight-bearing MODEL request whose READABLE weight is absent or incomplete. Delegated to
      ``_selected_readable_weight_complete``, which applies the request's ignore filter and scope
      uniformly: the weight the load reads (variant vs canonical, root vs in-scope) must be present, and
      its in-scope shard set complete. A co-resident out-of-scope / ignored-format partial neither masks
      an incomplete readable weight nor false-rejects a complete download."""
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
    return True


def _snapshot_payload_incomplete(
    payload: Any, *, repo_type: str, allow_patterns: Any, ignore_patterns: Any,
    variant: Optional[str] = None,
) -> bool:
    """True when a snapshot download returned a real directory not usable for the request (see
    ``_download_result_usable``). Guarded to an existing dir, so a mocked / non-path payload (tests) is
    trusted rather than rejected; in production the child always returns a real snapshot dir."""
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
    variant: Optional[str] = None,
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
            # Purge a non-HTTP partial first: an HTTP resume over a sparse Xet/hf_transfer partial
            # silently corrupts the blob. Scope the purge to the partials the stalled child owned, so
            # a concurrent same-repo sibling's partial is spared. An injected (Studio) hook owns its
            # own cache accounting, so it keeps the plain (repo_type, repo_id) signature.
            owned_incomplete = params.pop("_owned_incomplete_blobs", None)
            try:
                if prepare_for_http_fn is None:
                    _default_prepare_for_http(
                        repo_type, repo_id, cache_dir = cache_dir, active_grace = stall_timeout,
                        owned_incomplete_blobs = owned_incomplete,
                    )
                else:
                    prepare_for_http_fn(repo_type, repo_id)
            except Exception as e:
                logger.debug("prepare_for_http failed for %s: %s", repo_id, e)
            # An unsafe partial that could not be cleared (locked file, permission error) would
            # corrupt the blob on an HTTP resume: force a clean re-download instead.
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
                variant = variant,
            ):
                # HF can hand back an existing incomplete snapshot dir (offline / timed-out request)
                # instead of fetching the missing files. Never load that in-process: retry over HTTP,
                # then fail loudly rather than load a broken cache. (Patterned / non-model requests are
                # judged by their own subset, so a valid weightless snapshot is not rejected.)
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
            # Deterministic failure (auth / not-found / gated / disk-full): the other transport fails
            # identically, so do not retry. _raise_child_error preserves the original exception type
            # across the spawn boundary so callers' typed except clauses still match.
            _raise_child_error(payload)
        if kind_result == "retryable_error":
            # Transient transport failure (hf_xet CAS timeout, 5xx, reset): HTTP may recover, so retry
            # once before surfacing it; if HTTP also failed there is no transport left -> raise.
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
            # Process-level crash with no captured exception: HTTP may still succeed, so retry once.
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
            # _safe_status: a raising status hook (disconnected client) must not abort the retry
            # before disable_xet is set, turning a recoverable stall into a failed download.
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

    Returns the local cache path. Raises ``RuntimeError("Cancelled")`` if *cancel_event* is set,
    re-raises a deterministic child error unchanged (no fallback), and raises ``DownloadStallError``
    only if BOTH transports stall. ``force_download=True`` re-fetches even if cached;
    ``local_files_only=True`` resolves from cache in-process with no child (HF offline semantics);
    ``subfolder`` is forwarded to ``hf_hub_download``.
    """
    repo_type = repo_type or "model"  # HF treats None as the default model repo.
    # Expand ~ (and normalize Path) as huggingface_hub does, so the probe and the child resolve to
    # the same on-disk location (else a warm cache is missed and we spawn a child for a cached file).
    if isinstance(cache_dir, (str, os.PathLike)):
        cache_dir = os.path.expanduser(os.fspath(cache_dir))
    # Honor an already-set cancellation before any probe: the short-circuits below return without
    # reaching _download_with_xet_fallback (which holds the only other cancel check).
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Cancelled")
    # Offline: resolve purely from cache. HF raises LocalEntryNotFoundError if uncached; let it propagate.
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
    # Finalized blob already cached: return it with no child and no network (skipped under
    # force_download). The cache stores a subfolder file under "<subfolder>/<filename>".
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
    stall_timeout: float = DEFAULT_STALL_TIMEOUT,
    interval: float = DEFAULT_HEARTBEAT_INTERVAL,
    grace_period: float = DEFAULT_GRACE_PERIOD,
    on_status: Optional[Callable[[str], None]] = None,
    prepare_for_http_fn: Optional[Callable[[str, str], None]] = None,
) -> str:
    """Download a whole repo snapshot with Xet primary and HTTP as a stall-only fallback, returning
    the local snapshot dir.

    Used by Unsloth's ``from_pretrained`` to warm the cache in a killable child BEFORE the in-process
    model load (which then hits a warm cache and cannot hang on a native Xet thread). A fully cached
    repo short-circuits in-process via ``local_files_only`` with no child. ``force_download=True``
    re-fetches in the killable child even if cached; ``local_files_only=True`` resolves from cache
    in-process with no child (HF offline semantics). ``variant`` (e.g. "fp16") forces the child even
    on a warm canonical cache, since the canonical gate cannot prove the variant-named weights present.
    """
    repo_type = repo_type or "model"  # HF treats None as the default model repo.
    # Expand ~ as huggingface_hub does, so the probe and the child resolve to the same cache location.
    if isinstance(cache_dir, (str, os.PathLike)):
        cache_dir = os.path.expanduser(os.fspath(cache_dir))
    # Honor an already-set cancellation before any probe: the short-circuits below return without
    # reaching _download_with_xet_fallback (which holds the only other cancel check).
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Cancelled")
    # Offline: resolve purely from cache. HF raises if uncached; let it propagate.
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
    # Fast path: everything already on disk -> resolve in-process (no Xet, no hang). Skipped under
    # force_download.
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
            # local_files_only returns a snapshot dir whenever refs/<rev> + snapshots/<sha> exist,
            # even one left by a prior interrupted or patterned download (config-only, partial shards).
            # Validate the EXACT returned revision dir: a full model warmup skips the child only when
            # its canonical weights are provably complete; a patterned / non-model request only needs
            # its referenced files. Scope to this snapshot, NOT the whole repo, so an unrelated
            # revision mid-download elsewhere in the repo cache does not force a needless re-fetch.
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
