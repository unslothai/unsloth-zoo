# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# This file is licensed under the GNU Affero General Public License v3.0 only
# (AGPL-3.0-only), unlike the rest of unsloth_zoo which is LGPL-3.0-or-later. It
# is the single shared home for the Xet -> HTTP stall fallback used by both
# Unsloth (FastModel.from_pretrained) and Unsloth Studio, which imports it.
# See <https://www.gnu.org/licenses/agpl-3.0.html>.

"""Xet-primary HF downloads with an automatic HTTP fallback on a no-progress stall.

Xet (``hf_xet``) is fast but can hang with no progress, no exception, and an un-killable native thread.
``HF_HUB_DISABLE_XET`` is read at import time, so the fallback runs in a fresh ``spawn`` child (not a
thread) that sets the env before importing ``huggingface_hub``. Cached files short-circuit with no
child; deterministic errors (401/403/404/disk-full) and cancellation propagate without a fallback.
``snapshot_download_with_xet_fallback`` warms a whole repo in a killable child before Unsloth's
in-process load; ``hf_hub_download_with_xet_fallback`` does a single file. Studio cache / secret /
process helpers are used best-effort (imported only if present) or injected. The child sets
``UNSLOTH_ZOO_DISABLE_GPU_INIT=1`` for unsloth_zoo's lightweight import path (no torch / transformers).
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
    _diffusers_declared_component_specs,
    _filter_paths,
    _has_glob,
    _has_incomplete_canonical_root_shards,
    _has_incomplete_variant_root_shards,
    _is_loadable_weight_file,
    _selected_shard_index_incomplete,
    _weight_shard_index_complete,
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

# Explicit list keeps stdlib imports out of Studio's `import *` re-export shim.
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

# Serializes the parent-env (and __main__.__file__) mutation around a child spawn so
# concurrent downloads cannot observe each other's transport env.
_SPAWN_ENV_LOCK = threading.Lock()

# Sentinel: "__main__.__file__ untouched for this spawn" (distinct from a saved None).
_UNSET = object()

# HF boolean env convention, case-insensitive.
_TRUTHY = {"1", "true", "yes", "on"}


def _is_true(value: Optional[str]) -> bool:
    return value is not None and str(value).strip().lower() in _TRUTHY


def _safe_status(callback: Optional[Callable[[str], None]], message: str) -> None:
    """Invoke a status callback; swallow its exceptions so a disconnected client cannot kill the
    daemon watchdog thread (stopping stall detection)."""
    if callback is None:
        return
    try:
        callback(message)
    except Exception as e:
        logger.debug("watchdog status callback raised (ignored): %s", e)


class DownloadStallError(RuntimeError):
    """Raised when no download progress is observed for too long. Studio re-imports this canonical type."""


def is_hf_xet_available() -> bool:
    """True iff the ``hf_xet`` extra is importable (Hub uses it automatically)."""
    try:
        return importlib.util.find_spec("hf_xet") is not None
    except Exception:
        return False


def xet_force_disabled() -> bool:
    """Whether the user asked to force HTTP up front via ``UNSLOTH_DISABLE_XET`` /
    ``UNSLOTH_STABLE_DOWNLOADS`` / ``HF_HUB_DISABLE_XET``."""
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
    # Redact the query of a presigned S3/CAS blob URL (temporary creds in the query string); keep
    # non-signed URLs (e.g. ...?download=true).
    def _redact_signed_query(match: "re.Match") -> str:
        base, query = match.group(1), match.group(2)
        if re.search(
            r"(X-Amz-|[Ss]ignature|(?:^|&)(?:sig|token|key|Expires|Policy|Key-Pair-Id)=)",
            query,
        ):
            return f"{base}?***"
        return match.group(0)

    # Stop the query at whitespace OR a structural delimiter (quote/bracket/brace/paren/angle/pipe): a
    # URL embedded in JSON / a dict repr has no trailing whitespace, so a greedy [^\s]* would swallow the
    # closing "} and corrupt the log line. Signed-query values percent-encode these chars, so a genuine
    # presigned URL is still fully redacted.
    out = re.sub(
        r"(https?://[^\s?]+)\?([^\s\"'()<>{}|[\]]*)", _redact_signed_query, out
    )
    return out


def _broken_link_has_active_partner(link: Path, *, active_grace: float) -> bool:
    """SPARE a dangling snapshot symlink iff a sibling is still writing its target blob. Discriminator
    is a FRESH ``.incomplete`` partner of the target, NOT the link mtime: our killed child's partner was
    static for the full stall timeout and is purged first (no partner -> link cleared), while a sibling
    mid-download still has a growing partner (link spared)."""
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
    """The ``<hash>.incomplete`` basename for a dangling symlink's target blob, or None."""
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
    ``iter_active_repo_cache_dirs`` is case-collision safe, so this destructive purge only touches an
    unambiguous repo cache dir. Studio injects its marker-aware version instead.

    *owned_incomplete_blobs* (basenames the stalled child held open, captured before the kill) SCOPES the
    purge so a same-repo sibling writing a DIFFERENT blob is spared even if aged past *active_grace*;
    None -> coarser mtime guard only.
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
                            # Spare a partial touched within active_grace (a slow sibling, not stalled);
                            # our killed partial has been static for the full stall timeout so it purges.
                            if time.time() - blob.stat().st_mtime < active_grace:
                                continue
                            blob.unlink()
                        except OSError:
                            continue  # a locked / denied blob must not abort the rest
            # Clear broken snapshot symlinks (also read as active incomplete state). Sweep EVERY snapshot,
            # else a dangling link under an older revision keeps the repo incomplete and re-trips.
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
                            # Spare a sibling's active link (target still has a fresh .incomplete).
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
    """Map ``{blob_filename: bytes_present}`` (sparse-aware) for the repo's ``*.incomplete`` partials, so
    the single-file watchdog follows only its own child's partials and a sibling download of a different
    file cannot mask this file's stall."""
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
    """Basenames of the ``*.incomplete`` blobs child *pid* has open -- exactly the partial THIS child is
    writing (incl. a resumed partial reusing a prior blob-hash name), not a sibling's (different pid).
    ``None`` when undeterminable (no ``psutil`` / ``/proc``, or process gone) -> caller uses a coarser
    measure; empty set -> child not yet writing (connect / metadata phase)."""
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
    """Return ``(total_on_disk_bytes, has_incomplete)`` for the HF cache being written (sparse-aware, so
    a partial Xet / ``hf_transfer`` blob is not read as full progress). Scans *cache_dir* or the active
    ``HF_HUB_CACHE``. Missing / empty cache -> ``(0, False)``; ``None`` only on a probe exception
    (unmeasurable -> callers skip stall logic this tick)."""
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
    """Start a daemon thread firing ``on_stall(message)`` once iff a ``*.incomplete`` is present AND the
    on-disk size is unchanged for *stall_timeout* seconds. The timer resets while no ``*.incomplete``
    exists, so post-download init is not misread as a stall. Returns a stop event the caller sets when
    the download phase ends.

    *watch_new_partials_only* (single-file) measures progress only over the child's own partial, so a
    sibling pull of a different file cannot keep a hung child alive. That partial is identified by the
    blobs *child_pid* has open (precise across a resume), else the partials not in
    *baseline_incomplete_blobs* (captured pre-spawn). Snapshots keep the repo-wide measurement."""
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
                # Only partials this child holds open (handles a resume reusing a baseline name, excludes
                # siblings). hf_xet holds the .incomplete fd continuously, so an EMPTY set means the child
                # owns no partial YET (connect / metadata phase), not a sibling's.
                owned = {name: n for name, n in sizes.items() if name in open_names}
                return (sum(owned.values()), len(owned) > 0)
            if child_pid:
                # pid given but open files uninspectable (no psutil AND no /proc: native Windows / macOS
                # without psutil). Post-baseline name filtering would forever EXCLUDE a resumed partial
                # reusing a baseline name, so a frozen Xet resume never trips -- defeating the fallback.
                # Fall back to the repo-wide measure (as snapshots do): the resume is watched, at the cost
                # that a same-repo sibling's progress may mask this child's stall (accepted tradeoff).
                return get_hf_download_state(
                    [single_repo_id], repo_type = repo_type, cache_dir = cache_dir
                )
            # No child pid at all: follow only newly-created (post-baseline) partials.
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
                # trip a false stall once readable again.
                last_change = now
                _safe_status(on_heartbeat, f"Downloading ({transport} transport)...")
                continue

            current_size, has_incomplete = state
            if current_size != last_size:
                last_size = current_size
                last_change = now

            # Reset unless .incomplete confirms an active download, so model init and lock waits
            # are not counted as a stall.
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
    """Redact secrets from a child error string, preferring Studio's patterns if present."""
    try:
        from hub.utils.download_registry import scrub_secrets  # type: ignore

        return scrub_secrets(text, hf_token = token)
    except Exception:
        return _default_scrub_secrets(text, hf_token = token)


# Deterministic Hub failures that recur identically over either transport, so retrying HTTP is
# pointless: surface them. Matched by class NAME so the parent need not import HF's error classes.
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
})
# TYPE reconstructed across the spawn but NOT retry-deterministic: ``HfHubHTTPError`` bases both
# deterministic 4xx and transient 5xx / 429, so its retry stays status-code driven while the parent
# still re-raises the real type (not RuntimeError) so ``except HfHubHTTPError`` keeps working.
_TYPE_PRESERVE_ONLY_NAMES = frozenset({
    "HfHubHTTPError",
})
# Substrings marking a transient transport failure (hf_xet / CAS error, timeout, reset, 5xx / 429)
# that an HTTP retry may recover.
_TRANSIENT_ERROR_HINTS = (
    "xet", "casclient", "cas_", "timeout", "timed out", "connection", "reset by peer",
    "temporarily", "try again", "incompleteread", "protocolerror", "remotedisconnected",
    "broken pipe", "ssl", "eof occurred", "502", "503", "504", "500 server", "429",
    "too many requests", "service unavailable", "bad gateway", "gateway time",
    "connection aborted",
)


def _resolve_exception_class(type_name: str) -> "Optional[type]":
    """Map a deterministic Hub / OS error class NAME back to its class so the parent re-raises the
    original type, not RuntimeError. Best-effort (unknown -> None); local imports keep it import-light
    and independent of the huggingface_hub layout."""
    if type_name == "OSError":
        return OSError
    # Preserve builtin OSError subclasses (PermissionError, FileNotFoundError, ...): deterministic FS
    # failures the child cannot retry away, so a caller's `except OSError` / `except PermissionError`
    # must still fire rather than see RuntimeError.
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
    """Build an *exc_cls* instance carrying *message*, robust to a finicky constructor: Hub errors
    subclass ``HfHubHTTPError`` whose ``response`` arg can be required, so ``exc_cls(message)`` may raise
    ``TypeError``. Try normal constructors, then BYPASS ``__init__`` via ``__new__`` so type + message
    survive. None only if ``__new__`` fails."""
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
    """Pull the errno out of a stringified OSError (CPython formats it ``[Errno 28] ...``), so a
    disk-full / quota error keeps its code across the spawn boundary for ``exc.errno`` branching."""
    match = re.search(r"\[Errno (\d+)\]", message)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _is_builtin_oserror(exc: BaseException) -> bool:
    """True iff *exc* is a BUILTIN ``OSError`` (or subclass) with a real errno. Excludes HF/requests HTTP
    errors (``OSError`` via ``requests -> IOError`` but no OS errno), so a bracketed ``[Errno N]`` in
    their message is not mistaken for one."""
    if not isinstance(exc, OSError):
        return False
    builtin = getattr(builtins, type(exc).__name__, None)
    return isinstance(builtin, type) and issubclass(builtin, OSError) and isinstance(exc, builtin)


def _raise_child_error(message: str) -> None:
    """Re-raise a deterministic child error preserving its original TYPE for a known Hub / OS error, so
    callers' ``except RepositoryNotFoundError`` / ``GatedRepoError`` / ``OSError`` still match across the
    spawn. Child reports ``"<ClassName>: <message>"``; an unrecognized / uninstantiable class ->
    ``RuntimeError``."""
    type_name = message.split(":", 1)[0].strip() if ":" in message else ""
    exc_cls = _resolve_exception_class(type_name)
    if exc_cls is None:
        raise RuntimeError(message)
    exc = _instantiate_preserving_type(exc_cls, message)
    if exc is None:
        raise RuntimeError(message)
    if _is_builtin_oserror(exc) and getattr(exc, "errno", None) is None:
        # Preserve errno (ENOSPC / EDQUOT ...) across the spawn for `except OSError` cleanup, for EVERY
        # builtin OSError subclass. Restricted to BUILTIN types (an HF HTTP error is also an OSError via
        # requests -> IOError, but its "[Errno N]" is not a real errno). Set as an attribute, not via the
        # two-arg constructor: a single-arg __init__ subclass (LocalEntryNotFoundError) rejects it, and
        # this avoids a doubled "[Errno N]" prefix.
        errno_val = _parse_errno(message)
        if errno_val is not None:
            try:
                exc.errno = errno_val
            except Exception:
                pass
    raise exc


def _is_retryable_download_error(exc: BaseException) -> bool:
    """True when a captured exception looks like a transient transport failure (``hf_xet`` / CAS error,
    reset, timeout, 5xx / 429) the OTHER transport may recover, vs a deterministic Hub error (auth,
    not-found, gated, disk-full). Unknown errors count as deterministic, so a real repeatable failure is
    surfaced rather than looped between transports."""
    name = type(exc).__name__
    # LocalEntryNotFoundError wraps BOTH a genuine offline / uncached miss (deterministic) AND a
    # TRANSIENT HEAD connection error / timeout for an uncached file. Retry the transient sub-case; a
    # true offline miss (no transient hint) falls through to the deterministic set below.
    if name == "LocalEntryNotFoundError" and any(
        hint in f"{name}: {exc}".lower() for hint in _TRANSIENT_ERROR_HINTS
    ):
        return True
    if name in _DETERMINISTIC_ERROR_NAMES:
        return False
    # Disk full / quota: another transport cannot help.
    if isinstance(exc, OSError) and getattr(exc, "errno", None) in (errno.ENOSPC, errno.EDQUOT):
        return False
    # HTTP status (HfHubHTTPError carries a requests / httpx response): 5xx / 429 / 408 transient,
    # other 4xx (401 / 403 / 404 / 416) deterministic.
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if not isinstance(status, int):
        status = getattr(exc, "status_code", None)
    if isinstance(status, int):
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
    """Spawn-child entrypoint (top-level + picklable): set the Xet env BEFORE importing huggingface_hub,
    form its own process group so the parent can kill the whole transfer, never log token / signed
    URLs."""
    # Die with the parent on Linux under Studio (best-effort; module absent standalone).
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
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    repo_id = params["repo_id"]

    # Test-only fault injection (never set in production): stall the Xet attempt to exercise the
    # watchdog + HTTP fallback against a real repo.
    if not disable_xet and os.environ.get("UNSLOTH_HF_XET_FORCE_STALL") == "1":
        _stall_fh = None
        try:
            from huggingface_hub.constants import HF_HUB_CACHE

            # Write the fake partial under the repo_type-correct dir the watchdog scans.
            cache_root = params.get("cache_dir") or HF_HUB_CACHE
            repo_dir_name = f"{repo_type or 'model'}s--" + repo_id.replace("/", "--")
            blobs = os.path.join(cache_root, repo_dir_name, "blobs")
            os.makedirs(blobs, exist_ok = True)
            # Hold the partial OPEN for the whole stall (the single-file watchdog counts only partials
            # this PID holds open); bound to a local so it stays open across the sleep.
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
        # Classify here where the exception (status, errno, type) is intact, so the parent retries a
        # transient failure over HTTP yet surfaces a deterministic one without a second attempt.
        result_queue.put({
            "ok": False,
            "error": _scrub_in_child(f"{type(e).__name__}: {e}", token),
            "retryable": _is_retryable_download_error(e),
        })


def _terminate_process_group(proc: "mp.process.BaseProcess", grace_period: float) -> None:
    """Kill *proc* and its whole process group (Xet may spawn helpers). The child ``os.setsid()``s so its
    pgid == pid; the group is signalled via ``os.killpg`` only once ``os.getpgid(pid) == pid`` confirms
    it. SIGTERM, then SIGKILL after *grace_period*."""
    pid = proc.pid

    def _signal_group(sig: int) -> None:
        # Signal the GROUP only once the child is its own leader (pgid == pid after setsid). Before setsid
        # it is still in OUR group and its pid could collide with a recycled group, so ``getpgid != pid``
        # guards ``killpg`` from the WRONG group (a reaped child raises here). Otherwise (also Windows: no
        # killpg / getpgid) signal the single process.
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
    # SIGKILL only while alive, so the pid (== pgid) is a live target: once join() reaps a leader, a busy
    # host could recycle its pid into an unrelated group and killpg would hit the WRONG one. hf_xet 1.5.x
    # spawns no helpers, so a reaped leader leaves nothing to clean up.
    if proc.is_alive():
        # -9 sentinel takes the force-kill branch on Windows (signal.SIGKILL undefined; moot there since
        # proc.kill() == proc.terminate()), keeping the call site consistent.
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
    """Run one download in a spawn child under the no-progress watchdog. Returns ``("ok", path)``,
    ``("stall", None)``, ``("cancelled", None)``, ``("crashed", message)`` (crash, no captured
    exception), ``("retryable_error", message)`` (transient, worth an HTTP retry), or ``("error",
    message)`` (deterministic Hub error). Tests monkeypatch this seam to avoid spawning."""
    # Single-file: snapshot the on-disk partials BEFORE spawning so the watchdog follows only the blob(s)
    # this child writes, not a sibling's. Snapshots stay repo-wide.
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
    # Set the transport env in THIS process around the spawn so the child inherits it from creation: HF
    # caches HF_HUB_DISABLE_XET at import time and the child re-imports huggingface_hub before its body,
    # so a child-side assignment lands too late. The child still sets it defensively.
    child_env = {
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        # Skip unsloth_zoo's heavy torch / transformers / device init in the child.
        "UNSLOTH_ZOO_DISABLE_GPU_INIT": "1",
    }
    if disable_xet:
        child_env["HF_HUB_DISABLE_XET"] = "1"
        child_env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    with _SPAWN_ENV_LOCK:
        # Cache Hub's transport constants in the PARENT from the REAL env NOW, before the child-only
        # HF_HUB_DISABLE_XET=1 is briefly set below: else a concurrent thread's FIRST `import
        # huggingface_hub` in the spawn window caches the disabled value and routes later in-process
        # downloads over HTTP. No-op once imported.
        try:
            import huggingface_hub.constants  # noqa: F401
        except Exception:
            pass
        saved_env = {k: os.environ.get(k) for k in child_env}
        # 'spawn' reconstructs __main__ from __main__.__file__: a pseudo-path ('<stdin>', a notebook)
        # fails to start, and a real UNGUARDED caller script re-imports as __mp_main__, re-running
        # top-level from_pretrained -> "start a process before bootstrapping" -> the child exits without
        # a result. Point __main__ at THIS side-effect-free module for the spawn.
        main_module = sys.modules.get("__main__")
        saved_main_file = _UNSET
        saved_main_spec = _UNSET
        if main_module is not None:
            saved_main_file = getattr(main_module, "__file__", _UNSET)
            main_module.__file__ = __file__
            # `python -m pkg`: spawn prefers __spec__.name and re-imports BY NAME. Clearing __spec__
            # forces the __file__ path branch.
            saved_main_spec = getattr(main_module, "__spec__", _UNSET)
            main_module.__spec__ = None
        try:
            os.environ.update(child_env)
            proc.start()
        except BaseException:
            # proc.start() can raise (OSError under fd / thread exhaustion). The lifecycle try/finally
            # that closes the queue's pipe fds runs only after a successful start, so close it here to
            # avoid an fd leak.
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
                # Prefer a result the child enqueued in the watchdog's fire window, so a just-succeeded
                # download is not killed. Its feeder may not have flushed a microseconds-earlier put, so
                # use a short timeout, not get_nowait().
                try:
                    result = result_queue.get(timeout = 1.0)
                    break
                except queue.Empty:
                    pass
                # Capture the partials THIS child owns BEFORE killing it, so HTTP prep scopes its purge to
                # them. Prefer the per-pid open-fd set; else post-baseline partials; None -> coarser mtime
                # guard.
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
            # Process exited; drain any enqueued result. Short timeout, not get_nowait(): the child can
            # exit just before its feeder flushes the pipe, which would spuriously look resultless.
            try:
                result = result_queue.get(timeout = 1.0)
            except queue.Empty:
                result = None
    finally:
        if stop_watchdog is not None:
            stop_watchdog.set()
        proc.join(timeout = grace_period)
        # No loop exit may leak the child; _terminate_process_group is idempotent so a redundant call
        # is a no-op.
        if proc.is_alive():
            _terminate_process_group(proc, grace_period)
        # Release the queue's pipe fds now rather than at GC. The result is extracted and a killed child
        # has nothing to flush, so cancel the feeder before close.
        try:
            result_queue.cancel_join_thread()
            result_queue.close()
        except Exception:
            pass

    if result is None:
        # Child exited resultless: a process-level crash (native hf_xet abort / segfault), not a captured
        # exception, so the other transport may still succeed -- report "crashed".
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
    """No interrupted-download evidence for the SELECTED files: no dangling requested symlink and every
    EXACT-named requested file present. A dangling EXCLUDED weight does not reject the subset."""
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
    """A weight a DEFAULT ``from_pretrained`` reads: safetensors or bin only. Excludes gguf / pt / onnx /
    msgpack / ... -- a stale cache holding only e.g. ``model.Q4_K_M.gguf`` must not satisfy the load, else
    it fetches the missing ``model.safetensors`` over un-killable Xet. Optimizer state is already excluded
    by ``_is_loadable_weight_file``."""
    return _is_loadable_weight_file(name) and name.endswith((".safetensors", ".bin"))


# CANONICAL root model weight a DEFAULT (no-variant) load reads: model.safetensors / pytorch_model.bin,
# single or numbered shard (dash infix, not a dotted variant token). A PEFT adapter, a variant
# (model.fp16.safetensors), a gguf, and a non-canonical root (consolidated.safetensors, tf_model.h5) are
# NOT matched -- a default from_pretrained probes only these names, so a cache holding only something
# else would fetch the missing canonical weight over un-killable Xet.
_CANONICAL_ROOT_MODEL_WEIGHT_RE = re.compile(
    r"^(?:model|pytorch_model)(?:-\d{5}-of-\d{5})?\.(?:safetensors|bin)$"
)

# CANONICAL (non-variant) diffusers component weight a PLAIN pipeline load reads in a component subfolder:
# a base with no intermediate dotted token, single or numbered shard, safetensors or bin. A VARIANT
# weight (diffusion_pytorch_model.fp16.safetensors) is EXCLUDED, so a variant='fp16' stale cache does not
# read as a warm PLAIN pipeline (which would fetch the non-variant name over un-killable Xet).
_CANONICAL_COMPONENT_WEIGHT_RE = re.compile(
    r"^[^.]+(?:-\d{5}-of-\d{5})?\.(?:safetensors|bin)$"
)

# SINGLE-FILE canonical root TF / Flax weight a from_tf / from_flax load reads instead of a PyTorch
# format. A SHARDED TF/Flax weight is judged through its index instead -- a lone shard here must NOT read
# as present, else an incomplete sharded set is loaded over Xet.
_CANONICAL_ROOT_TF_FLAX_WEIGHT_RE = re.compile(r"^(?:tf_model\.h5|flax_model\.msgpack)$")

# The shard-index sidecars a sharded TF / Flax weight is enumerated through.
_TF_FLAX_WEIGHT_INDEX_NAMES = ("tf_model.h5.index.json", "flax_model.msgpack.index.json")


def _pytorch_root_weight_formats_ignored(ignore_patterns: Any) -> bool:
    """True when the ignore filter drops BOTH canonical PyTorch root weights (``model.safetensors`` AND
    ``pytorch_model.bin``) -- the ``from_tf`` / ``from_flax`` signature. Lets the readable-weight check
    count the TF/Flax weight the load actually reads rather than false-reject a complete h5/msgpack
    download. Never true for a normal load (which keeps a PyTorch format)."""
    return not _filter_paths(
        ["model.safetensors", "pytorch_model.bin"], None, ignore_patterns
    )

# A training-checkpoint subdir (checkpoint-500/): its weights are never read as diffusers pipeline
# COMPONENTS, so they must not mask missing unet/vae/text-encoder weights.
_CHECKPOINT_DIR_RE = re.compile(r"^checkpoint[-_]\d+$")


def _diffusers_active_component_dirs(specs: dict) -> set:
    """Declared components a pipeline actually loads: spec is a ``[library, class]`` pair with both
    non-null. A ``[null, null]`` (a disabled / optional component such as safety_checker) is excluded --
    the load skips it, so it is not required to be present."""
    active: set = set()
    for name, spec in specs.items():
        if (
            isinstance(spec, (list, tuple)) and len(spec) >= 2
            and spec[0] is not None and spec[1] is not None
        ):
            active.add(name)
    return active


def _diffusers_component_weights_complete(
    snapshot_dir: Path, *, variant: Optional[str], ignore_patterns: Any = None,
) -> bool:
    """True when a diffusers pipeline warm holds every weight a plain / variant load reads. Beyond "some
    declared component weight is present" it requires each DECLARED ACTIVE component to be materialised (a
    non-empty subfolder) AND each model-style component (one carrying ``config.json``) to hold a readable
    weight of the read format -- so a stale partial missing a whole component (unet present, vae absent) or
    holding a component's config without its weight is retried over un-killable Xet, not loaded. Excludes
    ROOT-level weights and training-checkpoint subtrees; applies the ignore filter (format the load reads).
    Fails OPEN on a malformed / empty ``model_index.json`` to the lenient any-component-weight check,
    preserving hang protection without false-rejecting. A variant load accepts a component's canonical
    weight as diffusers' per-component fallback."""
    specs = _diffusers_declared_component_specs(snapshot_dir)
    declared = set(specs) if specs else None
    active = _diffusers_active_component_dirs(specs) if specs else None
    infix_dot = f".{variant}." if variant else ""
    infix_dash = f".{variant}-" if variant else ""
    per_comp_canon: dict = {}
    per_comp_variant: dict = {}
    try:
        for entry in snapshot_dir.rglob("*"):
            name = entry.name
            if not _is_default_load_weight_file(name):
                continue
            try:
                if not entry.is_file():
                    continue
                rel = entry.relative_to(snapshot_dir).as_posix()
            except (OSError, ValueError):
                continue
            parts = rel.split("/")
            if len(parts) < 2:
                continue  # a ROOT-level weight is not a component
            comp = parts[0]
            if declared is not None and comp not in declared:
                continue  # an UNDECLARED subtree the load does not read
            if any(_CHECKPOINT_DIR_RE.match(p) for p in parts[:-1]):
                continue  # a training-checkpoint subtree, not a component
            if variant and (infix_dot in name or infix_dash in name):
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
                present = comp_dir.is_dir() and any(comp_dir.iterdir())
            except OSError:
                present = False
            if not present:
                return False  # a declared active component was never materialised
            try:
                has_config = (comp_dir / "config.json").is_file()
            except OSError:
                has_config = False
            if has_config and not _has_read_weight(comp):
                return False  # a model-style component holds its config but no readable weight
    # Floor: at least one component holds a weight of the READ format -- rejects a variant-only-for-plain,
    # config-only, checkpoint-only, or undeclared-leftover-only stale snapshot.
    if variant:
        return any(_has_variant(c) for c in (declared or per_comp_variant))
    return any(_has_canon(c) for c in (declared or per_comp_canon))


def _has_diffusers_component_weight(snapshot_dir: Path, *, ignore_patterns: Any = None) -> bool:
    """Whether a PLAIN (non-variant) diffusers pipeline warm is complete for the weights a load reads
    (see ``_diffusers_component_weights_complete``)."""
    return _diffusers_component_weights_complete(
        snapshot_dir, variant = None, ignore_patterns = ignore_patterns
    )


def _root_model_has_weight(snapshot_dir: Path, *, ignore_patterns: Any = None) -> bool:
    """Whether an UNPATTERNED model warm holds a weight a default load reads: a CANONICAL ROOT weight
    (single or numbered shard), or -- for a diffusers pipeline (root ``model_index.json``) -- a
    component-subfolder weight. Counting ANY subtree weight would accept a stale checkpoint-only snapshot
    then fetch the root weights over un-killable Xet; diffusers is the one layout with weights in
    subfolders. Only canonical names count (a VARIANT root, PEFT adapter, gguf, or NON-canonical
    consolidated.* is retried over HTTP -- its canonical weight is still missing). The ignore filter is
    applied so a partial holding only the format the load will NOT read does not count."""
    try:
        is_diffusers = (snapshot_dir / "model_index.json").is_file()
    except OSError:
        is_diffusers = False
    if is_diffusers:
        return _has_diffusers_component_weight(snapshot_dir, ignore_patterns = ignore_patterns)
    rels: list = []
    tf_flax_rels: list = []
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
            elif _CANONICAL_ROOT_TF_FLAX_WEIGHT_RE.match(name):
                tf_flax_rels.append(name)  # TF/Flax root weight (from_tf / from_flax)
    except OSError:
        return False
    if _filter_paths(rels, None, ignore_patterns):
        return True
    # from_tf / from_flax (both PyTorch formats ignored): count a SINGLE-FILE TF/Flax weight or a COMPLETE
    # sharded set (index + every listed shard present), so a complete h5/msgpack download is not
    # false-rejected, while an INCOMPLETE set is retried over HTTP. Gated so a normal load is unchanged
    # and a stray leftover never counts.
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
    """True if a CANONICAL ROOT model weight carrying the requested *variant* token (kept by the ignore
    filter) is present. transformers reads ``model.<variant>.safetensors`` and
    ``model.<variant>-00001-of-00002.safetensors`` (``.<variant>-`` shard infix), matched by
    ``_ROOT_MODEL_VARIANT_WEIGHT_RE`` plus the variant infix. A non-canonical base, PEFT adapter, or
    non-``model`` variant is excluded -> a cache holding only those is retried over HTTP. The ignore
    filter is applied so an ignored-format partial does not count."""
    infix_dot = f".{variant}."
    infix_dash = f".{variant}-"
    rels: list = []
    try:
        for entry in snapshot_dir.iterdir():
            name = entry.name
            if infix_dot not in name and infix_dash not in name:
                continue  # not the requested variant token
            if not _ROOT_MODEL_VARIANT_WEIGHT_RE.match(name):
                continue  # only a canonical model / pytorch_model variant weight, not adapter / gguf
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
    """Variant analog of ``_has_diffusers_component_weight``: whether a diffusers *variant* pipeline warm
    is complete (a pipeline's variant weights are component-scoped, not root ``model.<variant>.*``, so a
    root-only check would false-reject a complete variant download). See
    ``_diffusers_component_weights_complete``."""
    return _diffusers_component_weights_complete(
        snapshot_dir, variant = variant, ignore_patterns = ignore_patterns
    )


def _root_model_has_variant_weight(
    snapshot_dir: Path, variant: str, *, ignore_patterns: Any = None
) -> bool:
    """Variant analog of ``_root_model_has_weight``: an UNPATTERNED variant warm holds a ROOT variant
    weight, or -- for a diffusers pipeline -- a component-subfolder variant weight (its weights live in
    subfolders, not root ``model.<variant>.*``, so the root-only check would false-reject them)."""
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
# ``["pytorch_model.bin", "model.safetensors"]`` (and the variant pair) is satisfied by ANY one, since HF
# allow patterns are ALTERNATIVES over the repo. Distinct logical weights (base AND adapter, a different
# variant) stay separate groups (each required).
_EITHER_FORMAT_WEIGHT_RE = re.compile(
    r"^(model|pytorch_model|adapter_model)(?:\.([^.]+))?\.(?:safetensors|bin)$"
)


def _exact_weight_logical(base: str) -> Any:
    """Equivalence key for an EXACT-named weight so either-format alternatives share a group:
    ``model.safetensors`` / ``pytorch_model.bin`` -> ``("root_model", None)``; same variant in both
    formats -> ``("root_model", "<variant>")``; ``adapter_model.*`` -> ``("adapter_model", ...)``. A
    non-weight / sharded name maps to itself (still required individually)."""
    m = _EITHER_FORMAT_WEIGHT_RE.match(base)
    if m is None:
        return base
    stem, variant = m.group(1), m.group(2)
    logical = "adapter_model" if stem == "adapter_model" else "root_model"
    return (logical, variant)


def _requested_exact_files_present_grouped(
    snapshot_dir: Path, *, allow_patterns: Any, ignore_patterns: Any,
) -> bool:
    """True unless an EXACT-named requested file is missing. Interchangeable weights need any one; distinct
    logical files (base AND adapter, a tokenizer file) each. A glob / unpatterned request is trivially
    satisfied here."""
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
    """True if a loadable weight the request SELECTS (allow / ignore filtered) is present, so a stale
    out-of-scope weight does not satisfy a patterned request."""
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
    allow / ignore scope with the variant infix check, so a patterned variant load whose partial kept
    only the canonical weight is retried over HTTP (else it fetches the variant over un-killable Xet)."""
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
    """Whether the allow / ignore filter keeps a canonical ROOT shard name. When False, an incomplete
    canonical root shard set is OUT of scope (a leftover a patterned adapter / gguf / subfolder load never
    reads), so the shard-completeness gate must NOT reject on it, else a complete patterned download is
    failed into a DownloadStallError."""
    probes = ["model-00001-of-00002.safetensors", "pytorch_model-00001-of-00002.bin"]
    return bool(_filter_paths(probes, allow_patterns, ignore_patterns))


def _request_selects_root_variant_weight(
    allow_patterns: Any, ignore_patterns: Any, variant: str,
) -> bool:
    """Whether the allow / ignore filter keeps a ROOT variant weight name. When False, a stale incomplete
    root variant shard set is OUT of scope (e.g. a subfolder request whose variant weights live under
    ``unet/``), so the ROOT variant-shard gate must not reject on it, else a complete in-scope variant
    download is failed."""
    probes = [
        f"model.{variant}.safetensors", f"model.{variant}-00001-of-00002.safetensors",
        f"pytorch_model.{variant}.bin", f"pytorch_model.{variant}-00001-of-00002.bin",
    ]
    return bool(_filter_paths(probes, allow_patterns, ignore_patterns))


def _cache_can_skip_download(
    snapshot_dir: Path, *, repo_type: str, allow_patterns: Any, ignore_patterns: Any,
    variant: Optional[str] = None,
) -> bool:
    """PRE-download: whether a cached snapshot is complete enough to skip the protective child. STRICT --
    a false True lets the load fetch a missing weight over un-killable Xet (the hang).

    Weight-bearing model request: only the conservative canonical gate (``snapshot_dir_is_complete``)
    skips; anything uncertain (diffusers, variants, patterns, sharded-without-index) spawns the child. A
    weightless / non-model request has no weight to hang on but is locally provable complete only when it
    names EXACT files (an intact exact-named subset short-circuits; unpatterned / glob defers)."""
    if repo_type in (None, "model") and request_can_include_weights(allow_patterns, ignore_patterns):
        # A variant load reads variant-named weights the canonical gate does not check, so a
        # canonical-only cache would fetch the variant over un-killable Xet. Defer to the child.
        if variant:
            return False
        # A default load probes model.safetensors before pytorch_model.bin, so a bin-only cache for a
        # repo that also publishes safetensors (unprovable locally) would fetch the safetensors over Xet.
        # prefer_safetensors defers it; a use_safetensors=False request still fast-paths its bin cache.
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
    """Invariant A (presence): a weight the in-process load will READ is present, ignore filter ALWAYS
    applied and scope matched to the request:

    - variant + UNPATTERNED -> a ROOT variant weight;
    - variant + PATTERNED   -> a SELECTED variant weight;
    - plain  + UNPATTERNED  -> a ROOT (or diffusers-component) weight, NOT a stray subfolder checkpoint;
    - plain  + PATTERNED    -> a SELECTED weight.

    A partial that kept only the ignored format does not count -> retried over HTTP."""
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
    """Invariant B (shard completeness): an IN-SCOPE shard set the load reads is incomplete (index present
    with a shard missing, or a lone numbered shard without its index) and must be retried. ALWAYS scoped
    to what the request selects, so a co-resident stale shard set the load never reads does not
    false-reject a complete download:

    - variant: ROOT variant-shard check (NON-diffusers) for UNPATTERNED, or a PATTERNED request selecting
      a ROOT variant weight; a subfolder-scoped variant request does not root-check.
    - plain: canonical-root-shard check (NON-diffusers) for UNPATTERNED, or a GLOBBED request selecting
      canonical root shards; an exact-named subset / out-of-scope request does not.
    - non-root: a PATTERNED request also checks any SELECTED shard index the root checks miss (a sharded
      adapter, a component subfolder) via ``_selected_shard_index_incomplete``.
    - diffusers: a pipeline reads COMPONENT subfolders, NOT root model shards, so root checks are SKIPPED;
      its component shard sets go through ``_diffusers_component_shards_incomplete`` (unpatterned) /
      ``_selected_shard_index_incomplete`` (patterned).

    The ignore filter is threaded through so completeness is judged for the FORMAT the load reads."""
    try:
        is_diffusers = (snapshot_dir / "model_index.json").is_file()
    except OSError:
        is_diffusers = False
    if _patterns_are_exact_names(allow_patterns):
        # An exact-named subset defers to the exact-file presence check: the load reads exactly the named
        # shard(s), so a lone exact shard is not judged against its unrequested index (else false-reject).
        return False
    if variant:
        if not is_diffusers and (
            allow_patterns is None
            or _request_selects_root_variant_weight(allow_patterns, ignore_patterns, variant)
        ):
            # Only a non-diffusers root variant load runs the root-shard check (a diffusers pipeline's
            # variant weights are component-scoped, handled below).
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
            # UNPATTERNED variant diffusers warm: a component variant shard index is incomplete (the root
            # variant check above only covers root model.<variant> shards).
            return True
        return False
    if allow_patterns is None:
        if not is_diffusers and _has_incomplete_canonical_root_shards(
            snapshot_dir, ignore_patterns = ignore_patterns
        ):
            # non-diffusers root model load (a diffusers stale root index is handled by the component
            # check below).
            return True
        if is_diffusers and _diffusers_component_shards_incomplete(
            snapshot_dir, variant = None, ignore_patterns = ignore_patterns
        ):
            # UNPATTERNED plain diffusers warm: a component shard index missing a shard is not covered by
            # the canonical ROOT-shard check.
            return True
        return False
    if not is_diffusers and _request_selects_canonical_root_shards(
        allow_patterns, ignore_patterns
    ) and _has_incomplete_canonical_root_shards(snapshot_dir, ignore_patterns = ignore_patterns):
        # non-diffusers only: a diffusers pipeline never reads root model shards (its component sets are
        # checked below), so a stale root index must not reject it.
        return True
    return _selected_shard_index_incomplete(
        snapshot_dir, allow_patterns = allow_patterns,
        ignore_patterns = ignore_patterns, variant = None,
    )


def _selected_readable_weight_complete(
    snapshot_dir: Path, *, allow_patterns: Any, ignore_patterns: Any, variant: Optional[str],
) -> bool:
    """Weight-bearing MODEL acceptance check: the weight the load will READ is present (Invariant A) AND
    its in-scope shard set is complete (Invariant B). Both apply the ignore filter and match scope
    uniformly, so a co-resident out-of-scope / ignored-format partial neither masks an incomplete weight
    (a silent Xet hang) nor false-rejects a complete download (a spurious ``DownloadStallError``)."""
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
    """POST-download: whether the child's result is usable or should be retried over HTTP.
    snapshot_download already did the authoritative manifest compare, so accept unless there is POSITIVE
    breakage evidence; LENIENT otherwise so a good download is never looped into a ``DownloadStallError``.
    A transient connection error during the child's metadata call makes ``snapshot_download`` silently
    return a stale / partial cache, so the checks apply the request's filters to the weight the load
    reads. Breakage:

    - A dangling REQUESTED symlink (a missing / still-``.incomplete`` blob).
    - A missing EXACT-named requested file (grouped by weight equivalence; globs stay lenient).
    - A weight-bearing MODEL request whose READABLE weight is absent or incomplete (delegated to
      ``_selected_readable_weight_complete``)."""
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
    trusted; production always returns a real snapshot dir."""
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
    # Read xet_force_disabled() under the lock a CONCURRENT download briefly sets HF_HUB_DISABLE_XET
    # under (around its spawn), so this download cannot observe the other's child-only value and wrongly
    # force itself onto HTTP from the start.
    with _SPAWN_ENV_LOCK:
        disable_xet = xet_force_disabled()

    for attempt in range(2):
        if disable_xet:
            # Purge a non-HTTP partial first (an HTTP resume over a sparse Xet/hf_transfer partial
            # silently corrupts the blob), scoped to the stalled child's own partials so a same-repo
            # sibling is spared. An injected (Studio) hook keeps the plain (repo_type, repo_id) signature.
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
            # An unsafe partial that could not be cleared (locked / permission) would corrupt the blob on
            # an HTTP resume: force a clean re-download instead.
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
                # HF can hand back an existing incomplete snapshot dir (offline / timed-out) instead of
                # fetching: never load it in-process. Retry over HTTP, then fail loudly. (Patterned /
                # non-model requests judge their own subset, so a valid weightless snapshot is not
                # rejected.)
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
            # identically. _raise_child_error preserves the original type across the spawn.
            _raise_child_error(payload)
        if kind_result == "retryable_error":
            # Transient transport failure (hf_xet CAS timeout, 5xx, reset): retry HTTP once, else raise.
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
            # _safe_status: a raising status hook must not abort the retry before disable_xet is set.
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
    """Download a single file with Xet primary and HTTP as a stall-only fallback; return the local path.

    Raises ``RuntimeError("Cancelled")`` if *cancel_event* is set, re-raises a deterministic child error
    unchanged, and raises ``DownloadStallError`` only if BOTH transports stall. ``local_files_only=True``
    resolves from cache in-process with no child (HF offline semantics).
    """
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
    # Finalized blob already cached: return it with no child (skipped under force_download). A subfolder
    # file is cached under "<subfolder>/<filename>".
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
    """Download a whole repo snapshot with Xet primary and HTTP as a stall-only fallback; return the
    local snapshot dir.

    Used by Unsloth's ``from_pretrained`` to warm the cache in a killable child BEFORE the in-process
    load (which then hits a warm cache and cannot hang on a native Xet thread). A fully cached repo
    short-circuits in-process with no child. ``local_files_only=True`` resolves from cache in-process
    (HF offline semantics). ``variant`` forces the child even on a warm canonical cache, since the
    canonical gate cannot prove the variant-named weights present.
    """
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
            # local_files_only returns a snapshot dir whenever refs/<rev> + snapshots/<sha> exist, even
            # one left by a prior interrupted / patterned download. Validate the EXACT returned revision
            # dir (scoped to this snapshot, not the whole repo, so an unrelated revision mid-download does
            # not force a needless re-fetch).
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
