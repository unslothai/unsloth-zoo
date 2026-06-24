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
inside Studio and standalone.
"""

from __future__ import annotations

import importlib.util
import multiprocessing as mp
import logging
import os
import queue
import re
import signal
import threading
import time
from typing import Any, Callable, Optional

from unsloth_zoo.hf_cache_state import (
    INCOMPLETE_SUFFIX,
    blob_bytes_present,
    has_active_incomplete_blobs,
    hf_cache_root,
    iter_active_repo_cache_dirs,
)

logger = logging.getLogger(__name__)

_CTX = mp.get_context("spawn")

# Defaults match the existing Studio inference watchdog and hub shutdown deadline.
DEFAULT_HEARTBEAT_INTERVAL = 30.0
DEFAULT_STALL_TIMEOUT = 180.0
DEFAULT_GRACE_PERIOD = 10.0
_POLL_INTERVAL = 0.5

# Serializes the brief parent-env mutation around a child spawn (below) so
# concurrent downloads cannot observe each other's transport env.
_SPAWN_ENV_LOCK = threading.Lock()

# Hugging Face boolean env convention: 1 / ON / YES / TRUE, case-insensitive.
_TRUTHY = {"1", "true", "yes", "on"}


def _is_true(value: Optional[str]) -> bool:
    return value is not None and str(value).strip().lower() in _TRUTHY


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
    if hf_token:
        out = out.replace(hf_token, "***")
    out = re.sub(r"hf_[A-Za-z0-9]{8,}", "***", out)
    out = re.sub(r"([Bb]earer\s+)[A-Za-z0-9._\-]+", r"\1***", out)
    return out


def _default_prepare_for_http(
    repo_type: str, repo_id: str, *, cache_dir: Optional[str] = None
) -> None:
    """Generic 'make the partial safe for an HTTP resume': delete the repo's active
    ``*.incomplete`` blobs (an HTTP resume over a sparse Xet/hf_transfer partial
    silently corrupts the blob). Studio injects its marker-aware version instead."""
    try:
        for entry in iter_active_repo_cache_dirs(repo_type, repo_id, cache_dir = cache_dir):
            blobs_dir = entry / "blobs"
            if not blobs_dir.is_dir():
                continue
            for blob in blobs_dir.iterdir():
                if blob.is_file() and blob.name.endswith(INCOMPLETE_SUFFIX):
                    try:
                        blob.unlink()
                    except OSError:
                        # A locked / permission-denied blob (common on Windows)
                        # must not abort cleanup of the rest of the partials.
                        continue
    except Exception as e:
        logger.debug("default prepare_for_http failed for %s: %s", repo_id, e)


def get_hf_download_state(
    repo_ids: Optional[list[str]] = None,
    *,
    repo_type: str = "model",
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
    repo_type: str = "model",
    cache_dir: Optional[str] = None,
    interval: float = DEFAULT_HEARTBEAT_INTERVAL,
    stall_timeout: float = DEFAULT_STALL_TIMEOUT,
    xet_disabled: bool = False,
    on_heartbeat: Optional[Callable[[str], None]] = None,
) -> threading.Event:
    """Start a daemon thread that fires ``on_stall(message)`` exactly once iff a
    ``*.incomplete`` is present AND the on-disk size is unchanged for
    *stall_timeout* seconds. The timer resets while no ``*.incomplete`` exists, so
    post-download init is never misread as a stall. Scans *cache_dir* when the
    download targets a caller-supplied cache, else the active ``HF_HUB_CACHE``.
    Returns a stop event the caller sets when the download phase ends.
    """
    stop = threading.Event()
    transport = "https" if xet_disabled else "xet"
    fired = False

    def _beat() -> None:
        nonlocal fired
        state = get_hf_download_state(repo_ids, repo_type = repo_type, cache_dir = cache_dir)
        last_size = state[0] if state is not None else 0
        last_change = time.monotonic()

        while not stop.wait(interval):
            state = get_hf_download_state(repo_ids, repo_type = repo_type, cache_dir = cache_dir)
            now = time.monotonic()

            if state is None:
                # Unmeasurable this tick (transient FS error): treat as progress
                # so a long unmeasurable gap cannot trip a false stall the instant
                # the state becomes readable again.
                last_change = now
                if on_heartbeat is not None:
                    on_heartbeat(f"Downloading ({transport} transport)...")
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

            if on_heartbeat is not None:
                on_heartbeat(f"Downloading ({transport} transport)...")

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
        )

    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id = params["repo_id"],
        filename = params["filename"],
        repo_type = repo_type,
        token = token,
        revision = params.get("revision"),
        cache_dir = params.get("cache_dir"),
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
        try:
            from huggingface_hub.constants import HF_HUB_CACHE

            blobs = os.path.join(HF_HUB_CACHE, "models--" + repo_id.replace("/", "--"), "blobs")
            os.makedirs(blobs, exist_ok = True)
            with open(os.path.join(blobs, "xet-force-stall.incomplete"), "wb") as fh:
                fh.write(b"\0" * 4096)
        except OSError:
            pass
        while True:
            time.sleep(3600)

    try:
        path = _child_download(kind = kind, params = params, token = token, repo_type = repo_type)
        result_queue.put({"ok": True, "path": path})
    except BaseException as e:  # noqa: BLE001 - report every failure to the parent
        result_queue.put({"ok": False, "error": _scrub_in_child(f"{type(e).__name__}: {e}", token)})


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

    Returns ``("ok", path)``, ``("stall", None)``, ``("cancelled", None)``, or
    ``("error", message)``. This is the seam tests monkeypatch to avoid spawning.
    """
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
    child_env = {"HF_HUB_DISABLE_PROGRESS_BARS": "1"}
    if disable_xet:
        child_env["HF_HUB_DISABLE_XET"] = "1"
        child_env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    with _SPAWN_ENV_LOCK:
        saved_env = {k: os.environ.get(k) for k in child_env}
        try:
            os.environ.update(child_env)
            proc.start()
        finally:
            for k, v in saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    # Bind the child to the parent lifetime when running under Studio (best-effort).
    try:
        from utils.process_lifetime import adopt_pid  # type: ignore

        adopt_pid(proc.pid)
    except Exception:
        pass

    stalled = threading.Event()
    stop_watchdog = start_watchdog(
        repo_ids = [repo_id],
        on_stall = lambda msg: stalled.set(),
        repo_type = repo_type,
        cache_dir = params.get("cache_dir"),
        interval = interval,
        stall_timeout = stall_timeout,
        xet_disabled = disable_xet,
        on_heartbeat = on_status,
    )

    result: Optional[dict] = None
    try:
        while proc.is_alive():
            if cancel_event is not None and cancel_event.is_set():
                _terminate_process_group(proc, grace_period)
                return ("cancelled", None)
            if stalled.is_set():
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
        stop_watchdog.set()
        proc.join(timeout = grace_period)

    if result is None:
        return (
            "error",
            f"download process for '{repo_id}' exited "
            f"(code={proc.exitcode}) without a result",
        )
    if result.get("ok"):
        return ("ok", result["path"])
    return ("error", result.get("error") or "unknown download error")


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
    # The Unsloth/HF knobs can force HTTP from the very first attempt.
    disable_xet = xet_force_disabled()

    for attempt in range(2):
        if disable_xet:
            # Purge a non-HTTP partial before resuming over HTTP: an HTTP resume
            # over a sparse Xet/hf_transfer partial silently corrupts the blob.
            # The generic purge is cache_dir-aware; an injected (Studio) hook owns
            # its own cache accounting and keeps the (repo_type, repo_id) signature.
            try:
                if prepare_for_http_fn is None:
                    _default_prepare_for_http(repo_type, repo_id, cache_dir = cache_dir)
                else:
                    prepare_for_http_fn(repo_type, repo_id)
            except Exception as e:
                logger.debug("prepare_for_http failed for %s: %s", repo_id, e)

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
            return payload  # type: ignore[return-value]
        if kind_result == "cancelled":
            raise RuntimeError("Cancelled")
        if kind_result == "error":
            # Deterministic failure: the other transport would fail identically.
            raise RuntimeError(payload)
        # kind_result == "stall"
        if not disable_xet:
            logger.warning(
                "Download stalled for '%s' -- retrying with HF_HUB_DISABLE_XET=1", label
            )
            if on_status is not None:
                on_status(f"{label}: Xet stalled, retrying over HTTP")
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
    token: Optional[str],
    *,
    cancel_event: Optional[threading.Event] = None,
    repo_type: str = "model",
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
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
    """
    # Finalized blob already cached: return it with no child and no network.
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(
            repo_id, filename, repo_type = repo_type, revision = revision, cache_dir = cache_dir
        )
        if isinstance(cached, str) and os.path.exists(cached):
            return cached
    except Exception as e:
        logger.debug("Cached probe failed for %s/%s: %s", repo_id, filename, e)

    return _download_with_xet_fallback(
        repo_id = repo_id,
        label = f"{repo_id}/{filename}",
        kind = "file",
        params = {"repo_id": repo_id, "filename": filename, "revision": revision, "cache_dir": cache_dir},
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
    repo_type: str = "model",
    cache_dir: Optional[str] = None,
    allow_patterns: Optional[Any] = None,
    ignore_patterns: Optional[Any] = None,
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
    via ``local_files_only`` with no child and no network.
    """
    # Fast path: everything already on disk -> resolve in-process (no Xet, no hang).
    try:
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
