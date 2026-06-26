# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# This file is licensed under the GNU Affero General Public License v3.0 only
# (AGPL-3.0-only), unlike the rest of unsloth_zoo which is LGPL-3.0-or-later. It
# is the single shared home for the sparse-aware Hugging Face cache primitives
# used by the Xet -> HTTP stall fallback (unsloth_zoo.hf_xet_fallback) and by
# Unsloth Studio's download manager, which re-exports the names below.
# See <https://www.gnu.org/licenses/agpl-3.0.html>.

"""Sparse-aware introspection of the active Hugging Face hub cache.

These helpers answer two questions for a repo's blobs under ``HF_HUB_CACHE``:
how many bytes are actually on disk (sparse-aware, so a partially written Xet /
``hf_transfer`` ``.incomplete`` is not mistaken for full-size progress) and
whether an ``.incomplete`` partial is present. The no-progress download watchdog
is built on exactly these two signals.

Only the single active cache root (``huggingface_hub.constants.HF_HUB_CACHE``) is
scanned here; multi-root / legacy-cache enumeration and transport-marker logic
are download-manager concerns that live in the consumer, not in this module.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator, Optional


__all__ = [
    "INCOMPLETE_SUFFIX",
    "hf_cache_root",
    "target_dir_name",
    "repo_cache_dir_name",
    "blob_bytes_present",
    "latest_snapshot_dir",
    "snapshot_dir_has_broken_symlinks",
    "iter_active_repo_cache_dirs",
    "repo_cache_dir_has_incomplete_blobs",
    "has_active_incomplete_blobs",
]


INCOMPLETE_SUFFIX = ".incomplete"


def _safe_is_dir(path: Path) -> bool:
    """``Path.is_dir()`` returning False instead of raising when the path or a
    parent is unreadable (e.g. a restricted ``~/.cache/huggingface/hub``), so
    cache enumeration skips that root rather than erroring."""
    try:
        return path.is_dir()
    except OSError:
        return False


def hf_cache_root(*, create: bool = False, cache_dir: "Optional[str | Path]" = None) -> Optional[Path]:
    """The hub cache root to scan, or None if unavailable.

    When *cache_dir* is given (a caller-supplied ``snapshot_download`` cache), it
    is used verbatim; otherwise the active ``HF_HUB_CACHE`` is read lazily so any
    redirect applied at import time (see
    ``unsloth_zoo.hf_cache.redirect_hf_cache_if_readonly``) is honored.
    """
    if cache_dir is not None:
        # Match huggingface_hub, which expands ~ before writing; scanning the
        # literal path would otherwise miss a partial under e.g. ~/hf-cache.
        root = Path(cache_dir).expanduser()
    else:
        try:
            from huggingface_hub import constants as hf_constants
        except ImportError:
            return None
        root = Path(hf_constants.HF_HUB_CACHE)
    if create:
        try:
            root.mkdir(parents = True, exist_ok = True)
        except OSError:
            return None
        return root
    return root if _safe_is_dir(root) else None


def target_dir_name(repo_type: Optional[str], repo_id: str) -> str:
    return repo_cache_dir_name(repo_type, repo_id).lower()


def repo_cache_dir_name(repo_type: Optional[str], repo_id: str) -> str:
    # Hugging Face treats repo_type=None as the default "model"; mirror that here
    # so a caller forwarding repo_type=None still resolves models--<id> (not
    # Nones--<id>, which would make the cache-state probe miss real partials).
    repo_type = repo_type or "model"
    return f"{repo_type}s--{repo_id.replace('/', '--')}"


def _blob_dir_is_partial(blobs_dir: Path) -> bool:
    try:
        for blob in blobs_dir.iterdir():
            if blob.is_file() and blob.name.endswith(INCOMPLETE_SUFFIX):
                return True
    except OSError:
        return False
    return False


def blob_bytes_present(path: Path) -> int:
    """Sparse-aware on-disk size: XET / ``hf_transfer`` ``.incomplete`` partials
    report a full ``st_size`` while only some blocks are allocated, so prefer
    ``st_blocks``, falling back to ``st_size`` where it is unreported (Windows,
    some network filesystems)."""
    st = path.stat()
    blocks = getattr(st, "st_blocks", 0)
    if blocks > 0:
        return min(blocks * 512, st.st_size)
    if sys.platform == "win32":
        allocated = _windows_allocated_size(path)
        if allocated is not None:
            return min(allocated, st.st_size)
    return st.st_size


def _windows_allocated_size(path: Path) -> Optional[int]:
    """Best-effort allocated-byte count for sparse files on Windows."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
        get_compressed_file_size = kernel32.GetCompressedFileSizeW
        get_compressed_file_size.argtypes = [
            wintypes.LPCWSTR,
            ctypes.POINTER(wintypes.DWORD),
        ]
        get_compressed_file_size.restype = wintypes.DWORD

        high = wintypes.DWORD(0)
        ctypes.set_last_error(0)
        low = get_compressed_file_size(str(path), ctypes.byref(high))
        if low == 0xFFFFFFFF and ctypes.get_last_error() != 0:
            return None
        return (int(high.value) << 32) + int(low)
    except Exception:
        return None


def latest_snapshot_dir(repo_dir: Path) -> Optional[Path]:
    """Newest immediate child of ``repo_dir/snapshots`` by mtime, or None.

    mtime is the signal huggingface_hub's from_pretrained resolves to, so this
    points at whatever snapshot most recently landed on disk.
    """
    snapshots_dir = repo_dir / "snapshots"
    try:
        if not snapshots_dir.is_dir():
            return None
        snapshots = [entry for entry in snapshots_dir.iterdir() if entry.is_dir()]
        if not snapshots:
            return None
        return max(snapshots, key = lambda entry: entry.stat().st_mtime)
    except OSError:
        return None


def snapshot_dir_has_broken_symlinks(snapshot_dir: Path) -> bool:
    """True if *snapshot_dir* contains a dangling symlink -- a file the snapshot
    references whose blob is missing or still an ``.incomplete`` partial, i.e. an
    interrupted download. Used to validate one specific (caller-requested)
    revision, not just the newest one on disk."""
    try:
        for entry in snapshot_dir.rglob("*"):
            if entry.is_symlink() and not entry.exists():
                return True
    except OSError:
        return False
    return False


def _iter_snapshot_dirs(repo_dir: Path) -> Iterator[Path]:
    snapshots_dir = repo_dir / "snapshots"
    try:
        if not snapshots_dir.is_dir():
            return
        children = [entry for entry in snapshots_dir.iterdir() if entry.is_dir()]
    except OSError:
        return
    yield from children


def _repo_dir_has_broken_snapshot_symlinks(repo_dir: Path) -> bool:
    # Check every snapshot, not just the newest by mtime: a caller may request an
    # older revision whose snapshot is broken while a more recent one is clean, so
    # a latest-only check would report the repo healthy and let the interrupted
    # revision load with missing files.
    return any(
        snapshot_dir_has_broken_symlinks(snapshot)
        for snapshot in _iter_snapshot_dirs(repo_dir)
    )


def _case_safe_repo_cache_dirs(root: Path, repo_type: Optional[str], repo_id: str) -> list:
    """Cache dirs that can be safely attributed to this exact repo id.

    The cache dir name is case-folded by the Hub, so a case-insensitive match is
    needed for compatibility, but a bare case-insensitive match is unsafe: on a
    case-sensitive filesystem ``models--Org--Repo`` and ``models--org--repo`` are
    distinct repos. Prefer an exact-case match; otherwise accept a single folded
    match ONLY when the filesystem is case-insensitive (so the folded dir really is
    the same entry); on a 2+ way collision attribute to neither, so a stale partial
    in one repo cannot be charged to the other (which would let the watchdog kill an
    unrelated active download or HTTP-prep purge the wrong repo).
    """
    target = repo_cache_dir_name(repo_type, repo_id)
    folded_target = target.lower()
    try:
        entries = [entry for entry in root.iterdir() if entry.name.lower() == folded_target]
    except OSError:
        return []
    exact = [entry for entry in entries if entry.name == target]
    if exact:
        return exact
    if len(entries) == 1:
        # A single folded-but-not-exact match. Attribute it to this repo only when
        # the filesystem is case-insensitive: looking up the exact-case name then
        # resolves to that same directory. On a case-sensitive filesystem the
        # exact-case path does not exist, so the folded dir is a DIFFERENT repo and
        # must not be charged here.
        try:
            if (root / target).exists():
                return entries
        except OSError:
            return []
    return []


def iter_active_repo_cache_dirs(
    repo_type: Optional[str], repo_id: str, *, cache_dir: "Optional[str | Path]" = None
) -> Iterator[Path]:
    """Yield the repo's cache dir(s) under *cache_dir* (or the active ``HF_HUB_CACHE``).

    Case-collision safe (see ``_case_safe_repo_cache_dirs``), so both the read /
    watchdog path and the destructive HTTP-prep path share one attribution rule.
    """
    root = hf_cache_root(cache_dir = cache_dir)
    if root is None:
        return
    yield from _case_safe_repo_cache_dirs(root, repo_type, repo_id)


def repo_cache_dir_has_incomplete_blobs(repo_dir: Path) -> bool:
    blobs_dir = repo_dir / "blobs"
    return (blobs_dir.is_dir() and _blob_dir_is_partial(blobs_dir)) or (
        _repo_dir_has_broken_snapshot_symlinks(repo_dir)
    )


def has_active_incomplete_blobs(
    repo_type: str, repo_id: str, *, cache_dir: "Optional[str | Path]" = None
) -> bool:
    for entry in iter_active_repo_cache_dirs(repo_type, repo_id, cache_dir = cache_dir):
        if repo_cache_dir_has_incomplete_blobs(entry):
            return True
    return False
