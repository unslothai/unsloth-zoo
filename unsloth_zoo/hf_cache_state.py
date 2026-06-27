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

import re
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
    "snapshot_dir_is_complete",
    "request_can_include_weights",
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


def _safe_is_file(path: Path) -> bool:
    """``Path.is_file()`` (follows symlinks) returning False instead of raising on an
    unreadable path or a dangling link, so snapshot enumeration never errors out."""
    try:
        return path.is_file()
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
    blocks = getattr(st, "st_blocks", None)
    if blocks is not None:
        # st_blocks is reported (POSIX): trust it even when 0. A freshly truncated
        # sparse .incomplete reports st_size == full but 0 allocated blocks, and
        # must count as 0 bytes present, not full size (a > 0 guard would fall
        # through to st_size and read an empty partial as complete).
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


# Model weight file extensions. A snapshot with none of these is config/tokenizer
# only (e.g. a prior AutoConfig fetch), so it is not a warm cache for a weight load.
_WEIGHT_FILE_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".gguf",
    ".ckpt",
    ".onnx",
    ".msgpack",
    ".h5",
    ".pdparams",
)

# Trainer / optimizer state files carry weight suffixes (.bin / .pt / .pth) but are NOT
# loadable model weights. A checkpoint dir or a patterned pull can leave only these behind,
# so they must not satisfy the "snapshot holds its weights" check (which would skip the
# killable download while from_pretrained still lacks real weights).
_NON_WEIGHT_BASENAMES = frozenset({
    "training_args.bin",
    "optimizer.bin",
    "optimizer.pt",
    "scheduler.bin",
    "scheduler.pt",
    "scaler.pt",
    "rng_state.pt",
    "rng_state.pth",
})
# Distributed trainer runs shard the RNG state as rng_state_0.pth, rng_state_1.pth, ...
_NON_WEIGHT_BASENAME_PREFIXES = ("rng_state_",)


def _is_loadable_weight_file(name: str) -> bool:
    """True if *name* is a loadable model-weight file: a recognized weight suffix that is
    not a known trainer / optimizer state artifact (training_args.bin, optimizer.pt,
    scheduler.pt, rng_state.pth, ...). Those share weight suffixes but are not model
    weights, so a cache holding only them is not a warm model cache."""
    if not name.endswith(_WEIGHT_FILE_SUFFIXES):
        return False
    lowered = name.lower()
    if lowered in _NON_WEIGHT_BASENAMES:
        return False
    if any(lowered.startswith(prefix) for prefix in _NON_WEIGHT_BASENAME_PREFIXES):
        return False
    return True


# Numbered shard naming, e.g. ``model-00001-of-00002.safetensors`` or
# ``pytorch_model-00003-of-00004.bin``: prefix, 1-based index, total, suffix.
_NUMBERED_SHARD_RE = re.compile(
    r"^(?P<prefix>.+)-(?P<idx>\d+)-of-(?P<total>\d+)(?P<suffix>\.[^.]+)$"
)


def _numbered_shard_set_present(entry: Path) -> bool:
    """For a numbered weight shard (``model-00001-of-00002.safetensors``), True only when
    every shard in its ``-of-NNNNN`` set is present in the same directory.

    A leftover single shard from an interrupted multi-shard download reads as a weight
    file on its own, so without this an incomplete pull (one shard on disk, the rest
    never fetched) would short-circuit as a warm cache. This catches that even when the
    shard *index* sidecar was never cached (so ``_weight_shard_index_complete`` has
    nothing to check). A non-numbered / single-file weight matches no shard pattern and
    is trivially satisfied."""
    match = _NUMBERED_SHARD_RE.match(entry.name)
    if match is None:
        return True
    total_str = match.group("total")
    try:
        total = int(total_str)
    except ValueError:
        return True
    if total <= 0:
        return True
    prefix = match.group("prefix")
    suffix = match.group("suffix")
    width = len(total_str)
    base = entry.parent
    for i in range(1, total + 1):
        shard_name = f"{prefix}-{i:0{width}d}-of-{total_str}{suffix}"
        try:
            if not (base / shard_name).exists():
                return False
        except OSError:
            return False
    return True


def _weight_shard_index_complete(index_path: Path) -> bool:
    """True if every shard a HF weight index (``model.safetensors.index.json`` /
    ``pytorch_model.bin.index.json``) lists is present next to the index. An unreadable
    or non-sharded index is treated as satisfied (nothing extra to verify), so this only
    ever rejects an index whose shards are demonstrably missing on disk."""
    import json

    try:
        with open(index_path, "r", encoding = "utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return True
    weight_map = data.get("weight_map") if isinstance(data, dict) else None
    if not isinstance(weight_map, dict):
        return True
    # weight_map values are filenames relative to the index file's own directory.
    base = index_path.parent
    for shard in set(weight_map.values()):
        try:
            if not (base / shard).exists():
                return False
        except OSError:
            return False
    return True


def snapshot_dir_is_complete(snapshot_dir: Path) -> bool:
    """Best-effort check that a cached snapshot actually holds its model weights.

    ``snapshot_download(local_files_only=True)`` returns a snapshot dir whenever
    ``refs/<rev>`` and ``snapshots/<sha>`` exist, even one left by a prior interrupted
    or patterned download (a config-only snapshot from an ``AutoConfig`` fetch, or a
    partial shard pull). A dangling-symlink check alone misses those: the missing files
    were never symlinked, so nothing dangles. Treating such a snapshot as a warm cache
    skips the killable child and lets the in-process load hit Xet on the absent weights.

    A snapshot is complete only when it has no dangling symlinks, every weight-shard
    index it ships resolves all its shards on disk, every numbered shard set present has
    all its members on disk (even with no index sidecar), and it contains at least one
    weight file. This does NOT assert that every non-weight file is present (no offline
    manifest exists for that); the killable child completes anything else still missing.
    The aim is simply to never short-circuit a snapshot whose weights are not on disk."""
    if snapshot_dir_has_broken_symlinks(snapshot_dir):
        return False
    try:
        entries = list(snapshot_dir.rglob("*"))
    except OSError:
        return False
    has_weight = False
    for entry in entries:
        name = entry.name
        if name.endswith((".safetensors.index.json", ".bin.index.json")):
            if not _safe_is_file(entry):
                continue
            if not _weight_shard_index_complete(entry):
                return False
            has_weight = True
        elif _is_loadable_weight_file(name) and _safe_is_file(entry):
            if not _numbered_shard_set_present(entry):
                return False
            has_weight = True
    return has_weight


# Representative loadable-weight filenames -- the probe set for "can this request include a
# weight file". One per recognized format and naming convention (full model, sharded, PEFT
# adapter, consolidated / original checkpoint, diffusers), so a weight-selecting glob like
# ``adapter_model.*`` or ``consolidated.*`` matches a probe and is not misread as weightless.
# The shard *index* sidecars (``*.safetensors.index.json`` / ``*.bin.index.json``) are
# intentionally absent: they are JSON metadata, not weights, so a metadata-only request such
# as ``allow_patterns=["*.json"]`` (or ``["*.index.json"]``) must read as weightless.
_WEIGHT_PROBE_NAMES = (
    "model.safetensors",
    "model-00001-of-00002.safetensors",
    "pytorch_model.bin",
    "pytorch_model-00001-of-00002.bin",
    "adapter_model.safetensors",
    "adapter_model.bin",
    "consolidated.00.pth",
    "consolidated.safetensors",
    "diffusion_pytorch_model.safetensors",
    "diffusion_pytorch_model.bin",
    "tf_model.h5",
    "flax_model.msgpack",
    "model.gguf",
    "model.pt",
    "model.pth",
    "model.ckpt",
    "model.onnx",
    "model.pdparams",
)

_GLOB_CHARS = ("*", "?", "[")


def _has_glob(text: str) -> bool:
    return any(ch in text for ch in _GLOB_CHARS)


def _as_pattern_list(patterns: "Optional[object]") -> "Optional[list]":
    """Normalize an allow / ignore pattern argument to a list. Hugging Face accepts a bare
    ``str`` as well as a list, and iterating the ``str`` form would walk it character by
    character (so ``"checkpoint-10/*"`` would never match), misclassifying the request."""
    if patterns is None:
        return None
    if isinstance(patterns, str):
        return [patterns]
    return list(patterns)


def _pattern_basename_targets_weight(pattern: str) -> bool:
    """True if *pattern*'s final path component looks like it selects a weight file: a
    catch-all (``*`` / ``**``) or a name / glob ending in a recognized weight suffix.
    Used only when the pattern's parent directory is itself globbed, so no concrete probe
    path can be formed."""
    base = pattern.rsplit("/", 1)[-1].lower()
    if base in ("*", "**"):
        return True
    return base.endswith(_WEIGHT_FILE_SUFFIXES)


def request_can_include_weights(
    allow_patterns: "Optional[list]" = None, ignore_patterns: "Optional[list]" = None
) -> bool:
    """Whether a download restricted by *allow_patterns* / *ignore_patterns* can still
    include a model weight file.

    Used to decide whether snapshot completeness should require weights: a request that
    filters every weight format out (e.g. ``ignore_patterns`` covering ``*.safetensors``
    and ``*.bin`` to fetch only config / tokenizer files from a model repo) legitimately
    yields a weightless snapshot, so requiring a weight there would reject a valid result.
    An unfiltered request -- or one any weight filename survives -- includes weights.

    Path-qualified requests are handled too: ``allow_patterns`` such as
    ``["checkpoint-10/*"]`` or ``["models/*.safetensors"]`` probe the canonical weight
    names re-rooted under that directory, and a bare non-first shard like
    ``["model-00002-of-00005.safetensors"]`` is probed verbatim, so a request that does
    target weights inside a subfolder / at a specific shard is not misread as weightless.

    *allow_patterns* / *ignore_patterns* accept the ``str`` or ``list[str]`` forms that
    Hugging Face itself accepts."""
    allow_patterns = _as_pattern_list(allow_patterns)
    ignore_patterns = _as_pattern_list(ignore_patterns)
    if not allow_patterns and not ignore_patterns:
        return True
    try:
        from huggingface_hub.utils import filter_repo_objects
    except Exception:
        return True  # cannot evaluate the filter -> assume weights are expected

    probes = list(_WEIGHT_PROBE_NAMES)
    for pat in (allow_patterns or ()):
        if "/" in pat:
            prefix = pat.rsplit("/", 1)[0]
            if _has_glob(prefix):
                # Globbed parent dir (e.g. "checkpoint-*/*.safetensors"): no concrete path
                # to test, so decide from the basename. Only a weight-targeting basename
                # flips this on, so config/tokenizer globs under a wildcard dir stay
                # weightless and are not forced into a strict weight check.
                if _pattern_basename_targets_weight(pat):
                    return True
                continue
            # Concrete parent dir: re-root the canonical weight probes under it so a
            # path-qualified request is checked inside that directory, not only at the root.
            probes.extend(f"{prefix}/{name}" for name in _WEIGHT_PROBE_NAMES)
        # A bare concrete weight filename (e.g. a specific non-first shard, or a
        # subfolder-qualified weight) is itself a probe the filter can match verbatim.
        if not _has_glob(pat) and pat.lower().endswith(_WEIGHT_FILE_SUFFIXES):
            probes.append(pat)

    try:
        kept = list(
            filter_repo_objects(
                probes, allow_patterns = allow_patterns, ignore_patterns = ignore_patterns
            )
        )
    except Exception:
        return True
    return len(kept) > 0


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
