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

These helpers report, for a repo's blobs under ``HF_HUB_CACHE``, how many bytes are actually on disk
(sparse-aware, so a partial Xet / ``hf_transfer`` ``.incomplete`` is not read as full progress) and
whether an ``.incomplete`` partial is present -- the two signals the no-progress watchdog runs on.

``snapshot_dir_is_complete`` is a CONSERVATIVE fast-path gate, not an authoritative verifier: it
returns "complete" only for unambiguous canonical model layouts, and defers everything else
(diffusers, variants, patterns, datasets) to the watched ``snapshot_download`` child. A false
"complete" is the only dangerous error (an in-process load could then fetch a missing weight over
un-killable Xet); a false "not complete" only spawns the cheap child, so the gate errs that way.

Only the active ``HF_HUB_CACHE`` root is scanned; multi-root / transport-marker logic is a
download-manager concern that lives in the consumer.
"""

from __future__ import annotations

import fnmatch
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
    """``Path.is_dir()`` that returns False instead of raising on an unreadable path."""
    try:
        return path.is_dir()
    except OSError:
        return False


def _safe_is_file(path: Path) -> bool:
    """``Path.is_file()`` that returns False instead of raising on an unreadable / dangling path."""
    try:
        return path.is_file()
    except OSError:
        return False


def hf_cache_root(*, create: bool = False, cache_dir: "Optional[str | Path]" = None) -> Optional[Path]:
    """The hub cache root to scan, or None if unavailable. A given *cache_dir* is used verbatim;
    otherwise ``HF_HUB_CACHE`` is read lazily so an import-time redirect is honored."""
    if cache_dir is not None:
        # Match huggingface_hub, which expands ~ before writing. expanduser() raises if no home can
        # be resolved (HOME unset in a container); fall back to the literal path rather than crash.
        try:
            root = Path(cache_dir).expanduser()
        except (RuntimeError, OSError):
            root = Path(cache_dir)
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
    # repo_type=None is HF's default "model"; mirror that so None resolves models--<id>, not Nones--<id>.
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
    """Sparse-aware on-disk size: a Xet / ``hf_transfer`` ``.incomplete`` reports full ``st_size``
    while only some blocks are allocated, so prefer ``st_blocks``, falling back to ``st_size`` where
    it is unreported (Windows, some network filesystems)."""
    st = path.stat()
    blocks = getattr(st, "st_blocks", None)
    if blocks is not None:
        # Trust st_blocks even when 0: a truncated sparse .incomplete reports full st_size but 0
        # blocks and must read as 0 bytes present (a > 0 guard would fall through to st_size).
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
    """Newest child of ``repo_dir/snapshots`` by mtime (the signal from_pretrained resolves to), or None."""
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
    """True if *snapshot_dir* holds a dangling symlink (a referenced blob that is missing or still
    ``.incomplete``) -- an interrupted download. Validates one requested revision, not just the newest."""
    try:
        for entry in snapshot_dir.rglob("*"):
            if entry.is_symlink() and not entry.exists():
                return True
    except OSError:
        return False
    return False


# ---------------------------------------------------------------------------
# Weight-file recognition
# ---------------------------------------------------------------------------

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

# Trainer / optimizer state carries weight suffixes (.bin / .pt / .pth) but is NOT a loadable weight,
# so a cache holding only these is not a warm model cache.
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
# Distributed runs shard the RNG state as rng_state_0.pth, rng_state_1.pth, ...
_NON_WEIGHT_BASENAME_PREFIXES = ("rng_state_",)


def _is_loadable_weight_file(name: str) -> bool:
    """True if *name* is a loadable model weight: a weight suffix that is not a trainer / optimizer
    state artifact (training_args.bin, optimizer.pt, rng_state.pth, ...)."""
    if not name.endswith(_WEIGHT_FILE_SUFFIXES):
        return False
    lowered = name.lower()
    if lowered in _NON_WEIGHT_BASENAMES:
        return False
    if any(lowered.startswith(prefix) for prefix in _NON_WEIGHT_BASENAME_PREFIXES):
        return False
    return True


def _is_weight_shard_index(name: str) -> bool:
    """True for a weight-shard index sidecar, canonical or variant (``model.safetensors.index.json``
    and ``model.safetensors.index.fp16.json``); a plain suffix test would miss the variant form."""
    return name.endswith(".json") and (".safetensors.index." in name or ".bin.index." in name)


def _is_canonical_weight_shard_index(name: str) -> bool:
    """True only for the CANONICAL (non-variant) index a default load probes
    (``model.safetensors.index.json`` / ``pytorch_model.bin.index.json``). A variant
    (``...index.fp16.json``) is rejected: the wrapper takes no variant param, so a variant-only cache
    must not satisfy the canonical fast path (its canonical weights are still missing)."""
    return name.endswith(".safetensors.index.json") or name.endswith(".bin.index.json")


def _weight_shard_index_complete(index_path: Path) -> bool:
    """True only if every shard a HF weight index lists is present next to it.

    Fail-CLOSED: an unreadable / truncated index, a non-dict payload or ``weight_map``, or an empty
    shard set return False, so a malformed index defers to the watched child rather than letting the
    in-process load skip it and then fail (or fetch over Xet)."""
    import json

    try:
        with open(index_path, "r", encoding = "utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return False
    weight_map = data.get("weight_map") if isinstance(data, dict) else None
    if not isinstance(weight_map, dict):
        return False
    # A non-string value is unhashable (breaks set()) and invalid for ``base / shard``; filter first.
    shards = {s for s in weight_map.values() if isinstance(s, str)}
    if not shards:
        return False
    base = index_path.parent
    for shard in shards:
        try:
            if not (base / shard).exists():
                return False
        except OSError:
            return False
    return True


# ---------------------------------------------------------------------------
# Pattern helpers (normalization + glob detection + HF filtering)
# ---------------------------------------------------------------------------

_GLOB_CHARS = ("*", "?", "[")


def _has_glob(text: str) -> bool:
    # A trailing-slash dir pattern ("unet/") is not an exact filename: HF expands it like "unet/*",
    # so treat it as a wildcard rather than look for a literal "unet/" entry.
    return text.endswith("/") or any(ch in text for ch in _GLOB_CHARS)


def _as_pattern_list(patterns: "Optional[object]") -> "Optional[list]":
    """Normalize an allow / ignore argument to a list. HF accepts a bare ``str``; iterating it would
    walk it character by character ("checkpoint-10/*" would never match)."""
    if patterns is None:
        return None
    if isinstance(patterns, str):
        return [patterns]
    return list(patterns)


def _filter_paths(
    paths: list,
    allow_patterns: "Optional[list]" = None,
    ignore_patterns: "Optional[list]" = None,
) -> list:
    """Filter repo-relative *paths* by HF allow / ignore patterns (as ``snapshot_download`` selects).
    Fails OPEN (returns all paths) so a snapshot that does hold weights is never rejected on an
    unevaluable filter."""
    try:
        from huggingface_hub.utils import filter_repo_objects

        return list(
            filter_repo_objects(
                paths, allow_patterns = allow_patterns, ignore_patterns = ignore_patterns
            )
        )
    except Exception:
        return list(paths)


def _broken_symlink_rel_paths(snapshot_dir: Path) -> list:
    """Repo-relative posix paths of every dangling symlink in *snapshot_dir* (empty when none), so the
    interrupted-download signal can be scoped to the files a request actually selects."""
    out: list = []
    try:
        for entry in snapshot_dir.rglob("*"):
            try:
                if entry.is_symlink() and not entry.exists():
                    try:
                        out.append(entry.relative_to(snapshot_dir).as_posix())
                    except ValueError:
                        out.append(entry.name)
            except OSError:
                continue
    except OSError:
        return out
    return out


def snapshot_has_requested_broken_symlinks(
    snapshot_dir: Path,
    *,
    allow_patterns: "Optional[object]" = None,
    ignore_patterns: "Optional[object]" = None,
    repo_type: "Optional[str]" = "model",
) -> bool:
    """True iff a dangling symlink in *snapshot_dir* is for a file the request actually SELECTS, so a
    dangling root ``model.safetensors`` does not fail a weightless ``allow=["config.json"]`` request
    whose config is on disk. (*repo_type* is kept for signature compatibility.)"""
    allow_patterns = _as_pattern_list(allow_patterns)
    ignore_patterns = _as_pattern_list(ignore_patterns)
    broken = _broken_symlink_rel_paths(snapshot_dir)
    if not broken:
        return False
    return bool(_filter_paths(broken, allow_patterns, ignore_patterns))


# ---------------------------------------------------------------------------
# The conservative fast-path completeness gate
# ---------------------------------------------------------------------------

# Canonical root weight filenames a default load reads (the single file or its shard index proves warm).
_CANONICAL_SINGLE_WEIGHTS = ("model.safetensors", "pytorch_model.bin")


def _ignore_strips_all_weights(ignore_patterns: "list") -> bool:
    """True iff the ignore set provably excludes EVERY weight format (a probe of each suffix matches a
    pattern). A partial strip is NOT weightless -- a surviving variant / .gguf / .pt weight could
    still be pulled, so the request stays weight-bearing (conservative)."""
    for suffix in _WEIGHT_FILE_SUFFIXES:
        probe = "weight" + suffix
        if not any(isinstance(p, str) and fnmatch.fnmatchcase(probe, p) for p in ignore_patterns):
            return False
    return True


# Representative weight names a glob allow pattern is probed against (via fnmatch): a glob matching one
# can select a weight; one matching none (``tokenizer*``, ``*.json``) is weightless. Covers canonical /
# variant / sharded / adapter / diffusers / consolidated and the non-safetensors formats.
_WEIGHT_PATTERN_PROBES = (
    "model.safetensors",
    "model.fp16.safetensors",
    "model-00001-of-00002.safetensors",
    "pytorch_model.bin",
    "pytorch_model-00001-of-00002.bin",
    "adapter_model.safetensors",
    "adapter_model.bin",
    "consolidated.safetensors",
    "consolidated.00.pth",
    "diffusion_pytorch_model.safetensors",
    "model.gguf",
    "model.pt",
    "model.pth",
    "model.h5",
    "model.msgpack",
    "tf_model.h5",
    "flax_model.msgpack",
)

# Subdirs that hold only metadata / config, so a ``dir/`` pattern scoped to one is weightless. Any
# other dir pattern stays weight-bearing (a component dir or a checkpoint dir can hold a weight).
_NON_WEIGHT_DIRS = frozenset({
    "tokenizer",
    "processor",
    "preprocessor",
    "feature_extractor",
    "image_processor",
    "video_processor",
    "scheduler",
})


def _pattern_can_select_weight(pattern: "object") -> bool:
    """Whether a single allow pattern could select a weight. A weight-suffix basename or a non-metadata
    directory pattern is weight-bearing; a glob basename is weight-bearing only if it matches a
    ``_WEIGHT_PATTERN_PROBES`` name (so ``tokenizer*`` / ``*.json`` stay weightless while
    ``model.?afetensors`` / ``unet/*`` do not); a concrete non-weight name is weightless. A false
    weight-bearing only spawns the cheap child; the probe set avoids a false weightless on real weights."""
    if not isinstance(pattern, str):
        return True
    if pattern.endswith("/"):
        dir_name = pattern.rstrip("/").rsplit("/", 1)[-1].lower()
        return dir_name not in _NON_WEIGHT_DIRS
    # A pattern scoped under a metadata dir ("tokenizer/*", "processor/*.json") is weightless like the
    # "tokenizer/" form, instead of letting a "*" basename match a weight probe.
    if "/" in pattern:
        parent = pattern.rsplit("/", 1)[0].rstrip("/").rsplit("/", 1)[-1].lower()
        if parent in _NON_WEIGHT_DIRS:
            return False
    base = pattern.rsplit("/", 1)[-1]
    if base.endswith(_WEIGHT_FILE_SUFFIXES):
        return True
    if any(ch in base for ch in _GLOB_CHARS):
        return any(fnmatch.fnmatchcase(probe, base) for probe in _WEIGHT_PATTERN_PROBES)
    return False


def request_can_include_weights(
    allow_patterns: "Optional[object]" = None, ignore_patterns: "Optional[object]" = None
) -> bool:
    """Whether a request restricted by *allow_patterns* / *ignore_patterns* can still include a weight.
    Conservative: True when uncertain, so the acceptance check requires a weight; False only for a
    clearly weightless request (a tokenizer / config allow list, an ignore list dropping every weight
    format, or an allow + ignore pair that strips them all), preserving the tokenizer-only short-circuit."""
    allow_patterns = _as_pattern_list(allow_patterns)
    ignore_patterns = _as_pattern_list(ignore_patterns)
    if allow_patterns is None and ignore_patterns is None:
        return True
    if allow_patterns is None:
        return not _ignore_strips_all_weights(ignore_patterns or [])
    if not allow_patterns:
        return False  # allow=[] selects nothing
    if not any(_pattern_can_select_weight(pat) for pat in allow_patterns):
        return False
    # A root-reachable allow (no required subdir) can still be left weightless by the ignore filter
    # (allow=["*"] + ignore=[every weight suffix]). Apply HF's allow-then-ignore semantics to the weight
    # probes; a subdir-scoped allow stays weight-bearing (its required dir is absent from the root probes).
    if ignore_patterns and all(isinstance(p, str) and "/" not in p for p in allow_patterns):
        if not _filter_paths(list(_WEIGHT_PATTERN_PROBES), allow_patterns, ignore_patterns):
            return False
    return True


def _canonical_root_weights_complete(
    snapshot_dir: Path, entries: list, ignore_patterns: "Optional[list]" = None
) -> bool:
    """True iff the snapshot holds a complete canonical ROOT weight set: a root
    ``model.safetensors`` / ``pytorch_model.bin``, OR a root shard index whose every shard is present.
    Numbered shards without an index, or subfolder-only weights, do NOT count.

    A weight whose FORMAT the ignore filter drops does not count (a stale ``pytorch_model.bin`` under
    ``ignore=['*.bin']`` is not proof the requested safetensors are on disk). The format probe also
    discards a ``pytorch_model.bin.index.json`` whose ``.json`` name would slip the raw filter."""
    root_files: set = set()
    root_indices: list = []
    for entry in entries:
        try:
            rel = entry.relative_to(snapshot_dir).as_posix()
        except ValueError:
            rel = entry.name
        if "/" in rel:
            continue  # ROOT files only
        if _is_canonical_weight_shard_index(entry.name):
            if _safe_is_file(entry):
                root_indices.append(entry)
        elif _safe_is_file(entry):
            root_files.add(entry.name)

    def _format_kept(weight_name: str) -> bool:
        # The format a load reads from *weight_name* must survive the ignore filter, else the file is
        # a stale artifact for an excluded format and proves nothing.
        if not ignore_patterns:
            return True
        return bool(_filter_paths([weight_name], None, ignore_patterns))

    for index_entry in root_indices:
        fmt_probe = (
            "model.safetensors"
            if ".safetensors.index." in index_entry.name
            else "pytorch_model.bin"
        )
        if _format_kept(fmt_probe) and _weight_shard_index_complete(index_entry):
            return True
    return any(
        name in root_files and _format_kept(name) for name in _CANONICAL_SINGLE_WEIGHTS
    )


def snapshot_dir_is_complete(
    snapshot_dir: Path,
    *,
    allow_patterns: "Optional[object]" = None,
    ignore_patterns: "Optional[object]" = None,
    require_named_weights: bool = False,
) -> bool:
    """Conservative fast-path gate: True only for an unambiguously complete canonical ROOT model cache,
    so an in-process load will not fetch a weight. True requires: an UNPATTERNED request
    (``allow_patterns is None``), not a diffusers pipeline (no root ``model_index.json``), no dangling
    symlink, and canonical root weights present. Everything else defers to the watched child. A false
    True risks a silent Xet fetch; a false False only spawns the cheap child. *require_named_weights*
    is accepted for signature compatibility (a named-weight request is patterned, so never fast-pathed).

    *ignore_patterns* need no eligibility gate: the canonical-weight check below is what the load reads,
    so an ignore that dropped some format (the common ``*.bin`` / subdir prefetch ignores) cannot make
    an incomplete cache read complete -- keeping the common warm ``from_pretrained`` cache eligible."""
    allow_patterns = _as_pattern_list(allow_patterns)
    ignore_patterns = _as_pattern_list(ignore_patterns)
    if allow_patterns is not None:
        return False  # any allow list scopes the on-disk set unprovably -> defer
    try:
        entries = list(snapshot_dir.rglob("*"))
    except OSError:
        return False
    if _safe_is_file(snapshot_dir / "model_index.json"):
        return False  # diffusers needs component reasoning we do not fast-path
    if snapshot_dir_has_broken_symlinks(snapshot_dir):
        return False  # interrupted blob
    return _canonical_root_weights_complete(snapshot_dir, entries, ignore_patterns)


# A canonical numbered root shard: the index sits IMMEDIATELY before the extension (no variant token),
# so ``model-00001-of-00002.safetensors`` matches but ``model-00001-of-00002.fp16.safetensors`` does not.
_CANONICAL_ROOT_SHARD_RE = re.compile(
    r"^(?:model|pytorch_model)-\d{5}-of-\d{5}\.(?:safetensors|bin)$"
)


def _has_incomplete_canonical_root_shards(snapshot_dir: Path) -> bool:
    """True when the root holds canonical numbered shards but is NOT a complete canonical model (index
    missing or a shard absent) -- a stale interrupted download a default load cannot read, so the
    post-download check rejects it and retries over HTTP. Variant shards are excluded, so a
    variant-only repo is not force-failed here."""
    try:
        names = [entry.name for entry in snapshot_dir.iterdir()]
    except OSError:
        return False
    if not any(_CANONICAL_ROOT_SHARD_RE.match(name) for name in names):
        return False
    return not snapshot_dir_is_complete(snapshot_dir)


def requested_named_files_present(
    snapshot_dir: Path,
    *,
    allow_patterns: "Optional[object]" = None,
    ignore_patterns: "Optional[object]" = None,
) -> bool:
    """For a request naming EXACT files (every entry glob-free), True only when each named file the
    ignore filter keeps is on disk -- ``snapshot_download(local_files_only=True)`` returns the revision
    dir even when config-only, so a ``["tokenizer.json"]`` request needs its file present. A request
    with ANY glob, or no allow list, is trivially satisfied (it cannot be turned into an exact manifest)."""
    allow_patterns = _as_pattern_list(allow_patterns)
    ignore_patterns = _as_pattern_list(ignore_patterns)
    if not allow_patterns or any(_has_glob(p) for p in allow_patterns):
        return True
    try:
        entries = list(snapshot_dir.rglob("*"))
    except OSError:
        return True
    present = set()
    for entry in entries:
        if _safe_is_file(entry):
            try:
                present.add(entry.relative_to(snapshot_dir).as_posix())
            except ValueError:
                present.add(entry.name)
    for pat in allow_patterns:
        # A named file the ignore filter drops is not actually requested.
        if ignore_patterns and not _filter_paths([pat], None, ignore_patterns):
            continue
        if pat not in present:
            return False
    return True


# ---------------------------------------------------------------------------
# Active-cache enumeration primitives (download-manager / watchdog support)
# ---------------------------------------------------------------------------

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
    # Check every snapshot, not just the newest: a requested older revision may be broken while a
    # newer one is clean, and a latest-only check would report the repo healthy.
    return any(
        snapshot_dir_has_broken_symlinks(snapshot)
        for snapshot in _iter_snapshot_dirs(repo_dir)
    )


def _case_safe_repo_cache_dirs(root: Path, repo_type: Optional[str], repo_id: str) -> list:
    """Cache dirs safely attributable to this exact repo id.

    The Hub case-folds the dir name, so a case-insensitive match is needed, but on a case-sensitive
    filesystem ``models--Org--Repo`` and ``models--org--repo`` are distinct repos. Prefer an
    exact-case match; otherwise accept a single folded match ONLY when the filesystem is
    case-insensitive (the exact-case name resolves to it); on a 2+ way collision attribute to neither,
    so a stale partial in one repo cannot make the watchdog kill / purge the other."""
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
        # Attribute a single folded-but-not-exact match only on a case-insensitive filesystem, where
        # the exact-case path resolves to the same dir; on a case-sensitive fs it is a DIFFERENT repo.
        try:
            if (root / target).exists():
                return entries
        except OSError:
            return []
    return []


def iter_active_repo_cache_dirs(
    repo_type: Optional[str], repo_id: str, *, cache_dir: "Optional[str | Path]" = None
) -> Iterator[Path]:
    """Yield the repo's cache dir(s) under *cache_dir* (or the active ``HF_HUB_CACHE``). Case-collision
    safe, so the read / watchdog path and the destructive HTTP-prep path share one attribution rule."""
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
    repo_type: "Optional[str]", repo_id: str, *, cache_dir: "Optional[str | Path]" = None
) -> bool:
    for entry in iter_active_repo_cache_dirs(repo_type, repo_id, cache_dir = cache_dir):
        if repo_cache_dir_has_incomplete_blobs(entry):
            return True
    return False
