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
        # Path.expanduser() raises RuntimeError when no home can be resolved (a restricted
        # container with HOME unset); fall back to the literal path rather than crash the probe.
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


def _filter_paths(
    paths: list,
    allow_patterns: "Optional[list]" = None,
    ignore_patterns: "Optional[list]" = None,
) -> list:
    """Filter repo-relative *paths* by Hugging Face allow / ignore patterns, mirroring how
    ``snapshot_download`` selects files. On any failure, treat all paths as selected so a
    snapshot that does hold weights is never rejected for an unevaluable filter."""
    try:
        from huggingface_hub.utils import filter_repo_objects

        return list(
            filter_repo_objects(
                paths, allow_patterns = allow_patterns, ignore_patterns = ignore_patterns
            )
        )
    except Exception:
        return list(paths)


def _numbered_shard_set_present(
    entry: Path,
    *,
    snapshot_dir: "Optional[Path]" = None,
    allow_patterns: "Optional[list]" = None,
    ignore_patterns: "Optional[list]" = None,
) -> bool:
    """For a numbered weight shard (``model-00001-of-00002.safetensors``), True only when
    every shard in its ``-of-NNNNN`` set that the request selects is present in the same
    directory.

    A leftover single shard from an interrupted multi-shard download reads as a weight
    file on its own, so without this an incomplete pull (one shard on disk, the rest
    never fetched) would short-circuit as a warm cache. This catches that even when the
    shard *index* sidecar was never cached (so ``_weight_shard_index_complete`` has
    nothing to check). A non-numbered / single-file weight matches no shard pattern and
    is trivially satisfied.

    When *allow_patterns* / *ignore_patterns* are given, a sibling shard is required only
    if the request actually selects it: a deliberate single-shard request
    (``allow_patterns=["model-00002-of-00005.safetensors"]``) is satisfied by that one shard
    and must not demand the rest."""
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
    scoped = bool(allow_patterns or ignore_patterns) and snapshot_dir is not None
    for i in range(1, total + 1):
        shard_path = base / f"{prefix}-{i:0{width}d}-of-{total_str}{suffix}"
        if scoped:
            try:
                rel = shard_path.relative_to(snapshot_dir).as_posix()
            except ValueError:
                rel = shard_path.name
            if not _filter_paths([rel], allow_patterns, ignore_patterns):
                continue  # this sibling is not part of the request -> do not require it
        try:
            if not shard_path.exists():
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
    # weight_map values are filenames relative to the index file's own directory. They come from
    # arbitrary JSON: a non-string (e.g. list/dict) value is both unhashable -- so it would break
    # set() -- and invalid for ``base / shard``, so filter to strings BEFORE de-duplicating rather
    # than crash (consistent with the fail-open parse handling above).
    base = index_path.parent
    for shard in {s for s in weight_map.values() if isinstance(s, str)}:
        try:
            if not (base / shard).exists():
                return False
        except OSError:
            return False
    return True


def _broken_symlink_rel_paths(snapshot_dir: Path) -> list:
    """Repo-relative posix paths of every dangling symlink in *snapshot_dir* -- a referenced file
    whose blob is missing or still an ``.incomplete`` partial (an interrupted download). Empty when
    none. Lets a completeness check scope the interrupted-download signal to the files a request
    actually selects, rather than rejecting the whole snapshot for a dangle outside the request."""
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


def _requested_scope_filter(
    rels: list, allow_patterns: "Optional[list]", ignore_patterns: "Optional[list]"
) -> list:
    """The subset of repo-relative *rels* a request selects. Applies the allow / ignore filter, and
    when there is no ``allow_patterns`` (an UNPATTERNED or IGNORE-ONLY request -- a bare
    ``from_pretrained`` that reads ROOT weights) also drops per-checkpoint-dir paths the root load
    never reads, so a checkpoint-dir file neither satisfies the warm nor (as a dangling symlink)
    blocks it. An explicit ``allow_patterns`` is trusted as-is: a caller that names a checkpoint
    path opts back into it."""
    kept = _filter_paths(list(rels), allow_patterns, ignore_patterns)
    if allow_patterns is None:
        kept = [r for r in kept if not _path_under_checkpoint_dir(r)]
    return kept


def snapshot_dir_is_complete(
    snapshot_dir: Path,
    *,
    allow_patterns: "Optional[object]" = None,
    ignore_patterns: "Optional[object]" = None,
    require_named_weights: bool = False,
) -> bool:
    """Best-effort check that a cached snapshot actually holds the requested model weights.

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
    The aim is simply to never short-circuit a snapshot whose weights are not on disk.

    When *allow_patterns* / *ignore_patterns* are given, the weight that must be present is
    one the request actually selects: a request for ``adapter_model.safetensors`` (or a
    specific checkpoint shard) is satisfied only by that weight on disk, not by some other
    weight the snapshot happens to also carry. A deliberate single-shard request likewise
    requires only that shard, not its whole ``-of-NNNNN`` set. With no patterns, any loadable
    weight does, and every numbered shard set present must be complete.

    *require_named_weights* additionally requires every explicitly named exact weight in
    *allow_patterns* (e.g. ``["model.safetensors", "adapter_model.safetensors"]``) to be on
    disk, so a stale cache holding only one of them is not treated as complete. Off by default
    (used by the pre-download cache probe); a glob still selects a subset freely."""
    try:
        entries = list(snapshot_dir.rglob("*"))
    except OSError:
        return False
    allow_patterns = _as_pattern_list(allow_patterns)
    ignore_patterns = _as_pattern_list(ignore_patterns)
    # An empty allow list is a real (select-nothing) filter, not "unpatterned": treat any
    # non-None patterns as a scoped request so allow_patterns=[] does not fall into the full
    # warmup branch (consistent with request_can_include_weights).
    has_patterns = allow_patterns is not None or ignore_patterns is not None

    # A dangling symlink marks an interrupted download, but only one for a file the request
    # actually selects should reject the snapshot. A stale dangling root model.safetensors must
    # not fail an allow_patterns=["adapter_model.safetensors"] probe whose adapter weight IS on
    # disk, so scope the broken-symlink check to the requested files (and, for a root warm with no
    # allow_patterns, drop checkpoint-dir paths the bare load never reads) -- the same selection
    # _requested_scope_filter applies to the weights below.
    broken = _broken_symlink_rel_paths(snapshot_dir)
    if broken and _requested_scope_filter(broken, allow_patterns, ignore_patterns):
        return False

    index_entries: list = []
    weight_entries: list = []  # (entry, repo-relative path)
    for entry in entries:
        name = entry.name
        if name.endswith((".safetensors.index.json", ".bin.index.json")):
            if _safe_is_file(entry):
                index_entries.append(entry)
        elif _is_loadable_weight_file(name) and _safe_is_file(entry):
            try:
                rel = entry.relative_to(snapshot_dir).as_posix()
            except ValueError:
                rel = name
            weight_entries.append((entry, rel))

    # The weights the request selects that are present on disk (any present root weight when the
    # request is unpatterned). The snapshot can carry an unrelated weight while the requested one
    # is missing, so a patterned request must find one it actually selects. _requested_scope_filter
    # also excludes per-checkpoint-dir weights (checkpoint-500/model.safetensors, left behind by a
    # prior allow_patterns=["checkpoint-500/*"] pull) whenever there is no allow_patterns -- an
    # UNPATTERNED *or* IGNORE-ONLY root warm (e.g. ignore_patterns=["*.onnx"]) is still a bare
    # from_pretrained reading ROOT weights, so a checkpoint-only snapshot must not read as warm.
    selected = set(_requested_scope_filter([rel for _, rel in weight_entries], allow_patterns, ignore_patterns))
    if not selected:
        return False

    # A request that explicitly names exact files needs EACH of them on disk, not just one, so a
    # stale cache holding a subset is not short-circuited past the guarded download. WHICH names
    # are required depends on the request shape:
    #   * An exact-file request (no globs) -- ["model.safetensors", "tokenizer.json"], or a base
    #     plus a PEFT adapter ["model.safetensors", "adapter_model.safetensors"] -- names every
    #     file it wants, so each concrete name (weight OR non-weight) must be present. A cache with
    #     just the weight must not accept-warm while the named tokenizer / config is still missing.
    #   * A request containing ANY glob is a broad "warm what matches" selection where named aux
    #     files are best-effort (an optional vocab.txt / spiece.model the repo may simply lack), so
    #     only its concrete WEIGHT names are required -- demanding every optional aux file there
    #     would defeat the warm-cache short-circuit for normal repos.
    # Enforced only at the pre-download probe (require_named_weights), so the post-download check
    # stays lenient and never errors on an "either format" name (["pytorch_model.bin",
    # "model.safetensors"] against a safetensors-only repo) that does not exist in the repo. A name
    # the ignore filter drops is not actually requested.
    if require_named_weights and allow_patterns:
        exact_only = not any(_has_glob(p) for p in allow_patterns)
        if exact_only:
            present = set()
            for entry in entries:
                if _safe_is_file(entry):
                    try:
                        present.add(entry.relative_to(snapshot_dir).as_posix())
                    except ValueError:
                        present.add(entry.name)
        else:
            present = set(rel for _, rel in weight_entries)
        for pat in allow_patterns:
            if _has_glob(pat):
                continue
            if not exact_only and not str(pat).lower().endswith(_WEIGHT_FILE_SUFFIXES):
                continue
            if ignore_patterns and not _filter_paths([pat], None, ignore_patterns):
                continue
            # pat is a concrete (glob-free) path, so presence is an exact match. A direct
            # membership test (not _filter_paths, which fails OPEN by returning all paths on a
            # filter error) keeps this strict check fail-SAFE: an unevaluable case requires the
            # guarded download rather than silently accepting a stale cache as warm.
            if pat not in present:
                return False

    # Every selected numbered shard needs the sibling shards the request also selects (the
    # whole set when unpatterned), so an interrupted multi-shard pull is not read as warm.
    for entry, rel in weight_entries:
        if rel not in selected:
            continue
        if not _numbered_shard_set_present(
            entry, snapshot_dir = snapshot_dir,
            allow_patterns = allow_patterns, ignore_patterns = ignore_patterns,
        ):
            return False

    # A full (unpatterned) warm also validates any shard index ships all its shards; a
    # patterned request may legitimately want only a subset, so the index is not enforced. A
    # per-checkpoint index (checkpoint-500/model.safetensors.index.json) does not gate a root
    # warm for the same reason its weights do not, so it is skipped here too.
    if not has_patterns:
        for index_entry in index_entries:
            try:
                index_rel = index_entry.relative_to(snapshot_dir).as_posix()
            except ValueError:
                index_rel = index_entry.name
            if _path_under_checkpoint_dir(index_rel):
                continue
            if not _weight_shard_index_complete(index_entry):
                return False
    return True


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


# Stems that, by convention, name a per-checkpoint DIRECTORY (whose weights live inside),
# not a file. Used to disambiguate a dotted no-slash glob like ``checkpoint-v1.*`` (a
# checkpoint directory, weights nested) from a file glob like ``tokenizer.*`` -- both are
# structurally ``<stem>.*`` but only the former can include weights.
_CHECKPOINT_DIR_PREFIXES = (
    "checkpoint", "ckpt", "epoch", "step", "global_step", "iter", "iteration",
)


def _looks_like_checkpoint_dir(pattern: str) -> bool:
    lowered = pattern.lower()
    return any(lowered.startswith(prefix) for prefix in _CHECKPOINT_DIR_PREFIXES)


def _path_under_checkpoint_dir(rel: str) -> bool:
    """True when a repo-relative *rel* lives inside a per-checkpoint directory
    (``checkpoint-500/model.safetensors``, ``global_step1000/pytorch_model.bin``). Only the
    PARENT components are checked -- the final component is the filename itself. Used to keep a
    checkpoint-dir weight from satisfying an unpatterned (root-model) warmup: such a weight is
    what a prior ``allow_patterns=["checkpoint-500/*"]`` pull leaves behind, not the root weight
    a bare ``from_pretrained`` reads."""
    parts = rel.split("/")
    return any(_looks_like_checkpoint_dir(p) for p in parts[:-1] if p)


def _bracket_member(content: str) -> str:
    """A single character that a glob ``[...]`` class *matches*, for concretizing a bracket
    expression into a probe that still satisfies the caller's own pattern. ``[0-9]`` -> ``0``,
    ``[a-z]`` -> ``a``; a negated class (``[!...]`` / ``[^...]``) -> a filler the class does
    not exclude. Replacing the class with a non-member (a literal ``x`` for ``[0-9]``) would
    make the probe fail the caller's pattern and misread the request as weightless."""
    negated = content[:1] in ("!", "^")
    if not negated:
        # The first listed item is a member: a literal char, or the low end of a leading range.
        return content[0] if content else "x"
    # Negated: pick a filler the class does not exclude (fnmatch mirrors HF's matcher).
    try:
        cls = "[" + content + "]"
        for cand in ("x", "0", "a", "z", "9", "_", "-", "A"):
            if fnmatch.fnmatch(cand, cls):
                return cand
    except Exception:
        pass
    return "x"


def _concretize_glob(pattern: str) -> str:
    """Replace glob wildcards in *pattern* with a literal filler so it can stand in as a
    concrete directory name (e.g. ``checkpoint-*`` -> ``checkpoint-x``). A ``[...]`` class
    collapses to one member char it actually matches (so the probe still satisfies the
    pattern). Used to probe weights nested under a no-slash directory glob, since Hugging
    Face's ``fnmatch`` ``*`` spans ``/``."""
    out = []
    i = 0
    n = len(pattern)
    while i < n:
        ch = pattern[i]
        if ch in ("*", "?"):
            out.append("x")
            i += 1
        elif ch == "[":
            j = pattern.find("]", i + 1)
            if j != -1:
                out.append(_bracket_member(pattern[i + 1 : j]))
                i = j + 1
            else:
                out.append("x")  # unterminated class: treat "[" as a literal filler
                i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


# Representative NON-weight files a catch-all ("*") or a config / tokenizer glob ("*.json")
# would also match -- used to tell a weight-specific basename (model.*, *.safetensors) from a
# catch-all when deciding whether a path-qualified request under a plain subfolder targets
# weights. Not exhaustive; just enough common names to disqualify a non-weight glob.
_NON_WEIGHT_PROBE_NAMES = (
    "config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "preprocessor_config.json",
    "vocab.json",
    "merges.txt",
    "readme.md",
    "training_args.bin",
    "optimizer.pt",
)


# Subfolders that, by convention, hold only auxiliary / telemetry / config files -- never model
# weights. A catch-all glob under one of these (tokenizer/*, runs/*, scheduler/*) is read as
# weightless. Kept deliberately narrow: an unknown subfolder (unet/, transformer/, original/, a
# new arch's component dir) must stay weight-including, so a weight-bearing dir is never misread
# as weightless (that would re-open the silent-Xet-hang accept-stale this module exists to
# prevent). The Diffusers pipeline components listed here (scheduler/, feature_extractor/, the
# extra tokenizers) ship only *_config.json / vocab files; the weight-bearing pipeline dirs
# (unet/, transformer/, vae/, text_encoder*/, image_encoder/, safety_checker/) are deliberately
# absent so a catch-all under them stays weight-including.
_NON_WEIGHT_DIR_NAMES = frozenset({
    "tokenizer", "tokenizer_2", "tokenizer_3", "runs", "run", "logs", "log", "samples", "sample",
    "tensorboard", "tb", "events", "eval", "evals", "evaluation", "metrics", "wandb", "assets",
    "images", "media", "scheduler", "feature_extractor",
})


def _basename_targets_weight(basename: str) -> bool:
    """True when a path-qualified pattern's basename specifically selects a model weight
    (``model.*``, ``adapter_model.*``, ``*.safetensors``), so the request is weight-including
    even under a non-weight parent dir. A catch-all (``*``) matches both weights and non-weights
    and a non-weight glob (``*.json``) matches no weight, so neither counts."""
    base = basename.lower()
    if not any(fnmatch.fnmatchcase(name, base) for name in _WEIGHT_PROBE_NAMES):
        return False
    return not any(fnmatch.fnmatchcase(name, base) for name in _NON_WEIGHT_PROBE_NAMES)


def _basename_is_non_weight(basename: str) -> bool:
    """True when a path-qualified pattern's basename clearly selects only non-weight files
    (``*.json``, ``config.json``, ``tokenizer.*``, ``*.txt``): it matches a known non-weight
    representative but no weight name. A catch-all (``*``) matches a weight too, so it is NOT
    clearly non-weight and stays weight-including (the parent dir may hold weights)."""
    base = basename.lower()
    if not any(fnmatch.fnmatchcase(name, base) for name in _NON_WEIGHT_PROBE_NAMES):
        return False
    return not any(fnmatch.fnmatchcase(name, base) for name in _WEIGHT_PROBE_NAMES)


def _parent_is_non_weight_dir(prefix: str) -> bool:
    """True when *prefix* is a known auxiliary / telemetry dir (tokenizer/, runs/, logs/) and no
    component looks like a checkpoint / weight dir, so a catch-all glob under it holds no weights.
    An unknown subfolder returns False (stays weight-including) to avoid accept-stale."""
    parts = [p.lower() for p in prefix.split("/") if p]
    if any(_looks_like_checkpoint_dir(p) for p in parts):
        return False
    return any(p in _NON_WEIGHT_DIR_NAMES for p in parts)


def _weight_self_probe(pattern: str) -> "Optional[str]":
    """A concretized stand-in for *pattern* when it names a loadable model weight by suffix
    (``lora_*.safetensors`` -> ``lora_x.safetensors``, ``checkpoint-10/lora_*.bin`` ->
    ``checkpoint-10/lora_x.bin``, a bare ``model-00002-of-00005.safetensors``), so a custom
    weight basename that matches no canonical probe is still recognized. Returns None when the
    suffix is not a weight suffix, or when the (concretized) basename is a known trainer /
    optimizer artifact (``optimizer.pt``, ``training_args.bin``, ``rng_state_*.pth``): those
    carry weight suffixes but the snapshot completeness check filters them out as non-weights,
    so classifying such a request as weight-including would loop the guarded download."""
    if not pattern.lower().endswith(_WEIGHT_FILE_SUFFIXES):
        return None
    concrete = _concretize_glob(pattern)
    basename = concrete.rsplit("/", 1)[-1]
    if not _is_loadable_weight_file(basename):
        return None
    return concrete


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
    # Only a truly unfiltered request (both None) is an unconditional weight warmup. An empty
    # allow list is NOT None: Hugging Face's filter_repo_objects treats allow_patterns=[] as
    # selecting NO objects, so the request is weightless -- collapsing [] with None here would
    # reject a legitimately empty snapshot and loop the guarded download.
    if allow_patterns is None and ignore_patterns is None:
        return True
    try:
        from huggingface_hub.utils import filter_repo_objects
    except Exception:
        return True  # cannot evaluate the filter -> assume weights are expected

    probes = list(_WEIGHT_PROBE_NAMES)
    for pat in (allow_patterns or ()):
        # A concretized stand-in when the pattern itself names a loadable weight by suffix
        # (lora_*.safetensors, checkpoint-10/lora_*.bin, a bare non-first shard). None for a
        # non-weight suffix and for a known trainer artifact (optimizer.pt, training_args.bin),
        # keeping this consistent with the snapshot completeness check.
        self_probe = _weight_self_probe(pat)
        if "/" in pat:
            # Path-qualified: re-root the canonical weight probes under the parent dir
            # (concretized when the parent is globbed) so the request is checked inside that
            # directory. Default to re-rooting (weight-including), because an unknown subfolder
            # (unet/, transformer/, original/, mp_rank_*/) may hold weights and reading it as
            # weightless would accept a stale config-only cache -> the silent Xet hang. Skip the
            # re-root only when the request is clearly non-weight: a non-weight basename glob
            # (*.json, tokenizer.*, *.txt), or a catch-all under a known auxiliary dir
            # (tokenizer/*, runs/*) that does not itself target a weight. A weight-suffix
            # basename is still recognized by self_probe below; the final filter applies ignores.
            prefix, base = pat.rsplit("/", 1)
            clearly_weightless = _basename_is_non_weight(base) or (
                _parent_is_non_weight_dir(prefix) and not _basename_targets_weight(base)
            )
            if not clearly_weightless:
                concrete_parent = _concretize_glob(prefix) if _has_glob(prefix) else prefix
                probes.extend(f"{concrete_parent}/{name}" for name in _WEIGHT_PROBE_NAMES)
        elif (
            _has_glob(pat)
            and ("." not in pat or _looks_like_checkpoint_dir(pat))
            and not _basename_is_non_weight(pat)
        ):
            # A no-slash DIRECTORY glob ("checkpoint-*", "global_step*", the dotted
            # "checkpoint-v1.*"): HF's fnmatch "*" spans "/", so it matches nested weights like
            # checkpoint-10/model.safetensors. Probe the canonical weights re-rooted under a
            # concretized form of the glob. A plain extension file glob ("*.json", "tokenizer.*")
            # is not a directory glob and stays weightless unless it names a weight (self_probe).
            # A no-slash glob whose stem is a known metadata family ("tokenizer*", "config*",
            # "vocab*", "special_tokens*") is a FILE glob, not a directory: _basename_is_non_weight
            # excludes it so a tokenizer*-only warm that fetched tokenizer.json is not rejected for
            # lacking a weight ("model*" / "pytorch_model*" stay weight-including -- they match a
            # weight probe, so _basename_is_non_weight is False for them).
            concrete = _concretize_glob(pat)
            probes.extend(f"{concrete}/{name}" for name in _WEIGHT_PROBE_NAMES)
        # A pattern that itself names a loadable weight -- a bare filename, a path-qualified
        # name, or a weight-suffix glob whose stem matches no canonical probe (lora_*.safetensors,
        # checkpoint-*/lora_*.bin) -- is recognized via its self-probe. "adapter_model.*" rides
        # the canonical adapter probe instead, and a trainer artifact yields no self-probe and
        # stays weightless. Everything is subject to the final ignore_patterns filter below.
        if self_probe is not None:
            probes.append(self_probe)

    try:
        kept = list(
            filter_repo_objects(
                probes, allow_patterns = allow_patterns, ignore_patterns = ignore_patterns
            )
        )
    except Exception:
        return True
    return len(kept) > 0


def requested_named_files_present(
    snapshot_dir: Path,
    *,
    allow_patterns: "Optional[object]" = None,
    ignore_patterns: "Optional[object]" = None,
) -> bool:
    """For a request that names EXACT files (every ``allow_patterns`` entry is glob-free), True only
    when each named file the ignore filter keeps is on disk.

    ``snapshot_download(local_files_only=True)`` returns a snapshot dir whenever the revision folder
    exists -- even a config-only one left by a prior ``AutoConfig`` fetch -- so for a weightless
    request like ``allow_patterns=["tokenizer.json"]`` a dangling-symlink check alone would accept a
    cache that does not actually contain the requested file. This makes that request require its
    named file before the snapshot is treated as warm.

    A request with ANY glob, or with no ``allow_patterns``, is a best-effort "warm what matches" and
    cannot be turned into an exact manifest (an optional ``vocab.txt`` the repo may simply lack would
    wrongly fail it), so it is trivially satisfied here -- the weight-bearing requests are gated by
    ``snapshot_dir_is_complete`` instead."""
    allow_patterns = _as_pattern_list(allow_patterns)
    ignore_patterns = _as_pattern_list(ignore_patterns)
    if not allow_patterns or any(_has_glob(p) for p in allow_patterns):
        return True
    try:
        entries = list(snapshot_dir.rglob("*"))
    except OSError:
        return True  # cannot enumerate -> do not reject on an unreadable dir
    present = set()
    for entry in entries:
        if _safe_is_file(entry):
            try:
                present.add(entry.relative_to(snapshot_dir).as_posix())
            except ValueError:
                present.add(entry.name)
    for pat in allow_patterns:
        # A named file the ignore filter drops is not actually requested. _filter_paths fails OPEN
        # (returns all on error), so an unevaluable filter keeps the strict presence check.
        if ignore_patterns and not _filter_paths([pat], None, ignore_patterns):
            continue
        if pat not in present:
            return False
    return True


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
    repo_type: "Optional[str]", repo_id: str, *, cache_dir: "Optional[str | Path]" = None
) -> bool:
    for entry in iter_active_repo_cache_dirs(repo_type, repo_id, cache_dir = cache_dir):
        if repo_cache_dir_has_incomplete_blobs(entry):
            return True
    return False
