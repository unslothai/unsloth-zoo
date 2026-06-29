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

The completeness check here is intentionally a CONSERVATIVE fast-path gate, not an
authoritative snapshot verifier. It returns "complete" only for the unambiguous
canonical model-cache layouts whose local evidence proves an in-process load will
not fetch a weight. Everything else (diffusers pipelines, weight variants,
non-trivial allow/ignore patterns, datasets, any layout needing inference) returns
"not complete" so the caller runs the authoritative Hugging Face download/resume in
the watched child. Returning a false "complete" is the only dangerous error (it can
send an in-process load to fetch a missing weight over un-killable Xet); returning a
false "not complete" only spawns the cheap watched child, so the gate errs that way.

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


# ---------------------------------------------------------------------------
# Weight-file recognition (small helpers the conservative completeness gate needs)
# ---------------------------------------------------------------------------

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
# so they must not count as a model weight on disk.
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


def _is_weight_shard_index(name: str) -> bool:
    """True if *name* is a weight-shard index sidecar: the canonical
    ``model.safetensors.index.json`` / ``pytorch_model.bin.index.json`` AND the variant form
    ``model.safetensors.index.fp16.json`` (transformers' ``_add_variant`` inserts the variant token
    before the trailing ``.json``). A plain ``*.safetensors.index.json`` suffix test misses the
    variant form, leaving its listed shards unvalidated."""
    return name.endswith(".json") and (".safetensors.index." in name or ".bin.index." in name)


def _is_canonical_weight_shard_index(name: str) -> bool:
    """True only for the CANONICAL (non-variant) shard index a default in-process load probes:
    ``model.safetensors.index.json`` / ``pytorch_model.bin.index.json`` (any stem). A variant form
    such as ``model.safetensors.index.fp16.json`` ends in ``.index.fp16.json`` and is rejected: the
    fallback wrapper takes no variant parameter, so a default ``from_pretrained`` reads the canonical
    index, and a variant-only cache must NOT satisfy the canonical fast path (its canonical weights
    are still missing -- skipping the child there would reintroduce the unprotected in-process fetch
    this gate prevents)."""
    return name.endswith(".safetensors.index.json") or name.endswith(".bin.index.json")


def _weight_shard_index_complete(index_path: Path) -> bool:
    """True only if every shard a HF weight index (``model.safetensors.index.json`` /
    ``pytorch_model.bin.index.json``) lists is present next to the index.

    Fail-CLOSED: an unreadable / truncated index, a non-dict payload, a missing or non-dict
    ``weight_map``, or an empty shard set all return False. This function feeds the fast-path
    completeness gate, where a malformed index proves nothing -- treating it as complete would let
    the in-process load skip the protective child and then fail (or fetch over Xet) on a truncated
    index, so the safe direction is to defer such an index to the watched ``snapshot_download``
    child. Only an index whose every listed shard is demonstrably on disk returns True."""
    import json

    try:
        with open(index_path, "r", encoding = "utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return False
    weight_map = data.get("weight_map") if isinstance(data, dict) else None
    if not isinstance(weight_map, dict):
        return False
    # weight_map values are filenames relative to the index file's own directory. They come from
    # arbitrary JSON: a non-string (e.g. list/dict) value is both unhashable -- so it would break
    # set() -- and invalid for ``base / shard``, so filter to strings BEFORE de-duplicating rather
    # than crash.
    shards = {s for s in weight_map.values() if isinstance(s, str)}
    if not shards:
        return False  # an empty / all-non-string weight_map cannot prove a complete shard set
    base = index_path.parent
    for shard in shards:
        try:
            if not (base / shard).exists():
                return False
        except OSError:
            return False
    return True


# ---------------------------------------------------------------------------
# Pattern helpers (kept small: normalization + glob detection + HF filtering)
# ---------------------------------------------------------------------------

_GLOB_CHARS = ("*", "?", "[")


def _has_glob(text: str) -> bool:
    # A trailing-slash directory pattern ("unet/", "checkpoint-10/") is NOT an exact filename:
    # Hugging Face's filter_repo_objects expands it to match everything under that directory (as
    # if "unet/*"). Treat it as a wildcard so the strict exact-name checks do not look for a
    # literal "unet/" entry and wrongly reject a fully cached directory / component download.
    return text.endswith("/") or any(ch in text for ch in _GLOB_CHARS)


def _as_pattern_list(patterns: "Optional[object]") -> "Optional[list]":
    """Normalize an allow / ignore pattern argument to a list. Hugging Face accepts a bare
    ``str`` as well as a list, and iterating the ``str`` form would walk it character by
    character (so ``"checkpoint-10/*"`` would never match), misclassifying the request."""
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


def _broken_symlink_rel_paths(snapshot_dir: Path) -> list:
    """Repo-relative posix paths of every dangling symlink in *snapshot_dir* -- a referenced file
    whose blob is missing or still an ``.incomplete`` partial (an interrupted download). Empty when
    none. Lets the broken-symlink check scope the interrupted-download signal to the files a request
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


def snapshot_has_requested_broken_symlinks(
    snapshot_dir: Path,
    *,
    allow_patterns: "Optional[object]" = None,
    ignore_patterns: "Optional[object]" = None,
    repo_type: "Optional[str]" = "model",
) -> bool:
    """True iff a dangling symlink in *snapshot_dir* is for a file the request actually SELECTS.

    A dangling symlink marks an interrupted download, but for a scoped request only one for a
    requested file should reject the snapshot: a dangling root ``model.safetensors`` left by an
    earlier interrupted pull must not fail a weightless ``allow_patterns=["config.json"]`` request
    whose config is on disk. The allow / ignore filter mirrors ``snapshot_download`` selection, so a
    dangle for an excluded file does not reject the cache. (``repo_type`` is accepted for signature
    compatibility; the scoping is now purely the allow/ignore filter.)"""
    allow_patterns = _as_pattern_list(allow_patterns)
    ignore_patterns = _as_pattern_list(ignore_patterns)
    broken = _broken_symlink_rel_paths(snapshot_dir)
    if not broken:
        return False
    return bool(_filter_paths(broken, allow_patterns, ignore_patterns))


# ---------------------------------------------------------------------------
# The conservative fast-path completeness gate
# ---------------------------------------------------------------------------

# Canonical root weight filenames an in-process model load reads. Used to prove a warm cache (the
# file or its shard index is present).
_CANONICAL_SINGLE_WEIGHTS = ("model.safetensors", "pytorch_model.bin")


def _ignore_strips_all_weights(ignore_patterns: "list") -> bool:
    """True iff the ignore set provably excludes EVERY weight format: for each weight suffix there is
    a pattern matching a representative filename with that suffix. Only then is an ignore-only request
    weightless. A partial strip -- only some suffixes, or only the canonical ``model.safetensors`` /
    ``pytorch_model.bin`` names while a variant (``model.fp16.safetensors``) or an other-format weight
    (``model.gguf``, a ``*.pt`` checkpoint) survives -- is NOT weightless, so the request reads as
    weight-bearing (conservative: never under-classify a request that could still pull a weight, which
    would let the fast path skip the protective child on a config-only cache and hang on Xet)."""
    for suffix in _WEIGHT_FILE_SUFFIXES:
        probe = "weight" + suffix
        if not any(isinstance(p, str) and fnmatch.fnmatchcase(probe, p) for p in ignore_patterns):
            return False
    return True


# Representative weight filenames a glob allow pattern is probed against (via fnmatch). A glob that
# matches one of these can select a weight; one that matches none (``tokenizer*``, ``*.json``) is
# weightless. Covers the canonical / variant / sharded / adapter / diffusers / mistral-consolidated
# and the non-safetensors weight formats so a real weight glob is never under-classified.
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

# Snapshot subdirectories that hold only metadata / config (never a loadable model weight), so a
# trailing-slash directory pattern scoped to one of them (``allow_patterns=['tokenizer/']``) is
# weightless. Any OTHER directory pattern stays conservatively weight-bearing: a component dir
# (``unet/``, ``vae/``) or a training-checkpoint dir (``checkpoint-10/``) can hold a weight, so the
# fast path must not skip the child on it.
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
    """Whether a single allow pattern could select a model weight file.

    - a non-string (unknown shape) -> conservative True;
    - a bare directory pattern (``unet/``) -> True (expands to everything under it, incl. weights);
    - a basename ending in a weight suffix (``*.safetensors``, ``model.gguf``) -> True;
    - a glob basename (``model.?afetensors``, ``model.[sp]afetensors``, ``*``) -> True iff it matches a
      representative weight name in ``_WEIGHT_PATTERN_PROBES`` -- so ``tokenizer*`` / ``*.json`` read
      weightless and keep their offline short-circuit, while ``model.?afetensors`` / ``unet/*`` are
      weight-bearing;
    - a concrete non-weight name (``config.json``, ``tokenizer.model``) -> False.

    A glob is matched on its basename so ``checkpoint-*/model.?afetensors`` is still recognized. Both
    directions are bounded: a false weight-bearing only makes the pre-download gate spawn the cheap
    child; a false weightless is avoided for every standard weight name by the probe set."""
    if not isinstance(pattern, str):
        return True  # unknown shape -> conservative
    if pattern.endswith("/"):
        # A bare directory pattern expands to everything under it. A known metadata dir holds no
        # weight (so it stays weightless and keeps its offline short-circuit); any other dir could.
        dir_name = pattern.rstrip("/").rsplit("/", 1)[-1].lower()
        return dir_name not in _NON_WEIGHT_DIRS
    base = pattern.rsplit("/", 1)[-1]
    if base.endswith(_WEIGHT_FILE_SUFFIXES):
        return True  # a concrete or wildcard-stem weight suffix
    if any(ch in base for ch in _GLOB_CHARS):
        # A glob basename selects a weight only if it can actually match a weight filename. This keeps
        # tokenizer / config globs weightless while catching single-char (?) and class ([]) globs.
        return any(fnmatch.fnmatchcase(probe, base) for probe in _WEIGHT_PATTERN_PROBES)
    return False


def request_can_include_weights(
    allow_patterns: "Optional[object]" = None, ignore_patterns: "Optional[object]" = None
) -> bool:
    """Whether a request restricted by *allow_patterns* / *ignore_patterns* can still include a model
    weight. Used to pick the weight-requiring vs weightless branch of the acceptance check.

    Conservative by design: when uncertain it returns True (treat the request as weight-bearing), so
    the acceptance check requires a weight and never short-circuits a config-only cache for a real
    weight load. It returns False only when the request is clearly weightless (a tokenizer / config
    allow list that matches no weight name, or an ignore list that drops every weight format), which
    preserves the offline short-circuit for a genuine tokenizer-only warm."""
    allow_patterns = _as_pattern_list(allow_patterns)
    ignore_patterns = _as_pattern_list(ignore_patterns)
    if allow_patterns is None and ignore_patterns is None:
        return True
    if allow_patterns is None:
        # Ignore-only request: weight-bearing unless the ignore list strips every weight format.
        return not _ignore_strips_all_weights(ignore_patterns or [])
    if not allow_patterns:
        # allow_patterns=[] selects nothing -> no weight (HF filter selects no objects).
        return False
    # An allow list includes weights iff SOME pattern could select a weight (wildcard basename,
    # weight-suffix basename, or a bare directory pattern). A list of only concrete non-weight names
    # (a tokenizer / config warm) is weightless and keeps its offline short-circuit.
    return any(_pattern_can_select_weight(pat) for pat in allow_patterns)


def _canonical_root_weights_complete(
    snapshot_dir: Path, entries: list, ignore_patterns: "Optional[list]" = None
) -> bool:
    """True iff the snapshot holds a complete canonical ROOT model weight set: a root
    ``model.safetensors`` / ``pytorch_model.bin`` single file, OR a root weight-shard index whose
    every listed shard is present. Numbered shard files without a valid index, or weights that live
    only in a subfolder, do NOT count -- those are deferred to the watched child.

    A root weight (or weight-shard index) whose FORMAT the request's ignore filter drops does NOT
    count: a stale ``pytorch_model.bin`` under ``ignore=['*.bin']`` is not proof that the
    safetensors weights an in-process load (e.g. ``use_safetensors=True``) will actually read are on
    disk, so it must not let the fast path skip the protective child and then hang fetching the real
    weight over Xet. The surviving-format check uses a representative weight name per format, so a
    ``*.bin`` ignore also discards a ``pytorch_model.bin.index.json`` (whose ``.json`` sidecar name
    would otherwise slip past the filter)."""
    root_files: set = set()
    root_indices: list = []
    for entry in entries:
        try:
            rel = entry.relative_to(snapshot_dir).as_posix()
        except ValueError:
            rel = entry.name
        if "/" in rel:
            continue  # a bare from_pretrained reads ROOT files only
        # Only the CANONICAL (non-variant) index counts here: a default load probes
        # model.safetensors.index.json, not a variant like model.safetensors.index.fp16.json.
        if _is_canonical_weight_shard_index(entry.name):
            if _safe_is_file(entry):
                root_indices.append(entry)
        elif _safe_is_file(entry):
            root_files.add(entry.name)

    def _format_kept(weight_name: str) -> bool:
        # The weight format an in-process load reads from *weight_name* must survive the request's
        # ignore filter; otherwise the file is a stale artifact for an excluded format and proves
        # nothing about what the load will fetch.
        if not ignore_patterns:
            return True
        return bool(_filter_paths([weight_name], None, ignore_patterns))

    # Sharded: a canonical root index whose format is kept and whose every listed shard is on disk.
    for index_entry in root_indices:
        fmt_probe = (
            "model.safetensors"
            if ".safetensors.index." in index_entry.name
            else "pytorch_model.bin"
        )
        if _format_kept(fmt_probe) and _weight_shard_index_complete(index_entry):
            return True
    # Single-file canonical weight (the file itself must survive the ignore filter).
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
    """Conservative fast-path gate: True only when *snapshot_dir* is an unambiguously complete
    canonical ROOT model cache, so an in-process load will not fetch any weight.

    This is intentionally NOT an authoritative snapshot verifier. It returns True only for:
      - an UNPATTERNED request (allow_patterns is None; ignore_patterns are fine),
      - that is not a diffusers pipeline (no root ``model_index.json``),
      - with no dangling symlink (interrupted blob),
      - whose canonical root weights are present (single file, or a shard index with every shard).
    Every other layout -- variants, diffusers, datasets, any allow pattern, sharded weights without
    an index -- returns False, deferring to the watched ``snapshot_download`` child (the authoritative
    manifest compare + resume). A false True risks a silent un-killable Xet fetch during the in-process
    load; a false False only spawns the cheap child. ``require_named_weights`` is accepted for signature
    compatibility (a named-weight request is non-trivially patterned and so is never fast-pathed here).

    ``ignore_patterns`` need no eligibility gate: the canonical-weight presence check below verifies
    what the in-process load actually reads (root ``model.safetensors`` / ``pytorch_model.bin`` or a
    complete shard index) is on disk, so an ignore that dropped some weight format (the common
    ``*.onnx`` / ``*.gguf`` / ``*.pt`` / ``*.bin`` prefetch ignores, or the subdir ``*/*.safetensors``
    drops) cannot make an incomplete cache read complete -- the surviving canonical weight is what is
    checked. This keeps the common warm ``from_pretrained`` cache fast-path eligible."""
    allow_patterns = _as_pattern_list(allow_patterns)
    ignore_patterns = _as_pattern_list(ignore_patterns)
    # 1. Only an UNPATTERNED request is eligible. Any allow list scopes the on-disk set to a subset
    #    whose relationship to the in-process load is not locally provable -> defer to the child.
    if allow_patterns is not None:
        return False
    try:
        entries = list(snapshot_dir.rglob("*"))
    except OSError:
        return False
    # 2. A diffusers pipeline (root model_index.json) needs component-completeness reasoning we do
    #    not fast-path -> defer to the child.
    if _safe_is_file(snapshot_dir / "model_index.json"):
        return False
    # 3. A dangling symlink = an interrupted blob (missing or still .incomplete) -> not complete.
    if snapshot_dir_has_broken_symlinks(snapshot_dir):
        return False
    # 4. Canonical root weights present and complete (a weight whose format the request ignores
    #    does not count -- see _canonical_root_weights_complete).
    return _canonical_root_weights_complete(snapshot_dir, entries, ignore_patterns)


# A canonical numbered weight shard at the snapshot root: the shard index sits IMMEDIATELY before the
# extension (no variant token), so ``model-00001-of-00002.safetensors`` matches but the variant
# ``model-00001-of-00002.fp16.safetensors`` does NOT.
_CANONICAL_ROOT_SHARD_RE = re.compile(
    r"^(?:model|pytorch_model)-\d{5}-of-\d{5}\.(?:safetensors|bin)$"
)


def _has_incomplete_canonical_root_shards(snapshot_dir: Path) -> bool:
    """True when the snapshot root holds canonical numbered weight shards
    (``model-00001-of-00002.safetensors`` / ``pytorch_model-...bin``) but is NOT a complete canonical
    model -- the shard index is missing or a listed shard is absent.

    Such a loose-shard layout is a stale / interrupted download: a default in-process load cannot read
    bare numbered shards without their index and would fetch the rest over un-killable Xet, so the
    post-download acceptance check rejects it and retries over HTTP. Variant shards
    (``model-...fp16.safetensors``) are intentionally excluded -- they never satisfy a default load, so
    a variant-only repo must not be force-failed here (it simply defers, like any non-canonical warm)."""
    try:
        names = [entry.name for entry in snapshot_dir.iterdir()]
    except OSError:
        return False
    if not any(_CANONICAL_ROOT_SHARD_RE.match(name) for name in names):
        return False
    # Canonical shards exist but no complete single-file / indexed canonical set covers them.
    return not snapshot_dir_is_complete(snapshot_dir)


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
