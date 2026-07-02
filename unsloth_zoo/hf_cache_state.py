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
    (``model.safetensors.index.json`` / ``pytorch_model.bin.index.json``). Exact names only: an
    ``adapter_model.safetensors.index.json`` (or a variant ``...index.fp16.json``) is rejected, so a
    sharded-adapter-only / variant-only cache does not satisfy the canonical fast path (its base
    canonical weights are still missing -> the load would fetch them over un-killable Xet)."""
    return name in ("model.safetensors.index.json", "pytorch_model.bin.index.json")


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
    values = list(weight_map.values())
    # A non-string shard value is a malformed index transformers cannot load; fail CLOSED (defer to the
    # watched child) rather than silently dropping the bad entry and reading the remaining shards as a
    # complete set.
    if not values or not all(isinstance(s, str) for s in values):
        return False
    shards = set(values)
    base = index_path.parent
    for shard in shards:
        # A well-formed HF index lists a relative shard basename. Reject an absolute / parent-escaping
        # value (a malformed or crafted index) rather than let ``base / shard`` resolve to an unrelated
        # existing file OUTSIDE the snapshot and read as "present".
        if shard.startswith(("/", "\\")) or ".." in shard.replace("\\", "/").split("/"):
            return False
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
    # An allow that can reach a weight can still be left weightless by the ignore filter: allow=["*"] +
    # ignore=[every weight suffix], OR a subdir warm allow=["unet/*"] that ignores every weight suffix to
    # fetch only that subdir's metadata / configs. Apply HF's allow-then-ignore semantics to representative
    # weight probes at the ROOT and UNDER each subdir-scoped allow, so a genuinely weightless request is not
    # required to hold a weight (which would false-reject a complete metadata-only subset after both
    # transports). A subdir allow that keeps its weight suffixes still matches a subdir probe and stays
    # weight-bearing.
    if ignore_patterns:
        probes = list(_WEIGHT_PATTERN_PROBES)
        for pat in allow_patterns:
            if isinstance(pat, str) and "/" in pat:
                head = pat.rsplit("/", 1)[0]
                probes.extend(f"{head}/{name}" for name in _WEIGHT_PATTERN_PROBES)
        if not _filter_paths(probes, allow_patterns, ignore_patterns):
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

    st_index = next((e for e in root_indices if ".safetensors.index." in e.name), None)
    bin_index = next((e for e in root_indices if ".bin.index." in e.name), None)
    # transformers' local weight-file precedence, mirrored exactly: a single model.safetensors is probed
    # BEFORE the safetensors index, safetensors before the .bin single, and the .bin single before the
    # .bin index. So a complete single weight is never masked by a co-resident stale index, and an
    # incomplete PREFERRED (safetensors) index is breakage a complete .bin must not mask (transformers
    # takes the safetensors-index branch and does not fall back to .bin). A format the ignore filter
    # drops is skipped so the next format the load actually reads is judged.
    if "model.safetensors" in root_files and _format_kept("model.safetensors"):
        return True
    if st_index is not None and _format_kept("model.safetensors"):
        return _weight_shard_index_complete(st_index)
    if "pytorch_model.bin" in root_files and _format_kept("pytorch_model.bin"):
        return True
    if bin_index is not None and _format_kept("pytorch_model.bin"):
        return _weight_shard_index_complete(bin_index)
    return False


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


def _has_incomplete_canonical_root_shards(
    snapshot_dir: Path, *, ignore_patterns: "Optional[object]" = None
) -> bool:
    """True when the root holds canonical numbered shards but is NOT a complete canonical model (index
    missing or a shard absent) for the format the request READS -- a stale interrupted download a
    default load cannot read, so the post-download check rejects it and retries over HTTP. The request's
    ignore filter is applied, so a complete safetensors set does not mask an incomplete ``.bin`` set the
    load reads under ``ignore=['*.safetensors']``. Variant shards are excluded (their names carry a
    ``.<variant>-`` infix), so a variant-only repo is not force-failed here."""
    try:
        names = [entry.name for entry in snapshot_dir.iterdir()]
    except OSError:
        return False
    # Canonical shard evidence = a numbered shard FILE, or a canonical shard INDEX. An index-only
    # partial (index present, no shards yet) is still an incomplete sharded checkpoint the load would
    # finish over Xet, so it must be caught here even before any shard file exists.
    has_shard_evidence = (
        any(_CANONICAL_ROOT_SHARD_RE.match(name) for name in names)
        or any(_is_canonical_weight_shard_index(name) for name in names)
    )
    if not has_shard_evidence:
        return False
    return not snapshot_dir_is_complete(snapshot_dir, ignore_patterns = ignore_patterns)


def _has_incomplete_variant_root_shards(
    snapshot_dir: Path, variant: str, *, ignore_patterns: "Optional[object]" = None
) -> bool:
    """True when the ROOT variant weight the load READS is an incomplete sharded set. transformers writes
    a sharded variant weight with a ``.<variant>-`` shard infix and its index as
    ``model.safetensors.index.<variant>.json`` (a ``.<variant>.`` infix before ``.json``); a single-file
    variant is ``model.<variant>.safetensors``. Incomplete means: a present variant shard INDEX whose
    listed shards are not all present (an index-only partial with no shard files counts), OR variant shard
    FILES with no complete index.

    The request's ignore filter is applied so a variant weight in an ignored format is not the read
    format, and safetensors is treated as read BEFORE bin (transformers' probe order): a present-but-
    incomplete variant safetensors index is breakage even with a complete variant bin. Positive-evidence:
    a single-file variant or a complete variant shard set returns False, so a complete or single-file
    variant download is never rejected. Only the ROOT ``model`` / ``pytorch_model`` variant weight is
    considered: a co-resident stale ``adapter_model`` variant index / shard set (which a default variant
    model load does not read) must not force-fail a complete model variant."""
    dot_infix = f".{variant}."     # variant index (model.safetensors.index.<variant>.json) or single file
    dash_infix = f".{variant}-"    # a sharded variant weight (model.<variant>-00001-of-00002.safetensors)
    ignore_patterns = _as_pattern_list(ignore_patterns)

    def _format_kept(weight_name: str) -> bool:
        # The format a load reads from *weight_name* must survive the ignore filter, else the file is a
        # stale artifact for an excluded format the load does not read.
        if not ignore_patterns:
            return True
        return bool(_filter_paths([weight_name], None, ignore_patterns))

    try:
        entries = list(snapshot_dir.iterdir())
    except OSError:
        return False
    st_index_incomplete = None   # tri-state: None absent, else present & (in)complete
    bin_index_incomplete = None
    has_st_shard = has_bin_shard = False
    has_single_st = has_single_bin = False
    for entry in entries:
        name = entry.name
        # Restrict to the ROOT model index (model.safetensors.index.<variant>.json /
        # pytorch_model.bin.index.<variant>.json); an adapter_model / other non-model variant index the
        # default load does not read is skipped so its incompleteness cannot force-fail the model variant.
        if dot_infix in name and _ROOT_MODEL_SHARD_INDEX_RE.match(name):
            is_safetensors = ".safetensors.index." in name
            fmt_probe = (
                f"model.{variant}.safetensors" if is_safetensors else f"pytorch_model.{variant}.bin"
            )
            if not _format_kept(fmt_probe):
                continue  # this format is ignored -> the load does not read it
            incomplete = not (_safe_is_file(entry) and _weight_shard_index_complete(entry))
            if is_safetensors:
                st_index_incomplete = incomplete
            else:
                bin_index_incomplete = incomplete
        elif dash_infix in name and _ROOT_MODEL_VARIANT_WEIGHT_RE.match(name):
            if _safe_is_file(entry) and _format_kept(name):
                if name.endswith(".safetensors"):
                    has_st_shard = True
                else:
                    has_bin_shard = True
        elif dot_infix in name and _ROOT_MODEL_VARIANT_WEIGHT_RE.match(name):
            # a single-file ROOT model variant weight (model.<variant>.safetensors / .bin).
            if _safe_is_file(entry) and _format_kept(name):
                if name.endswith(".safetensors"):
                    has_single_st = True
                else:
                    has_single_bin = True
    # transformers' local precedence, mirrored: a single-file model.<variant>.safetensors is probed
    # BEFORE the safetensors index, safetensors before .bin, and the single .bin before the .bin index.
    # So a complete single-file variant is never masked by a co-resident stale index (that would force a
    # spurious HTTP retry and DownloadStallError on a usable cache), and an incomplete PREFERRED
    # (safetensors) index is still breakage a complete .bin must not mask.
    if has_single_st:
        return False  # a complete single-file safetensors variant, probed before the index
    if st_index_incomplete is not None:
        return st_index_incomplete
    if has_st_shard:
        return True   # variant safetensors shard files with no index -> incomplete
    if has_single_bin:
        return False  # a complete single-file bin variant, probed before the .bin index
    if bin_index_incomplete is not None:
        return bin_index_incomplete
    if has_bin_shard:
        return True   # variant bin shard files with no index -> incomplete
    return False


_VARIANT_SHARD_INDEX_RE = re.compile(r"\.(?:safetensors|bin)\.index\.([^.]+)\.json$")

# The ROOT canonical / variant MODEL shard index (owned by the canonical / variant root checks):
# model.safetensors.index.json, pytorch_model.bin.index.json, and their variant forms.
_ROOT_MODEL_SHARD_INDEX_RE = re.compile(
    r"^(?:model\.safetensors|pytorch_model\.bin)\.index(?:\.[^.]+)?\.json$"
)

# A ROOT model VARIANT weight (single or sharded): the variant token sits between the model / pytorch_model
# base and the extension / shard suffix (model.fp16.safetensors, pytorch_model.fp16-00001-of-00002.bin).
# Excludes a PEFT adapter (adapter_model.<variant>.*) the default variant model load does not read.
_ROOT_MODEL_VARIANT_WEIGHT_RE = re.compile(
    r"^(?:model|pytorch_model)\.[^.]+(?:-\d{5}-of-\d{5})?\.(?:safetensors|bin)$"
)


def _index_variant_token(name: str) -> "Optional[str]":
    """The variant token of a weight-shard INDEX basename, or None for the canonical (non-variant) form.
    ``model.safetensors.index.json`` -> None; ``model.safetensors.index.fp16.json`` -> ``"fp16"``. Lets
    the selected-index check read only the indices a load reads (a variant load reads variant indices, a
    plain load reads canonical ones)."""
    if name.endswith(".safetensors.index.json") or name.endswith(".bin.index.json"):
        return None
    m = _VARIANT_SHARD_INDEX_RE.search(name)
    return m.group(1) if m else None


def _index_shard_rel_paths(index_path: Path, dir_rel: str) -> "Optional[list]":
    """The snapshot-relative posix paths of the shards a weight index lists, or None if the index is
    unreadable / malformed -- mirrors the fail-CLOSED rules of ``_weight_shard_index_complete`` (a
    non-dict payload or ``weight_map``, an empty shard set, or a non-string / absolute / parent-escaping
    shard value all return None). *dir_rel* is the index's snapshot-relative directory ("" at root), so a
    listed basename is joined back to a full repo-relative path for the request filter."""
    import json

    try:
        with open(index_path, "r", encoding = "utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    weight_map = data.get("weight_map") if isinstance(data, dict) else None
    if not isinstance(weight_map, dict):
        return None
    values = list(weight_map.values())
    if not values or not all(isinstance(s, str) for s in values):
        return None
    prefix = f"{dir_rel}/" if dir_rel else ""
    out: list = []
    for shard in set(values):
        if shard.startswith(("/", "\\")) or ".." in shard.replace("\\", "/").split("/"):
            return None
        out.append(f"{prefix}{shard}")
    return out


def _selected_shard_index_incomplete(
    snapshot_dir: Path, *, allow_patterns: "Optional[object]", ignore_patterns: "Optional[object]",
    variant: "Optional[str]",
) -> bool:
    """True when a weight-shard INDEX the in-process load READS -- a sharded ADAPTER or a component
    SUBFOLDER set that the canonical / variant ROOT-model checks do not cover -- lists a shard that is
    absent (or the index is malformed). Scoped to the request so a complete download is never
    false-rejected:

    - variant: a variant load reads only variant indices (token == variant); a plain load reads only
      canonical (token is None) indices.
    - allow / ignore: an index is read only when its listed shards survive the request filter.
    - precedence: within a directory transformers reads safetensors before bin, so when both a
      safetensors and a bin index are selected only the safetensors set's completeness is required.

    Also rejects a SELECTED numbered shard FILE (adapter_model-00001-of-00002.safetensors,
    unet/diffusion_pytorch_model-00001-of-00002.safetensors) whose directory has NO index of the read
    format: the load enumerates a sharded weight through its index, so a shard set without one is
    incomplete and would fetch the index and remaining shards over Xet.

    The ROOT canonical / variant MODEL shard set is skipped -- ``_has_incomplete_canonical_root_shards`` /
    ``_has_incomplete_variant_root_shards`` own it (with their own precedence handling)."""
    allow_patterns = _as_pattern_list(allow_patterns)
    ignore_patterns = _as_pattern_list(ignore_patterns)
    want_variant = variant or None
    if want_variant is None:
        shard_file_re = re.compile(r"^[^.]+-\d{5}-of-\d{5}\.(?:safetensors|bin)$")
    else:
        v = re.escape(want_variant)
        shard_file_re = re.compile(rf"^[^.]+\.{v}-\d{{5}}-of-\d{{5}}\.(?:safetensors|bin)$")
    try:
        entries = list(snapshot_dir.rglob("*"))
    except OSError:
        return False
    per_dir: dict = {}       # dir_rel -> {"safetensors": [shard_rels, ...], "bin": [...]} (from indices)
    index_fmts: dict = {}    # dir_rel -> {fmt} an index of the read variant is present (non-root-model)
    shard_fmts: dict = {}    # dir_rel -> {fmt} a SELECTED numbered shard file is present (non-root-model)
    for entry in entries:
        name = entry.name
        if not _safe_is_file(entry):
            continue
        try:
            rel = entry.relative_to(snapshot_dir).as_posix()
        except ValueError:
            continue
        dir_rel = rel.rsplit("/", 1)[0] if "/" in rel else ""
        if _is_weight_shard_index(name):
            if _index_variant_token(name) != want_variant:
                continue  # a wrong-variant index the load does not read
            if dir_rel == "" and _ROOT_MODEL_SHARD_INDEX_RE.match(name):
                continue  # the ROOT model index -- owned by the canonical / variant root checks
            fmt = "safetensors" if ".safetensors.index." in name else "bin"
            index_fmts.setdefault(dir_rel, set()).add(fmt)
            shard_rels = _index_shard_rel_paths(entry, dir_rel)
            if shard_rels is None:
                # A malformed / non-string index. Defer to the watched child only when the REQUEST
                # selects this index (the load would read it to enumerate its shards); a co-resident
                # stale malformed index the request does NOT select (a leftover adapter index under a
                # base ['model*'] / subfolder warm) is not read, so it must not force a spurious retry.
                if _filter_paths([rel], allow_patterns, ignore_patterns):
                    return True
                continue
            if not _filter_paths(shard_rels, allow_patterns, ignore_patterns):
                continue  # the load does not read this set (out of scope / ignored format)
            per_dir.setdefault(dir_rel, {}).setdefault(fmt, []).append(shard_rels)
        elif shard_file_re.match(name):
            # a numbered weight shard FILE of the read variant. Skip the ROOT model shard set (owned by
            # the canonical / variant root-shard checks) and any training-checkpoint subtree.
            if dir_rel == "" and (
                (want_variant is None and _CANONICAL_ROOT_SHARD_RE.match(name))
                or (want_variant is not None and _ROOT_MODEL_VARIANT_WEIGHT_RE.match(name))
            ):
                continue
            if any(_CHECKPOINT_DIR_RE.match(p) for p in rel.split("/")[:-1]):
                continue
            if not _filter_paths([rel], allow_patterns, ignore_patterns):
                continue  # the load does not read this shard (out of scope / ignored format)
            fmt = "safetensors" if name.endswith(".safetensors") else "bin"
            shard_fmts.setdefault(dir_rel, set()).add(fmt)
    for by_fmt in per_dir.values():
        # safetensors read before bin: require only the preferred format present in this directory.
        for shard_rels in by_fmt.get("safetensors") or by_fmt.get("bin") or []:
            for shard in shard_rels:
                try:
                    if not (snapshot_dir / shard).exists():
                        return True
                except OSError:
                    return True
    for dir_rel, fmts in shard_fmts.items():
        # a numbered shard of the read (preferred) format with NO index in its directory: the load cannot
        # enumerate the set and would fetch the index + remaining shards over Xet.
        preferred = "safetensors" if "safetensors" in fmts else "bin"
        if preferred not in index_fmts.get(dir_rel, set()):
            return True
    return False


# A training-checkpoint subdir (checkpoint-500/, checkpoint_7/): its weights are never read as diffusers
# pipeline COMPONENTS, so an incomplete shard index under it must not force-fail a complete pipeline.
_CHECKPOINT_DIR_RE = re.compile(r"^checkpoint[-_]\d+$")


def _diffusers_declared_components(snapshot_dir: Path) -> "Optional[set]":
    """The component subfolder names a diffusers ``model_index.json`` declares (top-level keys mapping to
    a ``[library, class]`` list; ``_``-prefixed metadata keys excluded). None when the file is absent /
    unreadable / malformed, so the caller falls back to treating every subfolder as a component (fail
    OPEN, preserving hang protection). Scopes the component shard check to what the pipeline actually
    reads, so a co-resident stale UNDECLARED subtree (a leftover adapter / controlnet dir the
    ``DiffusionPipeline`` load never reads) cannot force-fail a complete pipeline download."""
    import json

    try:
        with open(snapshot_dir / "model_index.json", "r", encoding = "utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    components = {
        key for key, value in data.items()
        if not key.startswith("_") and isinstance(value, (list, tuple))
    }
    # A real pipeline always declares components; an empty / all-metadata model_index.json is degenerate
    # or malformed -> fail OPEN (None) so the caller checks every subfolder, preserving hang protection.
    return components or None


def _diffusers_component_shards_incomplete(
    snapshot_dir: Path, *, variant: "Optional[str]" = None,
    ignore_patterns: "Optional[object]" = None,
) -> bool:
    """True when a diffusers pipeline COMPONENT subfolder (unet/, vae/, text_encoder/, ...) holds a
    weight-shard INDEX of the read variant that lists a shard that is absent (or the index is malformed)
    -- an interrupted component pull the in-process pipeline load would finish over un-killable Xet.

    Scoped so a complete pipeline is never false-rejected: the check is limited to the components
    ``model_index.json`` declares (a stale UNDECLARED subtree the pipeline load never reads is skipped),
    a ROOT index (owned by the canonical / variant root-model checks) and a training-checkpoint subtree
    (checkpoint-N/) are skipped, and the request's ignore filter selects the read format. Per directory,
    safetensors is read before bin, so only the preferred format's set must be complete. A plain load
    reads canonical component indices (token None); a variant load reads variant ones. Also rejects a
    component holding a numbered shard FILE with NO index of the read format (the pipeline cannot
    enumerate the set and would fetch the index + remaining shards over Xet). Positive-evidence: a
    single-file component or a complete component shard set is not flagged, so a complete download passes."""
    want_variant = variant or None
    ignore_patterns = _as_pattern_list(ignore_patterns)
    declared = _diffusers_declared_components(snapshot_dir)
    if want_variant is None:
        shard_file_re = re.compile(r"^[^.]+-\d{5}-of-\d{5}\.(?:safetensors|bin)$")
    else:
        v = re.escape(want_variant)
        shard_file_re = re.compile(rf"^[^.]+\.{v}-\d{{5}}-of-\d{{5}}\.(?:safetensors|bin)$")
    try:
        entries = list(snapshot_dir.rglob("*"))
    except OSError:
        return False
    per_dir: dict = {}
    index_fmts: dict = {}   # component dir_rel -> {fmt} an index of the read variant is present
    shard_fmts: dict = {}   # component dir_rel -> {fmt} a numbered shard file (ignore-kept) is present
    for entry in entries:
        name = entry.name
        if not _safe_is_file(entry):
            continue
        try:
            rel = entry.relative_to(snapshot_dir).as_posix()
        except ValueError:
            continue
        parts = rel.split("/")
        if len(parts) < 2:
            continue  # a ROOT file -- owned by the canonical / variant root-model checks
        if declared is not None and parts[0] not in declared:
            continue  # an UNDECLARED subtree the DiffusionPipeline load does not read
        if any(_CHECKPOINT_DIR_RE.match(p) for p in parts[:-1]):
            continue  # a training-checkpoint subtree, not a pipeline component
        dir_rel = rel.rsplit("/", 1)[0]
        if _is_weight_shard_index(name):
            if _index_variant_token(name) != want_variant:
                continue  # a wrong-variant index the load does not read
            fmt = "safetensors" if ".safetensors.index." in name else "bin"
            index_fmts.setdefault(dir_rel, set()).add(fmt)
            shard_rels = _index_shard_rel_paths(entry, dir_rel)
            if shard_rels is None:
                return True  # a malformed / non-string index -> defer to the watched child
            if not _filter_paths(shard_rels, None, ignore_patterns):
                continue  # the load does not read this set (ignored format)
            per_dir.setdefault(dir_rel, {}).setdefault(fmt, []).append(shard_rels)
        elif shard_file_re.match(name) and _filter_paths([rel], None, ignore_patterns):
            fmt = "safetensors" if name.endswith(".safetensors") else "bin"
            shard_fmts.setdefault(dir_rel, set()).add(fmt)
    for by_fmt in per_dir.values():
        for shard_rels in by_fmt.get("safetensors") or by_fmt.get("bin") or []:
            for shard in shard_rels:
                try:
                    if not (snapshot_dir / shard).exists():
                        return True
                except OSError:
                    return True
    for dir_rel, fmts in shard_fmts.items():
        preferred = "safetensors" if "safetensors" in fmts else "bin"
        if preferred not in index_fmts.get(dir_rel, set()):
            return True  # a component numbered shard with no index of the read format
    return False


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
