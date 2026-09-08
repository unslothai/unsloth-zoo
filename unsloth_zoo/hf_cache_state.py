# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Sparse-aware introspection of the active Hugging Face hub cache.

Reports on-disk bytes (sparse-aware, so a partial Xet / ``hf_transfer`` ``.incomplete`` is not read as
full progress) and whether an ``.incomplete`` partial is present -- the signals the no-progress
watchdog runs on.

``snapshot_dir_is_complete`` is a CONSERVATIVE fast-path gate, not an authoritative verifier: it
returns "complete" only for unambiguous canonical model layouts and defers everything else to the
watched ``snapshot_download`` child. A false "complete" is the only dangerous error (an in-process
load could then fetch a missing weight over un-killable Xet); a false "not complete" only spawns the
cheap child. Only the active ``HF_HUB_CACHE`` root is scanned.
"""

from __future__ import annotations

import fnmatch
import re
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
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
    """``Path.is_dir()`` returning False instead of raising."""
    try:
        return path.is_dir()
    except OSError:
        return False


def _safe_is_file(path: Path) -> bool:
    """``Path.is_file()`` returning False instead of raising."""
    try:
        return path.is_file()
    except OSError:
        return False


def hf_cache_root(*, create: bool = False, cache_dir: "Optional[str | Path]" = None) -> Optional[Path]:
    """The hub cache root to scan, or None if unavailable. *cache_dir* is used verbatim; otherwise
    ``HF_HUB_CACHE`` is read lazily so an import-time redirect is honored."""
    if cache_dir is not None:
        # Match huggingface_hub's ~ expansion; expanduser() raises with no resolvable HOME (container),
        # so fall back to the literal path rather than crash.
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
    # repo_type=None is HF's default "model", so None resolves models--<id> not Nones--<id>.
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
    """Sparse-aware on-disk size: a sparse ``.incomplete`` reports full ``st_size`` while only some
    blocks are allocated, so prefer ``st_blocks``, falling back to ``st_size`` where it is unreported
    (Windows, some network filesystems)."""
    st = path.stat()
    blocks = getattr(st, "st_blocks", None)
    if blocks is not None:
        # Trust st_blocks even at 0: a truncated sparse .incomplete reports full st_size but 0 blocks
        # and must read as 0 bytes (a > 0 guard would fall through to st_size).
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
    """Newest child of ``repo_dir/snapshots`` by mtime (what from_pretrained resolves to), or None."""
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
    """True if *snapshot_dir* holds a dangling symlink (a referenced blob missing or still
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

# Trainer / optimizer state carries weight suffixes but is NOT a loadable weight, so a cache holding
# only these is not a warm model cache.
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
    """True if *name* is a loadable weight: a weight suffix that is not trainer / optimizer state."""
    if not name.endswith(_WEIGHT_FILE_SUFFIXES):
        return False
    lowered = name.lower()
    if lowered in _NON_WEIGHT_BASENAMES:
        return False
    if any(lowered.startswith(prefix) for prefix in _NON_WEIGHT_BASENAME_PREFIXES):
        return False
    return True


def _is_weight_shard_index(name: str) -> bool:
    """True for a weight-shard index sidecar, canonical or variant; a plain suffix test would miss the
    variant form (``model.safetensors.index.fp16.json``)."""
    return name.endswith(".json") and (".safetensors.index." in name or ".bin.index." in name)


def _is_canonical_weight_shard_index(name: str) -> bool:
    """True only for the CANONICAL (non-variant) index a default load probes. Exact names only, so a
    sharded-adapter-only / variant-only cache does not satisfy the canonical fast path (its base
    canonical weights are still missing -> the load would fetch them over un-killable Xet)."""
    return name in ("model.safetensors.index.json", "pytorch_model.bin.index.json")


# The EXACT canonical / single-variant diffusers COMPONENT shard index (anchored, one optional variant
# token), so a stale sidecar like diffusion_pytorch_model.safetensors.index.fp16.extra.json is rejected.
# The base is captured so the ignore-filter probe uses the component's OWN weight name.
_COMPONENT_SHARD_INDEX_RE = re.compile(
    r"^(?P<base>diffusion_pytorch_model|model|pytorch_model)\.(?P<ext>safetensors|bin)"
    r"\.index(?:\.(?P<variant>[^.]+))?\.json$"
)


def _is_canonical_component_shard_index(name: str) -> bool:
    """True only for the EXACT canonical / single-variant diffusers COMPONENT shard index
    (``diffusion_pytorch_model.safetensors.index.json`` / ``...index.fp16.json``, plus the ``model`` /
    ``pytorch_model`` bases). The component analog of ``_is_canonical_weight_shard_index``: lets a component
    sharded with NON-standard shard names be recognized via its index, while a stale malformed sidecar the
    pipeline never probes is rejected."""
    return _COMPONENT_SHARD_INDEX_RE.match(name) is not None


def _component_index_weight_probe(index_name: str, dir_rel: str) -> "Optional[str]":
    """The component-relative weight path a canonical component shard *index* enumerates, preserving the
    component's OWN base (``diffusion_pytorch_model`` / ``model`` / ``pytorch_model``), format, and variant
    token, so the ignore filter is judged on the real weight name -- a component-scoped ignore like
    ``unet/diffusion_pytorch_model*`` matches. None when *index_name* is not such an index."""
    m = _COMPONENT_SHARD_INDEX_RE.match(index_name)
    if m is None:
        return None
    token = f".{m['variant']}" if m["variant"] else ""
    prefix = f"{dir_rel}/" if dir_rel else ""
    return f"{prefix}{m['base']}{token}.{m['ext']}"


def _is_unsafe_shard_ref(shard: str) -> bool:
    """True if a ``weight_map`` value is NOT a safe relative path inside the snapshot (absolute, Windows
    drive-letter, UNC, or ``..``-escaping). Judged under BOTH POSIX and Windows semantics so a crafted
    index is rejected regardless of the running OS (on Windows ``base / "C:\\x"`` resolves OUTSIDE the
    snapshot; ``startswith(("/", "\\"))`` alone misses a drive letter). A well-formed relative basename
    is never rejected."""
    if not shard or shard.startswith(("/", "\\")):
        return True
    win = PureWindowsPath(shard)
    if win.is_absolute() or win.drive or ".." in win.parts:
        return True
    posix = PurePosixPath(shard)
    if posix.is_absolute() or ".." in posix.parts:
        return True
    return False


def _weight_shard_index_complete(index_path: Path) -> bool:
    """True only if every shard a HF weight index lists is present next to it.

    Fail-CLOSED: an unreadable / truncated index, a non-dict payload or ``weight_map``, or an empty
    shard set return False, deferring a malformed index to the watched child rather than letting the
    load skip it and then fail (or fetch over Xet)."""
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
    # A non-string shard value is malformed; fail CLOSED rather than drop it and read the rest as complete.
    if not values or not all(isinstance(s, str) for s in values):
        return False
    shards = set(values)
    base = index_path.parent
    for shard in shards:
        # Reject an unsafe ref rather than let ``base / shard`` resolve to a file OUTSIDE the snapshot.
        if _is_unsafe_shard_ref(shard):
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
    # A trailing-slash dir pattern ("unet/") is a wildcard: HF expands it like "unet/*".
    return text.endswith("/") or any(ch in text for ch in _GLOB_CHARS)


def _as_pattern_list(patterns: "Optional[object]") -> "Optional[list]":
    """Normalize an allow / ignore argument to a list. HF accepts a bare ``str``, which would otherwise
    be iterated character by character."""
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
    Fails OPEN (returns all paths) so a snapshot holding weights is never rejected on an unevaluable
    filter."""
    try:
        from huggingface_hub.utils import filter_repo_objects

        return list(
            filter_repo_objects(
                paths, allow_patterns = allow_patterns, ignore_patterns = ignore_patterns
            )
        )
    except Exception:
        return list(paths)


def _read_format_kept(weight_name: str, ignore_patterns: "Optional[list]") -> bool:
    """Whether a weight of *weight_name*'s format survives the ignore filter -- i.e. the load actually
    reads it, so its presence (or an index enumerating it) is real evidence and not a stale
    excluded-format artifact (a ``pytorch_model.bin`` under ``ignore=['*.bin']``). Shared by the presence
    checks (Invariant A) and the shard-completeness checks (Invariant B) so both judge the ignore filter
    identically. *ignore_patterns* must be a normalized list (see ``_as_pattern_list``) or None."""
    if not ignore_patterns:
        return True
    return bool(_filter_paths([weight_name], None, ignore_patterns))


def _index_weight_probe(index_name: str, variant: "Optional[str]" = None) -> str:
    """The canonical ROOT weight filename whose FORMAT a shard *index* enumerates, used as the ignore
    filter probe: a ``.safetensors.index.`` index -> ``model[.<variant>].safetensors``, any other (``.bin
    .index.``) -> ``pytorch_model[.<variant>].bin``. Centralizes the safetensors-vs-bin split so the
    presence checks (Invariant A) and completeness checks (Invariant B) map an index to the SAME probe."""
    token = f".{variant}" if variant else ""
    if ".safetensors.index." in index_name:
        return f"model{token}.safetensors"
    return f"pytorch_model{token}.bin"


def _broken_symlink_rel_paths(snapshot_dir: Path) -> list:
    """Repo-relative posix paths of every dangling symlink in *snapshot_dir*, so the interrupted-download
    signal can be scoped to the files a request selects."""
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
    """True iff a dangling symlink in *snapshot_dir* is for a file the request SELECTS, so a dangling root
    ``model.safetensors`` does not fail a weightless ``allow=["config.json"]`` request whose config is on
    disk. (*repo_type* kept for signature compatibility.)"""
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
    """True iff the ignore set provably excludes EVERY weight format. A partial strip is NOT weightless
    (a surviving weight could still be pulled), so the request stays weight-bearing (conservative)."""
    for suffix in _WEIGHT_FILE_SUFFIXES:
        probe = "weight" + suffix
        if not any(isinstance(p, str) and fnmatch.fnmatchcase(probe, p) for p in ignore_patterns):
            return False
    return True


# Representative weight names a glob allow pattern is probed against: a glob matching one can select a
# weight; one matching none (``tokenizer*``, ``*.json``) is weightless. Covers canonical / variant /
# sharded / adapter / diffusers / consolidated and the non-safetensors formats.
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
    ``_WEIGHT_PATTERN_PROBES`` name; a concrete non-weight name is weightless. A false weight-bearing only
    spawns the cheap child; the probe set avoids a false weightless on real weights."""
    if not isinstance(pattern, str):
        return True
    if pattern.endswith("/"):
        dir_name = pattern.rstrip("/").rsplit("/", 1)[-1].lower()
        return dir_name not in _NON_WEIGHT_DIRS
    # A pattern scoped under a metadata dir ("tokenizer/*") is weightless like the "tokenizer/" form.
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
    Conservative: True when uncertain; False only for a clearly weightless request (a tokenizer / config
    allow list, an ignore list dropping every weight format, or an allow + ignore pair that strips them
    all), preserving the tokenizer-only short-circuit."""
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
    # An allow that reaches a weight can still be left weightless by the ignore filter (allow=["*"] +
    # ignore=[every weight suffix], or a subdir allow=["unet/*"] ignoring every weight suffix). Apply HF's
    # allow-then-ignore semantics to weight probes at the ROOT and under each subdir-scoped allow, so a
    # genuinely weightless request is not required to hold a weight (a subdir allow keeping its weight
    # suffixes still matches a subdir probe and stays weight-bearing).
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
    snapshot_dir: Path, entries: list, ignore_patterns: "Optional[list]" = None,
    *, prefer_safetensors: bool = False,
) -> bool:
    """True iff the snapshot holds a complete canonical ROOT weight set: a root
    ``model.safetensors`` / ``pytorch_model.bin``, OR a root shard index whose every shard is present.
    Numbered shards without an index, or subfolder-only weights, do NOT count.

    A weight whose FORMAT the ignore filter drops does not count (a stale ``pytorch_model.bin`` under
    ``ignore=['*.bin']`` is no proof the requested safetensors are on disk). The format probe also
    discards a ``pytorch_model.bin.index.json`` whose ``.json`` name would slip the raw filter.

    *prefer_safetensors* is set by the STRICT pre-download gate: a default load probes safetensors
    BEFORE bin, so when safetensors is read (not ignored) a bin-only cache cannot be proven complete
    (the local cache cannot show the preferred safetensors is absent remotely) and skipping the child
    would fetch it over un-killable Xet. So ``.bin`` satisfies the gate only when safetensors is IGNORED.
    The lenient POST path leaves this False: a finished bin-only download is a genuinely bin-only repo
    and must not be false-rejected into a ``DownloadStallError``."""
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

    # Map each present root index to the weight it enumerates, so the format split is single-sourced in
    # _index_weight_probe (each canonical index maps to a distinct format, so no probe collides).
    index_by_read = {_index_weight_probe(e.name): e for e in root_indices}
    st_index = index_by_read.get("model.safetensors")
    bin_index = index_by_read.get("pytorch_model.bin")
    # transformers' local precedence, mirrored: single safetensors before the safetensors index,
    # safetensors before the .bin single, .bin single before the .bin index. So a complete single weight
    # is never masked by a stale index, and an incomplete PREFERRED safetensors index is breakage a
    # complete .bin must not mask. An ignore-dropped format is skipped so the next read format is judged.
    if "model.safetensors" in root_files and _read_format_kept("model.safetensors", ignore_patterns):
        return True
    if st_index is not None and _read_format_kept("model.safetensors", ignore_patterns):
        return _weight_shard_index_complete(st_index)
    if prefer_safetensors and _read_format_kept("model.safetensors", ignore_patterns):
        # STRICT gate: safetensors is preferred but absent, and a bin-only cache cannot prove it absent
        # remotely, so a default load would fetch it over Xet -> defer to the child.
        return False
    if "pytorch_model.bin" in root_files and _read_format_kept("pytorch_model.bin", ignore_patterns):
        return True
    if bin_index is not None and _read_format_kept("pytorch_model.bin", ignore_patterns):
        return _weight_shard_index_complete(bin_index)
    return False


def snapshot_dir_is_complete(
    snapshot_dir: Path,
    *,
    allow_patterns: "Optional[object]" = None,
    ignore_patterns: "Optional[object]" = None,
    require_named_weights: bool = False,
    prefer_safetensors: bool = False,
) -> bool:
    """Conservative fast-path gate: True only for an unambiguously complete canonical ROOT model cache,
    so an in-process load will not fetch a weight. True requires: an UNPATTERNED request, not a diffusers
    pipeline (no root ``model_index.json``), no dangling symlink, and canonical root weights present.
    Everything else defers to the watched child. A false True risks a silent Xet fetch; a false False
    only spawns the cheap child. *require_named_weights* is accepted for signature compatibility (a
    named-weight request is patterned, so never fast-pathed).

    *prefer_safetensors* (set by the strict pre-download gate) rejects a bin-only cache when a default
    load prefers safetensors: the cache cannot prove the preferred file absent remotely, so fast-pathing
    would fetch it over Xet. The POST caller leaves it False so a genuinely bin-only download is accepted.

    *ignore_patterns* need no eligibility gate: the canonical-weight check below is what the load reads,
    so an ignore dropping some format cannot make an incomplete cache read complete."""
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
    return _canonical_root_weights_complete(
        snapshot_dir, entries, ignore_patterns, prefer_safetensors = prefer_safetensors
    )


# sentence-transformers ``modules.json`` module types that ship NO weight (config-only), and those that
# DO ship a weight (a Dense / CNN / LSTM head saving model.safetensors / pytorch_model.bin), matched on
# the final dotted component of the "type".
_ST_WEIGHTLESS_MODULE_TYPES = frozenset({"pooling", "normalize", "layernorm", "dropout"})
_ST_WEIGHTED_MODULE_TYPES = frozenset({"dense", "cnn", "lstm"})


def _sentence_transformers_subfolder_incomplete(
    snapshot_dir: Path, *, prefer_safetensors: bool = False,
    ignore_patterns: "Optional[object]" = None, strict: bool = True,
) -> bool:
    """True when a sentence-transformers ``modules.json`` declares a weight-bearing module in a SUBFOLDER
    (e.g. ``2_Dense``) whose READABLE weight is absent. Such a subfolder weight is read by the ST load but
    is NOT covered by the canonical ROOT weight check, so a partial cache holding the root weight yet
    missing the subfolder weight would be fast-pathed / accepted and then fetch it over un-killable Xet.

    The format preference and ignore filter are honored exactly as the root check: with *prefer_safetensors*
    (the STRICT pre gate) a bin-only subfolder is incomplete when safetensors is the read format; the
    lenient POST path leaves it False so a finished bin-only subfolder is accepted.

    *strict* selects which modules must hold a weight. The PRE gate (strict=True) requires one for every
    module NOT known to be weightless, so an unknown / custom type conservatively defers -- a false
    requirement only spawns the cheap child. The POST path (strict=False) requires one only for a module
    KNOWN to be weight-bearing (dense / cnn / lstm), so an unknown weightless module can never false-reject
    a finished download into a ``DownloadStallError``.

    Present but unreadable ``modules.json`` -> *strict* (defer PRE, do not reject a finished POST download);
    absent -> False (not an ST multi-module repo)."""
    modules_path = snapshot_dir / "modules.json"
    if not _safe_is_file(modules_path):
        return False
    import json

    ignore_patterns = _as_pattern_list(ignore_patterns)
    try:
        with open(modules_path, "r", encoding = "utf-8") as f:
            modules = json.load(f)
    except (OSError, ValueError):
        return strict  # unreadable layout: defer PRE (safe), never reject a finished download POST
    if not isinstance(modules, list):
        return strict
    for module in modules:
        if not isinstance(module, dict):
            continue
        path = module.get("path")
        if not isinstance(path, str) or not path.strip("/"):
            continue  # the root module (path "") -- covered by the canonical root check
        module_type = module.get("type")
        leaf = module_type.rsplit(".", 1)[-1].lower() if isinstance(module_type, str) else ""
        if strict:
            if leaf in _ST_WEIGHTLESS_MODULE_TYPES:
                continue  # a config-only module (pooling / normalize / ...) needs no weight
        elif leaf not in _ST_WEIGHTED_MODULE_TYPES:
            continue  # POST rejects only a KNOWN weight-bearing module (no false-reject on unknowns)
        path = path.strip("/")
        sub = snapshot_dir / path
        try:
            names = {entry.name for entry in sub.iterdir()} if _safe_is_dir(sub) else set()
        except OSError:
            names = set()
        st_read = _read_format_kept(f"{path}/model.safetensors", ignore_patterns)
        bin_read = _read_format_kept(f"{path}/pytorch_model.bin", ignore_patterns)
        st_present = "model.safetensors" in names and st_read
        bin_present = "pytorch_model.bin" in names and bin_read
        if prefer_safetensors and st_read:
            # safetensors is the read format and not ignored: a bin-only subfolder cannot prove the
            # preferred safetensors absent remotely, so the load would fetch it over Xet -> defer.
            if not st_present:
                return True
            continue
        if not (st_present or bin_present):
            return True  # no readable weight of the format the load reads
    return False


# A canonical numbered root shard (no variant token): ``model-00001-of-00002.safetensors`` matches but
# ``model-00001-of-00002.fp16.safetensors`` does not.
_CANONICAL_ROOT_SHARD_RE = re.compile(
    r"^(?:model|pytorch_model)-\d{5}-of-\d{5}\.(?:safetensors|bin)$"
)


def _has_incomplete_canonical_root_shards(
    snapshot_dir: Path, *, ignore_patterns: "Optional[object]" = None
) -> bool:
    """True when the root holds canonical numbered shards but is NOT a complete canonical model (index
    missing or a shard absent) for the format the request READS -- a stale interrupted download the post
    check rejects and retries over HTTP. The request's ignore filter is applied, so a complete
    safetensors set does not mask an incomplete ``.bin`` set read under ``ignore=['*.safetensors']``.
    Variant shards (``.<variant>-`` infix) are excluded, so a variant-only repo is not force-failed."""
    try:
        names = [entry.name for entry in snapshot_dir.iterdir()]
    except OSError:
        return False
    # Shard evidence = a numbered shard FILE or a canonical shard INDEX. An index-only partial (index
    # present, no shards yet) is still incomplete and must be caught before any shard file exists.
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
    """True when the ROOT variant weight the load READS is an incomplete sharded set. Incomplete means:
    a present variant shard INDEX whose listed shards are not all present (an index-only partial counts),
    OR variant shard FILES with no complete index.

    The request's ignore filter is applied, and safetensors is read BEFORE bin (transformers' probe
    order): an incomplete variant safetensors index is breakage even with a complete variant bin.
    Positive-evidence: a single-file variant or a complete variant shard set returns False. Only the ROOT
    ``model`` / ``pytorch_model`` variant weight is considered, so a stale ``adapter_model`` variant set
    (which the default variant model load does not read) must not force-fail a complete model variant."""
    dot_infix = f".{variant}."     # variant index or single file
    dash_infix = f".{variant}-"    # a sharded variant weight (model.<variant>-00001-of-00002.safetensors)
    ignore_patterns = _as_pattern_list(ignore_patterns)

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
        # Restrict to the ROOT model index; an adapter_model / other non-model variant index the default
        # load does not read is skipped so its incompleteness cannot force-fail the model variant.
        if dot_infix in name and _ROOT_MODEL_SHARD_INDEX_RE.match(name):
            probe = _index_weight_probe(name, variant)
            if not _read_format_kept(probe, ignore_patterns):
                continue  # this format is ignored -> the load does not read it
            incomplete = not (_safe_is_file(entry) and _weight_shard_index_complete(entry))
            if probe.endswith(".safetensors"):
                st_index_incomplete = incomplete
            else:
                bin_index_incomplete = incomplete
        elif dash_infix in name and _ROOT_MODEL_VARIANT_WEIGHT_RE.match(name):
            if _safe_is_file(entry) and _read_format_kept(name, ignore_patterns):
                if name.endswith(".safetensors"):
                    has_st_shard = True
                else:
                    has_bin_shard = True
        elif dot_infix in name and _ROOT_MODEL_VARIANT_WEIGHT_RE.match(name):
            # a single-file ROOT model variant weight (model.<variant>.safetensors / .bin).
            if _safe_is_file(entry) and _read_format_kept(name, ignore_patterns):
                if name.endswith(".safetensors"):
                    has_single_st = True
                else:
                    has_single_bin = True
    # transformers' local precedence, mirrored: single safetensors before the safetensors index,
    # safetensors before .bin, single .bin before the .bin index. So a complete single-file variant is
    # never masked by a stale index (which would force a spurious DownloadStallError), and an incomplete
    # PREFERRED safetensors index is still breakage a complete .bin must not mask.
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

# A ROOT model VARIANT weight, single or sharded (model.fp16.safetensors,
# pytorch_model.fp16-00001-of-00002.bin). Excludes a PEFT adapter the variant model load does not read.
_ROOT_MODEL_VARIANT_WEIGHT_RE = re.compile(
    r"^(?:model|pytorch_model)\.[^.]+(?:-\d{5}-of-\d{5})?\.(?:safetensors|bin)$"
)


def _index_variant_token(name: str) -> "Optional[str]":
    """The variant token of a weight-shard INDEX basename, or None for the canonical form
    (``model.safetensors.index.fp16.json`` -> ``"fp16"``). Lets the selected-index check read only the
    indices a load reads (variant load -> variant indices, plain load -> canonical)."""
    if name.endswith(".safetensors.index.json") or name.endswith(".bin.index.json"):
        return None
    m = _VARIANT_SHARD_INDEX_RE.search(name)
    return m.group(1) if m else None


def _index_shard_rel_paths(index_path: Path, dir_rel: str) -> "Optional[list]":
    """Snapshot-relative posix paths of the shards a weight index lists, or None if the index is
    unreadable / malformed (fail-CLOSED, mirroring ``_weight_shard_index_complete``). *dir_rel* is the
    index's snapshot-relative dir ("" at root), so a listed basename is joined back to a full
    repo-relative path for the request filter."""
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
        if _is_unsafe_shard_ref(shard):
            return None
        out.append(f"{prefix}{shard}")
    return out


def _index_shard_probe(index_name: str, dir_rel: str) -> "Optional[str]":
    """A representative numbered-shard path for a malformed weight-shard INDEX (index base + format as a
    first shard, under *dir_rel*), so the scope check judges the request filter on the WEIGHT the load
    reads, not the ``.json`` filename -- ``ignore=['*.bin']`` misses ``pytorch_model.bin.index.json`` but
    the load never reads that ignored-format index. None when the name is not a recognizable shard index."""
    for marker, ext in ((".safetensors.index.", "safetensors"), (".bin.index.", "bin")):
        if marker in index_name:
            base = index_name.split(marker, 1)[0]
            if not base:
                return None
            prefix = f"{dir_rel}/" if dir_rel else ""
            return f"{prefix}{base}-00001-of-00002.{ext}"
    return None


def _request_scopes_into_dir(allow_patterns: "Optional[list]", dir_name: str) -> bool:
    """True when an allow pattern names *dir_name* among its LITERAL leading path segments
    (``allow=['checkpoint-7/*']`` -> True for ``checkpoint-7``), i.e. the load reads INTO that directory.
    Lets the shard check skip a leftover checkpoint subtree the request does not target while still
    validating one it explicitly loads from. Segments are read only up to the first glob."""
    for p in allow_patterns or ():
        if not isinstance(p, str) or "/" not in p:
            continue
        for seg in p.split("/"):
            if _has_glob(seg):
                break  # a wildcard segment is not a literal directory target
            if seg == dir_name:
                return True
    return False


def _selected_shard_index_incomplete(
    snapshot_dir: Path, *, allow_patterns: "Optional[object]", ignore_patterns: "Optional[object]",
    variant: "Optional[str]",
) -> bool:
    """True when a weight-shard INDEX the load READS -- a sharded ADAPTER or a component SUBFOLDER set not
    covered by the canonical / variant ROOT-model checks -- lists an absent shard (or is malformed).
    Scoped to the request so a complete download is never false-rejected:

    - variant: a variant load reads only variant indices; a plain load only canonical ones.
    - allow / ignore: an index is read only when its listed shards survive the request filter.
    - precedence: safetensors read before bin, so when both are selected only the safetensors set's
      completeness is required.

    Also rejects a SELECTED numbered shard FILE whose directory has NO index of the read format: the load
    enumerates a sharded weight through its index, so a shard set without one is incomplete and would
    fetch the index + remaining shards over Xet. The ROOT canonical / variant MODEL shard set is skipped
    (the root-shard checks own it)."""
    allow_patterns = _as_pattern_list(allow_patterns)
    ignore_patterns = _as_pattern_list(ignore_patterns)
    want_variant = variant or None
    # Canonical single-weight bases whose presence beats a stale same-dir shard index. Includes
    # adapter_model so a single adapter_model.safetensors (a PEFT adapter load, allow=['adapter_model*'])
    # is not false-rejected by a co-resident stale adapter_model.safetensors.index.json.
    _canon = r"(?:diffusion_pytorch_model|model|pytorch_model|adapter_model)"
    if want_variant is None:
        shard_file_re = re.compile(r"^[^.]+-\d{5}-of-\d{5}\.(?:safetensors|bin)$")
        single_file_re = re.compile(rf"^{_canon}\.(?:safetensors|bin)$")
    else:
        v = re.escape(want_variant)
        shard_file_re = re.compile(rf"^[^.]+\.{v}-\d{{5}}-of-\d{{5}}\.(?:safetensors|bin)$")
        single_file_re = re.compile(rf"^{_canon}\.{v}\.(?:safetensors|bin)$")
    try:
        entries = list(snapshot_dir.rglob("*"))
    except OSError:
        return False
    per_dir: dict = {}       # dir_rel -> {"safetensors": [shard_rels, ...], "bin": [...]} (from indices)
    index_fmts: dict = {}    # dir_rel -> {fmt} an index of the read variant is present (non-root-model)
    shard_fmts: dict = {}    # dir_rel -> {fmt} a SELECTED numbered shard file is present (non-root-model)
    single_fmts: dict = {}   # dir_rel -> {fmt} a SELECTED single weight is present (beats that dir's index)
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
                # A malformed index. Defer only when the REQUEST reads its weight set, judged on a
                # representative shard of the index's OWN base + format (not the .json filename). So a
                # stale malformed index the request does NOT select, or one for an IGNORED format, must
                # not force a spurious retry; an unrecognizable index defers to the child.
                probe = _index_shard_probe(name, dir_rel)
                if probe is None or _filter_paths([probe], allow_patterns, ignore_patterns):
                    return True
                continue
            if not _filter_paths(shard_rels, allow_patterns, ignore_patterns):
                continue  # the load does not read this set (out of scope / ignored format)
            per_dir.setdefault(dir_rel, {}).setdefault(fmt, []).append(shard_rels)
        elif shard_file_re.match(name):
            # a numbered shard FILE of the read variant. Skip the ROOT model shard set (root checks own
            # it) and any training-checkpoint subtree.
            if dir_rel == "" and (
                (want_variant is None and _CANONICAL_ROOT_SHARD_RE.match(name))
                or (want_variant is not None and _ROOT_MODEL_VARIANT_WEIGHT_RE.match(name))
            ):
                continue
            ckpt_dirs = [p for p in rel.split("/")[:-1] if _CHECKPOINT_DIR_RE.match(p)]
            if ckpt_dirs and not _request_scopes_into_dir(allow_patterns, ckpt_dirs[0]):
                # a leftover checkpoint subtree the request does not target. An EXPLICIT checkpoint load
                # (allow=['checkpoint-N/*']) DOES read it, so that set is checked rather than skipped.
                continue
            if not _filter_paths([rel], allow_patterns, ignore_patterns):
                continue  # the load does not read this shard (out of scope / ignored format)
            fmt = "safetensors" if name.endswith(".safetensors") else "bin"
            shard_fmts.setdefault(dir_rel, set()).add(fmt)
        elif single_file_re.match(name):
            # a SINGLE canonical weight of the read variant: transformers/diffusers read it before a
            # same-format shard index in the SAME dir, so a stale co-resident index must not reject it.
            if _filter_paths([rel], allow_patterns, ignore_patterns):
                fmt = "safetensors" if name.endswith(".safetensors") else "bin"
                single_fmts.setdefault(dir_rel, set()).add(fmt)
    for dir_rel, by_fmt in per_dir.items():
        singles = single_fmts.get(dir_rel, set())
        # safetensors read before bin, and a single weight of a format beats that format's stale index.
        if "safetensors" in singles:
            continue
        required = by_fmt.get("safetensors")
        if required is None:
            if "bin" in singles:
                continue
            required = by_fmt.get("bin") or []
        for shard_rels in required:
            for shard in shard_rels:
                try:
                    if not (snapshot_dir / shard).exists():
                        return True
                except OSError:
                    return True
    for dir_rel, fmts in shard_fmts.items():
        # a numbered shard of the read format with NO index in its dir: the load cannot enumerate the set
        # and would fetch the index + remaining shards over Xet.
        preferred = "safetensors" if "safetensors" in fmts else "bin"
        if preferred not in index_fmts.get(dir_rel, set()):
            return True
    return False


# A training-checkpoint subdir (checkpoint-500/): never read as a diffusers pipeline COMPONENT, so an
# incomplete shard index under it must not force-fail a complete pipeline.
_CHECKPOINT_DIR_RE = re.compile(r"^checkpoint[-_]\d+$")


def _diffusers_declared_component_specs(snapshot_dir: Path) -> "Optional[dict]":
    """name -> declared ``[library, class]`` spec from a diffusers ``model_index.json`` (``_``-prefixed
    metadata excluded). None when absent / unreadable / malformed / empty, so the caller fails OPEN."""
    import json

    try:
        with open(snapshot_dir / "model_index.json", "r", encoding = "utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    specs = {
        key: value for key, value in data.items()
        if not key.startswith("_") and isinstance(value, (list, tuple))
    }
    # An empty / all-metadata model_index.json is degenerate -> fail OPEN (None).
    return specs or None


def _diffusers_declared_components(snapshot_dir: Path) -> "Optional[set]":
    """The component subfolder names a diffusers ``model_index.json`` declares. None when absent /
    unreadable / malformed so the caller falls back to every subfolder (fail OPEN). Scopes the component
    check to what the pipeline reads, so a stale UNDECLARED subtree cannot force-fail a complete download."""
    specs = _diffusers_declared_component_specs(snapshot_dir)
    return set(specs) if specs else None


def _diffusers_active_components(snapshot_dir: Path) -> "Optional[set]":
    """The ACTIVE (loaded) component names: a declared ``[library, class]`` pair with BOTH non-null. A
    ``[null, null]`` disabled / optional component (e.g. safety_checker) is excluded -- the pipeline does
    not load it, so a stale incomplete shard set under it must not force-fail a complete download. None
    when the model_index is absent / malformed (fail OPEN to every subfolder)."""
    specs = _diffusers_declared_component_specs(snapshot_dir)
    if not specs:
        return None
    return {
        name for name, spec in specs.items()
        if isinstance(spec, (list, tuple)) and len(spec) >= 2
        and spec[0] is not None and spec[1] is not None
    }


def _diffusers_component_shards_incomplete(
    snapshot_dir: Path, *, variant: "Optional[str]" = None,
    ignore_patterns: "Optional[object]" = None,
) -> bool:
    """True when a diffusers pipeline COMPONENT subfolder (unet/, vae/, ...) holds a weight-shard INDEX of
    the read variant listing an absent shard (or a malformed index) -- an interrupted component pull the
    pipeline load would finish over un-killable Xet.

    Scoped so a complete pipeline is never false-rejected: limited to declared components (a stale
    UNDECLARED subtree is skipped), ROOT indices (root checks own them) and checkpoint subtrees are
    skipped, and the ignore filter selects the read format. Per directory safetensors is read before bin.
    A plain load reads canonical component indices, a variant load variant ones. Also rejects a component
    numbered shard FILE with NO index of the read format. Positive-evidence: a single-file or complete
    component set passes."""
    want_variant = variant or None
    ignore_patterns = _as_pattern_list(ignore_patterns)
    # ACTIVE components only: a [null, null] disabled component (safety_checker) is not loaded, so a stale
    # incomplete shard index under it must not force-fail a complete pipeline into a DownloadStallError.
    declared = _diffusers_active_components(snapshot_dir)
    if want_variant is None:
        shard_file_re = re.compile(r"^[^.]+-\d{5}-of-\d{5}\.(?:safetensors|bin)$")
        single_file_re = re.compile(
            r"^(?:diffusion_pytorch_model|model|pytorch_model)\.(?:safetensors|bin)$"
        )
    else:
        v = re.escape(want_variant)
        shard_file_re = re.compile(rf"^[^.]+\.{v}-\d{{5}}-of-\d{{5}}\.(?:safetensors|bin)$")
        single_file_re = re.compile(
            rf"^(?:diffusion_pytorch_model|model|pytorch_model)\.{v}\.(?:safetensors|bin)$"
        )
    try:
        entries = list(snapshot_dir.rglob("*"))
    except OSError:
        return False
    per_dir: dict = {}
    index_fmts: dict = {}   # component dir_rel -> {fmt} an index of the read variant is present
    shard_fmts: dict = {}   # component dir_rel -> {fmt} a numbered shard file (ignore-kept) is present
    single_fmts: dict = {}  # component dir_rel -> {fmt} a SINGLE component weight (beats that dir's index)
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
                # A malformed index. Defer only when its FORMAT is read (a representative shard survives
                # the ignore filter); a stale malformed index for an IGNORED format is not read and must
                # not force a spurious retry of a complete other-format pipeline.
                probe = _index_shard_probe(name, dir_rel)
                if probe is None or _filter_paths([probe], None, ignore_patterns):
                    return True
                continue
            if not _filter_paths(shard_rels, None, ignore_patterns):
                continue  # the load does not read this set (ignored format)
            per_dir.setdefault(dir_rel, {}).setdefault(fmt, []).append(shard_rels)
        elif shard_file_re.match(name) and _filter_paths([rel], None, ignore_patterns):
            fmt = "safetensors" if name.endswith(".safetensors") else "bin"
            shard_fmts.setdefault(dir_rel, set()).add(fmt)
        elif single_file_re.match(name) and _filter_paths([rel], None, ignore_patterns):
            # a SINGLE canonical component weight: the pipeline reads it before a same-format shard index
            # in the SAME component dir, so a stale co-resident index must not reject a complete component.
            fmt = "safetensors" if name.endswith(".safetensors") else "bin"
            single_fmts.setdefault(dir_rel, set()).add(fmt)
    for dir_rel, by_fmt in per_dir.items():
        singles = single_fmts.get(dir_rel, set())
        # safetensors is read before bin, and a single weight of a format beats that format's stale index.
        if "safetensors" in singles:
            continue
        required = by_fmt.get("safetensors")
        if required is None:
            if "bin" in singles:
                continue
            required = by_fmt.get("bin") or []
        for shard_rels in required:
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
    """For a request naming EXACT files (every entry glob-free), True only when each named file the ignore
    filter keeps is on disk -- ``local_files_only`` returns the revision dir even when config-only, so a
    ``["tokenizer.json"]`` request needs its file present. A request with ANY glob, or no allow list, is
    trivially satisfied."""
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
    # Check every snapshot, not just the newest: an older revision may be broken while a newer is clean.
    return any(
        snapshot_dir_has_broken_symlinks(snapshot)
        for snapshot in _iter_snapshot_dirs(repo_dir)
    )


def _case_safe_repo_cache_dirs(root: Path, repo_type: Optional[str], repo_id: str) -> list:
    """Cache dirs safely attributable to this exact repo id.

    The Hub case-folds the dir name, so a case-insensitive match is needed, but on a case-sensitive fs
    ``models--Org--Repo`` and ``models--org--repo`` are distinct repos. Prefer exact case; else accept a
    single folded match ONLY on a case-insensitive fs; on a 2+ way collision attribute to neither, so a
    stale partial in one repo cannot make the watchdog kill / purge the other."""
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
        # A single folded-but-not-exact match is the same dir only on a case-insensitive fs; on a
        # case-sensitive fs it is a DIFFERENT repo.
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
    safe, so the read / watchdog path and the destructive HTTP-prep path share one rule."""
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
