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

"""Sparse-aware introspection of the active HF_HUB_CACHE (on-disk bytes, .incomplete partials) for the stall watchdog;
snapshot_dir_is_complete is conservative: a false complete would let a load fetch over un-killable Xet."""

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
    """Hub cache root to scan, or None if unavailable; an explicit cache_dir wins, else HF_HUB_CACHE is read lazily so import-time redirects apply."""
    if cache_dir is not None:
        # Match huggingface_hub's ~ expansion; expanduser() raises with no resolvable HOME (container), so fall back to the literal path.
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
    """Sparse-aware on-disk size: prefers st_blocks because a sparse .incomplete reports full st_size; falls back to st_size where st_blocks is unreported."""
    st = path.stat()
    blocks = getattr(st, "st_blocks", None)
    if blocks is not None:
        # Trust st_blocks even at 0: a truncated sparse .incomplete reports full st_size with 0 blocks and must read as
        # 0 bytes; a > 0 guard would fall through to st_size.
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
    """True if a snapshot symlink dangles (blob missing or still .incomplete), i.e. an interrupted download; checks the requested revision, not the newest."""
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

# Trainer/optimizer state carries weight suffixes but is not loadable; a cache holding only these is not a warm model cache.
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
    """True for a canonical or variant weight-shard index; a plain suffix test misses the variant form (model.safetensors.index.fp16.json)."""
    return name.endswith(".json") and (".safetensors.index." in name or ".bin.index." in name)


def _is_canonical_weight_shard_index(name: str) -> bool:
    """Exact canonical index names only: a variant- or adapter-only cache lacks the base canonical weights, so the load would fetch them over un-killable Xet."""
    return name in ("model.safetensors.index.json", "pytorch_model.bin.index.json")


# Exact canonical or single-variant component shard index (one variant token); rejects stale sidecars like
# diffusion_pytorch_model.safetensors.index.fp16.extra.json; keeps the base for the probe.
_COMPONENT_SHARD_INDEX_RE = re.compile(
    r"^(?P<base>diffusion_pytorch_model|model|pytorch_model)\.(?P<ext>safetensors|bin)"
    r"\.index(?:\.(?P<variant>[^.]+))?\.json$"
)


def _is_canonical_component_shard_index(name: str) -> bool:
    """Exact canonical or single-variant diffusers component shard index; finds components with non-standard shard names via their index, rejects stale sidecars."""
    return _COMPONENT_SHARD_INDEX_RE.match(name) is not None


def _component_index_weight_probe(index_name: str, dir_rel: str) -> "Optional[str]":
    """Component-relative weight path a shard index enumerates, keeping its base, format and variant so ignore globs
    (unet/diffusion_pytorch_model*) match; None if not an index."""
    m = _COMPONENT_SHARD_INDEX_RE.match(index_name)
    if m is None:
        return None
    token = f".{m['variant']}" if m["variant"] else ""
    prefix = f"{dir_rel}/" if dir_rel else ""
    return f"{prefix}{m['base']}{token}.{m['ext']}"


def _is_unsafe_shard_ref(shard: str) -> bool:
    """True if a weight_map value is not a safe relative path (absolute, drive letter, UNC, or ..-escaping); checked
    under POSIX and Windows rules so a crafted index fails on any OS."""
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
    """True only if every shard the weight index lists is present; fails closed on unreadable, non-dict or empty indexes, deferring them to the watched child."""
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
    """Normalize an allow/ignore argument to a list; a bare str is wrapped, since HF accepts one and iterating it would split it into characters."""
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
    """Filter repo paths by HF allow/ignore patterns as snapshot_download does; fails open (all paths) so a weight is never rejected on an unevaluable filter."""
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
    """Format survives the ignore filter; ignore_patterns must be normalized (_as_pattern_list) or None"""
    if not ignore_patterns:
        return True
    return bool(_filter_paths([weight_name], None, ignore_patterns))


def _index_weight_probe(index_name: str, variant: "Optional[str]" = None) -> str:
    """Root weight name a shard index's format maps to (safetensors gives model[.variant].safetensors, else
    pytorch_model[.variant].bin); one source for presence and completeness checks."""
    token = f".{variant}" if variant else ""
    if ".safetensors.index." in index_name:
        return f"model{token}.safetensors"
    return f"pytorch_model{token}.bin"


def _broken_symlink_rel_paths(snapshot_dir: Path) -> list:
    """Repo-relative posix paths of every dangling symlink in the snapshot, so the interrupted-download signal can be scoped to the files a request selects."""
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
    """True iff a dangling symlink is for a file the request selects, so a dangling root model.safetensors does not fail
    a weightless allow=['config.json'] request; repo_type is kept for compatibility."""
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
    """True iff the ignore set excludes every weight format; a partial strip stays weight-bearing since a surviving format could still be pulled."""
    for suffix in _WEIGHT_FILE_SUFFIXES:
        probe = "weight" + suffix
        if not any(isinstance(p, str) and fnmatch.fnmatchcase(probe, p) for p in ignore_patterns):
            return False
    return True


# Representative weight names a glob allow pattern is probed against; a glob matching none (tokenizer*, *.json) is
# weightless; covers variant, sharded, adapter, diffusers and .bin forms.
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

# Metadata-only subdirs: a dir pattern scoped to one is weightless; other dir patterns stay weight-bearing, since component or checkpoint dirs can hold weights.
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
    """Whether one allow pattern can select a weight: weight suffixes and non-metadata dirs count, globs only if they
    match a _WEIGHT_PATTERN_PROBES name, concrete non-weight names do not; a false weight-bearing only spawns the
    child."""
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
    """Whether an allow/ignore-restricted request can still include a weight; conservative, so False only for a clearly
    weightless request (tokenizer/config allow, or ignore stripping all weights)."""
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
    """True iff the root holds a complete canonical weight set (single file or full index) in formats not ignored;
    strict prefer_safetensors rejects bin-only unless safetensors is ignored (else the load fetches over Xet); the
    lenient POST path accepts a finished bin-only repo."""
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

    # Maps each root index to the weight it enumerates; the format split lives in _index_weight_probe, so each canonical index maps to a distinct probe.
    index_by_read = {_index_weight_probe(e.name): e for e in root_indices}
    st_index = index_by_read.get("model.safetensors")
    bin_index = index_by_read.get("pytorch_model.bin")
    # Mirrors transformers' local precedence (single safetensors, safetensors index, single bin, bin index): a complete
    # single weight beats a stale index, and an incomplete preferred safetensors index is not masked by a complete .bin;
    # ignored formats are skipped.
    if "model.safetensors" in root_files and _read_format_kept("model.safetensors", ignore_patterns):
        return True
    if st_index is not None and _read_format_kept("model.safetensors", ignore_patterns):
        return _weight_shard_index_complete(st_index)
    if prefer_safetensors and _read_format_kept("model.safetensors", ignore_patterns):
        # Strict gate: safetensors is preferred but absent, and a bin-only cache cannot prove it absent remotely, so a
        # default load would fetch it over Xet; defer.
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
    """Conservative fast path: True only for an unpatterned, non-diffusers cache with no dangling symlink and complete
    root weights (a false True risks a silent Xet fetch); strict prefer_safetensors also rejects a bin-only cache."""
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


# sentence-transformers modules.json types: weightless (pooling, normalize, layernorm, dropout) vs weight-bearing
# (dense, cnn, lstm), matched on the final dotted component.
_ST_WEIGHTLESS_MODULE_TYPES = frozenset({"pooling", "normalize", "layernorm", "dropout"})
_ST_WEIGHTED_MODULE_TYPES = frozenset({"dense", "cnn", "lstm"})


def _sentence_transformers_subfolder_incomplete(
    snapshot_dir: Path, *, prefer_safetensors: bool = False,
    ignore_patterns: "Optional[object]" = None, strict: bool = True,
) -> bool:
    """True when an ST modules.json subfolder weight (e.g. 2_Dense) is absent, as the root check does not cover it;
    strict PRE requires a weight for every module not known weightless, POST only for known dense/cnn/lstm; an
    unreadable modules.json returns the strict flag (PRE defers, POST accepts)."""
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
            # When safetensors is read and not ignored, a bin-only subfolder cannot prove the preferred safetensors
            # absent remotely, so defer (the load would fetch it over Xet).
            if not st_present:
                return True
            continue
        if not (st_present or bin_present):
            return True  # no readable weight of the format the load reads
    return False


# Canonical numbered root shard with no variant token: model-00001-of-00002.safetensors matches, model-00001-of-00002.fp16.safetensors does not.
_CANONICAL_ROOT_SHARD_RE = re.compile(
    r"^(?:model|pytorch_model)-\d{5}-of-\d{5}\.(?:safetensors|bin)$"
)


def _has_incomplete_canonical_root_shards(
    snapshot_dir: Path, *, ignore_patterns: "Optional[object]" = None
) -> bool:
    """True when root canonical numbered shards lack a complete index or shard set for the format read (ignore filter
    applied), a stale download the POST check retries over HTTP; variant shards excluded."""
    try:
        names = [entry.name for entry in snapshot_dir.iterdir()]
    except OSError:
        return False
    # Shard evidence is a numbered shard file or canonical shard index; an index-only partial (no shards yet) is still
    # incomplete, caught before any shard exists.
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
    """True when the root variant weight the load reads is an incomplete shard set (partial index, or shards with no
    index); safetensors is checked before bin; only root model variants count, so a stale adapter_model set cannot
    force-fail it."""
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
        # Only the ROOT model index counts: a non-model variant index (e.g. adapter_model) the default load does not
        # read is skipped so it cannot fail the model variant.
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
    # Order: safetensors > safetensors index > bin > bin index; complete single file beats stale index
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

# Root canonical/variant model shard indexes (model.safetensors.index.json, pytorch_model.bin.index.json and variant forms); owned by the root checks.
_ROOT_MODEL_SHARD_INDEX_RE = re.compile(
    r"^(?:model\.safetensors|pytorch_model\.bin)\.index(?:\.[^.]+)?\.json$"
)

# Root model variant weight, single or sharded (model.fp16.safetensors, pytorch_model.fp16-00001-of-00002.bin); excludes
# PEFT adapters the variant load does not read.
_ROOT_MODEL_VARIANT_WEIGHT_RE = re.compile(
    r"^(?:model|pytorch_model)\.[^.]+(?:-\d{5}-of-\d{5})?\.(?:safetensors|bin)$"
)


def _index_variant_token(name: str) -> "Optional[str]":
    """Variant token of a shard index basename (model.safetensors.index.fp16.json gives fp16), None for canonical; lets checks read only the indices a load reads."""
    if name.endswith(".safetensors.index.json") or name.endswith(".bin.index.json"):
        return None
    m = _VARIANT_SHARD_INDEX_RE.search(name)
    return m.group(1) if m else None


def _index_shard_rel_paths(index_path: Path, dir_rel: str) -> "Optional[list]":
    """Snapshot-relative paths of the shards an index lists, joined under dir_rel so the request filter sees repo paths;
    None if unreadable or malformed (fail closed)."""
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
    """Representative first-shard path (index base + format under dir_rel) for a malformed index, so filters judge the
    weight not the .json name; None if unrecognized."""
    for marker, ext in ((".safetensors.index.", "safetensors"), (".bin.index.", "bin")):
        if marker in index_name:
            base = index_name.split(marker, 1)[0]
            if not base:
                return None
            prefix = f"{dir_rel}/" if dir_rel else ""
            return f"{prefix}{base}-00001-of-00002.{ext}"
    return None


def _request_scopes_into_dir(allow_patterns: "Optional[list]", dir_name: str) -> bool:
    """True when an allow pattern names dir_name in its literal leading segments (read up to the first glob), e.g. allow=['checkpoint-7/*'] for checkpoint-7."""
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
    # Canonical single-weight bases whose file beats a stale same-dir shard index; includes adapter_model so a PEFT
    # adapter file is not rejected by a stale adapter index.
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
                # Malformed index: defer only if the request reads its weight set (judged on a representative shard of
                # its own base and format, not the .json name); unrecognizable defers.
                probe = _index_shard_probe(name, dir_rel)
                if probe is None or _filter_paths([probe], allow_patterns, ignore_patterns):
                    return True
                continue
            if not _filter_paths(shard_rels, allow_patterns, ignore_patterns):
                continue  # the load does not read this set (out of scope / ignored format)
            per_dir.setdefault(dir_rel, {}).setdefault(fmt, []).append(shard_rels)
        elif shard_file_re.match(name):
            # Numbered shard file of the read variant; skips the root model shard set (root checks own it) and training-checkpoint subtrees.
            if dir_rel == "" and (
                (want_variant is None and _CANONICAL_ROOT_SHARD_RE.match(name))
                or (want_variant is not None and _ROOT_MODEL_VARIANT_WEIGHT_RE.match(name))
            ):
                continue
            ckpt_dirs = [p for p in rel.split("/")[:-1] if _CHECKPOINT_DIR_RE.match(p)]
            if ckpt_dirs and not _request_scopes_into_dir(allow_patterns, ckpt_dirs[0]):
                # Skips a leftover checkpoint subtree the request does not target; an explicit load
                # (allow=['checkpoint-N/*']) does read it, so that set is checked.
                continue
            if not _filter_paths([rel], allow_patterns, ignore_patterns):
                continue  # the load does not read this shard (out of scope / ignored format)
            fmt = "safetensors" if name.endswith(".safetensors") else "bin"
            shard_fmts.setdefault(dir_rel, set()).add(fmt)
        elif single_file_re.match(name):
            # A single canonical weight is read before a same-format shard index in the same dir, so a stale co-resident index must not reject it.
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
        # A numbered shard of the read format with no index in its dir is incomplete: the load cannot enumerate it and would fetch the rest over Xet.
        preferred = "safetensors" if "safetensors" in fmts else "bin"
        if preferred not in index_fmts.get(dir_rel, set()):
            return True
    return False


# Training-checkpoint subdir (checkpoint-500/) is never a diffusers pipeline component, so its incomplete shard index must not fail a complete pipeline.
_CHECKPOINT_DIR_RE = re.compile(r"^checkpoint[-_]\d+$")


def _diffusers_declared_component_specs(snapshot_dir: Path) -> "Optional[dict]":
    """Map component name to its declared [library, class] spec from model_index.json (underscore metadata excluded);
    None when absent, unreadable or empty (fail open)."""
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
    """Component subfolders declared in model_index.json, so stale undeclared subtrees cannot force-fail a download;
    None (check all subfolders) when absent or malformed."""
    specs = _diffusers_declared_component_specs(snapshot_dir)
    return set(specs) if specs else None


def _diffusers_active_components(snapshot_dir: Path) -> "Optional[set]":
    """Active components: declared [library, class] pairs with both non-null, so disabled [null, null] components like
    safety_checker are skipped; None (fail open) if absent or malformed."""
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
    """True when an active diffusers component subfolder has a read-variant shard index with an absent or malformed
    shard, or a numbered shard with no index; root and checkpoint dirs skipped."""
    want_variant = variant or None
    ignore_patterns = _as_pattern_list(ignore_patterns)
    # Only ACTIVE components count: a disabled [null, null] component (safety_checker) is not loaded, so its stale
    # incomplete index must not raise DownloadStallError.
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
                # A malformed index defers only if its format is read (a shard survives the ignore filter); an ignored
                # format's stale index must not force a retry.
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
            # A single canonical component weight is read before a same-format shard index in the same dir, so a stale
            # co-resident index must not reject the component.
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
    """For exact-name allow entries, True only when each named file the ignore filter keeps is on disk (local_files_only
    returns the dir even when config-only); any glob or no allow list passes."""
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
    """Hub case-folds cache dir names: prefer exact case, accept a folded match only on a case-insensitive fs, and
    attribute nothing on a collision so no other repo is purged."""
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
        # A single case-folded but not exact match is the same dir only on a case-insensitive fs; on a case-sensitive fs it is a different repo.
        try:
            if (root / target).exists():
                return entries
        except OSError:
            return []
    return []


def iter_active_repo_cache_dirs(
    repo_type: Optional[str], repo_id: str, *, cache_dir: "Optional[str | Path]" = None
) -> Iterator[Path]:
    """Yield the repo's cache dirs under cache_dir or the active HF_HUB_CACHE, case-collision safe; the watchdog and destructive HTTP-prep paths share this rule."""
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
