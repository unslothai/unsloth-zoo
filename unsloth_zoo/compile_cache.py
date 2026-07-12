# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Persistent torch.compile artifact cache for Unsloth training (Mega-cache).

``unsloth_compiled_cache/`` only caches the GENERATED SOURCE; Dynamo tracing,
AOTAutograd partitioning, Inductor codegen and Triton autotuning are redone by
every new process unless torch's own on-disk caches (default
``/tmp/torchinductor_<user>``, which rarely survives reboots or container
restarts) are warm. This module persists those artifacts via the portable
Mega-cache API (``torch.compiler.save/load_cache_artifacts``, torch >= 2.7) so
a model family's compile cost is paid once per IDENTICAL environment.

A bundle is only valid for the exact torch / Triton / CUDA build, GPU
architecture, transformers version and generated-source configuration, so the
key is an exact-match fingerprint over all of them; a mismatch is just a cache
miss. Loading is best-effort and never raises.

Environment knobs:
  UNSLOTH_MEGA_CACHE       = "auto" (default) | "0" | "1"
      auto -> load a matching bundle AND save at exit when the recorded
              artifacts differ from disk (torch records hits too, so saves
              merge loaded + new artifacts).
      1    -> like auto, but always rewrite the bundle at exit.
      0    -> fully disabled (kill switch).
  UNSLOTH_MEGA_CACHE_DIR   = bundle root (default ~/.cache/unsloth/mega_cache).
"""

__all__ = [
    "megacache_is_enabled",
    "megacache_load",
    "megacache_save",
    "megacache_status",
]

import atexit
import hashlib
import json
import os
import time

_UNSLOTH_ENABLE_LOGGING = os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1"

_FORMAT_VERSION = 1
_BUNDLE_NAME = "megacache.bin"
_MANIFEST_NAME = "manifest.json"

# Process-level state: each unsloth_compile_transformers call extends the key.
_STATE = {
    "armed": False,
    "loaded_key": None,
    "hit": False,
    "model_keys": [],
    "atexit_registered": False,
}


def _log(message):
    if _UNSLOTH_ENABLE_LOGGING:
        print(f"Unsloth: Mega-cache {message}", flush = True)
pass


def megacache_is_enabled():
    return (os.environ.get("UNSLOTH_MEGA_CACHE", "auto").strip().lower()
            not in ("0", "off", "false", "no"))
pass


def _refresh_enabled():
    return os.environ.get("UNSLOTH_MEGA_CACHE", "auto").strip().lower() in ("1", "on", "true", "yes")
pass


def _cache_root():
    root = os.path.expanduser(os.environ.get("UNSLOTH_MEGA_CACHE_DIR", ""))
    if root == "":
        home = os.path.expanduser("~")
        # No resolvable home: disable rather than write a literal "~" dir.
        if home == "~" or not os.path.isdir(home):
            return None
        root = os.path.join(home, ".cache", "unsloth", "mega_cache")
    return root
pass


def _mega_cache_supported():
    try:
        import torch
        return (
            hasattr(torch.compiler, "save_cache_artifacts")
            and hasattr(torch.compiler, "load_cache_artifacts")
        )
    except Exception:
        return False
pass


def _environment_fingerprint():
    """Hard portability dimensions - any difference invalidates a bundle."""
    fingerprint = {
        "format": _FORMAT_VERSION,
        "torch": None,
        "cuda": None,
        "triton": None,
        "transformers": None,
        "unsloth_zoo": None,
        "gpu_name": None,
        "gpu_capability": None,
    }
    try:
        import torch
        fingerprint["torch"] = str(torch.__version__)
        fingerprint["cuda"] = str(torch.version.cuda)
        if torch.cuda.is_available():
            # Bound device, not device 0: mixed-architecture hosts must not
            # save one GPU's kernels under another's key.
            device = torch.cuda.current_device()
            fingerprint["gpu_name"] = torch.cuda.get_device_name(device)
            capability = torch.cuda.get_device_capability(device)
            fingerprint["gpu_capability"] = f"sm_{capability[0]}{capability[1]}"
    except Exception:
        pass
    try:
        import triton
        fingerprint["triton"] = str(getattr(triton, "__version__", None))
    except Exception:
        pass
    try:
        import transformers
        fingerprint["transformers"] = str(transformers.__version__)
    except Exception:
        pass
    try:
        from . import __version__ as zoo_version
        fingerprint["unsloth_zoo"] = str(zoo_version)
    except Exception:
        pass
    return fingerprint
pass


def _model_key(model_type, compile_kwargs, torch_compile_options):
    """One model's contribution to the bundle key: the arguments that change
    the GENERATED SOURCE (hence the traced graphs) plus the Inductor options
    baked into the generated ``@torch.compile`` decorators."""
    payload = {
        "model_type": str(model_type),
        "compile_kwargs": {k: compile_kwargs[k] for k in sorted(compile_kwargs)},
        "torch_compile_options": {k: torch_compile_options[k] for k in sorted(torch_compile_options)},
    }
    return payload
pass


def _bundle_key():
    payload = json.dumps(
        {"env": _environment_fingerprint(), "models": _STATE["model_keys"]},
        sort_keys = True, default = str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
pass


def _bundle_paths(key):
    root = _cache_root()
    if root is None:
        return None, None
    directory = os.path.join(root, key)
    return directory, os.path.join(directory, _MANIFEST_NAME)
pass


def _read_disk_bundle(directory, manifest_path):
    """Read (manifest, bundle bytes), or ``(None, None)`` unless the manifest
    checksum matches the bundle file it names. Bundle files are
    content-addressed (same name implies same bytes), so a manifest can never
    validate a different writer's bundle."""
    try:
        with open(manifest_path, "r") as file:
            manifest = json.load(file)
        bundle_name = os.path.basename(str(manifest.get("bundle", _BUNDLE_NAME)))
        with open(os.path.join(directory, bundle_name), "rb") as file:
            data = file.read()
        if manifest.get("sha256") != hashlib.sha256(data).hexdigest():
            return None, None
        return manifest, data
    except Exception:
        return None, None
pass


def _merge_recorded_artifacts(data):
    """Re-record the on-disk bundle's artifacts so the next
    ``save_cache_artifacts`` serializes the UNION of bundle + this process.
    torch only records what this run exercised, so without this a run hitting
    a subset of a warm bundle would replace it with that subset. Best-effort:
    a failure just means this save may narrow the bundle."""
    try:
        from torch.compiler._cache import CacheArtifactManager
        artifacts = CacheArtifactManager.deserialize(data)
        if not artifacts:
            return
        merged = 0
        for artifact_type, items in artifacts.items():
            for artifact in items:
                # Value-equality dedup skips artifacts this run re-recorded.
                if artifact in CacheArtifactManager._seen_artifacts:
                    continue
                CacheArtifactManager._new_cache_artifacts[artifact_type].append(artifact)
                CacheArtifactManager._seen_artifacts.add(artifact)
                merged += 1
        if merged:
            _log(f"merged {merged} artifact(s) from the existing bundle")
    except Exception as error:
        _log(f"bundle merge skipped ({error})")
pass


def megacache_load(model_type, compile_kwargs = None, torch_compile_options = None):
    """Pre-load compile artifacts for ``model_type`` and register the
    exit-time save. Runs during ``from_pretrained``, strictly before any
    ``@torch.compile`` region consults the caches this pre-populates.
    Never raises; a miss just means a normal local compile."""
    if not megacache_is_enabled():
        return False
    if not _mega_cache_supported():
        _log("unavailable (needs torch >= 2.7); skipping")
        return False

    _STATE["model_keys"].append(
        _model_key(model_type, compile_kwargs or {}, torch_compile_options or {})
    )
    key = _bundle_key()
    directory, manifest_path = _bundle_paths(key)
    if manifest_path is None:
        _log("no writable cache root; skipping")
        return False

    _STATE["armed"] = True
    _STATE["loaded_key"] = key
    # A hit for an earlier cumulative key does not cover this extended one.
    _STATE["hit"] = False
    if not _STATE["atexit_registered"]:
        atexit.register(megacache_save)
        _STATE["atexit_registered"] = True

    manifest, data = _read_disk_bundle(directory, manifest_path)
    if manifest is None:
        _log(f"no bundle for key {key} (will compile locally and save on exit)")
        return False

    try:
        import torch
        info = torch.compiler.load_cache_artifacts(data)
        if info is None:
            _log("torch reported no loadable artifacts; ignoring")
            return False
        _STATE["hit"] = True
        _log(f"loaded bundle for key {key} ({len(data)} bytes)")
        return True
    except Exception as error:
        _log(f"load failed ({error}); continuing with local compile")
        return False
pass


def megacache_save():
    """Persist this process' compile artifacts (atexit or on demand). torch
    records hits as well as writes, but only what this run exercised - so the
    on-disk bundle is merged back first and the save is always the UNION of
    bundle + process. An unchanged bundle is skipped. Never raises."""
    if not (_STATE["armed"] and megacache_is_enabled() and _mega_cache_supported()):
        return False
    # One writer per NODE (cache roots default to node-local disks); shared
    # roots are guarded by the pid-unique temp + atomic replace below.
    if os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")) != "0":
        return False
    key = _STATE["loaded_key"]
    if key is None:
        return False
    try:
        import torch
        directory, manifest_path = _bundle_paths(key)
        if manifest_path is None:
            return False
        # Union semantics: fold the on-disk bundle back in first so a run
        # that exercised only part of it cannot narrow it.
        _, existing = _read_disk_bundle(directory, manifest_path)
        if existing is not None:
            _merge_recorded_artifacts(existing)
        result = torch.compiler.save_cache_artifacts()
        if not result or not result[0]:
            _log("nothing to save (no compile artifacts recorded)")
            return False
        data = result[0]
        new_sha = hashlib.sha256(data).hexdigest()
        if not _refresh_enabled() and existing is not None:
            if hashlib.sha256(existing).hexdigest() == new_sha:
                _log("bundle unchanged; skipping save")
                return False
        os.makedirs(directory, exist_ok = True)
        # Content-addressed bundle + manifest naming it = consistency without
        # a cross-file lock: concurrent writers produce differently named
        # bundles, the manifest replace is atomic, and the last manifest
        # points at its own complete file. Same name implies same bytes.
        bundle_name = f"megacache-{new_sha[:16]}.bin"
        bundle_path = os.path.join(directory, bundle_name)
        if not os.path.isfile(bundle_path):
            temporary_path = f"{bundle_path}.tmp.{os.getpid()}"
            with open(temporary_path, "wb") as file:
                file.write(data)
            os.replace(temporary_path, bundle_path)
        manifest = {
            "format": _FORMAT_VERSION,
            "key": key,
            "created": time.time(),
            "bytes": len(data),
            "sha256": new_sha,
            "bundle": bundle_name,
            "env": _environment_fingerprint(),
            "models": _STATE["model_keys"],
        }
        temporary_manifest = f"{manifest_path}.tmp.{os.getpid()}"
        with open(temporary_manifest, "w") as file:
            json.dump(manifest, file, indent = 2, sort_keys = True, default = str)
        os.replace(temporary_manifest, manifest_path)
        _log(f"saved bundle for key {key} ({len(data)} bytes)")
        # Best-effort GC of superseded bundles (incl. legacy fixed name).
        # Open readers keep their handle; later readers re-read the manifest.
        try:
            for name in os.listdir(directory):
                is_stale_bundle = (
                    name == _BUNDLE_NAME
                    or (name.startswith("megacache-") and name.endswith(".bin"))
                    or ".bin.tmp." in name
                )
                if is_stale_bundle and name != bundle_name:
                    os.unlink(os.path.join(directory, name))
        except Exception:
            pass
        return True
    except Exception as error:
        _log(f"save failed ({error})")
        return False
pass


def megacache_status():
    """Introspection helper for tests and debugging."""
    return {
        "enabled": megacache_is_enabled(),
        "supported": _mega_cache_supported(),
        "armed": _STATE["armed"],
        "hit": _STATE["hit"],
        "key": _STATE["loaded_key"],
        "root": _cache_root(),
    }
pass
