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

``unsloth_compiled_cache/`` only caches the GENERATED SOURCE of the patched
modules. The expensive part of a first training step - Dynamo tracing,
AOTAutograd partitioning, Inductor codegen and Triton autotuning for the
forward AND backward graphs - is redone by every new process unless torch's
own on-disk caches (FX graph cache, AOTAutograd cache, autotune cache) are
warm. Those caches live under ``TORCHINDUCTOR_CACHE_DIR`` (by default a
``/tmp/torchinductor_<user>`` directory that rarely survives reboots or
container restarts, and is never shipped to users).

This module persists the same artifacts through the portable Mega-cache API
(``torch.compiler.save_cache_artifacts`` / ``load_cache_artifacts``,
torch >= 2.7) so the one-time compile cost of a model family can be paid once
per environment and reused across processes, container restarts and machines
with an IDENTICAL environment.

A bundle is only valid for the exact same torch / Triton / CUDA build, GPU
architecture, transformers version and generated-source configuration. The
key is therefore an exact-match fingerprint over all of those dimensions; a
mismatch simply means a cache miss and a normal local compile. Loading is
always best-effort and never raises.

Environment knobs:
  UNSLOTH_MEGA_CACHE       = "auto" (default) | "0" | "1"
      auto -> load a matching bundle if one exists AND save a bundle at
              process exit when compilation happened with no bundle present.
      1    -> same as auto, plus refresh the bundle even when one was loaded.
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

# Process-level state: one fingerprint accumulates all models loaded in this
# process (multiple unsloth_compile_transformers calls extend the key).
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
        # Restricted containers can have no resolvable home; disable rather
        # than writing a literal "~" directory into the CWD.
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
            fingerprint["gpu_name"] = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
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
    """One model's contribution to the bundle key.

    ``compile_kwargs`` are the unsloth_compile_transformers arguments that
    change the GENERATED SOURCE (and therefore the traced graphs);
    ``torch_compile_options`` is the Inductor options dict baked into the
    generated ``@torch.compile`` decorators.
    """
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
        return None, None, None
    directory = os.path.join(root, key)
    return (
        directory,
        os.path.join(directory, _BUNDLE_NAME),
        os.path.join(directory, _MANIFEST_NAME),
    )
pass


def megacache_load(model_type, compile_kwargs = None, torch_compile_options = None):
    """Try to pre-load compile artifacts for ``model_type`` before the first
    compiled forward. Registers the exit-time save as a side effect.

    Called from ``unsloth_compile_transformers`` which runs during
    ``from_pretrained`` - i.e. strictly before any ``@torch.compile`` region
    executes, which is when Dynamo consults the caches this pre-populates.

    Never raises; a miss just means a normal local compile.
    """
    if not megacache_is_enabled():
        return False
    if not _mega_cache_supported():
        _log("unavailable (needs torch >= 2.7); skipping")
        return False

    _STATE["model_keys"].append(
        _model_key(model_type, compile_kwargs or {}, torch_compile_options or {})
    )
    key = _bundle_key()
    directory, bundle_path, manifest_path = _bundle_paths(key)
    if bundle_path is None:
        _log("no writable cache root; skipping")
        return False

    _STATE["armed"] = True
    _STATE["loaded_key"] = key
    # A hit for an earlier cumulative key does not cover this one: reset so a
    # miss here still saves the extended bundle at exit.
    _STATE["hit"] = False
    if not _STATE["atexit_registered"]:
        atexit.register(megacache_save)
        _STATE["atexit_registered"] = True

    if not (os.path.isfile(bundle_path) and os.path.isfile(manifest_path)):
        _log(f"no bundle for key {key} (will compile locally and save on exit)")
        return False

    try:
        with open(manifest_path, "r") as file:
            manifest = json.load(file)
        with open(bundle_path, "rb") as file:
            data = file.read()
        # Defence in depth: torch validates internally as well, but a stale or
        # corrupted bundle should be an explicit miss, not a surprise.
        if manifest.get("sha256") != hashlib.sha256(data).hexdigest():
            _log("bundle checksum mismatch; ignoring")
            return False
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
    """Persist this process' compile artifacts. Runs at exit (atexit) or on
    demand. Saves only when something new would be stored: a fresh bundle on
    a miss, or a refresh when UNSLOTH_MEGA_CACHE=1. Never raises.
    """
    if not (_STATE["armed"] and megacache_is_enabled() and _mega_cache_supported()):
        return False
    if _STATE["hit"] and not _refresh_enabled():
        return False
    # Distributed launches: every rank records the same artifacts; concurrent
    # writers could leave the manifest checksum pointing at another rank's
    # bytes, so only the main rank persists the bundle.
    if os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) != "0":
        return False
    key = _STATE["loaded_key"]
    if key is None:
        return False
    try:
        import torch
        result = torch.compiler.save_cache_artifacts()
        if not result or not result[0]:
            _log("nothing to save (no compile artifacts recorded)")
            return False
        data = result[0]
        directory, bundle_path, manifest_path = _bundle_paths(key)
        if bundle_path is None:
            return False
        os.makedirs(directory, exist_ok = True)
        # PID-unique temp + atomic replace: concurrent processes cannot
        # truncate or interleave each other's bytes.
        temporary_path = f"{bundle_path}.tmp.{os.getpid()}"
        with open(temporary_path, "wb") as file:
            file.write(data)
        os.replace(temporary_path, bundle_path)
        manifest = {
            "format": _FORMAT_VERSION,
            "key": key,
            "created": time.time(),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "env": _environment_fingerprint(),
            "models": _STATE["model_keys"],
        }
        temporary_manifest = f"{manifest_path}.tmp.{os.getpid()}"
        with open(temporary_manifest, "w") as file:
            json.dump(manifest, file, indent = 2, sort_keys = True, default = str)
        os.replace(temporary_manifest, manifest_path)
        _log(f"saved bundle for key {key} ({len(data)} bytes)")
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
