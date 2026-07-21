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
  UNSLOTH_MEGA_CACHE       = "1" (default, enabled) | "0" (kill switch)
      Enabled by default: load and save artifacts, but only from a cache
      directory that passes the POSIX trust checks below (owned by the current
      user, not group/other writable, no symlinked components). torch.compile
      artifacts are executable/deserialized data, so an untrusted cache
      directory is refused rather than loaded - the feature fails closed.
      0 / off / false / no -> fully disabled (kill switch).
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
import stat
import time

_UNSLOTH_ENABLE_LOGGING = os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1"

_FORMAT_VERSION = 1
_BUNDLE_NAME = "megacache.bin"
_MANIFEST_NAME = "manifest.json"

# One fingerprint accumulates every model loaded in this process; each
# unsloth_compile_transformers call extends the key.
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
    # On by default. Bundles are executable/deserialized torch.compile
    # artifacts, so the POSIX trust checks (owner-only, no group/other write,
    # no symlinks) are the safeguard: an untrusted cache is refused, not
    # loaded, so the feature fails closed. Kill switch: UNSLOTH_MEGA_CACHE=0.
    return os.environ.get("UNSLOTH_MEGA_CACHE", "1").strip().lower() not in (
        "0", "off", "false", "no",
    )
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
    # Canonicalize symlinks and ".." so the path we validate is exactly the one
    # later reads and writes resolve to. abspath only collapses ".." lexically,
    # so a symlinked component followed by ".." would validate one directory
    # while the kernel accesses another.
    return os.path.realpath(root)
pass


def _is_trusted_directory(path, *, allow_missing = True):
    """Return whether an existing cache directory is safe to load from.

    A checksum stored next to a bundle only detects corruption; it cannot
    authenticate files to an attacker who can write the same directory. On
    POSIX, reject symlinks, directories owned by another user, and directories
    writable by group or other users. The explicit feature opt-in remains the
    trust boundary on platforms where POSIX ownership and mode bits do not
    describe filesystem access controls.

    Missing directories are allowed so a cache miss can be created safely by
    ``_ensure_private_directory`` during the save path.
    """
    if path is None:
        return True
    try:
        if os.name != "posix":
            try:
                directory_stat = os.lstat(path)
            except FileNotFoundError:
                return allow_missing
            return stat.S_ISDIR(directory_stat.st_mode)

        # Validate the whole path, not just the leaf: a 0700 cache under a
        # non-sticky group/world-writable parent can be renamed and swapped
        # between this check and open(), defeating the leaf permission check.
        target = os.path.abspath(os.fspath(path))
        current = target
        target_exists = False
        while True:
            try:
                directory_stat = os.lstat(current)
                if not stat.S_ISDIR(directory_stat.st_mode):
                    return False
                if current == target:
                    target_exists = True
                    if directory_stat.st_uid != os.geteuid():
                        return False
                writable_by_others = directory_stat.st_mode & 0o022
                # A sticky dir's owner can still rename its entries, so trust
                # the sticky exception only for root- or self-owned parents
                # (/tmp), never an attacker-owned sticky directory.
                sticky_trusted = (directory_stat.st_mode & stat.S_ISVTX) and (
                    directory_stat.st_uid in (0, os.geteuid())
                )
                if writable_by_others and not sticky_trusted:
                    return False
            except FileNotFoundError:
                pass

            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent

        return target_exists or allow_missing
    except Exception:
        return False
pass


def _ensure_private_directory(path):
    """Create a trusted, owner-only cache directory tree.

    Create each missing component with an explicit 0700 mode. That mode
    survives any normal umask (it has no group/other bits to strip), unlike
    ``makedirs`` whose mode applies only to the leaf so a lax umask would leave
    a fresh parent group-writable and then rejected. Pre-existing components
    are left untouched, so a configured root such as
    ``UNSLOTH_MEGA_CACHE_DIR=/tmp`` is never re-permissioned; it is used as-is
    only if ``_is_trusted_directory`` accepts it. No process-wide umask change."""
    try:
        target = os.path.abspath(os.fspath(path))
        if os.name != "posix":
            os.makedirs(target, exist_ok = True)
            return _is_trusted_directory(target)
        components = []
        current = target
        while True:
            components.append(current)
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
        for directory in reversed(components):
            try:
                os.mkdir(directory, 0o700)
            except FileExistsError:
                pass
        return _is_trusted_directory(target)
    except Exception:
        return False
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
            # The bound device, not device 0: mixed-arch hosts must not save
            # one GPU's kernels under another's key.
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
        return None, None
    directory = os.path.join(root, key)
    if not (_is_trusted_directory(root) and _is_trusted_directory(directory)):
        _log("cache directory is not trusted; skipping")
        return None, None
    return directory, os.path.join(directory, _MANIFEST_NAME)
pass


def _read_trusted_file(path, *, binary = False):
    """Read a cache file, refusing symlinks and foreign-owned or group/other
    writable files. Directory trust stops an attacker creating files here, but
    a group-writable file already present can have its bytes (and matching
    checksum) rewritten in place; reject those. Returns None if the file is
    missing or untrusted, and never blocks on a FIFO/device."""
    mode = "rb" if binary else "r"
    if os.name != "posix":
        try:
            with open(path, mode) as file:
                return file.read()
        except OSError:
            return None
    # O_NOFOLLOW rejects a symlinked file; O_NONBLOCK stops a stray FIFO/device
    # here from blocking the open so the S_ISREG check below can reject it.
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return None
    try:
        file_stat = os.fstat(fd)
        if (not stat.S_ISREG(file_stat.st_mode)
                or file_stat.st_uid != os.geteuid()
                or file_stat.st_mode & 0o022):
            return None
        with os.fdopen(fd, mode) as file:
            fd = -1
            return file.read()
    finally:
        if fd >= 0:
            os.close(fd)
pass


def _read_disk_bundle(directory, manifest_path):
    """Read the (manifest, bundle bytes) pair for a key, returning
    ``(None, None)`` unless both files pass ``_read_trusted_file`` and the
    manifest checksum matches the bundle file it points at. The caller must
    establish directory trust first: this checksum detects corruption but does
    not authenticate a bundle against a writer who can also replace both."""
    try:
        manifest_text = _read_trusted_file(manifest_path)
        if manifest_text is None:
            return None, None
        manifest = json.loads(manifest_text)
        bundle_name = os.path.basename(str(manifest.get("bundle", _BUNDLE_NAME)))
        data = _read_trusted_file(os.path.join(directory, bundle_name), binary = True)
        if data is None:
            return None, None
        if manifest.get("sha256") != hashlib.sha256(data).hexdigest():
            return None, None
        return manifest, data
    except Exception:
        return None, None
pass


def _merge_recorded_artifacts(data):
    """Re-record an existing bundle's artifacts into torch's artifact manager
    so the next ``save_cache_artifacts`` serializes the UNION of the on-disk
    bundle and this process' artifacts. torch only records what this run
    loaded-and-hit or compiled, so without this a run that exercises a subset
    of a warm bundle (a different shape, one of several cached models) would
    replace the bundle with that subset and silently drop the rest.
    Best-effort: a failure just means this save may narrow the bundle."""
    try:
        from torch.compiler._cache import CacheArtifactManager
        artifacts = CacheArtifactManager.deserialize(data)
        if not artifacts:
            return
        merged = 0
        for artifact_type, items in artifacts.items():
            for artifact in items:
                # Skip artifacts already present from this run's own hits
                # (value-equality dedup).
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
    directory, manifest_path = _bundle_paths(key)
    if manifest_path is None:
        _log("no trusted cache root; skipping")
        return False

    _STATE["armed"] = True
    _STATE["loaded_key"] = key
    # A hit on an earlier cumulative key does not cover this one: reset so a
    # miss here still saves the extended bundle at exit.
    _STATE["hit"] = False
    if not _STATE["atexit_registered"]:
        atexit.register(megacache_save)
        _STATE["atexit_registered"] = True

    # Missing paths are safe to create later but never safe to read from.
    # Revalidate both dirs as existing right before opening the manifest so an
    # attacker cannot win a create-after-check race under a sticky shared
    # parent such as /tmp.
    root = os.path.dirname(directory)
    if not (
        _is_trusted_directory(root, allow_missing = False)
        and _is_trusted_directory(directory, allow_missing = False)
    ):
        _log(f"no trusted bundle for key {key} (will compile locally and save on exit)")
        return False

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
    """Persist this process' compile artifacts. Runs at exit (atexit) or on
    demand. torch records artifacts on cache HITS as well as writes, but only
    for the entries this run actually exercised - so the on-disk bundle is
    first merged back into the recorder and the saved bundle is always the
    UNION of the existing bundle and this process' artifacts. A run that
    changes nothing serializes back to the same content and is skipped.
    Never raises.
    """
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
        root = os.path.dirname(directory)
        # Establish owned private dirs before reading/deserializing any prior
        # bundle; closes the analogous race on the save path.
        if not (
            _ensure_private_directory(root)
            and _ensure_private_directory(directory)
        ):
            _log("cache directory is not trusted; skipping save")
            return False
        # Fold the on-disk bundle back into the manager before serializing so a
        # run that exercised only part of a warm bundle cannot narrow it (union
        # semantics; see _merge_recorded_artifacts).
        _, existing = _read_disk_bundle(directory, manifest_path)
        if existing is not None:
            _merge_recorded_artifacts(existing)
        result = torch.compiler.save_cache_artifacts()
        if not result or not result[0]:
            _log("nothing to save (no compile artifacts recorded)")
            return False
        data = result[0]
        new_sha = hashlib.sha256(data).hexdigest()
        # Skip rewriting a byte-identical bundle so a no-op run doesn't churn
        # the cache (and its mtime) every exit.
        if existing is not None and hashlib.sha256(existing).hexdigest() == new_sha:
            _log("bundle unchanged; skipping save")
            return False
        # The bundle is content-addressed and the manifest names it, so the
        # pair is consistent without a cross-file lock: concurrent writers
        # produce differently named bundles, the manifest replace is atomic, and
        # the last manifest to land points at its own complete file. Same name
        # always implies same bytes.
        bundle_name = f"megacache-{new_sha[:16]}.bin"
        bundle_path = os.path.join(directory, bundle_name)
        # Content-addressing means a trusted file of this name already holds
        # these bytes, but a leftover from an older writer could be group-
        # writable; skipping the write would leave the manifest pointing at an
        # untrusted bundle that every load rejects. (Re)write unless the
        # existing file already passes the trusted-file checks.
        if _read_trusted_file(bundle_path, binary = True) is None:
            temporary_path = f"{bundle_path}.tmp.{os.getpid()}"
            with open(temporary_path, "wb") as file:
                file.write(data)
            # Owner-only so a same-group user cannot rewrite the bytes later.
            if os.name == "posix":
                os.chmod(temporary_path, 0o600)
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
        if os.name == "posix":
            os.chmod(temporary_manifest, 0o600)
        os.replace(temporary_manifest, manifest_path)
        _log(f"saved bundle for key {key} ({len(data)} bytes)")
        # Best-effort GC of superseded bundle files (including the legacy
        # fixed-name one). A reader that already opened one keeps its handle;
        # a later reader re-reads the manifest and finds the new name.
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
