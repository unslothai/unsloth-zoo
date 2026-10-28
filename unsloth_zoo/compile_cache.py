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

"""Mega-cache of torch.compile artifacts (torch >= 2.7), exact torch/Triton/CUDA/GPU/transformers match only; loads
never raise; UNSLOTH_MEGA_CACHE 0 disables, unset is on for POSIX (untrusted refused, fails closed), opt-in elsewhere
as artifacts are executable; UNSLOTH_MEGA_CACHE_DIR sets root"""

__all__ = [
    "megacache_is_enabled",
    "megacache_load",
    "megacache_save",
    "megacache_status",
]

import atexit
import contextlib
import errno
import hashlib
import json
import os
import stat
import sys
import time

_UNSLOTH_ENABLE_LOGGING = os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1"

_FORMAT_VERSION = 1
_BUNDLE_NAME = "megacache.bin"
_MANIFEST_NAME = "manifest.json"
_LOCK_NAME = ".lock"
# Bounded so a wedged writer cannot hang interpreter exit; on timeout the save is skipped.
_LOCK_TIMEOUT = 60

# The key is the SET of models loaded in this process, not the call order: torch's entries are content-addressed, so order would only mint keys that never hit.
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
    # Bundles are executable torch.compile artifacts: on by default on POSIX, where owner, mode and symlink checks
    # refuse untrusted caches (fail closed); non-POSIX needs opt-in; 0 disables.
    value = os.environ.get("UNSLOTH_MEGA_CACHE", "").strip().lower()
    if value in ("0", "off", "false", "no"):
        return False
    if value in ("1", "on", "true", "yes"):
        return True
    return os.name == "posix"
pass


def _cache_root():
    root = os.path.expanduser(os.environ.get("UNSLOTH_MEGA_CACHE_DIR", ""))
    if root == "":
        home = os.path.expanduser("~")
        # Restricted containers can have no resolvable home; disable rather than write a literal ~ directory into the CWD.
        if home == "~" or not os.path.isdir(home):
            return None
        root = os.path.join(home, ".cache", "unsloth", "mega_cache")
    # Canonicalize with realpath so the validated path is the one later reads and writes use; abspath collapses .. only
    # lexically, so a symlink followed by .. validates one directory while the kernel opens another.
    return os.path.realpath(root)
pass


def _is_trusted_directory(path, *, allow_missing = True):
    """Whether an existing cache dir is safe to load: on POSIX reject symlinks, foreign-owned and group/other-writable
    dirs (a checksum only detects corruption); missing dirs are allowed, to be created by _ensure_private_directory
    on save; non-POSIX relies on the opt-in."""
    if path is None:
        return True
    try:
        if os.name != "posix":
            try:
                directory_stat = os.lstat(path)
            except FileNotFoundError:
                return allow_missing
            return stat.S_ISDIR(directory_stat.st_mode)

        # Validate every ancestor, not just the leaf: a 0700 cache under a non-sticky group/world-writable parent can be
        # renamed and swapped before open(), defeating the leaf permission check.
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
                # Sticky exception only for root- or self-owned parents such as /tmp: a sticky dir's owner can still
                # rename its entries, so never trust an attacker-owned one.
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
    """Create missing components with an explicit 0700 that survives any umask (unlike makedirs, whose mode reaches only
    the leaf); existing components are never re-permissioned, and a configured root is used only if
    _is_trusted_directory accepts it; no process-wide umask change."""
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
        "hip": None,
        "python": "%d.%d" % sys.version_info[:2],
        "triton": None,
        "transformers": None,
        "unsloth_zoo": None,
        "gpu_name": None,
        "gpu_capability": None,
        "gpu_arch": None,
        "xpu_name": None,
    }
    try:
        import torch
        fingerprint["torch"] = str(torch.__version__)
        fingerprint["cuda"] = str(torch.version.cuda)
        fingerprint["hip"] = str(getattr(torch.version, "hip", None))
        if torch.cuda.is_available():
            # Query the bound device, not device 0: on mixed-arch hosts one GPU's kernels must not be saved under another GPU's key.
            device = torch.cuda.current_device()
            fingerprint["gpu_name"] = torch.cuda.get_device_name(device)
            capability = torch.cuda.get_device_capability(device)
            fingerprint["gpu_capability"] = f"sm_{capability[0]}{capability[1]}"
            # ROCm reports one capability for several ISAs (gfx1100/gfx1101).
            if getattr(torch.version, "hip", None):
                fingerprint["gpu_arch"] = getattr(
                    torch.cuda.get_device_properties(device), "gcnArchName", None
                )
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            fingerprint["xpu_name"] = torch.xpu.get_device_name(torch.xpu.current_device())
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


def _register_model_key(entry):
    """Add a model entry to the order-free key set; returns whether it changed."""
    if entry in _STATE["model_keys"]:
        return False
    _STATE["model_keys"] = sorted(
        [*_STATE["model_keys"], entry],
        key = lambda item: json.dumps(item, sort_keys = True, default = str),
    )
    return True
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
    # Validate the per-key dir: its walk also vets the root as an ancestor (sticky root-owned /tmp is fine); the key dir itself must be owned by us.
    if not _is_trusted_directory(directory):
        _log("cache directory is not trusted; skipping")
        return None, None
    return directory, os.path.join(directory, _MANIFEST_NAME)
pass


def _read_trusted_file(path, *, binary = False):
    """Read a cache file only if it is not a symlink, foreign-owned, or group/other-writable (directory trust does not
    stop in-place rewrites); returns None if missing or untrusted; never blocks on a FIFO or device."""
    mode = "rb" if binary else "r"
    if os.name != "posix":
        try:
            with open(path, mode) as file:
                return file.read()
        except OSError:
            return None
    # O_NOFOLLOW rejects a symlinked file; O_NONBLOCK stops a stray FIFO or device blocking the open so the S_ISREG check can reject it.
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


# Kinds torch stores as one file per key are merged by key so this run's copy wins; Inductor / AOTAutograd keep guarded
# variants per key, so they stay by content.
_SINGLE_ENTRY_KINDS = frozenset((
    "AutotuneCacheArtifact", "PGOCacheArtifact", "PrecompileCacheArtifact",
    "AUTOTUNE", "PGO",  # torch 2.7: one CacheArtifact class with a type enum
))
# Only re-timed autotune entries are interchangeable; a changed PGO profile or precompile package is a real update and still counts as new when saving.
_RETIMED_KINDS = frozenset(("AutotuneCacheArtifact", "AUTOTUNE"))


def _artifact_identity(artifact, key_only_kinds = _SINGLE_ENTRY_KINDS):
    kind = type(artifact).__name__
    artifact_type = getattr(artifact, "type", None)
    if kind == "CacheArtifact" and hasattr(artifact_type, "name"):
        kind = artifact_type.name
    key = getattr(artifact, "key", None)
    if key is None:
        return (kind, artifact)
    if kind in key_only_kinds:
        return (kind, key)
    content = getattr(artifact, "content", None)
    if isinstance(content, (bytes, bytearray, memoryview)):
        return (kind, key, hashlib.sha256(content).hexdigest())
    return (kind, key, artifact)
pass


def _merge_recorded_artifacts(data):
    """Re-records the on-disk bundle so saves write the union with this run's artifacts (torch records only what a run
    hits or compiles, else a subset run drops the rest); one copy per identity, this run's wins; returns disk
    identities (None if unreadable); best-effort: a failure may narrow the bundle."""
    try:
        from torch.compiler._cache import CacheArtifactManager
        artifacts = CacheArtifactManager.deserialize(data)
        if not artifacts:
            return None
        disk_ids = set()
        seen_ids = {_artifact_identity(a) for a in list(CacheArtifactManager._seen_artifacts)}
        merged = 0
        for artifact_type, items in list(artifacts.items()):
            for artifact in list(items):
                identity = _artifact_identity(artifact)
                disk_ids.add(_artifact_identity(artifact, _RETIMED_KINDS))
                if identity in seen_ids:
                    continue
                seen_ids.add(identity)
                CacheArtifactManager._new_cache_artifacts[artifact_type].append(artifact)
                CacheArtifactManager._seen_artifacts.add(artifact)
                merged += 1
        if merged:
            _log(f"merged {merged} artifact(s) from the existing bundle")
        return disk_ids
    except Exception as error:
        _log(f"bundle merge skipped ({error})")
    return None
pass


def _recorded_identities():
    """Identities torch's artifact manager holds now, or None."""
    try:
        from torch.compiler._cache import CacheArtifactManager
        return {
            _artifact_identity(a, _RETIMED_KINDS)
            for a in list(CacheArtifactManager._seen_artifacts)
        }
    except Exception:
        return None
pass


def _prepare_lock_file(path):
    """Create the lock file owner-only; refuse anything but a single-link regular file we own, so it cannot be redirected via a symlink or hard link."""
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except OSError:
        return False
    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
            return False
        if os.name == "posix" and (
            file_stat.st_uid != os.geteuid() or file_stat.st_mode & 0o022
        ):
            return False
        return True
    finally:
        os.close(fd)
pass


def _platform_lock(path):
    """Use the OS lock, never SoftFileLock: a killed writer's existence lock would block every later save; older filelock lacks fallback_to_soft and mode."""
    import filelock
    lock_class = filelock.WindowsFileLock if os.name == "nt" else filelock.UnixFileLock
    for kwargs in ({"mode": 0o600, "fallback_to_soft": False}, {"mode": 0o600}, {}):
        try:
            return lock_class(path, timeout = _LOCK_TIMEOUT, **kwargs)
        except TypeError:
            continue
    return lock_class(path, timeout = _LOCK_TIMEOUT)
pass


def _locking_unsupported(error):
    if isinstance(error, (ImportError, NotImplementedError)):
        return True
    unsupported = {getattr(errno, name) for name in ("ENOSYS", "ENOLCK", "EOPNOTSUPP", "ENOTSUP") if hasattr(errno, name)}
    return isinstance(error, OSError) and error.errno in unsupported
pass


def _probe_os_lock(path):
    """Non-blocking flock probe before filelock's acquire: filelock retries every flock error but ENOSYS until the
    timeout, so ENOLCK / EOPNOTSUPP (NFS without a lock manager) would stall exit for _LOCK_TIMEOUT."""
    if os.name != "posix":
        return
    import fcntl
    fd = os.open(path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        if error.errno not in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES):
            raise
    finally:
        os.close(fd)
pass


@contextlib.contextmanager
def _key_lock(directory):
    """Serialize savers of one key across processes, else one writer collects a bundle another's manifest names and
    concurrent unions drop artifacts; yields whether to save (unlocked only where locking is unsupported; a timeout
    or untrusted lock file skips)."""
    lock = None
    may_save = True
    try:
        if os.name not in ("posix", "nt"):
            raise NotImplementedError(os.name)
        path = os.path.join(directory, _LOCK_NAME)
        if _prepare_lock_file(path):
            _probe_os_lock(path)
            lock = _platform_lock(path)
            lock.acquire()
        else:
            _log("lock file is not a private regular file; skipping save")
            may_save = False
    except Exception as error:
        lock = None
        may_save = _locking_unsupported(error)
        if may_save:
            _log(f"save lock unsupported ({error!r}); saving unlocked")
        else:
            _log(f"save lock not acquired ({error!r}); skipping save")
    try:
        yield may_save
    finally:
        if lock is not None:
            try:
                lock.release()
            except Exception:
                pass
pass


def megacache_load(model_type, compile_kwargs = None, torch_compile_options = None):
    """Pre-load artifacts for model_type before any @torch.compile region runs (via unsloth_compile_transformers in
    from_pretrained); registers the exit save; never raises; a miss compiles locally."""
    if not megacache_is_enabled():
        return False
    if not _mega_cache_supported():
        _log("unavailable (needs torch >= 2.7); skipping")
        return False

    _register_model_key(
        _model_key(model_type, compile_kwargs or {}, torch_compile_options or {})
    )
    key = _bundle_key()
    directory, manifest_path = _bundle_paths(key)
    if manifest_path is None:
        _log("no trusted cache root; skipping")
        return False

    _STATE["armed"] = True
    _STATE["loaded_key"] = key
    # A hit on an earlier cumulative key does not cover this one: reset the hit flag so a miss here still saves the extended bundle at exit.
    _STATE["hit"] = False
    if not _STATE["atexit_registered"]:
        atexit.register(megacache_save)
        _STATE["atexit_registered"] = True

    # Missing paths are safe to create but never to read from; revalidate the existing key dir right before opening the
    # manifest (an attacker could win a create-after-check race under a sticky shared parent such as /tmp).
    if not _is_trusted_directory(directory, allow_missing = False):
        _log(f"no trusted bundle for key {key} (will compile locally and save on exit)")
        return False

    manifest, data = _read_disk_bundle(directory, manifest_path)
    if manifest is None and os.path.exists(manifest_path):
        # A concurrent save may have replaced the manifest between our two reads.
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
    """Persist artifacts at exit (atexit) or on demand; the on-disk bundle is merged in first so the saved bundle is the
    union; an unchanged run is skipped; never raises."""
    if not (_STATE["armed"] and megacache_is_enabled() and _mega_cache_supported()):
        return False
    # Only LOCAL_RANK (else RANK) 0 saves: one writer per node since roots default to node-local disks; shared roots
    # rely on the pid-unique temp file and atomic replace.
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
        # Create the owned 0700 key dir before reading or deserializing any prior bundle (closes the save-path race); a
        # pre-existing sticky shared root is left as-is and only vetted as an ancestor.
        if not _ensure_private_directory(directory):
            _log("cache directory is not trusted; skipping save")
            return False
        with _key_lock(directory) as may_save:
            if not may_save:
                return False
            return _save_locked(torch, key, directory, manifest_path)
    except Exception as error:
        _log(f"save failed ({error})")
        return False
pass


def _save_locked(torch, key, directory, manifest_path):
    """Body of ``megacache_save`` run under the per-key lock."""
    # Fold the on-disk bundle into the manager before serializing so a partial run cannot narrow it (union semantics; see _merge_recorded_artifacts).
    _, existing = _read_disk_bundle(directory, manifest_path)
    disk_ids = _merge_recorded_artifacts(existing) if existing is not None else None
    # Always serialize: it also records deferred Dynamo precompile entries.
    result = torch.compiler.save_cache_artifacts()
    if not result or not result[0]:
        _log("nothing to save (no compile artifacts recorded)")
        return False
    data = result[0]
    new_sha = hashlib.sha256(data).hexdigest()
    # Skip a no-op rewrite; bytes alone miss it (recording order, re-timed autotune entries), so identities are compared too.
    if existing is not None and hashlib.sha256(existing).hexdigest() == new_sha:
        _log("bundle unchanged; skipping save")
        return False
    if disk_ids is not None:
        recorded = _recorded_identities()
        if recorded is not None and recorded <= disk_ids:
            _log("bundle already holds every recorded artifact; skipping save")
            return False
    # Content-addressed bundle plus atomic manifest replace keep the pair consistent even for an unlocked (fallback) writer.
    bundle_name = f"megacache-{new_sha[:16]}.bin"
    bundle_path = os.path.join(directory, bundle_name)
    # A same-named leftover may be group-writable (every load would reject it), so rewrite unless it passes the trusted-file checks.
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
    # Best-effort GC of superseded bundles (incl. legacy fixed-name one); a reader keeps its open handle and a later
    # reader re-reads the manifest; per file, as Windows refuses to unlink a held bundle.
    try:
        names = os.listdir(directory)
    except Exception:
        names = ()
    for name in names:
        is_stale_bundle = (
            name == _BUNDLE_NAME
            or (name.startswith("megacache-") and name.endswith(".bin"))
            or ".bin.tmp." in name
        )
        if is_stale_bundle and name != bundle_name:
            try:
                os.unlink(os.path.join(directory, name))
            except Exception:
                pass
    return True
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
