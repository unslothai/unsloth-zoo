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

"""PR #967: the compiled-cache write guard must not be evaluated rank-locally.

distributed_function() broadcasts and barriers, so a rank that decides on its own
whether to call it can skip a collective the other ranks are inside, desynchronise
the group and hang the job on the NCCL watchdog. Rank 0 decides for everyone and
broadcasts both the decision and a digest of the bytes it will import.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import importlib
import importlib.util
import os
import pathlib
import py_compile
import sys
import threading
import time

import pytest

# Importing the compiler pulls in the bitsandbytes and triton substitutes on a host
# that ships neither, which is every macOS and Windows runner. Do it at collection
# time: left to the `compiler` fixture, the substitute for bitsandbytes.nn lands
# inside the first test that uses it, and conftest's sys.modules leak gate
# attributes it to that test and fails its teardown. It has to be the submodule --
# importing the package alone does not reach bitsandbytes.nn. Guarded because the
# static source tests below must still run where the import cannot succeed.
try:
    import unsloth_zoo.compiler  # noqa: F401
except Exception:
    pass


_COMPILER_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "unsloth_zoo"
    / "compiler.py"
)
_CACHE_LEAF = "persistent-cache"


def _compiler_source() -> str:
    """Read compiler.py without importing the unsloth_zoo package."""
    return _COMPILER_PATH.read_text(encoding="utf-8")


def _create_new_function_ast() -> ast.FunctionDef:
    tree = ast.parse(_compiler_source())
    return next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "create_new_function"
    )


@pytest.fixture
def compiler():
    """Runtime compiler module, skipped when the companion package is absent."""
    if importlib.util.find_spec("unsloth") is None:
        pytest.skip("requires the companion `unsloth` package")
    try:
        return importlib.import_module("unsloth_zoo.compiler")
    except ImportError as error:
        if "Please install Unsloth" in str(error):
            pytest.skip("requires the companion `unsloth` package")
        raise


@pytest.fixture(autouse=True)
def _temp_mode_does_not_leak():
    """Catch a prior test that forgot to let monkeypatch restore temp mode."""
    module = sys.modules.get("unsloth_zoo.compiler")
    if module is not None:
        assert module.UNSLOTH_COMPILE_USE_TEMP is False


@pytest.fixture
def probe(compiler):
    """Compile generated probes and always remove both module aliases."""
    names = []

    def create(name, body="return x", *, overwrite=True):
        names.append(name)
        indented_body = "\n".join(f"    {line}" for line in body.splitlines())
        return compiler.create_new_function(
            name,
            f"def {name}_fn(x):\n{indented_body}\n",
            "pr967",
            {},
            overwrite=overwrite,
        )

    yield create
    for name in names:
        sys.modules.pop(name, None)
        sys.modules.pop(f"unsloth_cache_{name}", None)


@pytest.fixture
def cache_dirs(tmp_path):
    primary = tmp_path / "primary"
    temp = tmp_path / "temp"
    primary.mkdir()
    temp.mkdir()
    return primary, temp


def _own_temp_mode(monkeypatch, compiler, value=False):
    """Let monkeypatch restore the global even when production reassigns it."""
    monkeypatch.setattr(compiler, "UNSLOTH_COMPILE_USE_TEMP", value)


def _stub_compile_folders(
    monkeypatch, compiler, primary, temp, *, temp_active=False,
):
    """Route persistent and node-local folders without a real collective."""
    _own_temp_mode(monkeypatch, compiler, temp_active)
    monkeypatch.setattr(
        compiler,
        "get_compile_folder",
        lambda use_tempfile=False: (
            (str(temp), True)
            if (compiler.UNSLOTH_COMPILE_USE_TEMP or use_tempfile)
            else (str(primary), False)
        ),
    )


def _configure_local_temp(tmp_path, monkeypatch, compiler):
    local_temp = tmp_path / "rank1-temp"
    monkeypatch.setattr(
        compiler, "UNSLOTH_COMPILE_LOCATION", str(tmp_path / _CACHE_LEAF),
    )
    _own_temp_mode(monkeypatch, compiler)
    monkeypatch.setattr(compiler.tempfile, "gettempdir", lambda: str(local_temp))
    return local_temp


@pytest.mark.parametrize(
    "fallback_kind", ["explicit", "rank0-fallback"],
)
def test_temp_compile_folder_is_resolved_locally(
    tmp_path, monkeypatch, compiler, fallback_kind,
):
    """Explicit and rank-0 fallback paths both resolve temp locally."""
    local_temp = _configure_local_temp(tmp_path, monkeypatch, compiler)
    calls = []

    if fallback_kind == "explicit":
        monkeypatch.setattr(
            compiler,
            "distributed_function",
            lambda *_args, **_kwargs: pytest.fail(
                "an explicit node-local temp path was broadcast"
            ),
        )
    else:
        rank0_temp = f"/rank0-private/tmp/{_CACHE_LEAF}"

        def rank0_fell_back(n, function, *args, **kwargs):
            calls.append((n, function, args, kwargs))
            return rank0_temp, True

        monkeypatch.setattr(compiler, "distributed_function", rank0_fell_back)

    location, use_temp = compiler.get_compile_folder(
        use_tempfile=fallback_kind == "explicit",
    )

    assert len(calls) == (0 if fallback_kind == "explicit" else 1)
    assert pathlib.Path(location) == local_temp / _CACHE_LEAF
    assert pathlib.Path(location).is_dir()
    if fallback_kind == "rank0-fallback":
        assert location != rank0_temp
    assert use_temp is True


@pytest.mark.parametrize("local_temp_active", [False, True])
def test_temp_mode_is_agreed_before_skipping_folder_collective(
    tmp_path, monkeypatch, compiler, local_temp_active,
):
    """Divergent pre-group fallback state must converge before any rank returns."""
    local_temp = _configure_local_temp(tmp_path, monkeypatch, compiler)
    _own_temp_mode(monkeypatch, compiler, local_temp_active)
    monkeypatch.setattr(compiler, "torch_distributed_is_initialized", lambda: True)
    calls = []
    monkeypatch.setattr(
        compiler,
        "distributed_any",
        lambda value: calls.append(value) or (
            True if len(calls) == 1 else bool(value)
        ),
    )
    monkeypatch.setattr(
        compiler,
        "distributed_function",
        lambda *_args, **_kwargs: pytest.fail(
            "a rank entered the folder broadcast after temp mode was agreed"
        ),
    )

    location, use_temp = compiler.get_compile_folder(use_tempfile=False)

    # First agree temp mode, then agree that local directory creation succeeded.
    assert calls == [local_temp_active, False]
    assert pathlib.Path(location) == local_temp / _CACHE_LEAF
    assert use_temp is True


def test_persistent_folder_failure_defers_temp_creation_to_each_rank(
    tmp_path, monkeypatch, compiler,
):
    """Rank 0 must broadcast fallback mode before resolving any local temp path."""
    persistent = tmp_path / "persistent"
    _own_temp_mode(monkeypatch, compiler)
    monkeypatch.setattr(compiler, "UNSLOTH_COMPILE_LOCATION", str(persistent))
    monkeypatch.setattr(
        compiler.os,
        "makedirs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            PermissionError("simulated persistent path failure")
        ),
    )
    monkeypatch.setattr(
        compiler.tempfile,
        "gettempdir",
        lambda: pytest.fail("rank 0 resolved its temp path before broadcasting"),
    )

    location, use_temp = compiler._get_compile_folder(use_tempfile=False)

    assert location is None
    assert use_temp is True
    assert compiler.UNSLOTH_COMPILE_USE_TEMP is True


@pytest.mark.parametrize("failure_site", ["temp-folder", "verification"])
@pytest.mark.parametrize("local_failure", [False, True], ids=["remote", "local"])
def test_rank_local_failures_are_agreed(
    tmp_path, monkeypatch, compiler, failure_site, local_failure,
):
    """Local and remote failures must make every rank raise together."""
    monkeypatch.setattr(compiler, "torch_distributed_is_initialized", lambda: True)
    local_error = (
        PermissionError("simulated local failure") if local_failure else None
    )
    calls = []
    monkeypatch.setattr(
        compiler,
        "distributed_any",
        lambda value: calls.append(value) or True,
    )

    expected_error = PermissionError if local_failure else RuntimeError
    expected_message = "local failure" if local_failure else "failed on another rank"

    if failure_site == "temp-folder":
        _own_temp_mode(monkeypatch, compiler, True)
        monkeypatch.setattr(
            compiler,
            "_get_compile_folder",
            lambda **_kwargs: (
                (str(tmp_path / "local-temp"), True)
                if local_error is None
                else (_ for _ in ()).throw(local_error)
            ),
        )
        operation = lambda: compiler.get_compile_folder(use_tempfile=False)
        expected_calls = [True, local_failure]
    else:
        monkeypatch.setattr(
            compiler,
            "_verify_compiled_cache_file",
            lambda *_args: (
                None
                if local_error is None
                else (_ for _ in ()).throw(local_error)
            ),
        )
        operation = lambda: compiler._verify_compiled_cache_file_collectively(
            str(tmp_path / "mod.py"), "digest",
        )
        expected_calls = [local_failure]

    with pytest.raises(expected_error, match=expected_message):
        operation()

    assert calls == expected_calls


def test_decision_is_broadcast_not_computed_locally(
    tmp_path, monkeypatch, compiler, probe,
):
    """Every rank must take the branch rank 0 chose."""
    calls = []
    real = compiler.distributed_function

    def spy(n, function, *args, **kwargs):
        calls.append(getattr(function, "__name__", str(function)))
        return real(n, function, *args, **kwargs)

    monkeypatch.setattr(compiler, "distributed_function", spy)
    monkeypatch.setattr(compiler, "UNSLOTH_COMPILE_LOCATION", str(tmp_path))
    _own_temp_mode(monkeypatch, compiler)
    monkeypatch.syspath_prepend(str(tmp_path))

    name = "pr967_probe"
    module = probe(name)

    assert (tmp_path / f"{name}.py").is_file(), "probe did not compile into tmp_path"
    assert pathlib.Path(module.__file__).parent == tmp_path, (
        "a successful persistent-cache import was mistaken for a failure and "
        "unnecessarily recovered to the tempfile cache"
    )
    assert compiler.UNSLOTH_COMPILE_USE_TEMP is False
    assert "_compiled_cache_decision" in calls, (
        "the write decision was not routed through distributed_function(); it is "
        "rank-local again -- regression of PR #967."
    )


def test_unreadable_preexisting_cache_reaches_write_decision(
    monkeypatch, compiler, probe, cache_dirs,
):
    """A rank-local read error must not escape before the first collective."""
    primary, temp = cache_dirs
    _stub_compile_folders(monkeypatch, compiler, primary, temp)
    name = "pr967_unreadable_precheck"
    location = primary / f"{name}.py"
    location.write_text("stale source", encoding="utf-8")
    real_open = builtins.open
    monkeypatch.setenv("UNSLOTH_COMPILE_OVERWRITE", "0")

    def fail_text_read(file, mode="r", *args, **kwargs):
        if pathlib.Path(file) == location and mode == "r":
            raise PermissionError("simulated rank-local cache read failure")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fail_text_read)

    module = probe(name, "return x * 2", overwrite=False)

    assert getattr(module, f"{name}_fn")(21) == 42
    assert b"return x * 2" in location.read_bytes()


def test_verified_cache_precedes_shadowing_sys_path_entry(
    tmp_path, monkeypatch, compiler, probe, cache_dirs,
):
    """Import the verified cache even when another directory has the same name."""
    primary, temp = cache_dirs
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    _stub_compile_folders(monkeypatch, compiler, primary, temp)
    name = "pr967_shadowed_cache"
    (shadow / f"{name}.py").write_text(
        f"def {name}_fn(x):\n    return x * 99\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(primary))
    monkeypatch.syspath_prepend(str(shadow))

    module = probe(name, "return x * 2")

    assert pathlib.Path(module.__file__).parent == primary
    assert getattr(module, f"{name}_fn")(21) == 42


def test_direct_recovery_invalidates_temp_folder_bytecode(
    monkeypatch, compiler, probe, cache_dirs,
):
    """Folder-switching recovery must discard a stale temp-folder pyc."""
    primary, temp = cache_dirs
    _stub_compile_folders(
        monkeypatch, compiler, primary, temp, temp_active=True,
    )
    name = "pr967_direct_pyc"
    first = probe(name, "return x * 2")
    assert getattr(first, f"{name}_fn")(21) == 42

    temp_source = temp / f"{name}.py"
    pyc = pathlib.Path(importlib.util.cache_from_source(str(temp_source)))
    stamp = int(time.time()) + 30
    os_times = (stamp, stamp)
    temp_source.touch()
    os.utime(temp_source, os_times)
    py_compile.compile(str(temp_source), cfile=str(pyc), doraise=True)

    # Re-enter through the persistent folder, fail that import, then switch back
    # to temp. Pin the rewritten temp source to the pyc's timestamp and size.
    compiler.UNSLOTH_COMPILE_USE_TEMP = False
    real_fsync = compiler.os.fsync

    def fsync_and_pin_temp(fd):
        real_fsync(fd)
        if temp_source.exists():
            os.utime(temp_source, os_times)

    real_import = compiler.importlib.import_module
    failed = False

    def fail_persistent_import_once(module_name, package=None):
        nonlocal failed
        if module_name == name and not failed:
            failed = True
            raise ImportError("force folder-switching recovery")
        return real_import(module_name, package)

    monkeypatch.setattr(compiler.os, "fsync", fsync_and_pin_temp)
    monkeypatch.setattr(
        compiler.importlib, "import_module", fail_persistent_import_once,
    )

    updated = probe(name, "return x * 3")

    assert getattr(updated, f"{name}_fn")(21) == 63


def test_undeletable_stale_bytecode_does_not_abandon_the_persistent_cache(
    monkeypatch, compiler, probe, cache_dirs,
):
    """A pyc we cannot unlink but CPython would reject must cost nothing.

    On Windows `os.remove` raises PermissionError whenever any other handle is
    open on the file, which a virus scanner, a sync client or a second
    interpreter all do routinely, and the compiled-cache lock excludes none of
    them. Treating that as fatal moved the whole group onto tempfile recovery
    on the first occurrence and aborted the job on the second, so an import
    that had always worked started failing. Here the rewrite changed the source
    size, so the pyc is already invalid and continuing is safe.
    """
    primary, temp = cache_dirs
    _stub_compile_folders(monkeypatch, compiler, primary, temp)
    real_remove = compiler.os.remove

    def deny_primary_pyc_removal(path):
        if primary in pathlib.Path(path).parents:
            raise PermissionError("simulated read-only pycache")
        return real_remove(path)

    monkeypatch.setattr(compiler.os, "remove", deny_primary_pyc_removal)
    name = "pr967_undeletable_pyc"

    module = probe(name, "return x * 2")

    assert pathlib.Path(module.__file__).parent == primary, (
        "a pyc that could not be unlinked pushed the cache into tempfile "
        "recovery, which is how this turned into a hard failure on Windows."
    )
    assert getattr(module, f"{name}_fn")(21) == 42


def test_undeletable_usable_bytecode_still_fails_over(
    monkeypatch, compiler, probe, cache_dirs, tmp_path,
):
    """A pyc CPython would still accept is the case the removal exists for.

    Same size, same timestamp second: importlib accepts the old bytecode, so
    source-digest verification would pass while this rank executes the previous
    implementation. Tolerating the unlink failure here is the one shape that is
    not safe, so it must still raise rather than warn.
    """
    primary, temp = cache_dirs
    _stub_compile_folders(monkeypatch, compiler, primary, temp)
    name = "pr967_usable_pyc"
    probe(name, "return x * 2")

    source = primary / f"{name}.py"
    bytecode = pathlib.Path(importlib.util.cache_from_source(str(source)))
    py_compile.compile(str(source), cfile=str(bytecode), doraise=True)

    # Compiled from this exact source, so the header records its mtime and size
    # and CPython would load it. No stubbing needed to reach the unsafe shape.
    assert compiler._bytecode_would_be_used(str(source), str(bytecode))

    monkeypatch.setattr(
        compiler.os, "remove",
        lambda *_a, **_k: (_ for _ in ()).throw(
            PermissionError("simulated locked pycache")),
    )

    with pytest.raises(RuntimeError, match="Cannot remove stale bytecode"):
        compiler._remove_compiled_cache_bytecode(str(source))


@pytest.mark.parametrize("overwrite", [False, True])
def test_warm_cache_removes_no_bytecode(
    monkeypatch, compiler, probe, cache_dirs, overwrite,
):
    """An unchanged cache is imported as-is, bytecode included.

    Every process start walks this path once per generated module. Deleting
    the pyc here forced CPython to re-parse and re-compile the whole generated
    source on every single import, and on Windows it is also the step that can
    fail.

    overwrite=True is parametrised because it is the DEFAULT, and the one both
    unsloth_compile_transformers and patch_lora_forwards use. It only grants
    permission to rewrite; when write_file() finds the bytes identical it writes
    nothing, so there is still no stale bytecode. Gating on the write decision
    rather than on the write left this case removing the pyc every time.
    """
    primary, temp = cache_dirs
    _stub_compile_folders(monkeypatch, compiler, primary, temp)
    name = f"pr967_warm_no_pyc_churn_{int(overwrite)}"
    probe(name, "return x * 2", overwrite=overwrite)

    removed = []
    real_remove = compiler.os.remove

    def record_removal(path):
        removed.append(str(path))
        return real_remove(path)

    monkeypatch.setattr(compiler.os, "remove", record_removal)
    sys.modules.pop(name, None)

    module = probe(name, "return x * 2", overwrite=overwrite)

    assert getattr(module, f"{name}_fn")(21) == 42
    assert not [p for p in removed if p.endswith(".pyc")], (
        f"an unchanged import removed bytecode (overwrite={overwrite}): {removed}"
    )


def test_rewriting_the_cache_still_removes_stale_bytecode(
    monkeypatch, compiler, probe, cache_dirs,
):
    """The rewrite path keeps the defence the warm path no longer needs.

    Pins that the gate is on "did this call rewrite the file", not on some
    weaker condition that would let a same-size rewrite import stale bytecode.
    """
    primary, temp = cache_dirs
    _stub_compile_folders(monkeypatch, compiler, primary, temp)
    name = "pr967_rewrite_drops_pyc"
    probe(name, "return x * 2", overwrite=False)

    removed = []
    real_remove = compiler.os.remove

    def record_removal(path):
        removed.append(str(path))
        return real_remove(path)

    monkeypatch.setattr(compiler.os, "remove", record_removal)
    sys.modules.pop(name, None)

    module = probe(name, "return x * 3", overwrite=True)

    assert getattr(module, f"{name}_fn")(21) == 63
    assert [p for p in removed if p.endswith(".pyc")], (
        "a rewrite left stale bytecode in place"
    )


@pytest.mark.parametrize("load_path", ["normal", "direct"])
def test_import_rechecks_digest_while_holding_lock(
    monkeypatch, compiler, probe, cache_dirs, load_path,
):
    """Normal and direct loads must reject rewrites after collective verification."""
    primary, temp = cache_dirs
    _stub_compile_folders(monkeypatch, compiler, primary, temp)
    monkeypatch.setattr(compiler, "torch_distributed_is_initialized", lambda: True)
    name = f"pr967_{load_path}_locked_digest"
    location = (primary if load_path == "normal" else temp) / f"{name}.py"
    real_get_lock = compiler.get_lock
    target_lock_count = 0

    class RewriteAfterLock:
        def __init__(self, lock):
            self.lock = lock

        def __enter__(self):
            value = self.lock.__enter__()
            location.write_text(
                f"def {name}_fn(x):\n    return x * 99\n",
                encoding="utf-8",
            )
            return value

        def __exit__(self, *args):
            return self.lock.__exit__(*args)

    def rewrite_on_import_lock(target, *args, **kwargs):
        nonlocal target_lock_count
        lock = real_get_lock(target, *args, **kwargs)
        if pathlib.Path(target) == location:
            target_lock_count += 1
            if target_lock_count == 2:
                return RewriteAfterLock(lock)
        return lock

    monkeypatch.setattr(compiler, "get_lock", rewrite_on_import_lock)

    if load_path == "direct":
        real_import = compiler.importlib.import_module
        failed = False

        def fail_initial_import(module_name, package=None):
            nonlocal failed
            if module_name == name and not failed:
                failed = True
                raise ImportError("force direct-load recovery")
            return real_import(module_name, package)

        monkeypatch.setattr(compiler.importlib, "import_module", fail_initial_import)
        with pytest.raises(RuntimeError, match="Direct module loading failed"):
            probe(name, "return x * 2")
        assert name not in sys.modules
        assert f"unsloth_cache_{name}" not in sys.modules
    else:
        module = probe(name, "return x * 2")

        assert pathlib.Path(module.__file__).parent == temp
        assert getattr(module, f"{name}_fn")(21) == 42


def test_decision_digests_disk_not_generated_source(
    tmp_path, monkeypatch, compiler,
):
    """UNSLOTH_COMPILE_OVERWRITE=0 keeps a cache file that != write_new_source.

    Digesting write_new_source instead of the bytes on disk would make every rank
    reject a cache file rank 0 deliberately kept.
    """
    monkeypatch.setattr(compiler, "torch_distributed_is_initialized", lambda: True)
    location = tmp_path / "mod.py"
    location.write_bytes(b"older cached source")

    should_write, digest = compiler._compiled_cache_decision(
        str(location), "freshly generated source", False,
    )
    assert should_write is False
    assert digest == hashlib.sha256(b"older cached source").hexdigest(), (
        "the digest describes write_new_source rather than the file rank 0 will "
        "import, which breaks UNSLOTH_COMPILE_OVERWRITE=0."
    )


def test_unknown_rank0_digest_forces_regeneration(
    tmp_path, monkeypatch, compiler,
):
    """Unreadable rank-0 bytes must not degrade to existence-only agreement."""
    location = tmp_path / "mod.py"
    location.write_bytes(b"retained bytes")
    source = "fresh generated source"
    real_open = builtins.open
    monkeypatch.setattr(compiler, "torch_distributed_is_initialized", lambda: True)

    def fail_rank0_binary_read(file, mode="r", *args, **kwargs):
        if pathlib.Path(file) == location and mode == "rb":
            raise PermissionError("simulated rank-0 digest failure")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fail_rank0_binary_read)

    should_write, digest = compiler._compiled_cache_decision(
        str(location), source, False,
    )

    assert should_write is True
    assert digest == hashlib.sha256(source.encode()).hexdigest()


def test_decision_skips_hashing_without_process_group(
    tmp_path, monkeypatch, compiler,
):
    """A launched process without a group keeps the old independent behavior."""
    monkeypatch.setattr(compiler, "torch_distributed_is_initialized", lambda: False)
    location = tmp_path / "mod.py"

    assert compiler._compiled_cache_decision(str(location), "src", False) == (True, None)
    location.write_bytes(b"cached")
    assert compiler._compiled_cache_decision(str(location), "src", False) == (False, None)
    assert compiler._compiled_cache_decision(str(location), "src", True) == (True, None)


def test_collective_verification_skips_without_process_group(
    tmp_path, monkeypatch, compiler,
):
    """Do not claim cross-rank agreement before collectives are available."""
    monkeypatch.setattr(compiler, "torch_distributed_is_initialized", lambda: False)
    monkeypatch.setattr(
        compiler,
        "_verify_compiled_cache_file",
        lambda *_args: pytest.fail("verification ran without a process group"),
    )
    monkeypatch.setattr(
        compiler,
        "distributed_any",
        lambda *_args: pytest.fail("collective agreement ran without a process group"),
    )

    assert compiler._cache_verification_error(
        str(tmp_path / "mod.py"), "digest",
    ) is None


def test_verify_rejects_divergent_file(tmp_path, monkeypatch, compiler):
    """A stale-but-present file must not be silently imported."""
    monkeypatch.setattr(compiler, "_COMPILED_CACHE_VISIBILITY_TIMEOUT", 0.2)
    location = tmp_path / "mod.py"
    location.write_bytes(b"stale bytes")
    rank0_digest = hashlib.sha256(b"current bytes").hexdigest()

    with pytest.raises(RuntimeError, match="differs between rank 0"):
        compiler._verify_compiled_cache_file(str(location), rank0_digest)


def test_verify_reports_missing_file(tmp_path, monkeypatch, compiler):
    monkeypatch.setattr(compiler, "_COMPILED_CACHE_VISIBILITY_TIMEOUT", 0.2)
    location = tmp_path / "never_written.py"

    with pytest.raises(FileNotFoundError, match="not readable on rank"):
        compiler._verify_compiled_cache_file(
            str(location), hashlib.sha256(b"x").hexdigest(),
        )


def test_verify_waits_out_filesystem_lag(tmp_path, monkeypatch, compiler):
    """A network filesystem may publish the file just after the barrier."""
    monkeypatch.setattr(compiler, "_COMPILED_CACHE_VISIBILITY_TIMEOUT", 5.0)
    location = tmp_path / "mod.py"
    digest = hashlib.sha256(b"late bytes").hexdigest()

    threading.Timer(0.5, lambda: location.write_bytes(b"late bytes")).start()
    started = time.monotonic()
    compiler._verify_compiled_cache_file(str(location), digest)  # must not raise
    assert time.monotonic() - started >= 0.4, "did not actually wait for the file"


@pytest.mark.parametrize("is_rank_zero", [True, False], ids=["rank0", "nonzero_rank"])
def test_write_path_verifies_the_file_on_every_rank(
    tmp_path, monkeypatch, is_rank_zero, compiler, probe,
):
    """A rank-local stale file must be checked even when rank 0 writes.

    Parametrised over the rank, because rank 0 is the only rank that writes: a
    verification guarded by is_main_process() would leave every other rank --
    the ones that can actually hold a divergent file -- unchecked, and a
    single-process test cannot tell the difference.
    """
    verified = []
    monkeypatch.setattr(compiler, "torch_distributed_is_initialized", lambda: True)
    monkeypatch.setattr(compiler, "is_main_process", lambda: is_rank_zero)
    _stub_compile_folders(
        monkeypatch, compiler, tmp_path, tmp_path / "temp-cache",
    )
    monkeypatch.setattr(
        compiler,
        "_cache_verification_error",
        lambda location, digest, *_args: verified.append((location, digest)),
    )

    name = "pr967_verify_after_write"
    probe(name)

    assert len(verified) == 1
    assert verified[0][0] == str(tmp_path / f"{name}.py")
    assert verified[0][1] is not None


@pytest.mark.parametrize("write_failure", ["silent", "rank0-only"])
def test_write_failure_recovers_to_node_local_temp(
    tmp_path, monkeypatch, compiler, probe, write_failure,
):
    """Missing persistent writes must recover on every local node."""
    cache = tmp_path / "cache"
    recovery = tmp_path / "recovery"
    cache.mkdir()
    recovery.mkdir()

    _stub_compile_folders(monkeypatch, compiler, cache, recovery)
    monkeypatch.setattr(compiler, "torch_distributed_is_initialized", lambda: True)
    monkeypatch.setattr(compiler, "distributed_any", lambda value: bool(value))
    monkeypatch.setattr(compiler, "_COMPILED_CACHE_VISIBILITY_TIMEOUT", 0.2)

    name = f"pr967_{write_failure.replace('-', '_')}_write_failure"
    blocked = cache / f"{name}.py"
    if write_failure == "silent":
        blocked.write_text("stale = 1\n")

    real_distributed_function = compiler.distributed_function

    def report_rank0_write_without_local_bytes(n, function, *args, **kwargs):
        if getattr(function, "__name__", "") == "write_file_outcome" and (
            write_failure == "rank0-only" or args[0] == str(blocked)
        ):
            # (ok, error, changed): rank 0 claims a clean write that never
            # reached this rank's disk.
            return True, "", True
        return real_distributed_function(n, function, *args, **kwargs)

    monkeypatch.setattr(
        compiler, "distributed_function", report_rank0_write_without_local_bytes,
    )

    module = probe(name, "return x * 2")

    assert pathlib.Path(module.__file__).parent == recovery
    assert getattr(module, f"{name}_fn")(21) == 42
    assert (recovery / f"{name}.py").is_file()
    if write_failure == "silent":
        assert blocked.read_text() == "stale = 1\n"
    else:
        assert not blocked.exists()


@pytest.mark.parametrize(
    ("cache_mode", "local_bytes"),
    [
        ("temp", None),
        ("temp", b"stale node-local source"),
        ("persistent", b"stale persistent source"),
    ],
    ids=["temp-missing", "temp-stale", "persistent-stale"],
)
def test_inconsistent_retained_cache_recovers_from_rank0(
    monkeypatch, compiler, probe, cache_dirs, cache_mode, local_bytes,
):
    """Missing/stale caches recover from rank 0's retained bytes."""
    primary, temp = cache_dirs
    active_folder = temp if cache_mode == "temp" else primary
    name = f"pr967_{cache_mode}_retained"
    location = active_folder / f"{name}.py"
    if local_bytes is not None:
        location.write_bytes(local_bytes)

    rank0_source = f"def {name}_fn(x):\n    return x * 2\n"
    rank0_digest = hashlib.sha256(rank0_source.encode()).hexdigest()

    monkeypatch.setenv("UNSLOTH_COMPILE_OVERWRITE", "0")
    monkeypatch.setattr(compiler, "torch_distributed_is_initialized", lambda: True)
    monkeypatch.setattr(compiler, "distributed_any", lambda value: bool(value))
    _stub_compile_folders(
        monkeypatch,
        compiler,
        primary,
        temp,
        temp_active=cache_mode == "temp",
    )

    def rank0_retained_its_copy(n, function, *args, **kwargs):
        if function is compiler._compiled_cache_decision:
            return False, rank0_digest
        if function is compiler._retained_cache_source:
            return rank0_source, rank0_digest, ""
        pytest.fail(f"unexpected distributed function: {function.__name__}")

    monkeypatch.setattr(
        compiler, "distributed_function", rank0_retained_its_copy,
    )

    seen_timeouts = []
    real_verification = compiler._cache_verification_error

    def record_timeout(location, digest, visibility_timeout=None):
        seen_timeouts.append(visibility_timeout)
        return real_verification(location, digest, 0)

    monkeypatch.setattr(
        compiler, "_cache_verification_error", record_timeout,
    )
    module = probe(name, "return x * 9", overwrite=False)

    assert pathlib.Path(module.__file__).parent == temp
    assert getattr(module, f"{name}_fn")(21) == 42
    assert "return x * 2" in (temp / f"{name}.py").read_text(encoding="utf-8")
    expected_timeout = 0 if cache_mode == "temp" else None
    assert seen_timeouts and seen_timeouts[0] == expected_timeout


def test_unreadable_rank0_retained_cache_regenerates(
    monkeypatch, compiler, probe, cache_dirs,
):
    """Unreadable retained bytes fall back to rank 0's generated source."""
    primary, temp = cache_dirs
    _stub_compile_folders(monkeypatch, compiler, primary, temp)
    monkeypatch.setenv("UNSLOTH_COMPILE_OVERWRITE", "0")
    monkeypatch.setattr(compiler, "torch_distributed_is_initialized", lambda: True)
    monkeypatch.setattr(compiler, "distributed_any", lambda value: bool(value))
    name = "pr967_unreadable_retained"
    rank0_source = f"def {name}_fn(x):\n    return x * 2\n"
    rank0_digest = hashlib.sha256(rank0_source.encode()).hexdigest()

    def rank0_cache_state(n, function, *args, **kwargs):
        if function is compiler._compiled_cache_decision:
            return False, None
        if function is compiler._retained_cache_source:
            return None, None, "PermissionError: simulated unreadable cache"
        if function is compiler._generated_cache_source:
            return rank0_source, rank0_digest
        pytest.fail(f"unexpected distributed function: {function.__name__}")

    verification_calls = 0
    repaired_digests = []
    real_verification = compiler._cache_verification_error

    def fail_retained_verification(location, digest, visibility_timeout=None):
        nonlocal verification_calls
        verification_calls += 1
        if verification_calls == 1:
            return PermissionError("simulated unreadable retained cache")
        repaired_digests.append(digest)
        return real_verification(location, digest, visibility_timeout)

    monkeypatch.setattr(compiler, "distributed_function", rank0_cache_state)
    monkeypatch.setattr(
        compiler, "_cache_verification_error", fail_retained_verification,
    )

    module = probe(name, "return x * 9")

    assert pathlib.Path(module.__file__).parent == temp
    assert getattr(module, f"{name}_fn")(21) == 42
    assert repaired_digests == [rank0_digest]


def test_write_file_is_never_run_bare_inside_the_collective():
    """get_lock() raises outside write_file()'s own guard.

    distributed_function() runs the function on rank 0 before broadcasting, so a
    raise there abandons the broadcast the other ranks are already inside and the
    group mis-pairs its next collective (gloo aborts on the payload size).
    """
    func = _create_new_function_ast()
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "id", None) != "distributed_function":
            continue
        if len(node.args) < 2:
            continue
        assert getattr(node.args[1], "id", None) != "write_file", (
            "distributed_function() runs write_file() directly; a get_lock() failure "
            "on a read-only cache directory then leaves rank 0 outside the broadcast."
        )


def test_verification_never_raises_rank_locally_from_create_new_function():
    """Every call site must go through the collective wrapper.

    _verify_compiled_cache_file() raises on the rank that sees the mismatch
    while the others walk into the next collective, which is the very hang the
    verifier exists to diagnose. The behavioural tests cannot catch a call site
    reverting to it, because a single process cannot tell the two apart.
    """
    func = _create_new_function_ast()
    bare = [
        node for node in ast.walk(func)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "_verify_compiled_cache_file"
    ]
    assert not bare, (
        "create_new_function calls _verify_compiled_cache_file directly; it must "
        "use _verify_compiled_cache_file_collectively or _cache_verification_error "
        "so that a mismatch on one rank is agreed before any rank raises."
    )


def test_import_recovery_is_agreed_across_ranks():
    """An import can fail on one rank only, but its recovery is collective."""
    func = _create_new_function_ast()
    guards = [
        node for node in ast.walk(func)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Call)
        and getattr(node.test.func, "id", None) == "distributed_any"
    ]
    assert guards, (
        "the tempfile import recovery is entered on a rank-local exception; a rank "
        "taking it alone calls get_compile_folder() and distributed_function(), "
        "which the other ranks never reach."
    )
    assert any(
        "get_compile_folder" in ast.dump(guard) for guard in guards
    ), "distributed_any() does not guard the collective part of the import recovery."


def test_write_failure_falls_back_to_tempfile(
    tmp_path, monkeypatch, compiler, probe,
):
    """A read-only cache directory must reach the tempfile fallback, not raise."""
    cache = tmp_path / "readonly_cache"
    cache.mkdir()
    monkeypatch.setattr(compiler, "UNSLOTH_COMPILE_LOCATION", str(cache))
    _own_temp_mode(monkeypatch, compiler)
    monkeypatch.setattr(
        compiler.tempfile,
        "gettempdir",
        lambda: str(tmp_path / "node-temp"),
    )

    real_get_lock = compiler.get_lock

    def fake_get_lock(target, *args, **kwargs):
        if str(target).startswith(str(cache)):
            raise OSError("simulated read-only cache directory")
        return real_get_lock(target, *args, **kwargs)

    monkeypatch.setattr(compiler, "get_lock", fake_get_lock)

    name = "pr967_readonly"
    module = probe(name, "return x * 3")

    assert module.pr967_readonly_fn(14) == 42
    assert not (cache / f"{name}.py").exists(), (
        "the write was expected to fail against the read-only directory"
    )


def test_current_rank_without_process_group(monkeypatch, compiler):
    """RANK is unset outside a launcher, and get_rank() would raise."""
    monkeypatch.delenv("RANK", raising=False)
    assert str(compiler.current_rank()) == "0"
    monkeypatch.setenv("RANK", "3")
    assert str(compiler.current_rank()) == "3"


def test_import_recovery_message_does_not_blame_this_rank():
    """A rank whose own import succeeded must not log 'failed ... : None'."""
    src = _compiler_source()
    assert "Standard import failed for {name}: {import_error}" not in src, (
        "import_error is None on a rank that imported fine but was pulled onto the "
        "recovery by another rank, so this logs a bare None."
    )
    assert 'reason = import_error or "an import failure on another rank"' in src


def test_failed_import_restores_sys_path(
    monkeypatch, compiler, probe, cache_dirs,
):
    """A local import failure must not leave the persistent cache on sys.path."""
    primary, temp = cache_dirs
    monkeypatch.setattr(compiler, "torch_distributed_is_initialized", lambda: True)
    _stub_compile_folders(monkeypatch, compiler, primary, temp)

    name = "pr967_import_path_cleanup"
    real_import = compiler.importlib.import_module
    failed = False

    def fail_once(module_name, package=None):
        nonlocal failed
        if module_name == name and not failed:
            failed = True
            raise ImportError("simulated local import failure")
        return real_import(module_name, package)

    monkeypatch.setattr(compiler.importlib, "import_module", fail_once)

    module = probe(name, "return x * 2")

    assert getattr(module, f"{name}_fn")(21) == 42
    assert primary not in map(pathlib.Path, sys.path)


def test_failed_direct_load_removes_partial_module_alias(
    monkeypatch, compiler, cache_dirs,
):
    """Direct execution should clean sys.modules like importlib does."""
    primary, temp = cache_dirs
    _stub_compile_folders(monkeypatch, compiler, primary, temp)
    name = "pr967_partial_direct_load"
    source = (
        "raise RuntimeError('simulated module-body failure')\n"
        f"def {name}_fn(x):\n"
        "    return x\n"
    )

    with pytest.raises(RuntimeError, match="Direct module loading failed"):
        compiler.create_new_function(name, source, "pr967", {}, overwrite=True)

    assert name not in sys.modules
    assert f"unsloth_cache_{name}" not in sys.modules


def test_remote_direct_load_failure_removes_successful_local_aliases(
    monkeypatch, compiler, probe, cache_dirs,
):
    """A global recovery failure must leave no importable module on any rank."""
    primary, temp = cache_dirs
    monkeypatch.setattr(compiler, "torch_distributed_is_initialized", lambda: True)
    _stub_compile_folders(monkeypatch, compiler, primary, temp)
    name = "pr967_remote_direct_failure"
    real_import = compiler.importlib.import_module
    failed = False

    def fail_initial_import(module_name, package=None):
        nonlocal failed
        if module_name == name and not failed:
            failed = True
            raise ImportError("force tempfile recovery")
        return real_import(module_name, package)

    real_agreed_error = compiler._agreed_error

    def fail_direct_load_remotely(local_error, operation):
        if operation.startswith("Direct module loading"):
            assert local_error is None
            return RuntimeError("direct load failed on another rank")
        return real_agreed_error(local_error, operation)

    monkeypatch.setattr(compiler.importlib, "import_module", fail_initial_import)
    monkeypatch.setattr(compiler, "_agreed_error", fail_direct_load_remotely)

    with pytest.raises(RuntimeError, match="another rank"):
        probe(name, "return x * 2")

    assert name not in sys.modules
    assert f"unsloth_cache_{name}" not in sys.modules


def test_temp_recovery_loads_by_path_on_every_rank(
    monkeypatch, compiler, probe, cache_dirs,
):
    """A successful rank must not reuse its old sys.modules entry."""
    primary, temp = cache_dirs
    monkeypatch.setattr(compiler, "torch_distributed_is_initialized", lambda: True)
    _stub_compile_folders(monkeypatch, compiler, primary, temp)
    name = "pr967_temp_reload"
    rank0_source = f"def {name}_fn(x):\n    return x * 2\n"
    rank0_digest = hashlib.sha256(rank0_source.encode()).hexdigest()
    real_distributed_function = compiler.distributed_function

    def rank0_collectives(n, function, *args, **kwargs):
        if function is compiler._compiled_cache_decision:
            return True, rank0_digest
        if function is compiler._generated_cache_source:
            return rank0_source, rank0_digest
        if getattr(function, "__name__", "") == "write_file_outcome":
            return real_distributed_function(
                n, function, args[0], rank0_source,
            )
        return real_distributed_function(n, function, *args, **kwargs)

    monkeypatch.setattr(compiler, "distributed_function", rank0_collectives)

    # Fire on the import guard, identified as the first collective after the
    # write-path verification where this rank itself saw no failure. Keyed on
    # that ordering rather than on a call count, so adding or removing a
    # collective in the write path cannot silently re-target the injection.
    write_phase_done = False
    already_fired = False
    real_cache_verification_error = compiler._cache_verification_error

    def note_end_of_write_phase(location, digest, visibility_timeout=None):
        nonlocal write_phase_done
        result = real_cache_verification_error(
            location, digest, visibility_timeout,
        )
        write_phase_done = True
        return result

    def another_rank_failed_to_import(value):
        nonlocal already_fired
        if write_phase_done and not value and not already_fired:
            already_fired = True
            return True
        return bool(value)

    monkeypatch.setattr(
        compiler, "_cache_verification_error", note_end_of_write_phase,
    )
    monkeypatch.setattr(compiler, "distributed_any", another_rank_failed_to_import)

    module = probe(name, "return x * 9")

    assert pathlib.Path(module.__file__).parent == temp
    assert module.pr967_temp_reload_fn(21) == 42
    # Importable by name again, but resolving to what the recovery loaded
    # rather than the pre-recovery module from the primary folder.
    assert sys.modules[name] is module, (
        "the recovered module is not importable by its plain name, so a "
        "later `import` fails where it succeeds without a recovery."
    )
    assert pathlib.Path(sys.modules[name].__file__).parent == temp, (
        "`import` by name resurrects the implementation the recovery replaced."
    )

    monkeypatch.setattr(
        compiler, "distributed_function", real_distributed_function,
    )
    updated = probe(name, "return x * 20")

    assert updated is not module
    assert pathlib.Path(updated.__file__).parent == temp
    assert getattr(updated, f"{name}_fn")(21) == 420
    assert sys.modules[name] is updated
    assert f"unsloth_cache_{name}" not in sys.modules


def test_verify_falls_back_to_existence_when_rank0_digest_unknown(
    tmp_path, compiler,
):
    """If rank 0 could not digest its copy, existence is all we can check."""
    location = tmp_path / "mod.py"
    location.write_bytes(b"anything")
    compiler._verify_compiled_cache_file(str(location), None)  # must not raise


def _patch_dtype_modules_twice(compiler, monkeypatch, tmp_path):
    """Run _patch_torch_dtype_modules for two model_locations, as a load does.

    unsloth/models/loader.py prepends "siglip" to model_types and
    _utils.py loops over them, so every vision load patches the global
    torch.nn classes more than once in one process.
    """
    import torch
    import transformers
    import transformers.models.llama.modeling_llama
    import transformers.models.clip.modeling_clip

    # eval(f"{model_location}.torch") resolves against the compiler globals.
    monkeypatch.setattr(compiler, "transformers", transformers, raising=False)
    monkeypatch.setattr(
        compiler, "UNSLOTH_COMPILE_LOCATION", str(tmp_path), raising=False)
    _own_temp_mode(monkeypatch, compiler, False)

    installed = []
    for model_location in (
        "transformers.models.llama.modeling_llama",
        "transformers.models.clip.modeling_clip",
    ):
        compiler._patch_torch_dtype_modules(
            model_location, [], {}, True, False, None,
        )
        installed.append(torch.nn.Conv2d.forward)
    return installed


def test_repeated_dtype_patching_does_not_stack_the_source_rewrite(
    monkeypatch, compiler, tmp_path,
):
    """A second pass must not rewrite its own rewrite.

    The source-rewrite branch reads inspect.getsource(forward). Left unmarked,
    a second pass reads back the forward the first pass generated and inserts
    another dtype prologue, so `original_dtype` binds to the weight dtype
    instead of the caller's and a bf16 activation returns fp32. Evicting
    sys.modules before importing is what makes the second rewrite take effect,
    so this only shows up once the compiled cache is reloaded properly.
    """
    import torch

    # _patch_torch_dtype_modules rewrites every name in _patch_functions, and with
    # disable=False patch_torch_functions() also replaces F.layer_norm. Restoring
    # only Conv2d would leave the rest patched for every later test in the session.
    patched = {
        module: getattr(torch.nn, module).forward
        for module in compiler._patch_functions
        if hasattr(getattr(torch.nn, module, None), "forward")
    }
    functional = {
        attr: getattr(torch.nn.functional, attr)
        for attr in ("layer_norm", "rms_norm", "group_norm", "batch_norm")
        if hasattr(torch.nn.functional, attr)
    }
    pristine = torch.nn.Conv2d.forward
    try:
        installed = _patch_dtype_modules_twice(compiler, monkeypatch, tmp_path)

        conv = torch.nn.Conv2d(3, 4, 3).to(torch.float32)
        activation = torch.randn(1, 3, 8, 8, dtype=torch.bfloat16)
        assert conv(activation).dtype == torch.bfloat16, (
            "the second dtype rewrite stacked, so the output dtype no longer "
            "follows the caller's activation."
        )
        assert installed[0] is installed[1], (
            "the second pass regenerated the forward instead of recognising "
            "its own marker."
        )
        assert getattr(installed[0], "__unsloth_dtype_wrapped__", False) is True
        assert installed[0].__unsloth_dtype_original__ is pristine, (
            "the marker must carry torch's own forward, so a later "
            "compile-mode change rebuilds from source rather than a rewrite."
        )
        source = pathlib.Path(tmp_path / "Conv2d.py").read_text(encoding="utf-8")
        assert source.count("original_dtype = input.dtype") == 1, source
    finally:
        for module, forward in patched.items():
            getattr(torch.nn, module).forward = forward
        for attr, function in functional.items():
            setattr(torch.nn.functional, attr, function)
        for module in compiler._patch_functions:
            sys.modules.pop(module, None)
            sys.modules.pop(f"unsloth_cache_{module}", None)


def test_direct_recovery_can_still_import_cache_helpers(
    monkeypatch, compiler, cache_dirs,
):
    """The recovered module must still resolve helpers next to the cache.

    exec_module() does not put the module's directory on sys.path, and
    import_module() restores sys.path before recovery runs. Generated MoE
    modules carry a bare `try: from moe_utils import ... except: pass`, so
    without the search path the direct load reports success while every MoE
    backend name is silently undefined and the first MoE forward fails instead.
    """
    primary, temp = cache_dirs
    _stub_compile_folders(monkeypatch, compiler, primary, temp)
    monkeypatch.setattr(compiler, "UNSLOTH_COMPILE_LOCATION", str(primary))
    (primary / "moe_utils.py").write_text(
        "forward_moe_backend = 'installed'\n", encoding="utf-8",
    )

    name = "pr967_recovery_helper"
    real_import = compiler.importlib.import_module
    failed = False

    def fail_once(module_name, package=None):
        nonlocal failed
        if module_name == name and not failed:
            failed = True
            raise ImportError("force direct-load recovery")
        return real_import(module_name, package)

    monkeypatch.setattr(compiler.importlib, "import_module", fail_once)

    try:
        module = compiler.create_new_function(
            name,
            f"def {name}_fn(x):\n    return x * 2\n",
            "pr967",
            {},
            prepend=(
                "try:\n"
                "    from moe_utils import forward_moe_backend\n"
                "except Exception:\n"
                "    pass\n"
            ),
            overwrite=True,
        )

        assert getattr(module, f"{name}_fn")(21) == 42
        assert getattr(module, "forward_moe_backend", None) == "installed", (
            "the recovered module could not import moe_utils from the compiled "
            "cache, so a generated MoE module would load with its backend names "
            "undefined and fail at the first forward."
        )
    finally:
        for alias in (name, f"unsloth_cache_{name}", "moe_utils"):
            sys.modules.pop(alias, None)
