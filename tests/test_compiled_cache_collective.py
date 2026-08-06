# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

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


def test_explicit_temp_compile_folder_is_resolved_locally(
    tmp_path, monkeypatch, compiler,
):
    """An explicitly requested temp path must not be broadcast from rank 0."""
    local_temp = _configure_local_temp(tmp_path, monkeypatch, compiler)
    monkeypatch.setattr(
        compiler,
        "distributed_function",
        lambda *_args, **_kwargs: pytest.fail(
            "an explicit node-local temp path was broadcast"
        ),
    )

    location, use_temp = compiler.get_compile_folder(use_tempfile=True)

    assert pathlib.Path(location) == local_temp / _CACHE_LEAF
    assert pathlib.Path(location).is_dir()
    assert use_temp is True


def test_rank0_temp_fallback_is_recomputed_locally(
    tmp_path, monkeypatch, compiler,
):
    """The fallback decision is shared, but rank 0's temp path is not."""
    local_temp = _configure_local_temp(tmp_path, monkeypatch, compiler)
    rank0_temp = f"/rank0-private/tmp/{_CACHE_LEAF}"
    calls = []

    def rank0_fell_back(n, function, *args, **kwargs):
        calls.append((n, function, args, kwargs))
        return rank0_temp, True

    monkeypatch.setattr(compiler, "distributed_function", rank0_fell_back)

    location, use_temp = compiler.get_compile_folder(use_tempfile=False)

    assert len(calls) == 1
    assert pathlib.Path(location) == local_temp / _CACHE_LEAF
    assert pathlib.Path(location).is_dir()
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


@pytest.mark.parametrize(
    ("local_error", "expected_error", "message"),
    [
        (
            None,
            RuntimeError,
            "failed on another rank",
        ),
        (
            PermissionError("simulated node-local temp failure"),
            PermissionError,
            "node-local temp failure",
        ),
    ],
    ids=["remote-failure", "local-failure"],
)
def test_node_local_temp_folder_failure_is_agreed(
    tmp_path, monkeypatch, compiler, local_error, expected_error, message,
):
    """One node's mkdir failure must make every rank fail before proceeding."""
    _own_temp_mode(monkeypatch, compiler, True)
    monkeypatch.setattr(compiler, "torch_distributed_is_initialized", lambda: True)
    calls = []
    monkeypatch.setattr(
        compiler,
        "distributed_any",
        lambda value: calls.append(value) or True,
    )
    monkeypatch.setattr(
        compiler,
        "_get_compile_folder",
        lambda **_kwargs: (
            (str(tmp_path / "local-temp"), True)
            if local_error is None
            else (_ for _ in ()).throw(local_error)
        ),
    )

    with pytest.raises(expected_error, match=message):
        compiler.get_compile_folder(use_tempfile=False)

    assert calls == [True, local_error is not None]


def test_write_guard_is_not_rank_local():
    """The guard's test must come from a collective, not a local os.path.isfile."""
    func = _create_new_function_ast()
    # The `if <cond>:` whose body calls distributed_function(..., write_file, ...)
    guards = [
        node for node in ast.walk(func)
        if isinstance(node, ast.If) and "write_file" in ast.dump(node.body[0])
    ]
    assert guards, "could not find the write_file guard in create_new_function"
    guard = guards[0]
    condition = ast.dump(guard.test)
    assert "isfile" not in condition, (
        "create_new_function decides whether to call distributed_function() with a "
        "rank-local os.path.isfile(). A rank arriving after rank 0 has written the "
        "file skips the collective and hangs the group -- regression of PR #967."
    )


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
    tmp_path, monkeypatch, compiler, probe,
):
    """A rank-local read error must not escape before the first collective."""
    primary = tmp_path / "primary"
    temp = tmp_path / "temp"
    primary.mkdir()
    temp.mkdir()
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
    tmp_path, monkeypatch, compiler, probe,
):
    """Import the verified cache even when another directory has the same name."""
    primary = tmp_path / "primary"
    temp = tmp_path / "temp"
    shadow = tmp_path / "shadow"
    primary.mkdir()
    temp.mkdir()
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


def test_same_size_recompile_invalidates_stale_bytecode(
    tmp_path, monkeypatch, compiler, probe,
):
    """A same-second rewrite must not reload the prior same-sized pyc."""
    primary = tmp_path / "primary"
    temp = tmp_path / "temp"
    primary.mkdir()
    temp.mkdir()
    _stub_compile_folders(monkeypatch, compiler, primary, temp)
    name = "pr967_same_size_recompile"

    first = probe(name, "return x * 2")
    updated = probe(name, "return x * 3")

    assert getattr(first, f"{name}_fn")(21) == 42
    assert getattr(updated, f"{name}_fn")(21) == 63


def test_direct_recovery_invalidates_temp_folder_bytecode(
    tmp_path, monkeypatch, compiler, probe,
):
    """Folder-switching recovery must discard a stale temp-folder pyc."""
    primary = tmp_path / "primary"
    temp = tmp_path / "temp"
    primary.mkdir()
    temp.mkdir()
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


def test_verify_accepts_matching_file(tmp_path, compiler):
    location = tmp_path / "mod.py"
    location.write_bytes(b"same bytes")
    digest = hashlib.sha256(b"same bytes").hexdigest()
    compiler._verify_compiled_cache_file(str(location), digest)  # must not raise


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


@pytest.mark.parametrize(
    ("local_error", "expected_match"),
    [
        (None, "failed on another rank"),
        (RuntimeError("local verification failed"), "local verification failed"),
    ],
)
def test_cache_verification_failure_is_coordinated(
    tmp_path, monkeypatch, local_error, expected_match, compiler,
):
    """Every rank enters the failure collective before any rank raises."""
    calls = []
    monkeypatch.setattr(compiler, "torch_distributed_is_initialized", lambda: True)
    monkeypatch.setattr(
        compiler,
        "_verify_compiled_cache_file",
        lambda *_args: (_ for _ in ()).throw(local_error) if local_error else None,
    )

    def any_rank_failed(value):
        calls.append(value)
        return True

    monkeypatch.setattr(compiler, "distributed_any", any_rank_failed)

    with pytest.raises(RuntimeError, match=expected_match):
        compiler._verify_compiled_cache_file_collectively(
            str(tmp_path / "mod.py"), "digest",
        )
    assert calls == [local_error is not None]


def test_a_silently_failed_write_recovers_instead_of_killing_the_job(
    tmp_path, monkeypatch, compiler, probe,
):
    """A write can be reported as successful without the bytes arriving.

    write_file() swallows its own write errors, and on a multi-node run rank 0's
    write lands on another node's disk. Verification is what notices, so it has
    to route into the collective tempfile fallback the same way an outright write
    failure does, rather than raise.
    """
    cache = tmp_path / "cache"
    recovery = tmp_path / "recovery"
    cache.mkdir()
    recovery.mkdir()

    _stub_compile_folders(monkeypatch, compiler, cache, recovery)
    monkeypatch.setattr(compiler, "torch_distributed_is_initialized", lambda: True)
    monkeypatch.setattr(compiler, "distributed_any", lambda value: bool(value))
    monkeypatch.setattr(compiler, "_COMPILED_CACHE_VISIBILITY_TIMEOUT", 0.2)

    name = "pr967_silent_write_failure"
    blocked = cache / f"{name}.py"
    blocked.write_text("stale = 1\n")

    real_distributed_function = compiler.distributed_function

    def report_a_write_that_never_lands(n, function, *args, **kwargs):
        if (
            getattr(function, "__name__", "") == "write_file_outcome"
            and args[0] == str(blocked)
        ):
            return True, ""
        return real_distributed_function(n, function, *args, **kwargs)

    monkeypatch.setattr(compiler, "distributed_function", report_a_write_that_never_lands)

    module = probe(name, "return x * 2")

    assert pathlib.Path(module.__file__).parent == recovery
    assert getattr(module, f"{name}_fn")(21) == 42
    assert blocked.read_text() == "stale = 1\n", "the write was not supposed to land"


def test_the_tempfile_fallback_writes_on_every_rank(
    tmp_path, monkeypatch, compiler, probe,
):
    """The temp cache is per node, so a rank-0-only write never reaches it.

    Simulates a non-zero rank on a second node: whatever rank 0 writes lands on
    a filesystem this rank cannot see, so unless this rank writes its own copy
    the fallback produces nothing for it to import and the recovery cannot work.
    """
    cache = tmp_path / "cache"
    node_local = tmp_path / "node_local_tmp"
    cache.mkdir()
    node_local.mkdir()

    monkeypatch.setattr(compiler, "torch_distributed_is_initialized", lambda: True)
    monkeypatch.setattr(compiler, "is_main_process", lambda: False)
    monkeypatch.setattr(compiler, "distributed_any", lambda value: bool(value))
    monkeypatch.setattr(compiler, "_COMPILED_CACHE_VISIBILITY_TIMEOUT", 0.2)
    _stub_compile_folders(monkeypatch, compiler, cache, node_local)

    real_distributed_function = compiler.distributed_function

    def rank_zero_writes_on_its_own_node(n, function, *args, **kwargs):
        # A write routed through here runs on rank 0, whose filesystem this rank
        # does not share: it reports success while nothing appears locally.
        if getattr(function, "__name__", "") == "write_file_outcome":
            return True, ""
        return real_distributed_function(n, function, *args, **kwargs)

    monkeypatch.setattr(
        compiler, "distributed_function", rank_zero_writes_on_its_own_node,
    )

    name = "pr967_fallback_every_rank"
    module = probe(name, "return x * 2")

    assert (node_local / f"{name}.py").is_file(), (
        "the fallback wrote on rank 0 only, so this rank's per-node temp "
        "cache is empty and the recovery cannot produce a module"
    )
    assert pathlib.Path(module.__file__).parent == node_local
    assert getattr(module, f"{name}_fn")(21) == 42


@pytest.mark.parametrize(
    "local_bytes",
    [None, b"stale node-local source"],
    ids=["missing", "stale"],
)
def test_inconsistent_retained_temp_cache_is_rebuilt(
    tmp_path, monkeypatch, compiler, probe, local_bytes,
):
    """A warm rank 0 must not make cold/stale worker nodes skip their write."""
    node_temp = tmp_path / "node-temp"
    node_temp.mkdir()
    name = "pr967_mixed_temp_cache"
    location = node_temp / f"{name}.py"
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
        tmp_path / _CACHE_LEAF,
        node_temp,
        temp_active=True,
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

    started = time.monotonic()
    module = probe(name, "return x * 9", overwrite=False)

    assert pathlib.Path(module.__file__).parent == node_temp
    assert getattr(module, f"{name}_fn")(21) == 42
    assert "return x * 2" in location.read_text(encoding="utf-8")
    assert time.monotonic() - started < 1


def test_unreadable_rank0_retained_cache_regenerates(
    tmp_path, monkeypatch, compiler, probe,
):
    """Unreadable retained bytes fall back to rank 0's generated source."""
    primary = tmp_path / "primary"
    temp = tmp_path / "temp"
    primary.mkdir()
    temp.mkdir()
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


def test_persistent_retained_divergence_recovers_to_temp(
    tmp_path, monkeypatch, compiler, probe,
):
    """A shared-cache race should recover using rank 0's retained bytes."""
    monkeypatch.setattr(compiler, "torch_distributed_is_initialized", lambda: True)
    monkeypatch.setattr(compiler, "distributed_any", lambda value: bool(value))
    monkeypatch.setattr(compiler, "_COMPILED_CACHE_VISIBILITY_TIMEOUT", 0)
    temp = tmp_path / "temp-cache"
    temp.mkdir()
    _stub_compile_folders(
        monkeypatch, compiler, tmp_path, temp,
    )
    name = "pr967_retained_divergent"
    rank0_source = f"def {name}_fn(x):\n    return x * 2\n"
    rank0_digest = hashlib.sha256(rank0_source.encode()).hexdigest()
    (tmp_path / f"{name}.py").write_bytes(b"this rank's bytes")

    def rank0_retained_its_copy(n, function, *args, **kwargs):
        if function is compiler._compiled_cache_decision:
            return False, rank0_digest
        if function is compiler._retained_cache_source:
            return rank0_source, rank0_digest, ""
        pytest.fail(f"unexpected distributed function: {function.__name__}")

    monkeypatch.setattr(
        compiler, "distributed_function", rank0_retained_its_copy,
    )

    module = probe(name, "return x * 9", overwrite=False)

    assert pathlib.Path(module.__file__).parent == temp
    assert getattr(module, f"{name}_fn")(21) == 42


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


def test_distributed_any_without_process_group(compiler):
    """Mirrors distributed_function's tolerance of an uninitialised group."""
    assert compiler.distributed_any(True) is True
    assert compiler.distributed_any(False) is False
    assert compiler.distributed_any("non-empty") is True


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
    tmp_path, monkeypatch, compiler, probe,
):
    """A local import failure must not leave the persistent cache on sys.path."""
    primary = tmp_path / "primary"
    recovery = tmp_path / "recovery"
    primary.mkdir()
    recovery.mkdir()
    monkeypatch.setattr(compiler, "torch_distributed_is_initialized", lambda: True)
    _stub_compile_folders(monkeypatch, compiler, primary, recovery)

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
    tmp_path, monkeypatch, compiler,
):
    """Direct execution should clean sys.modules like importlib does."""
    primary = tmp_path / "primary"
    recovery = tmp_path / "recovery"
    primary.mkdir()
    recovery.mkdir()
    _stub_compile_folders(monkeypatch, compiler, primary, recovery)
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
    tmp_path, monkeypatch, compiler, probe,
):
    """A global recovery failure must leave no importable module on any rank."""
    primary = tmp_path / "primary"
    recovery = tmp_path / "recovery"
    primary.mkdir()
    recovery.mkdir()
    monkeypatch.setattr(compiler, "torch_distributed_is_initialized", lambda: True)
    _stub_compile_folders(monkeypatch, compiler, primary, recovery)
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
    tmp_path, monkeypatch, compiler, probe,
):
    """A successful rank must not reuse its old sys.modules entry."""
    primary = tmp_path / "primary"
    recovery = tmp_path / "recovery"
    primary.mkdir()
    recovery.mkdir()
    monkeypatch.setattr(compiler, "torch_distributed_is_initialized", lambda: True)
    _stub_compile_folders(monkeypatch, compiler, primary, recovery)
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

    assert pathlib.Path(module.__file__).parent == recovery
    assert module.pr967_temp_reload_fn(21) == 42
    # Importable by name again, but resolving to what the recovery loaded
    # rather than the pre-recovery module from the primary folder.
    assert sys.modules[name] is module, (
        "the recovered module is not importable by its plain name, so a "
        "later `import` fails where it succeeds without a recovery."
    )
    assert pathlib.Path(sys.modules[name].__file__).parent == recovery, (
        "`import` by name resurrects the implementation the recovery replaced."
    )

    monkeypatch.setattr(
        compiler, "distributed_function", real_distributed_function,
    )
    updated = probe(name, "return x * 20")

    assert updated is not module
    assert pathlib.Path(updated.__file__).parent == recovery
    assert getattr(updated, f"{name}_fn")(21) == 420
    assert sys.modules[name] is updated


def test_verify_falls_back_to_existence_when_rank0_digest_unknown(
    tmp_path, compiler,
):
    """If rank 0 could not digest its copy, existence is all we can check."""
    location = tmp_path / "mod.py"
    location.write_bytes(b"anything")
    compiler._verify_compiled_cache_file(str(location), None)  # must not raise
