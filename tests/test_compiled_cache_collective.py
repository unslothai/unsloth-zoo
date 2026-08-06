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
import hashlib
import importlib.util
import pathlib
import sys
import threading
import time

import pytest


def _compiler_source() -> str:
    spec = importlib.util.find_spec("unsloth_zoo.compiler")
    assert spec is not None and spec.origin, "cannot locate unsloth_zoo.compiler"
    return pathlib.Path(spec.origin).read_text(encoding="utf-8")


def test_write_guard_is_not_rank_local():
    """The guard's test must come from a collective, not a local os.path.isfile."""
    tree = ast.parse(_compiler_source())
    func = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "create_new_function"
    )
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


def test_decision_is_broadcast_not_computed_locally(tmp_path, monkeypatch):
    """Every rank must take the branch rank 0 chose."""
    from unsloth_zoo import compiler

    calls = []
    real = compiler.distributed_function

    def spy(n, function, *args, **kwargs):
        calls.append(getattr(function, "__name__", str(function)))
        return real(n, function, *args, **kwargs)

    monkeypatch.setattr(compiler, "distributed_function", spy)
    monkeypatch.setattr(compiler, "is_distributed", lambda: False)
    monkeypatch.setattr(compiler, "UNSLOTH_COMPILE_LOCATION", str(tmp_path))
    monkeypatch.setattr(compiler, "UNSLOTH_COMPILE_USE_TEMP", False)

    compiler.create_new_function(
        "pr967_probe", "def pr967_probe_fn(x):\n    return x\n", "pr967", {},
        overwrite=True,
    )
    assert (tmp_path / "pr967_probe.py").is_file(), "probe did not compile into tmp_path"
    assert "_compiled_cache_decision" in calls, (
        "the write decision was not routed through distributed_function(); it is "
        "rank-local again -- regression of PR #967."
    )


def test_decision_digests_disk_not_generated_source(tmp_path, monkeypatch):
    """UNSLOTH_COMPILE_OVERWRITE=0 keeps a cache file that != write_new_source.

    Digesting write_new_source instead of the bytes on disk would make every rank
    reject a cache file rank 0 deliberately kept.
    """
    from unsloth_zoo import compiler

    monkeypatch.setattr(compiler, "is_distributed", lambda: True)
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


def test_decision_skips_hashing_when_not_distributed(tmp_path, monkeypatch):
    """Single process: same answer as the old expression, and no extra file read."""
    from unsloth_zoo import compiler

    monkeypatch.setattr(compiler, "is_distributed", lambda: False)
    location = tmp_path / "mod.py"

    assert compiler._compiled_cache_decision(str(location), "src", False) == (True, None)
    location.write_bytes(b"cached")
    assert compiler._compiled_cache_decision(str(location), "src", False) == (False, None)
    assert compiler._compiled_cache_decision(str(location), "src", True) == (True, None)


def test_verify_accepts_matching_file(tmp_path):
    from unsloth_zoo import compiler

    location = tmp_path / "mod.py"
    location.write_bytes(b"same bytes")
    digest = hashlib.sha256(b"same bytes").hexdigest()
    compiler._verify_compiled_cache_file(str(location), digest)  # must not raise


def test_verify_rejects_divergent_file(tmp_path, monkeypatch):
    """A stale-but-present file must not be silently imported."""
    from unsloth_zoo import compiler

    monkeypatch.setattr(compiler, "_COMPILED_CACHE_VISIBILITY_TIMEOUT", 0.2)
    location = tmp_path / "mod.py"
    location.write_bytes(b"stale bytes")
    rank0_digest = hashlib.sha256(b"current bytes").hexdigest()

    with pytest.raises(RuntimeError, match="differs between rank 0"):
        compiler._verify_compiled_cache_file(str(location), rank0_digest)


def test_verify_reports_missing_file(tmp_path, monkeypatch):
    from unsloth_zoo import compiler

    monkeypatch.setattr(compiler, "_COMPILED_CACHE_VISIBILITY_TIMEOUT", 0.2)
    location = tmp_path / "never_written.py"

    with pytest.raises(FileNotFoundError, match="not readable on rank"):
        compiler._verify_compiled_cache_file(
            str(location), hashlib.sha256(b"x").hexdigest(),
        )


def test_verify_waits_out_filesystem_lag(tmp_path, monkeypatch):
    """A network filesystem may publish the file just after the barrier."""
    from unsloth_zoo import compiler

    monkeypatch.setattr(compiler, "_COMPILED_CACHE_VISIBILITY_TIMEOUT", 5.0)
    location = tmp_path / "mod.py"
    digest = hashlib.sha256(b"late bytes").hexdigest()

    threading.Timer(0.5, lambda: location.write_bytes(b"late bytes")).start()
    started = time.monotonic()
    compiler._verify_compiled_cache_file(str(location), digest)  # must not raise
    assert time.monotonic() - started >= 0.4, "did not actually wait for the file"


@pytest.mark.parametrize("is_rank_zero", [True, False], ids=["rank0", "nonzero_rank"])
def test_write_path_verifies_the_file_on_every_rank(
    tmp_path, monkeypatch, is_rank_zero,
):
    """A rank-local stale file must be checked even when rank 0 writes.

    Parametrised over the rank, because rank 0 is the only rank that writes: a
    verification guarded by is_main_process() would leave every other rank --
    the ones that can actually hold a divergent file -- unchecked, and a
    single-process test cannot tell the difference.
    """
    from unsloth_zoo import compiler

    verified = []
    monkeypatch.setattr(compiler, "is_distributed", lambda: True)
    monkeypatch.setattr(compiler, "is_main_process", lambda: is_rank_zero)
    monkeypatch.setattr(compiler, "UNSLOTH_COMPILE_USE_TEMP", False)
    monkeypatch.setattr(
        compiler,
        "get_compile_folder",
        lambda use_tempfile=False: (str(tmp_path), False),
    )
    monkeypatch.setattr(
        compiler,
        "_cache_verification_error",
        lambda location, digest: verified.append((location, digest)),
    )

    name = "pr967_verify_after_write"
    try:
        compiler.create_new_function(
            name,
            f"def {name}_fn(x):\n    return x\n",
            "pr967",
            {},
            overwrite=True,
        )
    finally:
        sys.modules.pop(name, None)

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
    tmp_path, monkeypatch, local_error, expected_match,
):
    """Every rank enters the failure collective before any rank raises."""
    from unsloth_zoo import compiler

    calls = []
    monkeypatch.setattr(compiler, "is_distributed", lambda: True)
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
    tmp_path, monkeypatch,
):
    """A write can be reported as successful without the bytes arriving.

    write_file() swallows its own write errors, and on a multi-node run rank 0's
    write lands on another node's disk. Verification is what notices, so it has
    to route into the collective tempfile fallback the same way an outright write
    failure does, rather than raise.
    """
    from unsloth_zoo import compiler

    cache = tmp_path / "cache"
    recovery = tmp_path / "recovery"
    cache.mkdir()
    recovery.mkdir()

    def fake_get_compile_folder(use_tempfile=False):
        return (str(recovery), True) if use_tempfile else (str(cache), False)

    monkeypatch.setattr(compiler, "get_compile_folder", fake_get_compile_folder)
    monkeypatch.setattr(compiler, "is_distributed", lambda: True)
    monkeypatch.setattr(compiler, "distributed_any", lambda value: bool(value))
    monkeypatch.setattr(compiler, "_COMPILED_CACHE_VISIBILITY_TIMEOUT", 0.2)
    # create_new_function() assigns this global, which monkeypatch cannot undo
    # unless it owns the attribute first.
    monkeypatch.setattr(compiler, "UNSLOTH_COMPILE_USE_TEMP", False)

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

    try:
        module = compiler.create_new_function(
            name,
            f"def {name}_fn(x):\n    return x * 2\n",
            "pr967",
            {},
            overwrite=True,
        )
        assert pathlib.Path(module.__file__).parent == recovery
        assert getattr(module, f"{name}_fn")(21) == 42
        assert blocked.read_text() == "stale = 1\n", "the write was not supposed to land"
    finally:
        sys.modules.pop(name, None)
        sys.modules.pop(f"unsloth_cache_{name}", None)


def test_the_tempfile_fallback_writes_on_every_rank(tmp_path, monkeypatch):
    """The temp cache is per node, so a rank-0-only write never reaches it.

    Simulates a non-zero rank on a second node: whatever rank 0 writes lands on
    a filesystem this rank cannot see, so unless this rank writes its own copy
    the fallback produces nothing for it to import and the recovery cannot work.
    """
    from unsloth_zoo import compiler

    cache = tmp_path / "cache"
    node_local = tmp_path / "node_local_tmp"
    cache.mkdir()
    node_local.mkdir()

    monkeypatch.setattr(compiler, "is_distributed", lambda: True)
    monkeypatch.setattr(compiler, "is_main_process", lambda: False)
    monkeypatch.setattr(compiler, "distributed_any", lambda value: bool(value))
    monkeypatch.setattr(compiler, "_COMPILED_CACHE_VISIBILITY_TIMEOUT", 0.2)
    monkeypatch.setattr(compiler, "UNSLOTH_COMPILE_USE_TEMP", False)
    monkeypatch.setattr(
        compiler,
        "get_compile_folder",
        lambda use_tempfile=False: (
            (str(node_local), True) if use_tempfile else (str(cache), False)
        ),
    )

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
    try:
        module = compiler.create_new_function(
            name,
            f"def {name}_fn(x):\n    return x * 2\n",
            "pr967",
            {},
            overwrite=True,
        )
        assert (node_local / f"{name}.py").is_file(), (
            "the fallback wrote on rank 0 only, so this rank's per-node temp "
            "cache is empty and the recovery cannot produce a module"
        )
        assert pathlib.Path(module.__file__).parent == node_local
        assert getattr(module, f"{name}_fn")(21) == 42
    finally:
        sys.modules.pop(name, None)
        sys.modules.pop(f"unsloth_cache_{name}", None)


def test_a_retained_but_divergent_file_still_fails_loudly(tmp_path, monkeypatch):
    """The counterpart: with no write to retry, a stale copy must raise.

    Falling back to a tempfile here would mask a genuinely misconfigured cache.
    """
    from unsloth_zoo import compiler

    monkeypatch.setattr(compiler, "is_distributed", lambda: True)
    monkeypatch.setattr(compiler, "distributed_any", lambda value: bool(value))
    monkeypatch.setattr(compiler, "_COMPILED_CACHE_VISIBILITY_TIMEOUT", 0.2)
    monkeypatch.setattr(compiler, "UNSLOTH_COMPILE_USE_TEMP", False)
    monkeypatch.setattr(
        compiler,
        "get_compile_folder",
        lambda use_tempfile=False: (str(tmp_path), False),
    )
    # Rank 0 keeps its file, and this rank's copy at the same path differs.
    monkeypatch.setattr(
        compiler,
        "_compiled_cache_decision",
        lambda *_args: (False, hashlib.sha256(b"rank 0 bytes").hexdigest()),
    )

    name = "pr967_retained_divergent"
    (tmp_path / f"{name}.py").write_bytes(b"this rank's bytes")

    with pytest.raises(RuntimeError, match="differs between rank 0"):
        compiler.create_new_function(
            name, f"def {name}_fn(x):\n    return x\n", "pr967", {}, overwrite=False,
        )


def _create_new_function_ast() -> ast.FunctionDef:
    tree = ast.parse(_compiler_source())
    return next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "create_new_function"
    )


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


def test_distributed_any_without_process_group():
    """Mirrors distributed_function's tolerance of an uninitialised group."""
    from unsloth_zoo.utils import distributed_any
    assert distributed_any(True) is True
    assert distributed_any(False) is False
    assert distributed_any("non-empty") is True


def test_write_failure_falls_back_to_tempfile(tmp_path, monkeypatch):
    """A read-only cache directory must reach the tempfile fallback, not raise."""
    from unsloth_zoo import compiler

    cache = tmp_path / "readonly_cache"
    cache.mkdir()
    monkeypatch.setattr(compiler, "UNSLOTH_COMPILE_LOCATION", str(cache))
    monkeypatch.setattr(compiler, "UNSLOTH_COMPILE_USE_TEMP", False)

    real_get_lock = compiler.get_lock

    def fake_get_lock(target, *args, **kwargs):
        if str(target).startswith(str(cache)):
            raise OSError("simulated read-only cache directory")
        return real_get_lock(target, *args, **kwargs)

    monkeypatch.setattr(compiler, "get_lock", fake_get_lock)

    module = compiler.create_new_function(
        "pr967_readonly", "def pr967_readonly_fn(x):\n    return x * 3\n", "pr967", {},
        overwrite=True,
    )
    assert module.pr967_readonly_fn(14) == 42
    assert not (cache / "pr967_readonly.py").exists(), (
        "the write was expected to fail against the read-only directory"
    )


def test_current_rank_without_process_group(monkeypatch):
    """RANK is unset outside a launcher, and get_rank() would raise."""
    from unsloth_zoo.utils import current_rank

    monkeypatch.delenv("RANK", raising=False)
    assert str(current_rank()) == "0"
    monkeypatch.setenv("RANK", "3")
    assert str(current_rank()) == "3"


def test_import_recovery_message_does_not_blame_this_rank():
    """A rank whose own import succeeded must not log 'failed ... : None'."""
    src = _compiler_source()
    assert "Standard import failed for {name}: {import_error}" not in src, (
        "import_error is None on a rank that imported fine but was pulled onto the "
        "recovery by another rank, so this logs a bare None."
    )
    assert 'reason = import_error or "an import failure on another rank"' in src


def test_temp_recovery_loads_by_path_on_every_rank(tmp_path, monkeypatch):
    """A successful rank must not reuse its old sys.modules entry."""
    from unsloth_zoo import compiler

    primary = tmp_path / "primary"
    recovery = tmp_path / "recovery"
    primary.mkdir()
    recovery.mkdir()
    monkeypatch.setattr(compiler, "is_distributed", lambda: True)
    monkeypatch.setattr(compiler, "UNSLOTH_COMPILE_USE_TEMP", False)

    def fake_get_compile_folder(use_tempfile=False):
        return (str(recovery), True) if use_tempfile else (str(primary), False)

    monkeypatch.setattr(compiler, "get_compile_folder", fake_get_compile_folder)

    # Fire on the import guard, identified as the first collective after the
    # write-path verification where this rank itself saw no failure. Keyed on
    # that ordering rather than on a call count, so adding or removing a
    # collective in the write path cannot silently re-target the injection.
    write_phase_done = False
    already_fired = False
    real_cache_verification_error = compiler._cache_verification_error

    def note_end_of_write_phase(location, digest):
        nonlocal write_phase_done
        result = real_cache_verification_error(location, digest)
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

    name = "pr967_temp_reload"
    try:
        module = compiler.create_new_function(
            name,
            f"def {name}_fn(x):\n    return x * 2\n",
            "pr967",
            {},
            overwrite=True,
        )
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
    finally:
        sys.modules.pop(name, None)
        sys.modules.pop(f"unsloth_cache_{name}", None)


def test_verify_falls_back_to_existence_when_rank0_digest_unknown(tmp_path):
    """If rank 0 could not digest its copy, existence is all we can check."""
    from unsloth_zoo import compiler

    location = tmp_path / "mod.py"
    location.write_bytes(b"anything")
    compiler._verify_compiled_cache_file(str(location), None)  # must not raise
