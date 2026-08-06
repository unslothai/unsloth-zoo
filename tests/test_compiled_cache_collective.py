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
    monkeypatch.syspath_prepend(str(tmp_path))

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


def test_verify_falls_back_to_existence_when_rank0_digest_unknown(tmp_path):
    """If rank 0 could not digest its copy, existence is all we can check."""
    from unsloth_zoo import compiler

    location = tmp_path / "mod.py"
    location.write_bytes(b"anything")
    compiler._verify_compiled_cache_file(str(location), None)  # must not raise
