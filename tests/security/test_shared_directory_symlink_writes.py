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

"""Writes into predictable shared directories must not follow a planted link.

Three sites write to a path another local user can guess ahead of time:
the compiled-cache module file, the gpt-oss flavor marker, and the
diffusion-studio request body. A symlink planted at any of them used to be
followed, giving a co-located user a truncate-and-overwrite of any file the
victim could write.
"""

import os
import stat
import sys

import pytest

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason = "symlink pre-planting is a POSIX shared-temp problem"
)


@pytest.fixture
def victim_file(tmp_path):
    """A file the attacker names as the symlink target. It must survive intact."""
    path = tmp_path / "precious.txt"
    path.write_text("do not overwrite me")
    return path


# ---------------------------------------------------------------- compiler.py


def test_compiled_cache_write_does_not_follow_a_symlink(tmp_path, victim_file):
    from unsloth_zoo.compiler import _write_compiled_cache_file

    cache = tmp_path / "unsloth_compiled_cache"
    cache.mkdir()
    planted = cache / "UnslothSFTTrainer.py"
    os.symlink(victim_file, planted)

    _write_compiled_cache_file(str(planted), b"generated = 1\n")

    assert victim_file.read_text() == "do not overwrite me"
    # The safe fallback replaces the link itself, so the cache still ends up holding
    # the generated module. Losing the write would be a different bug.
    assert not os.path.islink(planted)
    assert planted.read_bytes() == b"generated = 1\n"


def test_compiled_cache_write_refuses_a_fifo(tmp_path):
    """O_NOFOLLOW says nothing about a FIFO, so the fstat has to."""
    from unsloth_zoo.compiler import _write_compiled_cache_file

    cache = tmp_path / "cache"
    cache.mkdir()
    target = cache / "unsloth_compiled_module_llama.py"
    os.mkfifo(target)

    _write_compiled_cache_file(str(target), b"generated = 1\n")

    assert not stat.S_ISFIFO(os.lstat(target).st_mode)
    assert target.read_bytes() == b"generated = 1\n"


def test_compiled_cache_write_still_writes_an_ordinary_file(tmp_path):
    from unsloth_zoo.compiler import _write_compiled_cache_file

    target = tmp_path / "plain.py"
    _write_compiled_cache_file(str(target), b"x = 2\n")
    assert target.read_bytes() == b"x = 2\n"


# -------------------------------------------- temporary_patches/gpt_oss.py


def test_gpt_oss_marker_skips_a_foreign_owned_cache_directory(tmp_path, monkeypatch):
    """`os.path.isdir` is satisfied by one mkdir from any local user."""
    from unsloth_zoo.temporary_patches import gpt_oss

    hostile = tmp_path / "unsloth_compiled_cache"
    hostile.mkdir(mode = 0o777)
    os.chmod(hostile, 0o777)  # world writable, so the trust check must refuse it

    monkeypatch.setattr(
        gpt_oss, "_gpt_oss_cache_locations", lambda: [str(hostile)],
    )
    gpt_oss._sync_gpt_oss_compiled_flavor("stock")

    assert not (hostile / gpt_oss._GPT_OSS_FLAVOR_MARKER).exists(), (
        "the marker was written into a directory every local user can write"
    )


def test_gpt_oss_marker_survives_a_group_writable_cache(tmp_path, monkeypatch):
    """umask 002 makes the library's own cache 0775, and that must keep working.

    Group write is a deliberate sharing choice and the compiled module next to the
    marker is written 0644 into that same directory anyway. Refusing it here bought
    nothing and silently stopped the flavor ever being recorded.
    """
    from unsloth_zoo.temporary_patches import gpt_oss

    parent = tmp_path / "shared"
    parent.mkdir(mode = 0o775)
    os.chmod(parent, 0o775)
    cache = parent / "unsloth_compiled_cache"
    cache.mkdir(mode = 0o775)
    os.chmod(cache, 0o775)

    monkeypatch.setattr(gpt_oss, "_gpt_oss_cache_locations", lambda: [str(cache)])
    gpt_oss._sync_gpt_oss_compiled_flavor("stock")

    assert (cache / gpt_oss._GPT_OSS_FLAVOR_MARKER).read_text() == "stock"


def test_gpt_oss_marker_skips_a_cache_owned_by_another_user(tmp_path, monkeypatch):
    """The attacker's `mkdir /tmp/unsloth_compiled_cache` case, by ownership."""
    from unsloth_zoo.temporary_patches import gpt_oss

    cache = tmp_path / "unsloth_compiled_cache"
    cache.mkdir(mode = 0o755)
    # A directory we cannot create as an unprivileged test: pretend to be someone else.
    monkeypatch.setattr(gpt_oss.os, "geteuid", lambda: os.geteuid() + 1)

    monkeypatch.setattr(gpt_oss, "_gpt_oss_cache_locations", lambda: [str(cache)])
    gpt_oss._sync_gpt_oss_compiled_flavor("stock")

    assert not (cache / gpt_oss._GPT_OSS_FLAVOR_MARKER).exists()


def test_gpt_oss_untrusted_cache_still_forces_regeneration(tmp_path, monkeypatch):
    """Refusing to write a marker must not also mean trusting the stale module.

    The compiler applies no trust gate when it loads
    `unsloth_compiled_module_gpt_oss.py`, so a cache directory we decline to write
    into is still one we will import from. Ignoring it in the mismatch scan would
    let a bnb4bit <-> stock switch reinstall the wrong router/experts layout.
    """
    from unsloth_zoo.temporary_patches import gpt_oss

    cache = tmp_path / "unsloth_compiled_cache"
    cache.mkdir(mode = 0o777)
    os.chmod(cache, 0o777)  # group/world writable, so the trust check refuses it
    module = cache / (gpt_oss._GPT_OSS_COMPILED_MODULE + ".py")
    module.write_text("# built for bnb4bit\n")
    (cache / gpt_oss._GPT_OSS_FLAVOR_MARKER).write_text("bnb4bit")

    monkeypatch.setattr(gpt_oss, "_gpt_oss_cache_locations", lambda: [str(cache)])
    gpt_oss._sync_gpt_oss_compiled_flavor("stock")

    assert not module.exists(), (
        "the stale compiled module survived a flavor switch in a rejected cache"
    )


def test_gpt_oss_marker_write_does_not_follow_a_symlink(tmp_path, victim_file):
    from unsloth_zoo.temporary_patches import gpt_oss

    cache = tmp_path / "cache"
    cache.mkdir(mode = 0o700)
    os.symlink(victim_file, cache / gpt_oss._GPT_OSS_FLAVOR_MARKER)

    with pytest.raises(OSError):
        gpt_oss._gpt_oss_write_marker(str(cache), "stock")
    assert victim_file.read_text() == "do not overwrite me"


def test_gpt_oss_marker_is_written_into_a_directory_we_own(tmp_path, monkeypatch):
    """The gate must not stop the ordinary case from recording the flavor."""
    from unsloth_zoo.temporary_patches import gpt_oss

    cache = tmp_path / "cache"
    cache.mkdir(mode = 0o700)
    monkeypatch.setattr(gpt_oss, "_gpt_oss_cache_locations", lambda: [str(cache)])

    gpt_oss._sync_gpt_oss_compiled_flavor("bnb4bit")

    marker = cache / gpt_oss._GPT_OSS_FLAVOR_MARKER
    assert marker.read_text() == "bnb4bit"
    assert stat.S_IMODE(os.stat(marker).st_mode) & 0o077 == 0


# ------------------------------------ diffusion_studio/visual_engine.py


def test_visual_server_request_path_is_private_and_unpredictable(monkeypatch, tmp_path):
    """The old name was `<shared dir>/dg_visual_<pid>.req`, guessable and 0644."""
    import unsloth_zoo.diffusion_studio.visual_engine as visual_engine

    monkeypatch.setattr(visual_engine.VisualServer, "_spawn", lambda self: None)
    monkeypatch.setattr(visual_engine, "_resolve_bin", lambda b: "/bin/true")
    monkeypatch.setattr(
        visual_engine, "_build_subprocess_env",
        lambda *a, **k: {"NGL": "0"},
    )
    monkeypatch.setattr(visual_engine.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(visual_engine.os.path, "isdir", lambda p: False)

    server = visual_engine.VisualServer("model.gguf")
    try:
        assert f"dg_visual_{os.getpid()}.req" not in server.req
        directory = os.path.dirname(server.req)
        assert stat.S_IMODE(os.stat(directory).st_mode) & 0o077 == 0, (
            "the request directory is readable or writable by other local users"
        )

        server.p = None
        monkeypatch.setattr(
            visual_engine.VisualServer, "restart", lambda self: None,
        )

        class _Stdin:
            def write(self, _): pass
            def flush(self): pass

        class _Process:
            stdin = _Stdin()
            def poll(self): return None

        server.p = _Process()
        server._send([{"role": "user", "content": "hello"}], 1, 0)
        assert stat.S_IMODE(os.stat(server.req).st_mode) & 0o077 == 0
    finally:
        if server._req_dir is not None:
            import shutil
            shutil.rmtree(server._req_dir, ignore_errors = True)


def test_visual_server_request_write_does_not_follow_a_symlink(
    monkeypatch, tmp_path, victim_file,
):
    import unsloth_zoo.diffusion_studio.visual_engine as visual_engine

    monkeypatch.setattr(visual_engine.VisualServer, "_spawn", lambda self: None)
    monkeypatch.setattr(visual_engine, "_resolve_bin", lambda b: "/bin/true")
    monkeypatch.setattr(
        visual_engine, "_build_subprocess_env", lambda *a, **k: {"NGL": "0"},
    )

    planted = tmp_path / "dg_visual_planted.req"
    os.symlink(victim_file, planted)
    server = visual_engine.VisualServer("model.gguf", req_path = str(planted))

    class _Stdin:
        def write(self, _): pass
        def flush(self): pass

    class _Process:
        stdin = _Stdin()
        def poll(self): return None

    server.p = _Process()
    with pytest.raises(OSError):
        server._send([{"role": "user", "content": "secret"}], 1, 0)
    assert victim_file.read_text() == "do not overwrite me"
