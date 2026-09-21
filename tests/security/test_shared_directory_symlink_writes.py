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

The compiled-cache module file, the gpt-oss flavor marker and the diffusion-studio
request body are all at paths another local user can guess ahead of time.
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


@pytest.fixture
def no_o_nofollow(monkeypatch):
    """Present the interpreter Windows presents: no O_NOFOLLOW constant at all."""
    monkeypatch.delattr(os, "O_NOFOLLOW", raising = False)


def test_compiled_cache_write_refuses_a_symlink_without_o_nofollow(
    tmp_path, victim_file, no_o_nofollow,
):
    """Windows has no O_NOFOLLOW, so the flag becomes 0 and O_TRUNC would truncate
    the victim before the fstat runs."""
    from unsloth_zoo.compiler import _write_compiled_cache_file

    cache = tmp_path / "unsloth_compiled_cache"
    cache.mkdir()
    planted = cache / "UnslothSFTTrainer.py"
    os.symlink(victim_file, planted)

    _write_compiled_cache_file(str(planted), b"generated = 1\n")

    assert victim_file.read_text() == "do not overwrite me"
    assert not os.path.islink(planted)
    assert planted.read_bytes() == b"generated = 1\n"


def test_gpt_oss_marker_write_replaces_a_symlink_without_o_nofollow(
    tmp_path, victim_file, no_o_nofollow,
):
    """No atomic no-follow open, so the marker lands on the name, not through it."""
    from unsloth_zoo.temporary_patches import gpt_oss

    cache = tmp_path / "cache"
    cache.mkdir(mode = 0o700)
    marker = cache / gpt_oss._GPT_OSS_FLAVOR_MARKER
    os.symlink(victim_file, marker)

    gpt_oss._gpt_oss_write_marker(str(cache), "stock")

    assert victim_file.read_text() == "do not overwrite me"
    assert not os.path.islink(marker)
    assert marker.read_text() == "stock"


def test_visual_server_request_replaces_a_symlink_without_o_nofollow(
    monkeypatch, tmp_path, victim_file, no_o_nofollow,
):
    import unsloth_zoo.diffusion_studio.visual_engine as visual_engine

    monkeypatch.setattr(visual_engine.VisualServer, "_spawn", lambda self: None)
    monkeypatch.setattr(visual_engine, "_resolve_bin", lambda b: "/bin/true")
    monkeypatch.setattr(
        visual_engine, "_build_subprocess_env", lambda *a, **k: {"NGL": "0"},
    )

    planted = tmp_path / "planted.req"
    os.symlink(victim_file, planted)
    server = visual_engine.VisualServer("model.gguf", req_path = str(planted))

    class _Stdin:
        def write(self, _): pass
        def flush(self): pass

    class _Process:
        stdin = _Stdin()
        def poll(self): return None

    server.p = _Process()
    server._send([{"role": "user", "content": "secret"}], 1, 0)

    assert victim_file.read_text() == "do not overwrite me"
    assert not os.path.islink(planted)
    assert "secret" in planted.read_text()


def test_compiled_cache_write_still_writes_an_ordinary_file(tmp_path):
    from unsloth_zoo.compiler import _write_compiled_cache_file

    target = tmp_path / "plain.py"
    _write_compiled_cache_file(str(target), b"x = 2\n")
    assert target.read_bytes() == b"x = 2\n"



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


def test_gpt_oss_rejected_candidate_does_not_invalidate_a_good_cache(tmp_path, monkeypatch):
    """One planted candidate must not delete the cache in every other location:
    the temp path is predictable, so anyone can create it first."""
    from unsloth_zoo.temporary_patches import gpt_oss

    primary = tmp_path / "unsloth_compiled_cache"
    primary.mkdir(mode = 0o755)
    good = primary / (gpt_oss._GPT_OSS_COMPILED_MODULE + ".py")
    good.write_text("# valid, built for stock\n")
    (primary / gpt_oss._GPT_OSS_FLAVOR_MARKER).write_text("stock")

    decoy = tmp_path / "tmp_unsloth_compiled_cache"
    decoy.mkdir(mode = 0o777)
    os.chmod(decoy, 0o777)          # world writable, so the trust check refuses it
    (decoy / (gpt_oss._GPT_OSS_COMPILED_MODULE + ".py")).write_text("# planted\n")

    monkeypatch.setattr(
        gpt_oss, "_gpt_oss_cache_locations", lambda: [str(primary), str(decoy)],
    )
    gpt_oss._sync_gpt_oss_compiled_flavor("stock")

    assert good.exists(), "a planted candidate deleted the valid primary cache"
    assert not (decoy / (gpt_oss._GPT_OSS_COMPILED_MODULE + ".py")).exists(), (
        "the rejected candidate's module should still be dropped where we can"
    )


def test_gpt_oss_group_member_can_update_a_marker_it_does_not_own(tmp_path, monkeypatch):
    """B switching flavor must record it, in a marker file A created.

    Owner-only there means B deletes the stale module but cannot write the new
    flavor, and the next load pairs a fresh module with the old marker.
    """
    from unsloth_zoo.temporary_patches import gpt_oss

    cache = tmp_path / "unsloth_compiled_cache"
    cache.mkdir(mode = 0o775)
    os.chmod(cache, 0o775)
    (cache / (gpt_oss._GPT_OSS_COMPILED_MODULE + ".py")).write_text("# stock build\n")
    marker = cache / gpt_oss._GPT_OSS_FLAVOR_MARKER
    marker.write_text("stock")
    os.chmod(marker, 0o400)     # what a marker owned by user A presents to user B

    monkeypatch.setattr(gpt_oss, "_gpt_oss_cache_locations", lambda: [str(cache)])
    gpt_oss._sync_gpt_oss_compiled_flavor("bnb4bit")

    assert marker.read_text() == "bnb4bit", "the marker still claims the old flavor"
    assert stat.S_IMODE(os.stat(marker).st_mode) & 0o020, (
        "a shared marker the next group member cannot rewrite is the same bug again"
    )
    assert not stat.S_IMODE(os.stat(marker).st_mode) & 0o002


def test_visual_server_cleans_its_directory_when_construction_fails(monkeypatch, tmp_path):
    """No caller can close() an object __init__ never returned."""
    import unsloth_zoo.diffusion_studio.visual_engine as visual_engine

    monkeypatch.setattr(visual_engine, "_resolve_bin", lambda b: "/bin/true")
    monkeypatch.setattr(
        visual_engine, "_build_subprocess_env", lambda *a, **k: {"NGL": "0"},
    )
    monkeypatch.setattr(visual_engine.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(visual_engine.os.path, "isdir", lambda p: False)

    def _explode(self):
        raise RuntimeError("visual server failed to start: ''")

    monkeypatch.setattr(visual_engine.VisualServer, "_spawn", _explode)

    with pytest.raises(RuntimeError):
        visual_engine.VisualServer("model.gguf")

    assert not [p for p in os.listdir(tmp_path) if p.startswith("dg_visual_")], (
        "a failed start left its private directory behind"
    )


def test_gpt_oss_marker_survives_a_group_writable_cache(tmp_path, monkeypatch):
    """umask 002 makes the library's own cache 0775, and that must keep working:
    the module beside the marker is 0644 in that same directory anyway."""
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
    """The attacker's `mkdir /tmp/unsloth_compiled_cache` case, by ownership. 0755,
    so not shared with our group either."""
    from unsloth_zoo.temporary_patches import gpt_oss

    cache = tmp_path / "unsloth_compiled_cache"
    cache.mkdir(mode = 0o755)
    # Capture the real euid FIRST: a lambda calling os.geteuid() after the patch
    # calls itself, and the RecursionError is swallowed into a False that looks like
    # the answer under test.
    other_uid = os.geteuid() + 1
    monkeypatch.setattr(gpt_oss.os, "geteuid", lambda: other_uid)

    monkeypatch.setattr(gpt_oss, "_gpt_oss_cache_locations", lambda: [str(cache)])
    gpt_oss._sync_gpt_oss_compiled_flavor("stock")

    assert not (cache / gpt_oss._GPT_OSS_FLAVOR_MARKER).exists()


def test_gpt_oss_keeps_a_group_shared_cache_built_by_another_user(tmp_path, monkeypatch):
    """User B must not destroy user A's module in a cache shared with their group:
    a 0775 cache is owned by whoever built it first, so everyone else fails an
    ownership test on the very cache they were given write access to."""
    from unsloth_zoo.temporary_patches import gpt_oss

    cache = tmp_path / "unsloth_compiled_cache"
    cache.mkdir(mode = 0o775)
    os.chmod(cache, 0o775)
    module = cache / (gpt_oss._GPT_OSS_COMPILED_MODULE + ".py")
    module.write_text("# built by user A\n")
    (cache / gpt_oss._GPT_OSS_FLAVOR_MARKER).write_text("stock")

    # We are user B: same group, write access, not the owner.
    other_uid = os.geteuid() + 1
    monkeypatch.setattr(gpt_oss.os, "geteuid", lambda: other_uid)
    monkeypatch.setattr(gpt_oss, "_gpt_oss_cache_locations", lambda: [str(cache)])
    assert os.lstat(cache).st_gid in set(os.getgroups()) | {os.getgid()}, (
        "the fixture cannot model a group-shared cache on this runner"
    )

    gpt_oss._sync_gpt_oss_compiled_flavor("stock")

    assert module.exists(), "a group member deleted the shared compiled module"
    assert (cache / gpt_oss._GPT_OSS_FLAVOR_MARKER).read_text() == "stock"


def test_gpt_oss_untrusted_cache_still_forces_regeneration(tmp_path, monkeypatch):
    """Refusing to write a marker must not also mean trusting the stale module: the
    compiler applies no such gate when it imports from that same directory."""
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


def test_gpt_oss_marker_write_refuses_a_fifo(tmp_path):
    """A planted FIFO must not block the load, nor silently eat the marker: without
    O_NONBLOCK the open waits for a reader forever, and with one attached it succeeds
    and the flavor is never recorded."""
    from unsloth_zoo.temporary_patches import gpt_oss

    cache = tmp_path / "cache"
    cache.mkdir(mode = 0o700)       # owner-only, so the in-place open is the path taken
    marker = cache / gpt_oss._GPT_OSS_FLAVOR_MARKER
    os.mkfifo(marker)

    with pytest.raises(OSError):
        gpt_oss._gpt_oss_write_marker(str(cache), "stock")
    assert stat.S_ISFIFO(os.lstat(marker).st_mode), "the FIFO should be refused, not replaced"


def test_gpt_oss_marker_replaces_a_fifo_in_a_shared_cache(tmp_path):
    """A shared cache lands the marker by replacement, which removes the plant.

    Refusing is right for the in-place path, where writing would mean writing INTO
    the pipe. Replacement never touches it, so succeeding is the better outcome.
    """
    from unsloth_zoo.temporary_patches import gpt_oss

    cache = tmp_path / "cache"
    cache.mkdir(mode = 0o775)
    os.chmod(cache, 0o775)
    marker = cache / gpt_oss._GPT_OSS_FLAVOR_MARKER
    os.mkfifo(marker)

    gpt_oss._gpt_oss_write_marker(str(cache), "stock")

    assert not stat.S_ISFIFO(os.lstat(marker).st_mode)
    assert marker.read_text() == "stock"


def test_visual_server_request_write_refuses_to_block_on_a_fifo(monkeypatch, tmp_path):
    """An explicit req_path pointing at a FIFO must fail the request, not hang it."""
    import unsloth_zoo.diffusion_studio.visual_engine as visual_engine

    monkeypatch.setattr(visual_engine.VisualServer, "_spawn", lambda self: None)
    monkeypatch.setattr(visual_engine, "_resolve_bin", lambda b: "/bin/true")
    monkeypatch.setattr(
        visual_engine, "_build_subprocess_env", lambda *a, **k: {"NGL": "0"},
    )

    planted = tmp_path / "planted.req"
    os.mkfifo(planted)
    server = visual_engine.VisualServer("model.gguf", req_path = str(planted))

    class _Stdin:
        def write(self, _): pass
        def flush(self): pass

    class _Process:
        stdin = _Stdin()
        def poll(self): return None

    server.p = _Process()
    with pytest.raises(OSError):
        server._send([{"role": "user", "content": "hello"}], 1, 0)


def test_visual_server_request_refuses_a_fifo_someone_is_reading(monkeypatch, tmp_path):
    """O_NONBLOCK only refuses a FIFO with NO reader: held open for reading the open
    succeeds, and only the fstat refuses this one."""
    import threading
    import unsloth_zoo.diffusion_studio.visual_engine as visual_engine

    monkeypatch.setattr(visual_engine.VisualServer, "_spawn", lambda self: None)
    monkeypatch.setattr(visual_engine, "_resolve_bin", lambda b: "/bin/true")
    monkeypatch.setattr(
        visual_engine, "_build_subprocess_env", lambda *a, **k: {"NGL": "0"},
    )

    planted = tmp_path / "planted.req"
    os.mkfifo(planted)
    holder = {}

    def eavesdrop():
        holder["fd"] = os.open(str(planted), os.O_RDONLY)

    reader = threading.Thread(target = eavesdrop, daemon = True)
    reader.start()

    server = visual_engine.VisualServer("model.gguf", req_path = str(planted))

    class _Stdin:
        def write(self, _): pass
        def flush(self): pass

    class _Process:
        stdin = _Stdin()
        def poll(self): return None

    server.p = _Process()
    try:
        with pytest.raises(OSError):
            server._send([{"role": "user", "content": "secret"}], 1, 0)
    finally:
        if "fd" in holder:
            os.close(holder["fd"])


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
