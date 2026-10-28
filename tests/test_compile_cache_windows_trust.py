# SPDX-License-Identifier: AGPL-3.0-only

"""Windows owner/DACL trust check for the Mega-cache (compile_cache.py).

The decision logic, the raw SID/ACL parsers and the directory walk run on any
OS (synthetic ACLs, stubbed handle queries). The tests marked windows_only use
the real advapi32 path and icacls, and run on a Windows host."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import struct
import subprocess
import sys
import types
from pathlib import Path

import pytest


_COMPILE_CACHE_PATH = (
    Path(__file__).resolve().parents[1] / "unsloth_zoo" / "compile_cache.py"
)

windows_only = pytest.mark.skipif(os.name != "nt", reason = "needs real Windows ACLs")

USER = "S-1-5-21-1111-2222-3333-1001"
OTHER = "S-1-5-21-1111-2222-3333-1002"
SYSTEM = "S-1-5-18"
ADMINS = "S-1-5-32-544"
USERS = "S-1-5-32-545"
EVERYONE = "S-1-1-0"
AUTH_USERS = "S-1-5-11"
CREATOR_OWNER = "S-1-3-0"

FA = 0x001F01FF          # full control
MODIFY = 0x001301BF      # icacls M
RX = 0x001200A9          # icacls RX
WRITE = 0x00120116       # icacls W
ADD_SUBDIR = 0x00000004  # icacls AD
DELETE_CHILD = 0x00000040
GENERIC_WRITE = 0x40000000
GENERIC_ALL = 0x10000000

ALLOW, DENY, ALLOW_OBJECT, ALLOW_CALLBACK = 0x00, 0x01, 0x05, 0x09
OI_CI_IO = 0x01 | 0x02 | 0x08
INHERITED = 0x10


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "zoo_compile_cache_windows_under_test", _COMPILE_CACHE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sid_bytes(sid):
    parts = sid.split("-")
    revision, authority, subs = int(parts[1]), int(parts[2]), [int(p) for p in parts[3:]]
    return (
        bytes([revision, len(subs)]) + authority.to_bytes(6, "big")
        + struct.pack("<%dI" % len(subs), *subs)
    )


def _ace(ace_type, flags, mask, sid):
    body = struct.pack("<I", mask) + _sid_bytes(sid)
    return struct.pack("<BBH", ace_type, flags, 4 + len(body)) + body


def _acl(*aces):
    data = b"".join(aces)
    return struct.pack("<BBHHH", 2, 0, 8 + len(data), len(aces), 0) + data


def _trusted(module, owner, aces, *, leaf, user_sids = (USER, USER)):
    return module._windows_acl_is_trusted(owner, aces, user_sids, leaf = leaf)


# --- parsers ---------------------------------------------------------------

def test_parse_sid_round_trips():
    module = _load_module()
    assert _sid_bytes(SYSTEM) == bytes.fromhex("010100000000000512000000")
    for sid in (SYSTEM, ADMINS, EVERYONE, USER, module._WIN_SID_TRUSTED_INSTALLER):
        assert module._windows_parse_sid(_sid_bytes(sid)) == sid


@pytest.mark.parametrize("raw", (b"", b"\x01\x01\x00", b"\x02\x01" + b"\x00" * 10, b"\x01\x02" + b"\x00" * 10))
def test_parse_sid_rejects_malformed(raw):
    module = _load_module()
    with pytest.raises(ValueError):
        module._windows_parse_sid(raw)


def test_parse_acl_reads_simple_and_flags_unknown_types():
    module = _load_module()
    object_ace = struct.pack("<BBH", ALLOW_OBJECT, 0, 8) + struct.pack("<I", FA)
    raw = _acl(_ace(ALLOW, 0, FA, USER), _ace(DENY, INHERITED, WRITE, EVERYONE), object_ace)
    assert module._windows_parse_acl(raw) == [
        (ALLOW, 0, FA, USER),
        (DENY, INHERITED, WRITE, EVERYONE),
        (ALLOW_OBJECT, 0, None, None),
    ]


def test_parse_acl_rejects_malformed():
    module = _load_module()
    good = _acl(_ace(ALLOW, 0, FA, USER))
    with pytest.raises(ValueError):
        module._windows_parse_acl(good[:6])
    with pytest.raises(ValueError):  # header claims more ACEs than present
        module._windows_parse_acl(struct.pack("<BBHHH", 2, 0, 8, 1, 0))
    with pytest.raises(ValueError):  # ACE size past the ACL
        module._windows_parse_acl(good[:-4])
    with pytest.raises(ValueError):  # bad revision
        module._windows_parse_acl(b"\x07" + good[1:])


# --- decision ------------------------------------------------------------

STOCK_PROFILE = [(ALLOW, 0, FA, SYSTEM), (ALLOW, 0, FA, ADMINS), (ALLOW, 0, FA, USER)]


def test_owner_only_dacl_is_trusted_as_leaf_and_ancestor():
    module = _load_module()
    assert _trusted(module, USER, STOCK_PROFILE, leaf = True)
    assert _trusted(module, USER, STOCK_PROFILE, leaf = False)
    assert _trusted(module, USER, [], leaf = True)  # empty DACL: owner only


def test_foreign_owner_is_refused():
    module = _load_module()
    assert not _trusted(module, OTHER, STOCK_PROFILE, leaf = True)
    assert not _trusted(module, OTHER, STOCK_PROFILE, leaf = False)


def test_system_owner_only_for_ancestors():
    module = _load_module()
    for owner in (SYSTEM, ADMINS, module._WIN_SID_TRUSTED_INSTALLER):
        assert not _trusted(module, owner, STOCK_PROFILE, leaf = True)
        assert _trusted(module, owner, STOCK_PROFILE, leaf = False)


def test_elevated_token_owner_is_accepted_for_the_leaf():
    module = _load_module()
    assert _trusted(module, ADMINS, STOCK_PROFILE, leaf = True, user_sids = (USER, ADMINS))


def test_null_dacl_and_unknown_identity_are_refused():
    module = _load_module()
    assert not _trusted(module, USER, None, leaf = True)
    assert not _trusted(module, USER, None, leaf = False)
    assert not _trusted(module, None, STOCK_PROFILE, leaf = True)
    assert not _trusted(module, USER, STOCK_PROFILE, leaf = True, user_sids = ())


@pytest.mark.parametrize("sid", (EVERYONE, AUTH_USERS, USERS, OTHER))
@pytest.mark.parametrize("mask", (WRITE, MODIFY, FA, ADD_SUBDIR, DELETE_CHILD, GENERIC_WRITE, GENERIC_ALL))
def test_any_foreign_write_refuses_the_leaf(sid, mask):
    module = _load_module()
    assert not _trusted(module, USER, STOCK_PROFILE + [(ALLOW, INHERITED, mask, sid)], leaf = True)


@pytest.mark.parametrize("sid", (EVERYONE, AUTH_USERS, USERS))
def test_foreign_read_is_fine(sid):
    module = _load_module()
    assert _trusted(module, USER, STOCK_PROFILE + [(ALLOW, 0, RX, sid)], leaf = True)


def test_ancestor_allows_create_but_not_replace():
    module = _load_module()
    # C:\ grants Authenticated Users "create folders" (AD) on itself: harmless.
    for mask in (ADD_SUBDIR, WRITE, GENERIC_WRITE):
        assert _trusted(module, SYSTEM, STOCK_PROFILE + [(ALLOW, 0, mask, AUTH_USERS)], leaf = False)
    # Anything that can delete, rename or re-ACL an existing child is not.
    for mask in (MODIFY, FA, DELETE_CHILD, 0x00010000, 0x00040000, 0x00080000, GENERIC_ALL):
        assert not _trusted(module, SYSTEM, STOCK_PROFILE + [(ALLOW, 0, mask, AUTH_USERS)], leaf = False)


def test_inherit_only_and_placeholder_aces_are_skipped():
    module = _load_module()
    # C:\Users / C:\: CREATOR OWNER full and Authenticated Users Modify, both
    # inherit-only, describe children, not this object.
    aces = STOCK_PROFILE + [
        (ALLOW, OI_CI_IO, FA, CREATOR_OWNER),
        (ALLOW, OI_CI_IO, MODIFY, AUTH_USERS),
        (ALLOW_OBJECT, OI_CI_IO, None, None),
    ]
    assert _trusted(module, USER, aces, leaf = True)
    # Effective CREATOR OWNER / OWNER RIGHTS ACEs never grant a stranger anything.
    assert _trusted(module, USER, STOCK_PROFILE + [(ALLOW, 0, FA, CREATOR_OWNER)], leaf = True)
    assert _trusted(module, USER, STOCK_PROFILE + [(ALLOW, 0, FA, "S-1-3-4")], leaf = True)


def test_deny_aces_are_ignored_and_unknown_allows_refused():
    module = _load_module()
    assert _trusted(module, USER, STOCK_PROFILE + [(DENY, 0, FA, EVERYONE)], leaf = True)
    assert not _trusted(module, USER, STOCK_PROFILE + [(ALLOW_OBJECT, 0, None, None)], leaf = True)
    assert not _trusted(module, USER, STOCK_PROFILE + [(0x11, 0, None, None)], leaf = False)
    # A conditional allow is judged as if it always applied.
    assert not _trusted(module, USER, STOCK_PROFILE + [(ALLOW_CALLBACK, 0, WRITE, EVERYONE)], leaf = True)
    assert _trusted(module, USER, STOCK_PROFILE + [(ALLOW_CALLBACK, 0, RX, EVERYONE)], leaf = True)


# --- walk (stubbed handle queries, any OS) ---------------------------------

def _stub_windows(monkeypatch, module, security_by_path):
    monkeypatch.setattr(module, "_windows_token_sids", lambda: (USER, USER))
    default = (0x10, SYSTEM, STOCK_PROFILE)

    def fake_security(path):
        return security_by_path.get(os.path.abspath(os.fspath(path)), default)

    monkeypatch.setattr(module, "_windows_path_security", fake_security)


def _is_trusted_as_nt(monkeypatch, module, path, **kwargs):
    with monkeypatch.context() as patch:
        patch.setattr(module.os, "name", "nt")
        return module._is_trusted_directory(str(path), **kwargs)


def test_walk_applies_leaf_and_ancestor_rules(monkeypatch, tmp_path):
    module = _load_module()
    root = tmp_path / "mega_cache"
    key = root / "deadbeef"
    key.mkdir(parents = True)
    owned = (0x10, USER, STOCK_PROFILE)
    table = {str(root): owned, str(key): owned}
    _stub_windows(monkeypatch, module, table)
    assert _is_trusted_as_nt(monkeypatch, module, key) is True

    # an ancestor with inherited Authenticated Users Modify (a dir under C:\)
    table[str(tmp_path)] = (0x10, USER, STOCK_PROFILE + [(ALLOW, INHERITED, MODIFY, AUTH_USERS)])
    assert _is_trusted_as_nt(monkeypatch, module, key) is False
    # the same ancestor granting only "create folders" is fine
    table[str(tmp_path)] = (0x10, USER, STOCK_PROFILE + [(ALLOW, 0, ADD_SUBDIR, AUTH_USERS)])
    assert _is_trusted_as_nt(monkeypatch, module, key) is True
    # the key dir itself owned by SYSTEM (strict leaf rule)
    table[str(key)] = (0x10, SYSTEM, STOCK_PROFILE)
    assert _is_trusted_as_nt(monkeypatch, module, key) is False
    table[str(key)] = owned
    # a junction / symlink anywhere on the walk
    table[str(root)] = (0x10 | 0x400, USER, STOCK_PROFILE)
    assert _is_trusted_as_nt(monkeypatch, module, key) is False
    table[str(root)] = owned
    # a file where a directory is expected
    table[str(key)] = (0x20, USER, STOCK_PROFILE)
    assert _is_trusted_as_nt(monkeypatch, module, key) is False


def test_walk_missing_target_follows_allow_missing(monkeypatch, tmp_path):
    module = _load_module()
    _stub_windows(monkeypatch, module, {str(tmp_path): (0x10, USER, STOCK_PROFILE)})
    missing = tmp_path / "not-yet"
    assert _is_trusted_as_nt(monkeypatch, module, missing) is True
    assert _is_trusted_as_nt(monkeypatch, module, missing, allow_missing = False) is False


def test_api_failure_means_untrusted(monkeypatch, tmp_path):
    module = _load_module()

    def broken():
        raise OSError("advapi32 unavailable")

    monkeypatch.setattr(module, "_windows_api", broken)
    target = tmp_path / "cache"
    target.mkdir()
    (target / "manifest.json").write_text("{}", encoding = "utf-8")
    with monkeypatch.context() as patch:
        patch.setattr(module.os, "name", "nt")
        assert module._is_trusted_directory(str(target)) is False
        assert module._read_trusted_file(str(target / "manifest.json")) is None
        assert module._ensure_private_directory(str(tmp_path / "new" / "dir")) is False


def _open_fds():
    return len(os.listdir("/proc/self/fd")) if os.path.isdir("/proc/self/fd") else None


@pytest.mark.skipif(os.name == "nt", reason = "stubs msvcrt; the real path is covered below")
def test_file_read_is_judged_on_the_open_handle(monkeypatch, tmp_path):
    """The Windows read path off Windows: owner / DACL / attributes come from
    the open handle, a file swapped between lstat and open is refused, and the
    descriptor is closed on every path."""
    module = _load_module()
    path = tmp_path / "manifest.json"
    path.write_bytes(b"{}")
    monkeypatch.setattr(module, "_windows_token_sids", lambda: (USER, USER))
    monkeypatch.setitem(sys.modules, "msvcrt", types.SimpleNamespace(get_osfhandle = lambda fd: fd))
    handles = []

    def read_with(security, **kwargs):
        def fake_handle_security(handle):
            handles.append(handle)
            return security
        monkeypatch.setattr(module, "_windows_handle_security", fake_handle_security)
        with monkeypatch.context() as patch:
            patch.setattr(module.os, "name", "nt")
            return module._read_trusted_file(str(path), **kwargs)

    before = _open_fds()
    assert read_with((0x20, USER, STOCK_PROFILE)) == "{}"
    assert read_with((0x20, USER, STOCK_PROFILE), binary = True) == b"{}"
    assert handles and all(isinstance(h, int) for h in handles)
    for security in (
        (0x20, OTHER, STOCK_PROFILE),                                         # foreign owner
        (0x20, USER, STOCK_PROFILE + [(ALLOW, INHERITED, WRITE, EVERYONE)]),  # foreign writer
        (0x20, USER, None),                                                   # NULL DACL
        (0x20 | 0x400, USER, STOCK_PROFILE),                                  # reparse point
        (0x20 | 0x40, USER, STOCK_PROFILE),                                   # device
        (0x10, USER, STOCK_PROFILE),                                          # a directory
    ):
        assert read_with(security) is None, security

    def broken(handle):
        raise OSError("GetSecurityInfo failed")
    monkeypatch.setattr(module, "_windows_handle_security", broken)
    with monkeypatch.context() as patch:
        patch.setattr(module.os, "name", "nt")
        assert module._read_trusted_file(str(path)) is None

    # Swapped between the lstat and the open: a different file id is refused
    # before its security is even queried.
    real_fstat = os.fstat

    def swapped_fstat(fd):
        st = real_fstat(fd)
        fields = list(st[:10])
        fields[1] = st.st_ino + 1
        return os.stat_result(fields)
    handles.clear()
    with monkeypatch.context() as patch:
        patch.setattr(module.os, "fstat", swapped_fstat)
        assert read_with((0x20, USER, STOCK_PROFILE)) is None
    assert handles == []
    if before is not None:
        assert _open_fds() == before


# --- real Windows ------------------------------------------------------------

def _icacls(path, *args):
    subprocess.run(["icacls", str(path), *args], check = True, capture_output = True)


@windows_only
def test_windows_default_cache_location_ancestors_are_trusted():
    module = _load_module()
    home = os.path.expanduser("~")
    probe = os.path.join(home, ".cache", "unsloth", "mega_cache", "trust-probe-%d" % os.getpid())
    assert not os.path.exists(probe)
    # Only existing components are checked: home and whatever of .cache exists.
    assert module._is_trusted_directory(probe) is True


@windows_only
def test_windows_tmp_dir_is_trusted_and_everyone_modify_is_not(tmp_path):
    module = _load_module()
    cache = tmp_path / "mega_cache"
    cache.mkdir()
    assert module._is_trusted_directory(str(cache)) is True
    _icacls(cache, "/grant", "*S-1-1-0:(OI)(CI)M")
    assert module._is_trusted_directory(str(cache)) is False


@windows_only
def test_windows_ancestor_rule(tmp_path):
    module = _load_module()
    root = tmp_path / "root"
    key = root / "key"
    key.mkdir(parents = True)
    assert module._is_trusted_directory(str(key)) is True
    _icacls(root, "/grant", "*S-1-1-0:(AD)")  # create folders, this folder only
    assert module._is_trusted_directory(str(key)) is True
    _icacls(root, "/grant", "*S-1-1-0:(DC)")  # delete child
    assert module._is_trusted_directory(str(key)) is False


@windows_only
def test_windows_junction_is_refused(tmp_path):
    module = _load_module()
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check = True, capture_output = True)
    assert module._is_trusted_directory(str(target)) is True
    assert module._is_trusted_directory(str(link)) is False
    assert module._is_trusted_directory(str(link / "child")) is False


@windows_only
def test_windows_file_with_foreign_write_is_not_read(tmp_path):
    module = _load_module()
    path = tmp_path / "manifest.json"
    path.write_text("{}", encoding = "utf-8")
    assert module._read_trusted_file(str(path)) == "{}"
    assert module._read_trusted_file(str(path), binary = True) == b"{}"
    _icacls(path, "/grant", "*S-1-1-0:W")
    assert module._read_trusted_file(str(path)) is None
    assert module._read_trusted_file(str(tmp_path / "missing.json")) is None


@windows_only
def test_windows_private_mkdir_drops_inherited_foreign_aces(tmp_path):
    module = _load_module()
    parent = tmp_path / "open"
    parent.mkdir()
    _icacls(parent, "/grant", "*S-1-1-0:(OI)(CI)M")
    plain = parent / "plain"
    plain.mkdir()
    assert any(sid == EVERYONE for _, _, _, sid in module._windows_path_security(plain)[2])
    private = parent / "private"
    module._windows_make_private_directory(str(private))
    sids = {sid for _, _, _, sid in module._windows_path_security(private)[2]}
    assert EVERYONE not in sids
    assert sids <= {module._windows_token_sids()[0], SYSTEM, ADMINS}
    with pytest.raises(FileExistsError):
        module._windows_make_private_directory(str(private))


@windows_only
def test_windows_save_then_load_round_trip(monkeypatch, tmp_path):
    module = _load_module()
    root = tmp_path / "mega_cache"
    data = b"locally generated artifact bundle"
    loaded = []
    fake_torch = types.SimpleNamespace(
        compiler = types.SimpleNamespace(
            load_cache_artifacts = lambda bundle: loaded.append(bundle) or {},
            save_cache_artifacts = lambda: (data, {}),
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("UNSLOTH_MEGA_CACHE", "1")
    monkeypatch.setenv("UNSLOTH_MEGA_CACHE_DIR", str(root))
    monkeypatch.setattr(module, "_bundle_key", lambda: "deadbeef")
    monkeypatch.setattr(module.atexit, "register", lambda function: None)
    module._STATE["armed"] = True
    module._STATE["loaded_key"] = "deadbeef"

    assert module.megacache_save() is True
    key_dir = root / "deadbeef"
    manifest = json.loads((key_dir / "manifest.json").read_text(encoding = "utf-8"))
    assert manifest["sha256"] == hashlib.sha256(data).hexdigest()
    assert module.megacache_load("windows-model") is True
    assert loaded == [data]

    # A foreign writer on the bundle file makes the next load a miss.
    _icacls(key_dir / manifest["bundle"], "/grant", "*S-1-1-0:W")
    loaded.clear()
    assert module.megacache_load("windows-model") is False
    assert loaded == []


_FRESH_PROCESS_LOAD = r'''
import importlib.util, json, sys, types
path, key = sys.argv[1], sys.argv[2]
loaded = []
sys.modules["torch"] = types.SimpleNamespace(compiler = types.SimpleNamespace(
    load_cache_artifacts = lambda bundle: loaded.append(bundle) or {},
    save_cache_artifacts = lambda: (b"", {})))
spec = importlib.util.spec_from_file_location("compile_cache_fresh", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module._bundle_key = lambda: key
module.atexit.register = lambda function: None
print(json.dumps({"loaded": module.megacache_load("windows-model"), "bytes": [len(b) for b in loaded],
                  "root": module._cache_root()}))
'''


@windows_only
def test_windows_stock_root_save_then_fresh_process_load(monkeypatch):
    """The real default root (%USERPROFILE%\\.cache\\unsloth\\mega_cache, no
    UNSLOTH_MEGA_CACHE_DIR): a key dir this code creates on save is trusted by
    the next process. Only the test's own key dir (and the root / its parents if
    the test created them) is removed afterwards."""
    import shutil
    import uuid

    module = _load_module()
    monkeypatch.delenv("UNSLOTH_MEGA_CACHE_DIR", raising = False)
    root = Path(module._cache_root())
    created = [p for p in (root.parent.parent, root.parent, root) if not p.exists()]
    key = "trust-roundtrip-%s" % uuid.uuid4().hex
    data = b"stock root artifact bundle"
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(compiler = types.SimpleNamespace(
        load_cache_artifacts = lambda bundle: {}, save_cache_artifacts = lambda: (data, {}))))
    monkeypatch.setenv("UNSLOTH_MEGA_CACHE", "1")
    monkeypatch.setattr(module, "_bundle_key", lambda: key)
    monkeypatch.setattr(module.atexit, "register", lambda function: None)
    module._STATE["armed"] = True
    module._STATE["loaded_key"] = key
    try:
        assert module.megacache_save() is True
        assert module._is_trusted_directory(str(root / key), allow_missing = False) is True
        env = dict(os.environ, UNSLOTH_MEGA_CACHE = "1")
        env.pop("UNSLOTH_MEGA_CACHE_DIR", None)
        out = subprocess.run([sys.executable, "-c", _FRESH_PROCESS_LOAD, str(_COMPILE_CACHE_PATH), key],
                             capture_output = True, text = True, env = env, timeout = 120)
        assert out.returncode == 0, out.stderr
        result = json.loads(out.stdout.strip().splitlines()[-1])
        assert result["root"] == str(root)
        assert result["loaded"] is True and result["bytes"] == [len(data)]
    finally:
        shutil.rmtree(root / key, ignore_errors = True)
        for directory in reversed(created):
            try:
                directory.rmdir()  # only if now empty
            except OSError:
                pass
