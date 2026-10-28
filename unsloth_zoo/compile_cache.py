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
  UNSLOTH_MEGA_CACHE       = "1" (enable) | "0" (disable) | unset (default)
      Unset -> enabled on POSIX, where the trust checks below enforce that a
      cache directory is owned by the current user and not group/other writable
      (an untrusted directory is refused, not loaded, so the feature fails
      closed). Unset -> disabled on non-POSIX platforms. On Windows an opted-in
      cache is checked by owner SID and DACL instead (_windows_acl_is_trusted),
      failing closed on any API error; the default there is still off.
      torch.compile artifacts are executable/deserialized data.
      0 / off / false / no -> fully disabled (kill switch), any platform.
      1 / on / true / yes  -> enabled, any platform.
  UNSLOTH_MEGA_CACHE_DIR   = bundle root (default ~/.cache/unsloth/mega_cache).
"""

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
import struct
import sys
import time

_UNSLOTH_ENABLE_LOGGING = os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1"

_FORMAT_VERSION = 1
_BUNDLE_NAME = "megacache.bin"
_MANIFEST_NAME = "manifest.json"
_LOCK_NAME = ".lock"
# Bounded so a wedged writer cannot hang interpreter exit; on timeout the save is skipped.
_LOCK_TIMEOUT = 60

# The key is the SET of models loaded in this process, not the call order: torch's
# entries are content-addressed, so order would only mint keys that never hit.
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
    # Bundles are executable/deserialized torch.compile artifacts. On by
    # default where the POSIX trust checks can enforce it (owner-only, no
    # group/other write, no symlinks) so an untrusted cache is refused, not
    # loaded - the feature fails closed. On non-POSIX platforms, whose ACLs
    # those checks cannot model, require an explicit opt-in. Kill switch: 0.
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


# ---------------------------------------------------------------------------
# Windows trust check (owner SID + DACL), the counterpart of the POSIX owner and
# mode-bit checks. ctypes over advapi32 / kernel32 only. The decision logic
# (_windows_acl_is_trusted, _windows_parse_acl, _windows_parse_sid) is pure
# Python so it is testable anywhere; the ctypes layer only extracts the owner
# SID, the raw DACL bytes and the file attributes from an open handle. Every
# API failure raises inside the helpers and the callers turn it into
# "untrusted", so the check fails closed.
#
# Rules, for each existing component of the path:
#   leaf (the cache dir being validated, and every cache file read):
#     owner is the current user (or the token's default owner, which is
#     BUILTIN\Administrators for an elevated process), and no ACE that applies
#     to the object grants ANY write / delete / re-ACL right to anyone other
#     than the current user, SYSTEM, Administrators or TrustedInstaller.
#   ancestors (the root and every parent up to the drive):
#     owner is one of those principals, and no ACE grants anyone else a right
#     that can rename, delete or re-ACL an existing entry (FILE_DELETE_CHILD,
#     DELETE, WRITE_DAC, WRITE_OWNER). Add-file / add-subdirectory rights are
#     allowed there, like a sticky POSIX parent: they can create new names but
#     not replace an existing one, and a name created first by someone else is
#     refused by the owner check when the walk reaches it.
#   any component that is a reparse point (symlink, junction): untrusted.
#   a NULL DACL (everyone full access, also what ACL-less FAT/exFAT reports),
#   an unparseable ACL, or an allow ACE of a type not understood: untrusted.
# Deny ACEs only remove rights and are ignored; inherit-only ACEs do not apply
# to the object itself and are skipped (they are judged on the children, which
# are checked themselves when they are on the walk).
# ---------------------------------------------------------------------------

_WIN_SID_SYSTEM = "S-1-5-18"
_WIN_SID_ADMINISTRATORS = "S-1-5-32-544"
_WIN_SID_TRUSTED_INSTALLER = "S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464"
_WIN_SID_CREATOR_OWNER = "S-1-3-0"
_WIN_SID_CREATOR_GROUP = "S-1-3-1"
_WIN_SID_OWNER_RIGHTS = "S-1-3-4"
_WIN_SYSTEM_PRINCIPALS = (
    _WIN_SID_SYSTEM, _WIN_SID_ADMINISTRATORS, _WIN_SID_TRUSTED_INSTALLER,
)
# Placeholders that grant nothing on the object itself (CREATOR OWNER / GROUP
# are substituted only at inheritance) or that resolve to the owner, whom the
# owner check already requires to be trusted (OWNER RIGHTS).
_WIN_PLACEHOLDER_SIDS = (
    _WIN_SID_CREATOR_OWNER, _WIN_SID_CREATOR_GROUP, _WIN_SID_OWNER_RIGHTS,
)

_WIN_ACCESS_ALLOWED = 0x00
_WIN_ACCESS_DENIED = 0x01
_WIN_ACCESS_ALLOWED_CALLBACK = 0x09
_WIN_ACCESS_DENIED_CALLBACK = 0x0A
_WIN_DENY_ACE_TYPES = (0x01, 0x06, 0x0A, 0x0C)  # plain, object, callback, callback object
_WIN_SIMPLE_ACE_TYPES = (0x00, 0x01, 0x09, 0x0A)  # header, mask, SID (+ callback data)
_WIN_INHERIT_ONLY_ACE = 0x08

_WIN_FILE_WRITE_DATA = 0x0002        # FILE_ADD_FILE on a directory
_WIN_FILE_APPEND_DATA = 0x0004       # FILE_ADD_SUBDIRECTORY on a directory
_WIN_FILE_WRITE_EA = 0x0010
_WIN_FILE_DELETE_CHILD = 0x0040
_WIN_FILE_WRITE_ATTRIBUTES = 0x0100
_WIN_DELETE = 0x00010000
_WIN_WRITE_DAC = 0x00040000
_WIN_WRITE_OWNER = 0x00080000
_WIN_MAXIMUM_ALLOWED = 0x02000000
_WIN_GENERIC_ALL = 0x10000000
_WIN_GENERIC_EXECUTE = 0x20000000
_WIN_GENERIC_WRITE = 0x40000000
_WIN_GENERIC_READ = 0x80000000
_WIN_FILE_ALL_ACCESS = 0x001F01FF
_WIN_GENERIC_MAP = (
    (_WIN_GENERIC_READ, 0x00120089),
    (_WIN_GENERIC_WRITE, 0x00120116),
    (_WIN_GENERIC_EXECUTE, 0x001200A0),
    (_WIN_GENERIC_ALL, _WIN_FILE_ALL_ACCESS),
    (_WIN_MAXIMUM_ALLOWED, _WIN_FILE_ALL_ACCESS),  # never valid in an ACE: treat as everything
)
_WIN_REPLACE_RIGHTS = _WIN_FILE_DELETE_CHILD | _WIN_DELETE | _WIN_WRITE_DAC | _WIN_WRITE_OWNER
_WIN_WRITE_RIGHTS = (
    _WIN_REPLACE_RIGHTS | _WIN_FILE_WRITE_DATA | _WIN_FILE_APPEND_DATA
    | _WIN_FILE_WRITE_EA | _WIN_FILE_WRITE_ATTRIBUTES
)

_WIN_FILE_ATTRIBUTE_DIRECTORY = 0x10
_WIN_FILE_ATTRIBUTE_DEVICE = 0x40
_WIN_FILE_ATTRIBUTE_REPARSE_POINT = 0x400

# Token SIDs are fixed for the life of the process; cached once read.
_WIN_TOKEN_SIDS = []
# Private WinDLL handles with our own argtypes (never the shared ctypes.windll).
_WIN_API = []


def _windows_parse_sid(data, offset = 0):
    """Binary SID -> "S-R-A-S1-..." string. Raises ValueError on a bad SID."""
    if len(data) < offset + 8:
        raise ValueError("truncated SID")
    revision, count = data[offset], data[offset + 1]
    end = offset + 8 + 4 * count
    if revision != 1 or count > 15 or len(data) < end:
        raise ValueError("invalid SID")
    authority = int.from_bytes(bytes(data[offset + 2 : offset + 8]), "big")
    subs = struct.unpack_from("<%dI" % count, data, offset + 8)
    return "S-%d-%d" % (revision, authority) + "".join("-%d" % s for s in subs)
pass


def _windows_parse_acl(data):
    """Raw ACL bytes -> [(ace_type, ace_flags, mask, sid)]. Types other than the
    simple allow/deny (and callback) layouts come back with mask/sid None, for
    the decision function to refuse. Raises ValueError on a malformed ACL."""
    if len(data) < 8:
        raise ValueError("truncated ACL")
    revision, _, acl_size, ace_count, _ = struct.unpack_from("<BBHHH", data, 0)
    if revision not in (2, 4) or acl_size < 8 or acl_size > len(data):
        raise ValueError("invalid ACL header")
    aces = []
    offset = 8
    for _ in range(ace_count):
        if offset + 4 > acl_size:
            raise ValueError("truncated ACE header")
        ace_type, ace_flags, ace_size = struct.unpack_from("<BBH", data, offset)
        if ace_size < 4 or offset + ace_size > acl_size:
            raise ValueError("invalid ACE size")
        mask = sid = None
        if ace_type in _WIN_SIMPLE_ACE_TYPES:
            body = bytes(data[offset : offset + ace_size])
            if len(body) < 16:
                raise ValueError("truncated ACE")
            mask = struct.unpack_from("<I", body, 4)[0]
            sid = _windows_parse_sid(body, 8)
        aces.append((ace_type, ace_flags, mask, sid))
        offset += ace_size
    return aces
pass


def _windows_expand_generic(mask):
    for generic, specific in _WIN_GENERIC_MAP:
        if mask & generic:
            mask |= specific
    return mask
pass


def _windows_acl_is_trusted(owner_sid, aces, user_sids, *, leaf):
    """Pure decision: may a cache object with this owner and DACL be trusted?

    ``user_sids`` are the current token's user SID and default owner SID.
    ``aces`` is ``_windows_parse_acl`` output, or None for a NULL DACL.
    ``leaf`` selects the strict rule (cache dir / file) over the ancestor rule.
    """
    user_sids = tuple(sid for sid in user_sids if sid)
    if not user_sids or not owner_sid:
        return False
    allowed_owners = user_sids if leaf else user_sids + _WIN_SYSTEM_PRINCIPALS
    if owner_sid not in allowed_owners:
        return False
    if aces is None:
        return False
    trusted_writers = user_sids + _WIN_SYSTEM_PRINCIPALS + _WIN_PLACEHOLDER_SIDS
    forbidden = _WIN_WRITE_RIGHTS if leaf else _WIN_REPLACE_RIGHTS
    for ace_type, ace_flags, mask, sid in aces:
        if ace_flags & _WIN_INHERIT_ONLY_ACE:
            continue
        if ace_type in _WIN_DENY_ACE_TYPES:
            continue
        # Conditional (callback) allows are judged as unconditional grants;
        # object ACEs and anything else in a DACL are not understood here.
        if ace_type not in (_WIN_ACCESS_ALLOWED, _WIN_ACCESS_ALLOWED_CALLBACK):
            return False
        if mask is None or sid is None:
            return False
        if sid in trusted_writers:
            continue
        if _windows_expand_generic(mask) & forbidden:
            return False
    return True
pass


def _windows_api():
    """advapi32 / kernel32 entry points with explicit signatures. Raises off
    Windows (no ctypes.WinDLL), which every caller treats as untrusted."""
    if _WIN_API:
        return _WIN_API[0]
    import ctypes
    from ctypes import wintypes

    advapi32 = ctypes.WinDLL("advapi32", use_last_error = True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
    PVOID = ctypes.c_void_p
    PPVOID = ctypes.POINTER(ctypes.c_void_p)

    advapi32.GetSecurityInfo.argtypes = [
        wintypes.HANDLE, ctypes.c_int, wintypes.DWORD, PPVOID, PPVOID, PPVOID, PPVOID, PPVOID,
    ]
    advapi32.GetSecurityInfo.restype = wintypes.DWORD
    advapi32.IsValidSid.argtypes = [PVOID]
    advapi32.IsValidSid.restype = wintypes.BOOL
    advapi32.GetLengthSid.argtypes = [PVOID]
    advapi32.GetLengthSid.restype = wintypes.DWORD
    advapi32.IsValidAcl.argtypes = [PVOID]
    advapi32.IsValidAcl.restype = wintypes.BOOL
    advapi32.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    advapi32.OpenProcessToken.restype = wintypes.BOOL
    advapi32.GetTokenInformation.argtypes = [
        wintypes.HANDLE, ctypes.c_int, PVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.GetTokenInformation.restype = wintypes.BOOL
    advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, PPVOID, ctypes.POINTER(wintypes.ULONG),
    ]
    advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = wintypes.BOOL

    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, PVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    ]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CreateDirectoryW.argtypes = [wintypes.LPCWSTR, PVOID]
    kernel32.CreateDirectoryW.restype = wintypes.BOOL
    kernel32.GetFileInformationByHandle.argtypes = [wintypes.HANDLE, PVOID]
    kernel32.GetFileInformationByHandle.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [PVOID]
    kernel32.LocalFree.restype = PVOID
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    _WIN_API[:] = [(ctypes, wintypes, advapi32, kernel32)]
    return _WIN_API[0]
pass


def _windows_sid_at(ctypes, advapi32, psid):
    if not psid or not advapi32.IsValidSid(psid):
        raise OSError("invalid SID")
    return _windows_parse_sid(ctypes.string_at(psid, advapi32.GetLengthSid(psid)))
pass


def _windows_token_sids():
    """(user SID, default owner SID) of this process' token."""
    if _WIN_TOKEN_SIDS:
        return tuple(_WIN_TOKEN_SIDS)
    ctypes, wintypes, advapi32, kernel32 = _windows_api()
    TOKEN_QUERY = 0x0008
    token = wintypes.HANDLE()
    if not advapi32.OpenProcessToken(kernel32.GetCurrentProcess(), TOKEN_QUERY, ctypes.byref(token)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        sids = []
        for token_class in (1, 4):  # TokenUser, TokenOwner: both start with a PSID
            size = wintypes.DWORD(0)
            advapi32.GetTokenInformation(token, token_class, None, 0, ctypes.byref(size))
            if size.value == 0:
                raise ctypes.WinError(ctypes.get_last_error())
            buffer = ctypes.create_string_buffer(size.value)
            if not advapi32.GetTokenInformation(token, token_class, buffer, size, ctypes.byref(size)):
                raise ctypes.WinError(ctypes.get_last_error())
            psid = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_void_p))[0]
            sids.append(_windows_sid_at(ctypes, advapi32, psid))
    finally:
        kernel32.CloseHandle(token)
    _WIN_TOKEN_SIDS[:] = sids
    return tuple(sids)
pass


def _windows_handle_security(handle):
    """(file attributes, owner SID, parsed DACL or None) for an open handle."""
    ctypes, wintypes, advapi32, kernel32 = _windows_api()
    info = (wintypes.DWORD * 13)()  # BY_HANDLE_FILE_INFORMATION, attributes first
    if not kernel32.GetFileInformationByHandle(handle, info):
        raise ctypes.WinError(ctypes.get_last_error())
    SE_FILE_OBJECT = 1
    OWNER_SECURITY_INFORMATION, DACL_SECURITY_INFORMATION = 0x1, 0x4
    owner, dacl, descriptor = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    status = advapi32.GetSecurityInfo(
        handle, SE_FILE_OBJECT, OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION,
        ctypes.byref(owner), None, ctypes.byref(dacl), None, ctypes.byref(descriptor),
    )
    if status != 0:
        raise ctypes.WinError(status)
    try:
        owner_sid = _windows_sid_at(ctypes, advapi32, owner.value)
        aces = None
        if dacl.value:
            if not advapi32.IsValidAcl(dacl.value):
                raise OSError("invalid ACL")
            acl_size = struct.unpack_from("<H", ctypes.string_at(dacl.value, 4), 2)[0]
            aces = _windows_parse_acl(ctypes.string_at(dacl.value, acl_size))
        return int(info[0]), owner_sid, aces
    finally:
        if descriptor.value:
            kernel32.LocalFree(descriptor.value)
pass


def _windows_path_security(path):
    """Like _windows_handle_security for a path, opening the path itself (never
    a reparse point's target) so the attributes, owner and DACL describe the
    same object."""
    ctypes, wintypes, advapi32, kernel32 = _windows_api()
    READ_CONTROL, FILE_READ_ATTRIBUTES = 0x00020000, 0x0080
    SHARE_ALL, OPEN_EXISTING = 0x7, 3
    FILE_FLAG_BACKUP_SEMANTICS, FILE_FLAG_OPEN_REPARSE_POINT = 0x02000000, 0x00200000
    handle = kernel32.CreateFileW(
        os.fspath(path), READ_CONTROL | FILE_READ_ATTRIBUTES, SHARE_ALL, None, OPEN_EXISTING,
        FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, None,
    )
    if handle is None or handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return _windows_handle_security(handle)
    finally:
        kernel32.CloseHandle(handle)
pass


def _windows_security_is_trusted(security, *, directory, leaf):
    attributes, owner_sid, aces = security
    if attributes & (_WIN_FILE_ATTRIBUTE_REPARSE_POINT | _WIN_FILE_ATTRIBUTE_DEVICE):
        return False
    if bool(attributes & _WIN_FILE_ATTRIBUTE_DIRECTORY) != directory:
        return False
    return _windows_acl_is_trusted(owner_sid, aces, _windows_token_sids(), leaf = leaf)
pass


def _windows_directory_is_trusted(path, allow_missing):
    """The Windows walk of _is_trusted_directory: the target as leaf, every
    existing parent up to the drive root as an ancestor. Raises on API errors;
    the caller maps that to untrusted."""
    target = os.path.abspath(os.fspath(path))
    current = target
    target_exists = False
    while True:
        try:
            os.lstat(current)
            exists = True
        except FileNotFoundError:
            exists = False
        if exists:
            is_target = current == target
            target_exists = target_exists or is_target
            security = _windows_path_security(current)
            if not _windows_security_is_trusted(security, directory = True, leaf = is_target):
                return False
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return target_exists or allow_missing
pass


def _windows_make_private_directory(path):
    """mkdir with a protected owner-only DACL (current user, SYSTEM,
    Administrators; inherited by the files saved inside), the counterpart of
    POSIX mkdir 0o700. Raises FileExistsError if the path exists."""
    ctypes, wintypes, advapi32, kernel32 = _windows_api()
    user_sid = _windows_token_sids()[0]
    sddl = "D:P(A;OICI;FA;;;%s)(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)" % user_sid
    descriptor = ctypes.c_void_p()
    if not advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
        sddl, 1, ctypes.byref(descriptor), None
    ):
        raise ctypes.WinError(ctypes.get_last_error())

    class SECURITY_ATTRIBUTES(ctypes.Structure):
        _fields_ = [
            ("nLength", wintypes.DWORD),
            ("lpSecurityDescriptor", ctypes.c_void_p),
            ("bInheritHandle", wintypes.BOOL),
        ]

    try:
        attributes = SECURITY_ATTRIBUTES(ctypes.sizeof(SECURITY_ATTRIBUTES), descriptor.value, False)
        if not kernel32.CreateDirectoryW(os.fspath(path), ctypes.byref(attributes)):
            error = ctypes.get_last_error()
            if error == 183:  # ERROR_ALREADY_EXISTS
                raise FileExistsError(path)
            raise ctypes.WinError(error)
    finally:
        kernel32.LocalFree(descriptor.value)
pass


def _windows_read_trusted_file(path, binary):
    """The Windows branch of _read_trusted_file: refuse links and non-regular
    files, open, confirm the handle is the file that was checked, then judge
    owner and DACL on the open handle (no path re-resolution)."""
    try:
        before = os.lstat(path)
    except OSError:
        return None
    if (not stat.S_ISREG(before.st_mode)
            or getattr(before, "st_file_attributes", 0) & _WIN_FILE_ATTRIBUTE_REPARSE_POINT):
        return None
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    except OSError:
        return None
    try:
        after = os.fstat(fd)
        if (not stat.S_ISREG(after.st_mode)
                or (after.st_ino, after.st_dev) != (before.st_ino, before.st_dev)):
            return None
        import msvcrt
        security = _windows_handle_security(msvcrt.get_osfhandle(fd))
        if not _windows_security_is_trusted(security, directory = False, leaf = True):
            return None
        with os.fdopen(fd, "rb" if binary else "r") as file:
            fd = -1
            return file.read()
    except Exception:
        return None
    finally:
        if fd >= 0:
            os.close(fd)
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
        if os.name == "nt":
            return _windows_directory_is_trusted(path, allow_missing)
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
        if os.name == "nt":
            # Only missing components are created (protected owner-only
            # DACL); pre-existing ones are judged as they are, like POSIX.
            components = []
            current = target
            while not os.path.lexists(current):
                components.append(current)
                parent = os.path.dirname(current)
                if parent == current:
                    break
                current = parent
            for directory in reversed(components):
                try:
                    _windows_make_private_directory(directory)
                except FileExistsError:
                    pass
            return _is_trusted_directory(target)
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
            # The bound device, not device 0: mixed-arch hosts must not save
            # one GPU's kernels under another's key.
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
    # Validate the per-key directory: its walk also vets the root as an
    # ancestor (a sticky root-owned parent like /tmp is fine), so a shared
    # sticky root with a private owned key dir is permitted, while the key dir
    # itself must still be owned by us.
    if not _is_trusted_directory(directory):
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
    if os.name == "nt":
        return _windows_read_trusted_file(path, binary)
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


# Kinds torch stores as one file per key: merged by key so this run's copy wins.
# Inductor / AOTAutograd keep guarded variants per key, so they stay by content.
_SINGLE_ENTRY_KINDS = frozenset((
    "AutotuneCacheArtifact", "PGOCacheArtifact", "PrecompileCacheArtifact",
    "AUTOTUNE", "PGO",  # torch 2.7: one CacheArtifact class with a type enum
))
# Only re-timed autotune entries are interchangeable; a changed PGO profile or
# precompile package is a real update, so it still counts as new when saving.
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
    """Re-record an existing bundle's artifacts into torch's artifact manager
    so the next ``save_cache_artifacts`` serializes the UNION of the on-disk
    bundle and this process' artifacts. torch only records what this run
    loaded-and-hit or compiled, so without this a run that exercises a subset
    of a warm bundle (a different shape, one of several cached models) would
    replace the bundle with that subset and silently drop the rest. One copy
    per identity; this run's copy wins. Returns the disk bundle's save
    identities (None if unreadable). Best-effort: a failure may narrow the bundle."""
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
    """Create the lock file owner-only; refuse anything but a single-link regular
    file we own, so it cannot be redirected through a symlink or hard link."""
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
    """The OS lock, never SoftFileLock: a killed writer's existence lock would
    block every later save. Older filelock lacks ``fallback_to_soft`` / ``mode``."""
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
    """Non-blocking flock before filelock's acquire: filelock retries every flock
    error but ENOSYS until the timeout, so ENOLCK / EOPNOTSUPP filesystems (NFS
    without a lock manager) would stall exit for _LOCK_TIMEOUT."""
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
    """Serialize savers of one key across processes, or one writer collects the
    bundle another's manifest names and concurrent unions drop artifacts.
    Yields whether to save: unlocked only where locking is unsupported; a
    timeout or an untrusted lock file skips the save."""
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
    # A hit on an earlier cumulative key does not cover this one: reset so a
    # miss here still saves the extended bundle at exit.
    _STATE["hit"] = False
    if not _STATE["atexit_registered"]:
        atexit.register(megacache_save)
        _STATE["atexit_registered"] = True

    # Missing paths are safe to create later but never safe to read from.
    # Revalidate the key dir as existing (its walk re-vets the root ancestor)
    # right before opening the manifest so an attacker cannot win a
    # create-after-check race under a sticky shared parent such as /tmp.
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
        # Establish the owned private key dir (creating any missing parents as
        # 0700) before reading/deserializing any prior bundle; closes the
        # analogous race on the save path. A pre-existing sticky shared root is
        # left as-is and vetted as an ancestor by the trust walk.
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
    # Fold the on-disk bundle back into the manager before serializing so a
    # run that exercised only part of a warm bundle cannot narrow it (union
    # semantics; see _merge_recorded_artifacts).
    _, existing = _read_disk_bundle(directory, manifest_path)
    disk_ids = _merge_recorded_artifacts(existing) if existing is not None else None
    # Always serialize: it also records deferred Dynamo precompile entries.
    result = torch.compiler.save_cache_artifacts()
    if not result or not result[0]:
        _log("nothing to save (no compile artifacts recorded)")
        return False
    data = result[0]
    new_sha = hashlib.sha256(data).hexdigest()
    # Skip a no-op rewrite. Bytes alone miss it (recording order, re-timed
    # autotune entries), so compare identities too.
    if existing is not None and hashlib.sha256(existing).hexdigest() == new_sha:
        _log("bundle unchanged; skipping save")
        return False
    if disk_ids is not None:
        recorded = _recorded_identities()
        if recorded is not None and recorded <= disk_ids:
            _log("bundle already holds every recorded artifact; skipping save")
            return False
    # Content-addressed bundle + atomic manifest replace keep the pair
    # consistent even for an unlocked (fallback) writer.
    bundle_name = f"megacache-{new_sha[:16]}.bin"
    bundle_path = os.path.join(directory, bundle_name)
    # A same-named leftover may be group-writable (every load would reject it):
    # rewrite unless it passes the trusted-file checks.
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
    # a later reader re-reads the manifest. Per file: Windows refuses to
    # unlink a bundle a reader holds open.
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
