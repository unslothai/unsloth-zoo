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

"""Stdlib only (imported by the MLX path on Mac, where torch may be absent)."""

import os
import shutil
import stat

__all__ = ["precreate_private_temp", "current_umask", "keep_mode", "copy_keep_mode", "copytree_keep_mode"]


def precreate_private_temp(temporary, final):
    """On a re-export over an existing file, create the staging temp at 0600 before the writer fills it, so weights
    the user restricted are never readable at the umask default while written (writers that open with truncate keep
    the mode; the caller copies the old file's mode on before os.replace). First exports and Windows are unchanged."""
    if os.name == "nt" or not os.path.isfile(str(final)):
        return
    temporary = str(temporary)
    try:
        os.unlink(temporary)  # a leftover temp may be wider, or already open elsewhere
    except FileNotFoundError:
        pass
    except OSError:
        return
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    except OSError:
        return
    os.close(fd)


def current_umask():
    """The process umask, read without changing it where /proc allows (else briefly narrowed to 077 and restored)."""
    try:
        with open("/proc/self/status", encoding = "ascii") as f:
            for line in f:
                if line.startswith("Umask:"):
                    return int(line.split()[1], 8)
    except (OSError, ValueError):
        pass
    old = os.umask(0o077)
    os.umask(old)
    return old


def keep_mode(temporary, final, source_mode = None, like = (), add = 0):
    """Before publishing `temporary` as `final`: take the mode of the existing `final` (or the first existing path in
    `like`), so a re-export never widens weights the user restricted. With nothing existing, a copied file's
    `source_mode` is narrowed by the umask; else the temp keeps the mode it was written with. `add` bits are ORed in.
    Idempotent, never raises; on Windows only the `add` bits are applied (a read-only copy would block the next replace)."""
    if os.name == "nt":
        if add:
            try:
                os.chmod(temporary, stat.S_IMODE(os.stat(temporary).st_mode) | add)
            except OSError:
                pass
        return
    mode = None
    for path in (final, *like):
        try:
            mode = stat.S_IMODE(os.stat(path).st_mode)
            break
        except (OSError, TypeError, ValueError):
            continue
    if mode is None:
        if source_mode is None and not add:
            return
        try:
            mode = stat.S_IMODE(os.stat(temporary).st_mode) if source_mode is None else source_mode & ~current_umask()
        except OSError:
            return
    try:
        os.chmod(temporary, mode | add)
    except OSError:
        pass


def copy_keep_mode(src, dst, *, follow_symlinks = True):
    """shutil.copy2 (also as a copytree copy_function) that keeps an existing `dst`'s mode, or narrows the source's
    mode by the umask on a first copy, instead of copying a wider source mode across. The bytes go in through
    copyfile (an existing dst keeps its mode while written), then the times and the mode."""
    try:
        existing = stat.S_IMODE(os.stat(dst).st_mode)
    except OSError:
        existing = None
    # A first copy starts at 0600 so a restricted source is never readable at the umask default while written.
    created = False
    if existing is None and os.name != "nt" and (follow_symlinks or not os.path.islink(src)):
        try:
            os.close(os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600))
            created = True
        except OSError:
            pass
    try:
        result = shutil.copyfile(src, dst, follow_symlinks = follow_symlinks)
    except BaseException:
        if created:
            try: os.unlink(dst)
            except OSError: pass
        raise
    try:
        info = os.stat(src)
        os.utime(result, ns = (info.st_atime_ns, info.st_mtime_ns))
        if os.name != "nt":
            os.chmod(result, existing if existing is not None else stat.S_IMODE(info.st_mode) & ~current_umask())
    except OSError:
        pass
    return result


def copytree_keep_mode(src, dst):
    """shutil.copytree(dirs_exist_ok = True) with copy_keep_mode, where a folder that already existed keeps its own mode
    instead of the source folder's (copytree copies that onto every folder last), so a re-save never reopens a folder
    the user restricted. New folders take the source's mode as copytree gives it."""
    existing = {}
    if os.name != "nt":
        for root, dirs, _ in os.walk(dst):
            for path in (root, *(os.path.join(root, d) for d in dirs)):
                try:
                    existing[os.path.abspath(path)] = stat.S_IMODE(os.lstat(path).st_mode)
                except OSError:
                    pass
    result = shutil.copytree(src, dst, dirs_exist_ok = True, copy_function = copy_keep_mode)
    for path, mode in existing.items():
        try:
            if not os.path.islink(path):
                os.chmod(path, mode)
        except OSError:
            pass
    return result
