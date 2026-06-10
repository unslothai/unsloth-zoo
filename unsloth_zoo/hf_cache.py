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

"""Redirect the Hugging Face cache when the default location is read-only.

On locked-down machines the default cache (~/.cache/huggingface) is often not
writable, so snapshot_download() fails. We probe the directories Hub writes to
and, if they are not writable, point HF_HOME / HF_HUB_CACHE / HF_XET_CACHE at
the first writable fallback (temp dir, then the working directory).
"""

from __future__ import annotations

import os
import re
import tempfile
import warnings
from pathlib import Path

__all__ = ["redirect_hf_cache_if_readonly"]


def _expand_env_path(value: str) -> Path:
    # Hub applies expandvars + expanduser to env-provided paths; mirror that.
    return Path(os.path.expandvars(value)).expanduser()


def _is_writable(path: Path) -> bool:
    # Create the dir and a throwaway file; any failure means not writable.
    # Reject symlinks so a planted link cannot route cache writes elsewhere.
    try:
        if path.is_symlink():
            return False
        path.mkdir(parents = True, exist_ok = True)
        with tempfile.NamedTemporaryFile(dir = path):
            pass
        return True
    except Exception:
        return False


def _is_safe_private_dir(path: Path) -> bool:
    # Fallbacks live under predictable names in shared locations (e.g. /tmp),
    # so a pre-existing dir could belong to another local user. Only accept a
    # non-symlink dir we own, and clamp it to 0700 so cached models (and the
    # token file Hub keeps under HF_HOME) are not readable by other users.
    # The ownership check is POSIX-only; Windows temp dirs are per-user.
    try:
        path.mkdir(parents = True, exist_ok = True)
        if path.is_symlink():
            return False
        if hasattr(os, "getuid") and path.stat().st_uid != os.getuid():
            return False
        os.chmod(path, 0o700)
        return True
    except Exception:
        return False


def _safe_user() -> str:
    user = os.environ.get("USER") or os.environ.get("USERNAME") or ""
    if not user and hasattr(os, "getuid"):
        user = str(os.getuid())
    return re.sub(r"[^\w.-]", "_", user) or "user"


def _fallback_bases() -> list[Path]:
    # Ordered writable candidates used when the default cache is read-only.
    user = _safe_user()
    bases = [Path(tempfile.gettempdir()) / f"huggingface_{user}"]
    try:
        bases.append(Path.cwd() / ".cache" / "huggingface")
    except Exception:
        # cwd can be deleted or unreadable; the temp candidate still stands.
        pass
    return bases


def _resolve_hf_home() -> Path | None:
    # Mirror Hub's HF_HOME layering: HF_HOME, else XDG_CACHE_HOME/huggingface,
    # else ~/.cache/huggingface. Path.home() (and expanduser on "~" paths) can
    # raise on locked-down machines with an unresolvable home directory; treat
    # that as "no usable default" rather than crashing import.
    try:
        hf_home_env = os.environ.get("HF_HOME")
        if hf_home_env:
            return _expand_env_path(hf_home_env)
        xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
        if xdg_cache_home:
            return _expand_env_path(xdg_cache_home) / "huggingface"
        return Path.home() / ".cache" / "huggingface"
    except Exception:
        return None


def _active_caches() -> tuple[Path | None, Path | None, Path | None]:
    # (hf_home, hub_cache, xet_cache) as Hub will resolve them. Explicit env
    # vars are resolved first so they never depend on home resolution; the
    # legacy HUGGINGFACE_HUB_CACHE name is still honored by Hub.
    hf_home = _resolve_hf_home()
    hub_cache_env = (
        os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    )
    xet_cache_env = os.environ.get("HF_XET_CACHE")
    try:
        if hub_cache_env:
            hub_cache = _expand_env_path(hub_cache_env)
        else:
            hub_cache = hf_home / "hub" if hf_home is not None else None
    except Exception:
        hub_cache = None
    try:
        if xet_cache_env:
            xet_cache = _expand_env_path(xet_cache_env)
        else:
            xet_cache = hf_home / "xet" if hf_home is not None else None
    except Exception:
        xet_cache = None
    return hf_home, hub_cache, xet_cache


def redirect_hf_cache_if_readonly() -> str | None:
    """Repoint HF_HOME and the hub/xet caches when the active hub cache is not
    writable; when only the xet cache is unusable, just HF_XET_CACHE moves.
    Returns the new HF_HOME, or None when HF_HOME did not change."""
    hf_home, hub_cache, xet_cache = _active_caches()
    hub_ok = hub_cache is not None and _is_writable(hub_cache)
    xet_ok = xet_cache is not None and _is_writable(xet_cache)
    if hub_ok and xet_ok:
        return None
    # An explicitly set writable HF_XET_CACHE is kept as the user chose.
    keep_explicit_xet = xet_ok and bool(os.environ.get("HF_XET_CACHE"))

    for base in _fallback_bases():
        if not _is_safe_private_dir(base):
            continue
        if hub_ok:
            # Only the xet cache is unusable. Keep the hub cache (and any
            # already-downloaded models) where they are; move just the xet dir.
            if not (_is_safe_private_dir(base / "xet") and _is_writable(base / "xet")):
                continue
            os.environ["HF_XET_CACHE"] = str(base / "xet")
            warnings.warn(
                f"Unsloth: Hugging Face xet cache '{xet_cache}' is not writable; "
                f"redirecting xet downloads to '{base / 'xet'}'.",
                stacklevel = 2,
            )
            return None
        if not (_is_safe_private_dir(base / "hub") and _is_writable(base / "hub")):
            continue
        if not keep_explicit_xet and not (
            _is_safe_private_dir(base / "xet") and _is_writable(base / "xet")
        ):
            continue
        os.environ["HF_HOME"] = str(base)
        os.environ["HF_HUB_CACHE"] = str(base / "hub")
        if not keep_explicit_xet:
            os.environ["HF_XET_CACHE"] = str(base / "xet")
        # Moving HF_HOME also moves Hub's default token lookup; keep a
        # readable token from the old location working.
        old_token = hf_home / "token" if hf_home is not None else None
        if (
            old_token is not None
            and "HF_TOKEN" not in os.environ
            and "HF_TOKEN_PATH" not in os.environ
            and old_token.is_file()
        ):
            os.environ["HF_TOKEN_PATH"] = str(old_token)
        reason = (
            f"cache '{hub_cache}' is not writable"
            if hub_cache is not None
            else "default cache location could not be resolved"
        )
        warnings.warn(
            f"Unsloth: Hugging Face {reason}; "
            f"redirecting downloads to '{base}'.",
            stacklevel = 2,
        )
        return str(base)
    return None
