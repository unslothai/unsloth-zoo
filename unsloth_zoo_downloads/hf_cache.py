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

"""Redirect HF_HOME, HF_HUB_CACHE and HF_XET_CACHE to a writable fallback (temp dir, then cwd) when the default cache is read-only (snapshot_download would fail)."""

from __future__ import annotations

import os
import re
import tempfile
import warnings
from pathlib import Path

__all__ = ["redirect_hf_cache_if_readonly"]


def _expand_env_path(value: str) -> Path:
    # Hub applies os.path.expandvars(os.path.expanduser(...)) to env-provided paths; mirror that exact order.
    return Path(os.path.expandvars(os.path.expanduser(value)))


def _is_writable(path: Path) -> bool:
    # Any failure to create the dir or a throwaway file means not writable; symlinks are allowed (users symlink caches
    # to large volumes); fallbacks are hardened in _is_safe_private_dir.
    try:
        path.mkdir(parents = True, exist_ok = True)
        with tempfile.NamedTemporaryFile(dir = path):
            pass
        return True
    except Exception:
        return False


def _is_safe_private_dir(path: Path) -> bool:
    # Fallback names are predictable in shared dirs like /tmp, so accept only non-symlink dirs we own, clamped to 0700
    # so cached models and the HF_HOME token stay private; ownership check is POSIX-only.
    try:
        if hasattr(os, "getuid"):
            # /tmp is race-safe thanks to the sticky bit; a world-writable non-sticky ancestor we do not own is not, so refuse a fallback there.
            ancestor = path
            while not ancestor.exists():
                ancestor = ancestor.parent
            st = ancestor.stat()
            if (
                st.st_mode & 0o002
                and not st.st_mode & 0o1000
                and st.st_uid != os.getuid()
            ):
                return False
        # mode applies only to the leaf, so a fresh dir is never looser than 0700 under a permissive umask; chmod covers pre-existing dirs.
        path.mkdir(mode = 0o700, parents = True, exist_ok = True)
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
    # Mirrors Hub's HF_HOME order (HF_HOME, XDG_CACHE_HOME/huggingface, ~/.cache/huggingface); Path.home() can raise
    # without a home dir, which is treated as no usable default.
    try:
        # Presence-based (not truthiness) env lookups mirror Hub's os.getenv: a set-but-empty value resolves to a relative path.
        hf_home_env = os.environ.get("HF_HOME")
        if hf_home_env is not None:
            return _expand_env_path(hf_home_env)
        xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
        if xdg_cache_home is not None:
            return _expand_env_path(xdg_cache_home) / "huggingface"
        return Path.home() / ".cache" / "huggingface"
    except Exception:
        return None


def _active_caches() -> tuple[Path | None, Path | None, Path | None]:
    # Resolve (hf_home, hub_cache, xet_cache) as Hub will; explicit env vars come first so they never depend on home
    # resolution, and legacy HUGGINGFACE_HUB_CACHE is still honored by Hub.
    hf_home = _resolve_hf_home()
    hub_cache_env = os.environ.get("HF_HUB_CACHE")
    if hub_cache_env is None:
        hub_cache_env = os.environ.get("HUGGINGFACE_HUB_CACHE")
    xet_cache_env = os.environ.get("HF_XET_CACHE")
    try:
        if hub_cache_env is not None:
            hub_cache = _expand_env_path(hub_cache_env)
        else:
            hub_cache = hf_home / "hub" if hf_home is not None else None
    except Exception:
        hub_cache = None
    try:
        if xet_cache_env is not None:
            # Hub reads HF_XET_CACHE literally (no expanduser/expandvars), unlike HF_HOME and HF_HUB_CACHE; probe the same literal path.
            xet_cache = Path(xet_cache_env)
        else:
            xet_cache = hf_home / "xet" if hf_home is not None else None
    except Exception:
        xet_cache = None
    return hf_home, hub_cache, xet_cache


def redirect_hf_cache_if_readonly() -> str | None:
    """Repoint HF_HOME and the hub/xet caches when the hub cache is not writable; if only xet is unusable, just
    HF_XET_CACHE moves; returns new HF_HOME or None if unchanged."""
    hf_home, hub_cache, xet_cache = _active_caches()
    hub_ok = hub_cache is not None and _is_writable(hub_cache)
    xet_ok = xet_cache is not None and _is_writable(xet_cache)
    if hub_ok and xet_ok:
        return None
    if hub_cache is None:
        reason = "default cache location could not be resolved"
    elif not hub_ok:
        reason = f"cache '{hub_cache}' is not writable"
    elif xet_cache is None:
        reason = "xet cache location could not be resolved"
    else:
        reason = f"xet cache '{xet_cache}' is not writable"
    # An explicitly set writable HF_XET_CACHE is kept as the user chose.
    keep_explicit_xet = xet_ok and "HF_XET_CACHE" in os.environ

    for base in _fallback_bases():
        if not _is_safe_private_dir(base):
            continue
        if hub_ok:
            # Only the xet cache is unusable: keep the hub cache and already-downloaded models where they are; move just the xet dir.
            if not (_is_safe_private_dir(base / "xet") and _is_writable(base / "xet")):
                continue
            os.environ["HF_XET_CACHE"] = str(base / "xet")
            warnings.warn(
                f"Unsloth: Hugging Face {reason}; "
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
        # Moving HF_HOME also moves Hub's default token lookup; keep a readable token from the old location working.
        old_token = hf_home / "token" if hf_home is not None else None
        if (
            old_token is not None
            and "HF_TOKEN" not in os.environ
            and "HF_TOKEN_PATH" not in os.environ
            and old_token.is_file()
        ):
            os.environ["HF_TOKEN_PATH"] = str(old_token)
        warnings.warn(
            f"Unsloth: Hugging Face {reason}; "
            f"redirecting downloads to '{base}'.",
            stacklevel = 2,
        )
        return str(base)
    warnings.warn(
        f"Unsloth: Hugging Face {reason} and no writable fallback was found; "
        f"downloads may fail. Set HF_HOME to a writable directory.",
        stacklevel = 2,
    )
    return None
