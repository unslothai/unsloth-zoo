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
writable, so snapshot_download() fails. We probe the directory Hub writes to
and, if it is not writable, point HF_HOME / HF_HUB_CACHE / HF_XET_CACHE at the
first writable fallback (temp dir, then the working directory).
"""

from __future__ import annotations

import os
import re
import tempfile
import warnings
from pathlib import Path

__all__ = ["redirect_hf_cache_if_readonly"]


def _is_writable(path: Path) -> bool:
    # Create the dir and a throwaway file; any failure means not writable.
    try:
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


def _default_caches() -> tuple[Path, Path] | None:
    # Mirror Hub's layering: HF_HUB_CACHE, else HF_HOME/hub, else
    # ~/.cache/huggingface/hub. Path.home() (and expanduser on "~" paths) can
    # raise on locked-down machines with an unresolvable home directory; treat
    # that as "no usable default" rather than crashing import.
    try:
        hf_home_env = os.environ.get("HF_HOME")
        if hf_home_env:
            hf_home = Path(hf_home_env).expanduser()
        else:
            hf_home = Path.home() / ".cache" / "huggingface"
        hub_cache_env = os.environ.get("HF_HUB_CACHE")
        if hub_cache_env:
            hub_cache = Path(hub_cache_env).expanduser()
        else:
            hub_cache = hf_home / "hub"
        return hf_home, hub_cache
    except Exception:
        return None


def redirect_hf_cache_if_readonly() -> str | None:
    """Repoint HF_HOME and the hub/xet caches when the active hub cache is not
    writable. Returns the new HF_HOME, or None when no change is needed."""
    resolved = _default_caches()
    if resolved is None:
        hf_home = hub_cache = None
    else:
        hf_home, hub_cache = resolved
        if _is_writable(hub_cache):
            return None
    for base in _fallback_bases():
        if not _is_safe_private_dir(base):
            continue
        if not (_is_writable(base / "hub") and _is_writable(base / "xet")):
            continue
        os.environ["HF_HOME"] = str(base)
        os.environ["HF_HUB_CACHE"] = str(base / "hub")
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
