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


def _safe_user() -> str:
    user = os.environ.get("USER") or os.environ.get("USERNAME") or ""
    if not user and hasattr(os, "getuid"):
        user = str(os.getuid())
    return re.sub(r"[^\w.-]", "_", user) or "user"


def _fallback_bases() -> list[Path]:
    # Ordered writable candidates used when the default cache is read-only.
    user = _safe_user()
    return [
        Path(tempfile.gettempdir()) / f"huggingface_{user}",
        Path.cwd() / ".cache" / "huggingface",
    ]


def redirect_hf_cache_if_readonly() -> str | None:
    """Repoint HF_HOME and the hub/xet caches when the active hub cache is not
    writable. Returns the new HF_HOME, or None when no change is needed."""
    hf_home = Path(
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    ).expanduser()
    hub_cache = Path(os.environ.get("HF_HUB_CACHE", hf_home / "hub")).expanduser()
    if _is_writable(hub_cache):
        return None
    for base in _fallback_bases():
        if _is_writable(base / "hub"):
            os.environ["HF_HOME"] = str(base)
            os.environ["HF_HUB_CACHE"] = str(base / "hub")
            os.environ["HF_XET_CACHE"] = str(base / "xet")
            warnings.warn(
                f"Unsloth: Hugging Face cache '{hub_cache}' is not writable; "
                f"redirecting downloads to '{base}'.",
                stacklevel = 2,
            )
            return str(base)
    return None
