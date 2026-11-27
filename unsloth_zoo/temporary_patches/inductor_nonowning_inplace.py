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

"""torch < 2.16: Inductor can reuse a torch.cat input in place when that input is a NonOwningLayout view into the
cat's own output buffer. The cat buffer is then freed early and a later allocation overwrites live data, so the
compiled graph silently returns wrong values. torch 2.16 adds ir.NonOwningLayout to the layouts that
BaseSchedulerNode.decide_inplace_update never reuses; this applies the same one-line change to older torch.
Only that decision changes: a hit keeps one extra buffer, every other graph compiles to the same code."""

import __future__
import inspect
import os
import re
import textwrap

__all__ = ["patch_inductor_nonowning_inplace"]

NONOWNING_INPLACE_CACHE_TAG = "unsloth/inductor-nonowning-inplace"
_ANCHOR = "ir.MutationLayoutSHOULDREMOVE,"
_FLAG = "_unsloth_nonowning_inplace_guard"


def _torch_below_2_16():
    # 2.16 dev builds count: nightlies from before the fix still have the bug, and the source check skips fixed ones.
    import torch
    version = str(torch.__version__)
    found = re.match(r"(\d+)\.(\d+)", version)
    if found is None:
        return False
    major_minor = (int(found.group(1)), int(found.group(2)))
    return major_minor < (2, 16) or (major_minor == (2, 16) and ".dev" in version)


def _tag_inductor_cache():
    # The FX graph cache key hashes torch's files on disk, not this patch, so a warm cache would keep serving code
    # compiled without it.
    try:
        import torch.compiler.config as config
    except Exception:
        return
    current = getattr(config, "cache_key_tag", None)
    if not isinstance(current, str):
        return
    tags = [t for t in current.split(",") if t]
    if NONOWNING_INPLACE_CACHE_TAG in tags:
        return
    tag = ",".join(tags + [NONOWNING_INPLACE_CACHE_TAG])
    from unsloth_zoo.torch_config import set_torch_config
    set_torch_config(config, "cache_key_tag", tag)
    # Inductor compile workers read the tag from the environment at startup.
    os.environ["TORCH_COMPILE_CACHE_KEY_TAG"] = tag


def _insert_nonowning(source):
    """The upstream one-line change, or None when the anchor is not alone on its own line."""
    match = re.search(r"(?m)^([ \t]*)" + re.escape(_ANCHOR) + r"[ \t]*$", source)
    if match is None:
        return None
    return source[:match.end()] + "\n" + match.group(1) + "ir.NonOwningLayout," + source[match.end():]


def patch_inductor_nonowning_inplace():
    """Returns True when the guard is active (installed now or before), False when torch is fixed or unpatchable."""
    try:
        from torch._inductor import ir, scheduler
    except Exception:
        return False
    node = getattr(scheduler, "BaseSchedulerNode", None)
    original = getattr(node, "decide_inplace_update", None)
    if original is None or not hasattr(ir, "NonOwningLayout"):
        return False
    if getattr(original, _FLAG, False):
        return True
    if os.environ.get("UNSLOTH_INDUCTOR_NONOWNING_GUARD", "1") == "0":
        return False
    if not _torch_below_2_16():
        return False
    try:
        source = textwrap.dedent(inspect.getsource(original))
    except (OSError, TypeError):
        return False
    # Already fixed (2.16, or a backport), or the tuple moved: leave it alone.
    if "NonOwningLayout" in source or source.count(_ANCHOR) != 1 or "super(" in source:
        return False
    source = _insert_nonowning(source)
    if source is None:
        return False
    namespace = {}
    # Annotations stay strings, as in a module with `from __future__ import annotations`.
    flags = __future__.annotations.compiler_flag
    exec(compile(source, "<unsloth decide_inplace_update>", "exec", flags = flags, dont_inherit = True), scheduler.__dict__, namespace)
    patched = namespace["decide_inplace_update"]
    patched.__qualname__ = original.__qualname__
    patched.__doc__ = original.__doc__
    setattr(patched, _FLAG, True)
    patched._unsloth_original = original
    node.decide_inplace_update = patched
    _tag_inductor_cache()
    return True


try:
    patch_inductor_nonowning_inplace()
except Exception as error:
    if os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1":
        from unsloth_zoo.log import logger
        logger.warning(f"Unsloth: Could not apply the Inductor NonOwningLayout in-place guard: {error!r}")
