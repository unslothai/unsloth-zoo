# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Auto-installer for the fused lm_head + cross_entropy forward.

Tier 1 swaps the forward via a hand-registered structural-hash allowlist
(empty by default; populate with `register_canonical`). Tier 2 falls back
to `ast_rewriter` which rewrites the canonical HF triplet in-place; misses
go to `_UNMATCHED` and the LOSS_MAPPING sweep stays as the backstop.

On by default. Set `UNSLOTH_FUSED_FORWARD=0` to disable. Soft floor at
transformers >= 4.56, the release where every `*ForCausalLM` settled on
the `outputs.last_hidden_state` + `self.loss_function(logits, labels,
vocab_size, **kwargs)` shape we match against.
"""

from __future__ import annotations

__all__ = [
    "install_modeling_import_hook",
    "install_for_module",
    "install_for_class",
    "register_canonical",
    "audit",
    "is_enabled",
    "EMPTY_LOGITS",
    "unsloth_fused_lm_head_loss",
]

import ast
import hashlib
import importlib.abc
import importlib.util
import inspect
import linecache
import logging
import os
import sys
import textwrap
import threading
import warnings
from typing import Any

from .ast_rewriter import rewrite_forward_source
from .forward_adapter import EMPTY_LOGITS, unsloth_fused_lm_head_loss


logger = logging.getLogger("unsloth_zoo.fused_forward")

_MIN_TRANSFORMERS = (4, 56, 0)

_REGISTRY_LOCK = threading.RLock()
_PATCHED: dict[str, dict[str, Any]] = {}   # qualname -> {tier, kind, hash, module}
_UNMATCHED: dict[str, str] = {}            # qualname -> reason
_FAILED: dict[str, str] = {}               # qualname -> error
_CANONICAL_FORWARDS: dict[str, Any] = {}   # forward_hash -> replacement forward fn

_INSTALL_DONE = False  # set once install_modeling_import_hook has run


def is_enabled() -> bool:
    # On by default; opt out via UNSLOTH_FUSED_FORWARD=0.
    return os.environ.get("UNSLOTH_FUSED_FORWARD", "1") != "0"


def register_canonical(forward_hash: str, replacement_forward) -> None:
    """Register a hand-written canonical forward for a known structural hash.
    Future installs that fingerprint to `forward_hash` get the replacement
    directly without the AST rewrite step."""
    with _REGISTRY_LOCK:
        _CANONICAL_FORWARDS[forward_hash] = replacement_forward


def audit() -> dict[str, Any]:
    """Snapshot of what's been patched / left alone / errored. JSON-safe."""
    with _REGISTRY_LOCK:
        return {
            "enabled": is_enabled(),
            "n_patched": len(_PATCHED),
            "n_unmatched": len(_UNMATCHED),
            "n_failed": len(_FAILED),
            "patched": dict(_PATCHED),
            "unmatched": dict(_UNMATCHED),
            "failed": dict(_FAILED),
            "canonical_hashes_registered": sorted(_CANONICAL_FORWARDS),
        }


def _transformers_version_ok() -> bool:
    try:
        import transformers  # noqa: PLC0415
    except Exception:
        return False
    v = getattr(transformers, "__version__", "0.0.0")
    parts = []
    for chunk in v.split("+")[0].split("."):
        try:
            parts.append(int(chunk))
        except ValueError:
            parts.append(0)
        if len(parts) == 3:
            break
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts) >= _MIN_TRANSFORMERS


def _structural_hash(fn) -> str | None:
    try:
        src = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        return None
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    # Strip docstrings so cosmetic changes do not bust the hash.
    _BODY_HOLDERS = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    for node in ast.walk(tree):
        if not isinstance(node, _BODY_HOLDERS):
            continue
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            body.pop(0)
    return hashlib.sha256(
        ast.dump(tree, annotate_fields=False, include_attributes=False).encode()
    ).hexdigest()[:16]


_LINEAR_HEAD_ATTRS = {
    "lm_head",
    "output_projection",
    "embed_out",
    "proj_out",
    "generator_lm_head",
    "head",
    "logits_dense",
    "codec_head",
}


def _is_eligible_class(cls) -> bool:
    # ForConditionalGeneration uses aligned labels; fused kernel hardcodes
    # a causal shift, so accept only ForCausalLM.
    name = getattr(cls, "__name__", "")
    if not name.endswith("ForCausalLM"):
        return False
    if not hasattr(cls, "forward"):
        return False
    return True


def install_for_class(cls) -> bool:
    """Try to install the fused forward on `cls`. Returns True on success."""
    if not is_enabled():
        return False
    if not _transformers_version_ok():
        return False
    if not _is_eligible_class(cls):
        return False
    qn = getattr(cls, "__qualname__", cls.__name__)
    with _REGISTRY_LOCK:
        if qn in _PATCHED:
            return True

    forward = cls.forward
    fhash = _structural_hash(forward)

    # Tier 1: hash-allowlisted function override.
    if fhash is not None:
        replacement = _CANONICAL_FORWARDS.get(fhash)
        if replacement is not None:
            try:
                replacement.__qualname__ = forward.__qualname__
                replacement.__module__ = forward.__module__
            except Exception:
                pass
            cls.forward = replacement
            with _REGISTRY_LOCK:
                _PATCHED[qn] = {
                    "tier": "1-function-override",
                    "kind": cls.__name__,
                    "hash": fhash,
                    "module": getattr(cls, "__module__", ""),
                }
            return True

    # Tier 2: AST triplet rewrite.
    try:
        src = textwrap.dedent(inspect.getsource(forward))
    except (OSError, TypeError) as exc:
        with _REGISTRY_LOCK:
            _UNMATCHED[qn] = f"source-unavailable: {exc}"
        return False

    new_src, cap = rewrite_forward_source(src)
    if new_src is None:
        with _REGISTRY_LOCK:
            _UNMATCHED[qn] = "no-canonical-triplet"
        return False
    # Composite heads (e.g. BigBird's BigBirdOnlyMLMHead via self.cls) lack
    # .weight/.bias and would crash inside the adapter.
    if cap.head_attr not in _LINEAR_HEAD_ATTRS:
        with _REGISTRY_LOCK:
            _UNMATCHED[qn] = f"non-linear-head: {cap.head_attr}"
        return False

    ns = dict(getattr(forward, "__globals__", {}))
    ns["unsloth_fused_lm_head_loss"] = unsloth_fused_lm_head_loss
    ns["EMPTY_LOGITS"] = EMPTY_LOGITS
    # The rewritten body reads UNSLOTH_RETURN_LOGITS via os.environ.get.
    ns.setdefault("os", os)
    try:
        from transformers.utils.generic import can_return_tuple
        ns.setdefault("can_return_tuple", can_return_tuple)
    except Exception:
        pass
    # Backfill transformers.modeling_outputs symbols; unsloth's compiled-cache
    # forwards reference CausalLMOutputWithPast & friends in the return line.
    try:
        import transformers.modeling_outputs as _mo
        for _name in dir(_mo):
            if _name.startswith("_"):
                continue
            ns.setdefault(_name, getattr(_mo, _name))
    except Exception:
        pass
    # Register rewritten source with linecache so inspect.getsource and
    # tracebacks see the installed body.
    synthetic_path = f"<unsloth-fused:{qn}>"
    linecache.cache[synthetic_path] = (
        len(new_src), None,
        [line + "\n" for line in new_src.splitlines()],
        synthetic_path,
    )
    try:
        code = compile(new_src, synthetic_path, "exec")
        exec(code, ns)
    except Exception as exc:
        with _REGISTRY_LOCK:
            _FAILED[qn] = f"compile-or-exec: {type(exc).__name__}: {exc}"
        return False

    new_forward = ns.get(forward.__name__)
    if not callable(new_forward):
        with _REGISTRY_LOCK:
            _FAILED[qn] = "rewritten-forward-missing"
        return False
    try:
        new_forward.__qualname__ = forward.__qualname__
        new_forward.__module__ = forward.__module__
        new_forward.__doc__ = forward.__doc__
    except Exception:
        pass

    cls.forward = new_forward
    with _REGISTRY_LOCK:
        _PATCHED[qn] = {
            "tier": "2-ast-triplet",
            "kind": cls.__name__,
            "hash": fhash,
            "module": getattr(cls, "__module__", ""),
            "head_attr": cap.head_attr,
        }
    return True


def install_for_module(module) -> int:
    """Scan a transformers `modeling_*` module and install where eligible.
    Returns the number of classes newly patched."""
    if not is_enabled():
        return 0
    if not _transformers_version_ok():
        return 0
    name = getattr(module, "__name__", "")
    if not (name.startswith("transformers.models.") and ".modeling_" in name):
        return 0
    n = 0
    for attr in dir(module):
        try:
            obj = getattr(module, attr)
        except Exception:
            continue
        if not isinstance(obj, type):
            continue
        if getattr(obj, "__module__", "") != name:
            continue  # skip re-exports
        try:
            if install_for_class(obj):
                n += 1
        except Exception as exc:
            qn = getattr(obj, "__qualname__", obj.__name__)
            with _REGISTRY_LOCK:
                _FAILED[qn] = f"install: {type(exc).__name__}: {exc}"
    return n


class _ModelingLoader(importlib.abc.Loader):
    """Wraps an inner loader, runs `install_for_module` after exec_module."""
    def __init__(self, inner):
        self._inner = inner

    def create_module(self, spec):
        if hasattr(self._inner, "create_module"):
            return self._inner.create_module(spec)
        return None

    def exec_module(self, module):
        self._inner.exec_module(module)
        try:
            install_for_module(module)
        except Exception as exc:
            logger.debug(
                "unsloth fused-forward install_for_module failed for %s: %s",
                getattr(module, "__name__", "?"), exc,
            )


class _ModelingFinder(importlib.abc.MetaPathFinder):
    """Intercepts `transformers.models.<X>.modeling_<X>` imports."""
    PREFIX = "transformers.models."

    def find_spec(self, fullname, path, target=None):
        if not (fullname.startswith(self.PREFIX) and ".modeling_" in fullname):
            return None
        if fullname in sys.modules:
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            try:
                spec = finder.find_spec(fullname, path, target)
            except Exception:
                continue
            if spec is None or spec.loader is None:
                continue
            spec.loader = _ModelingLoader(spec.loader)
            return spec
        return None


def install_modeling_import_hook() -> bool:
    """Register the meta-path finder + scan already-imported modeling modules.
    Returns True if the hook was installed (or already present); False if the
    install was skipped (feature disabled, transformers missing, version too
    old)."""
    global _INSTALL_DONE
    if _INSTALL_DONE:
        return True
    if not is_enabled():
        return False
    if not _transformers_version_ok():
        warnings.warn(
            "Unsloth fused-forward install skipped: requires transformers >= "
            f"{'.'.join(map(str, _MIN_TRANSFORMERS))}.",
            stacklevel=2,
        )
        return False
    if not any(isinstance(f, _ModelingFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _ModelingFinder())
    # Catch modules already imported before zoo loaded.
    for name in list(sys.modules):
        if name.startswith("transformers.models.") and ".modeling_" in name:
            mod = sys.modules.get(name)
            if mod is None:
                continue
            try:
                install_for_module(mod)
            except Exception:
                continue
    _INSTALL_DONE = True
    return True
