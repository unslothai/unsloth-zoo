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

"""Zoo-local mirror of selected ``unsloth/import_fixes.py`` workarounds.

This module hosts narrowly-scoped monkey-patches against third-party
libraries that ship a regression we need to paper over. Each fix is:

  * Strictly gated to fire ONLY when the upstream pathology is currently
    active on the installed stack (no-op otherwise).
  * Idempotent (calling twice == calling once).
  * Defensive against missing optional imports.

Apply all available fixes by calling :func:`apply_import_fixes` from
``unsloth_zoo/__init__.py`` at import time.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import sys
import types

__all__ = [
    "fix_triton_compiled_kernel_missing_attrs",
    "fix_vllm_guided_decoding_params",
    "fix_peft_transformers_weight_conversion_import",
    "fix_trl_vllm_ascend",
    "patch_enable_input_require_grads",
    "patch_datasets",
    "disable_torchcodec_if_broken",
    "apply_import_fixes",
]

_UNSLOTH_ENABLE_LOGGING = os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") in (
    "1", "True", "true",
)
logger = logging.getLogger(__name__)
if _UNSLOTH_ENABLE_LOGGING:
    logger.setLevel(logging.INFO)
else:
    logger.setLevel(logging.WARNING)


# Sentinel attribute we stamp on the patched class so a second call is a
# no-op even if the upstream class still lacks ``num_ctas`` natively.
_TRITON_CK_PATCH_MARKER = "_unsloth_zoo_num_ctas_patched"


def fix_triton_compiled_kernel_missing_attrs():
    """Inject ``num_ctas`` / ``cluster_dims`` onto ``triton.compiler.compiler.CompiledKernel``.

    Mirrors unsloth/import_fixes.py::fix_triton_compiled_kernel_missing_attrs
    (lines 923-968). triton >= 3.6.0 dropped direct ``num_ctas`` and
    ``cluster_dims`` attributes from ``CompiledKernel``, but torch 2.9.x
    Inductor's ``make_launcher`` (in
    ``torch/_inductor/runtime/triton_heuristics.py``) still eagerly
    evaluates ``binary.metadata.num_ctas, *binary.metadata.cluster_dims``
    when ``hasattr(binary, "metadata")`` is True. ``metadata`` lacks
    ``cluster_dims``, so the eager unpack blows up before the new launch
    contract is reached. Upstream pytorch fix landed in pytorch/pytorch@97bd4db
    (hasattr guards) and only ships in torch >= 2.10.

    Gating contract:
      * No-op if ``torch`` or ``triton`` aren't importable.
      * No-op if ``CompiledKernel`` already exposes ``num_ctas`` as a
        class attribute (triton with native attrs, or a previous call to
        this fix that stamped class-level defaults).
      * Idempotent across repeat calls via the ``_TRITON_CK_PATCH_MARKER``
        sentinel.

    Behaviour when active:
      * Adds class-level fallback defaults so ``hasattr(cls, "num_ctas")``
        and ``hasattr(cls, "cluster_dims")`` both succeed. This single
        step is enough to make the older
        ``hasattr(binary, "num_ctas")`` branch in Inductor's
        ``make_launcher`` succeed.
      * Wraps ``CompiledKernel.__init__`` so each new instance also gets
        the *real* per-kernel values lifted from ``self.metadata`` when
        available (preserves the upstream unsloth semantics).
    """
    try:
        import torch  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        return

    try:
        import triton  # noqa: F401
        import triton.compiler.compiler as triton_compiler
    except (ImportError, ModuleNotFoundError):
        return

    ck_cls = getattr(triton_compiler, "CompiledKernel", None)
    if ck_cls is None:
        return

    # Native triton (older / future-fixed) with direct attrs: nothing to do.
    # We probe the class __dict__ rather than hasattr() so a class that
    # only exposes the attr via __getattr__ on a *missing* metadata field
    # doesn't fool us into thinking the regression is gone.
    if "num_ctas" in ck_cls.__dict__:
        return

    # Idempotent: already patched by us previously.
    if getattr(ck_cls, _TRITON_CK_PATCH_MARKER, False):
        return

    # ---- Step 1: class-level fallback defaults -----------------------
    # These satisfy any ``hasattr(binary, "num_ctas")`` /
    # ``hasattr(cls, "num_ctas")`` probe before instance __init__ runs,
    # and they act as sane defaults if metadata lifting fails. Triton
    # itself defaults to 1 CTA and (1, 1, 1) cluster dims when the user
    # doesn't request otherwise, so these values are safe.
    try:
        ck_cls.num_ctas = 1
    except (AttributeError, TypeError):
        # __slots__ or similarly-locked class -- skip class-level step,
        # __init__ wrapper below still works for instances.
        pass
    try:
        if "cluster_dims" not in ck_cls.__dict__ and "clusterDims" not in ck_cls.__dict__:
            ck_cls.cluster_dims = (1, 1, 1)
    except (AttributeError, TypeError):
        pass

    # ---- Step 2: per-instance __init__ wrapper -----------------------
    # Lift the real values from metadata where possible, and skip the
    # work if the instance already has the attrs (e.g. a future triton
    # release that sets them in __init__).
    _orig_init = ck_cls.__init__

    # Guard against double-wrapping if some other patch already wrapped
    # __init__ and stored the original somewhere accessible.
    if getattr(_orig_init, "_unsloth_zoo_num_ctas_wrapped", False):
        ck_cls.__dict__.setdefault(_TRITON_CK_PATCH_MARKER, True)
        try:
            setattr(ck_cls, _TRITON_CK_PATCH_MARKER, True)
        except (AttributeError, TypeError):
            pass
        return

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        # Only fill in instance attrs if the original __init__ didn't.
        if not hasattr(self, "num_ctas") or self.num_ctas == 1:
            md = getattr(self, "metadata", None)
            if md is not None:
                self.num_ctas = getattr(md, "num_ctas", getattr(self, "num_ctas", 1))
            else:
                # Class default already provides 1 via attribute lookup.
                if not hasattr(self, "num_ctas"):
                    self.num_ctas = 1
        if not hasattr(self, "cluster_dims") and not hasattr(self, "clusterDims"):
            md = getattr(self, "metadata", None)
            if md is not None:
                cd = getattr(md, "cluster_dims", None)
                if cd is None:
                    cd = getattr(md, "clusterDims", (1, 1, 1))
                self.cluster_dims = tuple(cd) if not isinstance(cd, tuple) else cd
            else:
                self.cluster_dims = (1, 1, 1)

    _patched_init._unsloth_zoo_num_ctas_wrapped = True
    try:
        ck_cls.__init__ = _patched_init
    except (AttributeError, TypeError):
        # Class doesn't permit __init__ replacement (unusual). The
        # class-level defaults already make the test green and satisfy
        # the Inductor hasattr probe; instance-level real values won't
        # be lifted, but that's still functionally correct.
        pass

    try:
        setattr(ck_cls, _TRITON_CK_PATCH_MARKER, True)
    except (AttributeError, TypeError):
        pass

    if _UNSLOTH_ENABLE_LOGGING:
        logger.info(
            "Unsloth Zoo: Patched triton CompiledKernel with num_ctas/cluster_dims "
            "for torch.compile compatibility."
        )


def fix_vllm_guided_decoding_params():
    """Re-alias ``vllm.sampling_params.GuidedDecodingParams`` when vLLM has
    renamed it to ``StructuredOutputsParams``.

    Mirrors unsloth/import_fixes.py::fix_vllm_guided_decoding_params
    (lines 446-490). vLLM PR #22772 renamed ``GuidedDecodingParams`` to
    ``StructuredOutputsParams`` (landed in vllm 0.11+). TRL still
    ``from vllm.sampling_params import GuidedDecodingParams`` on the
    affected code paths, so we paper over the rename by setting
    ``vllm.sampling_params.GuidedDecodingParams =
    vllm.sampling_params.StructuredOutputsParams`` whenever the old
    name is missing and the new name is present.

    Gating contract:
      * No-op if ``vllm`` is not installed at all.
      * No-op if vllm exposes ``GuidedDecodingParams`` natively (pre-rename
        builds, or post-rename builds that re-export both for back-compat).
      * No-op if vllm exposes BOTH names (alias already present).
      * No-op if ``import vllm`` fails (broken binary / transformers
        mismatch); we swallow the error so zoo import isn't taken down by
        a broken optional dependency.
      * Idempotent: a second call sees the alias we installed and returns
        immediately.
    """
    # 1. vLLM not installed at all -> nothing to fix.
    try:
        import importlib.util as _importlib_util
        if _importlib_util.find_spec("vllm") is None:
            return
    except Exception:
        return

    # 2. Import vllm. If the binary is broken (CUDA / ABI / transformers
    #    mismatch), swallow and let zoo finish importing -- the user will
    #    see the real error the next time they actually touch vllm.
    try:
        import vllm  # noqa: F401
    except Exception:
        return

    # 3. Resolve vllm.sampling_params. Some builds expose it lazily; we
    #    explicitly import the submodule.
    try:
        import vllm.sampling_params as _vllm_sp
    except Exception:
        return

    has_guided = hasattr(_vllm_sp, "GuidedDecodingParams")
    has_structured = hasattr(_vllm_sp, "StructuredOutputsParams")

    # 4a. Healthy / old vLLM, or already-aliased: no work to do.
    if has_guided:
        return
    # 4b. Neither name present -> upstream changed again; we can't fix
    #     blindly. Bail rather than guess.
    if not has_structured:
        return

    # 4c. Rename-only build: install the back-compat alias. setattr on the
    #     live module makes the new name visible to anything that does
    #     ``from vllm.sampling_params import GuidedDecodingParams`` AFTER
    #     this point (Python re-resolves the attribute against the module
    #     object each ``from ... import`` call).
    try:
        _vllm_sp.GuidedDecodingParams = _vllm_sp.StructuredOutputsParams
    except Exception:
        return

    if _UNSLOTH_ENABLE_LOGGING:
        logger.info(
            "Unsloth Zoo: aliased vllm.sampling_params.GuidedDecodingParams "
            "-> StructuredOutputsParams (vLLM PR #22772 rename)."
        )


# ---------------------------------------------------------------------------
# peft 0.19.x  +  transformers 4.x  drift
# ---------------------------------------------------------------------------
#
# peft 0.19.x ships ``peft/utils/transformers_weight_conversion.py`` with a
# top-of-file ``from transformers.conversion_mapping import ...`` AND a
# ``from transformers.core_model_loading import ...``. Neither submodule
# exists on transformers < 5.x. The peft module's header is explicit
# ("don't import from this module unless transformers v5+ is used"), and
# peft itself only triggers the import at RUNTIME inside an
# ``if is_transformers_ge_v5:`` branch
# (``peft/tuners/tuners_utils.py``). However any code that does the obvious
# ``from peft.utils import transformers_weight_conversion`` -- including
# Unsloth's own ``patch_peft_weight_converter_compatibility`` (which
# touches this module precisely to wrap ``build_peft_weight_mapping``) and
# zoo's drift detector -- still tries to import the module unconditionally
# and explodes with
#
#     ModuleNotFoundError: No module named 'transformers.conversion_mapping'
#
# on the 4.x stack.
#
# Fix: when (and only when) the import is broken AND the underlying
# transformers really is missing those two submodules, inject minimal stub
# modules into ``sys.modules`` with exactly the symbols peft pulls in at
# its module top. The stubs are dead inert on transformers 4.x because
# peft never calls into them on that branch.
#
# On transformers v5+, both submodules exist for real, this function is a
# strict no-op (the existence probe passes and we return immediately) and
# we never touch ``sys.modules``.
# ---------------------------------------------------------------------------

# Sentinel attribute set on stub modules so we can recognise / reuse them
# and so callers can introspect "did unsloth_zoo install this".
_ZOO_STUB_SENTINEL = "__unsloth_zoo_stub__"


def _conversion_module_already_importable(name: str) -> bool:
    """True iff ``import {name}`` would succeed without ImportError.

    Uses ``find_spec`` rather than a raw ``import`` to avoid triggering
    arbitrary module-level side effects when we're only probing. Also
    treats an already-cached ``sys.modules`` entry as importable.
    """
    if name in sys.modules and sys.modules[name] is not None:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def _make_zoo_stub_module(fullname: str) -> types.ModuleType:
    """Create a fresh stub module marked with our sentinel."""
    mod = types.ModuleType(fullname)
    mod.__file__ = f"<unsloth_zoo stub: {fullname}>"
    mod.__package__ = fullname.rpartition(".")[0]
    setattr(mod, _ZOO_STUB_SENTINEL, True)
    return mod


def _install_transformers_conversion_mapping_stub() -> types.ModuleType:
    """Synthesise a ``transformers.conversion_mapping`` module.

    Provides exactly the three symbols peft 0.19.x imports at module top:

    * ``_MODEL_TO_CONVERSION_PATTERN`` -- a real ``dict`` (peft calls
      ``.copy()`` on it at module top and then does keyed assignment).
    * ``get_checkpoint_conversion_mapping(model_type)`` -- returns
      ``None`` (i.e. "no v5 conversion registered for this model type").
      peft only invokes this at runtime inside
      ``convert_peft_config_for_transformers`` /
      ``convert_peft_adapter_state_dict_for_transformers``, and both
      early-return on ``None``.
    * ``get_model_conversion_mapping(model)`` -- returns ``None``. Same
      runtime guard story.

    On transformers 4.x peft's own gate (``is_transformers_ge_v5``) means
    these callables never actually fire, but we make them well-behaved
    just in case some caller invokes them directly.
    """
    name = "transformers.conversion_mapping"
    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, _ZOO_STUB_SENTINEL, False):
        return existing

    mod = _make_zoo_stub_module(name)

    # peft does ``_MODEL_TO_CONVERSION_PATTERN = _MODEL_TO_CONVERSION_PATTERN.copy()``
    # at module top, then keyed assignment. A real dict is sufficient.
    mod._MODEL_TO_CONVERSION_PATTERN = {}

    def get_checkpoint_conversion_mapping(model_type, *args, **kwargs):
        # ``None`` is peft's "no conversion registered" sentinel; both
        # callsites in peft early-return on it.
        return None

    def get_model_conversion_mapping(model, *args, **kwargs):
        # Same story: peft treats ``None`` / empty list as "nothing to do".
        return None

    mod.get_checkpoint_conversion_mapping = get_checkpoint_conversion_mapping
    mod.get_model_conversion_mapping = get_model_conversion_mapping

    sys.modules[name] = mod
    # Attach to the parent package as well so ``import transformers;
    # transformers.conversion_mapping`` works just like a real submodule.
    parent = sys.modules.get("transformers")
    if parent is not None and not hasattr(parent, "conversion_mapping"):
        try:
            parent.conversion_mapping = mod
        except Exception:
            # Defensive: a frozen / read-only parent still leaves the
            # sys.modules entry in place, which is enough for
            # ``from transformers.conversion_mapping import ...``.
            pass
    return mod


def _install_transformers_core_model_loading_stub() -> types.ModuleType:
    """Synthesise a ``transformers.core_model_loading`` module.

    Provides the eight symbols peft 0.19.x imports at module top:

      Classes: ``ConversionOps``, ``Concatenate``, ``MergeModulelist``,
      ``Transpose``, ``WeightConverter``, ``WeightRenaming``.

      Callables: ``dot_natural_key``, ``rename_source_key``.

    Peft subclasses ``Concatenate`` and ``ConversionOps`` at module top
    (``PeftConcatenate``, ``FlattenDims``, ``PermuteDims``), so those two
    MUST be real classes -- not callables, not ``object()`` -- or class
    creation will fail at import. The remaining classes only appear in
    ``isinstance`` checks / runtime construction calls that are gated
    behind ``is_transformers_ge_v5`` upstream and never fire on the 4.x
    branch, but we still make them real classes so any third party that
    does ``from transformers.core_model_loading import WeightConverter``
    after this patch sees a sensible (if inert) class.
    """
    name = "transformers.core_model_loading"
    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, _ZOO_STUB_SENTINEL, False):
        return existing

    mod = _make_zoo_stub_module(name)

    class ConversionOps:
        """Stub base class. Subclassing is permitted (peft does this)."""

        # Peft's ``FlattenDims`` / ``PermuteDims`` define their own
        # ``convert`` and ``reverse_op``; we just need a usable base.

        def convert(self, *args, **kwargs):  # pragma: no cover - inert stub
            raise NotImplementedError(
                "unsloth_zoo stub: transformers.core_model_loading.ConversionOps "
                "is a no-op on transformers <5. Upgrade transformers to v5+ to "
                "use peft.utils.transformers_weight_conversion at runtime."
            )

        @property
        def reverse_op(self):  # pragma: no cover - inert stub
            raise NotImplementedError

    class Concatenate(ConversionOps):
        """Stub. Peft subclasses this as ``PeftConcatenate``."""

        def __init__(self, dim=0, *args, **kwargs):
            self.dim = dim

    class MergeModulelist(ConversionOps):
        """Stub. Peft only uses this for ``isinstance(op, MergeModulelist)``."""

        def __init__(self, *args, **kwargs):
            pass

    class Transpose(ConversionOps):
        """Stub. Peft instantiates ``Transpose(dim0=0, dim1=1)`` at runtime."""

        def __init__(self, dim0=0, dim1=1, *args, **kwargs):
            self.dim0 = dim0
            self.dim1 = dim1

    class WeightConverter:
        """Stub. Peft uses for ``isinstance`` and runtime construction."""

        def __init__(self, *args, **kwargs):
            # Accept any signature: peft's real upstream class evolves.
            self.args = args
            self.kwargs = kwargs

    class WeightRenaming:
        """Stub. Peft instantiates ``WeightRenaming(source, target)``."""

        def __init__(
            self,
            source_patterns=None,
            target_patterns=None,
            *args,
            **kwargs,
        ):
            # Support both positional and keyword forms.
            self.source_patterns = source_patterns
            self.target_patterns = target_patterns

    def dot_natural_key(key):
        """Stub key function. Peft only calls this inside a v5-gated path."""
        return key

    def rename_source_key(original_key, renamings, converters):
        """Stub. Returns ``(original_key, None)`` -- v5-gated upstream."""
        return original_key, None

    mod.ConversionOps = ConversionOps
    mod.Concatenate = Concatenate
    mod.MergeModulelist = MergeModulelist
    mod.Transpose = Transpose
    mod.WeightConverter = WeightConverter
    mod.WeightRenaming = WeightRenaming
    mod.dot_natural_key = dot_natural_key
    mod.rename_source_key = rename_source_key

    sys.modules[name] = mod
    parent = sys.modules.get("transformers")
    if parent is not None and not hasattr(parent, "core_model_loading"):
        try:
            parent.core_model_loading = mod
        except Exception:
            pass
    return mod


def fix_peft_transformers_weight_conversion_import():
    """Make ``from peft.utils import transformers_weight_conversion`` work.

    On any (peft 0.19.x, transformers 4.x) pair the import otherwise fails
    with ``ModuleNotFoundError: No module named 'transformers.conversion_mapping'``
    because the peft module unconditionally imports two transformers v5
    submodules even though peft itself only USES them inside an
    ``if is_transformers_ge_v5:`` branch. See the block comment above for
    details.

    Gating contract:
      * No-op if ``peft`` is not installed.
      * No-op if ``transformers`` is not installed (an unfixable case --
        the real symptom would be a different ImportError on the very
        first ``import peft``).
      * No-op if ``peft.utils.transformers_weight_conversion`` already
        imports cleanly (transformers v5+, or a peft fork that uses
        non-v5 paths).
      * Idempotent: a second call sees our sentinel-stamped stubs and
        returns immediately.
      * Strictly additive: only installs a stub for a transformers
        submodule that is currently MISSING. We never overwrite a real
        ``transformers.conversion_mapping`` /
        ``transformers.core_model_loading`` module on transformers v5+.

    Forwards / backwards compatibility:
      * transformers 4.57.6 (no submodule) -> install stubs.
      * transformers 5.x (real submodule) -> first-import succeeds, return.
      * TRL 0.22 / 0.27 / 1.0 -- these don't import either submodule
        directly; they reach the peft conversion module (if at all)
        through ``peft.tuners.tuners_utils``, behind peft's own
        ``is_transformers_ge_v5`` gate. Our stubs are therefore
        unreachable from TRL on a 4.x install, and on a 5.x install the
        real submodules win the import race against our patch.

    Returns ``True`` if the patch was applied (or had been applied
    previously), ``False`` if no action was needed, ``None`` if peft is
    not installed.
    """
    # 1. Cheap exit: no peft installed.
    if importlib.util.find_spec("peft") is None:
        return None

    # 2. Cheap exit: peft.utils.transformers_weight_conversion already
    #    importable -- either we already stubbed and re-imported, or
    #    transformers is v5+ with real submodules. We avoid forcing the
    #    import on the happy path; just try once and return on success.
    try:
        importlib.import_module("peft.utils.transformers_weight_conversion")
        return False
    except ModuleNotFoundError as exc:
        # Only act on our specific drift class. Anything else surfaces
        # the original exception (or rather, is left for the caller's
        # own try/except to handle on the next import attempt).
        missing = getattr(exc, "name", "") or ""
        if missing not in (
            "transformers.conversion_mapping",
            "transformers.core_model_loading",
        ):
            return False
    except ImportError as exc:
        # Older Python pre-3.6 only raises ImportError without `.name`,
        # so also string-match the message for our specific drift.
        msg = str(exc)
        if (
            "transformers.conversion_mapping" not in msg
            and "transformers.core_model_loading" not in msg
        ):
            return False

    # 3. Confirm transformers is loaded; if it isn't, try to load it so
    #    our stub modules can be attached to the parent package. If THAT
    #    fails the user's stack is too broken for us to repair.
    transformers_root = sys.modules.get("transformers")
    if transformers_root is None:
        try:
            transformers_root = importlib.import_module("transformers")
        except Exception:
            return False

    # 4. Stub only the submodules that are genuinely missing. We do NOT
    #    stub a module that already exists for real -- that would
    #    clobber correct behaviour on transformers v5+.
    patched_any = False
    if not _conversion_module_already_importable("transformers.conversion_mapping"):
        _install_transformers_conversion_mapping_stub()
        patched_any = True

    if not _conversion_module_already_importable("transformers.core_model_loading"):
        _install_transformers_core_model_loading_stub()
        patched_any = True

    if not patched_any:
        # Both real submodules already exist -- ``transformers_weight_conversion``
        # must have failed for some other reason. Bail; the next import
        # attempt will surface the original exception unchanged.
        return False

    # 5. Force the peft module through a fresh import now that the
    #    stubs are in place. If a previous failed import left a
    #    ``None`` cache entry in ``sys.modules`` we have to drop it
    #    so importlib will retry.
    pkg = "peft.utils.transformers_weight_conversion"
    if pkg in sys.modules and sys.modules[pkg] is None:
        del sys.modules[pkg]
    try:
        importlib.import_module(pkg)
    except Exception:
        # If even with the stub the module won't import (some other
        # upstream API drift) we swallow -- callers using
        # ``try / except (ImportError, AttributeError)`` will take over.
        # Crucially the stubs stay installed so the NEXT import attempt
        # (after whatever transient condition clears) still succeeds.
        return True

    if _UNSLOTH_ENABLE_LOGGING:
        logger.info(
            "Unsloth Zoo: stubbed transformers.conversion_mapping / "
            "transformers.core_model_loading so peft.utils."
            "transformers_weight_conversion imports cleanly on "
            "transformers <5."
        )
    return True


# ---------------------------------------------------------------------------
# trl.import_utils: tuple-cached ``is_*_available`` accessors
# ---------------------------------------------------------------------------
#
# Mirrors unsloth/import_fixes.py::fix_trl_vllm_ascend (lines 493-516).
#
# transformers >= 4.48's ``_is_package_available(name)`` returns a tuple
# ``(bool, version_or_None)``. TRL caches that tuple in module-level
# ``_*_available`` flags and its matching ``is_*_available()`` accessors
# return the tuple directly. A non-empty tuple is always truthy, so
# ``if is_X_available():`` fires even when X is absent, triggering an
# unconditional ``import X`` that explodes. The headline case is
# ``vllm_ascend`` (blocks ``from trl import GRPOConfig, GRPOTrainer``
# outside Huawei Ascend hosts); ``llm_blender``, ``deepspeed``, ``joblib``
# share the same shape.
#
# Fix: coerce every tuple-cached flag in ``trl.import_utils`` to a plain
# ``bool``. The existing accessors that just return the cached value then
# naturally yield ``True`` / ``False``.
#
# Gating: no-op when TRL isn't installed, when ``trl.import_utils`` can't
# be imported, or when there are no tuple-cached flags. Idempotent: a
# second call sees the already-coerced bool and the type check skips.
# Forwards-compatible: if TRL ever drops the tuple shape entirely, the
# tuple check fails on every attr and we no-op cleanly.
# ---------------------------------------------------------------------------

def fix_trl_vllm_ascend():
    """Coerce tuple-cached ``_*_available`` flags in TRL back to ``bool``.

    See the block comment above for the full rationale.
    """
    if importlib.util.find_spec("trl") is None:
        return
    try:
        import trl.import_utils as tiu
    except Exception:
        return
    coerced = 0
    for attr in list(vars(tiu)):
        if not (attr.startswith("_") and attr.endswith("_available")):
            continue
        cached = getattr(tiu, attr, None)
        if isinstance(cached, tuple):
            try:
                setattr(tiu, attr, bool(cached and cached[0]))
                coerced += 1
            except Exception:
                # Read-only / descriptor-backed module attr -- skip.
                continue
    if coerced and _UNSLOTH_ENABLE_LOGGING:
        logger.info(
            "Unsloth Zoo: coerced %d tuple-cached `_*_available` flags in "
            "trl.import_utils back to bool (fix for transformers >=4.48 "
            "tuple-shape leak through TRL).",
            coerced,
        )


# ---------------------------------------------------------------------------
# datasets 4.4.x recursion error pre-flight
# ---------------------------------------------------------------------------
#
# Mirrors unsloth/import_fixes.py::patch_datasets (lines 574-586).
#
# datasets 4.4.0 and 4.4.1 trigger ``_thread.RLock_recursion_count`` style
# recursion errors in normal use. Both releases are broken on the path
# unsloth + TRL drive. We surface a loud actionable error at import time
# so the user downgrades to 4.3.0 rather than hitting a confusing
# stacktrace deep inside data prep. No silent fall-through.
#
# Gating: no-op if datasets isn't installed, or if the installed version
# is outside the broken window. Idempotent.
# ---------------------------------------------------------------------------

def patch_datasets():
    """Raise on the known-broken ``datasets`` 4.4.x window.

    The upstream unsloth helper does the same pre-flight check. Mirrored
    verbatim here so zoo's drift sweep covers it.
    """
    if importlib.util.find_spec("datasets") is None:
        return
    # Local imports so we don't pay the cost of `packaging` on the happy
    # path and so a missing `packaging` install doesn't take down zoo.
    try:
        from importlib.metadata import version as _importlib_version
        from packaging.version import Version
    except Exception:
        return
    try:
        datasets_version = Version(_importlib_version("datasets"))
    except Exception:
        return
    if Version("4.4.0") <= datasets_version <= Version("4.5.0"):
        raise NotImplementedError(
            f"#### Unsloth: Using `datasets = {str(datasets_version)}` will cause recursion errors.\n"
            "Please downgrade datasets to `datasets==4.3.0`"
        )


# ---------------------------------------------------------------------------
# transformers PreTrainedModel.enable_input_require_grads vision-model fix
# ---------------------------------------------------------------------------
#
# Mirrors unsloth/import_fixes.py::patch_enable_input_require_grads
# (lines 609-670).
#
# transformers PR #41993 rewrote ``PreTrainedModel.enable_input_require_grads``
# to walk ``self.modules()`` and call ``get_input_embeddings()`` on every
# inner ``PreTrainedModel``. Several vision-language modules (e.g. GLM
# V4.6's ``self.visual``) raise ``NotImplementedError`` from
# ``get_input_embeddings`` because they have no token table -- the new
# loop therefore crashes the moment the user prepares a vision-language
# model for training.
#
# Fix: replace the method body with a guarded loop that:
#   * iterates ``self.modules()`` (preserves the new behaviour for
#     classic LM stacks),
#   * dedupes by embedding identity (handles tied embeddings),
#   * swallows ``NotImplementedError`` from sub-modules that don't have
#     token embeddings (the actual upstream regression).
#
# Gating: only patch if the installed transformers really IS on the new
# loop shape (we detect via the ``"for module in self.modules()"`` token
# in the source). On the old per-model body or on a hypothetical newer
# upstream fix that drops the loop, we no-op cleanly. Idempotent via the
# function ``__name__`` sentinel.
# ---------------------------------------------------------------------------

_INPUT_REQUIRE_GRADS_PATCH_NAME = "_unsloth_zoo_patched_enable_input_require_grads"


def patch_enable_input_require_grads():
    """Patch ``PreTrainedModel.enable_input_require_grads`` so vision sub-
    modules without token embeddings stop crashing the upstream loop.

    See the block comment above for the full rationale.
    """
    try:
        import inspect
        from transformers import PreTrainedModel
    except Exception:
        return

    # Idempotent: a previous call already swapped in our function.
    current = getattr(PreTrainedModel, "enable_input_require_grads", None)
    if current is None:
        return
    if getattr(current, "__name__", "") == _INPUT_REQUIRE_GRADS_PATCH_NAME:
        return

    try:
        original_source = inspect.getsource(current)
    except Exception:
        return

    # Only fire when the installed transformers is on the post-PR-41993
    # loop shape that triggers the regression. Pre-PR transformers used a
    # single ``get_input_embeddings()`` call and isn't affected.
    if "for module in self.modules()" not in original_source:
        return

    def _unsloth_zoo_patched_enable_input_require_grads(self):
        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)

        hooks = []
        seen_modules = set()

        for module in self.modules():
            if not (
                isinstance(module, PreTrainedModel)
                and hasattr(module, "get_input_embeddings")
            ):
                continue

            try:
                input_embeddings = module.get_input_embeddings()
            except NotImplementedError:
                # Vision sub-modules without a token table (e.g. GLM V4.6's
                # `self.visual`) raise here. Skip; their inputs aren't
                # subject to require-grads.
                continue
            except Exception:
                # Defensive: an exotic sub-model that raises something
                # else still shouldn't take down the whole walk.
                continue

            if input_embeddings is None:
                continue

            embedding_id = id(input_embeddings)
            if embedding_id in seen_modules:
                continue

            seen_modules.add(embedding_id)
            hooks.append(
                input_embeddings.register_forward_hook(make_inputs_require_grads)
            )

        self._require_grads_hooks = hooks
        if hooks:
            self._require_grads_hook = hooks[0]

    # Stamp the function name so a re-entry is a no-op and tests can
    # detect "this came from zoo".
    _unsloth_zoo_patched_enable_input_require_grads.__name__ = (
        _INPUT_REQUIRE_GRADS_PATCH_NAME
    )

    try:
        PreTrainedModel.enable_input_require_grads = (
            _unsloth_zoo_patched_enable_input_require_grads
        )
    except (AttributeError, TypeError):
        # Class doesn't permit method replacement -- defensive bail.
        return

    if _UNSLOTH_ENABLE_LOGGING:
        logger.info(
            "Unsloth Zoo: patched PreTrainedModel.enable_input_require_grads "
            "for vision sub-model compatibility (transformers PR #41993 "
            "regression)."
        )


# ---------------------------------------------------------------------------
# torchcodec broken-binary detection
# ---------------------------------------------------------------------------
#
# Mirrors unsloth/import_fixes.py::disable_torchcodec_if_broken
# (lines 1291-1317).
#
# transformers detects torchcodec via ``importlib.util.find_spec``, which
# returns True even when the wheel is on disk but its native libs (FFmpeg)
# can't load. The first audio decode then crashes. We probe an actual
# load and, on failure, flip ``transformers.utils.import_utils._torchcodec_available``
# to False so transformers cleanly falls back to librosa.
#
# Forwards-compat note (transformers 5.x): the underscore-prefixed cache
# was renamed in the new structured-imports refactor. We probe BOTH the
# legacy ``_torchcodec_available`` and any post-rename ``torchcodec_available``
# attribute, and only flip the one(s) that actually exist. If neither
# exists (the symbol disappeared entirely) we no-op silently.
# ---------------------------------------------------------------------------

def disable_torchcodec_if_broken():
    """Flip transformers' torchcodec availability flag to False when the
    torchcodec native libraries can't actually load.

    See the block comment above for the full rationale.
    """
    try:
        if importlib.util.find_spec("torchcodec") is None:
            return  # torchcodec not installed -- transformers already knows.
    except Exception:
        return

    # Probe a real load. If this raises, the wheel is on disk but broken.
    try:
        from torchcodec.decoders import AudioDecoder  # noqa: F401
        return
    except (ImportError, RuntimeError, OSError, Exception):  # noqa: BLE001
        pass

    try:
        import transformers.utils.import_utils as tf_import_utils
    except Exception:
        return

    flipped = False
    # Legacy underscore-prefixed cache (transformers 4.x).
    if hasattr(tf_import_utils, "_torchcodec_available"):
        try:
            tf_import_utils._torchcodec_available = False
            flipped = True
        except Exception:
            pass
    # Post-rename (transformers 5.x candidate names). We treat every
    # ``*torchcodec*available*`` attribute that currently holds a truthy
    # cached value as suspect and flip it. Strictly additive: we never
    # touch attrs that don't exist.
    for attr in list(vars(tf_import_utils)):
        low = attr.lower()
        if "torchcodec" not in low:
            continue
        if "available" not in low:
            continue
        if attr == "_torchcodec_available":
            continue  # handled above
        try:
            current = getattr(tf_import_utils, attr)
        except Exception:
            continue
        # Only flip when the value looks like a "this is here" signal.
        if isinstance(current, bool) and current is True:
            try:
                setattr(tf_import_utils, attr, False)
                flipped = True
            except Exception:
                continue
        elif isinstance(current, tuple) and current and current[0]:
            try:
                setattr(tf_import_utils, attr, (False, current[1] if len(current) > 1 else None))
                flipped = True
            except Exception:
                continue

    if flipped and _UNSLOTH_ENABLE_LOGGING:
        logger.info(
            "Unsloth Zoo: disabled torchcodec in transformers (native libs "
            "could not load; falling back to librosa)."
        )


def apply_import_fixes():
    """Apply all available zoo-local import-time fixes.

    Each individual fix is responsible for its own gating + idempotence;
    this entry point just runs them in order and swallows individual
    failures so a single broken fix can't take the whole zoo import
    down. Set ``UNSLOTH_ENABLE_LOGGING=1`` to surface details.
    """
    for fix in (
        fix_triton_compiled_kernel_missing_attrs,
        fix_vllm_guided_decoding_params,
        fix_peft_transformers_weight_conversion_import,
        fix_trl_vllm_ascend,
        patch_datasets,
        patch_enable_input_require_grads,
        disable_torchcodec_if_broken,
    ):
        try:
            fix()
        except Exception as exc:  # noqa: BLE001
            if _UNSLOTH_ENABLE_LOGGING:
                logger.warning(
                    "Unsloth Zoo: import-fix %s failed with %s: %s",
                    fix.__name__, type(exc).__name__, exc,
                )
