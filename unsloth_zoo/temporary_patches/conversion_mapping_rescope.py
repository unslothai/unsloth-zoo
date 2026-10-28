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
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Re-scopes the leaked submodule renamings that transformers 5.4.0 (PR 44300) merges into a composite model, dropping
bnb quant_state sidecars; applies to 5.4.0 and 5.5.0 through 5.5.4 (upstream fixed it in 5.6.0, PR 45567); neither
gate is a version check."""

__all__ = [
    "patch_transformers_composite_prefix_renaming",
]

import functools
import sys

from .common import (
    TEMPORARY_PATCHES,
    RESCOPE_PATCH_FLAG,
    UNSLOTH_ENABLE_LOGGING,
    WRAPPER_INNER_ATTR,
    logger,
)
from .utils import raise_error

# unsloth's own wrapper carries this. Either mark means the repair is already live.
_UNSLOTH_PATCH_FLAG = "_unsloth_patched_composite_prefix_renaming"

# A prefix renaming matches all of a submodule's names or none, so one sample would do; eight also covers a mapping that rewrites only some leaf names.
_RENAMING_SAMPLE = 8

# Bounded: a chain can end early, and a malformed one must not spin.
_MAX_WRAPPER_DEPTH = 8


def _transformers_rescopes_submodule_prefix_renamings():
    """Asks the API, not a version: True (leave transformers alone) when any of the model_prefix argument,
    PrefixChange.with_submodel_prefix or scope_prefix (5.10+) exists, or the build is unrecognised or has no
    conversion_mapping."""
    try:
        import inspect
        from transformers import conversion_mapping
        from transformers import core_model_loading
    except Exception:
        return True
    extract = getattr(conversion_mapping, "extract_weight_conversions_for_model", None)
    if extract is None:
        # No per-submodule extraction means no recursion to correct; also the answer for any future build whose machinery is unrecognised.
        return True
    transform = getattr(core_model_loading, "WeightTransform", None)
    if transform is not None and hasattr(transform, "scope_prefix"):
        return True
    prefix_change = getattr(core_model_loading, "PrefixChange", None)
    if prefix_change is not None and hasattr(prefix_change, "with_submodel_prefix"):
        return True
    try:
        return "model_prefix" in inspect.signature(extract).parameters
    except Exception:
        return True


def _patterns(conversion):
    """Unprocessed patterns where the release keeps them: 5.5.x has _original_source_patterns and
    _original_target_patterns because __post_init__ rewrites the live ones; 5.4.0 lacks them, so use live patterns."""
    source = getattr(conversion, "_original_source_patterns", None) or conversion.source_patterns
    target = getattr(conversion, "_original_target_patterns", None) or conversion.target_patterns
    return list(source), list(target)


def _renaming_signature(conversion):
    """Compares by value: get_checkpoint_conversion_mapping returns deep copies, so a submodule's renaming is never the same object as the parent's list entry."""
    source, target = _patterns(conversion)
    return (type(conversion).__name__, tuple(source), tuple(target))


def _sample_submodule_keys(submodule, prefix):
    """A few of the submodule's real parameter names, spelled as the PARENT spells them."""
    keys = []
    try:
        for name, _ in submodule.named_parameters(recurse = True):
            keys.append(f"{prefix}.{name}")
            if len(keys) >= _RENAMING_SAMPLE:
                return keys
        for name, _ in submodule.named_buffers(recurse = True):
            keys.append(f"{prefix}.{name}")
            if len(keys) >= _RENAMING_SAMPLE:
                break
    except Exception:
        return []
    return keys


def _renaming_destroys_keys(conversion, sample_keys, model_keys):
    """Does this renaming move a real model key onto a name the model lacks? The same entry is inert on its own text
    model but wrong merged into the composite; one rename onto a real name answers False."""
    destroys = False
    for key in sample_keys:
        try:
            renamed, matched = conversion.rename_source_key(key)
        except Exception:
            return False
        if matched is None or renamed == key:
            continue
        if renamed in model_keys:
            return False
        destroys = True
    return destroys


def _prefixed_pattern(pattern, prefix):
    """Push a pattern down into `prefix`, keeping a start anchor anchored."""
    if pattern.startswith("^"):
        return f"^{prefix}.{pattern[1:]}"
    return f"{prefix}.{pattern}"


def _rescoped_renaming(conversion, prefix, sample_keys, model_keys):
    """Renaming re-anchored under its submodule prefix, or None: only ^-anchored patterns can be scoped, and it is returned only if inert on the model's own names."""
    from transformers.core_model_loading import WeightRenaming

    source, target = _patterns(conversion)
    if not all(pattern.startswith("^") for pattern in source):
        return None

    patterns = {
        "source_patterns": [_prefixed_pattern(p, prefix) for p in source],
        "target_patterns": [_prefixed_pattern(p, prefix) for p in target],
    }
    try:
        rescoped = type(conversion)(**patterns)
    except Exception:
        # PrefixChange takes prefixes, not patterns; fall back to the base class WeightRenaming, since a renaming is all this produces, rather than giving up.
        try:
            rescoped = WeightRenaming(**patterns)
        except Exception:
            return None
    for key in sample_keys:
        try:
            renamed, matched = rescoped.rename_source_key(key)
        except Exception:
            return None
        if matched is not None and renamed != key and renamed not in model_keys:
            return None
    return rescoped


def _leaked_submodule_prefix_renamings(model):
    """Renamings the submodule recursion merged into the model's mapping where they do not belong; walks submodules in
    the same order with first model type winning; ignores renamings the parent registers for itself; empty if nothing
    leaked."""
    from transformers.conversion_mapping import extract_weight_conversions_for_model
    from transformers.core_model_loading import WeightRenaming
    from transformers.modeling_utils import PreTrainedModel

    def extract(module, prefix):
        # Some releases also take the submodule's dotted path; try that signature and fall back on TypeError, since the check is on the function, not a version.
        try:
            return extract_weight_conversions_for_model(module, prefix)
        except TypeError:
            return extract_weight_conversions_for_model(module)

    model_keys = set()
    for name, _ in model.named_parameters(remove_duplicate = False):
        model_keys.add(name)
    for name, _ in model.named_buffers(remove_duplicate = False):
        model_keys.add(name)

    own = set()
    seen_model_types = set()
    own_conversions = extract(model, "")
    if own_conversions is not None:
        seen_model_types.add(getattr(model.config, "model_type", None))
        own.update(_renaming_signature(c) for c in own_conversions)

    leaked = {}
    for name, submodule in model.named_modules():
        if submodule is model or not name or not isinstance(submodule, PreTrainedModel):
            continue
        model_type = getattr(getattr(submodule, "config", None), "model_type", None)
        if model_type is None or model_type in seen_model_types:
            continue
        conversions = extract(submodule, name)
        if conversions is None:
            continue
        seen_model_types.add(model_type)
        sample_keys = _sample_submodule_keys(submodule, name)
        if not sample_keys:
            continue
        for conversion in conversions:
            if not isinstance(conversion, WeightRenaming):
                continue
            signature = _renaming_signature(conversion)
            if signature in own or signature in leaked:
                continue
            if _renaming_destroys_keys(conversion, sample_keys, model_keys):
                leaked[signature] = (name, sample_keys)
    return leaked, model_keys


def _rescope_conversions(model, conversions):
    """Replace every leaked renaming in `conversions` with its scoped form, or drop it."""
    from transformers.core_model_loading import WeightRenaming

    leaked, model_keys = _leaked_submodule_prefix_renamings(model)
    if not leaked:
        # The caller's own list object, so a non-composite model is measurably untouched.
        return conversions

    rescoped_conversions = []
    fixed = 0
    dropped = 0
    for conversion in conversions:
        scoped_by_transformers = isinstance(conversion, WeightRenaming) and (
            getattr(conversion, "scope_prefix", None) is not None
        )
        entry = (
            leaked.get(_renaming_signature(conversion))
            if isinstance(conversion, WeightRenaming) and not scoped_by_transformers
            else None
        )
        if entry is None:
            rescoped_conversions.append(conversion)
            continue
        prefix, sample_keys = entry
        replacement = _rescoped_renaming(conversion, prefix, sample_keys, model_keys)
        if replacement is None:
            dropped += 1
            continue
        rescoped_conversions.append(replacement)
        fixed += 1

    if not fixed and not dropped:
        # Nothing replaced: return the caller's own list, not a copy, so 'this patch changed nothing' stays checkable by
        # identity if the install gate is ever bypassed.
        return conversions

    if UNSLOTH_ENABLE_LOGGING:
        logger.info(
            f"Unsloth: re-scoped {fixed} and dropped {dropped} checkpoint renaming(s) that "
            f"transformers merged into {type(model).__name__}'s conversion mapping from a "
            f"submodule, where they rewrite the model's own weight names into names it does "
            f"not have (transformers PR #45567, released in 5.6.0)"
        )
    return rescoped_conversions


def _wrapper_chain(function):
    """`function` and everything it wraps, by either link. Bounded, so a cycle terminates."""
    chain = []
    seen = 0
    while function is not None and seen < _MAX_WRAPPER_DEPTH:
        if any(function is already for already in chain):
            break
        chain.append(function)
        function = (
            getattr(function, "__wrapped__", None)
            or getattr(function, WRAPPER_INNER_ATTR, None)
        )
        seen += 1
    return chain


def _repair_already_installed(function):
    """Is either package's repair on this callable or down its wrapper chain? Follows WRAPPER_INNER_ATTR as well as
    __wrapped__, since a wrapper that must survive rescope cannot publish __wrapped__ (moe_utils_bnb4bit.py)."""
    seen = 0
    while function is not None and seen < _MAX_WRAPPER_DEPTH:
        if getattr(function, RESCOPE_PATCH_FLAG, False):
            return True
        if getattr(function, _UNSLOTH_PATCH_FLAG, False):
            return True
        function = getattr(function, "__wrapped__", None) or getattr(function, WRAPPER_INNER_ATTR, None)
        seen += 1
    return False


def patch_transformers_composite_prefix_renaming():
    if _transformers_rescopes_submodule_prefix_renamings():
        return
    try:
        from transformers import conversion_mapping
    except Exception as e:
        return raise_error("transformers.conversion_mapping", e)

    original = getattr(conversion_mapping, "get_model_conversion_mapping", None)
    if original is None:
        return
    # Both marks, whole chain: unsloth may have installed its copy first, under
    # `moe_utils_bnb4bit.py`'s wrapper. A second repair is inert but is a wrapper nobody needs.
    if _repair_already_installed(original):
        return
    # Wrap WHATEVER IS LIVE, never what it wraps. This used to unwrap `__wrapped__` first, to
    # avoid re-wrapping a wrapper of ours that had lost its mark -- but the check above already
    # declines when either mark is anywhere in the chain, so that case cannot reach here, while
    # unwrapping a THIRD PARTY's `functools.wraps` wrapper and then replacing the module
    # attribute dropped their behaviour entirely. Measured on transformers 5.5.4: their wrapper
    # was absent from the resulting chain.
    # Every callable our wrapper now sits in front of, so the sweep below can tell a binding it
    # SUPERSEDES from one it would silently destroy.
    superseded = _wrapper_chain(original)

    @functools.wraps(original)
    def get_model_conversion_mapping(*args, **kwargs):
        conversions = original(*args, **kwargs)
        try:
            model = kwargs["model"] if "model" in kwargs else (args[0] if args else None)
            if model is None or not conversions:
                return conversions
            return _rescope_conversions(model, conversions)
        except Exception as e:
            # A mapping we could not reason about is still the mapping transformers built.
            if UNSLOTH_ENABLE_LOGGING:
                logger.info(f"Unsloth: Could not re-scope the conversion mapping ({e})")
            return conversions

    # Explicit, not just via functools.wraps: a wraps-less edit must not make the patch
    # un-probeable and un-undoable.
    get_model_conversion_mapping.__wrapped__ = original
    setattr(get_model_conversion_mapping, RESCOPE_PATCH_FLAG, True)

    try:
        conversion_mapping.get_model_conversion_mapping = get_model_conversion_mapping
        # A module that already did `from .conversion_mapping import ...` holds the OBJECT,
        # not the attribute (modeling_utils, integrations.peft, peft itself). Restricted to
        # packages importing this name from transformers, because the test below cannot be an
        # identity one and an unrestricted sweep would replace ANY callable so named.
        owning_packages = ("transformers", "peft", "unsloth_zoo", "unsloth")
        for module_name, module in list(sys.modules.items()):
            if module is None or module is conversion_mapping:
                continue
            root = module_name.partition(".")[0]
            if root not in owning_packages:
                continue
            namespace = getattr(module, "__dict__", None)
            if not isinstance(namespace, dict):
                continue
            # `__dict__`, never `getattr`: transformers' lazy modules answer any name through
            # `__getattr__`, so a getattr sweep walks the whole model zoo printing deprecations.
            try:
                bound = namespace.get("get_model_conversion_mapping", None)
                if not callable(bound) or bound is get_model_conversion_mapping:
                    continue
                # Only a binding we supersede. Anything else is SOMEONE ELSE'S wrapper on the
                # same name, and overwriting it drops their behaviour with no trace: our
                # wrapper closes over the module attribute, not over whatever an alias held,
                # so their work would simply not be in the chain. Leave it, even though it
                # then misses this repair -- a partly-unrepaired load beats silently
                # disabling unrelated conversion logic.
                if not any(bound is known for known in superseded):
                    if UNSLOTH_ENABLE_LOGGING:
                        logger.info(
                            f"Unsloth: leaving {module_name}.get_model_conversion_mapping "
                            f"alone, it holds another wrapper we do not supersede"
                        )
                    continue
                module.get_model_conversion_mapping = get_model_conversion_mapping
            except Exception:
                continue
        if UNSLOTH_ENABLE_LOGGING:
            logger.info(
                "Unsloth: Patching transformers `get_model_conversion_mapping` so a "
                "pre-quantized multimodal checkpoint keeps its bitsandbytes quant_state "
                "(transformers PR #45567)"
            )
    except Exception as e:
        return raise_error("transformers.get_model_conversion_mapping", e)
pass
TEMPORARY_PATCHES.append(patch_transformers_composite_prefix_renaming)
