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

"""Keep a pre-quantized multimodal checkpoint's bitsandbytes quant_state on transformers 5.4/5.5.

transformers 5.4.0 (PR #44300) made `get_model_conversion_mapping` recurse into
`PreTrainedModel` submodules and merge each submodule's registered conversions into the
parent's mapping. A submodule's mapping is written against ITS OWN key space, so a renaming
anchored at the start of the key is meaningless once the submodule is nested. The standalone
`qwen3_5_text` / `qwen3_5_moe_text` / `gemma3n_text` models register
`^model.language_model.` -> `^model.` to strip a prefix their own checkpoints carry, and the
recursion hands that renaming to the COMPOSITE model, whose weights really ARE named
`model.language_model....`.

Renamings all run before any `WeightConverter` (`core_model_loading.py` splits the two lists
and chains every renaming first), so every checkpoint key loses the `language_model.` segment.
The packed `weight` is rescued by original key with no converter attached and loads as a bare
uint8 parameter, while `weight.absmax`, `weight.quant_map`, `weight.nested_absmax`,
`weight.nested_quant_map` and `weight.quant_state.bitsandbytes__nf4` match nothing and are
discarded as unexpected. `Bnb4bitDeserialize.convert` then takes its `len(input_dict) == 1`
early return and hands back the raw packed tensor WITHOUT RAISING, which is why the failure
only surfaces at the first forward, as a shape error about a blob.

Upstream fixed it in 5.6.0 (PR #45567) by scoping each transform non-destructively instead of
rewriting its patterns. This patch re-scopes the leaked renaming the same way for the releases
that cannot, which is 5.4.0 and 5.5.0 through 5.5.4.

It lives here, and not only in `unsloth/import_fixes.py`, because this package is the one that
owns the bitsandbytes `Linear4bit` patch that reports the failure, is importable without
`unsloth`, and caps transformers at 5.5.0 on Apple Silicon in its own `pyproject.toml` -- which
puts every Mac install inside the defect window by construction. `unsloth`'s copy defers to
this one when it is present, so only ever one of the two installs.

Neither gate is a version number. The install gate asks whether this transformers has any of
the shapes upstream has used for per-submodule scoping; the call gate only touches a renaming
that demonstrably rewrites this model's real parameter names into names it does not have.
"""

__all__ = [
    "patch_transformers_composite_prefix_renaming",
]

import functools
import sys

from .common import (
    TEMPORARY_PATCHES,
    RESCOPE_PATCH_FLAG,
    UNSLOTH_ENABLE_LOGGING,
    logger,
)
from .utils import raise_error

# unsloth's own wrapper carries this. Either mark means the repair is already live.
_UNSLOTH_PATCH_FLAG = "_unsloth_patched_composite_prefix_renaming"

# How many of a submodule's own parameter names to try a renaming against. A prefix renaming
# either matches every name under the submodule or none of them, so one would do; eight costs
# nothing and covers a mapping that only rewrites some leaf names.
_RENAMING_SAMPLE = 8

# `__wrapped__` chains are walked rather than followed blindly: `moe_utils_bnb4bit.py` wraps the
# same function without setting `__wrapped__` at all, so a chain can end early, and a malformed
# one must not spin.
_MAX_WRAPPER_DEPTH = 8


def _transformers_rescopes_submodule_prefix_renamings():
    """Does this transformers scope a submodule's conversion mapping to where the submodule lives?

    Asked of the API, never of a version number. A transformers without `conversion_mapping` at
    all (4.x) has no recursion to correct, so it answers True as well -- True means "leave
    transformers alone".

    Upstream has spelled the scoping three ways, so all three count: the `model_prefix` argument
    of `extract_weight_conversions_for_model`, `PrefixChange.with_submodel_prefix`, and the
    `scope_prefix` field every transform carries from 5.10 on. Any one of them present means the
    recursion knows where a submodule's mapping belongs; none of them present is the defect.
    Anything unrecognisable answers True, because guessing wrong in that direction only leaves
    transformers as it was.
    """
    try:
        import inspect
        from transformers import conversion_mapping
        from transformers import core_model_loading
    except Exception:
        return True
    extract = getattr(conversion_mapping, "extract_weight_conversions_for_model", None)
    if extract is None:
        # No per-submodule extraction, so no recursion to correct. Every 5.x before 5.4.0 reads
        # the top model's mapping and stops, and this is also the answer for any future build
        # whose machinery we cannot recognise.
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
    """A conversion's source and target patterns, unprocessed where the release keeps them.

    5.5.x stores `_original_source_patterns` because `__post_init__` rewrites the live ones;
    5.4.0 has no such field at all, so fall through to the live patterns rather than assuming
    an attribute that release never had.
    """
    source = getattr(conversion, "_original_source_patterns", None) or conversion.source_patterns
    target = getattr(conversion, "_original_target_patterns", None) or conversion.target_patterns
    return list(source), list(target)


def _renaming_signature(conversion):
    """Identity of a renaming by VALUE, because the mapping is handed out as deep copies.

    `get_checkpoint_conversion_mapping` returns `deepcopy(...)`, so the object collected from a
    submodule here is never the object that ended up in the parent's list, and `is` can never
    match them. The patterns are what a renaming is, so compare those.
    """
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
    """Does this renaming rewrite the model's OWN parameter names into names it does not have?

    This is the whole discriminator, and it is a behaviour, not a name. A renaming exists to map
    CHECKPOINT keys onto MODEL keys, so one that fires on a key the model really has and
    produces one it does not is wrong wherever it came from.

    It also separates the two directions the same entry serves. The standalone text model
    registers `^model.language_model.` -> `^model.` because its checkpoint carries the longer
    keys and the model carries the shorter ones -- fired against that model's own names it
    matches nothing, and answers False here. Merged into the composite model, whose names ARE
    the longer ones, it matches all of them and lands off the map. One rule, both cases.

    A single rename onto a name the model really has is enough to answer False: that is a
    mapping doing its job, and no amount of other evidence should override it.
    """
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
    """The same renaming, scoped to the submodule it came from, or None if that cannot be built.

    This is what upstream's per-submodule scoping produces, spelled for a transformers that has
    none: `^model.language_model.` -> `^model.` collected from the submodule at
    `model.language_model` becomes `^model.language_model.model.language_model.` ->
    `model.language_model.model.`, which can only fire on a genuinely doubled prefix and
    therefore leaves every real key alone.

    Only a renaming anchored at the start of the key can be scoped this way, which is the same
    restriction upstream has. An unanchored pattern says "wherever this appears", and pushing a
    prefix in front of it produces a pattern that means neither thing, so this answers None and
    lets the caller drop the entry instead of pretending to have placed it. Nothing legitimate
    is lost -- the caller only reaches here for an entry already shown to rewrite real weight
    names off the map.

    Returned only after checking that it really is inert on this model's own names. A renaming
    that still rewrites real keys into nonexistent ones is worse than no renaming at all.
    """
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
        # A subclass whose __init__ takes something else entirely -- upstream's `PrefixChange`
        # takes prefixes, not patterns. It is still a WeightRenaming, and a renaming is all this
        # produces, so fall back to the base class rather than giving up.
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
    """Which renamings did the recursion merge into `model`'s mapping where they do not belong?

    Walks the submodules the way `get_model_conversion_mapping` walks them -- same order, same
    "first model type wins" rule -- so the entries found here are the entries it collected. A
    renaming that the parent registers for ITSELF is never considered, even if a submodule
    registers the same one.

    Returns `(signature -> (prefix, sample_keys), model_keys)`, empty when nothing leaked.
    """
    from transformers.conversion_mapping import extract_weight_conversions_for_model
    from transformers.core_model_loading import WeightRenaming
    from transformers.modeling_utils import PreTrainedModel

    def extract(module, prefix):
        # Some releases take the submodule's dotted path as a second argument, everything else
        # takes the module alone. Asked of the function rather than of a version.
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
        # Nothing was actually replaced. A leaked SIGNATURE can still match an entry that this
        # transformers already scoped for itself -- which is what a fixed release looks like if
        # the install gate is ever bypassed -- and returning the caller's own list rather than a
        # copy of it is what makes "this patch changed nothing" checkable by identity.
        return conversions

    if UNSLOTH_ENABLE_LOGGING:
        logger.info(
            f"Unsloth: re-scoped {fixed} and dropped {dropped} checkpoint renaming(s) that "
            f"transformers merged into {type(model).__name__}'s conversion mapping from a "
            f"submodule, where they rewrite the model's own weight names into names it does "
            f"not have (transformers PR #45567, released in 5.6.0)"
        )
    return rescoped_conversions


def _repair_already_installed(function):
    """Is either package's repair already on this callable, anywhere down the wrapper chain?"""
    seen = 0
    while function is not None and seen < _MAX_WRAPPER_DEPTH:
        if getattr(function, RESCOPE_PATCH_FLAG, False):
            return True
        if getattr(function, _UNSLOTH_PATCH_FLAG, False):
            return True
        function = getattr(function, "__wrapped__", None)
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
    # Both marks, and the whole chain: `unsloth` may have installed its copy first, and
    # `moe_utils_bnb4bit.py` may have put its own unmarked wrapper on top of that. Installing a
    # second repair is measurably inert -- the first pass leaves no signature for the second to
    # match -- but it is still a wrapper nobody needs.
    if _repair_already_installed(original):
        return
    # Probe and wrap the ORIGINAL, never a wrapper of ours that lost its mark.
    original = getattr(original, "__wrapped__", original)

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

    # functools.wraps sets __wrapped__, but set it explicitly: the probe and the tests both read
    # it, and a wraps-less edit must not make the patch un-probeable and un-undoable.
    get_model_conversion_mapping.__wrapped__ = original
    setattr(get_model_conversion_mapping, RESCOPE_PATCH_FLAG, True)

    try:
        conversion_mapping.get_model_conversion_mapping = get_model_conversion_mapping
        # Every module that already did `from .conversion_mapping import
        # get_model_conversion_mapping` holds the OBJECT, not the attribute:
        # transformers.modeling_utils and transformers.integrations.peft do, and so does peft
        # itself. Modules that import it later pick the patched one up from the module above.
        #
        # Restricted to the packages that import this name from transformers, because the test
        # below cannot be an identity test: `moe_utils_bnb4bit.py` wraps the same function
        # without `functools.wraps` and without `__wrapped__`, so when it goes first a module
        # holding the pre-zoo function matches neither object. Without the restriction the sweep
        # would replace ANY callable of that name, including one a notebook defined for itself.
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
            # `__dict__`, never `getattr`: transformers' lazy modules answer ANY attribute name
            # through `__getattr__`, which imports a submodule, prints a deprecation notice and
            # hands back an alias. Sweeping with getattr walks the whole model zoo and says
            # "Accessing get_model_conversion_mapping from ..." several hundred times.
            try:
                bound = namespace.get("get_model_conversion_mapping", None)
                if callable(bound) and bound is not get_model_conversion_mapping:
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
