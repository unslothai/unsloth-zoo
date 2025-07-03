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

import inspect
import typing as t
from typing import Any, Callable, Dict, List, Tuple
import logging
from .common import UNSLOTH_ENABLE_LOGGING

logger = logging.getLogger(__name__)

EMPTY = inspect._empty


def canonicalize_annotation(annotation: Any) -> Any:
    """
    Canonicalize type annotations for consistent comparison.
    Makes List[int], typing.List[int], list[int] equivalent.
    """
    if annotation is EMPTY:
        return EMPTY

    if hasattr(t, "get_origin"):
        origin = t.get_origin(annotation)
        if origin is not None:
            args = tuple(canonicalize_annotation(arg) for arg in t.get_args(annotation))
            return (origin, args)

    # Normalize common built-in types to their typing equivalents
    type_mappings = {
        list: t.List,
        dict: t.Dict,
        set: t.Set,
        tuple: t.Tuple,
    }

    return type_mappings.get(annotation, annotation)


def get_function_fingerprint(func: Callable) -> List[Dict[str, Any]]:
    """
    Return a fingerprint we can use to compare function signatures.
    Returns: [{'name': str, 'kind': int, 'is_required': bool, 'annotation': Any}]
    """
    try:
        signature = inspect.signature(func)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Cannot inspect function signature: {e}")

    fingerprint = []
    for param in signature.parameters.values():
        fingerprint.append({
            'name': param.name,
            'kind': param.kind.value,  # Use .value for cleaner comparison
            'is_required': param.default is EMPTY,  # True = required
            'annotation': canonicalize_annotation(param.annotation),
        })

    return fingerprint


def can_safely_patch(original_func: Callable, new_func: Callable, 
                    match_level: str = "strict") -> Tuple[bool, str]:
    """
    Check if it's safe to patch original_func with new_func.
    """
    if match_level not in ("strict", "relaxed"):
        return False, f"Invalid match_level: {match_level}. Use 'strict' or 'relaxed'"

    try:
        old_fp = get_function_fingerprint(original_func)
        new_fp = get_function_fingerprint(new_func)
    except ValueError as e:
        return False, f"Signature inspection failed: {e}"

    if len(old_fp) != len(new_fp):
        return False, f"Parameter count mismatch: {len(old_fp)} vs {len(new_fp)}"

    for old_param, new_param in zip(old_fp, new_fp):
        if (old_param['name'], old_param['kind']) != (new_param['name'], new_param['kind']):
            return False, f"Parameter '{old_param['name']}' signature changed"

        if new_param['is_required'] and not old_param['is_required']:
            return False, f"Parameter '{new_param['name']}' changed from optional to required"

        # For strict matching, also check type annotations
        if match_level == "strict" and old_param['annotation'] != new_param['annotation']:
            return False, f"Parameter '{old_param['name']}' type annotation changed: {old_param['annotation']} -> {new_param['annotation']}"

    return True, ""


def _get_unique_storage_name(target_obj: Any, attr_name: str) -> str:
    """
    Generate a unique name for storing the original function.
    """
    if hasattr(target_obj, '__name__'):
        obj_name = target_obj.__name__
    elif hasattr(target_obj, '__class__'):
        obj_name = target_obj.__class__.__name__
    else:
        obj_name = str(type(target_obj).__name__)

    # Include module if available for extra uniqueness
    if hasattr(target_obj, '__module__'):
        module_name = target_obj.__module__.split('.')[-1]  # Just the last part
        return f"_original_{module_name}_{obj_name}_{attr_name}"
    else:
        return f"_original_{obj_name}_{attr_name}"


def patch_function(target_obj: Any, attr_name: str, new_func: Callable, 
                  force: bool = False, store_original: bool = True, 
                  match_level: str = "strict") -> bool:
    """
    Patch a function/method on an object.
    """
    if not hasattr(target_obj, attr_name):
        if UNSLOTH_ENABLE_LOGGING:
            logger.error(f"Unsloth: Attribute '{attr_name}' not found on {target_obj}")
        return False

    original_func = getattr(target_obj, attr_name)

    # Store original for potential restoration with unique name
    if store_original:
        unique_name = _get_unique_storage_name(target_obj, attr_name)
        setattr(target_obj, unique_name, original_func)
        if UNSLOTH_ENABLE_LOGGING:
            logger.info(f"Unsloth: Stored original as {unique_name}")

    if not force:
        is_safe, reason = can_safely_patch(original_func, new_func, match_level)
        if not is_safe:
            if UNSLOTH_ENABLE_LOGGING:
                logger.error(f"Unsloth: Patch of {attr_name} skipped: {reason}")
            return False

    try:
        setattr(target_obj, attr_name, new_func)
        return True
    except Exception as e:
        if UNSLOTH_ENABLE_LOGGING:
            logger.error(f"Unsloth: Failed to patch {attr_name}: {e}")
        return False


def patch_multiple(patches: List[Tuple[Any, str, Callable]], 
                  force: bool = False, 
                  fail_fast: bool = True,
                  match_level: str = "strict") -> Dict[str, bool]:
    """
    Apply multiple patches at once.
    """
    results = {}

    for target_obj, attr_name, new_func in patches:
        key = f"{getattr(target_obj, '__name__', str(target_obj))}.{attr_name}"
        success = patch_function(target_obj, attr_name, new_func, force=force, match_level=match_level)
        results[key] = success

        if fail_fast and not success:
            if UNSLOTH_ENABLE_LOGGING:
                logger.error(f"Unsloth: Stopping patch process due to failure on {key}")
            break

    return results


def restore_original(target_obj: Any, attr_name: str) -> bool:
    """
    Restore original function if it was stored.
    """
    unique_name = _get_unique_storage_name(target_obj, attr_name)

    if not hasattr(target_obj, unique_name):
        if UNSLOTH_ENABLE_LOGGING:
            logger.error(f"Unsloth: No stored original found for {attr_name} (looked for {unique_name})")
        return False

    try:
        original_func = getattr(target_obj, unique_name)
        setattr(target_obj, attr_name, original_func)
        delattr(target_obj, unique_name)  # Clean up
        if UNSLOTH_ENABLE_LOGGING:
            logger.info(f"Unsloth: Restored original {attr_name}")
        return True
    except Exception as e:
        if UNSLOTH_ENABLE_LOGGING:
            logger.error(f"Unsloth: Failed to restore {attr_name}: {e}")
        return False


def list_stored_originals(target_obj: Any) -> List[str]:
    """
    List all stored original functions on a target object.
    """
    stored = []
    for attr_name in dir(target_obj):
        if attr_name.startswith('_original_') and not attr_name.startswith('_original___'):
            # Extract the original method name from the unique storage name
            # Format: _original_{module}_{class}_{method} or _original_{class}_{method}
            parts = attr_name.split('_')[2:]  # Skip '_original_'
            if len(parts) >= 2:
                method_name = parts[-1]  # Last part is the method name
                stored.append(method_name)

    return sorted(list(set(stored)))  # Remove duplicates and sort


def restore_multiple(target_objs_and_attrs: List[Tuple[Any, str]]) -> Dict[str, bool]:
    """
    Restore multiple original functions.
    """
    results = {}

    for target_obj, attr_name in target_objs_and_attrs:
        key = f"{getattr(target_obj, '__name__', str(target_obj))}.{attr_name}"
        results[key] = restore_original(target_obj, attr_name)

    return results
