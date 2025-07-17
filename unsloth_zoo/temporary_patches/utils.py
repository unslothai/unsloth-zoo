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

__all__ = [
    "patch_function",
    "process_return",
    "process_output_options",
    "KWARGS_TYPE",
    "raise_error"
    "Unpack",
    "Cache",
    "DynamicCache",
    "HybridCache",
    "StaticCache",
    "TextInput",
    "PreTokenizedInput",
    "ImageInput",
    "ImagesKwargs",
    "MultiModalData",
    "ProcessingKwargs",
    "ProcessorMixin",
]
import inspect
import typing as t
import torch
from typing import Any, Callable, Dict, List, Tuple
try:
    t_TypedDictMeta = t._TypedDictMeta
except:
    from typing_extensions import _TypedDictMeta as t_TypedDictMeta

from packaging.version import Version
from .common import UNSLOTH_ENABLE_LOGGING, UNSLOTH_COMPILE_DISABLE, torch_compile_options, logger

EMPTY = inspect._empty

def raise_error(f: str, exception: Any):
    # Raises error only if logging is on
    if UNSLOTH_ENABLE_LOGGING:
        logger.error(
            f"==================\n"\
            f"Failed to patch {f}. Error\n"\
            f"{str(exception)}\n"\
            f"==================\n"
        )
    return
pass

# Output classes sometimes might remove objects, so we make a fastpath
global PROCESS_RETURN_ALLOWED_TYPES
PROCESS_RETURN_ALLOWED_TYPES = {}
def process_return(
    output_class : type,
    return_dict : Dict,
) -> Any:
    """ CausalLMOutputWithPast(...) might change arguments! """
    try:
        if output_class in PROCESS_RETURN_ALLOWED_TYPES:
            allowed_keys = PROCESS_RETURN_ALLOWED_TYPES[output_class]
            chosen_keys = allowed_keys & return_dict.keys()
            return_dict = {key : return_dict[key] for key in chosen_keys}
        return output_class(**return_dict)
    except:
        # We inspect the argument then only allow those arguments
        return_dict_keys = return_dict.keys()
        allowed_keys = set(inspect.signature(output_class).parameters.keys())
        chosen_keys  = allowed_keys & return_dict_keys
        return_dict  = {key : return_dict[key] for key in chosen_keys}
        logger.warning_once(
            f"Unsloth: Returning {output_class.__name__} changed return args.\n"\
            f"Previously we wanted {return_dict_keys}\n"\
            f"Now we can only use  {allowed_keys}\n"\
            f"These keys are gone: {return_dict_keys - allowed_keys}"
        )
        try:
            PROCESS_RETURN_ALLOWED_TYPES[output_class] = allowed_keys
            return output_class(**return_dict)
        except Exception as e:
            raise RuntimeError(str(e))
    pass
pass

# Get Unpack
# Python 3.10 doesnt have t_Unpack!
try:
    t_Unpack = t.Unpack
except:
    from typing_extensions import Unpack as t_Unpack
try:
    from transformers.processing_utils import Unpack
    assert \
        type(Unpack) is type(t_Unpack), \
        "Unsloth: Unpack type changed! Please file a bug report asap!"
except Exception as e:
    raise RuntimeError(
        f"Unsloth: Unpack has been moved! Other error = {str(e)}.\n"\
        "Please file a bug report asap!"
    )
pass
KWARGS_TYPE = t_Unpack[t_TypedDictMeta]


# Sometimes output classes change! Account for this
def process_output_options(
    self : Any,
    locals_items : Dict,
    kwargs : Dict,
) -> Dict:
    """ Latest transformers also deletes output_attentions and output_hidden_states """
    # Preserve old transformers style
    # 4.54.0 removes output_attentions and output_hidden_states
    output_attentions    = locals_items.get("output_attentions",    False)
    output_hidden_states = locals_items.get("output_hidden_states", False)

    output_attentions = output_attentions if output_attentions is not None else getattr(self.config, "output_attentions", False)
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else getattr(self.config, "output_hidden_states", False)
    )
    # Move to kwargs
    kwargs["output_attentions"]    = output_attentions
    kwargs["output_hidden_states"] = output_hidden_states
    return kwargs
pass


# Latest transformers 4.54.0 changed to TransformersKwargs
TransformersKwargs = t_TypedDictMeta
try:
    from transformers.utils import TransformersKwargs
    assert \
        type(TransformersKwargs) is t_TypedDictMeta, \
        "Unsloth: TransformersKwargs type changed! Please file a bug report asap!"
except Exception as e:
    from transformers import __version__ as transformers_version
    if Version(transformers_version) >= Version("4.54.0.dev0"):
        raise RuntimeError(
            f"Unsloth: TransformersKwargs has been moved! Other error = {str(e)}.\n"\
            "Please file a bug report asap!"
        )
    else:
        pass
pass

# Get FlashAttentionKwargs
FlashAttentionKwargs = t_TypedDictMeta
try:
    from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
    assert \
        type(FlashAttentionKwargs) is t_TypedDictMeta, \
        "Unsloth: FlashAttentionKwargs type changed! Please file a bug report asap!"
except:
    # No more FlashAttentionKwargs can ignore!
    pass
pass

# Get LossKwargs and KwargsForCausalLM
LossKwargs = t_TypedDictMeta
KwargsForCausalLM = t_TypedDictMeta
try:
    from transformers.utils import LossKwargs
    assert \
        type(LossKwargs) is t_TypedDictMeta, \
        "Unsloth: LossKwargs type changed! Please file a bug report asap!"
    if FlashAttentionKwargs != t_TypedDictMeta:
        class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...
except:
    # New transformers changed KwargsForCausalLM to TransformersKwargs
    KwargsForCausalLM = TransformersKwargs
    if KwargsForCausalLM == t_TypedDictMeta:
        logger.error(
            "Unsloth: KwargsForCausalLM cannot be inherited from "\
            f"TransformersKwargs since it's of type = {type(TransformersKwargs)}"
        )
pass

# Get Cache
Cache = t.Any
try: from transformers.cache_utils import Cache
except: pass
DynamicCache = t.Any
try: from transformers.cache_utils import DynamicCache
except: pass
HybridCache = t.Any
try: from transformers.cache_utils import HybridCache
except: pass
StaticCache = t.Any
try: from transformers.cache_utils import StaticCache
except: pass

# Get text and image utils and typings
TextInput = str
try: from transformers.tokenization_utils_base import TextInput
except: pass
PreTokenizedInput = List[str]
try: from transformers.tokenization_utils_base import PreTokenizedInput
except: pass
ImageInput = t.Any
try: from transformers.image_utils import ImageInput
except: pass
ImagesKwargs = t.Any
try: from transformers.processing_utils import ImagesKwargs
except: pass
MultiModalData = t.Any
try: from transformers.processing_utils import MultiModalData
except: pass
ProcessingKwargs = t.Any
try: from transformers.processing_utils import ProcessingKwargs
except: pass
ProcessorMixin = t.Any
try: from transformers.processing_utils import ProcessorMixin
except: pass

# Normalize common built-in types to their typing equivalents
VAR_KEYWORD_ID = inspect.Parameter.VAR_KEYWORD.value
TYPE_MAPPINGS = {
    torch.Tensor         : torch.Tensor,
    torch.IntTensor      : torch.Tensor,
    torch.FloatTensor    : torch.Tensor,
    list                 : t.List,
    dict                 : t.Dict,
    set                  : t.Set,
    tuple                : t.Tuple,
    frozenset            : t.FrozenSet,
    Unpack               : t_Unpack,
    KWARGS_TYPE          : t_Unpack[t_TypedDictMeta],
    Cache                : t.Any,
    DynamicCache         : t.Any,
    HybridCache          : t.Any,
    StaticCache          : t.Any,
    ImageInput           : t.Any,
    ImagesKwargs         : t.Any,
    MultiModalData       : t.Any,
    ProcessingKwargs     : t.Any,
    ProcessorMixin       : t.Any,
}
if TextInput         != str:       TYPE_MAPPINGS[TextInput]         = t.Any
if PreTokenizedInput != List[str]: TYPE_MAPPINGS[PreTokenizedInput] = t.Any

if TransformersKwargs   != t_TypedDictMeta: TYPE_MAPPINGS[TransformersKwargs]   = t_TypedDictMeta
if FlashAttentionKwargs != t_TypedDictMeta: TYPE_MAPPINGS[FlashAttentionKwargs] = t_TypedDictMeta
if LossKwargs           != t_TypedDictMeta: TYPE_MAPPINGS[LossKwargs]           = t_TypedDictMeta

def _canonicalize_annotation(annotation: Any) -> Any:
    """
    Canonicalize type annotations for consistent comparison.
    Makes List[int], typing.List[int], list[int] equivalent.
    """
    if annotation is EMPTY:
        return EMPTY

    if hasattr(t, "get_origin"):
        origin = t.get_origin(annotation)
        if origin is not None:
            args = t.get_args(annotation)
            args = tuple(canonicalize_annotation(arg) for arg in args)
            return (origin, args)
    return TYPE_MAPPINGS.get(annotation, annotation)
pass
def canonicalize_annotation(annotation: Any) -> Any:
    annotation = _canonicalize_annotation(annotation)
    if type(annotation) is tuple and len(annotation) == 2:
        # Fix up reordering of Union with sorting
        # Union[str, List[str], list[str]] gets reduced to Union[str, list[str]]
        # due to duplicates. We also sort Unions
        if annotation[0] == t.Union:
            annotation = (annotation[0], tuple(set(annotation[1])),)
        # Fix up kwargs
        # (typing.Unpack, (<class 'transformers.models.csm.modeling_csm.KwargsForCausalLM'>,)) to
        # (typing.Unpack, (<class 'typing._TypedDictMeta'>,))
        elif annotation[0] == t_Unpack and \
            type(annotation[1]) is tuple and \
            len(annotation[1]) == 1 and \
            "Kwargs" in str(annotation[1][0]):
            annotation = (t_Unpack, (t_TypedDictMeta,),)

        # (typing.Unpack, <class 'typing._TypedDictMeta'>,)
        elif annotation[0] == t_Unpack and \
            type(annotation[1]) is type and \
            "Kwargs" in str(annotation[1]):
            annotation = (t_Unpack, (t_TypedDictMeta,),)

    return annotation
pass


def get_function_fingerprint(func: Callable) -> List[Dict[str, Any]]:
    """
    Return a fingerprint we can use to compare function signatures.
    Returns: [{'name': str, 'kind': int, 'is_required': bool, 'annotation': Any}]
    """
    try:
        signature = inspect.signature(func)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Unsloth: Cannot inspect function signature: {e}")
    fingerprint = []
    signature_parameters = signature.parameters.values()
    
    for kk, param in enumerate(signature_parameters):
        param_name = str(param.name)
        param_kind = param.kind.value # 4 is type VAR_KEYWORD **kwargs
        annotation = param.annotation

        # If **kwargs is seen, then canonicalize name to simply kwargs
        if "kwargs" in param_name.lower():
            param_name = "kwargs"
            # Also if no type set, set it to a default
            if \
                (param_kind == VAR_KEYWORD_ID) and \
                (annotation == EMPTY) and \
                (len(signature_parameters)-1 == kk):
                annotation = (t_Unpack, (t_TypedDictMeta,),)
        pass
        # If name is simply x, and annotation is empty, set to torch.Tensor
        # For eg def forward(self, x)
        if \
            (param_name == "x") and \
            (len(signature_parameters) == 2) and \
            (func.__name__ == "forward") and \
            (annotation == EMPTY):
            annotation = torch.Tensor
        pass
        fingerprint.append({
            'name': param_name,
            'kind': param_kind,
            'is_required': param.default is EMPTY, # True = required
            'annotation' : canonicalize_annotation(annotation),
        })
    return fingerprint
pass


def removed_flags(
    old_fp : List[Dict],
    new_fp : List[Dict],
) -> List[str]:
    old_params = set(x["name"] for x in old_fp)
    new_params = set(x["name"] for x in new_fp)
    removed_params = list(old_params ^ new_params)
    removed_params.sort()
    return tuple(removed_params)
pass


def can_safely_patch(
    original_func: Callable,
    new_func: Callable, 
    match_level: str = "strict",
) -> Tuple[bool, str]:
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
        # New transformers 4.54.0 removed output_attentions and output_hidden_states
        # We check it and ignore if the old function has both these, and the new removed them
        if removed_flags(old_fp, new_fp) == ("output_attentions", "output_hidden_states",):
            return False, f"New function removed output_attentions and output_hidden_states"
        return False, f"Parameter count mismatch: {len(old_fp)} vs {len(new_fp)}"

    for old_param, new_param in zip(old_fp, new_fp):
        if (old_param['name'], old_param['kind']) != (new_param['name'], new_param['kind']):
            return False, f"Parameter '{old_param['name']}' signature changed"

        if new_param['is_required'] and not old_param['is_required']:
            return False, f"Parameter '{new_param['name']}' changed from optional to required"

        # For strict matching, also check type annotations
        if match_level == "strict" and old_param['annotation'] != new_param['annotation']:
            return False, \
            f"Parameter '{old_param['name']}' type annotation changed from:\n"\
            f"{old_param['annotation']} to\n"\
            f"{new_param['annotation']}"

    return True, ""
pass


def _get_unique_storage_name(
    target_obj: Any,
    attr_name: str,
) -> str:
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
pass


def patch_function(
    target_obj: Any,
    attr_name: str,
    new_func: Callable, 
    force: bool = False,
    store_original: bool = True, 
    match_level: str = "strict",
    fullgraph = None,
    dynamic = True,
) -> bool:
    """
    Patch a function/method on an object.
    """
    if not hasattr(target_obj, attr_name):
        if UNSLOTH_ENABLE_LOGGING:
            logger.error(f"Unsloth: Attribute '{attr_name}' not found on {target_obj.__name__}")
        return False

    original_func = getattr(target_obj, attr_name)

    # torch.compile function if requested
    if fullgraph is not None and type(fullgraph) is bool and not UNSLOTH_COMPILE_DISABLE:
        # Get wrapped function if already compiled
        if hasattr(new_func, "get_compiler_config"):
            new_func = new_func.__wrapped__
        if hasattr(original_func, "get_compiler_config"):
            original_func = original_func.__wrapped__
        new_func = torch.compile(
            new_func,
            fullgraph = fullgraph,
            dynamic = dynamic,
            options = torch_compile_options,
        )
    pass

    # Store original for potential restoration with unique name
    if store_original:
        unique_name = _get_unique_storage_name(target_obj, attr_name)
        setattr(target_obj, unique_name, original_func)
        # if UNSLOTH_ENABLE_LOGGING:
        #     logger.info(f"Unsloth: Stored original as {unique_name}")
    pass

    if not force:
        is_safe, reason = can_safely_patch(original_func, new_func, match_level)
        if not is_safe:
            if UNSLOTH_ENABLE_LOGGING:
                logger.error(f"Unsloth: Skipped {target_obj.__name__}.{attr_name}\nReason: {reason}")
            return False
    pass
    try:
        setattr(target_obj, attr_name, new_func)
        if UNSLOTH_ENABLE_LOGGING:
            logger.info(f"Unsloth: Patched {target_obj.__name__}.{attr_name}.")
        return True
    except Exception as e:
        if UNSLOTH_ENABLE_LOGGING:
            logger.error(f"Unsloth: Failed to patch {target_obj.__name__}.{attr_name}: {e}")
        return False
    pass
pass


def patch_multiple(
    patches: List[Tuple[Any, str, Callable]], 
    force: bool = False, 
    fail_fast: bool = True,
    match_level: str = "strict",
    fullgraph = None,
    dynamic = True,
) -> Dict[str, bool]:
    """
    Apply multiple patches at once.
    """
    results = {}

    for target_obj, attr_name, new_func in patches:
        key = f"{getattr(target_obj, '__name__', str(target_obj))}.{attr_name}"
        success = patch_function(
            target_obj,
            attr_name,
            new_func,
            force = force,
            match_level = match_level,
            fullgraph = fullgraph,
            dynamic = dynamic,
        )
        results[key] = success

        if fail_fast and not success:
            if UNSLOTH_ENABLE_LOGGING:
                logger.error(f"Unsloth: Stopping patch process due to failure on {key}")
            break

    return results
pass


def restore_original(
    target_obj: Any,
    attr_name: str,
) -> bool:
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
pass


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
pass


def restore_multiple(target_objs_and_attrs: List[Tuple[Any, str]]) -> Dict[str, bool]:
    """
    Restore multiple original functions.
    """
    results = {}

    for target_obj, attr_name in target_objs_and_attrs:
        key = f"{getattr(target_obj, '__name__', str(target_obj))}.{attr_name}"
        results[key] = restore_original(target_obj, attr_name)

    return results
pass
