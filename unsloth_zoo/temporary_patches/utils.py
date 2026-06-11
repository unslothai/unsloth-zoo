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
    "patch_function_past_key_values",
    "process_return",
    "process_output_options",
    "KWARGS_TYPE",
    "raise_error",
    "Unpack",
    "Cache",
    "DynamicCache",
    "HybridCache",
    "HAS_HYBRID_CACHE",
    "StaticCache",
    "TextInput",
    "PreTokenizedInput",
    "ImageInput",
    "ImagesKwargs",
    "MultiModalData",
    "ProcessingKwargs",
    "ProcessorMixin",
    "_get_unique_storage_name",
    "dedent",
]
import inspect
import typing as t
import torch
from textwrap import dedent
from typing import Any, Callable, Dict, List, Tuple, Union
try:
    t_TypedDictMeta = t._TypedDictMeta
except:
    from typing_extensions import _TypedDictMeta as t_TypedDictMeta

from ..utils import Version
from .common import UNSLOTH_ENABLE_LOGGING, UNSLOTH_COMPILE_DISABLE, torch_compile_options, logger

EMPTY = inspect._empty

def raise_error(f: str, exception: Any = None):
    # Raises error only if logging is on
    if UNSLOTH_ENABLE_LOGGING:
        logger.error(
            f"==================\n"\
            f"Failed to patch {f}. Error\n"\
            f"{str(exception) if exception is not None else ''}\n"\
            f"==================\n"
        )
    return
pass

# Fastpath: output classes sometimes drop args.
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

# Get Unpack (Python 3.10 lacks t.Unpack).
try:
    t_Unpack = t.Unpack
except:
    from typing_extensions import Unpack as t_Unpack
# Fix stale module caching (Kaggle/Colab after upgrading packages mid-session):
# old modules stay cached in sys.modules and fail when new on-disk files reference
# upgraded-only symbols. PIL: clear sys.modules. numpy/scipy: C extensions cannot
# be reloaded, so raise a clear restart error.
import sys as _sys
from importlib.metadata import version as _get_pkg_version
from packaging.version import parse as parse_version

# numpy: C extensions cannot be reloaded, so must restart.
_np_mod = _sys.modules.get("numpy")
if _np_mod is not None and hasattr(_np_mod, "__version__"):
    try:
        _installed_numpy = _get_pkg_version("numpy")
        if parse_version(_np_mod.__version__).public != parse_version(_installed_numpy).public:
            raise RuntimeError(
                f"***** numpy was upgraded mid-session (loaded: {_np_mod.__version__}, "
                f"installed: {_installed_numpy}) but the kernel still has the old version "
                f"in memory. numpy uses C extensions that cannot be reloaded without "
                f"restarting. Please restart your runtime/kernel after installing packages. *****"
            )
    except RuntimeError:
        raise
    except Exception:
        pass # Best-effort; non-critical if importlib.metadata is unavailable.

# PIL: fixable by clearing sys.modules.
_pil_mod = _sys.modules.get("PIL")
if _pil_mod is not None and hasattr(_pil_mod, "__version__"):
    try:
        _installed_pillow = _get_pkg_version("Pillow")
        if _pil_mod.__version__ != _installed_pillow:
            for _k in [k for k in list(_sys.modules.keys()) if k == "PIL" or k.startswith("PIL.")]:
                del _sys.modules[_k]
    except Exception:
        pass
del _sys, _pil_mod

# ROCm on Windows ships PyTorch without the full torch.distributed C-extension
# stack. torchao pulls the entire distributed chain in at module-import time,
# cascading into ImportError even for code paths that never use distributed
# features (e.g. plain LoRA training).
# Fix: if torchao can't be imported, install a sys.meta_path hook intercepting
# all "torchao"/"torchao.*" imports and returning self-contained stub modules.
# Each stub satisfies `from torchao.X import Y` via a no-op sentinel class, so
# transformers can define TorchAoHfQuantizer at import time. Any runtime call
# needing a real torchao op still fails loudly.
# Ref: https://github.com/ROCm/TheRock/issues/3284
import sys as _sys_rocm_stub, types as _types_rocm_stub
from importlib.abc import MetaPathFinder as _MetaPathFinder, Loader as _Loader
from importlib.machinery import ModuleSpec as _ModuleSpec


# Metaclass making sentinel classes chainable via attribute access
# (AffineQuantizedTensor.subattr -> another sentinel). peft does
# isinstance(weight, AffineQuantizedTensor), which needs a real type.
class _ROCmSentinelMeta(type):
    def __getattr__(cls, name):
        child = _ROCmSentinelMeta(name, (), {"__module__": cls.__module__})
        setattr(cls, name, child)
        return child


def _rocm_make_sentinel(attr, parent_name):
    """Return a sentinel class that is a proper type (works in isinstance())."""
    return _ROCmSentinelMeta(attr, (), {"__module__": parent_name})


def _rocm_make_torchao_stub(name):
    """Create a stub module for a torchao path.

    Sub-module imports get module stubs via the meta_path finder; direct
    attribute access returns a sentinel CLASS so isinstance() works (always
    False, since no real weight is an instance of the sentinel).
    """
    import sys as _s, types as _t
    from importlib.machinery import ModuleSpec as _MS

    mod = _t.ModuleType(name)
    mod.__path__    = []
    mod.__package__ = name
    mod.__spec__    = _MS(name, loader=None)

    def _getattr(attr):
        full = f"{name}.{attr}"
        # Reuse an already-imported sub-module; else a sentinel class.
        if full in _s.modules:
            obj = _s.modules[full]
        else:
            obj = _rocm_make_sentinel(attr, name)
        setattr(mod, attr, obj)
        return obj

    mod.__getattr__ = _getattr
    return mod


class _ROCmTorchaoLoader(_Loader):
    """Loader that creates a recursive stub module for any torchao path."""

    def create_module(self, spec):
        return _rocm_make_torchao_stub(spec.name)

    def exec_module(self, module):
        pass  # _rocm_make_torchao_stub already configured it


class _ROCmTorchaoFinder(_MetaPathFinder):
    """Intercepts torchao.* imports on Windows ROCm where torch.distributed is incomplete."""
    _loader = _ROCmTorchaoLoader()

    def find_spec(self, fullname, path, target=None):
        if fullname == "torchao" or fullname.startswith("torchao."):
            from importlib.machinery import ModuleSpec as _MS
            return _MS(fullname, self._loader, is_package=True)
        return None

    def find_module(self, fullname, path=None):   # Python < 3.12 shim
        return None


# Only Windows + ROCm (HIP) PyTorch needs this stub -- the one build where
# `import torchao` crashes on the missing torch.distributed C-extension stack.
# Elsewhere a failing import just means torchao isn't installed (transformers
# handles that), and the stub would be harmful: is_torchao_available() reads a
# sentinel torchao.__version__ and crashes in packaging.version.parse() with
# "'_ROCmSentinelMeta' object is not iterable".
_is_windows_rocm = False
if _sys_rocm_stub.platform == "win32":
    try:
        import torch as _torch_rocm_probe
        _is_windows_rocm = bool(
            getattr(getattr(_torch_rocm_probe, "version", None), "hip", None)
            or "rocm" in getattr(_torch_rocm_probe, "__version__", "").lower()
        )
        del _torch_rocm_probe
    except Exception:
        _is_windows_rocm = False

if _is_windows_rocm and "torchao" not in _sys_rocm_stub.modules:
    try:
        import torchao  # noqa: F401
    except Exception:
        # torchao import failed on Windows ROCm: install the meta path hook so
        # subsequent "import torchao.*" gets a harmless stub.
        _sys_rocm_stub.meta_path.insert(0, _ROCmTorchaoFinder())

# Keep _rocm_make_torchao_stub / _rocm_make_sentinel / _ROCmSentinelMeta alive;
# the loader and sentinel classes call them at runtime.
del _ROCmTorchaoLoader, _ROCmTorchaoFinder
del _MetaPathFinder, _Loader, _ModuleSpec, _sys_rocm_stub, _types_rocm_stub
del _is_windows_rocm

try:
    from transformers.processing_utils import Unpack
    assert \
        type(Unpack) is type(t_Unpack), \
        "Unsloth: Unpack type changed! Please file a bug report asap!"
except ImportError as e:
    e = str(e)
    if "cannot import name '_center' from 'numpy._core.umath'" in e:
        raise RuntimeError(
            f"***** You might have used uv to install packages, and they broke numpy. Try restarting your runtime. *****"
        )
    elif "torchvision::nms does not exist" in e:
        raise RuntimeError(
            f"***** Please update and reinstall torchvision - it broke! `pip install --upgrade --force-reinstall --no-cache-dir torchvision` *****"
        )
    elif "PIL" in e or "_Ink" in e or "Pillow" in e:
        raise RuntimeError(
            f"***** Your Pillow (PIL) version is incompatible with torchvision. "
            f"Please run `pip install --upgrade --force-reinstall Pillow` then restart your runtime/kernel. *****"
        )
    elif "Unpack" not in e:
        raise Exception(e)
    raise RuntimeError(
        f"Unsloth: Unpack has been moved! Other error = {str(e)}.\n"\
        "Please file a bug report asap!"
    )
except Exception as e:
    e_str = str(e)
    if "numpy" in e_str and ("_blas" in e_str or "_multiarray" in e_str):
        raise RuntimeError(
            f"***** numpy was likely upgraded mid-session without restarting the kernel. "
            f"numpy C extensions cannot be reloaded in-place. "
            f"Please restart your runtime/kernel after installing packages. "
            f"Original error: {e_str} *****"
        )
    raise
pass
KWARGS_TYPE = t_Unpack[t_TypedDictMeta]


# Account for output classes changing across versions.
def process_output_options(
    self : Any,
    locals_items : Dict,
    kwargs : Dict,
) -> Dict:
    """ Latest transformers also deletes output_attentions and output_hidden_states """
    # transformers 4.54.0 removed output_attentions/output_hidden_states.
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
except ImportError as e:
    from transformers import __version__ as transformers_version
    if Version(transformers_version) >= Version("4.54.0.dev0"):
        raise RuntimeError(
            f"Unsloth: TransformersKwargs has been moved! Other error = {str(e)}.\n"\
            "Please file a bug report asap!"
        )
    else:
        pass
except Exception as e:
    raise Exception(e)
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
HAS_HYBRID_CACHE = False
try:
    from transformers.cache_utils import HybridCache
    HAS_HYBRID_CACHE = True
except Exception:
    pass
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
VAR_KEYWORD_ID    = inspect.Parameter.VAR_KEYWORD.value
VAR_POSITIONAL_ID = inspect.Parameter.VAR_POSITIONAL.value
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

try:
    import types
    TYPE_MAPPINGS[types.UnionType] = t.Union
except:
    pass

def _canonicalize_annotation(annotation: Any) -> Any:
    """Canonicalize annotations so List[int]/typing.List[int]/list[int] match."""
    if annotation is EMPTY:
        return EMPTY

    if hasattr(t, "get_origin"):
        origin = t.get_origin(annotation)
        if origin is not None:
            args = t.get_args(annotation)
            args = tuple(canonicalize_annotation(arg) for arg in args)
            # Canonicalize origin (types.UnionType -> typing.Union) so
            # `int | str` and `Union[int, str]` match.
            origin = TYPE_MAPPINGS.get(origin, origin)
            return (origin, args)
    return TYPE_MAPPINGS.get(annotation, annotation)
pass
def canonicalize_annotation(annotation: Any) -> Any:
    annotation = _canonicalize_annotation(annotation)
    if type(annotation) is tuple and len(annotation) == 2:
        # Dedupe + sort Union args (Union[str, List[str], list[str]] ->
        # Union[str, list[str]]); sort by str(x) since sets are unordered.
        if annotation[0] == t.Union:
            args = list(set(annotation[1]))
            args.sort(key = lambda x: str(x))
            args = tuple(args)
            annotation = (annotation[0], args,)
        # Normalize Unpack[...Kwargs] to Unpack[_TypedDictMeta].
        elif annotation[0] == t_Unpack and \
            type(annotation[1]) is tuple and \
            len(annotation[1]) == 1 and \
            "Kwargs" in str(annotation[1][0]):
            annotation = (t_Unpack, (t_TypedDictMeta,),)

        # Same normalization for the bare-type Unpack form.
        elif annotation[0] == t_Unpack and \
            type(annotation[1]) is type and \
            "Kwargs" in str(annotation[1]):
            annotation = (t_Unpack, (t_TypedDictMeta,),)

    return annotation
pass


def get_function_fingerprint(func: Callable) -> List[Dict[str, Any]]:
    """Fingerprint for comparing function signatures.

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

        # Canonicalize any **kwargs name to "kwargs".
        if "kwargs" in param_name.lower():
            param_name = "kwargs"
            # Default the annotation when untyped.
            if \
                (param_kind == VAR_KEYWORD_ID) and \
                (annotation == EMPTY) and \
                (len(signature_parameters)-1 == kk):
                annotation = (t_Unpack, (t_TypedDictMeta,),)
        pass
        # forward(self, x) with untyped x -> torch.Tensor.
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

    # If relaxed, allow matching with *args, **kwargs
    def check_args_kwargs(old_fp, new_fp, removed_flags_list):
        if (len(new_fp) >= 2) and (
            new_fp[-1]["kind"] == VAR_KEYWORD_ID and new_fp[-1]["name"] == "kwargs"
        ) and (
            new_fp[-2]["kind"] == VAR_POSITIONAL_ID and new_fp[-2]["name"] == "args"
        ):
            # Check removed flags must not have any gaps!
            removed_flags_list = set(removed_flags_list)
            removed_flags_list -= set({"args", "kwargs"})
            i = 0
            fail = False
            while i < len(old_fp):
                old_arg = old_fp[i]
                if old_arg["name"] in removed_flags_list:
                    # Go to the end
                    i += 1
                    while i < len(old_fp):
                        old_arg = old_fp[i]
                        if old_arg["name"] not in removed_flags_list:
                            # Hole seen but ignore args, kwargs
                            if (old_arg["name"] in ("args", "kwargs",)) and \
                                (old_arg["kind"] in (VAR_KEYWORD_ID, VAR_POSITIONAL_ID,)):
                                pass
                            else:
                                fail = True
                                break
                        i += 1
                i += 1
            if not fail:
                return True, f"Replacing with *args, **kwargs"
        return False, ""
    pass


    if len(old_fp) != len(new_fp):
        # transformers 4.54.0 dropped output_attentions/output_hidden_states;
        # tolerate exactly that removal.
        removed_flags_list = removed_flags(old_fp, new_fp)
        if removed_flags_list == ("output_attentions", "output_hidden_states",):
            return False, f"New function removed output_attentions and output_hidden_states"
        result, error = check_args_kwargs(old_fp, new_fp, removed_flags_list)
        if result == True:
            return True, error
        return False, f"Parameter count mismatch: {len(old_fp)} vs {len(new_fp)}"
    pass

    for old_param, new_param in zip(old_fp, new_fp):
        if (old_param['name'], old_param['kind']) != (new_param['name'], new_param['kind']):
            if match_level == "relaxed":
                # Last chance: *args, **kwargs replacement.
                removed_flags_list = removed_flags(old_fp, new_fp)
                result, error = check_args_kwargs(old_fp, new_fp, removed_flags_list)
                if result == True:
                    return True, error
            return False, f"Parameter '{old_param['name']}' signature changed"

        if new_param['is_required'] and not old_param['is_required']:
            return False, f"Parameter '{new_param['name']}' changed from optional to required"

        # Strict matching also compares type annotations.
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
    """Unique attribute name for stashing the original function."""
    if hasattr(target_obj, '__name__'):
        obj_name = target_obj.__name__
    elif hasattr(target_obj, '__class__'):
        obj_name = target_obj.__class__.__name__
    else:
        obj_name = str(type(target_obj).__name__)

    # Include module for extra uniqueness when available.
    if hasattr(target_obj, '__module__'):
        module_name = target_obj.__module__.split('.')[-1]
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
    """Patch a function/method on an object."""
    if not hasattr(target_obj, attr_name):
        if UNSLOTH_ENABLE_LOGGING:
            logger.error(f"Unsloth: Attribute '{attr_name}' not found on {target_obj.__name__}")
        return False

    original_func = getattr(target_obj, attr_name)

    # torch.compile if requested.
    if fullgraph is not None and type(fullgraph) is bool and not UNSLOTH_COMPILE_DISABLE:
        # Unwrap already-compiled functions.
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

    # Stash original under a unique name for later restoration.
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


def patch_function_past_key_values(
    target_obj: Any,
    attr_name: str,
    new_functions: Union[Callable, List[Callable]], 
    force: bool = False,
    store_original: bool = True, 
    match_level: str = "strict",
    fullgraph = None,
    dynamic = True,
) -> bool:
    """ Patch either past_key_value or past_key_values """
    if not hasattr(target_obj, attr_name):
        if UNSLOTH_ENABLE_LOGGING:
            logger.error(f"Unsloth: Attribute '{attr_name}' not found on {target_obj.__name__}")
        return False

    original_func = getattr(target_obj, attr_name)
    try:
        old_keys = inspect.signature(original_func).parameters.keys()
    except:
        logger.error(f"Unsloth: Cannot inspect {target_obj.__name__}")
        return False
    success = False
    error = ""
    for func in new_functions:
        try:
            new_keys = inspect.signature(func).parameters.keys()
        except Exception as e:
            error = str(e)
            continue
        # Check if either is provided
        for key in ("past_key_value", "past_key_values",):
            if key in new_keys and key in old_keys:
                try:
                    success = patch_function(
                        target_obj = target_obj,
                        attr_name = attr_name,
                        new_func = func, 
                        force = force,
                        store_original = store_original,
                        match_level = match_level,
                        fullgraph = fullgraph,
                        dynamic = dynamic,
                    )
                    if success: break
                except Exception as e:
                    error = str(e)
                    continue
    if not success and UNSLOTH_ENABLE_LOGGING:
        logger.error(f"Unsloth: Failed to patch {target_obj.__name__}.{attr_name}: {error}")
    return success
pass


def patch_multiple(
    patches: List[Tuple[Any, str, Callable]], 
    force: bool = False, 
    fail_fast: bool = True,
    match_level: str = "strict",
    fullgraph = None,
    dynamic = True,
) -> Dict[str, bool]:
    """Apply multiple patches at once."""
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
    """Restore the original function if it was stored."""
    unique_name = _get_unique_storage_name(target_obj, attr_name)

    if not hasattr(target_obj, unique_name):
        if UNSLOTH_ENABLE_LOGGING:
            logger.error(f"Unsloth: No stored original found for {attr_name} (looked for {unique_name})")
        return False

    try:
        original_func = getattr(target_obj, unique_name)
        setattr(target_obj, attr_name, original_func)
        delattr(target_obj, unique_name)
        if UNSLOTH_ENABLE_LOGGING:
            logger.info(f"Unsloth: Restored original {attr_name}")
        return True
    except Exception as e:
        if UNSLOTH_ENABLE_LOGGING:
            logger.error(f"Unsloth: Failed to restore {attr_name}: {e}")
        return False
pass


def list_stored_originals(target_obj: Any) -> List[str]:
    """List all stored original functions on a target object."""
    stored = []
    for attr_name in dir(target_obj):
        if attr_name.startswith('_original_') and not attr_name.startswith('_original___'):
            # Name format: _original_{module}_{class}_{method} (method = last part).
            parts = attr_name.split('_')[2:]
            if len(parts) >= 2:
                method_name = parts[-1]
                stored.append(method_name)

    return sorted(list(set(stored)))
pass


def restore_multiple(target_objs_and_attrs: List[Tuple[Any, str]]) -> Dict[str, bool]:
    """Restore multiple original functions."""
    results = {}

    for target_obj, attr_name in target_objs_and_attrs:
        key = f"{getattr(target_obj, '__name__', str(target_obj))}.{attr_name}"
        results[key] = restore_original(target_obj, attr_name)

    return results
pass
