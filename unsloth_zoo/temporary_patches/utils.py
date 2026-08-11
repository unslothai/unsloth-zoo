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
    "compile_with_eager_fallback",
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
    "eager_fallback_state",
    "force_eager_fallback",
    "apply_pending_eager_fallbacks",
    "torch_compile_with_fallback",
]
import functools
import inspect
import weakref
import typing as t
import torch
from textwrap import dedent
from typing import Any, Callable, Dict, List, Tuple, Union
try:
    t_TypedDictMeta = t._TypedDictMeta
except:
    from typing_extensions import _TypedDictMeta as t_TypedDictMeta

from ..utils import Version
from .common import UNSLOTH_ENABLE_LOGGING, UNSLOTH_COMPILE_DISABLE, torch_compile_options, logger, unwrap_already_compiled

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
try:
    from packaging.version import parse as parse_version
except Exception:
    parse_version = None # Fall back to raw string compare below.

# numpy: C extensions cannot be reloaded, so must restart.
_np_mod = _sys.modules.get("numpy")
if _np_mod is not None and hasattr(_np_mod, "__version__"):
    try:
        _installed_numpy = _get_pkg_version("numpy")
        # Compare public versions so local labels (e.g. +computecanada) don't false-trigger.
        _public = (lambda v: parse_version(v).public) if parse_version is not None else (lambda v: v)
        if _public(_np_mod.__version__) != _public(_installed_numpy):
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

_TORCHVISION_BROKE = (
    "***** Please update and reinstall torchvision - it broke! "
    "`pip install --upgrade --force-reinstall --no-cache-dir torchvision` *****"
)

# A torchao built for a newer torch imports symbols straight out of torch (e.g.
# `from torch.nn.functional import ScalingType`). On an older pinned torch the
# symbol is absent, and the ImportError arrives here naming neither package.
_TORCHAO_TORCH_SYMBOLS = ("ScalingType", "ScalingGranularity", "Float8Tensor")


def _torchao_is_newer_than_torch(message):
    """Is this ImportError torchao reaching into a torch that lacks the symbol?

    Narrow on purpose: both the symbol and the failing module must be torch's,
    so unrelated errors keep their own text.
    """
    try:
        message = str(message)
    except Exception:                                    # noqa: BLE001
        return False
    if "cannot import name" not in message:
        return False
    if "torch" not in message:
        return False
    return any(sym in message for sym in _TORCHAO_TORCH_SYMBOLS)


def _torchao_torch_mismatch_message(message):
    """Name both versions, because the raw error names neither."""
    def _ver(mod):
        try:
            import importlib.metadata as _md
            return _md.version(mod)
        except Exception:                                # noqa: BLE001
            return "unknown"
    torch_v, ao_v = _ver("torch"), _ver("torchao")
    return (
        f"***** Unsloth: torchao {ao_v} was built for a newer torch than the "
        f"torch {torch_v} you have installed, so importing it fails with:\n"
        f"    {message}\n"
        f"Install a torchao that matches your torch, e.g.\n"
        f"    pip install --upgrade --force-reinstall --no-cache-dir "
        f"\"torchao<0.18\"\n"
        f"or upgrade torch instead. Then restart your runtime/kernel. *****"
    )


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
        raise RuntimeError(_TORCHVISION_BROKE)
    elif "No module named 'torchvision.io.video'" in e or \
         "No module named 'torchvision.io._video'" in e:
        # A half-installed torchvision, like the nms arm above. No release raises
        # this alone (0.25 ships `io/video.py`, 0.26 dropped it with its importer),
        # so only a tree partly overwritten by a venv install gets here.
        # The MISSING-MODULE form only: `cannot import name 'read_video' from
        # 'torchvision.io.video'` carries the same substring while the module is
        # right there, and the message below would then be a lie.
        raise RuntimeError(
            f"***** Your torchvision install is incomplete: `torchvision.io` "
            f"imports a `video` module that is not there. Please run "
            f"`pip install --upgrade --force-reinstall --no-cache-dir torchvision` "
            f"then restart your runtime/kernel. Original error = {e} *****"
        )
    elif "PIL" in e or "_Ink" in e or "Pillow" in e:
        raise RuntimeError(
            f"***** Your Pillow (PIL) version is incompatible with torchvision. "
            f"Please run `pip install --upgrade --force-reinstall Pillow` then restart your runtime/kernel. *****"
        )
    elif _torchao_is_newer_than_torch(e):
        raise RuntimeError(_torchao_torch_mismatch_message(e)) from None
    elif "Unpack" not in e:
        raise Exception(e)
    raise RuntimeError(
        f"Unsloth: Unpack has been moved! Other error = {str(e)}.\n"\
        "Please file a bug report asap!"
    )
except Exception as e:
    e_str = str(e)
    # The nms arm above, for the case that never arrives as an ImportError: a
    # torchvision whose compiled ops do not match torch fails inside
    # `_meta_registrations` at `register_fake("torchvision::nms")`, a RuntimeError.
    if "torchvision::nms does not exist" in e_str:
        raise RuntimeError(_TORCHVISION_BROKE)
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


def _recompile_limit_errors():
    """Dynamo's cache-exhaustion exceptions on whichever torch is installed.

    Looked up by name: torch 2.6 to 2.11 do not agree on which exist, and a
    missing one must not stop the tuple from being built.
    """
    try:
        import torch._dynamo.exc as _exc
    except Exception:
        return ()
    found = []
    for _n in ("FailOnRecompileLimitHit", "RecompileLimitExceeded",
               "CacheLimitExceeded"):
        _e = getattr(_exc, _n, None)
        if isinstance(_e, type) and issubclass(_e, BaseException):
            found.append(_e)
    return tuple(found)


_UNKNOWN = object()


def _saved_tensor_hook_accessor():
    """torch 2.8 added this; 2.4 to 2.7 cannot read the hook stack. Its own
    function so a test can stand in for an older torch without writing to the
    read-only `torch._C._autograd`."""
    return getattr(torch._C._autograd, "_top_saved_tensors_default_hooks", None)
# `CheckpointFunction.forward`/`.backward`, the reentrant path's own frames.
# Reentrant checkpointing re-differentiates what it recomputes, so a mode flip
# cannot strand it, and both phases carry one of these.
_REENTRANT_FRAMES = ("forward", "backward")

# `checkpoint_sequential` opens no region of its own: it calls `checkpoint` per
# segment and deliberately runs the LAST segment outside every one of them, so
# the catch-all below must skip it rather than read it as proof of a region.
_SEQUENTIAL_FRAME = "checkpoint_sequential"


def _frame_local(frame, name):
    """One local off a live frame, `_UNKNOWN` when it is not there."""
    try:
        return frame.f_locals.get(name, _UNKNOWN)
    except Exception:
        return _UNKNOWN


def _walk_for_checkpoint_frame():
    """The pre-2.8 answer: is any `torch.utils.checkpoint` frame on the stack?

    Never depth-capped: `checkpoint()` sits well over 60 frames above the
    compiled call under nested module dispatch, so stopping early answered "no
    region" and switched to eager inside one. The decision paths (give-up,
    `_in_non_reentrant_checkpoint`) are bounded instead by the bump caps, to a
    handful of calls per process.

    The one hot-path caller is the per-call probe on a torch with no hook
    accessor: 11.2us at depth 60, 33.2us at depth 200, versus 0.13us for the
    call itself. It pays instead with a per-step budget of 64 misses in
    `_probe_walk`, reset each step by `apply_pending_eager_fallbacks`; going
    over answers False, costing only an optimistic latch, and the give-up path
    re-derives the answer unbudgeted before anything acts on it.

    Frames rather than the accessor because 2.4 to 2.7 expose none, and None
    there would quietly restore the old behaviour on releases pyproject still
    supports. The module file is stable across every supported version.

    A reentrant region does not end the walk: it nests inside non-reentrant
    checkpointing, whose outer region a mode flip strands just the same, so skip
    its own `checkpoint` frame (the next one outward) and keep scanning.
    """
    try:
        import sys                              # module-level `_sys` is deleted above
        import torch.utils.checkpoint as _checkpoint
        origin = _checkpoint.__file__
    except Exception:
        return None
    if not origin: return None
    frame, reentrant = sys._getframe(1), 0
    while frame is not None:
        if frame.f_code.co_filename == origin:
            name = frame.f_code.co_name
            if name == _SEQUENTIAL_FRAME:
                pass                            # keep scanning outward
            elif name in _REENTRANT_FRAMES:
                # `checkpoint_sequential`'s per-segment closure is also called
                # `forward` in this same file, so only the locals tell them
                # apart: the autograd Function frames take a `ctx`.
                if _frame_local(frame, "ctx") is not _UNKNOWN:
                    reentrant += 1
            elif name == "checkpoint":
                use_reentrant = _frame_local(frame, "use_reentrant")
                if use_reentrant is False: return True
                if reentrant: reentrant -= 1
                elif use_reentrant is _UNKNOWN: return True
            else:
                return True                     # a pack or recompute frame
        frame = frame.f_back
    return False


def _checkpoint_early_stop_errors():
    """Checkpoint's "recomputation finished" signal, as an except-able tuple.

    Private and version-dependent, so resolve by name and fall back to an empty
    tuple, which `except` accepts and never matches."""
    try:
        import torch.utils.checkpoint as _checkpoint
    except Exception:
        return ()
    found = tuple(
        cls for cls in (getattr(_checkpoint, "_StopRecomputationError", None),)
        if isinstance(cls, type) and issubclass(cls, BaseException)
    )
    return found


def _is_checkpoint_pack_hook(pack):
    """Is this `torch.utils.checkpoint`'s own pack hook?"""
    return (getattr(pack, "__module__", "") == "torch.utils.checkpoint" and
            getattr(pack, "__qualname__", "").startswith(
                ("_checkpoint_hook", "_recomputation_hook")))


def _in_non_reentrant_checkpoint():
    """Are we inside a `use_reentrant = False` region? None when torch cannot say.

    The only saved-tensor hooks `torch.utils.checkpoint` installs are the
    non-reentrant path's two, whose qualnames (unchanged from 2.6 to current
    main) separate the forward pack from the backward recompute. The accessor
    arrived in 2.8, so 2.6 and 2.7 answer None and keep the old behaviour.
    Reentrant checkpointing installs no hooks and answers False, rightly: it
    re-differentiates what it recomputes, so a mode flip cannot strand it.

    Only the TOP of the hook stack is reported, and a user's own
    `saved_tensors_hooks`/`save_on_cpu` entered inside the region sits above
    ours, so an unrecognised hook is "cannot tell from here", not "no region":
    ask the frames instead.
    """
    top = _saved_tensor_hook_accessor()
    if top is not None:
        try:
            hooks = top(True)                   # ignore_is_tracing
        except Exception:
            hooks = _UNKNOWN
        if hooks is not _UNKNOWN:
            if not hooks: return False
            if _is_checkpoint_pack_hook(hooks[0]): return True
    return _walk_for_checkpoint_frame()


# Set when a compiled call ran with a non-reentrant checkpoint's pack hook on
# top: activations packed COMPILED, a backward still owed. Asking only whether
# the CURRENT call is in a region (as the give-up path used to) is false once
# that layer's forward returned, so latching mid-step left it to recompute
# eagerly in backward -- an abort, or wrong gradients when the shapes line up.
_PACKED_COMPILED_IN_CHECKPOINT = False


def _dynamo_is_tracing():
    """True while Dynamo is compiling the caller, across the supported torches."""
    for mod, name in ((getattr(torch, "compiler", None), "is_compiling"),
                      (getattr(torch, "_dynamo", None), "is_compiling")):
        fn = getattr(mod, name, None)
        if fn is None:
            continue
        try:
            return bool(fn())
        except Exception:
            continue
    return False


# Fruitless walks the per-call probe pays for in one step before it stops
# asking. With no hook accessor the walk is the only signal, ~15us at stack
# depth 60 against ~0.1us for the wrapper itself. Non-reentrant checkpointing
# latches on one of its first calls (layer 0 is already in a region) so never
# spends this; a run without it would pay the walk on every call forever, so
# cap the loss at ~1ms per step. Reset at each step boundary, and by a latch.
_CHECKPOINT_PROBE_MISSES = 0
_CHECKPOINT_PROBE_MISS_BUDGET = 64


def _probe_walk():
    """The walk, on this step's remaining budget. False once it is spent."""
    global _CHECKPOINT_PROBE_MISSES
    if _CHECKPOINT_PROBE_MISSES >= _CHECKPOINT_PROBE_MISS_BUDGET: return False
    if _walk_for_checkpoint_frame():
        _CHECKPOINT_PROBE_MISSES = 0
        return True
    _CHECKPOINT_PROBE_MISSES += 1
    return False


def _note_packed_under_checkpoint():
    """Cheap probe, run per compiled call: the hook accessor only, no frame walk.

    Skipped while Dynamo is tracing: this sits in the wrapper body, so a nested
    compiled region traces it, and the accessor is a pybind builtin Dynamo
    refuses to enter ("Attempted to call function marked as skipped
    torch._C._autograd...._top_saved_tensors_default_hooks"), which under
    `fullgraph = True` is fatal rather than a graph break -- it killed
    Gemma4_(E2B)-Vision at cell 15. Nothing is lost: the answer is meaningless
    mid-trace, and the same wrapper is entered from eager on the call that packs.
    """
    global _PACKED_COMPILED_IN_CHECKPOINT
    if _PACKED_COMPILED_IN_CHECKPOINT or _dynamo_is_tracing():
        return
    # Inference mode, NOT `is_grad_enabled()`. Grad-off looks like the exact
    # test -- nothing packed, no backward owed -- but `autograd.Function.forward`
    # runs with grad DISABLED and Unsloth's gradient checkpointing IS a custom
    # Function, so every patched kernel inside a checkpointed forward sees grad
    # off; gating on it skipped the probe in the one place it must fire and sent
    # Gemma4_(E2B)-Vision back to aborting on the checkpoint assert. An autograd
    # Function does not enter inference mode, and generation runs under it, so
    # this keeps the win without the hole.
    if torch.is_inference_mode_enabled():
        return
    top = _saved_tensor_hook_accessor()
    if top is None:
        # torch < 2.8 has no accessor, so ask the frames. Latching regardless
        # marked EVERY call on 2.4-2.7 as packed, making `_give_up` rethrow -- the
        # one path this wrapper exists to avoid. Dropping the observation was
        # wrong the other way: the give-up walk only sees a region open around
        # the FAILING call, so a layer that packed compiled and returned got
        # latched eager by a later wrapper exhausting the budget outside any
        # region. Self-limiting: the flag latches on the first hit.
        if _probe_walk(): _PACKED_COMPILED_IN_CHECKPOINT = True
        return
    try:
        hooks = top(True)                       # ignore_is_tracing
    except Exception:
        _PACKED_COMPILED_IN_CHECKPOINT = True
        return
    if not hooks:
        return                                  # no region open, definitively
    if _is_checkpoint_pack_hook(hooks[0]):
        _PACKED_COMPILED_IN_CHECKPOINT = True
        return
    # Only the TOP of the stack is reported, and a user's own
    # `saved_tensors_hooks` / `save_on_cpu` entered inside the region sits above
    # ours, so an unrecognised hook is "cannot tell from here", not "no region":
    # ask the frames, as `_in_non_reentrant_checkpoint` does. Cheap enough here,
    # reached only under a foreign hook and latching on the first hit.
    if _probe_walk(): _PACKED_COMPILED_IN_CHECKPOINT = True


def _wants_hard_recompile_failure():
    """Did the user ask Dynamo to make cache exhaustion fatal?

    torch raises FailOnRecompileLimitHit from two branches with the same class
    -- this flag, and fullgraph=True -- so the exception cannot tell them
    apart. Falling back to eager is right for the second and wrong for the
    first, whose whole purpose is to stop the run. Read the flag instead.
    Renamed in 2.7 (fail_on_cache_limit_hit -> fail_on_recompile_limit_hit)
    with the old name kept as an alias, so check both to cover 2.6.
    """
    try:
        import torch._dynamo.config as _config
    except Exception:
        return False
    return bool(getattr(_config, "fail_on_recompile_limit_hit", False) or
                getattr(_config, "fail_on_cache_limit_hit", False))


def _disabled_hook_graph_break_error():
    """Dynamo's graph-break exception, if this torch has one."""
    try:
        import torch._dynamo.exc as _exc
    except Exception:
        return ()
    _e = getattr(_exc, "Unsupported", None)
    if isinstance(_e, type) and issubclass(_e, BaseException):
        return (_e,)
    return ()


def _is_recompile_limit_unsupported(exc):
    """Cache exhaustion reported as a plain graph break.

    torch 2.4 has no exception class for it: `convert_frame` ends the
    cache-limit branch in `unimplemented(f"{limit_type} reached")`, which
    raises `Unsupported`, so `_recompile_limit_errors()` is empty there and
    the message is all that is left to match on. 2.5 added `CacheLimitExceeded`
    and 2.6 `RecompileLimitExceeded`, but both still fall through to the same
    `unimplemented` when `skip_code_recursive_on_cache_limit_hit` is off.

    `limit_type` is `cache_size_limit` or `accumulated_cache_size_limit`,
    renamed to `recompile_limit` / `accumulated_recompile_limit` in torch 2.7.
    """
    try:
        text = str(exc)
    except Exception:
        return False
    return ("cache_size_limit reached" in text
            or "recompile_limit reached" in text)


# The complete set of texts Dynamo uses to refuse a disabled callable, one per
# wording, each a literal from the single `_torchdynamo_disable` branch of
# `UserFunctionVariable.call_function` in `torch/_dynamo/variables/functions.py`.
# Nothing else in Dynamo emits them, which is what keeps the match narrow.
#
#   torch 2.4 to 2.6  unimplemented(f"call torch._dynamo.disable() wrapped
#                     function {self.value}")   (2.4.0 L604, 2.5.1 L655, 2.6.0 L620)
#   torch 2.7+        unimplemented_v2(gb_type="Skip calling
#                     `torch.compiler.disable()`d function", ...)  (2.7.0 L1173)
#
# Both spellings stay listed: pyproject allows torch>=2.4, and the older one is
# all those releases ever emit.
_DISABLED_HOOK_SIGNATURES = (
    ("Skip calling", "torch.compiler.disable"),
    ("torch._dynamo.disable() wrapped function",),
)


def _is_our_own_disabled_hook(exc):
    """Did we break our own graph with our own `torch.compiler.disable`?

    Our `torch.compiler.disable`d requires_grad hooks can be invoked from
    inside a `fullgraph = True` region, and Dynamo refuses with

        Unsupported: Skip calling `torch.compiler.disable()`d function

    or, before torch 2.7 renamed it,

        Unsupported: call torch._dynamo.disable() wrapped function <...>

    Matched narrowly on those signatures: any other graph break under fullgraph
    must still raise, since those point at real problems.
    """
    try:
        text = str(exc)
    except Exception:
        return False
    return any(
        all(part in text for part in signature)
        for signature in _DISABLED_HOOK_SIGNATURES
    )


# torch 2.7 renamed both recompile budgets, keeping the old names as aliases,
# so bumping whichever of the pair the installed torch actually reads is the
# only portable way to buy the current step a few more compilations.
_RECOMPILE_LIMIT_NAMES = (
    ("recompile_limit", "cache_size_limit"),
    ("accumulated_recompile_limit", "accumulated_cache_size_limit"),
)

# Headroom per bump, and bumps one wrapper may ask for before going eager.
# Bounded on purpose: the limit exists to stop unbounded compilation.
_RECOMPILE_LIMIT_BUMP = 16
_MAX_RECOMPILE_LIMIT_BUMPS = 4

# The budgets live on process-global torch._dynamo.config, so the per-wrapper
# cap bounds nothing on its own: N wrappers, or several models in one process,
# each spend their 4 bumps and the global limit ends up hundreds higher,
# permanently, for every unrelated compiled function. Cap the total and put the
# originals back once the debt is settled.
_MAX_TOTAL_RECOMPILE_LIMIT_BUMPS = 8
_GLOBAL_BUMPS = 0
_ORIGINAL_RECOMPILE_LIMITS = {}
# name -> {bumped value: value it was bumped from}. Restore only a name still
# holding a value we wrote, back to the baseline THAT value came from: bumps nest
# inside `torch._dynamo.config.patch`, and the two orderings need opposite
# answers, which one recorded original cannot give.
#   bump 1024->1040, patch, bump inside, exit  -> dynamo hands back 1040, owes 1024
#   patch(2), bump 2->18, exit, bump 8->24     -> live 24 owes 8, NOT the scoped 2
_BUMPED_RECOMPILE_LIMITS = {}


def _bump_recompile_limits(extra = _RECOMPILE_LIMIT_BUMP):
    """Raise both recompile budgets. False if this torch exposes neither, or if
    the process-wide bump budget is spent."""
    global _GLOBAL_BUMPS
    try:
        import torch._dynamo.config as _config
    except Exception:
        return False
    # Measured here, not carried over from the last restore: that may have run
    # under a `config.patch` hiding our value, where the count reads 0, while
    # the borrow below happens outside it against the real raised limit.
    _GLOBAL_BUMPS = max((_live_bump_depth(_config, n)
                         for n in _BUMPED_RECOMPILE_LIMITS), default = 0)
    if _GLOBAL_BUMPS >= _MAX_TOTAL_RECOMPILE_LIMIT_BUMPS:
        return False
    bumped = False
    for names in _RECOMPILE_LIMIT_NAMES:
        for name in names:
            current = getattr(_config, name, None)
            if isinstance(current, int) and not isinstance(current, bool):
                try:
                    setattr(_config, name, current + extra)
                except Exception:
                    continue
                # Key the baseline by the value we wrote: only the live value
                # at restore time says which bump is still standing. A stack per
                # value, not one entry, because a scoped patch can repeat a
                # baseline -- bump 8->24, patch to 8, bump to 24 again -- and one
                # entry collapsed the two debts.
                _BUMPED_RECOMPILE_LIMITS.setdefault(name, {}).setdefault(
                    current + extra, []).append(current)
                _ORIGINAL_RECOMPILE_LIMITS.setdefault(name, current)
                bumped = True
                # Only the name this torch really reads; the alias follows.
                break
    if bumped:
        _GLOBAL_BUMPS += 1
    return bumped


def _restore_recompile_limits_if_idle():
    """Restore only when no live wrapper still relies on the extra headroom.

    A wrapper that borrowed budget and has not gone eager is still compiling
    against the raised limit; taking it away mid-flight would push it into the
    fallback we just paid to avoid."""
    for ref in _EAGER_FALLBACK_WRAPPERS:
        w = ref()
        if w is None:
            continue
        st = w._unsloth_fallback_state
        if st.get("bumps") and not st.get("eager"):
            return 0
    return _restore_recompile_limits()


def _restore_recompile_limits():
    """Put the budgets back. Returns how many names were restored.

    Called once every borrower has gone eager, so the process stops carrying a
    raised limit around."""
    global _GLOBAL_BUMPS, _PACKED_COMPILED_IN_CHECKPOINT, _CHECKPOINT_PROBE_MISSES
    # A new step packs its own activations; last step's are long since freed.
    _PACKED_COMPILED_IN_CHECKPOINT = False
    _CHECKPOINT_PROBE_MISSES = 0
    if not _ORIGINAL_RECOMPILE_LIMITS:
        return 0
    try:
        import torch._dynamo.config as _config
    except Exception:
        return 0
    restored = 0
    for name in list(_ORIGINAL_RECOMPILE_LIMITS):
        try:
            live = getattr(_config, name, None)
            baselines = _BUMPED_RECOMPILE_LIMITS.get(name, {})
            if live not in baselines:
                # Not our value: someone else's write, or a live `config.patch`
                # hiding ours, indistinguishably. Dropping the claim loses the
                # original when that patch exits and hands our bump back, so
                # keep it and settle at a later boundary.
                continue
            # Follow the chain back: successive bumps record 1040->1024,
            # 1056->1040 and so on, so one hop would hand back the previous bump
            # rather than the starting value. `chain` also stops a cycle if a
            # bump ever lands on an earlier one.
            original, chain = baselines[live][-1], [live]
            while original in baselines and original not in chain:
                chain.append(original)
                original = baselines[original][-1]
            setattr(_config, name, original)
            restored += 1
        except Exception:
            continue
        # Only the branch just unwound: dropping the whole map also threw away
        # branches a live `config.patch` still hides -- bump 8->24, patch to 2,
        # bump 2->18, restore killed the 24->8 debt, and the patch exit then
        # resurrected 24 permanently.
        for value in chain:
            stack = baselines.get(value)
            if stack: stack.pop()
            if not stack: baselines.pop(value, None)
        if not baselines:
            _ORIGINAL_RECOMPILE_LIMITS.pop(name, None)
            _BUMPED_RECOMPILE_LIMITS.pop(name, None)
    # Repay only the debts actually settled. Zeroing while a scoped patch still
    # hid a live bump handed the next wrapper the whole global allowance again,
    # so repeated patches walked straight past the total cap.
    _GLOBAL_BUMPS = max((_live_bump_depth(_config, n)
                         for n in _BUMPED_RECOMPILE_LIMITS), default = 0)
    return restored


def _live_bump_depth(_config, name):
    """How many of our bumps are IN EFFECT on this name right now.

    Counting every recorded branch was wrong both ways. A branch a completed
    `config.patch` rolled back is gone from the limit, yet kept consuming the
    process-wide allowance forever, so repeated scoped patches starved every
    wrapper of budget nobody was using. A branch hidden by a LIVE patch is
    equally not in effect: the limit is the patched value, borrowing is bounded
    by it, and the debt is counted again when the patch exits and hands it back.

    Walking back from the live value answers both, along the same chain the
    restore follows, so the two cannot disagree."""
    live = getattr(_config, name, None)
    baselines = _BUMPED_RECOMPILE_LIMITS.get(name, {})
    depth, seen = 0, set()
    while live in baselines and live not in seen:
        seen.add(live)
        depth += 1
        live = baselines[live][-1]
    return depth


def compile_with_eager_fallback(func, label, fullgraph = True, dynamic = True):
    """`torch_compile` a standalone kernel, keeping `patch_function`'s eager fallback.

    A bare `@torch_compile(fullgraph = True)` skips
    `_fall_back_to_eager_on_recompile_limit` (only `patch_function` applies it), so
    cache exhaustion aborts training instead of latching to eager.
    """
    # The RAW compile: `torch_compile` now routes fullgraph regions through the
    # fallback itself, so wrapping its result again would leave the inner
    # wrapper swallowing the exhaustion under a label nobody looks up, and the
    # outer one returned here never latching.
    from .common import _raw_torch_compile, unwrap_already_compiled
    # No-op for the plain module-level functions every caller passes today, and
    # the thing that keeps torch 2.11's bare
    # `assert not hasattr(compile_wrapper, "get_compiler_config")` out of reach
    # if one ever passes a callable that has already been through here.
    func = unwrap_already_compiled(func)
    compiled = _raw_torch_compile(fullgraph = fullgraph, dynamic = dynamic)(func)
    # Without fullgraph Dynamo already falls back by itself.
    if not fullgraph:
        return compiled
    return _fall_back_to_eager_on_recompile_limit(compiled, func, label)


# Call sites that have already given up, by label. Outlives the wrappers, which
# the registry holds only weakly.
_LATCHED_EAGER_LABELS: set = set()

# Call sites that deferred their switch to the next step boundary. Same reason:
# a wrapper built inside a forward is unreachable before the boundary arrives,
# so the deferral has to be recorded somewhere that survives it.
_PENDING_EAGER_LABELS: set = set()

# The same latches, but only since the last settle. `_LATCHED_EAGER_LABELS` is
# deliberately permanent, which makes it the wrong thing to read as EVIDENCE
# that something fell back just now: train two models in one process and the
# first one's labels answer for the second. Cleared at every settle, so this
# one says "in this step", which is the question being asked.
_RECENT_EAGER_LABELS: set = set()


def _fall_back_to_eager_on_recompile_limit(compiled_func, eager_func, label):
    """Run eager instead of dying when the recompile cache is exhausted.

    `fullgraph = True` makes Dynamo raise on cache exhaustion rather than fall
    back, turning a performance problem into a hard training failure:

        FailOnRecompileLimitHit: recompile_limit reached with fullgraph=True

    Only cache exhaustion and our own disabled-hook graph break are caught, so
    any other graph break under fullgraph still raises exactly as before.

    The fallback LATCHES. Do not make it retry per call. Non-reentrant
    activation checkpointing recomputes each packed forward during backward and
    compares the saved intermediates, and a compiled pack recomputed eagerly
    aborts the backward with "Something went unexpectedly wrong in activation
    checkpoint". Per-call retry does not fix that: the pack and the recompute
    run under different guards (grad mode differs, and the recompute is inside
    backward), so the compiler can succeed for one and raise for the other in
    either direction. Latching makes every call after the first failure eager,
    so every later pack and recompute agree.

    That leaves one mixed step, the one during which the latch flips. See
    `force_eager_fallback` below, which unsloth uses to close it.
    """
    errors = _recompile_limit_errors()
    graph_break_errors = _disabled_hook_graph_break_error()
    if not errors and not graph_break_errors:
        return compiled_func

    # Warn once. The condition repeats every call, and the log should not.
    # Eager/pending come from the LABEL, not the object: a wrapper built inside
    # a forward (GRPO's `accumulate_chunk` closes over per-call accumulators, so
    # it cannot be hoisted) dies with that forward and the registry holds it
    # weakly, so latching it bought nothing -- the next step compiled a fresh
    # one, borrowed again, and the bounded transition to eager never happened.
    state = {"warned": False, "eager": label in _LATCHED_EAGER_LABELS,
             "pending_eager": label in _PENDING_EAGER_LABELS, "bumps": 0}

    def _warn(message):
        if not state["warned"]:
            state["warned"] = True
            logger.warning(message)

    # Nothing is a valid return value, so the retry needs its own sentinel.
    _NO_RESULT = object()

    def _latch_all_to_eager():
        """Give up on the budget, taking the other wrappers in the same crisis.

        Switching only this one leaves the step with some regions compiled and
        some eager -- the mismatch this path exists to avoid, since one
        checkpointed region routinely spans several patched functions.

        Only borrowers come along: a wrapper that never bumped was never in
        trouble and never changed mode, so knocking it eager would cost
        compilation for nothing and break the per-wrapper latch.

        Anything already packed compiled earlier in this step is still at risk.
        Buying budget first is what makes that rare; this is the last resort."""
        state["eager"] = True
        state["pending_eager"] = False
        _LATCHED_EAGER_LABELS.add(label)
        _RECENT_EAGER_LABELS.add(label)
        for ref in _EAGER_FALLBACK_WRAPPERS:
            w = ref()
            if w is None:
                continue
            st = w._unsloth_fallback_state
            if not st.get("bumps"):
                continue
            st["eager"] = True
            st["pending_eager"] = False
            _wl = getattr(w, "_unsloth_fallback_label", None)
            if _wl is not None:
                _LATCHED_EAGER_LABELS.add(_wl)
                _RECENT_EAGER_LABELS.add(_wl)
        _restore_recompile_limits_if_idle()

    def _release_borrowed_budget():
        """Hand back the bump a retry took out but never got to use.

        The call died of the model's own trouble, not the compiler's, so the
        wrapper is not compiling against that headroom. Leaving the bump counted
        keeps it looking like a live borrower forever -- neither eager nor
        pending, so no boundary settles it, `_restore_recompile_limits_if_idle`
        keeps declining, and the raised process-global limit and the spent
        shared allowance outlive the run."""
        if state["bumps"] > 0:
            state["bumps"] -= 1
        # Declines while any other wrapper still relies on the headroom.
        _restore_recompile_limits_if_idle()

    def _retry_with_more_budget(args, kwargs):
        """Finish THIS call the way it started, if we can.

        Switching to eager here is what breaks non-reentrant activation
        checkpointing: the pack and the recompute of one region then run in
        different compile modes, and torch aborts the backward with "Something
        went unexpectedly wrong in activation checkpoint" or the neighbouring
        "A different number of tensors was saved" / "different metadata".
        Buying a little more budget keeps the call compiled and defers the
        switch to the next step boundary, where nothing is half-packed.
        Bounded, so a model that recompiles forever still ends up eager."""
        if state["bumps"] >= _MAX_RECOMPILE_LIMIT_BUMPS:
            return _NO_RESULT
        if not _bump_recompile_limits():
            return _NO_RESULT
        state["bumps"] += 1
        try:
            result = compiled_func(*args, **kwargs)
        except errors:
            # Keep the bump: the caller latches everything that borrowed to
            # eager next, and that hand-back is what restores the budgets.
            return _NO_RESULT
        except graph_break_errors as e:
            if _is_recompile_limit_unsupported(e) or _is_our_own_disabled_hook(e):
                return _NO_RESULT
            _release_borrowed_budget()
            raise
        except _checkpoint_early_stop_errors() as e:
            # Not a failure: with early-stop on, checkpoint's recompute hook
            # raises this once every needed tensor is back and the machinery
            # swallows it as a successful recomputation. Releasing the bump and
            # returning left the wrapper compiled with its counters reset, so
            # every new guard variant could borrow again and walk past both the
            # per-wrapper and global caps. The retry DID finish; treat it as one
            # and still let the signal out.
            state["pending_eager"] = True
            _PENDING_EAGER_LABELS.add(label)
            _warn(
                f"Unsloth: torch.compile ran out of recompilation cache for "
                f"{label} during checkpoint recomputation; switching to eager "
                f"at the next step. Training is unaffected apart from speed."
            )
            raise
        except BaseException:
            # A real failure of the model, not of the compiler: falling through
            # to eager would run the call twice, reapplying any mutation it
            # already made, and bury the error. The caller may catch it and
            # carry on (skipping a bad batch, say), so hand the budget back
            # rather than strand it on a call that never completed.
            _release_borrowed_budget()
            raise
        state["pending_eager"] = True
        _PENDING_EAGER_LABELS.add(label)
        # Deferring is for checkpointed training, where flipping mid-step makes
        # the pack and the recompute of one region disagree. Outside a region
        # nothing is half-packed, and no boundary is coming either:
        # `apply_pending_eager_fallbacks` runs from the trainer's step hook, so
        # a compiled inference function (GPT-OSS's module-level `_torch_compile`
        # ones) would hold the process-global limits raised for the rest of the
        # process and never make the switch it just promised. Settle it here
        # instead, but only on a definite False: None means torch could not say,
        # and guessing "no region" is the answer that corrupts a backward.
        if _in_non_reentrant_checkpoint() is False and \
            not _PACKED_COMPILED_IN_CHECKPOINT:
            # THIS wrapper only. Taking every live borrower along is what a step
            # boundary wants, not this: one inference function out of cache
            # would knock every patched kernel in the process eager mid-step,
            # and nothing spans this call the way a checkpointed region does.
            state["eager"] = True
            state["pending_eager"] = False
            _PENDING_EAGER_LABELS.discard(label)
            _LATCHED_EAGER_LABELS.add(label)
            _RECENT_EAGER_LABELS.add(label)
            while state["bumps"] > 0:
                _release_borrowed_budget()
            _warn(
                f"Unsloth: torch.compile ran out of recompilation cache for "
                f"{label}; running it eagerly from here. Speed is the only "
                f"thing affected."
            )
            return result
        _warn(
            f"Unsloth: torch.compile ran out of recompilation cache for "
            f"{label}; finishing this step compiled and switching to eager at "
            f"the next step. Training is unaffected apart from speed."
        )
        return result

    def _give_up(e, args, kwargs):
        """No budget left. Latch, then decide whether eager is even survivable.

        Inside a non-reentrant checkpoint region it is not: whatever this step
        packed compiled gets recomputed eagerly, and the backward either aborts
        or, when the shapes line up, hands back wrong gradients (torch compares
        only shape/dtype/device). Nothing is left to keep this call compiled
        with, so end the step instead. Every borrower is eager by now, so the
        caller's retry is consistent -- the contract `force_eager_fallback`
        offers reactively."""
        # Read BEFORE latching, which hands the budget back, and that settlement
        # is a step boundary that clears the marker. Not just "in a region now":
        # anything packed compiled this step owes a backward that would now
        # recompute eagerly.
        packed = _in_non_reentrant_checkpoint() or _PACKED_COMPILED_IN_CHECKPOINT
        _latch_all_to_eager()
        _warn(
            f"Unsloth: torch.compile ran out of recompilation cache for "
            f"{label}; running it eagerly from here. Training is "
            f"unaffected apart from speed. ({type(e).__name__})"
        )
        if packed:
            global _RAISED_INSIDE_CHECKPOINT, _CHECKPOINT_SETTLE_ATTEMPTS
            _RAISED_INSIDE_CHECKPOINT = True
            _CHECKPOINT_SETTLE_ATTEMPTS = 0
            raise e
        return eager_func(*args, **kwargs)

    @functools.wraps(eager_func)
    def wrapper(*args, **kwargs):
        if state["eager"]:
            return eager_func(*args, **kwargs)
        try:
            _note_packed_under_checkpoint()
            return compiled_func(*args, **kwargs)
        except errors as e:
            if _wants_hard_recompile_failure():
                raise
            result = _retry_with_more_budget(args, kwargs)
            if result is not _NO_RESULT:
                return result
            return _give_up(e, args, kwargs)
        except graph_break_errors as e:
            # Cache exhaustion on a torch that has no exception class for it
            # (2.4), then our own `torch.compiler.disable` hook. Anything else
            # is a real graph break and must keep raising.
            if _is_recompile_limit_unsupported(e):
                result = _retry_with_more_budget(args, kwargs)
                if result is not _NO_RESULT:
                    return result
                return _give_up(e, args, kwargs)
            if not _is_our_own_disabled_hook(e):
                raise
            state["eager"] = True
            _warn(
                f"Unsloth: torch.compile hit one of Unsloth's own "
                f"`torch.compiler.disable`d gradient-checkpointing hooks "
                f"inside {label}; running it eagerly from here. Training is "
                f"unaffected apart from speed."
            )
            return eager_func(*args, **kwargs)

    # Keep the compiled callable reachable for anything that unwraps it, and
    # keep `get_compiler_config` present so the unwrap check below still sees
    # a compiled function and reaches `__wrapped__` (the eager original).
    wrapper._unsloth_compiled_func = compiled_func
    # Anything asking "is this compiled?" (torch included) looks here, and the
    # wrapper is what callers now hold, so forward it rather than let the
    # fallback look like it un-compiled the region.
    _orig = getattr(compiled_func, "_torchdynamo_orig_callable", None)
    if _orig is not None:
        wrapper._torchdynamo_orig_callable = _orig
    _gcc = getattr(compiled_func, "get_compiler_config", None)
    if _gcc is not None:
        wrapper.get_compiler_config = _gcc

    wrapper._unsloth_fallback_state = state
    wrapper._unsloth_fallback_label = label
    _EAGER_FALLBACK_WRAPPERS.append(weakref.ref(wrapper))
    # GRPO rebuilds and re-wraps `accumulate_chunk` inside every backward, so the
    # registry gained an entry per step forever and every scan above walked them
    # all: quadratic over a long run. Compacting once the list has doubled is
    # amortised O(1) per append and keeps scans proportional to live wrappers.
    global _EAGER_FALLBACK_PRUNE_AT
    if len(_EAGER_FALLBACK_WRAPPERS) >= _EAGER_FALLBACK_PRUNE_AT:
        _EAGER_FALLBACK_WRAPPERS[:] = [
            r for r in _EAGER_FALLBACK_WRAPPERS if r() is not None]
        _EAGER_FALLBACK_PRUNE_AT = max(64, 2 * len(_EAGER_FALLBACK_WRAPPERS))
    return wrapper


# Every wrapper built above, weakly, so a patched-then-discarded model does not
# keep its functions alive. Weak refs also mean the registry cannot report a
# switch for something nobody is calling any more.
_EAGER_FALLBACK_WRAPPERS: list = []
_EAGER_FALLBACK_PRUNE_AT: int = 64


def eager_fallback_state() -> dict[str, bool]:
    """{label: already fell back} for every live wrapper. For tests and logs."""
    out = {}
    for ref in _EAGER_FALLBACK_WRAPPERS:
        w = ref()
        if w is not None:
            label = w._unsloth_fallback_label
            # OR, not assign: labels can collide, and a later False would hide an
            # earlier True, reporting compiled for an already-eager path.
            out[label] = out.get(label, False) or bool(w._unsloth_fallback_state["eager"])
    return out


_RAISED_INSIDE_CHECKPOINT = False
# Retries are bounded: one collection per step forever would cost more than the
# stale hooks do, and torch < 2.8 cannot report them at all.
_MAX_CHECKPOINT_SETTLE_ATTEMPTS = 8
_CHECKPOINT_SETTLE_ATTEMPTS = 0


def _checkpoint_hooks_left_installed():
    """Are checkpoint's hooks still on top? `_UNKNOWN` when torch cannot say.

    Before 2.8 there is no accessor, and answering False there let settlement
    give up after one failed collection -- exactly the case needing the retry,
    since a traceback held by `except ... as exc` roots the generator until the
    caller drops it."""
    top = _saved_tensor_hook_accessor()
    if top is None: return _UNKNOWN
    try:
        hooks = top(True)
    except Exception:
        return _UNKNOWN
    return bool(hooks) and _is_checkpoint_pack_hook(hooks[0])


def _settle_abandoned_checkpoint_generator():
    """Finalise the checkpoint generator our give-up raise left mid-flight.

    `_checkpoint_without_reentrant_generator` holds `with _checkpoint_hook(...)`
    open across its yield, so an exception escaping the region abandons the
    generator with its saved-tensor hooks installed and every later region sees
    them. Refcounting usually finalises it at once; raising through the compiled
    frames leaves a cycle, so it waits for a collection. One collect here makes
    the promised retry deterministic. Only after a give-up raise, which the bump
    caps bound to a handful per process."""
    global _RAISED_INSIDE_CHECKPOINT, _CHECKPOINT_SETTLE_ATTEMPTS
    if not _RAISED_INSIDE_CHECKPOINT: return False
    import gc
    gc.collect()
    _CHECKPOINT_SETTLE_ATTEMPTS += 1
    # A caller retrying from inside `except ... as exc` roots the traceback, so
    # the generator survives every collection: stay pending and try again at the
    # next boundary. `_UNKNOWN` counts as pending, since before 2.8 nothing can
    # be inspected and that is precisely when the retry is needed.
    _left = _checkpoint_hooks_left_installed()
    if (_left is _UNKNOWN or _left) and \
        _CHECKPOINT_SETTLE_ATTEMPTS < _MAX_CHECKPOINT_SETTLE_ATTEMPTS:
        return False
    _RAISED_INSIDE_CHECKPOINT = False
    _CHECKPOINT_SETTLE_ATTEMPTS = 0
    return True


def torch_compile_with_fallback(fullgraph = False, **compile_kwargs):
    """`torch.compile` that survives cache exhaustion under `fullgraph = True`.

    `patch_function` already routes its own `fullgraph = True` compiles through
    `_fall_back_to_eager_on_recompile_limit`, but the generated modules in
    `unsloth_compiled_cache` decorate directly and never reach it, so those
    regions kept the hard failure the wrapper exists to remove. Gemma4 has ten,
    and the vision tower drives `Gemma4RMSNorm_forward` far past the budget:

        FailOnRecompileLimitHit: Hard failure due to fullgraph=True

    out of `unsloth_compiled_module_gemma4.py`, ending training at step 0.
    Reproducible on any GPU by lowering `recompile_limit`; a T4 gets there
    on its own.

    `fullgraph = False` is returned untouched: Dynamo already falls back by
    itself there, so wrapping would add a layer that can never fire.

    This is a BARE decorator: `compiler.py` writes it into every generated
    `unsloth_compiled_cache` module and `rl_replacements.py` applies it directly,
    so it reaches `torch.compile` without passing through `_compile_or_fall_back`
    and needs its own `unwrap_already_compiled`. What it returns is a
    `functools.wraps` copy of a compiled function, which is precisely the shape
    torch 2.11+ refuses to compile again.
    """
    def _decorate(func):
        func = unwrap_already_compiled(func)
        compiled = torch.compile(func, fullgraph = fullgraph, **compile_kwargs)
        if not fullgraph:
            return compiled
        return _fall_back_to_eager_on_recompile_limit(
            compiled, func, getattr(func, "__qualname__", None) or repr(func),
        )
    return _decorate


def apply_pending_eager_fallbacks() -> int:
    """Latch every wrapper that deferred its switch. Returns how many flipped.

    Call this at a training-step boundary, where no activation-checkpoint region
    is half-packed. A wrapper out of recompile budget mid-step stayed compiled
    for the rest of that step (see `_fall_back_to_eager_on_recompile_limit`) so
    the pack and the recompute of every checkpointed region agree; this settles
    that debt.

    Flipping one wrapper flips all of them, for the same reason
    `force_eager_fallback` does: one checkpointed region usually spans several
    patched functions, and leaving the rest compiled just moves the mismatch.

    Safe to call on every step. Nothing pending means nothing happens.
    """
    # A new step: whatever was packed compiled belonged to the last one. Only
    # `_restore_recompile_limits` cleared the marker, which an ordinary
    # successful step never reaches, so it stayed true for the rest of the run
    # and made `_give_up` re-raise for a later call nowhere near a checkpoint.
    # Cleared first, before any early return below.
    global _PACKED_COMPILED_IN_CHECKPOINT, _CHECKPOINT_PROBE_MISSES
    _PACKED_COMPILED_IN_CHECKPOINT = False
    _CHECKPOINT_PROBE_MISSES = 0             # a new step, a new probe budget
    _settled = _settle_abandoned_checkpoint_generator()
    if _RAISED_INSIDE_CHECKPOINT and not _settled:
        # Still rooted, so its saved-tensor hooks are still on the stack and the
        # next region would pack under them. Almost always a caller retrying
        # from inside `except ... as exc`, whose traceback holds the frames
        # alive. Say so rather than let the flip read as a clean boundary.
        logger.warning_once(
            "Unsloth: a checkpoint region left by the compile-mode fallback has "
            "not been finalised yet, so its saved-tensor hooks are still "
            "installed. Retry the step outside the `except` block (or clear the "
            "exception first) so the region can close."
        )
    live = [w for w in (ref() for ref in _EAGER_FALLBACK_WRAPPERS)
            if w is not None]
    # By label as well as by object: a wrapper built inside a forward is already
    # collected by the time the boundary arrives, so asking only the live ones
    # answered "nothing pending" and the next step compiled it again, and the
    # bounded transition never happened for GRPO's `accumulate_chunk`.
    if not _PENDING_EAGER_LABELS and \
        not any(w._unsloth_fallback_state.get("pending_eager") for w in live):
        # Cleared here too, not just on the settle path below: a wrapper giving
        # up OUTSIDE a checkpoint records its label without a pending entry, so
        # every later boundary took this early return and the label never
        # expired. A genuine checkpoint failure in a later step, or in a second
        # model in the same process, then read it as evidence of a compile-mode
        # flip, latched healthy wrappers eager and asked for a retry instead of
        # letting the real error through.
        _RECENT_EAGER_LABELS.clear()
        # Nothing to flip, but a borrower that bumped and was then collected
        # (training aborted, or the patched object replaced) would otherwise
        # leave the process-wide limit raised and its allowance spent forever.
        # The helper declines while any live wrapper still needs the headroom.
        _restore_recompile_limits_if_idle()
        return 0
    flipped = 0
    for w in live:
        if not w._unsloth_fallback_state["eager"]:
            flipped += 1
        w._unsloth_fallback_state["eager"] = True
        w._unsloth_fallback_state["pending_eager"] = False
    # The deferral is settled, so those call sites stay eager however often
    # their wrappers are rebuilt.
    _LATCHED_EAGER_LABELS.update(_PENDING_EAGER_LABELS)
    _PENDING_EAGER_LABELS.clear()
    _RECENT_EAGER_LABELS.clear()
    # Everything that borrowed headroom is eager now, so hand the budget back
    # rather than leaving the process permanently raised.
    _restore_recompile_limits()
    return flipped


def force_eager_fallback(only_if_already_triggered: bool = True) -> int:
    """Latch every wrapper to eager. Returns how many are eager afterwards.

    For unsloth to call when a backward dies inside activation checkpointing,
    so the retry of that step is consistent. The limit is hit per function, so
    one fallback can strand a checkpointed region spanning several; switching
    them together removes the whole class of mismatch.

    Returns a count of call sites that are eager, not of changes: the wrapper
    that caused the trouble has already latched itself, so "how many changed"
    would be 0 in exactly the case that matters.

    `only_if_already_triggered` refuses to switch off compilation for a model
    that never fell back. A caller that gets 0 has learned the checkpoint
    assertion was not a compile-mode flip, and should re-raise rather than
    retry.
    """
    live = [w for w in (ref() for ref in _EAGER_FALLBACK_WRAPPERS)
            if w is not None]
    # The label sets count too: a wrapper built inside a forward (GRPO's
    # `accumulate_chunk`) is collected before backward, so its record of having
    # failed survives only there, and asking the live ones alone returned 0 --
    # the caller then re-raised the very failure this exists to retry past.
    if only_if_already_triggered and not (
            _RECENT_EAGER_LABELS or _PENDING_EAGER_LABELS or any(
            w._unsloth_fallback_state["eager"]
            or w._unsloth_fallback_state.get("pending_eager")
            for w in live)):
        return 0
    for w in live:
        w._unsloth_fallback_state["eager"] = True
        w._unsloth_fallback_state["pending_eager"] = False
        _label = getattr(w, "_unsloth_fallback_label", None)
        if _label is not None: _LATCHED_EAGER_LABELS.add(_label)
    # Settle deferrals too, or the per-call wrapper the retry rebuilds reads a
    # still-pending label and compiles again.
    _LATCHED_EAGER_LABELS.update(_PENDING_EAGER_LABELS)
    _PENDING_EAGER_LABELS.clear()
    _RECENT_EAGER_LABELS.clear()
    _restore_recompile_limits()
    # Latched labels, not live objects: a call site with no wrapper alive right
    # now is still eager, and returning 0 would read as "nothing fell back".
    return max(len(live), len(_LATCHED_EAGER_LABELS))


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
        # Unwrap already-compiled functions. The shared helper rather than a
        # bare `.__wrapped__`: a carrier without one (an `OptimizedModule`, say)
        # raised `AttributeError` here, a bound method's `__wrapped__` is the
        # UNBOUND original and silently dropped the receiver, and one hop is not
        # always enough to reach the eager function.
        new_func = unwrap_already_compiled(new_func)
        original_func = unwrap_already_compiled(original_func)
        _eager_func = new_func
        # The compile is guarded the way `_compile_or_fall_back` guards its own,
        # and for the same reason: a patch that cannot be compiled is a
        # performance problem, and the eager function is still correct, so it
        # must not end the model load. Reachable on torch 2.11+ for whatever
        # `unwrap_already_compiled` deliberately leaves alone -- a compiled BOUND
        # method keeps its receiver, and torch refuses to compile it again.
        try:
            new_func = torch.compile(
                new_func,
                fullgraph = fullgraph,
                dynamic = dynamic,
                options = torch_compile_options,
            )
        except Exception as exception:
            _label = f"{getattr(target_obj, '__name__', target_obj)}.{attr_name}"
            logger.warning(
                f"Unsloth: torch.compile refused to wrap {_label}; running it "
                f"eagerly. ({type(exception).__name__}: {exception})"
            )
            new_func = _eager_func
        else:
            if fullgraph:
                # Only fullgraph turns cache exhaustion into a raise; without it
                # Dynamo already falls back on its own.
                new_func = _fall_back_to_eager_on_recompile_limit(
                    new_func, _eager_func,
                    f"{getattr(target_obj, '__name__', target_obj)}.{attr_name}",
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
