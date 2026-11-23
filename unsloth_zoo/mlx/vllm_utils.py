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

"""vLLM on Apple Silicon through the vllm-metal platform plugin: the MLX counterpart of ``unsloth_zoo.vllm_utils``."""

import atexit
import contextlib
import importlib.util
import itertools
import os
import shutil
import tempfile
import warnings

from .peft_interop import MLX_WEIGHTS_FILE, PEFT_WEIGHTS_FILE, convert_mlx_dir_to_peft, detect_adapter_format

__all__ = [
    "VLLM_ENV",
    "vllm_metal_available",
    "use_vllm",
    "load_vllm",
    "attach_vllm_engine",
    "save_lora",
    "load_lora",
    "release_lora",
]

VLLM_ENV = "UNSLOTH_MLX_VLLM"
_INSTALL_HINT = (
    "vLLM runs on Apple Silicon through vllm-metal, which ships for Python 3.12 only: "
    "https://github.com/vllm-project/vllm-metal"
)
_LORA_IDS = itertools.count(1)
_scratch_root = None


def _installed():
    return importlib.util.find_spec("vllm") is not None and importlib.util.find_spec("vllm_metal") is not None


def _metal_platform_error():
    """None when vLLM imports and resolves vllm-metal's Metal platform, otherwise why it does not."""
    if not _installed():
        return "vLLM and vllm-metal are not installed"
    try:
        # LLM is a lazy attribute with imports of its own, so load it here too.
        from vllm import LLM  # noqa: F401
        from vllm.platforms import current_platform
    except Exception as error:
        return f"vLLM failed to import ({type(error).__name__}: {error})"
    # Also covers VLLM_PLUGINS leaving out the metal plugin.
    platform = type(current_platform)
    if not platform.__module__.startswith("vllm_metal."):
        return f"vLLM resolved the {platform.__name__} platform, not vllm-metal's Metal platform"
    return None


def vllm_metal_available():
    return _metal_platform_error() is None


def use_vllm(fast_inference):
    """Whether ``fast_inference=True`` gets a vLLM engine: vllm-metal when it works, MLX batched generation otherwise."""
    choice = os.environ.get(VLLM_ENV, "").strip()
    if choice not in ("", "0", "1"):
        raise ValueError(f"Unsloth MLX: {VLLM_ENV}={choice!r}; expected 0 (MLX generation) or 1 (vLLM).")
    if not fast_inference or choice == "0":
        return False
    error = _metal_platform_error()
    if error is None:
        return True
    if choice == "1":
        raise ImportError(f"Unsloth MLX: {VLLM_ENV}=1 needs vLLM, but {error}. {_INSTALL_HINT}")
    if _installed():
        warnings.warn(f"Unsloth MLX: fast_inference keeps MLX generation because {error}.", stacklevel=2)
    return False


def _share_weights(engine_model, model):
    """Point the engine's (lazily loaded) arrays at ``model``'s when every name, shape and dtype matches."""
    from mlx.utils import tree_flatten, tree_unflatten

    wanted = dict(tree_flatten(engine_model.parameters()))
    trained = dict(tree_flatten(model.parameters()))
    if any(name not in trained or trained[name].shape != array.shape or trained[name].dtype != array.dtype
           for name, array in wanted.items()):
        return False
    engine_model.update(tree_unflatten([(name, trained[name]) for name in wanted]))
    return True


@contextlib.contextmanager
def _in_process_engine_loads(model):
    """While an in-process engine starts, its mlx-lm model takes the training model's weights instead of a second
    copy, and its layers skip Unsloth's class-level gradient checkpointing."""
    try:
        import vllm_metal.v1.model_lifecycle as lifecycle
    except ImportError:
        yield
        return
    from .utils import exempt_from_gradient_checkpointing

    original, original_vlm = lifecycle.mlx_lm_load_model, lifecycle.mlx_vlm_load_model

    def load(model_path, lazy=False, **kwargs):
        # Lazy, so the checkpoint's arrays are never read when the training model's replace them. This has to
        # happen inside the load: the runner compiles blocks over the weights it holds right after it.
        engine_model, config = original(model_path, lazy=True, **kwargs)
        if not _share_weights(engine_model, model):
            engine_model, config = original(model_path, lazy=lazy, **kwargs)
        return exempt_from_gradient_checkpointing(engine_model), config

    def load_vlm(*args, **kwargs):
        return exempt_from_gradient_checkpointing(original_vlm(*args, **kwargs))

    lifecycle.mlx_lm_load_model, lifecycle.mlx_vlm_load_model = load, load_vlm
    try:
        yield
    finally:
        lifecycle.mlx_lm_load_model, lifecycle.mlx_vlm_load_model = original, original_vlm


def load_vllm(model, *, max_seq_length, gpu_memory_utilization=0.5, max_lora_rank=64, enable_lora=True, seed=0):
    """Start an in-process vllm-metal engine on the checkpoint ``model`` was loaded from, sharing its weights."""
    # In-process engine core: a spawned one re-imports the notebook or script's __main__.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    error = _metal_platform_error()
    if error is not None:
        raise ImportError(f"Unsloth MLX: fast_inference needs vLLM, but {error}. {_INSTALL_HINT}")
    from vllm import LLM

    source = getattr(model, "_src_path", None) or getattr(model, "_hf_repo", None)
    if not source:
        raise RuntimeError("Unsloth MLX: fast_inference needs a model loaded by FastLanguageModel.from_pretrained.")
    engine_kwargs = dict(
        model=str(source),
        max_model_len=int(max_seq_length),
        gpu_memory_utilization=float(gpu_memory_utilization),
        seed=seed,
    )
    if enable_lora:
        engine_kwargs.update(enable_lora=True, max_lora_rank=int(max_lora_rank), max_loras=1)
    if os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] != "0":
        return LLM(**engine_kwargs)
    with _in_process_engine_loads(model):
        return LLM(**engine_kwargs)


def attach_vllm_engine(model, engine):
    model.vllm_engine = engine
    model.fast_generate = engine.generate
    return model


def _scratch_dir(parent=None):
    global _scratch_root
    if parent is not None:
        return tempfile.mkdtemp(prefix=".unsloth_save_lora_", dir=parent)
    if _scratch_root is None:
        _scratch_root = tempfile.mkdtemp(prefix="unsloth_mlx_vllm_")
        atexit.register(shutil.rmtree, _scratch_root, True)
    return tempfile.mkdtemp(dir=_scratch_root)


def save_lora(model, save_directory):
    """Write the live LoRA in PEFT format, replacing the adapter files of an earlier save as PEFT's save_pretrained does."""
    target = os.path.abspath(os.fspath(save_directory))
    if os.path.exists(os.path.join(target, MLX_WEIGHTS_FILE)):
        raise ValueError(
            f"Unsloth MLX: {target!r} holds an MLX-format adapter ({MLX_WEIGHTS_FILE}); "
            "save_lora writes PEFT format, so pick another directory."
        )
    parent = os.path.dirname(target)
    os.makedirs(parent, exist_ok=True)
    scratch = _scratch_dir(parent)
    try:
        staged = os.path.join(scratch, "peft")
        model.save_lora_adapters(staged, adapter_format="peft")
        os.makedirs(target, exist_ok=True)
        for name in os.listdir(staged):
            os.replace(os.path.join(staged, name), os.path.join(target, name))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return target


def _peft_dir(path):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Unsloth MLX: no LoRA adapter directory at {path!r}; save one with model.save_lora first.")
    if detect_adapter_format(path) == "peft":
        return path
    converted = os.path.join(_scratch_dir(), "peft")
    convert_mlx_dir_to_peft(path, converted)
    return converted


def load_lora(model, save_directory=None, load_tensors=False, lora_request_id=None):
    """A ``LoRARequest`` for the engine: the adapter saved at ``save_directory``, or the live adapter when ``load_tensors``."""
    if getattr(model, "vllm_engine", None) is None:
        raise RuntimeError(
            "Unsloth MLX: load_lora needs the vLLM engine that from_pretrained(fast_inference=True) "
            f"starts when vllm-metal is installed. {_INSTALL_HINT}"
        )
    from vllm.lora.request import LoRARequest

    if load_tensors:
        # vllm-metal reads adapters only from PEFT directories, so the live weights go through one.
        path = os.path.join(_scratch_dir(), "peft")
        model.save_lora_adapters(path, adapter_format="peft")
    else:
        if save_directory is None:
            raise ValueError("Unsloth MLX: load_lora needs save_directory unless load_tensors=True.")
        path = _peft_dir(os.path.abspath(os.fspath(save_directory)))
    # Every call advances the counter, explicit ids included, as on CUDA.
    request_id = next(_LORA_IDS)
    if lora_request_id is not None:
        request_id = int(lora_request_id)
    return LoRARequest(str(request_id), request_id, path)


def release_lora(model, request):
    """Drop a request's adapter from the engine and delete its directory when Unsloth created it."""
    model.vllm_engine.llm_engine.remove_lora(request.lora_int_id)
    if _scratch_root is not None and os.path.commonpath([_scratch_root, request.lora_path]) == _scratch_root:
        shutil.rmtree(os.path.dirname(request.lora_path), ignore_errors=True)
