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
import weakref

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
# The in-process engine whose live adapter (``engine._unsloth_live_lora``) a request path names; no file is there.
_IN_MEMORY_LORAS = weakref.WeakValueDictionary()
_IN_MEMORY_ROOT = "/unsloth-in-memory-lora/"
_IN_MEMORY_PATHS = (f"{_IN_MEMORY_ROOT}{n}" for n in itertools.count(1))


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


def _training_dtype(model):
    import collections

    import mlx.core as mx
    from mlx.utils import tree_flatten

    counts = collections.Counter(a.dtype for _, a in tree_flatten(model.parameters()) if mx.issubdtype(a.dtype, mx.floating))
    return counts.most_common(1)[0][0] if counts else None


def _bare_path(model):
    """Maps ``model``'s paths to the bare model's: a LoRA wrapper's base layer (e.g. ``q_proj.linear``) to the wrapper's
    path, as when ``from_pretrained`` applied a saved adapter before the engine starts."""
    from .utils import iter_mlx_lora_modules

    bases = {}
    for name, module in iter_mlx_lora_modules(model):
        for child in ("linear", "embedding"):
            if hasattr(module, child):
                bases[f"{name}.{child}" if name else child] = name
                break

    def bare(path):
        parts = path.split(".")
        for end in range(len(parts), 0, -1):
            prefix = ".".join(parts[:end])
            if prefix in bases:
                return ".".join(filter(None, (bases[prefix], *parts[end:])))
        return path

    return bare


def _quantize_like(engine_model, model):
    """Quantize each engine layer the training model has quantized, with its group size, bits and mode."""
    import mlx.nn as nn

    bare = _bare_path(model)
    quantized = {bare(path): m for path, m in model.named_modules() if hasattr(m, "scales") and hasattr(m, "bits")}
    if quantized:
        nn.quantize(engine_model, class_predicate=lambda path, module: (
            path in quantized and hasattr(module, "to_quantized")
            and dict(group_size=quantized[path].group_size, bits=quantized[path].bits, mode=getattr(quantized[path], "mode", "affine"))))
    return engine_model


def _counterpart(engine_model, model):
    """The part of ``model`` laid out like ``engine_model``: itself, or its language model when the engine serves a
    multimodal checkpoint with a text-only mlx-lm model that names its weights without ``language_model.``."""
    paths = {path for path, _ in engine_model.named_modules()}
    for part in (model, getattr(model, "language_model", None)):
        if part is not None and paths <= {path for path, _ in part.named_modules()}:
            return part
    return model


def _share_weights(engine_model, model):
    """Point the engine's (lazily loaded) arrays at ``model``'s when every name and shape matches and dtypes differ only
    in float precision; the engine then runs at the training model's dtypes, float32 parameters a model keeps included."""
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    wanted = dict(tree_flatten(engine_model.parameters()))
    bare = _bare_path(model)
    trained = {bare(name): array for name, array in tree_flatten(model.parameters())}
    floating = lambda a: mx.issubdtype(a.dtype, mx.floating)
    # Integer dtypes must match: packed weights and uint8 scales of another quantization mode can share a shape.
    if any(name not in trained or trained[name].shape != array.shape
           or (trained[name].dtype != array.dtype and not (floating(array) and floating(trained[name])))
           for name, array in wanted.items()):
        return False
    engine_model.update(tree_unflatten([(name, trained[name]) for name in wanted]))
    return True


@contextlib.contextmanager
def _in_process_engine_loads(model):
    """While an in-process engine starts, its mlx-lm or mlx-vlm model (and Gemma 4's vision sidecar) takes the training
    model's weights instead of a second copy, and its layers skip Unsloth's class-level gradient checkpointing."""
    try:
        import vllm_metal.v1.model_lifecycle as lifecycle
    except ImportError:
        yield
        return
    from .utils import exempt_from_gradient_checkpointing

    original, original_vlm = lifecycle.mlx_lm_load_model, lifecycle.mlx_vlm_load_model

    def shared(load_engine, lazy):
        # Lazy, so the checkpoint's arrays are never read when the training model's replace them. This has to
        # happen inside the load: the runner compiles blocks over the weights it holds right after it.
        engine_model, *rest = load_engine(True)
        source = _counterpart(engine_model, model)
        if not _share_weights(_quantize_like(engine_model, source), source):
            engine_model, *rest = load_engine(lazy)
            if _training_dtype(model) is not None:  # load_vllm gives vLLM this dtype for the KV cache
                from .loader import _convert_mlx_dtype

                _convert_mlx_dtype(engine_model, _training_dtype(model))
        return exempt_from_gradient_checkpointing(engine_model), *rest

    def load(model_path, lazy=False, **kwargs):
        return shared(lambda lazy: original(model_path, lazy=lazy, **kwargs), lazy)

    def load_vlm(model_path, lazy=False, **kwargs):
        return shared(lambda lazy: (original_vlm(model_path, lazy=lazy, **kwargs),), lazy)[0]

    try:  # Gemma 4 serves vision from towers of an mlx-vlm composite it loads itself, next to the mlx-lm text model.
        from vllm_metal.multimodal.gemma4.sidecar import Gemma4VisionSidecar
    except ImportError:
        Gemma4VisionSidecar = None
    original_sidecar = Gemma4VisionSidecar and Gemma4VisionSidecar.__dict__["load"]

    def load_sidecar(cls, model_path, **kwargs):
        return original_sidecar.__func__(cls, model_path, **{"load_composite": lambda path: load_vlm(path, lazy=True), **kwargs})

    lifecycle.mlx_lm_load_model, lifecycle.mlx_vlm_load_model = load, load_vlm
    if Gemma4VisionSidecar is not None:
        Gemma4VisionSidecar.load = classmethod(load_sidecar)
    try:
        yield
    finally:
        lifecycle.mlx_lm_load_model, lifecycle.mlx_vlm_load_model = original, original_vlm
        if Gemma4VisionSidecar is not None:
            Gemma4VisionSidecar.load = original_sidecar


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
    dtype = _training_dtype(model)
    if dtype is not None:
        # The engine shares the training model's arrays at this dtype; vllm-metal sizes its KV cache by it.
        engine_kwargs["dtype"] = str(dtype).rsplit(".", 1)[-1]
    with _in_process_engine_loads(model):
        engine = LLM(**engine_kwargs)
    engine._unsloth_in_memory_lora = _serve_in_memory_loras(engine)
    return engine


def _serve_in_memory_loras(engine):
    """Let vllm-metal's adapter loader answer in-memory request paths from ``_IN_MEMORY_LORAS``; False without it."""
    try:
        from vllm_metal.v1.lora import worker_manager
    except ImportError:
        return False
    add_request = engine.llm_engine.add_request

    def checked_add_request(*args, lora_request=None, **kwargs):
        # Refuse here, where vLLM aborts the batch cleanly: a load failing mid-step leaves the engine unusable.
        path = getattr(lora_request, "lora_path", "")
        if path.startswith(_IN_MEMORY_ROOT) and path not in _IN_MEMORY_LORAS:
            raise ValueError(
                "Unsloth MLX: this live LoRA request was released or replaced by a newer "
                "load_lora(load_tensors=True); call load_lora(load_tensors=True) again."
            )
        return add_request(*args, lora_request=lora_request, **kwargs)

    engine.llm_engine.add_request = checked_add_request
    original = worker_manager.load_peft_adapter
    if getattr(original, "_unsloth_in_memory", False):
        return True

    def load_peft_adapter(adapter_path, *, lora_id, lora_config=None, **kwargs):
        engine = _IN_MEMORY_LORAS.get(str(adapter_path))
        if engine is None:
            return original(adapter_path, lora_id=lora_id, lora_config=lora_config, **kwargs)
        adapter = engine._unsloth_live_lora[1]
        from vllm_metal.v1.lora.peft_loader import LoadedLoRA, LoRALayerWeightsMLX

        rank = max(int(a.shape[0]) for a, _, _ in adapter.values())
        if lora_config is not None and rank > lora_config.max_lora_rank:
            raise ValueError(f"LoRA rank {rank} is greater than max_lora_rank {lora_config.max_lora_rank}.")
        weights = {name: LoRALayerWeightsMLX(name, int(a.shape[0]), a, b, scaling) for name, (a, b, scaling) in adapter.items()}
        return LoadedLoRA(lora_id=lora_id, rank=rank, weights=weights)

    load_peft_adapter._unsloth_in_memory = True
    worker_manager.load_peft_adapter = load_peft_adapter
    return True


def _live_adapter(model):
    """The live LoRA in PEFT layout per module (A = lora_a.T, B = lora_b.T, scale), or None when only a PEFT directory
    can carry it (DoRA, other wrappers, full-state modules)."""
    import mlx.core as mx
    from mlx_lm.tuner.lora import LoRALinear

    from .utils import iter_mlx_lora_modules

    modules = list(iter_mlx_lora_modules(model))
    full_modules = getattr(model, "_unsloth_full_state_modules", None) or getattr(model, "_unsloth_cpt_full_module_weight_keys", None)
    if not modules or full_modules or any(type(m) is not LoRALinear for _, m in modules):
        return None
    adapter = {name: (module.lora_a.T, module.lora_b.T, float(module.scale)) for name, module in modules}
    mx.eval([array for a, b, _ in adapter.values() for array in (a, b)])
    return adapter


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

    adapter = _live_adapter(model) if load_tensors and getattr(model.vllm_engine, "_unsloth_in_memory_lora", False) else None
    if adapter is not None:
        # One live adapter per engine: vllm-metal keeps every request it served, so each unreleased snapshot would stay.
        if getattr(model.vllm_engine, "_unsloth_live_lora", None) is not None:
            release_lora(model, model.vllm_engine._unsloth_live_lora[0])
        path = next(_IN_MEMORY_PATHS)
    elif load_tensors:
        # A multiprocess engine, or an adapter only PEFT can describe, reads the live weights from a PEFT directory.
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
    request = LoRARequest(str(request_id), request_id, path)
    if adapter is not None:
        model.vllm_engine._unsloth_live_lora = (request, adapter)
        _IN_MEMORY_LORAS[path] = model.vllm_engine
    return request


def release_lora(model, request):
    """Drop a request's adapter from the engine and delete its in-memory arrays or directory when Unsloth created it."""
    model.vllm_engine.llm_engine.remove_lora(request.lora_int_id)
    if _IN_MEMORY_LORAS.pop(request.lora_path, None) is not None:
        model.vllm_engine._unsloth_live_lora = None
    if _scratch_root is not None and os.path.commonpath([_scratch_root, request.lora_path]) == _scratch_root:
        shutil.rmtree(os.path.dirname(request.lora_path), ignore_errors=True)
