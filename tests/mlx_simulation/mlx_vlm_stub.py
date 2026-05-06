# Unsloth Zoo - Utilities for Unsloth
# mlx_vlm stub — Studio inference + permissive submodules for 40+ VLM archs
"""
mlx_vlm — Vision Language Model wrapper.

PR-A imports `mlx_vlm.models.{arch}.{arch,vision,language}` for ~40
different architectures.  These all auto-resolve through the
_MLXFinder; no per-arch helpers needed.

PR-B uses `mlx_vlm.stream_generate` directly.
"""

from __future__ import annotations

import sys
import types


def stream_generate(model, processor, prompt, image=None, *args, **kwargs):
    from .mlx_helpers.stream_generate import vlm_stream_generate
    yield from vlm_stream_generate(model, processor, prompt, image, *args, **kwargs)


def generate_step(*args, **kwargs):
    raise NotImplementedError("mlx-shim: mlx_vlm.generate_step not implemented")


def load(*args, **kwargs):
    raise NotImplementedError(
        "mlx-shim: mlx_vlm.load not implemented; PR-B's tests assert this is "
        "NOT called by Studio. If you hit this, Studio dispatch is broken."
    )


# Submodules
def _pkg(name):
    """Make a module that's also a package (so finders can resolve submodules)."""
    m = types.ModuleType(name)
    m.__path__ = []
    return m


utils_module = _pkg("mlx_vlm.utils")
utils_module.MODEL_REMAPPING = {}


def _skip_multimodal_module(*args, **kwargs):
    return False


def _vlm_load_config(*args, **kwargs):
    return {}


utils_module.skip_multimodal_module = _skip_multimodal_module
utils_module.load_config = _vlm_load_config

prompt_utils_module = _pkg("mlx_vlm.prompt_utils")
models_module = _pkg("mlx_vlm.models")
chat_module = _pkg("mlx_vlm.chat")
generate_module = _pkg("mlx_vlm.generate")
server_module = _pkg("mlx_vlm.server")
evals_module = _pkg("mlx_vlm.evals")
evals_utils_module = _pkg("mlx_vlm.evals.utils")


__path__ = []


def __getattr__(name):
    from .mlx_stub import _Noop
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    return _Noop(f"mlx_vlm.{name}")


def inject_into_sys_modules():
    this = sys.modules[__name__]
    this.utils = utils_module
    this.prompt_utils = prompt_utils_module
    this.models = models_module
    this.chat = chat_module
    this.generate = generate_module
    this.server = server_module
    this.evals = evals_module
    evals_module.utils = evals_utils_module
    sys.modules.update({
        "mlx_vlm": this,
        "mlx_vlm.utils": utils_module,
        "mlx_vlm.prompt_utils": prompt_utils_module,
        "mlx_vlm.models": models_module,
        "mlx_vlm.chat": chat_module,
        "mlx_vlm.generate": generate_module,
        "mlx_vlm.server": server_module,
        "mlx_vlm.evals": evals_module,
        "mlx_vlm.evals.utils": evals_utils_module,
    })
    # Sub-architecture modules under mlx_vlm.models.* are auto-created on
    # first import via the _MLXFinder seeded by mlx_stub.inject_into_sys_modules.
