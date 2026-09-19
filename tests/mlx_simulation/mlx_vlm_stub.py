# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Unsloth Zoo - Utilities for Unsloth
"""
mlx_vlm — Vision Language Model wrapper.

PR-A imports `mlx_vlm.models.{arch}.{arch,vision,language}` for ~40
different architectures.  These all auto-resolve through the
_MLXFinder; no per-arch helpers needed.

PR-B uses `mlx_vlm.stream_generate` directly.
"""

from __future__ import annotations

import dataclasses
import math
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
        "NOT called by Unsloth. If you hit this, Unsloth dispatch is broken."
    )


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


def _prepare_inputs(*args, **kwargs):
    return {}


def _process_image(image, *args, **kwargs):
    return image


utils_module.skip_multimodal_module = _skip_multimodal_module
utils_module.load_config = _vlm_load_config
# The batched vision adapter binds both at construction time.
utils_module.prepare_inputs = _prepare_inputs
utils_module.process_image = _process_image

prompt_utils_module = _pkg("mlx_vlm.prompt_utils")
models_module = _pkg("mlx_vlm.models")
chat_module = _pkg("mlx_vlm.chat")
generate_module = _pkg("mlx_vlm.generate")

# The batched vision path introspects the INSTALLED mlx-vlm: it reads
# BatchGenerator.__init__'s signature to decide which KV-quantization controls are
# forwarded, and calls turboquant_enabled to classify the scheme. The module-level
# __getattr__ below answers any name with a _Noop, so without concrete definitions
# those probes resolve to something signature-less and every control is reported
# refused. Shapes copied from mlx-vlm 0.6.4, the newest release this repo's
# transformers<=5.5.0 cap resolves.
generate_ar_module = _pkg("mlx_vlm.generate.ar")


@dataclasses.dataclass
class _BatchResponse:
    uid: int
    token: int
    token_logprob: float
    finish_reason: object = None
    top_logprobs: object = None


class _BatchGenerator:
    Response = _BatchResponse

    def __init__(self, model, processor, max_tokens=256, stop_tokens=None,
                 sampler=None, completion_batch_size=32, prefill_batch_size=8,
                 prefill_step_size=512, prompt_cache=None, kv_bits=None,
                 kv_group_size=64, kv_quant_scheme=None, quantized_kv_start=0,
                 compute_logprobs=False, top_logprobs_k=0, logits_processors=None,
                 stream=None, apc_manager=None, draft_model=None, draft_kind=None,
                 draft_block_size=None, greedy_sampling=False):
        self.kwargs = {
            "kv_bits": kv_bits, "kv_group_size": kv_group_size,
            "kv_quant_scheme": kv_quant_scheme,
            "quantized_kv_start": quantized_kv_start,
        }

    def insert(self, prompts, max_tokens=None, prompt_kwargs=None,
               logits_processors=None, thinking_budget_criteria=None):
        raise NotImplementedError("mlx-shim: BatchGenerator.insert not implemented")

    def next(self):
        raise NotImplementedError("mlx-shim: BatchGenerator.next not implemented")

    def remove(self, uid):
        raise NotImplementedError("mlx-shim: BatchGenerator.remove not implemented")

    def close(self):
        pass


def turboquant_enabled(bits, scheme=None):
    if bits is None:
        return False
    if scheme == "turboquant":
        return True
    bits = float(bits)
    return not math.isclose(bits, round(bits), abs_tol=1e-6)


# The engine rebinds to the module BatchGenerator is DEFINED in, "where the private
# helpers live", so these must name the module they are published from rather than
# this one, or the rebind lands back here and finds none of them.
_BatchGenerator.__module__ = "mlx_vlm.generate.ar"
turboquant_enabled.__module__ = "mlx_vlm.generate.ar"

generate_ar_module.BatchGenerator = _BatchGenerator
generate_ar_module.turboquant_enabled = turboquant_enabled
generate_module.BatchGenerator = _BatchGenerator
generate_module.turboquant_enabled = turboquant_enabled

# mlx_vlm.kv_quant only exists from 0.6.12, so it stays absent here: the engine's
# own fallback names the two schemes, and this shim tracks the pinned release.
turboquant_module = _pkg("mlx_vlm.turboquant")
turboquant_module.turboquant_enabled = turboquant_enabled

# models.cache has to be concrete for one absence. should_quantize_kv_layer arrived in
# 0.6.6; at the pinned 0.6.4 the engine must fall back to the policy ar.py applies
# inline. Left to the permissive finder the name would resolve to a _Noop that imports
# fine and then raises when called, which is neither release's behaviour.
# Deliberately NOT _pkg: with a __path__ the finder treats the absent name as a
# submodule to auto-create, so `from ... import should_quantize_kv_layer` would
# succeed with a module object instead of raising.
models_cache_module = types.ModuleType("mlx_vlm.models.cache")
_CACHE_ABSENT_BEFORE_0_6_6 = ("should_quantize_kv_layer",)


def _models_cache_getattr(name):
    from .mlx_stub import _Noop
    if name in _CACHE_ABSENT_BEFORE_0_6_6 or (
            name.startswith("__") and name.endswith("__")):
        raise AttributeError(name)
    return _Noop(f"mlx_vlm.models.cache.{name}")


models_cache_module.__getattr__ = _models_cache_getattr
server_module = _pkg("mlx_vlm.server")
evals_module = _pkg("mlx_vlm.evals")
evals_utils_module = _pkg("mlx_vlm.evals.utils")


__path__ = []


# Absent at the pinned 0.6.4. kv_quant arrived in 0.6.12; answering it with a _Noop
# would make `from mlx_vlm import kv_quant` succeed and hand back _Noop sentinels
# where the engine expects the scheme name strings, so every scheme looks unknown.
_ABSENT_BEFORE_0_6_12 = ("kv_quant",)


def __getattr__(name):
    from .mlx_stub import _Noop
    if name in _ABSENT_BEFORE_0_6_12 or (
            name.startswith("__") and name.endswith("__")):
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
    this.turboquant = turboquant_module
    models_module.cache = models_cache_module
    evals_module.utils = evals_utils_module
    generate_module.ar = generate_ar_module
    sys.modules.update({
        "mlx_vlm": this,
        "mlx_vlm.utils": utils_module,
        "mlx_vlm.prompt_utils": prompt_utils_module,
        "mlx_vlm.models": models_module,
        "mlx_vlm.chat": chat_module,
        "mlx_vlm.generate": generate_module,
        "mlx_vlm.generate.ar": generate_ar_module,
        "mlx_vlm.turboquant": turboquant_module,
        "mlx_vlm.models.cache": models_cache_module,
        # None makes the import raise ImportError instead of the finder inventing it.
        "mlx_vlm.kv_quant": None,
        "mlx_vlm.server": server_module,
        "mlx_vlm.evals": evals_module,
        "mlx_vlm.evals.utils": evals_utils_module,
    })
    # Sub-architecture modules under mlx_vlm.models.* are auto-created on
    # first import via the _MLXFinder seeded by mlx_stub.inject_into_sys_modules.
