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

import torch
import torch.nn as nn
import inspect
import importlib
from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
from .common import TEMPORARY_PATCHES, torch_compile, RESCOPE_PATCH_FLAG, WRAPPER_INNER_ATTR
from .utils import (
    patch_function,
    process_output_options,
    process_return,
    KWARGS_TYPE,
    raise_error,
    ImageInput,
    PreTokenizedInput,
    TextInput,
    Cache,
    StaticCache,
    HybridCache,
    Unpack,
    _get_unique_storage_name,
)
from textwrap import dedent
import re


# transformers 5.4.0 and 5.5.x drop the bnb quant_state sidecars of some composite checkpoints.
#
# Both bounds carry `.dev0` so a prerelease sorts with the release line it belongs to. Without
# it `Version("5.6.0.dev0") < Version("5.6.0")` puts a 5.6.0 release candidate INSIDE the
# window and tells someone running a build that carries the fix that their transformers is the
# problem. The same applies at the bottom: a 5.4.0 prerelease already has the defect.
#
# Compared with `packaging.version.Version` below, and deliberately NOT with
# `unsloth_zoo.utils.Version`, which is the wrapper most of this package uses: that one
# rewrites a prerelease suffix and answers `Version('5.6.0.1')` for the string "5.6.0.dev0",
# which sorts AFTER the release it has to sort before, so the window would swallow 5.6.0.
_QUANT_STATE_BROKEN_TRANSFORMERS = ("5.4.0.dev0", "5.6.0.dev0")


def _transformers_drops_prequantized_quant_state():
    """True when the installed transformers is inside the quant_state defect window.

    transformers 5.4.0 (PR #44300) made the conversion mapping recurse into
    `PreTrainedModel` submodules, which pulled the text model's
    `^model.language_model` -> `model` `WeightRenaming` into the composite model's
    mapping. That renaming rewrites both the packed `weight` and its
    `weight.absmax`, `weight.quant_map`, `weight.nested_absmax`,
    `weight.nested_quant_map` and `weight.quant_state.bitsandbytes__nf4` sidecars
    into keys the model does not have. The packed weight survives anyway, because
    the loader retries with the ORIGINAL key when that key is a model parameter; a
    sidecar's original key never is, so the retry cannot fire and the sidecars are
    discarded as unexpected. transformers PR #45567 scoped the prefix surgery with
    `with_submodel_prefix` in 5.6.0, so the window is exactly 5.4.0 and 5.5.0
    through 5.5.4 (PyPI has no 5.4.1 and no 5.5.5).

    Only model types whose text submodel mapping STRIPS the composite prefix are
    affected -- `qwen3_5_text`, `gemma3n_text` and their aliases. Most composite
    mappings add a prefix instead and are untouched. This predicate reads the
    version only, so it is necessary, not sufficient.
    """
    installed = _installed_transformers_version()
    if installed == "unknown": return False
    try:
        from packaging.version import Version as _Version
        parsed = _Version(installed)
        low, high = _QUANT_STATE_BROKEN_TRANSFORMERS
        return _Version(low) <= parsed < _Version(high)
    except Exception:
        return False


def _installed_transformers_version():
    """The installed transformers version, or "unknown".

    Falls back to an already-imported `transformers.__version__` because a
    checkout without `.dist-info`, or a frozen bundle, has no metadata to read
    and would otherwise lose the defect-window explanation entirely. Reads
    `sys.modules` only, so it never triggers an import.
    """
    try:
        from importlib.metadata import version as _version
        return _version("transformers")
    except Exception:
        pass
    try:
        import sys
        return getattr(sys.modules.get("transformers"), "__version__", "unknown")
    except Exception:
        return "unknown"


def _composite_renaming_repair_installed():
    """Is the runtime repair for that defect live in THIS process?

    `temporary_patches/conversion_mapping_rescope.py` here, and
    `fix_transformers_composite_prefix_renaming` in unsloth's `import_fixes.py`, both re-scope
    the leaked renaming before it can rename anything. Either mark counts, and the whole chain
    is walked, because the two compose in either order and `moe_utils_bnb4bit.py` puts a third
    wrapper on the same function. That third one publishes `WRAPPER_INNER_ATTR` rather than
    `__wrapped__`, so follow both: it must survive a rescope install, and the rescope unwraps
    `__wrapped__` to decide what to wrap.

    Asked so the message cannot send a user to change a transformers version that is no longer
    what is failing them. With the repair live, a module that still reaches this guard has a
    checkpoint whose sidecars really are absent -- which is the one case where re-quantizing IS
    the answer, and the version branch would have told them the opposite.
    """
    try:
        from transformers import conversion_mapping
    except Exception:
        return False
    function = getattr(conversion_mapping, "get_model_conversion_mapping", None)
    seen = 0
    while function is not None and seen < 8:
        if getattr(function, RESCOPE_PATCH_FLAG, False):
            return True
        if getattr(function, "_unsloth_patched_composite_prefix_renaming", False):
            return True
        function = getattr(function, "__wrapped__", None) or getattr(function, WRAPPER_INNER_ATTR, None)
        seen += 1
    return False


def _packed_weight_without_quant_state_error(module):
    """Message for a Linear4bit whose packed weight arrived with no quant_state."""
    transformers_version = _installed_transformers_version()
    shape = tuple(module.weight.shape)
    # The cause belongs in the version-specific branch below, never here. This function
    # inspects the module, never the checkpoint, so "lost while loading" is a claim it
    # cannot make: a checkpoint whose sidecars really are absent reaches this same line.
    head = (
        f"Unsloth: a bitsandbytes Linear4bit holds what looks like its PACKED 4-bit "
        f"weight (shape {shape}, dtype {module.weight.dtype}, out_features "
        f"{getattr(module, 'out_features', 'unknown')}) but has no quant_state, so it "
        f"cannot be dequantized. Its quantization metadata is missing."
    )
    if _transformers_drops_prequantized_quant_state() and _composite_renaming_repair_installed():
        # Inside the window, but the repair is live, so the renaming never fired and the
        # version is not what went wrong here.
        return (
            f"{head}\nInstalled transformers=={transformers_version}, which is inside the "
            f"window that discards quant_state sidecars -- but Unsloth's runtime repair for "
            f"that is installed in this process, so it is not the explanation here. Check "
            f"that the checkpoint's `weight.absmax`, `weight.quant_map` and "
            f"`weight.quant_state.bitsandbytes__nf4` tensors are present and were not "
            f"reported as unexpected keys during loading."
        )
    if _transformers_drops_prequantized_quant_state():
        return (
            f"{head} The most likely reason is that it was lost while LOADING, not "
            f"while saving.\n"
            f"This is transformers=={transformers_version}: releases 5.4.0 and "
            f"5.5.0 to 5.5.4 discard the quant_state sidecar tensors of pre-quantized "
            f"composite (multimodal) checkpoints whose text submodel strips the "
            f"composite prefix, which covers Qwen3.5 and Gemma 3n. Introduced by "
            f"transformers PR #44300, fixed by PR #45567 in 5.6.0. Install "
            f"transformers>=5.6.0, or fall back to 5.3.0 (on Apple Silicon Unsloth "
            f"caps transformers at 5.5.0, so 5.3.0 is the option there), and try "
            f"again before regenerating anything: if that is what happened here the "
            f"checkpoint is intact and re-quantizing it will not help. If a supported "
            f"transformers still reports this, the sidecar tensors really are missing "
            f"from the files and the checkpoint does need rebuilding."
        )
    return (
        f"{head}\nInstalled transformers=={transformers_version}. Check that the "
        f"checkpoint's `weight.absmax`, `weight.quant_map` and "
        f"`weight.quant_state.bitsandbytes__nf4` tensors are present and were not "
        f"reported as unexpected keys during loading."
    )


def patch_bitsandbytes_linear4bit_forward():
    # Fixes torch.compile complaining about multiple things
    try:
        import bitsandbytes
        bitsandbytes.nn.modules.Linear4bit
        Params4bit = bitsandbytes.nn.modules.Params4bit
        fix_4bit_weight_quant_state_from_module = bitsandbytes.nn.modules.fix_4bit_weight_quant_state_from_module
    except Exception as e:
        return raise_error("bitsandbytes.Linear4bit", e)

    # Fix Params4bit.__torch_function__ infinite recursion under torch.compile.
    # bnb >= 0.46's Params4bit.__torch_function__ delegates to super() (except
    # chunk/split), which re-dispatches back to it since Params4bit is still in
    # the types tuple. Eager mode blocks this via _disabled_torch_function_impl,
    # but torch.compile's AOT autograd runtime does not (seen on T4 with torch
    # 2.8.0 + bnb 0.49.2). Removing it falls back to Parameter/Tensor C-level
    # dispatch that cannot recurse; bnb only added chunk/split handling anyway.
    if hasattr(Params4bit, "__torch_function__") and \
       "__torch_function__" in Params4bit.__dict__:
        delattr(Params4bit, "__torch_function__")
    pass

    def forward(self, x: torch.Tensor):
        # In transformers 5.0+, weights may not be in packed format yet during init
        if self.weight.shape[-1] == 1:
            fix_4bit_weight_quant_state_from_module(self)

        # Some layers may not be quantized (no quant_state) - fall back to regular matmul
        quant_state = getattr(self.weight, "quant_state", None)
        if quant_state is None:
            bias = None if self.bias is None else self.bias
            weight = self.weight
            # A layer that is genuinely unquantized holds an ordinary [out, in] weight,
            # and the fallback below is correct for it. A PACKED 4-bit buffer is [N, 1]
            # uint8, and handing that to F.linear only ever produces
            #   RuntimeError: mat1 and mat2 shapes cannot be multiplied (8x5120 and 1x15728640)
            # which reads like a corrupt checkpoint and sent the reporters of unsloth
            # #9867, #10010, #10017 and #10276 off regenerating good ones. It is not the
            # checkpoint: transformers 5.4.0 and 5.5.x discard the quant_state sidecars of
            # pre-quantized composite (multimodal) checkpoints while loading them. Name
            # that instead of letting the shape error stand. Note the recovery attempt
            # above cannot help here: fix_4bit_weight_quant_state_from_module only copies
            # module.quant_state onto the weight, and module.quant_state is itself None
            # because Params4bit.from_prequantized never ran.
            # A packed blob is (out_features * in_features // (2 * quant_storage.itemsize), 1)
            # (bitsandbytes _ops.py quantize_4bit), so it is [N, 1] whatever quant_storage
            # is. The ONE legitimate unquantized weight that is also [N, 1] is a layer with
            # a single input feature, and then N is exactly out_features. So ask that
            # question directly rather than comparing row counts alone: comparing only
            # shape[0] against out_features also excused a packed blob whenever
            # in_features == 2 * quant_storage.itemsize (2 for uint8, 4 for float16 and
            # bfloat16, 8 for float32), where the row count coincides by arithmetic and
            # the user was handed the shape error again.
            #
            # The exemption also requires a float weight, because in_features ==
            # out_features == 1 packs to (1, 1) and so satisfies the shape test on its own.
            # An unquantized weight is always floating point; a packed one carries
            # quant_storage, uint8 by default. Dtype NARROWS the exemption here, it does not
            # gate the raise: making the raise itself conditional on uint8 would disarm the
            # guard for a checkpoint packed with a float quant_storage, which is why it is
            # written this way round. No float-storage packing reaches the exemption anyway,
            # since shape[0] == out_features with in_features == 1 needs
            # out // (2 * itemsize) == out, which holds for no itemsize >= 1.
            in_features  = getattr(self, "in_features",  None)
            out_features = getattr(self, "out_features", None)
            if (
                weight.dim() == 2
                and weight.shape[-1] == 1
                and not (
                    in_features == 1
                    and weight.shape[0] == out_features
                    and weight.is_floating_point()
                )
            ):
                raise RuntimeError(_packed_weight_without_quant_state_error(self))
            if weight.dtype != x.dtype:
                weight = weight.to(x.dtype)
            if bias is not None and bias.dtype != x.dtype:
                bias = bias.to(x.dtype)
            return torch.nn.functional.linear(x, weight, bias)

        # weights are cast automatically as Int8Params, but the bias has to be cast manually

        # ** Errors out in torch.compile so remove it
        # if self.bias is not None and self.bias.dtype != x.dtype:
        #     self.bias.data = self.bias.data.to(x.dtype)

        if not self.compute_type_is_set:
            self.set_compute_type(x)
            self.compute_type_is_set = True

        inp_dtype = x.dtype
        if self.compute_dtype is not None:
            x = x.to(self.compute_dtype)

        bias = None if self.bias is None else self.bias.to(self.compute_dtype)
        # ** Errors out in torch.compile
        # weight = self.weight.t() if self.weight.dim() == 2 else self.weight

        weight = self.weight.data.t()

        return bitsandbytes.matmul_4bit(x, weight, bias=bias, quant_state=quant_state).to(inp_dtype)

    patch_function(bitsandbytes.nn.modules.Linear4bit, "forward", forward)
    try:
        patch_function(bitsandbytes.nn.Linear4bit, "forward", forward)
    except:
        pass
pass
TEMPORARY_PATCHES.append(patch_bitsandbytes_linear4bit_forward)
