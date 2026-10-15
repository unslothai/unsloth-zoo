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

# Granite-4 dense (granite-4.0-350m, -micro) shared MLP activations reach ~65k, past float16's 65504, so float16
# LoRA training overflows. residual_multiplier is applied only afterwards and output_linear is linear, so float16
# folds it into `up` before the product. Other dtypes run the stock forward.

import functools
import inspect
import re
import textwrap

import torch

from .common import TEMPORARY_PATCHES
from .utils import raise_error

__all__ = []


def _granite_scaled_shared_mlp(layer, hidden_states):
    mlp, scale = layer.shared_mlp, layer.residual_multiplier
    gate, up = mlp.input_linear(hidden_states).chunk(2, dim = -1)
    out = mlp.output_linear(mlp.activation(gate) * (up * scale))
    # A bias (PEFT lora_bias=True on lora_B) escapes the fold: scale * f(x) == f(scale * x) + (scale - 1) * f(0).
    if any(getattr(sub, "bias", None) is not None for sub in mlp.output_linear.modules()):
        out = out + (scale - 1) * mlp.output_linear(up.new_zeros(1, up.shape[-1]))
    return out


def _rewrite_decoder_source(source):
    source = textwrap.dedent(source)
    # Drop decorators (auto_docstring, deprecate_kwarg): the dispatcher below already calls the stock forward for them.
    source = source[source.index("def forward"):]
    # The final residual add is the one after the MLP; the attention one keeps its multiplier.
    final_add = "hidden_states = residual + hidden_states * self.residual_multiplier"
    if source.count(final_add) != 2: return None
    head, _, tail = source.rpartition(final_add)
    tail = "hidden_states = residual + hidden_states" + tail
    head, n_moe = re.subn(
        r"(moe_hidden_states)\s*\+\s*self\.shared_mlp\(hidden_states\)",
        r"\1 * self.residual_multiplier + _granite_scaled_shared_mlp(self, hidden_states)",
        head,
    )
    head, n_dense = re.subn(
        r"hidden_states = self\.shared_mlp\(hidden_states\)",
        r"hidden_states = _granite_scaled_shared_mlp(self, hidden_states)",
        head,
    )
    if n_moe != 1 or n_dense != 1: return None
    return head + tail


def _build_float16_decoder_forward(module, original_forward):
    source = _rewrite_decoder_source(inspect.getsource(original_forward))
    if source is None: return None
    # torch.compile guards resolve globals through sys.modules[__name__], so the helper must live on the module.
    module._granite_scaled_shared_mlp = _granite_scaled_shared_mlp
    namespace = dict(vars(module))
    exec(compile(source, f"<unsloth granitemoehybrid float16 {module.__name__}>", "exec"), namespace)
    return namespace["forward"]


def patch_GraniteMoeHybridDecoderLayer_float16():
    try:
        import transformers.models.granitemoehybrid.modeling_granitemoehybrid as module
        layer_class = module.GraniteMoeHybridDecoderLayer
    except Exception as e:
        return raise_error("GraniteMoeHybridDecoderLayer.forward", e)
    original_forward = layer_class.forward
    if getattr(original_forward, "_unsloth_granite_float16", False): return
    try:
        float16_forward = _build_float16_decoder_forward(module, original_forward)
    except Exception as e:
        return raise_error("GraniteMoeHybridDecoderLayer.forward", e)
    if float16_forward is None:
        return raise_error("GraniteMoeHybridDecoderLayer.forward", "source layout changed")

    @functools.wraps(original_forward)
    def forward(self, hidden_states, *args, **kwargs):
        if hidden_states.dtype == torch.float16:
            return float16_forward(self, hidden_states, *args, **kwargs)
        return original_forward(self, hidden_states, *args, **kwargs)
    forward._unsloth_granite_float16 = True
    layer_class.forward = forward
TEMPORARY_PATCHES.append(patch_GraniteMoeHybridDecoderLayer_float16)
