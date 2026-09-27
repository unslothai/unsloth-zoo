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

# Granite-4 dense checkpoints (granite-4.0-350m, granite-4.0-micro) carry massive activations
# in the shared MLP: in bf16, silu(gate) * up reaches ~65k and output_linear ~47k, against
# float16's 65504, so float16 LoRA training overflows to inf within a few steps. The decoder
# only scales that output by residual_multiplier (~0.25) afterwards. output_linear has no
# bias, so in float16 the multiplier is folded into `up` before the product instead. Hybrid
# checkpoints peak ~2.7k and are unaffected; other dtypes run the stock forward.

import inspect
import re
import textwrap

import torch

from .common import TEMPORARY_PATCHES
from .utils import raise_error

__all__ = []


def _granite_scaled_shared_mlp(layer, hidden_states):
    mlp = layer.shared_mlp
    gate, up = mlp.input_linear(hidden_states).chunk(2, dim = -1)
    return mlp.output_linear(mlp.activation(gate) * (up * layer.residual_multiplier))


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
    namespace = dict(vars(module))
    namespace["_granite_scaled_shared_mlp"] = _granite_scaled_shared_mlp
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

    def forward(self, hidden_states, *args, **kwargs):
        if hidden_states.dtype == torch.float16:
            return float16_forward(self, hidden_states, *args, **kwargs)
        return original_forward(self, hidden_states, *args, **kwargs)
    forward._unsloth_granite_float16 = True
    forward.__wrapped__ = original_forward
    layer_class.forward = forward
TEMPORARY_PATCHES.append(patch_GraniteMoeHybridDecoderLayer_float16)
