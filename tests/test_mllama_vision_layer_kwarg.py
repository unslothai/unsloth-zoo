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

"""The patched Mllama vision encoder layer runs under the installed transformers' encoder call
(`hidden_states=` on 5.x, `hidden_state=` before), so Llama 3.2 Vision images encode."""
import pytest
import torch

mllama = pytest.importorskip("transformers.models.mllama.modeling_mllama")

from unsloth_zoo.temporary_patches.misc import patch_MllamaVisionEncoderLayer


def _vision_config():
    from transformers.models.mllama.configuration_mllama import MllamaVisionConfig
    config = MllamaVisionConfig(hidden_size = 16, attention_heads = 2, intermediate_size = 32,
                                num_hidden_layers = 2, num_global_layers = 1, image_size = 28, patch_size = 14,
                                intermediate_layers_indices = [0], vision_output_dim = 32)
    config._attn_implementation = "eager"
    return config


def test_patched_layer_runs_under_the_installed_encoder():
    patch_MllamaVisionEncoderLayer()
    config = _vision_config()
    encoder = mllama.MllamaVisionEncoder(config, num_layers = 2, is_gated = False).eval()
    torch.manual_seed(0)
    hidden = torch.randn(1, 5, 16)
    with torch.no_grad():
        out = encoder(hidden)
    states = out[0] if isinstance(out, (tuple, list)) else getattr(out, "last_hidden_state", out)
    assert states.shape == (1, 5, 16) and torch.isfinite(states).all()


def test_patched_layer_takes_either_argument_name():
    patch_MllamaVisionEncoderLayer()
    layer = mllama.MllamaVisionEncoderLayer(_vision_config(), is_gated = True).eval()
    x = torch.randn(1, 5, 16)
    with torch.no_grad():
        a = layer(hidden_state = x)
        b = layer(hidden_states = x)
        c = layer(x)
    assert torch.equal(a, b) and torch.equal(a, c)
