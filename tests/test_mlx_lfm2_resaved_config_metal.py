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

import json

import pytest

try:
    import mlx.core as mx
    from mlx.utils import tree_flatten
except Exception:
    pytest.skip("requires mlx", allow_module_level=True)
from mlx_simulation import mlx_is_simulated  # noqa: E402

if mlx_is_simulated():
    pytest.skip("needs real mlx-lm lfm2 modules", allow_module_level=True)

# unsloth/LFM2.5-230M shape, shrunk; a transformers 5 save keeps only intermediate_size.
_CONFIG = {
    "model_type": "lfm2",
    "vocab_size": 64,
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 128,
    "norm_eps": 1e-5,
    "conv_bias": False,
    "conv_L_cache": 3,
    "block_dim": 32,
    "block_multiple_of": 16,
    "block_ffn_dim_multiplier": 1.0,
    "block_auto_adjust_ff_dim": True,
    "intermediate_size": 96,
    "layer_types": ["conv", "full_attention"],
    "tie_word_embeddings": True,
}


def _save_resaved_lfm2(directory):
    from mlx_lm.models import lfm2

    mx.random.seed(3407)
    model = lfm2.Model(lfm2.ModelArgs.from_dict({**_CONFIG, "block_ff_dim": 96}))
    mx.save_safetensors(str(directory / "model.safetensors"), dict(tree_flatten(model.parameters())))
    (directory / "config.json").write_text(json.dumps(_CONFIG))
    return model


def test_real_mlx_lm_loads_an_lfm2_config_without_block_ff_dim(monkeypatch, tmp_path):
    import mlx_lm.utils as lm_utils
    import unsloth_zoo.mlx.loader as loader

    reference = _save_resaved_lfm2(tmp_path)
    monkeypatch.setattr(lm_utils, "_download", lambda *a, **k: tmp_path)
    monkeypatch.setattr(lm_utils, "load_tokenizer", lambda *a, **k: None)

    model, _, config = loader._load_mlx_lm_with_strict_fallback(
        "unsloth/LFM2.5-230M", "lfm2", None, {"return_config": True},
    )
    assert config["block_ff_dim"] == 96
    tokens = mx.array([[1, 5, 9, 2]])
    assert mx.array_equal(model(tokens), reference(tokens))
