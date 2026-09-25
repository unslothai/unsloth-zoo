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

"""The MLX decision model against a torch reference of laya's DecisionModel, on real Metal."""

from __future__ import annotations

import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from mlx_simulation import mlx_is_simulated  # noqa: E402

if mlx_is_simulated():
    pytest.skip("needs real MLX: mx.fast attention and RoPE", allow_module_level = True)

from safetensors.torch import save_file  # noqa: E402

from unsloth_zoo.mlx.decision import load_decision_model  # noqa: E402


class _Reference(torch.nn.Module):
    # laya.common.DecisionModel up to its decision logits.
    def __init__(self, encoder, head_layers):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        layer = torch.nn.TransformerEncoderLayer(d, max(1, d // 64), 4 * d, 0.0, batch_first = True, norm_first = True)
        self.head = torch.nn.TransformerEncoder(layer, head_layers, enable_nested_tensor = False)
        self.type_emb = torch.nn.Embedding(3, d)
        self.scorer = torch.nn.Sequential(
            torch.nn.LayerNorm(d), torch.nn.Linear(d, d), torch.nn.GELU(), torch.nn.Linear(d, 1)
        )
        self.act_head = torch.nn.Sequential(torch.nn.Linear(d + 4, 256), torch.nn.GELU(), torch.nn.Linear(256, 2))
        self.register_buffer("temperature", torch.ones(3))

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        h = self.encoder(input_ids = input_ids, attention_mask = attention_mask).last_hidden_state
        h = h + self.type_emb(qtype)[:, None, :]
        for layer in self.head.layers:
            h = layer(h, src_key_padding_mask = ~attention_mask.bool())
        index = marker_pos.clamp(min = 0)[:, :, None].expand(-1, -1, h.size(-1))
        logits = self.scorer(torch.gather(h, 1, index)).squeeze(-1).float()
        return logits.masked_fill(~marker_mask, -1e4)


# 1 and 0 head layers exercise the trimmed last layer alone and the no-head gather.
@pytest.fixture(scope = "module", params = [2, 1, 0])
def checkpoint(request, tmp_path_factory):
    torch.manual_seed(0)
    config = transformers.ModernBertConfig(
        vocab_size = 97,
        hidden_size = 128,
        intermediate_size = 96,
        num_hidden_layers = 4,
        num_attention_heads = 2,
        # Short window and distinct thetas, so the sliding layers and their RoPE base both matter.
        local_attention = 8,
        global_attn_every_n_layers = 3,
        rope_parameters = {
            "full_attention": {"rope_type": "default", "rope_theta": 160000.0},
            "sliding_attention": {"rope_type": "default", "rope_theta": 50.0},
        },
        pad_token_id = 0,
    )
    reference = _Reference(transformers.ModernBertModel(config), head_layers = request.param).eval()
    for p in reference.parameters():
        # Default init leaves LayerNorm at identity and biases at zero, which would hide their mapping.
        p.data.add_(0.1 * torch.randn_like(p))
    folder = tmp_path_factory.mktemp("laya")
    (folder / "encoder").mkdir()
    config.save_pretrained(folder / "encoder")
    (folder / "rl_agent_config.json").write_text(json.dumps({"head_layers": request.param}))
    save_file({k: v.contiguous() for k, v in reference.state_dict().items()}, folder / "model.safetensors")
    return reference, folder


def _batch():
    # Row 1 is padded far past the window, so its padding queries have no valid key in range.
    lengths, markers = [30, 11], [[3, 17, 26], [2, 9, 0]]
    ids = np.zeros((2, 30), np.int64)
    rng = np.random.default_rng(0)
    for row, length in enumerate(lengths):
        ids[row, :length] = rng.integers(1, 97, length)
    return {
        "input_ids": ids,
        "attention_mask": (np.arange(30)[None] < np.array(lengths)[:, None]).astype(np.int64),
        "marker_pos": np.array(markers),
        "marker_mask": np.array([[True, True, True], [True, True, False]]),
        "qtype": np.array([0, 2]),
    }


def test_logits_match_the_torch_reference(checkpoint):
    reference, folder = checkpoint
    batch = _batch()
    with torch.inference_mode():
        expected = reference(**{k: torch.from_numpy(v) for k, v in batch.items()}).numpy()
    got = load_decision_model(folder, dtype = mx.float32).logits(batch)
    np.testing.assert_allclose(got, expected, atol = 2e-5, rtol = 0)
    assert got[1, 2] == -1e4
