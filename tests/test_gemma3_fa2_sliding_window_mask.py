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

"""Gemma3 sliding layers keep their window when transformers hands them no mask.

With attn_implementation="flash_attention_2" an unpadded batch gets
attention_mask=None and the window is left to the FA2 kernel, but the patched
Gemma3 attention runs SDPA, so a None mask used to mean full causal attention.
The reference is the same patched forward under "sdpa", where transformers
builds the explicit 4D sliding mask itself.
"""

import pytest
import torch

pytest.importorskip("transformers")
try:
    from transformers.models.gemma3 import modeling_gemma3 as gemma3
    from transformers.models.gemma3 import Gemma3TextConfig
except Exception:  # pragma: no cover
    pytest.skip("transformers gemma3 not installed", allow_module_level = True)

from unsloth_zoo.temporary_patches import gemma as gemma_patches
from unsloth_zoo.temporary_patches import utils as patch_utils
from unsloth_zoo.temporary_patches.gemma import _prepare_gemma3_sdpa_attention_mask

WINDOW = 4


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "0")
    monkeypatch.setattr(patch_utils, "UNSLOTH_COMPILE_DISABLE", True)
    monkeypatch.setattr(gemma_patches, "torch_compile", lambda function, *args, **kwargs: function)
    monkeypatch.setattr(gemma3.Gemma3Attention, "forward", gemma3.Gemma3Attention.forward)
    before = gemma3.Gemma3Attention.forward
    gemma_patches.patch_Gemma3Attention_generic()
    assert gemma3.Gemma3Attention.forward is not before, "generic Gemma3 attention patch did not install"
    config = Gemma3TextConfig(
        vocab_size = 64, hidden_size = 16, intermediate_size = 32, num_hidden_layers = 2,
        num_attention_heads = 2, num_key_value_heads = 1, head_dim = 8,
        sliding_window = WINDOW, layer_types = ["sliding_attention", "full_attention"],
    )
    torch.manual_seed(0)
    return gemma3.Gemma3ForCausalLM(config).float().eval()


def _logits(model, attn_implementation, input_ids, **kwargs):
    for module in model.modules():
        if hasattr(module, "config"):
            module.config._attn_implementation = attn_implementation
    with torch.no_grad():
        return model(input_ids, **kwargs).logits


def _reference_mask(q_len, kv_len, window):
    q = torch.arange(q_len)[:, None] + (kv_len - q_len)
    k = torch.arange(kv_len)[None, :]
    return ((k <= q) & (k > q - window))[None, None]


def test_unpadded_prefill_longer_than_window_matches_sdpa(model):
    input_ids = torch.randint(0, 64, (2, 3 * WINDOW))
    reference = _logits(model, "sdpa", input_ids)
    flash_path = _logits(model, "flash_attention_2", input_ids)
    torch.testing.assert_close(flash_path, reference, rtol = 1e-5, atol = 1e-5)


def test_short_prefill_keeps_mask_free_causal_path():
    q = torch.randn(1, 2, WINDOW, 8)
    assert _prepare_gemma3_sdpa_attention_mask(None, q, q, WINDOW) is None
    long = torch.randn(1, 2, WINDOW + 1, 8)
    assert _prepare_gemma3_sdpa_attention_mask(None, long, long, None) is None


@pytest.mark.parametrize("q_len, kv_len", [(WINDOW + 3, WINDOW + 3), (3, WINDOW + 5), (1, WINDOW + 2), (2, 3)])
def test_none_mask_gets_window_and_cache_offset(q_len, kv_len):
    q = torch.randn(2, 2, q_len, 8)
    k = torch.randn(2, 2, kv_len, 8)
    mask = _prepare_gemma3_sdpa_attention_mask(None, q, k, WINDOW)
    assert mask is not None and mask.dtype == torch.bool
    torch.testing.assert_close(mask.expand(2, 1, q_len, kv_len), _reference_mask(q_len, kv_len, WINDOW).expand(2, 1, q_len, kv_len))


def test_single_token_decode_with_padding_mask_over_window_is_windowed():
    q = torch.randn(1, 2, 1, 8)
    k = torch.randn(1, 2, WINDOW + 3, 8)
    mask = _prepare_gemma3_sdpa_attention_mask(torch.ones(1, WINDOW + 3, dtype = torch.long), q, k, WINDOW)
    torch.testing.assert_close(mask, _reference_mask(1, WINDOW + 3, WINDOW))
