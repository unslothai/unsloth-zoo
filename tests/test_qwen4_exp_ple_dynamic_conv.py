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

"""Qwen4Exp's PLE short conv must not recompile per sequence length: Inductor's conv backward
specializes the length, so the traced path uses shifted multiply-adds."""

import os

import pytest
import torch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
modeling = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")

from unsloth_zoo.temporary_patches import qwen4_exp as zoo_qwen4_exp  # noqa: E402

LENGTHS = (37, 91, 150, 203)


def _ple(device = "cpu", dtype = torch.float32):
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    zoo_qwen4_exp.patch_qwen4_exp()
    cfg = Qwen4ExpTextConfig(
        hidden_size = 64, num_hidden_layers = 4, num_attention_heads = 4, num_key_value_heads = 2, head_dim = 16,
        indexer_n_heads = 2, indexer_kv_heads = 1, indexer_head_dim = 16, indexer_budget = 16,
        indexer_compress_ratio = 4, linear_num_key_heads = 2,
        linear_num_value_heads = 2, linear_key_head_dim = 16, linear_value_head_dim = 16, num_experts = 4,
        num_experts_per_tok = 2, moe_intermediate_size = 16, shared_expert_intermediate_size = 16, vocab_size = 128,
        hc_count = 2, hc_lowrank = 8, ple_layer_ids = [2], ple_embed_dim = 64, ngram_vocab_size_base = 100,
        eos_token_id = 1,
    )
    torch.manual_seed(0)
    ple = modeling.Qwen4ExpTextPLELayer(cfg, layer_idx = 1, ple_layer_index = 0).to(device, dtype)
    with torch.no_grad():
        ple.conv1d.weight.normal_()
    return ple


def _conv_step(fn, ple, length, device, dtype, seed = 0):
    g = torch.Generator().manual_seed(seed + length)
    x = torch.randn(2, length, ple.conv1d.in_channels, generator = g, dtype = torch.float64).to(device, dtype)
    x.requires_grad_(True)
    torch._dynamo.maybe_mark_dynamic(x, 1)
    out = fn(x)
    dy = torch.randn(out.shape, generator = g, dtype = torch.float64).to(device, dtype)
    xgrad, wgrad = torch.autograd.grad((out * dy).sum(), [x, ple.conv1d.weight])
    return out.detach(), xgrad, wgrad


def test_ple_conv_is_the_dynamic_subclass_and_keeps_its_state():
    ple = _ple()
    assert type(ple.conv1d) is zoo_qwen4_exp.UnslothDilatedDepthwiseConv1d
    assert isinstance(ple.conv1d, torch.nn.Conv1d)
    assert list(ple.conv1d.state_dict()) == ["weight"] and ple.conv1d.dilation[0] > 1


def test_kill_switch_keeps_nn_conv1d(monkeypatch):
    monkeypatch.setenv("UNSLOTH_QWEN4_EXP_PLE_DYNAMIC_CONV", "0")
    assert type(_ple().conv1d) is torch.nn.Conv1d


def test_traced_conv_matches_nn_conv1d_in_float64():
    ple = _ple(dtype = torch.float64)
    reference = torch.nn.Conv1d.forward
    compiled = torch.compile(lambda x: ple._short_conv(x, None), backend = "aot_eager", fullgraph = True)
    torch._dynamo.reset()
    try:
        for length in LENGTHS[:3]:
            got = _conv_step(compiled, ple, length, "cpu", torch.float64)
            ple.conv1d.__class__.forward, saved = reference, ple.conv1d.__class__.forward
            try:
                want = _conv_step(lambda x: ple._short_conv(x, None), ple, length, "cpu", torch.float64)
            finally:
                ple.conv1d.__class__.forward = saved
            for a, b in zip(got, want):
                torch.testing.assert_close(a, b, rtol = 1e-12, atol = 1e-12)
    finally:
        torch._dynamo.reset()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason = "Inductor's convolution backward specialization is a GPU lowering")
def test_inductor_compiles_the_ple_conv_once_across_lengths():
    from torch._dynamo.utils import counters
    ple = _ple("cuda")
    compiled = torch.compile(lambda x: ple._short_conv(x, None))
    torch._dynamo.reset()
    counters.clear()
    try:
        for length in LENGTHS:
            got = _conv_step(compiled, ple, length, "cuda", torch.float32)
            want = _conv_step(lambda x: ple._short_conv(x, None), ple, length, "cuda", torch.float32)
            for a, b in zip(got, want):
                torch.testing.assert_close(a, b, rtol = 1e-4, atol = 1e-4)
        assert counters["stats"]["unique_graphs"] == 1, dict(counters["stats"])
    finally:
        torch._dynamo.reset()
