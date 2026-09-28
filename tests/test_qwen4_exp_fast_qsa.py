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

"""Qwen4Exp QSA fast path equals the reference whenever kv_length < budget + ratio."""
import pytest
import torch

modeling = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig


def transformers_forward():
    from unsloth_zoo.temporary_patches import qwen4_exp as patch
    fwd = modeling.Qwen4ExpTextQSAIndexer.forward
    if fwd is patch.qwen4_exp_qsa_indexer_forward:
        fwd = patch._reference_qsa_forward
    assert fwd.__module__.startswith("transformers."), fwd
    return fwd


def make_indexer(budget=16, ratio=4):
    cfg = Qwen4ExpTextConfig(
        hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=16, indexer_budget=budget,
        indexer_compress_ratio=ratio, linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=16, linear_value_head_dim=16, num_experts=4, num_experts_per_tok=2,
        moe_intermediate_size=16, shared_expert_intermediate_size=16, vocab_size=128,
        hc_count=2, hc_lowrank=8, ple_layer_ids=[], ngram_vocab_size_base=100,
    )
    torch.manual_seed(0)
    idx = modeling.Qwen4ExpTextQSAIndexer(cfg, layer_idx=3)
    for p in idx.parameters():
        torch.nn.init.normal_(p, 0, 0.5)
    return idx


def rope(T, dim=16, B=2):
    pos = torch.arange(T, dtype=torch.float32)
    freqs = pos[:, None] / (10000 ** (torch.arange(0, dim, 2) / dim))[None]
    emb = torch.cat([freqs, freqs], -1)
    return emb.cos()[None].expand(B, -1, -1), emb.sin()[None].expand(B, -1, -1)


def masks(T, kind, B=2):
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    m = causal[None, None].repeat(B, 1, 1, 1)
    if kind == "leftpad":
        m[1, :, :, :3] = False
    elif kind == "packed":
        cut = T // 2
        m[:, :, cut:, :cut] = False
    return m


def as_float(m, dtype=torch.float32):
    return torch.where(m, torch.zeros((), dtype=dtype), torch.finfo(dtype).min)


class RecordingCache:
    def __init__(self):
        self.keys = []

    def update_indexer(self, raw_keys, layer_idx):
        self.keys.append((layer_idx, raw_keys.detach().clone()))
        return raw_keys


@pytest.mark.parametrize("kind", ["causal", "leftpad", "packed"])
@pytest.mark.parametrize("floaty", [False, True])
@pytest.mark.parametrize("T", [5, 19, 20, 33])
def test_fast_path_matches_reference(kind, floaty, T):
    from unsloth_zoo.temporary_patches.qwen4_exp import _make_fast_indexer_forward
    idx = make_indexer()
    reference = transformers_forward()
    fast = _make_fast_indexer_forward(reference)
    h = torch.randn(2, T, 64)
    m = masks(T, kind)
    if floaty:
        m = as_float(m)
    pe = rope(T)
    c_ref, c_fast = RecordingCache(), RecordingCache()
    with torch.no_grad():
        want = reference(idx, h, pe, m, c_ref)
        got = fast(idx, h, pe, m, c_fast)
    assert got.dtype == want.dtype and got.shape == want.shape
    assert torch.equal(got, want)
    assert len(c_ref.keys) == len(c_fast.keys) == 1
    assert torch.equal(c_ref.keys[0][1], c_fast.keys[0][1])


def test_long_context_still_selects():
    """Beyond the budget the reference really drops blocks; the patch must defer to it."""
    from unsloth_zoo.temporary_patches.qwen4_exp import _make_fast_indexer_forward
    idx = make_indexer()
    reference = transformers_forward()
    T = 48
    m = masks(T, "causal")
    with torch.no_grad():
        want = reference(idx, torch.randn(2, T, 64), rope(T), m, None)
    assert not torch.equal(want, m)
    fast = _make_fast_indexer_forward(reference)
    torch.manual_seed(1)
    h = torch.randn(2, T, 64)
    with torch.no_grad():
        assert torch.equal(fast(idx, h, rope(T), m, None), reference(idx, h, rope(T), m, None))


def test_patch_installed_and_kill_switch(monkeypatch):
    from unsloth_zoo.temporary_patches import qwen4_exp as patch
    cls = modeling.Qwen4ExpTextQSAIndexer
    original = cls.forward
    if original is patch.qwen4_exp_qsa_indexer_forward:
        original = patch._reference_qsa_forward
    try:
        cls.forward = original
        monkeypatch.setenv("UNSLOTH_QWEN4_EXP_FAST_QSA", "0")
        patch.patch_qwen4_exp()
        assert cls.forward is original  # kill switch: untouched
        monkeypatch.delenv("UNSLOTH_QWEN4_EXP_FAST_QSA")
        patch.patch_qwen4_exp()
        assert cls.forward is patch.qwen4_exp_qsa_indexer_forward
        assert patch._reference_qsa_forward is original
        patch.patch_qwen4_exp()
        assert patch._reference_qsa_forward is original
        idx = make_indexer()
        T = 12
        m = masks(T, "causal")
        calls = []
        real_nonzero = torch.nonzero
        monkeypatch.setattr(torch, "nonzero", lambda *a, **k: calls.append(1) or real_nonzero(*a, **k))
        with torch.no_grad():
            out = idx(torch.randn(2, T, 64), rope(T), m, None)
        assert torch.equal(out, m) and calls == []
        # The compiler's generated copy lives outside transformers: never taken as the reference.
        def compiled_copy(self, *a, **k):
            return patch.qwen4_exp_qsa_indexer_forward(self, *a, **k)
        compiled_copy.__module__ = original.__module__  # the compiler keeps the identity
        cls.forward = compiled_copy
        patch.patch_qwen4_exp()
        assert cls.forward is compiled_copy and patch._reference_qsa_forward is original
    finally:
        cls.forward = original


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for torch._grouped_mm")
def test_grouped_mm_experts_take_fp32_input_under_autocast():
    # Unsloth's generate runs under bf16 autocast, where Qwen4Exp's PLE sum makes the residual
    # fp32; torch._grouped_mm is not autocast-cast and raised on the bf16 expert stacks.
    from unsloth_zoo.temporary_patches import moe_utils
    if not moe_utils._check_torch_grouped_mm_supported():
        pytest.skip("torch._grouped_mm unsupported on this device")
    cfg = Qwen4ExpTextConfig(
        hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16,
        linear_value_head_dim=16, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16,
        shared_expert_intermediate_size=16, vocab_size=128, hc_count=2, hc_lowrank=8,
        ple_layer_ids=[], ngram_vocab_size_base=100,
    )
    torch.manual_seed(0)
    experts = modeling.Qwen4ExpTextExperts(cfg).to("cuda", torch.bfloat16)
    for p in experts.parameters():
        torch.nn.init.normal_(p, 0, 0.1)
    x = torch.randn(6, 64, device="cuda")
    idx = torch.randint(0, 4, (6, 2), device="cuda")
    w = torch.rand(6, 2, device="cuda")
    with torch.no_grad():
        ref = moe_utils.forward_native_grouped_mm(experts, x.bfloat16(), idx, w)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = moe_utils.forward_native_grouped_mm(experts, x, idx, w)
    torch.testing.assert_close(out.float(), ref.float())
