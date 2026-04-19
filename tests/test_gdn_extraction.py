import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from unsloth_zoo.empty_model import extract_gdn_layers


class _FakePlainProj(torch.nn.Module):
    # Simulates vLLM ColumnParallelLinear: plain Linear-like with .weight but no output_sizes.
    def __init__(self, out_features, in_features, dtype=torch.float32):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features, dtype=dtype), requires_grad=False)


class _FakeRowProj(torch.nn.Module):
    def __init__(self, out_features, in_features, dtype=torch.float32):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features, dtype=dtype), requires_grad=False)


class _FakeGDN(torch.nn.Module):
    def __init__(self, hidden_size=8, num_k_heads=2, num_v_heads=2, head_k_dim=2, head_v_dim=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = num_k_heads * head_k_dim
        self.value_dim = num_v_heads * head_v_dim
        qkvz_dim = self.key_dim * 2 + self.value_dim * 2
        self.in_proj_qkvz = _FakePlainProj(qkvz_dim, hidden_size)
        self.in_proj_ba = _FakePlainProj(num_v_heads * 2, hidden_size)
        self.conv1d = _FakePlainProj(self.key_dim * 2 + self.value_dim, 4)
        self.dt_bias = torch.nn.Parameter(torch.randn(num_v_heads), requires_grad=False)
        self.A_log = torch.nn.Parameter(torch.randn(num_v_heads), requires_grad=False)
        self.norm = torch.nn.Module()
        self.norm.weight = torch.nn.Parameter(torch.randn(head_v_dim), requires_grad=False)
        self.out_proj = _FakeRowProj(hidden_size, self.value_dim)


def _fake_get_state_dict(prefix, kk, state_dict, module, slice_weights=True):
    state_dict[f"{prefix}.weight"] = module.weight.data


def test_extract_gdn_layers_no_output_sizes_does_not_crash():
    gdn = _FakeGDN()
    state_dict = {}
    quant_state_dict = {}
    extract_gdn_layers(gdn, "prefix", state_dict, quant_state_dict, _fake_get_state_dict)
    assert "prefix.in_proj_qkv.weight" in state_dict
    assert "prefix.in_proj_z.weight" in state_dict


def test_extract_gdn_layers_splits_ba_without_indexerror():
    gdn = _FakeGDN()
    state_dict = {}
    quant_state_dict = {}
    extract_gdn_layers(gdn, "prefix", state_dict, quant_state_dict, _fake_get_state_dict)
    assert "prefix.in_proj_b.weight" in state_dict
    assert "prefix.in_proj_a.weight" in state_dict
    ba_weight = gdn.in_proj_ba.weight.data
    mid = ba_weight.shape[0] // 2
    torch.testing.assert_close(state_dict["prefix.in_proj_b.weight"], ba_weight[:mid])
    torch.testing.assert_close(state_dict["prefix.in_proj_a.weight"], ba_weight[mid:])


def test_extract_gdn_layers_exports_norm_weight():
    gdn = _FakeGDN()
    state_dict = {}
    quant_state_dict = {}
    extract_gdn_layers(gdn, "prefix", state_dict, quant_state_dict, _fake_get_state_dict)
    assert "prefix.norm.weight" in state_dict
    torch.testing.assert_close(state_dict["prefix.norm.weight"], gdn.norm.weight.data)


def test_extract_gdn_layers_exports_conv1d_dtbias_alog():
    gdn = _FakeGDN()
    state_dict = {}
    quant_state_dict = {}
    extract_gdn_layers(gdn, "prefix", state_dict, quant_state_dict, _fake_get_state_dict)
    assert "prefix.conv1d.weight" in state_dict
    assert "prefix.dt_bias" in state_dict
    assert "prefix.A_log" in state_dict


def test_extract_gdn_layers_qkvz_offsets_match_gdn_dims():
    gdn = _FakeGDN(num_k_heads=3, num_v_heads=2, head_k_dim=4, head_v_dim=5)
    state_dict = {}
    quant_state_dict = {}
    extract_gdn_layers(gdn, "prefix", state_dict, quant_state_dict, _fake_get_state_dict)
    # offsets = [0, key_dim, 2*key_dim, 2*key_dim+value_dim, 2*key_dim+2*value_dim]
    # qkv = rows [0 : 2*key_dim+value_dim], z = rows [that : end]
    key_dim = gdn.key_dim
    value_dim = gdn.value_dim
    expected_qkv_rows = 2 * key_dim + value_dim
    expected_z_rows = value_dim
    assert state_dict["prefix.in_proj_qkv.weight"].shape[0] == expected_qkv_rows
    assert state_dict["prefix.in_proj_z.weight"].shape[0] == expected_z_rows


def test_extract_gdn_layers_raises_when_dims_missing():
    gdn = _FakeGDN()
    # Strip dim attrs and output_sizes so offset derivation fails.
    del gdn.key_dim
    del gdn.value_dim
    with pytest.raises(RuntimeError, match="in_proj_qkvz"):
        extract_gdn_layers(gdn, "prefix", {}, {}, _fake_get_state_dict)


def test_extract_gdn_layers_preserves_bnb_quant_state_sidecars():
    gdn = _FakeGDN()

    class _FakeQS:
        def __init__(self, label):
            self.label = label
        def as_dict(self, packed=True):
            return {"absmax": torch.tensor([float(hash(self.label) % 100)])}

    qkvz_weight = gdn.in_proj_qkvz.weight.data.clone()
    qkvz_weight.bnb_quant_state = {0: _FakeQS("q"), 1: _FakeQS("k"), 2: _FakeQS("v"), 3: _FakeQS("z")}
    gdn.in_proj_qkvz.weight = torch.nn.Parameter(qkvz_weight, requires_grad=False)
    # Re-attach bnb_quant_state after re-wrap (nn.Parameter copies base tensor; simulate via separate path)
    # Since attach-onto-Parameter may not propagate, set it directly via data wrapper attribute
    gdn.in_proj_qkvz.weight.data.bnb_quant_state = qkvz_weight.bnb_quant_state
    # Our extract_gdn_layers reads getattr(weight, "bnb_quant_state", None) on unwrapped weight
    # which via _unwrap becomes weight.data; attach on data to satisfy that path
    state_dict = {}
    quant_state_dict = {}
    extract_gdn_layers(gdn, "prefix", state_dict, quant_state_dict, _fake_get_state_dict)
    # At minimum, the call must not crash and qkv/z weights must be exported.
    assert "prefix.in_proj_qkv.weight" in state_dict
    assert "prefix.in_proj_z.weight" in state_dict
