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


"""load_in_fp8 on a 16-bit checkpoint: vLLM native online FP8 routing and the float8-safe finalize cast."""
import pytest
import torch

from unsloth_zoo.vllm_utils import _get_vllm_online_fp8_quantization, _vllm_online_fp8_moe_ignore
from unsloth_zoo.empty_model import _cast_unquantized_, _has_quantized_tensors


_BASE = ["fp8", "torchao", "bitsandbytes", "compressed-tensors"]
VLLM_METHODS = {
    "0.11.2": _BASE,
    "0.19.0": _BASE,
    "0.20.0": _BASE + ["online", "fp8_per_tensor", "fp8_per_block"],
    "0.24.0": _BASE + ["online", "fp8_per_tensor", "fp8_per_block", "fp8_per_channel"],
    "0.31.0": _BASE + ["online", "fp8_per_tensor", "fp8_per_block", "fp8_per_channel", "mxfp8"],
}


@pytest.mark.parametrize(
    "version, mode, expected",
    [
        ("0.11.2", "row", None),
        ("0.11.2", "block", None),
        ("0.19.0", "block", None),
        ("0.20.0", "block", "fp8_per_block"),
        ("0.20.0", "row", None),
        ("0.24.0", "row", "fp8_per_channel"),
        ("0.24.0", "block", "fp8_per_block"),
        ("0.31.0", "row", "fp8_per_channel"),
        ("0.31.0", "block", "fp8_per_block"),
    ],
)
def test_routing_table(version, mode, expected, monkeypatch):
    monkeypatch.delenv("UNSLOTH_VLLM_ONLINE_FP8", raising = False)
    assert _get_vllm_online_fp8_quantization(mode, VLLM_METHODS[version], (9, 0)) == expected


@pytest.mark.parametrize("version", ["0.24.0", "0.31.0"])
def test_blackwell_block_keeps_torchao(version, monkeypatch, capsys):
    # Online per-block FP8 crashes on SM100 (DeepGEMM UE8M0): block stays on torchao.
    monkeypatch.delenv("UNSLOTH_VLLM_ONLINE_FP8", raising = False)
    assert _get_vllm_online_fp8_quantization("block", VLLM_METHODS[version], (10, 0)) is None
    assert "cannot backprop" in capsys.readouterr().out
    assert _get_vllm_online_fp8_quantization("block", VLLM_METHODS["0.19.0"], (10, 0)) is None
    assert _get_vllm_online_fp8_quantization("row", VLLM_METHODS[version], (10, 0)) == "fp8_per_channel"
    monkeypatch.setenv("UNSLOTH_VLLM_ONLINE_FP8", "native")
    assert _get_vllm_online_fp8_quantization("block", VLLM_METHODS[version], (10, 0)) == "fp8_per_block"


@pytest.mark.parametrize(
    "config, expected",
    [
        ({"num_experts": 8}, True),
        ({"num_local_experts": 8}, True),
        ({"text_config": {"n_routed_experts": 64}}, True),
        ({"hidden_size": 64}, False),
    ],
)
def test_moe_online_ignore_keeps_routers_and_experts_16bit(config, expected):
    from types import SimpleNamespace
    def ns(d):
        return SimpleNamespace(**{k: ns(v) if isinstance(v, dict) else v for k, v in d.items()})
    ignore = _vllm_online_fp8_moe_ignore(ns(config))
    if not expected:
        assert ignore is None
        return
    import re
    regs = [re.compile(x[3:]) for x in ignore]
    hit = lambda name: any(r.match(name) for r in regs)  # vLLM: re.match
    for name in ("model.layers.0.mlp.gate", "model.layers.3.mlp.shared_expert_gate", "model.layers.0.mlp.experts"):
        assert hit(name), name
    for name in ("model.layers.0.mlp.gate_up_proj", "model.layers.0.self_attn.qkv_proj",
                 "model.layers.0.mlp.shared_expert.gate_up_proj", "model.layers.0.mlp.experts.w13"):
        assert not hit(name), name


def test_never_legacy_fp8_online():
    for methods in VLLM_METHODS.values():
        for mode in ("row", "block"):
            assert _get_vllm_online_fp8_quantization(mode, methods, (9, 0)) != "fp8"


def test_torchao_override(monkeypatch):
    monkeypatch.setenv("UNSLOTH_VLLM_ONLINE_FP8", "torchao")
    assert _get_vllm_online_fp8_quantization("block", VLLM_METHODS["0.31.0"], (9, 0)) is None


def test_bad_mode():
    with pytest.raises(ValueError):
        _get_vllm_online_fp8_quantization("tensor", VLLM_METHODS["0.31.0"])


def test_installed_vllm_matches_its_own_table(monkeypatch):
    monkeypatch.delenv("UNSLOTH_VLLM_ONLINE_FP8", raising = False)
    quant = pytest.importorskip("vllm.model_executor.layers.quantization")
    for mode, name in (("row", "fp8_per_channel"), ("block", "fp8_per_block")):
        got = _get_vllm_online_fp8_quantization(mode, device_capability = (9, 0))
        assert got == (name if name in quant.QUANTIZATION_METHODS else None)


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason = "no float8 dtype")
def test_finalize_cast_keeps_float8_and_ties():
    model = torch.nn.Module()
    model.q = torch.nn.Linear(4, 4, bias = False)
    model.q.weight = torch.nn.Parameter(torch.randn(4, 4).to(torch.float8_e4m3fn), requires_grad = False)
    model.q.register_buffer("weight_scale", torch.ones(4, 1, dtype = torch.float32))
    model.emb = torch.nn.Embedding(8, 4)
    model.head = torch.nn.Linear(4, 8, bias = False)
    model.head.weight = model.emb.weight  # tied
    model.norm = torch.nn.LayerNorm(4)
    model.register_buffer("pos", torch.arange(4))
    fp8_before = model.q.weight
    assert _has_quantized_tensors(model)
    _cast_unquantized_(model, "cpu", torch.bfloat16)
    assert model.q.weight is fp8_before and model.q.weight.dtype == torch.float8_e4m3fn
    assert model.emb.weight.dtype == torch.bfloat16 and model.norm.weight.dtype == torch.bfloat16
    assert model.head.weight is model.emb.weight
    assert model.pos.dtype == torch.int64
    assert model.q.weight_scale.dtype == torch.float32


def test_plain_model_reports_no_quantized_tensors():
    assert not _has_quantized_tensors(torch.nn.Linear(2, 2))
