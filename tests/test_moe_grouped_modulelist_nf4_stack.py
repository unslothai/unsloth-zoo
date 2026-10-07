"""Pointer-table NF4 stacks for the transformers<5 ModuleList grouped MoE path.

_build_gate_up_stack / _build_down_stack build the frozen expert stacks from one Triton launch
per projection kind (gpt_oss_routed._build_table + nf4_dequant_expert_stack). They must be
bitwise equal to the bitsandbytes per-expert builders and fall back to them otherwise.
"""
import pytest
import torch
import torch.nn as nn

from unsloth_zoo.temporary_patches import moe_grouped_modulelist as mgm

bnb = pytest.importorskip("bitsandbytes")
pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and mgm.HAS_BNB and torch.version.hip is None),
    reason="needs CUDA + bitsandbytes",
)

QWEN = mgm._BLOCK_SPECS["Qwen3MoeSparseMoeBlock"]
MIXTRAL = mgm._BLOCK_SPECS["MixtralSparseMoeBlock"]


def _lin4(n_in, n_out, nested, qdtype, seed):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(n_out, n_in, generator=g) * 0.02
    lin = bnb.nn.Linear4bit(n_in, n_out, bias=False, compute_dtype=torch.bfloat16,
                            compress_statistics=nested, quant_type="nf4")
    lin.weight = bnb.nn.Params4bit(w.to(qdtype), requires_grad=False, compress_statistics=nested,
                                   quant_type="nf4")
    return lin.cuda()


def _experts(E, hidden, inter, spec, nested=True, qdtype=torch.bfloat16, seed=0):
    out = nn.ModuleList()
    for e in range(E):
        ex = nn.Module()
        setattr(ex, spec[0], _lin4(hidden, inter, nested, qdtype, seed + 3 * e))
        setattr(ex, spec[1], _lin4(hidden, inter, nested, qdtype, seed + 3 * e + 1))
        setattr(ex, spec[2], _lin4(inter, hidden, nested, qdtype, seed + 3 * e + 2))
        out.append(ex)
    return out


def _check(experts, spec, dtype=torch.bfloat16, expect_stacked=True):
    before = dict(mgm._NF4_STACK_CALLS)
    gu = mgm._build_gate_up_stack(experts, spec, dtype)
    dn = mgm._build_down_stack(experts, spec, dtype)
    ref_gu = mgm._bnb_build_gate_up_stack(experts, spec, dtype)
    ref_dn = mgm._bnb_build_down_stack(experts, spec, dtype)
    for got, ref in ((gu, ref_gu), (dn, ref_dn)):
        assert got.shape == ref.shape and got.dtype == ref.dtype and got.stride() == ref.stride()
        assert torch.equal(got.view(torch.int16), ref.view(torch.int16))
    stacked = mgm._NF4_STACK_CALLS["stacked"] - before["stacked"]
    assert stacked == (2 * expect_stacked if isinstance(expect_stacked, bool) else expect_stacked)


@pytest.mark.parametrize("nested", [True, False])
@pytest.mark.parametrize("E", [1, 3, 7, 8])
@pytest.mark.parametrize("spec", [QWEN, MIXTRAL], ids=["qwen3", "mixtral"])
def test_bitwise_vs_bnb(nested, E, spec):
    _check(_experts(E, 128, 192, spec, nested=nested, seed=E), spec)


@pytest.mark.parametrize("nested", [True, False])
def test_fp32_quant_state_rounds_once(nested):
    _check(_experts(5, 128, 64, QWEN, nested=nested, qdtype=torch.float32), QWEN)


def test_fp16_quant_state_falls_back():
    # bnb dequantizes to fp16 then rounds to bf16: two roundings, so the kernel declines.
    _check(_experts(3, 128, 64, QWEN, qdtype=torch.float16), QWEN, expect_stacked=False)


def test_fp16_target_dtype():
    _check(_experts(3, 128, 64, QWEN, qdtype=torch.float16), QWEN, dtype=torch.float16)


def test_peft_wrapped_experts():
    peft = pytest.importorskip("peft")
    holder = nn.Module()
    holder.experts = _experts(5, 128, 64, QWEN, seed=11)
    cfg = peft.LoraConfig(r=8, lora_alpha=16, target_modules=["gate_proj", "up_proj", "down_proj"])
    holder = peft.inject_adapter_in_model(cfg, holder)
    assert hasattr(holder.experts[0].gate_proj, "base_layer")
    # The bnb reference reads lin.weight; PEFT forwards .weight to the base layer.
    _check(holder.experts, QWEN)


def test_kill_switch_and_plain_weights(monkeypatch):
    experts = _experts(3, 128, 64, QWEN)
    monkeypatch.setenv("UNSLOTH_MOE_GROUPED_NF4_STACK", "0")
    _check(experts, QWEN, expect_stacked=False)
    monkeypatch.delenv("UNSLOTH_MOE_GROUPED_NF4_STACK")
    plain = nn.ModuleList()
    for _ in range(3):
        ex = nn.Module()
        for name, (i, o) in zip(QWEN[:3], ((128, 64), (128, 64), (64, 128))):
            setattr(ex, name, nn.Linear(i, o, bias=False).cuda().bfloat16().requires_grad_(False))
        plain.append(ex)
    _check(plain, QWEN, expect_stacked=False)


def test_table_rebuilt_after_weight_swap():
    experts = _experts(3, 128, 64, QWEN, seed=1)
    _check(experts, QWEN)
    donor = _experts(1, 128, 64, QWEN, seed=99)
    experts[1].up_proj = donor[0].up_proj      # new weight, new addresses
    experts[2].down_proj.weight.quant_state.absmax.mul_(2)   # in-place edit, same address
    _check(experts, QWEN)


def test_mixed_shapes_fall_back():
    experts = _experts(2, 128, 64, QWEN)
    experts[1].gate_proj = _lin4(128, 64, False, torch.bfloat16, 5)   # nested differs
    _check(experts, QWEN, expect_stacked=1)   # gate_up falls back, down still stacks
