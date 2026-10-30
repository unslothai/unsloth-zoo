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

"""Fused MoE router (moe_router_triton) tests vs an fp64 oracle, eager parity, finite differences, ties, extremes, torch.compile and opcheck; CUDA + Triton only."""

import math
import os

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level = True)

from unsloth_zoo.temporary_patches import moe_router_triton as mr

DEV = "cuda"
DTYPES = (torch.float32, torch.bfloat16, torch.float16)
TOL_W = {torch.float32: 2e-6, torch.bfloat16: 2e-6, torch.float16: 2e-6}
TOL_G = {torch.float32: 2e-6, torch.bfloat16: 1e-2, torch.float16: 2e-3}


def _same_ids(a, b):
    """Same selected set per row; order differs only for exactly tied values (equal weights), since torch.topk's CUDA
    sort is unstable while the kernel puts the lower index first."""
    return torch.equal(a.sort(-1)[0], b.sort(-1)[0])


def _canon(ids, w):
    """(ids, weights) sorted by expert id: k-order inside exact ties is not part of the contract."""
    i, order = ids.sort(-1)
    return i, w.gather(1, order)


def _wdiff(ids, w, ide, we):
    return (_canon(ids, w)[1] - _canon(ide, we)[1]).abs().max().item()


def _dw_for(ids, seed = 0):
    """Upstream weight gradient as a function of (token, expert), as the expert outputs give it."""
    g = torch.Generator(device = ids.device).manual_seed(seed)
    c = torch.randn(ids.shape[0], 4096, device = ids.device, generator = g)
    return c.gather(1, ids)


def _oracle(z, ids, mode, renorm, scale = 1.0):
    """fp64 weights and dlogits for fixed ids (selection is not differentiable)."""
    z = z.detach().double().requires_grad_(True)
    if mode == "softmax_topk":
        w = z.softmax(-1).gather(1, ids)
        if renorm:
            w = w / w.sum(-1, keepdim = True)
    elif mode == "topk_softmax":
        w = z.gather(1, ids).softmax(-1)
    else:
        w = z.sigmoid().gather(1, ids)
        if renorm:
            w = w / (w.sum(-1, keepdim = True) + 1e-20)
    return w * scale, z


def _cases():
    out = []
    for E in (8, 60, 64, 128, 256):
        for k in (1, 2, 8):
            if k > E:
                continue
            for mode in ("softmax_topk", "topk_softmax"):
                for renorm in ((False, True) if mode == "softmax_topk" else (False,)):
                    out.append((E, k, mode, renorm, 1, 1))
            out.append((E, k, "sigmoid", True, 1, 1))
            out.append((E, k, "sigmoid", False, 1, 1))
    for E, G, TG, k in ((256, 8, 4, 8), (64, 8, 3, 6), (60, 4, 2, 4), (160, 10, 3, 8), (128, 4, 1, 2)):
        out.append((E, k, "sigmoid", True, G, TG))
        out.append((E, k, "sigmoid", False, G, TG))
    return out


def _route(z, k, mode, renorm, G, TG, bias, scale, mask_zero = False):
    return mr.moe_router(z, k, mode, renorm, bias if mode == "sigmoid" else None, G, TG, scale, mask_zero)


def _eager(z, k, mode, renorm, G, TG, bias, scale, mask_zero = False):
    return mr.moe_router_reference(z, k, mode, renorm, bias if mode == "sigmoid" else None, G, TG, scale, mask_zero)


@pytest.mark.parametrize("dtype", DTYPES, ids = str)
@pytest.mark.parametrize("E,k,mode,renorm,G,TG", _cases())
def test_oracle_and_eager_parity(E, k, mode, renorm, G, TG, dtype):
    torch.manual_seed(E * 131 + k * 7 + G)
    T = 37
    scale = 2.5 if mode == "sigmoid" else 1.0
    z0 = (torch.randn(T, E, device = DEV) * 2).to(dtype)
    bias = (torch.randn(E, device = DEV) * 0.1) if mode == "sigmoid" else None
    z = z0.clone().requires_grad_(True)
    w, ids = _route(z, k, mode, renorm, G, TG, bias, scale)
    (dz,) = torch.autograd.grad(w, z, _dw_for(ids))
    assert w.dtype == torch.float32 and ids.dtype == torch.int64 and dz.dtype == dtype

    ze = z0.clone().requires_grad_(True)
    we, ide = _eager(ze, k, mode, renorm, G, TG, bias, scale)
    (dze,) = torch.autograd.grad(we, ze, _dw_for(ide))
    assert _same_ids(ids, ide), "ids differ from eager"
    assert _wdiff(ids, w, ide, we) <= TOL_W[dtype] * max(1.0, scale)
    assert (dz.float() - dze.float()).abs().max().item() <= TOL_G[dtype] * max(1.0, scale) * max(1.0, dze.float().abs().max().item())

    wo, zo = _oracle(z0, ids, mode, renorm, scale)
    (dzo,) = torch.autograd.grad(wo, zo, _dw_for(ids).double())
    assert (w.double() - wo).abs().max().item() <= 4e-6 * max(1.0, scale)
    assert (dz.double() - dzo).abs().max().item() <= TOL_G[dtype] * max(1.0, scale) * max(1.0, dzo.abs().max().item()) + 4e-6
    if mode == "softmax_topk" and not renorm and k < E:
        unsel = torch.ones_like(dzo, dtype = torch.bool).scatter_(1, ids, False)
        assert dzo[unsel].abs().max() > 0 and dz.double()[unsel].abs().max() > 0, "unselected softmax logits must get gradient"


@pytest.mark.parametrize("mask_zero", (False, True))
def test_group_mask_value(mask_zero):
    # Negative biases make the 0.0 fill (transformers 4) and the -inf fill (transformers 5) differ.
    torch.manual_seed(0)
    z = torch.randn(64, 64, device = DEV)
    bias = torch.randn(64, device = DEV) - 1.5
    w, ids = mr.moe_router(z, 8, "sigmoid", True, bias, 8, 2, 1.0, mask_zero)
    we, ide = mr.moe_router_reference(z, 8, "sigmoid", True, bias, 8, 2, 1.0, mask_zero)
    assert _same_ids(ids, ide)
    assert _wdiff(ids, w, ide, we) <= 1e-6


def test_hf_v5_sigmoid_router_exact_code():
    """The transformers 5 DeepSeek-V3 code verbatim (sorted=False) vs the fused op, ids as sets."""
    torch.manual_seed(1)
    T, E, G, TG, k = 50, 256, 8, 4, 8
    z = torch.randn(T, E, device = DEV)
    bias = torch.randn(E, device = DEV) * 0.05
    scores = z.sigmoid()
    sfc = scores + bias
    gs = sfc.view(-1, G, E // G).topk(2, dim = -1)[0].sum(dim = -1)
    gi = torch.topk(gs, k = TG, dim = -1, sorted = False)[1]
    gm = torch.zeros_like(gs).scatter_(1, gi, 1)
    sm = gm.unsqueeze(-1).expand(-1, G, E // G).reshape(-1, E)
    sfc = sfc.masked_fill(~sm.bool(), float("-inf"))
    ti = torch.topk(sfc, k = k, dim = -1, sorted = False)[1]
    tw = scores.gather(1, ti)
    tw = tw / (tw.sum(dim = -1, keepdim = True) + 1e-20) * 2.5
    w, ids = mr.moe_router(z, k, "sigmoid", True, bias, G, TG, 2.5)
    a, ia = ti.sort(-1)
    b, ib = ids.sort(-1)
    assert torch.equal(a, b)
    torch.testing.assert_close(w.gather(1, ib), tw.gather(1, ia), rtol = 1e-6, atol = 1e-6)


@pytest.mark.parametrize("mode", ("softmax_topk", "topk_softmax", "sigmoid"))
def test_finite_difference(mode):
    torch.manual_seed(3)
    T, E, k = 6, 60, 4
    renorm = mode != "topk_softmax"
    # Well separated logits so a +-h step never changes the selection.
    z = (torch.randperm(E, device = DEV).float() * 0.1 - 3).repeat(T, 1)
    z = z + torch.randn(T, 1, device = DEV) * 0.01
    z.requires_grad_(True)
    c = torch.randn(T, k, device = DEV)
    w, ids = mr.moe_router(z, k, mode, renorm)
    (dz,) = torch.autograd.grad((w * c).sum(), z)
    h = 1e-2
    zd = z.detach().double()
    for t, e in [(0, int(ids[0, 0])), (1, int(ids[1, -1])), (2, int((ids[2, 0] + 7) % E)), (3, 5)]:
        vals = []
        for sgn in (1, -1):
            zz = zd.clone()
            zz[t, e] += sgn * h
            wv, iv = mr.moe_router_reference(zz, k, mode, renorm)
            assert torch.equal(iv, ids), "step crossed a selection boundary"
            vals.append((wv * c.double()).sum().item())
        fd = (vals[0] - vals[1]) / (2 * h)
        assert abs(fd - dz[t, e].item()) <= 1e-4 + 1e-3 * abs(fd), (t, e, fd, dz[t, e].item())


@pytest.mark.parametrize("mode", ("softmax_topk", "topk_softmax", "sigmoid"))
def test_explicit_ties_match_torch_topk(mode):
    E, k = 64, 8
    rows = [
        torch.zeros(E),
        torch.ones(E) * 3,
        torch.arange(E).float().remainder(4),
        torch.tensor([1.0] * 3 + [2.0] * 10 + [0.5] * (E - 13)),
        torch.arange(E).float().flip(0).floor_divide(8),
    ]
    g = torch.Generator().manual_seed(5)
    for _ in range(20):
        rows.append(torch.randint(0, 5, (E,), generator = g).float())
    z = torch.stack(rows).to(DEV)
    for dtype in DTYPES:
        zz = z.to(dtype)
        w, ids = mr.moe_router(zz, k, mode, True)
        we, ide = mr.moe_router_reference(zz, k, mode, True)
        assert _same_ids(ids, ide), (dtype, ids[:3], ide[:3])
        assert _wdiff(ids, w, ide, we) <= 2e-6
    if mode == "sigmoid":
        bias = torch.zeros(E, device = DEV)
        w, ids = mr.moe_router(z, k, mode, True, bias, 8, 3, 1.0)
        we, ide = mr.moe_router_reference(z, k, mode, True, bias, 8, 3, 1.0)
        assert _same_ids(ids, ide)
        assert _wdiff(ids, w, ide, we) <= 2e-6


@pytest.mark.parametrize("mode", ("softmax_topk", "topk_softmax", "sigmoid"))
def test_extreme_logits(mode):
    torch.manual_seed(7)
    T, E, k = 33, 128, 8
    z = torch.randn(T, E, device = DEV) * 40
    z[0] = torch.linspace(-1e4, 1e4, E, device = DEV)
    z[1, :] = -80.0
    z[1, 17] = 80.0
    z[2] = torch.randn(E, device = DEV) * 1e3
    if mode == "softmax_topk":
        z[3, ::3] = float("-inf")
    for dtype in (torch.float32, torch.bfloat16):
        zz = z.to(dtype).requires_grad_(True)
        w, ids = mr.moe_router(zz, k, mode, mode != "topk_softmax")
        (dz,) = torch.autograd.grad(w, zz, _dw_for(ids))
        ze = z.to(dtype).requires_grad_(True)
        we, ide = mr.moe_router_reference(ze, k, mode, mode != "topk_softmax")
        (dze,) = torch.autograd.grad(we, ze, _dw_for(ide))
        assert torch.isfinite(w).all() and torch.isfinite(dz.float()).all()
        assert _same_ids(ids, ide)
        assert _wdiff(ids, w, ide, we) <= 1e-5
        torch.testing.assert_close(dz.float(), dze.float(), rtol = 1e-2, atol = 1e-5)


def test_aux_loss_gradient_kept():
    torch.manual_seed(11)
    z = torch.randn(20, 64, device = DEV, requires_grad = True)
    w, ids = mr.moe_router(z, 4, "softmax_topk", True)
    zloss = torch.logsumexp(z, -1).square().mean()
    (g1,) = torch.autograd.grad((w.sum() * 0.3 + zloss), z)
    ze = z.detach().clone().requires_grad_(True)
    we, _ = mr.moe_router_reference(ze, 4, "softmax_topk", True)
    (g2,) = torch.autograd.grad((we.sum() * 0.3 + torch.logsumexp(ze, -1).square().mean()), ze)
    torch.testing.assert_close(g1, g2, rtol = 1e-5, atol = 1e-6)


@pytest.mark.parametrize("mode", ("softmax_topk", "topk_softmax", "sigmoid"))
def test_compile_fullgraph_no_recompiles(mode):
    import torch._dynamo
    from torch._dynamo.utils import counters
    torch._dynamo.reset()
    counters.clear()
    bias = torch.randn(256, device = DEV) * 0.1 if mode == "sigmoid" else None
    G, TG = (8, 4) if mode == "sigmoid" else (1, 1)

    def f(x, W):
        logits = x @ W
        w, ids = mr.moe_router(logits, 8, mode, True, bias, G, TG, 1.5)
        return (w * ids.float()).sum() + logits.float().square().mean()

    cf = torch.compile(f, fullgraph = True)
    W = torch.randn(64, 256, device = DEV, requires_grad = True)
    for i, T in enumerate((17, 64, 129, 300, 1000)):
        x = torch.randn(T, 64, device = DEV)
        torch._dynamo.mark_dynamic(x, 0)
        out = cf(x, W)
        (gw,) = torch.autograd.grad(out, W)
        (gr,) = torch.autograd.grad(f(x, W), W)
        torch.testing.assert_close(gw, gr, rtol = 1e-4, atol = 1e-4)
    assert sum(counters["graph_break"].values()) == 0
    assert counters["stats"]["unique_graphs"] == 1, dict(counters["stats"])


@pytest.mark.parametrize("mode", ("softmax_topk", "topk_softmax", "sigmoid"))
def test_opcheck(mode):
    z = torch.randn(19, 64, device = DEV, requires_grad = True)
    bias = torch.randn(64, device = DEV) if mode == "sigmoid" else None
    args = (z, bias, 4, mr._MODES[mode], True, 4 if mode == "sigmoid" else 1, 2, 1.0, False)
    res = torch.library.opcheck(mr._router_fwd_op, args)
    assert all(v == "SUCCESS" for v in res.values()), res


def test_softmax_route_matches_modulelist_router():
    from unsloth_zoo.temporary_patches import moe_grouped_modulelist as gm

    class Blk:
        norm_topk_prob = True
    torch.manual_seed(2)
    z = torch.randn(40, 60, device = DEV, dtype = torch.bfloat16)
    old = os.environ.get("UNSLOTH_MOE_FUSED_ROUTER")
    try:
        os.environ["UNSLOTH_MOE_FUSED_ROUTER"] = "0"
        w0, i0 = gm._route_softmax_topk(Blk(), z, 4)
        n0 = mr.ROUTER_CALLS["fused"]
        os.environ["UNSLOTH_MOE_FUSED_ROUTER"] = "1"
        w1, i1 = gm._route_softmax_topk(Blk(), z, 4)
        assert mr.ROUTER_CALLS["fused"] == n0 + 1
    finally:
        if old is None:
            os.environ.pop("UNSLOTH_MOE_FUSED_ROUTER", None)
        else:
            os.environ["UNSLOTH_MOE_FUSED_ROUTER"] = old
    assert _same_ids(i0, i1)
    assert _wdiff(i0, w0, i1, w1) <= 2e-6


class _Env:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        self.old = os.environ.get("UNSLOTH_MOE_FUSED_ROUTER")
        os.environ["UNSLOTH_MOE_FUSED_ROUTER"] = self.value

    def __exit__(self, *a):
        if self.old is None:
            os.environ.pop("UNSLOTH_MOE_FUSED_ROUTER", None)
        else:
            os.environ["UNSLOTH_MOE_FUSED_ROUTER"] = self.old


def test_env_gate_compiles_fullgraph():
    import torch._dynamo
    from torch._dynamo.utils import counters
    torch._dynamo.reset()
    counters.clear()

    def f(z):
        r = mr.fused_route_or_none(z, 4, "softmax_topk", True)
        return r[0].sum() if r is not None else z.float().softmax(-1).topk(4)[0].sum()
    with _Env("1"):
        n = mr.ROUTER_CALLS["fused"]
        torch.compile(f, fullgraph = True, dynamic = True)(torch.randn(33, 60, device = DEV))
        assert mr.ROUTER_CALLS["fused"] == n + 1
    assert sum(counters["graph_break"].values()) == 0


_HF_ROUTERS = [
    ("qwen3_moe", "Qwen3MoeTopKRouter", "Qwen3MoeConfig", dict(num_experts = 60, num_experts_per_tok = 8, norm_topk_prob = True)),
    ("qwen3_moe", "Qwen3MoeTopKRouter", "Qwen3MoeConfig", dict(num_experts = 60, num_experts_per_tok = 8, norm_topk_prob = False)),
    ("qwen2_moe", "Qwen2MoeTopKRouter", "Qwen2MoeConfig", dict(num_experts = 60, num_experts_per_tok = 4, norm_topk_prob = False)),
    ("olmoe", "OlmoeTopKRouter", "OlmoeConfig", dict(num_experts = 64, num_experts_per_tok = 8, norm_topk_prob = False)),
    ("mixtral", "MixtralTopKRouter", "MixtralConfig", dict(num_local_experts = 8, num_experts_per_tok = 2)),
    ("qwen3_vl_moe", "Qwen3VLMoeTextTopKRouter", "Qwen3VLMoeTextConfig", dict(num_experts = 128, num_experts_per_tok = 8)),
    ("qwen3_5_moe", "Qwen3_5MoeTopKRouter", "Qwen3_5MoeTextConfig", dict(num_experts = 256, num_experts_per_tok = 8)),
    ("gpt_oss", "GptOssTopKRouter", "GptOssConfig", dict(num_local_experts = 32, num_experts_per_tok = 4)),
    ("deepseek_v3", "DeepseekV3TopkRouter", "DeepseekV3Config",
     dict(n_routed_experts = 256, num_experts_per_tok = 8, n_group = 8, topk_group = 4, norm_topk_prob = True, routed_scaling_factor = 2.5)),
    ("glm4_moe", "Glm4MoeTopkRouter", "Glm4MoeConfig",
     dict(n_routed_experts = 160, num_experts_per_tok = 8, n_group = 1, topk_group = 1, norm_topk_prob = True, routed_scaling_factor = 2.5)),
]


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16), ids = str)
@pytest.mark.parametrize("module_name,cls_name,cfg_name,over", _HF_ROUTERS)
def test_hf_router_class_parity(module_name, cls_name, cfg_name, over, dtype):
    import importlib
    try:
        m = importlib.import_module(f"transformers.models.{module_name}.modeling_{module_name}")
        cfgm = importlib.import_module(f"transformers.models.{module_name}.configuration_{module_name}")
        cls, cfg_cls = getattr(m, cls_name), getattr(cfgm, cfg_name)
    except Exception as e:
        pytest.skip(f"{cls_name} unavailable: {e}")
    orig = getattr(cls.forward, "__wrapped__", cls.forward)
    form = mr._router_form(type(cls_name, (cls,), {"forward": orig}))
    if form is None:
        pytest.skip(f"{cls_name} forward does not match a known form in this transformers")
    cfg = cfg_cls(hidden_size = 64, **over)
    torch.manual_seed(0)
    router = cls(cfg).to(DEV, dtype)
    torch.nn.init.normal_(router.weight, std = 0.3)
    if getattr(router, "bias", None) is not None and isinstance(router.bias, torch.nn.Parameter):
        torch.nn.init.normal_(router.bias, std = 0.3)
    if hasattr(router, "e_score_correction_bias"):
        router.e_score_correction_bias.data = torch.randn_like(router.e_score_correction_bias.float()) * 0.05
    x = torch.randn(77, 64, device = DEV, dtype = dtype)
    fused_fwd = mr._make_router_forward(form, orig)

    def run(fn):
        router.zero_grad(set_to_none = True)
        xx = x.clone().requires_grad_(True)
        out = fn(router, xx)
        logits, w, ids = out if len(out) == 3 else (None, out[1], out[0])
        loss = (w.float() * _dw_for(ids)).sum()
        if logits is not None:
            loss = loss + 1e-2 * torch.logsumexp(logits.float(), -1).square().mean()   # z-loss on the logits
        loss.backward()
        return logits, w, ids, router.weight.grad.float().clone(), xx.grad.float().clone()

    with _Env("1"):
        n = mr.ROUTER_CALLS["fused"]
        l1, w1, i1, g1, x1 = run(fused_fwd)
        assert mr.ROUTER_CALLS["fused"] == n + 1
    l0, w0, i0, g0, x0 = run(orig)
    assert w1.dtype == w0.dtype and i1.dtype == i0.dtype and w1.shape == w0.shape
    if l0 is not None:
        assert torch.equal(l0, l1)
    assert _same_ids(i0, i1)
    tol = 1e-2 if dtype == torch.bfloat16 else 2e-3
    assert _wdiff(i0, w0.float(), i1, w1.float()) <= tol * max(1.0, w0.float().abs().max().item())
    assert (g0 - g1).norm() <= 2e-2 * g0.norm() + 1e-6, ((g0 - g1).norm(), g0.norm())
    assert (x0 - x1).norm() <= 2e-2 * x0.norm() + 1e-6


@pytest.mark.parametrize("cls_name", ("GptOssTopKRouter",))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16), ids = str)
def test_zoo_gpt_oss_router_parity(cls_name, dtype):
    from unsloth_zoo.temporary_patches import gpt_oss as zg
    try:
        from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
    except Exception as e:
        pytest.skip(str(e))
    cfg = GptOssConfig(hidden_size = 64, num_local_experts = 32, num_experts_per_tok = 4)
    cfg.dtype = dtype
    torch.manual_seed(0)
    router = getattr(zg, cls_name)(cfg).to(DEV, dtype)
    with torch.no_grad():
        router.weight.normal_(0, 0.3)
        router.bias.normal_(0, 0.3)
    x = torch.randn(50, 64, device = DEV, dtype = dtype)

    def run(on):
        with _Env("1" if on else "0"):
            router.zero_grad(set_to_none = True)
            out = router(x)
            scores, ids = out[-2], out[-1]
            (scores.float() * torch.randn(scores.shape, device = DEV, generator = torch.Generator(DEV).manual_seed(1))).sum().backward()
            return scores, ids, router.weight.grad.float().clone()
    n = mr.ROUTER_CALLS["fused"]
    s1, i1, g1 = run(True)
    assert mr.ROUTER_CALLS["fused"] == n + 1
    s0, i0, g0 = run(False)
    assert s0.dtype == s1.dtype and _same_ids(i0, i1)
    assert (s0.float() - s1.float()).abs().max() <= (1e-2 if dtype == torch.bfloat16 else 2e-3)
    assert (g0 - g1).norm() <= 2e-2 * g0.norm() + 1e-6
