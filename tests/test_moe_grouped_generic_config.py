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

"""generic_gemm_config / generic_wgrad_config tiles: every table config vs fp64, GROUP_M / EVEN_K bitwise."""
import os

import pytest
import torch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level = True)
pytest.importorskip("triton")

from unsloth_zoo.temporary_patches import moe_grouped_fp16 as mg

DEV = torch.device("cuda", torch.cuda.current_device())
if not mg.fp16_grouped_available(DEV) or mg._capability(DEV) < (8, 0):
    pytest.skip("needs the Triton grouped GEMM on sm80+", allow_module_level = True)


@pytest.fixture(autouse = True)
def _triton(monkeypatch):
    monkeypatch.setenv("UNSLOTH_GPTOSS_FP16_GEMM", "triton")


def _counts(E, seed):
    g = torch.Generator().manual_seed(seed)
    c = torch.randint(0, 33, (E,), generator = g)
    c[0] = 0
    c[1] = 389
    c[-1] = 1
    return c.to(DEV, torch.int32)


def _ref_gemm(a, w, counts, b_trans):
    out, st = [], 0
    for e, c in enumerate(counts.tolist()):
        we = w[e].double()
        out.append(a[st:st + c].double() @ (we.T if b_trans else we))
        st += c
    return torch.cat(out, 0)


def _ref_wgrad(g, x, counts):
    out, st = [], 0
    for c in counts.tolist():
        out.append(g[st:st + c].double().T @ x[st:st + c].double())
        st += c
    return torch.stack(out, 0)


def _rel(y, ref):
    return ((y.double() - ref).norm() / ref.norm()).item()


def _tables_configs(kind):
    table = mg._GENERIC_GEMM if kind == "gemm" else mg._GENERIC_WGRAD
    out = set()
    for fam in table.values():
        for cls, rows in fam.items():
            for _, cfg in rows:
                out.add((cls, cfg))
    return sorted(out)


def _fits(cfg, elt, wgrad):
    BM, BN, BK, _, stages = cfg[:5]
    need = (stages - 1) * (BM * (BN + BK) if wgrad else BK * (BM + BN)) * elt
    return need <= mg._smem_limit(DEV)


def _shape_for(cls, big_n, big_k):
    return {"big": (big_n, big_k), "N": (16, big_k), "K": (big_n, 16)}[cls]


def _torch_err(fn, ref):
    if not hasattr(torch, "_grouped_mm"):
        return None
    try:
        return _rel(fn(), ref)
    except Exception:
        return None


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("b_trans", [True, False])
@pytest.mark.parametrize("cls_cfg", _tables_configs("gemm"), ids = lambda c: f"{c[0]}-{'_'.join(map(str, c[1]))}")
def test_table_gemm_configs_vs_fp64(cls_cfg, b_trans, dtype):
    cls, cfg = cls_cfg
    BM, BN, BK, warps, stages, group_m = cfg
    E = 8
    N, K = _shape_for(cls, 1408, 2880)   # odd tile tails in N and K
    if cls == "N":
        BN = 16
    if cls == "K":
        BK = 16
    cfg = (BM, BN, BK, warps, stages, group_m)
    if not _fits(cfg, 2, False):
        pytest.skip("over this GPU's shared memory")
    counts = _counts(E, 0)
    M = int(counts.sum())
    g = torch.Generator().manual_seed(1)
    a = torch.randn(M, K, generator = g).to(DEV, dtype)
    w = (torch.randn(*((E, N, K) if b_trans else (E, K, N)), generator = g) / K ** 0.5).to(DEV, dtype)
    ref = _ref_gemm(a, w, counts, b_trans)
    y = mg.grouped_gemm(a, w, counts, dtype, b_trans = b_trans, config = cfg)
    err = _rel(y, ref)
    offs = torch.cumsum(counts, 0, dtype = torch.int32)
    te = _torch_err(lambda: torch._grouped_mm(a, w.transpose(-2, -1) if b_trans else w, offs = offs), ref)
    bound = 1.02 * te if te else (4e-3 if dtype == torch.bfloat16 else 5e-4)
    assert err <= bound, (cfg, err, te)
    # GROUP_M / EVEN_K change only the walk order and the K masks: bitwise equal to the 2D grid.
    plain = mg.grouped_gemm(a, w, counts, dtype, b_trans = b_trans, config = cfg[:5])
    assert torch.equal(y, plain)
    swz = mg.grouped_gemm(a, w, counts, dtype, b_trans = b_trans, config = cfg[:5] + (8 - group_m,))
    assert torch.equal(swz, plain)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("cls_cfg", _tables_configs("wgrad"), ids = lambda c: f"{c[0]}-{'_'.join(map(str, c[1]))}")
def test_table_wgrad_configs_vs_fp64(cls_cfg, dtype):
    cls, cfg = cls_cfg
    BM, BN, BK, warps, stages = cfg
    E = 8
    N, K = _shape_for(cls, 1408, 2880)
    if cls == "N":
        BN = 16
    if cls == "K":
        BK = 16
    cfg = (BM, BN, BK, warps, stages)
    if not _fits(cfg, 2, True):
        pytest.skip("over this GPU's shared memory")
    counts = _counts(E, 2)
    M = int(counts.sum())
    gen = torch.Generator().manual_seed(3)
    gr = torch.randn(M, N, generator = gen).to(DEV, dtype)
    x = torch.randn(M, K, generator = gen).to(DEV, dtype)
    ref = _ref_wgrad(gr, x, counts)
    dw = mg.grouped_wgrad(gr, x, counts, dtype, num_experts = E, config = cfg)
    assert torch.equal(dw[0], torch.zeros_like(dw[0]))   # empty expert
    err = _rel(dw, ref)
    offs = torch.cumsum(counts, 0, dtype = torch.int32)
    te = _torch_err(lambda: torch._grouped_mm(gr.t(), x, offs = offs), ref)
    bound = 1.02 * te if te else (4e-3 if dtype == torch.bfloat16 else 5e-4)
    assert err <= bound, (cfg, err, te)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rank", [8, 16, 32, 64])
@pytest.mark.parametrize("rows", [16, 128, 1024])
def test_generic_configs_end_to_end(rows, rank, dtype):
    """The functions' own picks for this GPU, LoRA A / B shapes and a base shape."""
    E, H, I2 = 8, 2048, 1536
    counts = torch.full((E,), rows, dtype = torch.int32, device = DEV)
    counts[0] = 0
    counts[1] = rows * 2
    M = int(counts.sum())
    g = torch.Generator().manual_seed(rows + rank)
    for N, K in ((rank, H), (I2, rank), (I2, H)):
        a = torch.randn(M, K, generator = g).to(DEV, dtype)
        w = (torch.randn(E, N, K, generator = g) / K ** 0.5).to(DEV, dtype)
        cfg = mg.generic_gemm_config(M, E, N, K, DEV, dtype)
        y = mg.grouped_gemm(a, w, counts, dtype, b_trans = True, config = cfg)
        assert _rel(y, _ref_gemm(a, w, counts, True)) < (8e-3 if dtype == torch.bfloat16 else 1e-3)
        gr = torch.randn(M, N, generator = g).to(DEV, dtype)
        wc = mg.generic_wgrad_config(M, E, N, K, DEV, dtype)
        dw = mg.grouped_wgrad(gr, a, counts, dtype, num_experts = E, config = wc)
        assert _rel(dw, _ref_wgrad(gr, a, counts)) < (8e-3 if dtype == torch.bfloat16 else 1e-3)


def test_config_helpers(monkeypatch):
    for fam, cap, smem in (("sm80", (8, 0), 166912), ("sm89", (8, 9), 101376), ("sm120", (12, 0), 101376),
                           ("sm89", (8, 6), 101376), ("sm80", (9, 0), 232448), ("sm89", (8, 7), 49152)):
        monkeypatch.setitem(mg._SM, DEV.index, cap)
        monkeypatch.setitem(mg._SMEM, DEV.index, smem)
        assert mg._generic_family(DEV) == fam
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            elt = torch.empty((), dtype = dtype).element_size()
            for rows in (1, 16, 100, 256, 5000):
                for N, K in ((16, 2048), (32, 4096), (8, 768), (2816, 16), (4096, 32), (64, 4096), (28672, 4096), (40, 24)):
                    M = rows * 64
                    c = mg.generic_gemm_config(M, 64, N, K, DEV, dtype)
                    assert len(c) == 6 and all(isinstance(v, int) for v in c)
                    assert c[1] <= max(16, mg._pow2(N)) and c[2] <= max(16, mg._pow2(K))
                    assert (c[4] - 1) * c[2] * (c[0] + c[1]) * elt <= mg._SMEM[DEV.index]
                    w = mg.generic_wgrad_config(M, 64, N, K, DEV, dtype)
                    assert len(w) == 5 and all(isinstance(v, int) for v in w)
                    assert (w[4] - 1) * w[0] * (w[1] + w[2]) * elt <= mg._SMEM[DEV.index]
    monkeypatch.setitem(mg._SM, DEV.index, (7, 5))
    assert mg.generic_gemm_config(4096, 32, 2880, 2880, DEV, torch.float16) == mg._gemm_config(4096, 32, 2880, 2880, DEV, False) + (0,)


def test_config_none_is_the_1591_tile(monkeypatch):
    """config = None launches _gemm_config's 5-tuple (no GROUP_M / EVEN_K): the gpt-oss path."""
    seen = {}
    real = mg._grouped_gemm_kernel

    class _Spy:
        def __getitem__(self, grid):
            def launch(*args, **kw):
                seen["grid"], seen["kw"] = grid, kw
                return real[grid](*args, **kw)
            return launch

    monkeypatch.setattr(mg, "_grouped_gemm_kernel", _Spy())
    counts = torch.tensor([300, 0, 212], dtype = torch.int32, device = DEV)
    a = torch.randn(512, 2880, device = DEV, dtype = torch.float16)
    w = torch.randn(3, 2880, 2880, device = DEV, dtype = torch.float16)
    mg.grouped_gemm(a, w, counts, torch.float16, b_trans = True)
    BM, BN, BK, warps, stages = mg._gemm_config(512, 3, 2880, 2880, DEV, False)
    assert "GROUP_M" not in seen["kw"] and "EVEN_K" not in seen["kw"]
    assert len(seen["grid"]) == 2
    assert (seen["kw"]["BLOCK_M"], seen["kw"]["BLOCK_N"], seen["kw"]["BLOCK_K"]) == (BM, BN, BK)
