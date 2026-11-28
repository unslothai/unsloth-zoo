# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""Vendored fla gate-gradient numerics: fp32 h * dh reduction (GDN chunk_o, KDA chunk_bwd),
g_gamma partial-chunk mask, fp32 gate-bias gradient, saturated KDA sigmoid derivative and
sequence ids with zero-length segments."""

import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
os.environ.setdefault("UNSLOTH_VENDORED_FLA_NO_AUTORUN", "1")

try:
    from unsloth_zoo.temporary_patches import fla_vendor
except ImportError as _e:
    pytest.skip(f"unsloth_zoo unavailable: {_e}", allow_module_level = True)

ZOO_ROOT = pathlib.Path(__file__).resolve().parents[1]
VENDORED = ZOO_ROOT / "unsloth_zoo" / "_vendored" / "fla"


def _gpu_ok():
    if not fla_vendor._vendored_injection_supported() or fla_vendor._gpu_lacks_dot_instructions():
        return False
    import torch
    return getattr(torch.version, "hip", None) is None  # fla.ops.kda is withheld on ROCm


_GPU = pytest.mark.skipif(not _gpu_ok(), reason = "needs the vendored fla Triton kernels on CUDA")


def _read(rel):
    return (VENDORED / rel).read_text(encoding = "utf-8")


def test_gate_grad_reductions_are_fp32_in_source():
    assert "b_dg_last += tl.sum(b_h.to(tl.float32) * b_dh.to(tl.float32))" in _read("ops/common/chunk_o.py")
    assert "b_dgk += tl.sum(b_h.to(tl.float32) * b_dh.to(tl.float32), axis=0)" in _read("ops/kda/chunk_bwd.py")
    assert "tl.sum(b_h * b_dh" not in _read("ops/common/chunk_o.py")
    assert "tl.sum(b_h * b_dh" not in _read("ops/kda/chunk_bwd.py")


def test_gate_bias_grad_is_reduced_before_the_downcast():
    for rel, view in (("ops/gated_delta_rule/gate.py", "(-1, H)"), ("ops/kda/gate.py", "(-1, H * K)")):
        code = _read(rel)
        bias = code.index(f"dbias = dg.view{view}.sum(0)")
        cast = code.index("dg = dg.view_as(g).type_as(g)")
        assert bias < cast, rel


def test_kda_sigmoid_derivative_does_not_saturate_in_source():
    code = _read("ops/kda/gate.py")
    assert "b_sig * (1.0 - b_sig)" not in code
    assert "b_dsig = b_r / ((1.0 + b_r) * (1.0 + b_r))" in code


def test_g_gamma_partial_chunk_is_masked_in_source():
    code = _read("ops/common/chunk_h.py")
    assert "b_v = (b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]).to(b_v.dtype)" in code


_SEQ_IDS_CHILD = textwrap.dedent(
    """
    import importlib.util, sys, types, torch
    # Load ops/utils/index.py alone: the rest of the package needs a GPU.
    root = sys.argv[1]
    fla = types.ModuleType("fla"); fla.__path__ = [root]; sys.modules["fla"] = fla
    utils = types.ModuleType("fla.utils")
    utils.tensor_cache = lambda f: f
    utils.autotune_cache_kwargs = {}
    sys.modules["fla.utils"] = utils
    spec = importlib.util.spec_from_file_location("fla.ops.utils.index", root + "/ops/utils/index.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    for cu in ([0, 0, 3, 5], [0, 3, 3, 3, 7], [0, 2, 4, 4]):
        cu = torch.tensor(cu)
        lens = torch.diff(cu)
        want = torch.repeat_interleave(torch.arange(lens.numel()), lens)
        got = mod.prepare_sequence_ids(cu)
        assert torch.equal(got.long(), want), (cu.tolist(), got.tolist(), want.tolist())
        tok = mod.prepare_token_indices(cu)
        assert torch.equal(tok[:, 0].long(), want), (cu.tolist(), tok.tolist())
    print("SEQ_IDS_OK")
    """
)


def test_sequence_ids_skip_zero_length_segments():
    proc = subprocess.run(
        [sys.executable, "-c", _SEQ_IDS_CHILD, str(VENDORED)],
        capture_output = True, text = True, timeout = 600,
    )
    assert "SEQ_IDS_OK" in proc.stdout, f"stdout=\n{proc.stdout}\nstderr=\n{proc.stderr[-4000:]}"


_GPU_CHILD = textwrap.dedent(
    """
    import os, sys
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    import torch
    import unsloth_zoo
    from unsloth_zoo.temporary_patches.fla_vendor import patch_vendor_fla
    patch_vendor_fla()
    import fla
    import fla.ops.common.chunk_o as chunk_o
    import fla.ops.common.chunk_h as chunk_h
    import fla.ops.kda.chunk_bwd as kda_bwd
    import fla.ops.kda.gate as kda_gate
    import fla.ops.gated_delta_rule.gate as gdn_gate
    vendored = sys.argv[1]
    # The kernels under test must be the vendored files of this checkout.
    for m in (unsloth_zoo, fla, chunk_o, chunk_h, kda_bwd, kda_gate, gdn_gate):
        assert os.path.realpath(m.__file__).startswith(os.path.realpath(sys.argv[2])), (m.__name__, m.__file__)
    for m in (chunk_o, chunk_h, kda_bwd, kda_gate, gdn_gate):
        assert os.path.realpath(m.__file__).startswith(os.path.realpath(vendored)), (m.__name__, m.__file__)
    print("chunk_o from", chunk_o.__file__)
    dev = "cuda"
    failures = []
    def check(ok, what):
        if not ok:
            failures.append(what)
            print("FAIL", what)
    rel = lambda a, b: float((a.double() - b.double()).norm() / b.double().norm())

    # 1. chunk_bwd_dqkwg: dg from the h * dh tile sum. The reference runs the same kernel on
    #    fp32 copies of the 16-bit inputs, so only in-kernel arithmetic differs.
    def dqkwg(dtype, shift, seed = 0):
        g0 = torch.Generator(device = dev).manual_seed(seed)
        B, T, H, D, BT = 1, 128, 2, 128, 64
        NT = T // BT
        mk = lambda *s: torch.randn(*s, device = dev, generator = g0) * 0.2
        q, k, v, do, w, dv = (mk(B, T, H, D) for _ in range(6))
        h = torch.randn(B, NT, H, D, D, device = dev, generator = g0) + shift
        dh = torch.randn(B, NT, H, D, D, device = dev, generator = g0) + shift
        g = -torch.rand(B, T, H, device = dev, generator = g0) * 0.1
        xs = [t.to(dtype) for t in (q, k, v, do, h, dh, w, dv)]
        run = chunk_o.chunk_bwd_dqkwg(*xs[:3], do = xs[3], h = xs[4], dh = xs[5], w = xs[6], dv = xs[7],
                                      g = g, scale = D ** -0.5, chunk_size = BT)
        f = [t.float() for t in xs]
        ref = chunk_o.chunk_bwd_dqkwg(*f[:3], do = f[3], h = f[4], dh = f[5], w = f[6], dv = f[7],
                                      g = g, scale = D ** -0.5, chunk_size = BT)
        return bool(torch.isfinite(run[3]).all()), rel(run[3], ref[3])

    fin, err = dqkwg(torch.float16, 4.5)
    print("dqkwg fp16 shift4.5 finite", fin, "err %.3e" % err)
    check(fin, "fp16 gate gradient overflowed")
    check(err < 1e-2, err)
    for shift in (0.0, 0.05, 1.0):
        fin, err = dqkwg(torch.bfloat16, shift)
        print("dqkwg bf16 shift", shift, "err %.3e" % err)
        check(fin and err < 5e-3, (shift, err))

    # 2. KDA chunk backward: dg vs the same kernels on fp32 copies of the bf16 inputs.
    from fla.ops.kda import chunk_kda
    def kda(seed, T = 1024, B = 1, H = 4, D = 128):
        g0 = torch.Generator(device = dev).manual_seed(seed)
        l2 = lambda x: torch.nn.functional.normalize(x, dim = -1)
        q = l2(torch.randn(B, T, H, D, device = dev, generator = g0)).bfloat16()
        k = l2(torch.randn(B, T, H, D, device = dev, generator = g0)).bfloat16()
        v = (torch.randn(B, T, H, D, device = dev, generator = g0) + 2.0).bfloat16()
        g = torch.nn.functional.logsigmoid(torch.randn(B, T, H, D, device = dev, generator = g0) + 6.0)
        beta = torch.sigmoid(torch.randn(B, T, H, device = dev, generator = g0)).bfloat16()
        do = (torch.randn(B, T, H, D, device = dev, generator = g0) + 2.0)
        out = {}
        for name, cast in (("ref", torch.float32), ("run", None)):
            xs = [(t.to(cast) if cast else t).clone().requires_grad_() for t in (q, k, v)]
            gg = g.clone().requires_grad_()
            bb = (beta.to(cast) if cast else beta).clone().requires_grad_()
            o, _ = chunk_kda(xs[0], xs[1], xs[2], gg, bb)
            (o.float() * do).sum().backward()
            out[name] = gg.grad
        return bool(torch.isfinite(out["run"]).all()), rel(out["run"], out["ref"])
    for seed in (0, 1):
        fin, err = kda(seed)
        print("kda bf16 seed", seed, "dg err %.3e" % err)
        check(fin and err < 5e-3, (seed, err))

    # 2b. The fused KDA backward kernel alone: the h * dh sum lands in dg on each chunk's last row.
    def kda_unit(dtype, shift, seed = 0):
        g0 = torch.Generator(device = dev).manual_seed(seed)
        B, T, H, D, BT = 1, 128, 2, 128, 64
        NT = T // BT
        mk = lambda *s: torch.randn(*s, device = dev, generator = g0) * 0.2
        q, k, v, v_new, do, dv = (mk(B, T, H, D) for _ in range(6))
        g = (-torch.rand(B, T, H, D, device = dev, generator = g0) * 0.1).view(B, NT, BT, H, D).cumsum(2).view(B, T, H, D)
        beta = torch.rand(B, T, H, device = dev, generator = g0)
        A = mk(B, T, H, BT)
        h = torch.randn(B, NT, H, D, D, device = dev, generator = g0) + shift
        dh = torch.randn(B, NT, H, D, D, device = dev, generator = g0) + shift
        def run(dt):
            c = lambda t: t.to(dtype).to(dt)
            return kda_bwd.chunk_kda_bwd_wy_dqkg_fused(c(q), c(k), c(v), c(v_new), g, c(beta), c(A), c(h), c(do), c(dh),
                                                       c(dv), scale = D ** -0.5, chunk_size = BT)[4]
        got, ref = run(dtype), run(torch.float32)
        last = torch.arange(BT - 1, T, BT, device = dev)
        return bool(torch.isfinite(got).all()), rel(got[:, last], ref[:, last])
    for shift in (0.0, 0.05):
        fin, err = kda_unit(torch.bfloat16, shift)
        print("kda unit bf16 shift", shift, "dg(last row) err %.3e" % err)
        check(fin and err < 5e-4, ("kda unit", shift, err))

    # 3. simple_gla with g_gamma at lengths that leave a partial last chunk.
    from fla.ops.simple_gla.chunk import chunk_simple_gla
    from fla.ops.simple_gla.fused_recurrent import fused_recurrent_simple_gla
    for T in (1024, 1000, 63):
        torch.manual_seed(0)
        H, D = 4, 64
        q, k, v = (torch.randn(2, T, H, D, device = dev) for _ in range(3))
        g_gamma = -(8 / H) * torch.arange(H, dtype = torch.float32, device = dev)
        # The state after a partial chunk only reaches the final state, so ask for it.
        o, ht = chunk_simple_gla(q, k, v, g_gamma = g_gamma, output_final_state = True)
        r, rt = fused_recurrent_simple_gla(q, k, v, g_gamma = g_gamma, output_final_state = True)
        print("g_gamma T", T, "o finite", bool(torch.isfinite(o).all()), "ht finite", bool(torch.isfinite(ht).all()))
        check(torch.isfinite(o).all() and torch.isfinite(ht).all(), ("g_gamma finite", T))
        check(rel(o, r) < 1e-3 and rel(ht, rt) < 1e-3, ("g_gamma", T, rel(o, r), rel(ht, rt)))

    # 4. Gate bias gradients vs fp64 autograd of the naive gate.
    def bias_err(kind):
        torch.manual_seed(0)
        if kind == "gdn":
            H = 16
            g = torch.randn(8, 4096, H, device = dev).bfloat16()
            A_log = torch.randn(H, device = dev)
            bias = torch.randn(H, device = dev)
            fn = lambda g_, a_, b_: gdn_gate.fused_gdn_gate(g_, a_, b_)
            ref = lambda g_, a_, b_: -a_.exp() * torch.nn.functional.softplus(g_ + b_)
        else:
            H, K = 4, 64
            g = torch.randn(4, 2048, H, K, device = dev).bfloat16()
            A_log = torch.randn(H, device = dev)
            bias = torch.randn(H * K, device = dev)
            fn = lambda g_, a_, b_: kda_gate.fused_kda_gate(g_, a_, b_)
            ref = lambda g_, a_, b_: -a_.exp().view(H, 1) * torch.nn.functional.softplus(g_ + b_.view(H, K))
        dy = torch.randn(g.shape, device = dev)
        dy = dy - dy.mean(dim = tuple(range(dy.dim() - (1 if kind == "gdn" else 2))), keepdim = True)
        b1 = bias.clone().requires_grad_()
        (fn(g.clone().requires_grad_(), A_log, b1).float() * dy).sum().backward()
        b2 = bias.double().clone().requires_grad_()
        (ref(g.double(), A_log.double(), b2) * dy.double()).sum().backward()
        return rel(b1.grad, b2.grad)
    for kind in ("gdn", "kda"):
        err = bias_err(kind)
        print("dbias", kind, "err %.3e" % err)
        check(err < 1e-4, (kind, err))

    # 5. KDA lower-bound gate in its saturated range: sigmoid'(x) is tiny, not 0.
    torch.manual_seed(0)
    H, K = 2, 64
    g = (torch.rand(1, 64, H, K, device = dev) * 6 + 18)
    A_log = torch.zeros(H, device = dev)
    gg = g.clone().requires_grad_()
    kda_gate.fused_kda_gate(gg, A_log, None, lower_bound = -5.0).sum().backward()
    gr = g.double().clone().requires_grad_()
    (-5.0 * torch.sigmoid(gr)).sum().backward()
    err = rel(gg.grad, gr.grad)
    print("kda saturated sigmoid grad err %.3e" % err, "zeros", int((gg.grad == 0).sum()))
    check(err < 1e-3, err)
    assert not failures, failures
    print("GATE_GRAD_OK")
    """
)


@_GPU
def test_vendored_gate_grads_match_reference_subprocess():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ZOO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _GPU_CHILD, str(VENDORED), str(ZOO_ROOT)],
        env = env, capture_output = True, text = True, timeout = 1800,
    )
    print(proc.stdout)
    assert "GATE_GRAD_OK" in proc.stdout, f"stdout=\n{proc.stdout}\nstderr=\n{proc.stderr[-4000:]}"
