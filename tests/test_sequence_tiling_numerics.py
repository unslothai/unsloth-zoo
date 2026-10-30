# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""sequence_tiling.tiled_call vs the untiled module: loss, input grad and every parameter grad; CPU only.
Frozen-input LoRA grads, width-changing projectors, router-logit grads and once-per-backward grad hooks are the cases
the removed tiled_mlp.TiledMLP got wrong."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from _sequence_tiling_models import (  # noqa: E402
    GatedMLP, TopKMoE, load_balancing_loss, lora_mlp, grads, H,
)
from unsloth_zoo.sequence_tiling import tiled_call, bucket_tiles  # noqa: E402

TOL = {torch.float32: (1e-5, 1e-6), torch.bfloat16: (2e-2, 2e-2), torch.float16: (2e-3, 2e-3)}


def _run(module, x, call, loss_fn = None):
    module.zero_grad(set_to_none = True)
    if x.requires_grad: x.grad = None
    out = call(module, x)
    y = out[0] if isinstance(out, tuple) else out
    loss = (y.float() ** 2).mean() if loss_fn is None else loss_fn(out)
    loss.backward()
    return loss.detach(), (None if x.grad is None else x.grad.clone()), grads(module), out


def _assert_close(a, b, dtype = torch.float32):
    rtol, atol = TOL[dtype]
    la, ga, pa, _ = a
    lb, gb, pb, _ = b
    torch.testing.assert_close(la, lb, rtol = rtol, atol = atol)
    if ga is None: assert gb is None
    else: torch.testing.assert_close(ga, gb, rtol = rtol, atol = atol)
    assert pa.keys() == pb.keys()
    for k in pa:
        if pa[k] is None:
            assert pb[k] is None, k
        else:
            assert pb[k] is not None, f"{k}: no gradient from the tiled call"
            torch.testing.assert_close(pa[k], pb[k], rtol = rtol, atol = atol, msg = k)


def _new(n_tiles, **kw):
    return lambda m, x: tiled_call(m, x, n_tiles = n_tiles, needs_rng = False, **kw)


@pytest.mark.parametrize("shape", [(2, 37, H), (1, 37, H), (37, H)])
@pytest.mark.parametrize("n_tiles", [1, 2, 3, 7, 64])
@pytest.mark.parametrize("bias", [False, True])
def test_dense_parity_fp32(shape, n_tiles, bias):
    torch.manual_seed(0)
    m = GatedMLP(bias = bias)
    x = torch.randn(*shape, requires_grad = True)
    ref = _run(m, x, lambda m, x: m(x))
    new = _run(m, x, _new(n_tiles))
    _assert_close(ref, new)
    assert new[3].shape == ref[3].shape


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_dtypes(dtype):
    torch.manual_seed(0)
    m = GatedMLP().to(dtype)
    x = torch.randn(2, 33, H, dtype = dtype, requires_grad = True)
    _assert_close(_run(m, x, lambda m, x: m(x)), _run(m, x, _new(4)), dtype)


def test_fp16_autocast_gradscaler_skip():
    # GradScaler sees the same inf in the tiled grads and skips the same step.
    torch.manual_seed(0)
    res = []
    for call in (lambda m, x: m(x), _new(4)):
        m = GatedMLP()
        torch.manual_seed(1); m = GatedMLP()
        opt = torch.optim.SGD(m.parameters(), lr = 0.1)
        scaler = torch.amp.GradScaler("cpu", init_scale = 2.0 ** 16)
        x = torch.randn(2, 16, H)
        x[0, 3, 0] = float("inf")
        with torch.autocast("cpu", dtype = torch.float16):
            y = call(m, x)
        scaler.scale(y.float().mean()).backward()
        before = m.down_proj.weight.detach().clone()
        scaler.step(opt); scaler.update()
        res.append((torch.equal(before, m.down_proj.weight), scaler.get_scale()))
    assert res[0] == res[1] and res[0][0] is True


def test_frozen_input_mlp_only_lora():
    # Layer 0, MLP-only LoRA: the MLP input does not require grad.
    m = lora_mlp()
    x = torch.randn(2, 40, H)
    ref = _run(m, x, lambda m, x: m(x))
    assert ref[2]["down_proj.lora_B.weight"] is not None
    _assert_close(ref, _run(m, x, _new(4)))


def test_full_ft_and_frozen_base():
    for m in (GatedMLP(), lora_mlp()):
        x = torch.randn(3, 21, H, requires_grad = True)
        _assert_close(_run(m, x, lambda m, x: m(x)), _run(m, x, _new(5)))


def test_width_changing_projector():
    torch.manual_seed(0)
    m = GatedMLP(hidden = H, inter = 40, out = 24)  # vision projector: H -> 24
    x = torch.randn(2, 30, H, requires_grad = True)
    ref = _run(m, x, lambda m, x: m(x))
    new = _run(m, x, _new(4))
    _assert_close(ref, new)
    assert new[3].shape == (2, 30, 24)


def _moe_loss(out, w = 0.37):
    y, logits = out
    return (y.float() ** 2).mean() + w * load_balancing_loss(logits, 4, 2) + 0.1 * (logits.float() ** 2).mean()


def test_moe_router_logits_and_aux_loss():
    torch.manual_seed(0)
    m = TopKMoE()
    x = torch.randn(2, 29, H, requires_grad = True)
    ref = _run(m, x, lambda m, x: m(x), _moe_loss)
    new = _run(m, x, _new(4, output_contract = ("rows", "rows_flat")), _moe_loss)
    _assert_close(ref, new)
    assert new[3][1].shape == ref[3][1].shape == (58, 4)
    # Inferred contract gives the same.
    _assert_close(ref, _run(m, x, _new(4), _moe_loss))


def test_sum_contract_and_token_args():
    torch.manual_seed(0)
    m = GatedMLP()
    x = torch.randn(2, 25, H, requires_grad = True)
    scale = torch.rand(2, 25, 1)

    def fn(x, s):
        y = m(x) * s
        return y, (y.float() ** 2).sum()

    y_ref, aux_ref = fn(x, scale)
    y_new, aux_new = tiled_call(fn, x, scale, n_tiles = 3, needs_rng = False, output_contract = ("rows", "sum"))
    torch.testing.assert_close(y_new, y_ref)
    torch.testing.assert_close(aux_new, aux_ref, rtol = 1e-5, atol = 1e-6)
    gr = torch.autograd.grad(aux_ref + y_ref.sum(), [x] + list(m.parameters()))
    gn = torch.autograd.grad(aux_new + y_new.sum(), [x] + list(m.parameters()))
    for a, b in zip(gr, gn): torch.testing.assert_close(a, b, rtol = 1e-5, atol = 1e-5)


def test_scalar_output_refused_without_contract():
    m = GatedMLP()
    with pytest.raises(ValueError):
        tiled_call(lambda x: (m(x), m(x).sum()), torch.randn(2, 8, H), n_tiles = 2)


def _tiled_plain(m, x, n_tiles):
    # Same tiles, plain autograd (no checkpoint): the RNG draw order the checkpointed tiles must replay.
    return torch.cat([m(t) for t in torch.tensor_split(x.reshape(1, -1, x.shape[-1]), n_tiles, dim = 1)], 1).reshape(
        *x.shape[:-1], -1)


@pytest.mark.parametrize("needs_rng", [True, False])
def test_dropout_rng_parity(needs_rng):
    torch.manual_seed(0)
    m = GatedMLP(dropout = 0.3).train()
    x = torch.randn(2, 30, H, requires_grad = True)
    torch.manual_seed(123); ref = _run(m, x, lambda m, x: _tiled_plain(m, x, 4))
    torch.manual_seed(123)
    new = _run(m, x, lambda m, x: tiled_call(m, x, n_tiles = 4, needs_rng = needs_rng))
    torch.testing.assert_close(ref[0], new[0])  # forward draws identical masks
    if needs_rng:
        _assert_close(ref, new)
    else:
        # Without the RNG replay the recompute draws fresh masks: the gradients are wrong (proves the test is live).
        assert not torch.allclose(ref[2]["gate_proj.weight"], new[2]["gate_proj.weight"])


def test_no_grad_path_matches_and_skips_checkpoint(monkeypatch):
    import unsloth_zoo.sequence_tiling as st
    m = GatedMLP()
    x = torch.randn(2, 30, H)
    calls = []
    real = st._checkpoint
    monkeypatch.setattr(st, "_checkpoint", lambda *a: calls.append(1) or real(*a))
    with torch.no_grad():
        torch.testing.assert_close(tiled_call(m, x, n_tiles = 4), m(x))
    assert calls == []
    with torch.inference_mode():
        torch.testing.assert_close(tiled_call(m, x, n_tiles = 4), m(x))
    assert calls == []


def test_single_tile_is_direct_call():
    seen = []
    def fn(x):
        seen.append(x)
        return x * 2
    x = torch.randn(1, 1, H)
    tiled_call(fn, x, n_tiles = 1)
    tiled_call(fn, x, n_tiles = 8)  # clamped to 1 row
    assert all(s is x for s in seen) and len(seen) == 2


def test_param_grad_hooks_fire_once():
    m = lora_mlp()
    fired = {}
    for n, p in m.named_parameters():
        if p.requires_grad:
            p.register_post_accumulate_grad_hook(lambda p, n = n: fired.__setitem__(n, fired.get(n, 0) + 1))
    x = torch.randn(2, 40, H, requires_grad = True)
    tiled_call(m, x, n_tiles = 4, needs_rng = False).pow(2).mean().backward()
    assert fired and all(v == 1 for v in fired.values()), fired


def _layer(m, tiles):
    def layer(h):
        h = h * 1.5 + 0.1
        return h + (tiled_call(m, h, n_tiles = tiles, needs_rng = False) if tiles else m(h))
    return layer


@pytest.mark.parametrize("gc", ["torch_reentrant", "unsloth_reentrant", "torch_nonreentrant", "nonreentrant_offload"])
def test_inside_outer_gradient_checkpointing(gc, monkeypatch):
    from torch.utils.checkpoint import checkpoint as torch_ckpt
    if gc == "unsloth_reentrant" and not torch.cuda.is_available():
        pytest.skip("UnslothCheckpointFunction pins host buffers: needs CUDA (run via the GPU correctness test)")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = lora_mlp().to(dev)
    x = torch.randn(2, 40, H, device = dev, requires_grad = True)  # reentrant GC needs a grad-requiring input (enable_input_require_grads)

    def outer(fn, h):
        if gc == "torch_reentrant":
            return torch_ckpt(fn, h, use_reentrant = True)
        if gc == "unsloth_reentrant":
            from unsloth_zoo.gradient_checkpointing import unsloth_checkpoint, initialize_unsloth_gradient_checkpointing
            initialize_unsloth_gradient_checkpointing(torch.float32)
            # The shim replaces torch's checkpoint process-wide: the engine must still use the non-reentrant one.
            monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", unsloth_checkpoint)
            return unsloth_checkpoint(fn, h)
        if gc == "torch_nonreentrant":
            return torch_ckpt(fn, h, use_reentrant = False)
        from unsloth_zoo.gradient_checkpointing import non_reentrant_offload_hooks
        with non_reentrant_offload_hooks():
            return torch_ckpt(fn, h, use_reentrant = False)

    ref = _run(m, x, lambda m, x: _layer(m, 0)(x))
    new = _run(m, x, lambda m, x: outer(_layer(m, 4), x))
    _assert_close(ref, new)


def test_unsloth_shim_installed_frozen_input(monkeypatch):
    from unsloth_zoo.gradient_checkpointing import unsloth_checkpoint
    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", unsloth_checkpoint)
    m = lora_mlp()
    x = torch.randn(2, 40, H)
    _assert_close(_run(m, x, lambda m, x: m(x)), _run(m, x, _new(4)))


def test_bucket():
    assert bucket_tiles(100, 128) == 1
    assert bucket_tiles(129, 128) == 2
    assert bucket_tiles(1000, 128) == 8
    assert bucket_tiles(10 ** 7, 128, cap = 64) == 64


@pytest.mark.parametrize("frozen", [False, True])
def test_offloader_never_sees_tile_views(frozen, monkeypatch):
    # Non-reentrant GC + offload hooks: the offloader packs the outer checkpoint's inputs only. The tile inputs are saved
    # inside the outer checkpoint's own hooks (forward) and recompute storage (backward), so they are never offloaded,
    # and the pack calls are the same as untiled.
    import unsloth_zoo.gradient_checkpointing as gcm
    from torch.utils.checkpoint import checkpoint as torch_ckpt
    off = gcm.NonReentrantActivationOffloader()
    seen = []
    real_pack = off.pack
    monkeypatch.setattr(off, "pack", lambda x: (seen.append((tuple(x.shape), x.storage_offset())), real_pack(x))[1])
    m = lora_mlp()
    res = {}
    for tiles in (0, 4):
        seen.clear()
        x = torch.randn(2, 40, H, requires_grad = not frozen)
        with off.hooks():
            y = torch_ckpt(_layer(m, tiles), x, use_reentrant = False)
        y.pow(2).mean().backward()
        res[tiles] = list(seen)
    assert res[0] == res[4], res
    assert all(off == 0 for _, off in res[4])
