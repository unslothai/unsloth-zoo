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

"""Chunk loops that checkpoint each chunk with use_reentrant = False (GRPO score centering, the sparse attention
reference, the CPU chunked log-probs, the EAGLE3 / DSpark / draft-head losses, the mHC projection) must reach
torch's own non-reentrant checkpoint while smart gradient checkpointing has replaced
torch.utils.checkpoint.checkpoint with unsloth_checkpoint. Through the replacement every chunk runs the
reentrant, CPU-offloading UnslothCheckpointFunction: it bumps LAST_GC_INDEX / CURRENT_GC_INDEX (so the
"skip offloading the last layer" slot lands on a head chunk instead of the last decoder layer) and drops the
grads of parameters a chunk only reaches through a closure. CPU only: the offload branch is kept out of reach."""

import functools
import importlib.util
import types

import pytest
import torch
import torch.utils.checkpoint as torch_checkpoint

gc = pytest.importorskip("unsloth_zoo.gradient_checkpointing")


def _install_smart_gc(monkeypatch):
    """unsloth_checkpoint installed as torch.utils.checkpoint.checkpoint, as patch_unsloth_smart_gradient_checkpointing
    does; the pinned offload buffers need an accelerator, so MINIMUM_SIZE keeps every activation on device."""
    gc._capture_pristine_checkpoint_once()
    pristine = torch_checkpoint._unsloth_pristine_checkpoint
    state = dict(
        LAST_GC_INDEX = 0, CURRENT_GC_INDEX = 0, FIRST_PASS = True, BACKWARD_PASS = False,
        MINIMUM_SIZE = 2 ** 62, CPU_BUFFERS = [], CPU_INDEX = 0, USE_DOUBLE_BUFFER = False,
    )
    for name, value in state.items():
        monkeypatch.setattr(gc, name, value, raising = False)
    offloads, pristine_calls = [], []
    apply = gc.UnslothCheckpointFunction.apply

    def counted_apply(*args):
        offloads.append(1)
        return apply(*args)

    @functools.wraps(pristine)
    def counted_pristine(*args, **kwargs):
        pristine_calls.append(1)
        return pristine(*args, **kwargs)

    monkeypatch.setattr(gc.UnslothCheckpointFunction, "apply", counted_apply)
    monkeypatch.setattr(torch_checkpoint, "_unsloth_pristine_checkpoint", counted_pristine)
    monkeypatch.setattr(torch_checkpoint, "_old_checkpoint", pristine, raising = False)
    monkeypatch.setattr(torch_checkpoint, "checkpoint", gc.unsloth_checkpoint)
    return types.SimpleNamespace(offloads = offloads, pristine_calls = pristine_calls)


@pytest.fixture
def smart_gc(monkeypatch):
    return _install_smart_gc(monkeypatch)


def _counters():
    return gc.LAST_GC_INDEX, gc.CURRENT_GC_INDEX, gc.FIRST_PASS


def _fresh(name):
    """A second copy of a zoo module, executed now: what a module imported after smart GC was installed binds."""
    spec = importlib.util.find_spec(name)
    spec = importlib.util.spec_from_file_location(name + "_fresh_copy", spec.origin)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gen():
    return torch.Generator().manual_seed(0)


def _score_centering(fresh):
    from unsloth_zoo.score_centering import grpo_topk_head_logprobs

    g = _gen()
    hidden = torch.randn(2, 6, 8, generator = g).requires_grad_()
    head = torch.randn(16, 8, generator = g).requires_grad_()
    ids = torch.randint(0, 16, (2, 6, 3), generator = g)
    return grpo_topk_head_logprobs(hidden, head, ids, chunks = 4).sum(), (hidden, head), 4


def _sparse_attention(fresh, monkeypatch):
    from unsloth_zoo import sparse_attention as sa

    monkeypatch.setattr(sa, "_TORCH_CHUNK_ELEMS", 3 * 4 * 4)  # 4 query rows per chunk
    g = _gen()
    q = torch.randn(1, 12, 2, 4, generator = g).requires_grad_()
    kv = torch.randn(1, 10, 4, generator = g).requires_grad_()
    sink = torch.randn(2, generator = g).requires_grad_()
    idxs = torch.randint(-1, 10, (1, 12, 3), generator = g)
    return sa.sparse_attn_reference(q, kv, sink, idxs, 0.5).sum(), (q, kv, sink), 3


def _cpu_log_probs(fresh):
    from unsloth_zoo.cpu.ops import _checkpointed_log_probs

    g = _gen()
    hidden = torch.randn(10, 8, generator = g).requires_grad_()
    weight = torch.randn(16, 8, generator = g).requires_grad_()
    targets = torch.randint(0, 16, (10,), generator = g)
    lp, ent, *_ = _checkpointed_log_probs(hidden, weight, None, targets, 1.0, None, 1.0, 3, ("entropy",))
    return lp.sum() + ent.sum(), (hidden, weight), 4


class _Drafter(torch.nn.Module):
    uses_draft_vocab = False

    def __init__(self):
        super().__init__()
        g = _gen()
        self.fc = torch.nn.Linear(8, 8, bias = False)
        self.head = torch.nn.Linear(8, 16, bias = False)  # only reached through the chunk's closure
        with torch.no_grad():
            self.fc.weight.copy_(torch.randn(8, 8, generator = g))
            self.head.weight.copy_(torch.randn(16, 8, generator = g))
        self.t2d = torch.arange(16)

    def logits(self, h):
        return self.head(h)


def _eagle3(fresh):
    from unsloth_zoo.speculative.eagle3 import eagle3_chunked_loss

    drafter = _Drafter()
    g = _gen()
    x = torch.randn(1, 9, 8, generator = g).requires_grad_()
    step_hidden = [drafter.fc(x)]
    final = torch.randn(1, 9, 8, generator = g)
    lm = torch.randn(16, 8, generator = g)
    loss, _ = eagle3_chunked_loss(drafter, step_hidden, final, lm, torch.ones(1, 9, dtype = torch.bool), chunk_rows = 4)
    return loss, (x, drafter.fc.weight, drafter.head.weight), 3


def _draft_head_losses(fresh):
    losses = _fresh("unsloth_zoo.speculative.losses") if fresh else __import__(
        "unsloth_zoo.speculative.losses", fromlist = ["chunked_lm_loss"]
    )
    g = _gen()
    hidden = torch.randn(20, 8, generator = g).requires_grad_()
    weight = torch.randn(16, 8, generator = g).requires_grad_()
    labels = torch.randint(0, 16, (20,), generator = g)
    loss_sum, _, _ = losses.chunked_lm_loss(hidden, weight, labels, "ce", chunk_size = 6)
    return loss_sum, (hidden, weight), 4


def _dspark(fresh):
    dspark = _fresh("unsloth_zoo.speculative.dspark") if fresh else __import__(
        "unsloth_zoo.speculative.dspark", fromlist = ["dspark_loss"]
    )
    g = _gen()
    lm_head = torch.nn.Linear(8, 16, bias = False)
    with torch.no_grad():
        lm_head.weight.copy_(torch.randn(16, 8, generator = g))
    drafter = types.SimpleNamespace(lm_head = lm_head, markov_head = None, confidence_head = None)
    hidden = torch.randn(1, 2, 4, 8, generator = g).requires_grad_()
    out = {
        "eval_mask": torch.ones(1, 2, 4, dtype = torch.bool),
        "hidden": hidden,
        "target_ids": torch.randint(0, 16, (1, 2, 4), generator = g),
        "prev_ids": torch.randint(0, 16, (1, 2, 4), generator = g),
    }
    loss, _ = dspark.dspark_loss(drafter, out, l1_alpha = 0.0, chunk_size = 3)
    return loss, (hidden, drafter.lm_head.weight), 3


def _mhc(fresh):
    mhc = _fresh("unsloth_zoo.temporary_patches.mhc_sinkhorn") if fresh else __import__(
        "unsloth_zoo.temporary_patches.mhc_sinkhorn", fromlist = ["unsloth_mhc_projection"]
    )
    g = _gen()
    streams = torch.randn(2, 3, 4, 8, generator = g).to(torch.bfloat16).requires_grad_()
    fn = torch.randn(5, 32, generator = g).requires_grad_()
    out = mhc.unsloth_mhc_projection(streams, torch.nn.Identity(), fn)
    out = out + mhc.unsloth_mhc_projection_rsqrt(streams, fn, 1e-6)
    return out.float().sum(), (streams, fn), 2


SITES = {
    "score_centering": _score_centering,
    "sparse_attention": _sparse_attention,
    "cpu_log_probs": _cpu_log_probs,
    "eagle3": _eagle3,
    "draft_head_losses": _draft_head_losses,
    "dspark": _dspark,
    "mhc_projection": _mhc,
}
# Sites that used to bind checkpoint at import: also checked as imported after smart GC was installed.
IMPORT_BOUND = ("draft_head_losses", "dspark", "mhc_projection")


def _run(site, monkeypatch, fresh = False):
    fn = SITES[site]
    if site == "sparse_attention":
        loss, leaves, chunks = fn(fresh, monkeypatch)
    else:
        loss, leaves, chunks = fn(fresh)
    loss.backward()  # not autograd.grad: the reentrant replacement refuses it, which would hide the counters
    return loss.detach(), [leaf.grad for leaf in leaves], chunks


CASES = [(site, False) for site in SITES] + [(site, True) for site in IMPORT_BOUND]


@pytest.mark.parametrize("site,fresh", CASES, ids = [f"{s}{'-imported_after_patch' if f else ''}" for s, f in CASES])
def test_chunk_loop_uses_torch_checkpoint_under_smart_gc(site, fresh, monkeypatch):
    want_loss, want_grads, _ = _run(site, monkeypatch)  # torch's own checkpoint installed
    with monkeypatch.context() as patch:
        engaged = _install_smart_gc(patch)
        before = _counters()
        loss, grads, chunks = _run(site, patch, fresh = fresh)
        after = _counters()
    assert engaged.offloads == [], f"{site}: {len(engaged.offloads)} chunks ran UnslothCheckpointFunction"
    assert len(engaged.pristine_calls) == chunks, (site, len(engaged.pristine_calls), chunks)
    assert after == before, f"{site}: offload counters moved {before} -> {after}"
    torch.testing.assert_close(loss, want_loss, rtol = 0, atol = 0)
    for i, (got, want) in enumerate(zip(grads, want_grads)):
        assert got is not None and want is not None, (site, i)
        torch.testing.assert_close(got, want, rtol = 1e-6, atol = 1e-6, msg = f"{site} grad {i}")


def test_head_chunks_leave_the_last_layer_slot_to_the_last_decoder_layer(smart_gc):
    """Three checkpointed decoder layers then the score-centering head chunks, two steps: LAST_GC_INDEX must count
    the layers only, so the next step still skips offloading the last decoder layer."""
    from unsloth_zoo.score_centering import grpo_topk_head_logprobs

    g = _gen()
    weights = [torch.randn(8, 8, generator = g).requires_grad_() for _ in range(3)]
    head = torch.randn(16, 8, generator = g).requires_grad_()
    ids = torch.randint(0, 16, (2, 6, 3), generator = g)
    for _ in range(2):
        h = torch.randn(2, 6, 8, generator = g).requires_grad_()
        for w in weights:
            h = torch_checkpoint.checkpoint(lambda x, w = w: torch.tanh(x @ w), h, use_reentrant = False)
        grpo_topk_head_logprobs(h, head, ids, chunks = 4).sum().backward()
    assert gc.LAST_GC_INDEX == 3, gc.LAST_GC_INDEX
    assert len(smart_gc.offloads) == 6  # the layers still go through Unsloth's offloading checkpoint
    assert all(w.grad is not None for w in weights) and head.grad is not None


def test_closure_only_params_keep_their_grads(smart_gc, monkeypatch):
    """EAGLE3's chunk reaches the drafter head only through its closure. With explicit inputs that need no grad
    the reentrant replacement records nothing for the chunk and the head gets no grad; torch's own checkpoint
    gives the head the grad an unpatched run does."""
    from unsloth_zoo.speculative.eagle3 import eagle3_chunked_loss

    def head_grad():
        drafter = _Drafter()
        g = _gen()
        x = torch.randn(1, 9, 8, generator = g)
        final = torch.randn(1, 9, 8, generator = g)
        lm = torch.randn(16, 8, generator = g)
        mask = torch.ones(1, 9, dtype = torch.bool)
        loss, _ = eagle3_chunked_loss(drafter, [x], final, lm, mask, chunk_rows = 4)
        loss = loss + 0.0 * drafter.fc.weight.sum()
        return torch.autograd.grad(loss, drafter.head.weight, allow_unused = True)[0]

    with monkeypatch.context() as patch:
        patch.setattr(torch_checkpoint, "checkpoint", torch_checkpoint._old_checkpoint)
        want = head_grad()
    smart_gc.pristine_calls.clear()
    got = head_grad()
    assert want is not None
    assert got is not None, "closure-only drafter head lost its grad"
    torch.testing.assert_close(got, want, rtol = 0, atol = 0)
    assert smart_gc.offloads == [] and len(smart_gc.pristine_calls) == 3


def test_torch_checkpoint_resolution(monkeypatch):
    pristine = getattr(torch_checkpoint, "_unsloth_pristine_checkpoint", None) or torch_checkpoint.checkpoint
    monkeypatch.delattr(torch_checkpoint, "_unsloth_pristine_checkpoint", raising = False)
    monkeypatch.setattr(torch_checkpoint, "checkpoint", pristine)
    assert gc._torch_checkpoint() is pristine  # never patched
    monkeypatch.setattr(torch_checkpoint, "_old_checkpoint", pristine, raising = False)
    monkeypatch.setattr(torch_checkpoint, "checkpoint", gc.unsloth_checkpoint)
    assert gc._torch_checkpoint() is pristine  # patched before the stash existed
    monkeypatch.setattr(torch_checkpoint, "_unsloth_pristine_checkpoint", pristine, raising = False)
    monkeypatch.setattr(torch_checkpoint, "_old_checkpoint", gc.unsloth_gradient_checkpoint)
    assert gc._torch_checkpoint() is pristine  # stacked shims: the set-once stash wins
