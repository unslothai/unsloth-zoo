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

"""The GRPO policy pass must keep packing and PrefixGrouper within one padded mini-batch's tokens (a batch over it
packs in several bins of whole rows), and keep both off when forwards are collective (sharded FSDP2), so every rank
runs the same forwards."""

import collections
import importlib.util
import os
import sys
import types

import pytest
import torch

import unsloth_zoo.rl_replacements as rr

V, H = 16, 8
_IDS = torch.tensor([
    [0, 0, 1, 2, 3, 4, 5, 6],
    [0, 0, 1, 2, 7, 8, 9, 0],
    [1, 2, 6, 7, 3, 4, 6, 9],
    [1, 2, 6, 7, 5, 0, 0, 0],
    [0, 3, 2, 3, 5, 9, 8, 7],
    [0, 3, 2, 3, 11, 12, 0, 0],
])
_COMPLETION = 4
ROWS, L = _IDS.shape


class _Model(torch.nn.Module):
    """Position-local, so packed, grouped and per-row forwards agree; records every forward."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(V, H)
        self.mix = torch.nn.Linear(H, H)
        self.head = torch.nn.Linear(H, V, bias = False)
        self.device = torch.device("cpu")
        self.calls = []

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids = None, prefix_seg_info = None, packed_seq_lengths = None, **kwargs):
        kind = "grouped" if prefix_seg_info is not None else ("packed" if packed_seq_lengths is not None else "padded")
        self.calls.append((kind, input_ids.numel()))
        return types.SimpleNamespace(logits = torch.tanh(self.mix(self.emb(input_ids))))


def _trainer(model, mini_batch):
    return types.SimpleNamespace(
        args = types.SimpleNamespace(
            unsloth_grpo_mini_batch = mini_batch, unsloth_logit_chunk_multiplier = 1, off_policy_mask_threshold = None,
        ),
        processing_class = types.SimpleNamespace(pad_token_id = 0),
        model = model,
        accelerator = types.SimpleNamespace(
            unwrap_model = lambda m, keep_fp32_wrapper = False: m, scaler = None, gather = lambda t: t,
        ),
        control = types.SimpleNamespace(should_evaluate = False),
        _metrics = {"train": collections.defaultdict(list), "eval": collections.defaultdict(list)},
        use_vllm = False,
        _autocast_dtype = None,
        beta = 0.0,
        num_generations = 2,
    )


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    import unsloth_zoo.fused_losses.fused_grpo as _fused_grpo
    monkeypatch.setattr(_fused_grpo, "GRPO_FWD_GRAD", False)
    monkeypatch.setenv("UNSLOTH_RETURN_HIDDEN_STATES", "0")
    monkeypatch.setenv("UNSLOTH_GRPO_SKIP_ZERO_ADVANTAGE", "0")
    monkeypatch.setenv("UNSLOTH_GRPO_SEQ_PACKING", "1")
    # grpo_accumulated_loss memoises its PrefixGrouper helpers on itself: start unset, restore after.
    monkeypatch.setattr(rr.grpo_accumulated_loss, "_pg_funcs", None, raising = False)


def _load_prefix_grouper():
    """unsloth's PrefixGrouper module from its own file: importing the unsloth package can hit an import cycle on
    CPU-only hosts, and only the layout builder is needed."""
    spec = importlib.util.find_spec("unsloth")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("needs unsloth's PrefixGrouper")
    root = os.path.join(list(spec.submodule_search_locations)[0], "utils")
    if not os.path.exists(os.path.join(root, "prefix_grouper.py")):
        pytest.skip("this unsloth has no PrefixGrouper")
    pkg = types.ModuleType("_pg_cap_stub")
    pkg.__path__ = [root]
    sys.modules.setdefault("_pg_cap_stub", pkg)
    for name in ("prefix_grouper_kernel", "prefix_grouper"):
        full = f"_pg_cap_stub.{name}"
        if full not in sys.modules:
            module_spec = importlib.util.spec_from_file_location(full, os.path.join(root, f"{name}.py"))
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[full] = module
            module_spec.loader.exec_module(module)
    return sys.modules["_pg_cap_stub.prefix_grouper"]


def _prefix_grouper(monkeypatch):
    """PrefixGrouper with the real layout builder, no token-ratio gate and no verify (trusted at once)."""
    pg = _load_prefix_grouper()
    build = lambda *a, **k: pg.build_group_layout(*a, apply_tokr_gate = False, **k)
    funcs = (build, lambda model = None: True, lambda: False, pg.tol_ok, pg.TOL_KILL)
    monkeypatch.setenv("UNSLOTH_GRPO_PREFIX_GROUPER", "1")
    monkeypatch.setattr(rr.grpo_accumulated_loss, "_pg_funcs", funcs, raising = False)


def _run(mini_batch, model = None, result = None):
    model = model or _Model()
    model.train(True)
    mask = (_IDS[:, -_COMPLETION:] != 0).long()
    advantages = torch.linspace(-1.0, 1.0, ROWS)
    out = rr.grpo_accumulated_loss(
        _trainer(model, mini_batch), _IDS, (_IDS != 0).long(), _COMPLETION, mask, advantages, None, None,
        loss_type = "dapo", max_completion_length = 8, num_items_in_batch = mask.sum().float(),
        current_gradient_accumulation_steps = 1, epsilon_low = 0.2, epsilon_high = 0.28,
    )
    out[0].backward()
    if result is not None:
        result["loss"] = out[0].detach()
        result["grads"] = {k: p.grad.detach().clone() for k, p in model.named_parameters() if p.grad is not None}
    return model.calls


_REAL = [int(n) for n in (_IDS != 0).sum(dim = 1)]


def test_packing_bins_are_whole_rows_within_the_cap():
    assert rr.grpo_packing_bins(_REAL, 3 * L) == [(0, 4), (4, 6)]
    assert rr.grpo_packing_bins(_REAL, sum(_REAL)) == [(0, ROWS)]
    assert rr.grpo_packing_bins(_REAL, L) == [(i, i + 1) for i in range(ROWS)]
    assert rr.grpo_packing_bins(_REAL, max(_REAL) - 1) is None
    assert rr.grpo_packing_bins([], L) == []


def test_packing_stays_within_the_mini_batch(monkeypatch):
    monkeypatch.setenv("UNSLOTH_GRPO_PREFIX_GROUPER", "0")
    # 6 splits: one row per forward. Single-row bins cannot test cross-row isolation, so unverified packing does
    # not run them: the padded loop does.
    calls = _run(mini_batch = ROWS)
    assert all(kind == "padded" for kind, _ in calls), calls
    assert max(n for _, n in calls) <= L, calls


def test_single_row_bins_never_widen_the_trusted_envelope(monkeypatch):
    # Already trusted packing (verified before): single-row bins run packed, but they verify nothing about
    # cross-row isolation, so a later multi-row bin of the same length is verified again, not trusted.
    monkeypatch.setenv("UNSLOTH_GRPO_PREFIX_GROUPER", "0")
    model = _Model()
    model._unsloth_seq_packing_grad_ok = True
    calls = _run(mini_batch = ROWS, model = model)
    assert [n for kind, n in calls if kind == "packed"] == _REAL, calls
    assert int(getattr(model, "_unsloth_seq_packing_grad_verified_T", 0)) == 0
    model.calls = []
    calls = _run(mini_batch = 2, model = model)
    packed = [n for kind, n in calls if kind == "packed"]
    assert packed[:2] == [sum(_REAL[:4]), sum(_REAL[4:])], calls
    assert any(kind == "padded" for kind, _ in calls), calls  # the per-row verify forwards ran


def test_a_failing_bin_releases_the_finished_bins(monkeypatch):
    # An OOM in the second packed bin: the first bin's graph must not stay alive through the padded fallback.
    monkeypatch.setenv("UNSLOTH_GRPO_PREFIX_GROUPER", "0")
    import gc
    import weakref

    class _Sentinel:
        pass

    class _Keep(torch.autograd.Function):
        # Identity whose backward context holds a sentinel: it lives exactly as long as the bin's graph.
        @staticmethod
        def forward(ctx, x, sentinel):
            ctx.sentinel = sentinel
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad):
            return grad, None

    model = _Model()
    model._unsloth_seq_packing_grad_ok = True
    seen, forward = [], model.forward

    def flaky(input_ids = None, packed_seq_lengths = None, **kwargs):
        out = forward(input_ids = input_ids, packed_seq_lengths = packed_seq_lengths, **kwargs)
        if packed_seq_lengths is not None:
            sentinel = _Sentinel()
            seen.append(weakref.ref(sentinel))
            out.logits = _Keep.apply(out.logits, sentinel)
            del sentinel
            if len(seen) == 2:
                raise torch.cuda.OutOfMemoryError("injected")
        elif seen:
            gc.collect()
            assert all(ref() is None for ref in seen), "a finished packed bin is still alive in the padded loop"
        return out

    model.forward = flaky
    calls = _run(mini_batch = 2, model = model)
    assert [kind for kind, _ in calls].count("packed") == 2, calls
    assert any(kind == "padded" for kind, _ in calls), calls


def test_batch_over_the_mini_batch_packs_in_bins(monkeypatch):
    # 2 splits of 3 rows: 36 real tokens over a 24-token mini-batch. Two packed forwards of whole rows
    # replace the padded loop, with the loss and gradients of one packed forward over the whole batch.
    monkeypatch.setenv("UNSLOTH_GRPO_PREFIX_GROUPER", "0")
    assert sum(_REAL) > 3 * L
    binned, whole = {}, {}
    calls = _run(mini_batch = 2, result = binned)
    packed = [n for kind, n in calls if kind == "packed"]
    assert packed == [sum(_REAL[:4]), sum(_REAL[4:])], calls
    assert max(n for _, n in calls) <= 3 * L, calls
    assert [n for _, n in _run(mini_batch = 1, result = whole) if _ == "packed"] == [sum(_REAL)]
    torch.testing.assert_close(binned["loss"], whole["loss"], rtol = 1e-6, atol = 1e-7)
    assert binned["grads"].keys() == whole["grads"].keys() and binned["grads"]
    for name, grad in binned["grads"].items():
        torch.testing.assert_close(grad, whole["grads"][name], rtol = 1e-5, atol = 1e-7)


def test_packing_still_runs_when_it_fits(monkeypatch):
    monkeypatch.setenv("UNSLOTH_GRPO_PREFIX_GROUPER", "0")
    assert "packed" in [kind for kind, _ in _run(mini_batch = 1)]


def test_prefix_grouper_stays_within_the_mini_batch(monkeypatch):
    # The grouped layout is over the cap, so it is declined; packing then runs one row per forward.
    _prefix_grouper(monkeypatch)
    calls = _run(mini_batch = ROWS)
    assert "grouped" not in [kind for kind, _ in calls], calls
    assert max(n for _, n in calls) <= L, calls


def test_prefix_grouper_still_runs_when_it_fits(monkeypatch):
    _prefix_grouper(monkeypatch)
    assert [kind for kind, _ in _run(mini_batch = 1)] == ["grouped"]


def _collective(monkeypatch, fsdp2 = True):
    monkeypatch.setattr(rr, "grpo_agree_mini_batch", lambda B, device: B)
    if fsdp2:
        monkeypatch.setattr(rr, "grpo_forwards_are_collective", lambda model: True)
        return
    # DeepSpeed ZeRO-3 on 2 ranks: a live ZeRO-3 config and an initialised 2-rank group.
    import torch.distributed as dist
    import unsloth_zoo.temporary_patches.utils as tp_utils
    monkeypatch.setattr(tp_utils, "zero3_partitioned_training", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda *a, **k: 2)


@pytest.mark.parametrize("fsdp2", [True, False], ids = ["fsdp2", "zero3"])
@pytest.mark.parametrize("prefix_grouper", [True, False], ids = ["pg", "packing"])
def test_rank_local_forwards_are_off_when_forwards_are_collective(monkeypatch, fsdp2, prefix_grouper):
    """Sharded FSDP2 / ZeRO-3 forwards all-gather: a rank whose batch packs or groups would run a different number
    of forwards than one that runs the chunked loop, and the collectives hang. The forward count must equal the
    split count, whatever the batch looks like."""
    if prefix_grouper:
        _prefix_grouper(monkeypatch)
    else:
        monkeypatch.setenv("UNSLOTH_GRPO_PREFIX_GROUPER", "0")
    _collective(monkeypatch, fsdp2)
    for mini_batch in (1, 2):
        calls = _run(mini_batch)
        assert [kind for kind, _ in calls] == ["padded"] * mini_batch, calls


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
