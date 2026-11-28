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

"""DeepSeek-V4.1 gradient checkpointing replays the sparse-attention indexer's top-k (and candidate blocks) in the
recompute instead of scoring again; CPU only, against every implementation on the path (tests/_dsv41_impl.py)."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _dsv41_impl import available  # noqa: E402

IMPLS = available()
pytestmark = pytest.mark.skipif(
    not IMPLS, reason = "needs a DeepSeek-V4.1 implementation (transformers port, or unsloth's native module)"
)


@pytest.fixture(params = IMPLS or [None], ids = lambda impl: getattr(impl, "name", "none"))
def impl(request):
    request.param.register()
    from unsloth_zoo.temporary_patches.deepseek_v41 import patch_deepseek_v41_gradient_checkpointing

    patch_deepseek_v41_gradient_checkpointing()
    return request.param


def _tiny_model(impl):
    # Two candidate-free index sources (ratio 2), a candidate source (4) and a reader (5); n_comp > index_topk.
    config = impl.TextConfig(
        vocab_size = 256, hidden_size = 64, num_hidden_layers = 6, num_attention_heads = 2, head_dim = 64,
        qk_rope_head_dim = 16, q_lora_rank = 32, o_groups = 2, o_lora_rank = 16, moe_intermediate_size = 32,
        n_routed_experts = 4, num_experts_per_tok = 2, sliding_window = 8,
        compress_ratios = [0, 0, 2, 2, 1, 1], kv_source_layer_ids = [2, 4], index_source_layer_ids = [2, 3, 4, 5],
        candidate_source_layer_id = 4, candidate_topk_blocks = 2, candidate_block_size = 2, index_n_heads = 2,
        index_head_dim = 32, index_topk = 4, hc_mult = 2, hc_sinkhorn_iters = 3, engram_layer_ids = [],
        num_nextn_predict_layers = 0, rms_norm_eps = 1e-6,
    )
    torch.manual_seed(0)
    model = impl.modeling.DeepseekV41ForCausalLM(config).float()
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim >= 2:
                p.normal_(0.0, 0.05)
    return model


def _indexers(model):
    return [m for m in model.modules() if type(m).__name__ == "DeepseekV41Indexer"]


def _step(model, ids, gradient_checkpointing, retain_graph = False):
    """(loss, grads, indexer scorings in forward, in backward). A scoring is counted at the indexer's own
    weights_proj, which runs only when the indexer really scores (not when its selection is replayed)."""
    counts = {"forward": 0, "backward": 0}
    phase = ["forward"]
    hooks = [
        indexer.weights_proj.register_forward_hook(lambda *_: counts.__setitem__(phase[0], counts[phase[0]] + 1))
        for indexer in _indexers(model)
    ]
    try:
        model.zero_grad(set_to_none = True)
        if gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
        else:
            model.gradient_checkpointing_disable()
        model.train()
        loss = model(input_ids = ids, labels = ids, use_cache = False).loss
        phase[0] = "backward"
        loss.backward(retain_graph = retain_graph)
        if retain_graph:
            loss.backward()
    finally:
        for hook in hooks:
            hook.remove()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    return loss.detach(), grads, counts["forward"], counts["backward"]


def _selection_stats():
    import collections

    from unsloth_zoo import gradient_checkpointing as gc_module

    stats = getattr(gc_module, "GC_SELECTION_STATS", collections.Counter())
    stats.clear()
    return stats


def _ids():
    return torch.randint(0, 256, (2, 24), generator = torch.Generator().manual_seed(0))


@pytest.fixture
def one_thread():
    # Multi-threaded CPU reductions differ run to run by about 1e-7; one thread makes "bitwise" meaningful.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def test_recompute_replays_the_indexer_selection(impl, monkeypatch, one_thread):
    model = _tiny_model(impl)
    n_index = len(_indexers(model))
    assert n_index == 4
    ids = _ids()

    monkeypatch.setenv("UNSLOTH_GC_REPLAY_SELECTION", "0")
    loss_off, g_off, fwd_off, bwd_off = _step(model, ids, True)
    monkeypatch.delenv("UNSLOTH_GC_REPLAY_SELECTION")
    stats = _selection_stats()
    loss_on, g_on, fwd_on, bwd_on = _step(model, ids, True)

    # Kill switch: the recompute scores again (one per index source); replay: it never does.
    assert (fwd_off, bwd_off) == (n_index, n_index)
    assert (fwd_on, bwd_on) == (n_index, 0)
    assert stats["stored"] == n_index
    assert stats["reused"] == n_index
    assert stats["mismatch"] == 0 and stats["unmatched"] == 0
    # Same selection, same arithmetic: bitwise equal to scoring again.
    assert torch.equal(loss_on, loss_off)
    assert g_on.keys() == g_off.keys()
    for name in g_off:
        assert torch.equal(g_on[name], g_off[name]), name

    # And equal to the run without checkpointing.
    loss_ref, g_ref, _, _ = _step(model, ids, False)
    torch.testing.assert_close(loss_on, loss_ref)
    assert g_ref.keys() == g_on.keys()
    for name in g_ref:
        torch.testing.assert_close(g_on[name], g_ref[name], rtol = 1e-4, atol = 1e-6, msg = name)


def test_second_backward_replays_again(impl):
    model = _tiny_model(impl)
    n_index = len(_indexers(model))
    ids = _ids()
    stats = _selection_stats()
    loss, grads, fwd, bwd = _step(model, ids, True, retain_graph = True)
    assert (fwd, bwd) == (n_index, 0)
    assert stats["reused"] == 2 * n_index
    loss1, grads1, _, _ = _step(model, ids, True)
    for name in grads1:
        torch.testing.assert_close(grads[name], 2 * grads1[name], rtol = 1e-5, atol = 1e-7, msg = name)


def test_no_frame_outside_checkpointing(impl):
    """Without checkpointing (and in no_grad) the indexer scores once per call and nothing is stored."""
    model = _tiny_model(impl)
    n_index = len(_indexers(model))
    ids = _ids()
    stats = _selection_stats()
    _, _, fwd, bwd = _step(model, ids, False)
    assert (fwd, bwd) == (n_index, 0)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
    with torch.no_grad():
        model(input_ids = ids, use_cache = False)
    assert sum(stats.values()) == 0


def test_replay_reaches_an_indexer_class_a_compiled_cache_installed(impl, monkeypatch, one_thread):
    """unsloth_compile_transformers replaces the modeling module's DeepseekV41Indexer with its compiled copy after the
    patch ran, keeping the (already patched) decoder layer; the post_compile run of the patch wraps the new class."""
    from unsloth_zoo.temporary_patches.deepseek_v41 import patch_deepseek_v41_gradient_checkpointing

    original = impl.modeling.DeepseekV41Indexer
    unwrapped = getattr(original.forward, "_unsloth_original", original.forward)
    # A new class with the stock (unwrapped) forward, like the compiled cache's copy of it.
    compiled = type("DeepseekV41Indexer", (original,), {"forward": unwrapped})
    monkeypatch.setattr(impl.modeling, "DeepseekV41Indexer", compiled)
    patch_deepseek_v41_gradient_checkpointing()  # the post_compile phase
    assert getattr(compiled.forward, "_unsloth_selection_replay", False)

    model = _tiny_model(impl)
    assert {type(m) for m in _indexers(model)} == {compiled}
    _, _, fwd_on, bwd_on = _step(model, _ids(), True)
    assert (fwd_on, bwd_on) == (len(_indexers(model)), 0)
