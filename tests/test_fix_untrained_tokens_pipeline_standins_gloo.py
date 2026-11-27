# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Per-stage fix_untrained_tokens matches a one-process run without copying lm_head to host."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest  # noqa: E402
import torch  # noqa: E402

VOCAB, HIDDEN = 4096, 64


class _Tokenizer:
    chat_template = None

    def __len__(self):
        return VOCAB

    def convert_ids_to_tokens(self, ids):
        return [f"<|extra_{i}|>" for i in ids]


class _Model(torch.nn.Module):
    def __init__(self, embed, head):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(VOCAB, HIDDEN)
        self.lm_head = torch.nn.Linear(HIDDEN, VOCAB, bias = False)
        self.embed_tokens.weight = torch.nn.Parameter(embed)
        self.lm_head.weight = torch.nn.Parameter(head)

        class _Config:
            _name_or_path = "unsloth/tiny-pipeline-test"

        self.config = _Config()

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


def _weights():
    g = torch.Generator().manual_seed(0)
    embed = torch.randn(VOCAB, HIDDEN, generator = g)
    head = torch.randn(VOCAB, HIDDEN, generator = g)
    embed[[5, 6, 7]] = 0.0  # untrained in the embedding
    head[6] = head[5]  # 5, 6 repeat in the lm_head: untrained; 7 does not
    head[9] = head[5]  # repeats too, but its embedding row is trained: never reset
    return embed, head


def _standin(t):
    return torch.zeros((), dtype = t.dtype).expand(t.shape)


DATA = [{"input_ids": [1, 5, 2]}, {"input_ids": [6, 9, 3]}]
MARK = "_unsloth_pp_preload"  # set by the per-stage pipeline loader on every rank's model
# rank -> (holds the embedding, holds the lm_head): split across ranks, or colocated on rank 0 (tied / pinned).
LAYOUTS = {"split": ((True, False), (False, True)), "colocated": ((True, True), (False, False))}


def _rank_model(layout, rank, marked = True):
    embed, head = _weights()
    has_embed, has_head = LAYOUTS[layout][rank]
    model = _Model(embed if has_embed else _standin(embed), head if has_head else _standin(head))
    if marked:
        model.__dict__[MARK] = {"rank": rank}
    return model


def _worker(rank, init_file, out):
    import datetime
    import torch.distributed as dist
    import tracemalloc
    import importlib.util

    if importlib.util.find_spec("unsloth") is None:
        os.environ.setdefault("UNSLOTH_ZOO_DISABLE_GPU_INIT", "1")
    from unsloth_zoo import tokenizer_utils

    # A rank entering a collective the other skips fails in seconds instead of hanging the suite.
    dist.init_process_group(
        "gloo", init_method = "file://" + init_file, rank = rank, world_size = 2,
        timeout = datetime.timedelta(seconds = 30),
    )
    calls = []
    all_reduce = dist.all_reduce

    def counting_all_reduce(*args, **kwargs):
        calls.append(1)
        return all_reduce(*args, **kwargs)

    dist.all_reduce = counting_all_reduce
    try:
        results = {}
        # Warm lazy imports before tracing host memory.
        tokenizer_utils.fix_untrained_tokens(_rank_model("split", rank), _Tokenizer(), DATA)
        for layout in LAYOUTS:
            model = _rank_model(layout, rank)
            del calls[:]
            tracemalloc.start()
            try:
                tokenizer_utils.fix_untrained_tokens(model, _Tokenizer(), DATA)
                res = {"error": None}
            except Exception as e:  # asserted in the parent
                res = {"error": repr(e)}
            res["host_peak"] = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            res["collectives"] = len(calls)
            res["free"] = [not any(w.stride()) for w in (model.embed_tokens.weight, model.lm_head.weight)]
            res["embed"] = model.embed_tokens.weight.detach().clone()
            res["head"] = model.lm_head.weight.detach().clone()
            results[layout] = res
        # No per-stage mark (DDP / FSDP2 replicas, one process): never a collective, even in a process group.
        if rank == 0:
            del calls[:]
            embed, head = _weights()
            model = _Model(embed, head)
            tokenizer_utils.fix_untrained_tokens(model, _Tokenizer(), DATA)
            results["unmarked"] = {"collectives": len(calls), "embed": model.embed_tokens.weight.detach().clone()}
            del calls[:]
            model = _rank_model("split", 1, marked = False)
            tokenizer_utils.fix_untrained_tokens(model, _Tokenizer(), DATA)
            results["unmarked_standin"] = {"collectives": len(calls), "head": model.lm_head.weight.detach().clone()}
        dist.barrier()
        torch.save(results, os.path.join(out, f"rank{rank}.pt"))
    finally:
        dist.all_reduce = all_reduce
        dist.destroy_process_group()


@pytest.fixture(scope = "module")
def ranks(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("pp_untrained"))
    torch.multiprocessing.spawn(_worker, args = (os.path.join(out, "init"), out), nprocs = 2, join = True)
    return [torch.load(os.path.join(out, f"rank{r}.pt")) for r in range(2)]


def _one_process():
    import importlib.util

    if importlib.util.find_spec("unsloth") is None:
        os.environ.setdefault("UNSLOTH_ZOO_DISABLE_GPU_INIT", "1")
    from unsloth_zoo import tokenizer_utils

    embed, head = _weights()
    full = _Model(embed.clone(), head.clone())
    tokenizer_utils.fix_untrained_tokens(full, _Tokenizer(), DATA)
    full_embed, full_head = full.embed_tokens.weight.detach(), full.lm_head.weight.detach()
    assert not torch.equal(full_embed[5], embed[5]) and torch.equal(full_head[9], head[9])
    return full_embed, full_head


@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_pipeline_ranks_reset_the_rows_one_process_resets(ranks, layout):
    full_embed, full_head = _one_process()
    for rank, res in enumerate(ranks):
        res = res[layout]
        assert res["error"] is None, (rank, res["error"])
        assert res["collectives"] == 2, (rank, res["collectives"])  # every rank, whatever it holds
        held = LAYOUTS[layout][rank]
        assert res["free"] == [not held[0], not held[1]], (rank, res["free"])  # stand-ins stay stand-ins
        if held[0]:
            torch.testing.assert_close(res["embed"], full_embed, rtol = 0, atol = 0)
        if held[1]:
            torch.testing.assert_close(res["head"], full_head, rtol = 0, atol = 0)


def test_unmarked_runs_never_enter_a_collective(ranks):
    full_embed, full_head = _one_process()
    res = ranks[0]["unmarked"]
    assert res["collectives"] == 0
    torch.testing.assert_close(res["embed"], full_embed, rtol = 0, atol = 0)
    # Holding only the lm_head with no pipeline group to judge the rest: skipped, untouched.
    _, head = _weights()
    res = ranks[0]["unmarked_standin"]
    assert res["collectives"] == 0 and torch.equal(res["head"], head)


def test_pipeline_rank_with_the_lm_head_copies_only_candidate_rows(ranks):
    # One float32 copy of the whole lm_head is VOCAB * HIDDEN * 4 = 1 MiB here (4.7 GiB on Qwen3.8-27B).
    for rank, layout in ((1, "split"), (0, "colocated")):
        peak = ranks[rank][layout]["host_peak"]
        assert peak < VOCAB * HIDDEN * 4 // 4, (rank, layout, peak)
