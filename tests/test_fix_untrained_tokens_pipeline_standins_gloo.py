# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""fix_untrained_tokens on a per-stage pipeline load: pipeline rank 0 holds the embedding and a storage-free
stand-in for the lm_head, the last rank the reverse. Each rank must reset exactly the rows a one-process run resets
in the matrix it holds, and judge the lm_head only on the rows the embedding marks: treating the stand-in embedding
as all-untrained copied the whole lm_head to the host (11 GiB on Qwen3.8-27B) and could flag other rows."""

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


def _worker(rank, init_file, out):
    import torch.distributed as dist
    import tracemalloc
    import importlib.util

    if importlib.util.find_spec("unsloth") is None:
        os.environ.setdefault("UNSLOTH_ZOO_DISABLE_GPU_INIT", "1")
    from unsloth_zoo import tokenizer_utils

    dist.init_process_group("gloo", init_method = "file://" + init_file, rank = rank, world_size = 2)
    try:
        embed, head = _weights()
        model = _Model(embed, _standin(head)) if rank == 0 else _Model(_standin(embed), head)
        tracemalloc.start()
        try:
            tokenizer_utils.fix_untrained_tokens(model, _Tokenizer(), DATA)
            res = {"error": None}
        except Exception as e:  # asserted in the parent
            res = {"error": repr(e)}
        res["host_peak"] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        res["free"] = [not any(w.stride()) for w in (model.embed_tokens.weight, model.lm_head.weight)]
        res["embed"] = model.embed_tokens.weight.detach().clone()
        res["head"] = model.lm_head.weight.detach().clone()
        torch.save(res, os.path.join(out, f"rank{rank}.pt"))
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope = "module")
def ranks(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("pp_untrained"))
    torch.multiprocessing.spawn(_worker, args = (os.path.join(out, "init"), out), nprocs = 2, join = True)
    return [torch.load(os.path.join(out, f"rank{r}.pt")) for r in range(2)]


def test_pipeline_ranks_reset_the_rows_one_process_resets(ranks):
    import importlib.util

    if importlib.util.find_spec("unsloth") is None:
        os.environ.setdefault("UNSLOTH_ZOO_DISABLE_GPU_INIT", "1")
    from unsloth_zoo import tokenizer_utils

    embed, head = _weights()
    full = _Model(embed.clone(), head.clone())
    tokenizer_utils.fix_untrained_tokens(full, _Tokenizer(), DATA)
    full_embed, full_head = full.embed_tokens.weight.detach(), full.lm_head.weight.detach()
    assert not torch.equal(full_embed[5], embed[5]) and torch.equal(full_head[9], head[9])
    for rank, res in enumerate(ranks):
        assert res["error"] is None, (rank, res["error"])
    torch.testing.assert_close(ranks[0]["embed"], full_embed, rtol = 0, atol = 0)
    torch.testing.assert_close(ranks[1]["head"], full_head, rtol = 0, atol = 0)
    # The stand-ins stay stand-ins.
    assert ranks[0]["free"] == [False, True] and ranks[1]["free"] == [True, False]


def test_pipeline_rank_with_the_lm_head_copies_only_candidate_rows(ranks):
    # One float32 copy of the whole lm_head is VOCAB * HIDDEN * 4 = 1 MiB here (4.7 GiB on Qwen3.8-27B).
    assert ranks[1]["host_peak"] < VOCAB * HIDDEN * 4 // 4, ranks[1]["host_peak"]
