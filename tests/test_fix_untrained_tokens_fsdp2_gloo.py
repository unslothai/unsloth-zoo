# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""fix_untrained_tokens on FSDP2 (DTensor) embeddings.

A second Trainer built on a model that native FSDP2 already sharded (the continued-pretraining
notebook: UnslothTrainer on Wikipedia, then UnslothTrainer on Alpaca, with trainable embed_tokens /
lm_head) reaches fix_untrained_tokens with DTensor weights. Slicing a DTensor under inference mode
raised "Cannot set version_counter for inference tensor". On a gloo CPU world of 2 the untrained rows
must be reset on every rank, the trained rows left alone, and the parameters must stay the same
DTensor objects FSDP2 manages."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest  # noqa: E402
import torch  # noqa: E402


class _Tokenizer:
    chat_template = "<|extra_0|> in the template"

    def __len__(self):
        return 8

    def convert_ids_to_tokens(self, ids):
        return [f"<|extra_{i}|>" for i in range(len(ids))]


class _Model(torch.nn.Module):
    def __init__(self, tied):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(8, 4)
        self.lm_head = torch.nn.Linear(4, 8, bias = False)
        if tied:
            self.lm_head.weight = self.embed_tokens.weight
        with torch.no_grad():
            for w in (self.embed_tokens.weight, self.lm_head.weight):
                w.fill_(1.0)
                w[4].zero_()
                w[5].zero_()

        class _Config:
            _name_or_path = "unsloth/tiny-test-model"

        self.config = _Config()

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


def _worker(rank, init_file, out):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from unsloth_zoo import tokenizer_utils

    dist.init_process_group("gloo", init_method = "file://" + init_file, rank = rank, world_size = 2)
    try:
        mesh = init_device_mesh("cpu", (2,))
        res = {}
        for tied in (False, True):
            model = _Model(tied)
            fully_shard(model.embed_tokens, mesh = mesh)
            if not tied:
                fully_shard(model.lm_head, mesh = mesh)
            fully_shard(model, mesh = mesh)
            before = [model.embed_tokens.weight, model.lm_head.weight]
            try:
                tokenizer_utils.fix_untrained_tokens(
                    model, _Tokenizer(), [{"input_ids" : [1, 2, 4]}, {"input_ids" : [2, 3, 5]}],
                )
            except Exception as e:  # asserted in the parent
                res[tied] = repr(e)
                continue
            same = all(a is b for a, b in zip(before, [model.embed_tokens.weight, model.lm_head.weight]))
            res[tied] = (
                same,
                type(model.embed_tokens.weight).__name__,
                model.embed_tokens.weight.full_tensor().detach().clone(),
                model.lm_head.weight.full_tensor().detach().clone(),
            )
        torch.save(res, os.path.join(out, f"rank{rank}.pt"))
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope = "module")
def ranks(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("fsdp2_untrained"))
    torch.multiprocessing.spawn(
        _worker, args = (os.path.join(out, "init"), out), nprocs = 2, join = True
    )
    return [torch.load(os.path.join(out, f"rank{r}.pt")) for r in range(2)]


@pytest.mark.parametrize("tied", [False, True], ids = ["untied", "tied"])
def test_fix_untrained_tokens_resets_rows_of_sharded_embeddings(ranks, tied):
    results = [res[tied] for res in ranks]
    for res in results:
        assert not isinstance(res, str), res
        same, kind, embed, head = res
        assert same, "the FSDP2 parameters must stay the objects FSDP2 manages"
        assert kind == "DTensor"
        for w in (embed, head):
            assert torch.all(w[:4] == 1.0) and torch.all(w[6:] == 1.0), w
            assert not torch.equal(w[4], torch.zeros(4)), w
            assert not torch.equal(w[5], torch.zeros(4)), w
    torch.testing.assert_close(results[0][2], results[1][2])
    torch.testing.assert_close(results[0][3], results[1][3])
