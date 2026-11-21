# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""add_new_tokens skips transformers' mean_resizing (fp32 copies + a d x d covariance per matrix) when it overwrites
every grown row anyway; the resulting embeddings and lm_head are bit-identical to the default call."""

import pytest
import torch

pytest.importorskip("transformers")

from transformers import AutoModelForCausalLM, AutoTokenizer

import unsloth_zoo.tokenizer_utils as tu

REPO = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _load():
    try:
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype = torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(REPO)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"tiny model unavailable: {exc}")
    return model, tokenizer


def _run(method, pad, force_default):
    model, tokenizer = _load()
    seen = []
    original = model.resize_token_embeddings

    def spy(*args, **kwargs):
        seen.append(kwargs.get("mean_resizing", "default"))
        if force_default:
            kwargs.pop("mean_resizing", None)
        return original(*args, **kwargs)

    model.resize_token_embeddings = spy
    # Past the physical rows, so the matrices really grow.
    n_new = max(model.get_input_embeddings().weight.shape[0] - len(tokenizer), 0) + 5
    torch.manual_seed(1)
    tu.add_new_tokens(model, tokenizer, new_tokens = [f"<mr_tok_{i}>" for i in range(n_new)], method = method,
                      pad_to_multiple_of = pad)
    return (seen, model.get_input_embeddings().weight.detach().clone(),
            model.get_output_embeddings().weight.detach().clone())


@pytest.mark.parametrize("method", ["mean", "interpolation"])
@pytest.mark.parametrize("pad", [None, 64])
def test_mean_resizing_skipped_and_weights_identical(method, pad):
    seen, emb, head = _run(method, pad, force_default = False)
    assert seen == [False]
    ref_seen, ref_emb, ref_head = _run(method, pad, force_default = True)
    assert torch.equal(emb, ref_emb) and torch.equal(head, ref_head)


def test_old_signature_keeps_the_plain_call():
    class Old:
        def resize_token_embeddings(self, new_num_tokens = None):
            pass

    assert tu._accepts_mean_resizing(Old()) is False
