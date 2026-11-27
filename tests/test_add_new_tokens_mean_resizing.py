# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Skipping mean_resizing in add_new_tokens is bit-identical when new rows are overwritten."""

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


def _run(method, pad, force_default, head_bias = False):
    model, tokenizer = _load()
    if head_bias:
        # An untied, biased output head (Phi): mean_resizing initialises the new bias entries, which nothing overwrites.
        old_head = model.get_output_embeddings()
        head = torch.nn.Linear(old_head.in_features, old_head.out_features, bias = True)
        with torch.no_grad():
            head.weight.copy_(old_head.weight)
            head.bias.fill_(2.0)
        model.set_output_embeddings(head)
        model.config.tie_word_embeddings = False
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
    out = model.get_output_embeddings()
    return (seen, model.get_input_embeddings().weight.detach().clone(), out.weight.detach().clone(),
            None if out.bias is None else out.bias.detach().clone())


@pytest.mark.parametrize("method", ["mean", "interpolation"])
@pytest.mark.parametrize("pad", [None, 64])
def test_mean_resizing_skipped_and_weights_identical(method, pad):
    seen, emb, head, _ = _run(method, pad, force_default = False)
    assert seen == [False]
    ref_seen, ref_emb, ref_head, _ = _run(method, pad, force_default = True)
    assert torch.equal(emb, ref_emb) and torch.equal(head, ref_head)


def test_biased_lm_head_keeps_mean_resizing():
    seen, _, _, bias = _run("mean", None, force_default = False, head_bias = True)
    assert seen == ["default"]
    # New bias entries follow the existing distribution (all 2.0), not a fresh zero init.
    assert torch.allclose(bias, torch.full_like(bias, 2.0))


def test_old_signature_keeps_the_plain_call():
    class Old:
        def resize_token_embeddings(self, new_num_tokens = None):
            pass

    assert tu._accepts_mean_resizing(Old()) is False
