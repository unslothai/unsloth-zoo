# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Checkpoints that cannot hand back a single input embedding matrix.

`hasattr(model, "get_input_embeddings")` does not answer the question.
transformers 5 defines the method on every PreTrainedModel with a base
implementation that raises NotImplementedError, measured True on both 4.57.6
and 5.17.0, so a composite checkpoint carrying more than one embedding
(Qwen3-Omni has a thinker and a talker) passes the hasattr check and then
raises. Remote code can also declare a signature that cannot be called with no
arguments: stepfun-ai/Step-3.7-Flash defines get_input_embeddings(self,
input_ids), which raises TypeError.

Both used to fail the run before training started. There is nothing to reset in
either case, so the pass is skipped.

The models here are stand-ins: the real checkpoints are 30B and larger. What is
reproduced is the offending shape, not the architecture.
"""

import datasets
import pytest
import torch
from torch import nn
from transformers import AutoModelForCausalLM, LlamaConfig, PreTrainedModel

from unsloth_zoo.tokenizer_utils import fix_untrained_tokens


def _config():
    return LlamaConfig(
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        vocab_size = 64,
        max_position_embeddings = 32,
    )


@pytest.fixture(scope = "module")
def tokenizer():
    """Built locally rather than pulled from the hub.

    None of these tests needs a particular vocabulary, and a hub tokenizer
    makes the whole file raise OSError under HF_HUB_OFFLINE=1 instead of
    testing anything.
    """
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<unk>": 0, "<pad>": 1, "hello": 2, "world": 3, "a": 4, "b": 5, "c": 6}
    backend = Tokenizer(models.WordLevel(vocab, unk_token = "<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object = backend,
        unk_token = "<unk>",
        pad_token = "<pad>",
    )


@pytest.fixture(scope = "module")
def dataset():
    return datasets.Dataset.from_dict({"text": ["hello world", "a b c"]})


def test_the_base_implementation_really_raises():
    """Negative control for the premise the whole guard rests on."""
    import inspect

    source = inspect.getsource(PreTrainedModel.get_input_embeddings)
    assert "NotImplementedError" in source


class _CompositeModel(PreTrainedModel):
    """Qwen3-Omni's shape: several towers, no single embedding, so the
    transformers base implementation raises NotImplementedError."""

    config_class = LlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.thinker = nn.Linear(4, 4)
        self.talker = nn.Linear(4, 4)


class _NonStandardSignature(PreTrainedModel):
    """Step-3.7-Flash's shape: get_input_embeddings(self, input_ids)."""

    config_class = LlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.emb = nn.Embedding(64, 16)

    def get_input_embeddings(self, input_ids):
        return self.emb(input_ids)


class _ReturnsNone(PreTrainedModel):
    """A model entitled to answer "I have none"."""

    config_class = LlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.emb = nn.Embedding(64, 16)

    def get_input_embeddings(self):
        return None

    def get_output_embeddings(self):
        return None


class _OnlyOutputIsNone(PreTrainedModel):
    """One embedding to hand back, no output embedding: transformers 5 returns
    None from get_output_embeddings for anything without an lm_head."""

    config_class = LlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.emb = nn.Embedding(64, 16)

    def get_input_embeddings(self):
        return self.emb

    def get_output_embeddings(self):
        return None


class _RaisesTypeErrorInsideTheBody(PreTrainedModel):
    """A callable accessor that fails internally, e.g. remote code against the
    wrong transformers. Not a signature we cannot call, so it must propagate."""

    config_class = LlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.emb = nn.Embedding(64, 16)

    def get_input_embeddings(self):
        raise TypeError("something inside the model went wrong")


@pytest.mark.parametrize(
    "model_class",
    [_CompositeModel, _NonStandardSignature, _ReturnsNone, _OnlyOutputIsNone],
    ids = ["NotImplementedError", "TypeError", "returns-None", "output-None"],
)
def test_the_pass_is_skipped_instead_of_failing_the_run(model_class, tokenizer, dataset):
    model = model_class(_config())
    # Must return, not raise: this runs before training starts.
    assert fix_untrained_tokens(model, tokenizer, dataset) is None


def test_the_shapes_really_are_unanswerable(tokenizer, dataset):
    """Without this, the tests above could pass for the wrong reason."""
    with pytest.raises(NotImplementedError):
        _CompositeModel(_config()).get_input_embeddings()
    with pytest.raises(TypeError):
        _NonStandardSignature(_config()).get_input_embeddings()
    assert _ReturnsNone(_config()).get_input_embeddings() is None
    model = _OnlyOutputIsNone(_config())
    assert model.get_input_embeddings() is not None
    assert model.get_output_embeddings() is None


class _BrokenInputAndUncallableOutput(PreTrainedModel):
    """The compound case: the input accessor fails in its body while the output
    accessor is the one that cannot be called. Checking either signature against
    either failure would let the genuine input failure disappear."""

    config_class = LlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.emb = nn.Embedding(64, 16)

    def get_input_embeddings(self):
        raise TypeError("something inside the model went wrong")

    def get_output_embeddings(self, input_ids):
        return self.emb(input_ids)


def test_an_uncallable_output_accessor_does_not_mask_a_broken_input_accessor(tokenizer, dataset):
    model = _BrokenInputAndUncallableOutput(_config())
    with pytest.raises(TypeError, match = "something inside the model went wrong"):
        fix_untrained_tokens(model, tokenizer, dataset)


def test_a_type_error_from_inside_the_accessor_still_propagates(tokenizer, dataset):
    """The skip is for signatures we cannot call, not for models that are broken.

    Catching every TypeError would turn a genuine failure into a training run
    quietly missing its NaN guard, which is the one outcome worse than the crash
    this whole guard exists to avoid.
    """
    model = _RaisesTypeErrorInsideTheBody(_config())
    with pytest.raises(TypeError, match = "something inside the model went wrong"):
        fix_untrained_tokens(model, tokenizer, dataset)


def test_the_skip_is_visible_without_opting_into_logging(caplog, tokenizer, dataset):
    """A silent skip would look exactly like a pass that ran.

    The package logger sits at WARNING unless UNSLOTH_ENABLE_LOGGING is set, so
    an info-level line here would never reach the user who needs it: the run
    they just started is going without the NaN guard this pass provides.
    """
    import logging

    from unsloth_zoo.log import logger as package_logger

    assert package_logger.getEffectiveLevel() <= logging.WARNING
    with caplog.at_level(logging.WARNING, logger = package_logger.name):
        fix_untrained_tokens(_CompositeModel(_config()), tokenizer, dataset)
    skips = [r for r in caplog.records if "Skipping the untrained token fix" in r.message]
    assert len(skips) == 1
    assert skips[0].levelno >= logging.WARNING


def test_an_ordinary_model_is_still_processed(tokenizer, dataset):
    """The guard must not swallow the models the pass exists for."""
    model = AutoModelForCausalLM.from_config(_config())
    before = model.get_input_embeddings().weight.detach().clone()
    fix_untrained_tokens(model, tokenizer, dataset)
    after = model.get_input_embeddings().weight.detach()
    # It ran (no exception) and left a well-trained tiny model alone.
    assert torch.equal(before, after)


def test_untrained_tokens_really_are_still_reset(tokenizer):
    """The negative control the test above cannot be.

    "Nothing changed" also holds when the pass is skipped for every model, so
    it would survive a guard that swallowed everything. This one has genuinely
    untrained rows and asserts they were reset, so it fails the moment the
    ordinary path stops running.
    """
    config = _config()
    config.tie_word_embeddings = False
    model = AutoModelForCausalLM.from_config(config)
    untrained = slice(5, 8)
    with torch.no_grad():
        model.get_input_embeddings ().weight[untrained] = 0.0
        model.get_output_embeddings().weight[untrained] = 0.0
    # An untrained id has to appear in the data for the reset to be triggered.
    dataset = datasets.Dataset.from_dict({"input_ids": [[2, 3, 4, 5]]})
    fix_untrained_tokens(model, tokenizer, dataset)
    assert model.get_input_embeddings ().weight[untrained].abs().sum() > 0
    assert model.get_output_embeddings().weight[untrained].abs().sum() > 0
