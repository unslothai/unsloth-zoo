# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Checkpoints that cannot return one input embedding matrix; stand-in models reproduce the offending shape since the
real Qwen3-Omni and stepfun-ai/Step-3.7-Flash are 30B and larger."""

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
    """Built locally because a hub tokenizer would raise OSError under HF_HUB_OFFLINE=1 and make the whole file fail instead of testing anything."""
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
    """Qwen3-Omni's shape: several towers, so the base implementation raises."""

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
    config_class = LlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.emb = nn.Embedding(64, 16)

    def get_input_embeddings(self):
        return None

    def get_output_embeddings(self):
        return None


class _OnlyOutputIsNone(PreTrainedModel):
    """transformers 5 returns None from get_output_embeddings without an lm_head."""

    config_class = LlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.emb = nn.Embedding(64, 16)

    def get_input_embeddings(self):
        return self.emb

    def get_output_embeddings(self):
        return None


class _RaisesTypeErrorInsideTheBody(PreTrainedModel):
    """Accessor that raises TypeError inside its body (e.g. remote code against the wrong transformers) must propagate, not be treated as an uncallable signature."""

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


class _UnreadableSignature(PreTrainedModel):
    """Callable whose inspect.signature() raises TypeError, the same exception bind() raises for a required argument."""

    config_class = LlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.emb = nn.Embedding(64, 16)
        self.head = nn.Linear(16, 64, bias = False)

    def get_input_embeddings(self):
        return self.emb

    def get_output_embeddings(self):
        return self.head


_UnreadableSignature.get_input_embeddings.__signature__ = 42


def test_an_unreadable_signature_counts_as_callable(tokenizer):
    """Stated policy, now pinned: unreadable metadata must not mean "skip"."""
    import inspect

    model = _UnreadableSignature(_config())
    with pytest.raises(TypeError):
        inspect.signature(model.get_input_embeddings)
    untrained = slice(5, 8)
    with torch.no_grad():
        model.emb .weight[untrained] = 0.0
        model.head.weight[untrained] = 0.0
    dataset = datasets.Dataset.from_dict({"input_ids": [[2, 3, 4, 5]]})
    fix_untrained_tokens(model, tokenizer, dataset)
    assert model.emb .weight[untrained].abs().sum() > 0
    assert model.head.weight[untrained].abs().sum() > 0


class _BrokenInputAndUncallableOutput(PreTrainedModel):
    """Input accessor fails in its body while the output accessor cannot be called; checking either signature against either failure would hide the first error."""

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
    """Catching every TypeError would let a real failure run silently without its NaN guard, the one outcome worse than the crash."""
    model = _RaisesTypeErrorInsideTheBody(_config())
    with pytest.raises(TypeError, match = "something inside the model went wrong"):
        fix_untrained_tokens(model, tokenizer, dataset)


def test_the_skip_is_visible_without_opting_into_logging(caplog, tokenizer, dataset):
    """The logger is at WARNING unless UNSLOTH_ENABLE_LOGGING is set, so the skip must be logged at warning level or the user never sees it."""
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
    assert torch.equal(before, after)


def test_untrained_tokens_really_are_still_reset(tokenizer):
    """An unchanged embedding also happens when the pass is skipped for every model, so this test uses genuinely untrained rows that must be reset."""
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
