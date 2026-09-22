# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""fix_untrained_tokens must find the output head of a wrapped CausalLM.

nvidia/Nemotron-3-Nano-Omni-30B-A3B wraps a complete NemotronHForCausalLM as
`language_model` inside a PreTrainedModel that defines neither embedding
accessor. transformers' default `get_input_embeddings` finds `embed_tokens`
through the wrapper, but `get_output_embeddings` returns None because the
wrapper has no `lm_head` of its own, so SFTTrainer's untrained-token pass died
with "'NoneType' object has no attribute 'weight'" right after get_peft_model.

Built on small PreTrainedModels, no downloads; each test states which arm it
measures.
"""
import pytest
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import PretrainedConfig, PreTrainedModel

from unsloth_zoo.tokenizer_utils import _get_embedding_modules, fix_untrained_tokens

VOCAB, DIM = 16, 4


class _Cfg(PretrainedConfig):
    model_type = "unsloth-test-nested-head"


class _CausalLM(PreTrainedModel):
    config_class = _Cfg

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(VOCAB, DIM)
        self.lm_head = nn.Linear(DIM, VOCAB, bias = False)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


class _Wrapper(PreTrainedModel):
    """The Nemotron-Omni shape: a wrapper with no accessors of its own."""
    config_class = _Cfg

    def __init__(self, config):
        super().__init__(config)
        self.language_model = _CausalLM(config)
        self.vision_model = nn.Linear(DIM, DIM)


class _Tokenizer:
    bos_token = eos_token = unk_token = sep_token = pad_token = None
    cls_token = mask_token = None
    chat_template = None
    all_special_ids = []
    def __len__(self): return VOCAB
    def convert_tokens_to_ids(self, x): return None
    def convert_ids_to_tokens(self, ids): return [f"<tok{i}>" for i in ids]


def _model_with_untrained_rows():
    torch.manual_seed(0)
    model = _Wrapper(_Cfg(_name_or_path = "test/nested-head"))
    lm = model.language_model
    with torch.no_grad():
        lm.embed_tokens.weight.normal_()
        lm.lm_head.weight.normal_()
        lm.embed_tokens.weight[VOCAB - 2:].zero_()   # two untrained tokens
        lm.lm_head.weight[VOCAB - 2:].zero_()
    return model


def test_wrapper_alone_has_no_head():
    """The precondition: transformers cannot see the head through the wrapper."""
    model = _model_with_untrained_rows()
    assert model.get_output_embeddings() is None


def test_resolver_looks_through_the_wrapper():
    model = _model_with_untrained_rows()
    embeddings, lm_head = _get_embedding_modules(model)
    assert lm_head is model.language_model.lm_head
    assert embeddings is model.language_model.embed_tokens


def test_resolver_prefers_the_model_s_own_accessors():
    model = _CausalLM(_Cfg())
    assert _get_embedding_modules(model) == (model.embed_tokens, model.lm_head)


class _TwoSubModels(PreTrainedModel):
    """A wrapper that registers a head-owning sub-model BEFORE its language model.

    named_modules() is registration order, so "the first nested module that
    answers" is whichever sub-model happens to be declared first.
    """
    config_class = _Cfg

    def __init__(self, config):
        super().__init__(config)
        self.vision_decoder = _CausalLM(config)     # registered first, own head
        self.language_model = _CausalLM(config)

    def get_input_embeddings(self):
        return self.language_model.embed_tokens


def test_resolver_prefers_the_submodel_the_model_points_at():
    """The arm that fails when the search takes the first answer it finds."""
    model = _TwoSubModels(_Cfg())
    embeddings, lm_head = _get_embedding_modules(model)
    assert embeddings is model.language_model.embed_tokens
    assert lm_head is model.language_model.lm_head, "picked the wrong sub-model's head"


def test_fix_untrained_tokens_writes_into_the_right_submodel():
    """A wrong head is silent, not loud: vocabularies are only compared by min(len),
    so the mean embeddings land in the other sub-model and nothing raises."""
    torch.manual_seed(0)
    model = _TwoSubModels(_Cfg(_name_or_path = "test/two-submodels"))
    for sub in (model.vision_decoder, model.language_model):
        with torch.no_grad():
            sub.embed_tokens.weight.normal_(); sub.lm_head.weight.normal_()
            sub.embed_tokens.weight[VOCAB - 2:].zero_(); sub.lm_head.weight[VOCAB - 2:].zero_()
    ds = Dataset.from_dict({"input_ids": [[1, 2, VOCAB - 1, VOCAB - 2]]})
    fix_untrained_tokens(model, _Tokenizer(), ds)
    assert not torch.all(model.language_model.lm_head.weight[VOCAB - 2:] == 0)
    assert torch.all(model.vision_decoder.lm_head.weight[VOCAB - 2:] == 0), \
        "the untrained token fix wrote into the wrong sub-model"


def test_resolver_returns_none_when_nothing_owns_a_head():
    class _Headless(PreTrainedModel):
        config_class = _Cfg
        def __init__(self, config):
            super().__init__(config)
            self.encoder = nn.Linear(DIM, DIM)
    embeddings, lm_head = _get_embedding_modules(_Headless(_Cfg()))
    assert lm_head is None


def test_fix_untrained_tokens_runs_through_the_wrapper():
    """The arm that fails on a tree without the resolver."""
    model = _model_with_untrained_rows()
    lm = model.language_model
    before = lm.lm_head.weight[VOCAB - 2:].clone()
    assert torch.all(before == 0)
    ds = Dataset.from_dict({"input_ids": [[1, 2, VOCAB - 1, VOCAB - 2]]})
    fix_untrained_tokens(model, _Tokenizer(), ds)
    # the untrained rows that the dataset uses were reset to the trained mean
    assert not torch.all(lm.lm_head.weight[VOCAB - 2:] == 0)
    assert not torch.all(lm.embed_tokens.weight[VOCAB - 2:] == 0)


def test_fix_untrained_tokens_skips_a_headless_model(caplog):
    import logging
    import unsloth_zoo.tokenizer_utils as tokenizer_utils
    class _Headless(PreTrainedModel):
        config_class = _Cfg
        def __init__(self, config):
            super().__init__(config)
            self.encoder = nn.Linear(DIM, DIM)
    ds = Dataset.from_dict({"input_ids": [[1, 2]]})
    with caplog.at_level(logging.WARNING, logger = tokenizer_utils.logger.name):
        fix_untrained_tokens(_Headless(_Cfg()), _Tokenizer(), ds)   # no raise
    assert any("Skipping the untrained token fix" in r.message for r in caplog.records)


def test_resolver_lets_a_type_error_inside_an_accessor_propagate():
    """A TypeError raised inside a callable accessor is a real failure, not
    "no embeddings": the main-line guard must survive the wrapper search."""
    class _Broken(PreTrainedModel):
        config_class = _Cfg
        def __init__(self, config):
            super().__init__(config)
            self.encoder = nn.Linear(DIM, DIM)
        def get_output_embeddings(self):
            raise TypeError("something inside the model went wrong")
    with pytest.raises(TypeError, match = "something inside the model went wrong"):
        _get_embedding_modules(_Broken(_Cfg()))
