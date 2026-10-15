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

"""fix_untrained_tokens must find the head of a wrapped CausalLM (Nemotron-3-Nano-Omni)."""
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
        lm.embed_tokens.weight[VOCAB - 2:].zero_()
        lm.lm_head.weight[VOCAB - 2:].zero_()
    return model


def test_wrapper_alone_has_no_head():
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
    config_class = _Cfg

    def __init__(self, config):
        super().__init__(config)
        self.vision_decoder = _CausalLM(config)  # must register before language_model
        self.language_model = _CausalLM(config)

    def get_input_embeddings(self):
        return self.language_model.embed_tokens


def test_resolver_prefers_the_submodel_the_model_points_at():
    model = _TwoSubModels(_Cfg())
    embeddings, lm_head = _get_embedding_modules(model)
    assert embeddings is model.language_model.embed_tokens
    assert lm_head is model.language_model.lm_head, "picked the wrong sub-model's head"


def test_fix_untrained_tokens_writes_into_the_right_submodel():
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
    model = _model_with_untrained_rows()
    lm = model.language_model
    before = lm.lm_head.weight[VOCAB - 2:].clone()
    assert torch.all(before == 0)
    ds = Dataset.from_dict({"input_ids": [[1, 2, VOCAB - 1, VOCAB - 2]]})
    fix_untrained_tokens(model, _Tokenizer(), ds)
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
        fix_untrained_tokens(_Headless(_Cfg()), _Tokenizer(), ds)
    assert any("Skipping the untrained token fix" in r.message for r in caplog.records)


def test_resolver_lets_a_type_error_inside_an_accessor_propagate():
    class _Broken(PreTrainedModel):
        config_class = _Cfg
        def __init__(self, config):
            super().__init__(config)
            self.encoder = nn.Linear(DIM, DIM)
        def get_output_embeddings(self):
            raise TypeError("something inside the model went wrong")
    with pytest.raises(TypeError, match = "something inside the model went wrong"):
        _get_embedding_modules(_Broken(_Cfg()))


class _TwoHeadsNoPointer(PreTrainedModel):
    """No top-level accessor answers; two nested heads (Qwen3-Omni: thinker and talker.code_predictor)."""

    config_class = _Cfg

    def __init__(self, config, first, second):
        super().__init__(config)
        setattr(self, first, _CausalLM(config))
        setattr(self, second, _CausalLM(config))

    def get_input_embeddings(self):
        raise NotImplementedError


def test_ambiguous_heads_pick_the_language_model_whatever_the_order():
    model = _TwoHeadsNoPointer(_Cfg(), "vision_decoder", "language_model")
    embeddings, head = _get_embedding_modules(model)
    assert head is model.language_model.lm_head
    assert embeddings is model.language_model.embed_tokens


def test_ambiguous_heads_without_a_language_model_are_skipped():
    model = _TwoHeadsNoPointer(_Cfg(), "vision_decoder", "audio_decoder")
    _, head = _get_embedding_modules(model)
    assert head is None
