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

"""Embedding rows added for a new pad token must be trainable-safe tensors.

`patch_tokenizer` runs under `torch.inference_mode`, so a resize done there used to
leave the embedding (and a tied lm_head) as inference tensors: eager training then
failed with "Inference tensors cannot be saved for backward" and torch.compile graph
broke on every fake tensor call touching them (Moonshine).
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from unsloth_zoo.pad_token import fix_pad_token


class _Tokenizer:
    """pad == eos and no reserved / pad-named token, so fix_pad_token must add one."""

    def __init__(self, n):
        self._vocab = {f"tok{i}": i for i in range(n)}
        self.pad_token = self.eos_token = "tok1"
        self.added_tokens_decoder = {}

    def __len__(self):
        return len(self._vocab)

    @property
    def pad_token_id(self):
        return self._vocab.get(self.pad_token)

    @property
    def eos_token_id(self):
        return self._vocab.get(self.eos_token)

    def get_vocab(self):
        return dict(self._vocab)

    def convert_ids_to_tokens(self, token_id):
        return next((t for t, i in self._vocab.items() if i == token_id), None)

    def convert_tokens_to_ids(self, token):
        return self._vocab.get(token)

    def __call__(self, text, add_special_tokens = False):
        return type("Enc", (), {"input_ids": [self._vocab.get(text, 0)]})()

    def add_special_tokens(self, mapping):
        token = mapping["pad_token"]
        self._vocab.setdefault(token, len(self._vocab))
        self.pad_token = token


def test_pad_token_resize_under_inference_mode_is_trainable():
    vocab = 16
    config = transformers.GPT2Config(
        vocab_size = vocab, n_positions = 8, n_embd = 8, n_layer = 1, n_head = 2,
        tie_word_embeddings = True,
    )
    model = transformers.GPT2LMHeadModel(config)
    tokenizer = _Tokenizer(vocab)
    with torch.inference_mode():
        try:
            fix_pad_token(tokenizer, model = model, model_config = model.config)
        except RuntimeError:
            pass  # the new id needed a resize; the rows are still grown first
    embed = model.get_input_embeddings().weight
    assert embed.shape[0] > vocab
    assert not embed.is_inference()
    assert not model.get_output_embeddings().weight.is_inference()

    for p in model.parameters():
        p.requires_grad_(False)
    model.transformer.h[0].mlp.c_fc.weight.requires_grad_(True)
    ids = torch.tensor([[2, 3, 4, 5]])
    loss = model(input_ids = ids, labels = ids).loss
    loss.backward()
    assert model.transformer.h[0].mlp.c_fc.weight.grad is not None
