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

"""Fidelity tests for the Gemma3Processor automatic longest-padding override.

The forced padding="longest" must not leak into two things the caller still controls:
an explicitly supplied padding (padding=None means do-not-pad, so key presence is what
counts), and the truncation HF derives from (max_length, padding is False, truncation
unset) -> "longest_first", which reading the forced value would silently discard.
"""
import pytest

transformers = pytest.importorskip("transformers")

from transformers.models.gemma3.processing_gemma3 import Gemma3Processor
from unsloth_zoo.temporary_patches.gemma import patch_Gemma3Processor

BOS, PAD = 2, 0
SHORT = "one two"                       # 2 words -> 3 ids after the double BOS strip
LONG = " ".join(f"w{i}" for i in range(30))   # 30 words -> 31 ids after the strip


class _StubTokenizer:
    """Minimal stand-in for the Gemma3 tokenizer: doubled BOS, PreTrainedTokenizerBase
    truncation semantics, and it records the kwargs it was handed."""
    bos_token_id = BOS
    pad_token_id = PAD
    image_token_id = 99
    padding_side = "left"
    model_max_length = 1024
    init_kwargs = {}

    def __init__(self):
        self.last_kwargs = None

    def __call__(self, text = None, **kwargs):
        self.last_kwargs = dict(kwargs)
        truncation = kwargs.get("truncation", False)
        max_length = kwargs.get("max_length", None)
        rows = []
        for prompt in text:
            ids = [BOS, BOS] + [10 + i for i in range(len(prompt.split()))]
            if truncation not in (False, None, "do_not_truncate") and max_length is not None:
                ids = ids[:max_length]
            rows.append(ids)
        return {"input_ids": rows, "attention_mask": [[1] * len(r) for r in rows]}


@pytest.fixture(scope = "module")
def processor():
    original_call = Gemma3Processor.__call__
    patch_Gemma3Processor()
    if "unsloth_zoo" not in getattr(Gemma3Processor.__call__, "__code__", original_call.__code__).co_filename:
        Gemma3Processor.__call__ = original_call
        pytest.skip("Gemma3Processor.__call__ was not patched by this transformers version")
    stub = Gemma3Processor.__new__(Gemma3Processor)
    stub.tokenizer = _StubTokenizer()
    stub.image_processor = None
    stub.boi_token = "<start_of_image>"
    stub.image_token_id = 99
    stub.full_image_sequence = "<img>"
    yield stub
    Gemma3Processor.__call__ = original_call


def row_lengths(batch):
    return [len(row) for row in batch["input_ids"]]


def test_ragged_multi_row_still_padded_without_padding_kwarg(processor):
    out = processor(text = [SHORT, LONG], return_tensors = None)
    assert row_lengths(out) == [31, 31], row_lengths(out)


def test_explicit_padding_none_is_not_treated_as_omitted(processor):
    out = processor(text = [SHORT, LONG], padding = None, return_tensors = None)
    assert row_lengths(out) == [3, 31], row_lengths(out)


def test_explicit_padding_none_in_text_kwargs_is_not_treated_as_omitted(processor):
    out = processor(text = [SHORT, LONG], text_kwargs = {"padding": None}, return_tensors = None)
    assert row_lengths(out) == [3, 31], row_lengths(out)


def test_explicit_padding_false_still_ragged(processor):
    out = processor(text = [SHORT, LONG], padding = False, return_tensors = None)
    assert row_lengths(out) == [3, 31], row_lengths(out)


def test_explicit_padding_max_length_still_honoured(processor):
    out = processor(text = [SHORT, LONG], padding = "max_length", max_length = 40, return_tensors = None)
    assert row_lengths(out) == [40, 40], row_lengths(out)


def test_max_length_alone_keeps_implicit_truncation(processor):
    # HF truncates on (max_length, no padding, no truncation); forcing "longest" padding
    # must not disable that.
    out = processor(text = [LONG, SHORT], max_length = 8, return_tensors = None)
    assert processor.tokenizer.last_kwargs["truncation"] == "longest_first"
    assert max(row_lengths(out)) <= 8, row_lengths(out)
    assert row_lengths(out) == [7, 7], row_lengths(out)


def test_max_length_alone_single_row_unchanged(processor):
    out = processor(text = [LONG], max_length = 8, return_tensors = None)
    assert processor.tokenizer.last_kwargs["truncation"] == "longest_first"
    assert row_lengths(out) == [7]


def test_explicit_padding_false_keeps_implicit_truncation(processor):
    out = processor(text = [LONG, SHORT], padding = False, max_length = 8, return_tensors = None)
    assert processor.tokenizer.last_kwargs["truncation"] == "longest_first"
    assert row_lengths(out) == [7, 3], row_lengths(out)


def test_explicit_padding_disables_implicit_truncation(processor):
    out = processor(text = [LONG, SHORT], padding = True, max_length = 8, return_tensors = None)
    assert processor.tokenizer.last_kwargs["truncation"] is False
    assert row_lengths(out) == [31, 31], row_lengths(out)


def test_explicit_truncation_preserved(processor):
    out = processor(text = [LONG, SHORT], max_length = 8, truncation = False, return_tensors = None)
    assert processor.tokenizer.last_kwargs["truncation"] is False
    assert row_lengths(out) == [31, 31], row_lengths(out)


@pytest.fixture(scope = "module")
def processor_tokenizer_padded():
    """Same stub, but the TOKENIZER was initialized with padding="max_length": _merge_kwargs
    copies that over the _defaults even with empty per-call kwargs."""
    original_call = Gemma3Processor.__call__
    patch_Gemma3Processor()
    if "unsloth_zoo" not in getattr(Gemma3Processor.__call__, "__code__", original_call.__code__).co_filename:
        Gemma3Processor.__call__ = original_call
        pytest.skip("Gemma3Processor.__call__ was not patched by this transformers version")
    stub = Gemma3Processor.__new__(Gemma3Processor)
    stub.tokenizer = _StubTokenizer()
    stub.tokenizer.init_kwargs = {"padding": "max_length", "max_length": 40}
    stub.image_processor = None
    stub.boi_token = "<start_of_image>"
    stub.image_token_id = 99
    stub.full_image_sequence = "<img>"
    yield stub
    Gemma3Processor.__call__ = original_call


def test_tokenizer_init_padding_is_not_overridden(processor_tokenizer_padded):
    # _merge_kwargs puts the tokenizer's init padding in text_kwargs, so the automatic
    # "longest" must stand down.
    out = processor_tokenizer_padded(text = [SHORT, LONG], return_tensors = None)
    assert row_lengths(out) == [40, 40], row_lengths(out)


def test_tokenizer_init_padding_still_overridable_at_call_site(processor_tokenizer_padded):
    out = processor_tokenizer_padded(text = [SHORT, LONG], padding = False, return_tensors = None)
    assert row_lengths(out) == [3, 31], row_lengths(out)


def test_non_dict_text_kwargs_still_raises(processor):
    with pytest.raises(AttributeError):
        processor(text = [SHORT, LONG], text_kwargs = None, return_tensors = None)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
