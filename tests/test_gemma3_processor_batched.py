"""Regression test for the Gemma3 batched-processor padding fix.

Batched Gemma3 tokenization used to leave ragged input_ids after the double-BOS strip (the strip
only shortened the row that still started with [bos, bos], i.e. the longest one), so BatchFeature
could not stack them. _fix_double_bos_and_pad strips the duplicate BOS on every row and pads all
sequence fields. Padding follows the HF contract: it honours "do_not_pad"/False, pads list output
even when return_tensors is None, and targets max_length when padding="max_length".
"""
from unsloth_zoo.temporary_patches.gemma import _fix_double_bos_and_pad

BOS, PAD, IMG = 2, 0, 99


def test_batched_ragged_rows_become_stackable():
    text_inputs = {
        "input_ids": [[BOS, BOS, 10, 11], [BOS, BOS, 20, 21, 22, 23]],
        "attention_mask": [[1, 1, 1, 1], [1, 1, 1, 1, 1, 1]],
    }
    out = _fix_double_bos_and_pad(text_inputs, BOS, PAD, IMG, True, True, "left", "pt")
    assert {len(x) for x in out["input_ids"]} == {5}, out["input_ids"]   # 6 - 1 stripped BOS
    assert all(row.count(BOS) == 1 for row in out["input_ids"])          # duplicate BOS gone everywhere
    assert out["input_ids"][0][0] == PAD and out["attention_mask"][0][0] == 0  # shorter row left-padded
    assert all(len(t) == 5 for t in out["token_type_ids"])


def test_single_row_strips_bos_without_padding():
    text_inputs = {"input_ids": [[BOS, BOS, 10, 11, 12]], "attention_mask": [[1, 1, 1, 1, 1]]}
    out = _fix_double_bos_and_pad(text_inputs, BOS, PAD, IMG, True, True, "left", "pt")
    assert out["input_ids"] == [[BOS, 10, 11, 12]]


def test_right_padding_side():
    text_inputs = {"input_ids": [[BOS, BOS, 10], [BOS, BOS, 20, 21, 22]], "attention_mask": [[1, 1, 1], [1, 1, 1, 1, 1]]}
    out = _fix_double_bos_and_pad(text_inputs, BOS, PAD, IMG, True, True, "right", "pt")
    assert out["input_ids"][0][-1] == PAD and out["attention_mask"][0][-1] == 0


def test_padding_true_pads_even_without_return_tensors():
    # HF contract: padding=True pads the list output even when return_tensors is None.
    text_inputs = {"input_ids": [[BOS, BOS, 10], [BOS, BOS, 20, 21]], "attention_mask": [[1, 1, 1], [1, 1, 1, 1]]}
    out = _fix_double_bos_and_pad(text_inputs, BOS, PAD, IMG, True, True, "left", None)
    assert {len(x) for x in out["input_ids"]} == {3}   # stripped then padded to batch max


def test_do_not_pad_leaves_ragged():
    # "do_not_pad" is truthy but must NOT pad.
    text_inputs = {"input_ids": [[BOS, BOS, 10], [BOS, BOS, 20, 21]], "attention_mask": [[1, 1, 1], [1, 1, 1, 1]]}
    out = _fix_double_bos_and_pad(text_inputs, BOS, PAD, IMG, True, "do_not_pad", "left", "pt")
    assert [len(x) for x in out["input_ids"]] == [2, 3]   # stripped, not padded


def test_padding_false_leaves_ragged():
    text_inputs = {"input_ids": [[BOS, BOS, 10], [BOS, BOS, 20, 21]], "attention_mask": [[1, 1, 1], [1, 1, 1, 1]]}
    out = _fix_double_bos_and_pad(text_inputs, BOS, PAD, IMG, True, False, "left", "pt")
    assert [len(x) for x in out["input_ids"]] == [2, 3]


def test_max_length_padding():
    text_inputs = {"input_ids": [[BOS, BOS, 10], [BOS, BOS, 20, 21]], "attention_mask": [[1, 1, 1], [1, 1, 1, 1]]}
    out = _fix_double_bos_and_pad(text_inputs, BOS, PAD, IMG, True, "max_length", "left", "pt", max_length=6)
    assert {len(x) for x in out["input_ids"]} == {6}


def test_existing_token_type_ids_stripped_in_lockstep():
    # Tokenizer-provided token_type_ids must be BOS-stripped too, or the row desyncs when
    # return_mm_token_type_ids is False.
    text_inputs = {
        "input_ids": [[BOS, BOS, 10, 11], [BOS, BOS, 20]],
        "attention_mask": [[1, 1, 1, 1], [1, 1, 1]],
        "token_type_ids": [[0, 0, 0, 0], [0, 0, 0]],
    }
    out = _fix_double_bos_and_pad(text_inputs, BOS, PAD, IMG, False, True, "left", "pt")
    assert all(len(t) == len(i) for t, i in zip(out["token_type_ids"], out["input_ids"]))


def test_special_tokens_mask_stripped_and_padded():
    # any per-token field (here special_tokens_mask) must be stripped and padded, not left ragged;
    # its pad fill is 1 (padding is a special token).
    text_inputs = {
        "input_ids": [[BOS, BOS, 10], [BOS, BOS, 20, 21]],
        "attention_mask": [[1, 1, 1], [1, 1, 1, 1]],
        "special_tokens_mask": [[1, 1, 0], [1, 1, 0, 0]],
    }
    out = _fix_double_bos_and_pad(text_inputs, BOS, PAD, IMG, False, True, "left", "pt")
    assert {len(x) for x in out["special_tokens_mask"]} == {3}
    assert all(len(m) == len(i) for m, i in zip(out["special_tokens_mask"], out["input_ids"]))
    assert out["special_tokens_mask"][0][0] == 1   # left pad filled as special


def test_offset_mapping_tuples_stripped_and_padded():
    # offset_mapping rows are (start, end) tuples; stripping drops the first pair and padding fills (0, 0).
    text_inputs = {
        "input_ids": [[BOS, BOS, 10], [BOS, BOS, 20, 21]],
        "attention_mask": [[1, 1, 1], [1, 1, 1, 1]],
        "offset_mapping": [[(0, 0), (0, 0), (0, 2)], [(0, 0), (0, 0), (0, 2), (2, 4)]],
    }
    out = _fix_double_bos_and_pad(text_inputs, BOS, PAD, IMG, False, True, "left", "pt")
    assert {len(x) for x in out["offset_mapping"]} == {3}
    assert out["offset_mapping"][0][0] == (0, 0)    # left pad filled with an offset pair


def test_pad_to_multiple_of():
    text_inputs = {"input_ids": [[BOS, BOS, 10], [BOS, BOS, 20, 21]], "attention_mask": [[1, 1, 1], [1, 1, 1, 1]]}
    out = _fix_double_bos_and_pad(text_inputs, BOS, PAD, IMG, True, True, "left", "pt", pad_to_multiple_of=8)
    assert {len(x) for x in out["input_ids"]} == {8}   # batch max 3 rounded up to 8


def test_max_length_uses_model_max_when_unset():
    text_inputs = {"input_ids": [[BOS, BOS, 10], [BOS, BOS, 20, 21]], "attention_mask": [[1, 1, 1], [1, 1, 1, 1]]}
    out = _fix_double_bos_and_pad(text_inputs, BOS, PAD, IMG, True, "max_length", "left", "pt", model_max_length=6)
    assert {len(x) for x in out["input_ids"]} == {6}


if __name__ == "__main__":
    test_batched_ragged_rows_become_stackable()
    test_single_row_strips_bos_without_padding()
    test_right_padding_side()
    test_padding_true_pads_even_without_return_tensors()
    test_do_not_pad_leaves_ragged()
    test_padding_false_leaves_ragged()
    test_max_length_padding()
    test_existing_token_type_ids_stripped_in_lockstep()
    test_special_tokens_mask_stripped_and_padded()
    test_offset_mapping_tuples_stripped_and_padded()
    test_pad_to_multiple_of()
    test_max_length_uses_model_max_when_unset()
    print("OK: all gemma3 batched-processor padding tests passed")
