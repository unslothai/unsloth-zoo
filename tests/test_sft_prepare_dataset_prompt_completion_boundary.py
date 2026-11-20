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

"""A prompt ending in whitespace merges into the completion's first token; the completion must start where the tokenizations diverge.
GPT-OSS deletes the analysis of tool-calling turns once a final answer follows; those spans are skipped so tool history stays masked."""

import pytest
from datasets import Dataset, IterableDataset

tokenizers = pytest.importorskip("tokenizers")
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

import unsloth_zoo.dataset_utils as dataset_utils  # noqa: E402
from unsloth_zoo.dataset_utils import _prompt_completion_boundary, sft_prepare_dataset  # noqa: E402


@pytest.fixture(autouse = True)
def _keep_completion_mask(monkeypatch):
    # Keep completion_mask rather than folding it into labels, which depends on the installed TRL.
    monkeypatch.setattr(dataset_utils, "trl_collator_applies_masks", lambda: True)


# Qwen2 / Qwen3 pre-tokenizer: a leading space stays attached to the following letters.
QWEN2_SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+|\s+(?!\S)|\s+"
)
EOS = "<|im_end|>"
# ChatML without trimming, like Qwen2.5 / Typhoon 2.5: the generation prompt ends in "\n".
CHATML = (
    "{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>\\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)

ROWS = [
    # The prompt's trailing space merges with the answer: " two", " สอง".
    {"prompt": "Question: one plus one? Answer: ", "completion": "two"},
    {"prompt": "คำถาม: หนึ่งบวกหนึ่ง คำตอบ: ", "completion": "สอง"},
    # Controls: the space is on the completion side.
    {"prompt": "Question: one plus one? Answer:", "completion": " two"},
    {"prompt": "คำถาม: หนึ่งบวกหนึ่ง คำตอบ:", "completion": " สอง"},
]


@pytest.fixture(scope = "module")
def tokenizer():
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(QWEN2_SPLIT), behavior = "isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space = False, use_regex = False),
    ])
    tok.decoder = decoders.ByteLevel()
    corpus = [row["prompt"].rstrip() + " " + row["completion"].strip() for row in ROWS] * 50
    trainer = trainers.BpeTrainer(
        vocab_size = 600, special_tokens = [EOS, "<|im_start|>"],
        initial_alphabet = pre_tokenizers.ByteLevel.alphabet(),
    )
    corpus += ["<|im_start|>user\nQuestion<|im_end|>\n<|im_start|>assistant\n\nAnswer.<|im_end|>\n"] * 50
    tok.train_from_iterator(corpus, trainer = trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object = tok, eos_token = EOS, pad_token = EOS)
    fast.chat_template = CHATML
    return fast


class Args:
    max_length = 128
    dataset_text_field = "text"
    remove_unused_columns = True
    dataset_num_proc = None


class DummyTrainer:
    def __init__(self):
        self.model = None
        self.data_collator = None


def _trained_texts(dataset, tok):
    prepared = sft_prepare_dataset(
        DummyTrainer(), dataset, tok, Args(), packing = False, formatting_func = None, dataset_name = "train",
    )
    out = []
    for row in prepared:
        ids, mask = list(row["input_ids"]), list(row["completion_mask"])
        assert len(ids) == len(mask)
        out.append(tok.decode([t for t, m in zip(ids, mask) if m]))
    return out


def test_premise_trailing_space_merges_into_the_answer(tokenizer):
    for row in ROWS[:2]:
        prompt_ids = tokenizer(row["prompt"]).input_ids
        joined_ids = tokenizer(row["prompt"] + row["completion"]).input_ids
        assert joined_ids[:len(prompt_ids)] != prompt_ids, row


@pytest.mark.parametrize("streaming", [False, True], ids = ["batched", "row_by_row"])
def test_answer_is_trained_when_prompt_ends_in_space(tokenizer, streaming):
    dataset = Dataset.from_list(ROWS)
    if streaming:
        dataset = IterableDataset.from_generator(lambda: iter(ROWS))
    texts = _trained_texts(dataset, tokenizer)
    for row, text in zip(ROWS, texts):
        assert text == " " + row["completion"].strip() + EOS, (row, text)
        assert row["prompt"].strip().split()[-1] not in text


def test_boundary_unchanged_when_prompt_is_a_prefix():
    assert _prompt_completion_boundary([1, 2, 3], [1, 2, 3, 4, 5]) == 3
    assert _prompt_completion_boundary([1, 2, 3], [1, 2]) == 2


def test_boundary_is_where_tokenizations_diverge():
    # [.., 7] is the prompt's lone trailing space; [.., 9] is " two" in the joined ids.
    assert _prompt_completion_boundary([1, 2, 7], [1, 2, 9, 5]) == 2
    assert _prompt_completion_boundary([1, 2, 7], [3, 9]) == 0


def test_conversational_rows_keep_the_answer_and_mask_the_prompt(tokenizer):
    rows = [
        {"prompt": [{"role": "user", "content": "Question"}], "completion": [{"role": "assistant", "content": c}]}
        for c in ("Answer.", "\nAnswer.")
    ]
    texts = _trained_texts(Dataset.from_list(rows), tokenizer)
    for text in texts:
        assert "Answer.<|im_end|>" in text, text
        assert "Question" not in text and "user" not in text and "<|im_start|>" not in text, text


def test_boundary_skips_spans_the_template_deleted_from_the_prompt():
    # The prompt alone renders [8, 9] that the joined ids drop; the rest of the prompt still follows.
    assert _prompt_completion_boundary([1, 2, 8, 9, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7, 8]) == 6
    # Two deleted spans (two tool-calling turns): [8] and [9, 9].
    assert _prompt_completion_boundary(
        [1, 2, 8, 3, 4, 5, 6, 9, 9, 10, 11, 12, 13, 14], [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 20],
    ) == 11
    # A deleted span, then the prompt's last token merges with the completion.
    assert _prompt_completion_boundary([1, 2, 8, 9, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 10, 11]) == 6


def test_boundary_keeps_the_divergence_for_prompt_only_closing_tokens():
    # Gemma 4's "<turn|>\n" at the end of the prompt only.
    assert _prompt_completion_boundary([1, 2, 8, 9], [1, 2, 5, 6]) == 2
    # A completion starting with the prompt's last token is not a deleted span.
    assert _prompt_completion_boundary([1, 2, 8, 9], [1, 2, 9, 6]) == 2
    # DeepSeek-R1: the prompt ends "<think>", "\n", the joined ids drop "<think>"; the completion's "\n" stays trained.
    assert _prompt_completion_boundary([1, 2, 30, 31, 32, 33, 40, 41], [1, 2, 30, 31, 32, 33, 41, 50, 60]) == 6


def test_boundary_starts_after_the_prompt_tail_when_it_cannot_line_up():
    # [8, 8] became [9, 9] and the prompt's last ids reappear later: start after them.
    assert _prompt_completion_boundary([1, 2, 8, 8, 3, 4, 5, 6], [1, 2, 9, 9, 3, 4, 5, 6, 20]) == 8
    # Deleted spans too close together to anchor: the answer [20, 21] is still trained whole.
    assert _prompt_completion_boundary([1, 2, 8, 3, 4, 9, 10, 5, 6, 7, 11], [1, 2, 3, 4, 5, 6, 7, 11, 20, 21]) == 8
    # Mistral v0.3: the joined ids drop the system text [30, 31, 32] and "Cap" "ital" becomes "▁Capital" [41].
    assert _prompt_completion_boundary([1, 2, 30, 31, 32, 40, 42, 5, 6, 7, 9], [1, 2, 41, 5, 6, 7, 9, 50, 51]) == 7


def test_boundary_warns_once_per_cause(monkeypatch, capsys):
    monkeypatch.setattr(dataset_utils, "_PROMPT_COMPLETION_BOUNDARY_WARNED", set())
    for _ in range(2):
        _prompt_completion_boundary([1, 2, 8, 9, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7, 8])
        _prompt_completion_boundary([1, 2, 7], [1, 2, 9, 5])
        _prompt_completion_boundary([1, 2, 8, 8, 3, 4, 5, 6], [1, 2, 9, 9, 3, 4, 5, 6, 20])
    out = capsys.readouterr().out
    assert out.count("renders earlier turns of the prompt differently") == 1
    assert out.count("merges with the completion's first token") == 1
    assert out.count("could not be lined up") == 1


GPT_OSS = ("openai/gpt-oss-20b", "6cee5e81ee83917806bbde320786a8fb61efebee")
GPT_OSS_ANSWER = "The instrument read 92 degrees Celsius at noon."


def _gpt_oss_row(*turns, field = "content"):
    # chat_template.jinja drops "content" / "thinking" of tool-calling turns before a final answer.
    prompt = [{"role": "user", "content": "What did the meter read at noon?"}]
    for text, name, result in turns:
        call = {"type": "function", "function": {"name": name, "arguments": {}}}
        # "content": "" renders the same and keeps Arrow from filling None.
        turn = {"role": "assistant", "content": "", field: text, "tool_calls": [call]}
        prompt += [turn, {"role": "tool", "content": result}]
    return {"prompt": prompt, "completion": [{"role": "assistant", "content": GPT_OSS_ANSWER}]}


GPT_OSS_ROWS = {
    # Rebuilt from the row described on unslothai/unsloth-zoo#2402 (its user question was not posted).
    "one_tool_call": _gpt_oss_row(("Read it.", "read_meter", "METER_RESULT_176208")),
    "two_tool_calls": _gpt_oss_row(
        ("Read it.", "read_meter", "METER_RESULT_176208"), ("Check units.", "read_units", "UNITS_RESULT_2"),
    ),
    "two_tool_calls_thinking": _gpt_oss_row(
        ("Read it.", "read_meter", "METER_RESULT_176208"), ("Check units.", "read_units", "UNITS_RESULT_2"),
        field = "thinking",
    ),
}


@pytest.fixture(scope = "module")
def gpt_oss_tokenizer():
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(GPT_OSS[0], revision = GPT_OSS[1])
    except Exception as e:
        pytest.skip(reason = f"offline runner, tokenizer download failed: {e}")
    if not tok.chat_template:
        pytest.skip(reason = "gpt-oss chat_template.jinja is not available")
    return tok


def _gpt_oss_trained(tok, example, monkeypatch, streaming = False, columns = "completion_mask", **overrides):
    if columns == "labels":
        monkeypatch.setattr(dataset_utils, "trl_collator_applies_masks", lambda: False)
        monkeypatch.setattr(dataset_utils, "trl_collator_ignores_labels", lambda collator = None: False)
    dataset = Dataset.from_list([example])
    if streaming:
        dataset = IterableDataset.from_generator(lambda: iter([example]))

    class GptOssArgs(Args):
        max_length = 1024
    for k, v in overrides.items():
        setattr(GptOssArgs, k, v)

    row = next(iter(sft_prepare_dataset(
        DummyTrainer(), dataset, tok, GptOssArgs(), packing = False, formatting_func = None, dataset_name = "train",
    )))
    ids = list(row["input_ids"])
    if columns == "labels":
        labels = list(row["labels"])
        assert all(label in (-100, t) for t, label in zip(ids, labels))
        return ids, [label != -100 for label in labels]
    return ids, [bool(m) for m in row["completion_mask"]]


def _assert_only_the_final_answer_is_trained(tok, ids, trained):
    assert tok.decode([t for t, m in zip(ids, trained) if m]) == \
        "<|channel|>final<|message|>" + GPT_OSS_ANSWER + "<|return|>"
    assert trained[-1] and ids[-1] == tok.convert_tokens_to_ids("<|return|>")


@pytest.mark.parametrize("columns", ["completion_mask", "labels"])
@pytest.mark.parametrize("streaming", [False, True], ids = ["batched", "row_by_row"])
@pytest.mark.parametrize("name", list(GPT_OSS_ROWS))
def test_gpt_oss_tool_history_stays_masked(gpt_oss_tokenizer, monkeypatch, name, streaming, columns):
    tok, example = gpt_oss_tokenizer, GPT_OSS_ROWS[name]
    ids, trained = _gpt_oss_trained(tok, example, monkeypatch, streaming, columns)
    _assert_only_the_final_answer_is_trained(tok, ids, trained)
    # Every token up to the end of the last tool response, each tool call included, is masked.
    last_result = example["prompt"][-1]["content"]
    tool_end = next(k for k in range(len(ids) + 1) if f'"{last_result}"<|end|>' in tok.decode(ids[:k]))
    history = tok.decode(ids[:tool_end])
    for message in example["prompt"]:
        for call in message.get("tool_calls", []):
            assert "to=functions." + call["function"]["name"] in history
        if message["role"] == "tool":
            assert message["content"] in history
    assert not any(trained[:tool_end])


@pytest.mark.parametrize("streaming", [False, True], ids = ["batched", "row_by_row"])
def test_gpt_oss_tool_history_stays_masked_with_keep_end(gpt_oss_tokenizer, monkeypatch, streaming):
    # The boundary is found on the full row, then keep_end cuts every column alike.
    tok, example = gpt_oss_tokenizer, GPT_OSS_ROWS["two_tool_calls"]
    full, _ = _gpt_oss_trained(tok, example, monkeypatch, streaming)
    ids, trained = _gpt_oss_trained(tok, example, monkeypatch, streaming, max_length = 96, truncation_mode = "keep_end")
    assert len(full) > 96 and ids == full[-96:]
    _assert_only_the_final_answer_is_trained(tok, ids, trained)
    assert "UNITS_RESULT_2" in tok.decode(ids)
