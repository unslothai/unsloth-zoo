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

"""Auto-detect must pick a response marker the LAST assistant turn carries.

ERNIE-4.5-Thinking renders history assistant turns as "<|im_start|>assistant\\n<response>\\n..."
but the last one as "<|im_start|>assistant\\n<think>\\n\\n</think>\\n<response>\\n...". The gap
mode of the 3-turn probe is the history header, which a single-turn row never contains, so
every single-turn row was masked away ("masked every label to -100 in eval_dataset").
Also: a raise on eval_dataset must leave train_dataset untouched, else a retry with other
markers intersects with the stale labels and masks everything.
"""
import types

import pytest


def _setup():
    from transformers import AutoTokenizer
    tok = None
    # Prefer a cached tokenizer; otherwise fetch the small hf-internal-testing one.
    for repo, offline in (("hf-internal-testing/llama-tokenizer", True), ("Qwen/Qwen2.5-0.5B-Instruct", True),
                          ("hf-internal-testing/llama-tokenizer", False)):
        try:
            tok = AutoTokenizer.from_pretrained(repo, local_files_only = offline)
            break
        except Exception:
            continue
    if tok is None:
        pytest.skip("no tokenizer available (offline and not cached)")
    tok.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]})
    try:
        from unsloth_zoo.dataset_utils import get_chat_template_parts, train_on_responses_only
    except ImportError as e:
        pytest.skip(f"unsloth_zoo unavailable: {e}")
    tok.chat_template = ERNIE
    return tok, get_chat_template_parts, train_on_responses_only


# ERNIE-4.5-Thinking shape after Unsloth wraps its always-on generation prompt.
ERNIE = (
    "{{- '<|im_start|>system\n<global_setting>\nthink_mode=True\n</global_setting><|im_end|>\n\n' }}"
    "{%- set ns = namespace(lu=-1) %}"
    "{%- for m in messages %}{%- if m['role'] == 'user' %}{%- set ns.lu = loop.index0 %}{%- endif %}{%- endfor %}"
    "{%- for m in messages %}"
    "{%- if m['role'] == 'user' %}{{ '<|im_start|>user\n' + m['content'] + '<|im_end|>\n\n' }}"
    "{%- else %}"
    "{%- if loop.index0 > ns.lu and loop.last %}{{ '<|im_start|>assistant\n<think>\n\n</think>\n' }}"
    "{%- else %}{{ '<|im_start|>assistant\n' }}{%- endif %}"
    "{{ '<response>\n' + m['content'] + '\n</response>\n<|im_end|>\n\n' }}"
    "{%- endif %}{%- endfor %}"
    "{%- if add_generation_prompt %}{{ '<|im_start|>assistant\n<think>\n' }}{%- endif %}"
)

SINGLE = [{"role": "user", "content": "Q1 alpha"}, {"role": "assistant", "content": "ANSWERONE"}]
MULTI = SINGLE + [{"role": "user", "content": "Q2 bravo"}, {"role": "assistant", "content": "ANSWERTWO"}]


def _labels(tok, fn, msgs):
    text = tok.apply_chat_template(msgs, tokenize = False, add_generation_prompt = False)
    enc = tok(text, add_special_tokens = False, return_offsets_mapping = True)
    labels = fn({"input_ids": [enc["input_ids"]]})["labels"][0]
    un = {i for i, lab in enumerate(labels) if lab != -100}

    def trained(sub):
        s = text.index(sub)
        e = s + len(sub)
        return all(k in un for k, (a, b) in enumerate(enc["offset_mapping"]) if b > a and a < e and b > s)
    return trained


def test_autodetected_marker_matches_last_turn():
    tok, get_chat_template_parts, train_on_responses_only = _setup()
    ins, res = get_chat_template_parts(tok)
    last = tok.apply_chat_template(SINGLE, tokenize = False)
    assert res in last, res
    assert "<response>" not in res and "user" not in res, res
    fn = train_on_responses_only(None, instruction_part = ins, response_part = res, tokenizer = tok, return_function = True)
    tr = _labels(tok, fn, SINGLE)
    assert tr("ANSWERONE") and not tr("Q1 alpha") and not tr("think_mode")
    tr = _labels(tok, fn, MULTI)
    assert tr("ANSWERONE") and tr("ANSWERTWO")
    assert not tr("Q1 alpha") and not tr("Q2 bravo")


def test_failed_eval_split_leaves_train_split_untouched():
    from datasets import Dataset
    tok, get_chat_template_parts, train_on_responses_only = _setup()

    def ds(convs):
        texts = [tok.apply_chat_template(c, tokenize = False) for c in convs]
        return Dataset.from_dict({"input_ids": tok(texts, add_special_tokens = False).input_ids})
    trainer = types.SimpleNamespace(
        train_dataset = ds([MULTI, SINGLE]), eval_dataset = ds([SINGLE]), processing_class = tok,
        tokenizer = tok, data_collator = None,
        args = types.SimpleNamespace(max_length = 512, dataset_text_field = "text", packing = False),
    )
    before = trainer.train_dataset
    history_only = "<|im_end|>\n\n<|im_start|>assistant\n<response>\n"  # misses the last turn
    with pytest.raises(ValueError, match = "eval_dataset"):
        train_on_responses_only(trainer, instruction_part = "<|im_end|>\n\n<|im_start|>user\n", response_part = history_only)
    assert trainer.train_dataset is before and "labels" not in trainer.train_dataset.column_names
    ins, res = get_chat_template_parts(tok)
    trainer = train_on_responses_only(trainer, instruction_part = ins, response_part = res)
    assert len(trainer.train_dataset) == 2 and len(trainer.eval_dataset) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
