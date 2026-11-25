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

"""Auto-detected train_on_responses_only markers must start at the role marker, never swallow the
previous turn's end-of-message tokens. Otherwise every assistant turn's end tokens are masked and
the model never learns to stop (Inkling-2 "<|end_message|><|content_model_end_sampling|>",
gpt-oss "<|end|>"). Real templates come from cached tokenizers only (skipped when not cached);
synthetic Inkling-shaped and ChatML templates run on a small base tokenizer."""
import functools

import pytest

USER_TEXTS = ("What is 2+2?", "And 3+3?", "Then 4+4?")
ANSWERS = ("It is 4.", "It is 6.", "It is 8.")
CONVO = [m for u, a in zip(USER_TEXTS, ANSWERS)
         for m in ({"role": "user", "content": u}, {"role": "assistant", "content": a})]

# Same shape as the real Inkling-2 template: a generic <|end_message|> closes every message, the
# assistant turn adds <|content_model_end_sampling|>, add_generation_prompt is a no-op.
INKLING_LIKE = (
    "{{- '<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>' -}}"
    "{%- for m in messages -%}"
    "{%- if m.role == 'user' -%}{{- '<|message_user|><|content_text|>' + m.content + '<|end_message|>' -}}"
    "{%- elif m.role == 'system' -%}{{- '<|message_system|><|content_text|>' + m.content + '<|end_message|>' -}}"
    "{%- else -%}{{- '<|message_model|><|content_text|>' + m.content + '<|end_message|><|content_model_end_sampling|>' -}}"
    "{%- endif -%}{%- endfor -%}"
)
# unsloth/Inkling-2-Mini at ae70da4f added a generation prompt; add_generation_prompt=False renders are unchanged.
INKLING_LIKE_GENPROMPT = INKLING_LIKE + "{%- if add_generation_prompt -%}{{- '<|message_model|>' -}}{%- endif -%}"
INKLING_SPECIALS = ["<|message_system|>", "<|message_user|>", "<|message_model|>", "<|content_text|>",
                    "<|end_message|>", "<|content_model_end_sampling|>"]
# GLM-4.x: every sample opens with "[gMASK]<sop>", no bos, no assistant end token.
GLM_LIKE = (
    "[gMASK]<sop>{%- for m in messages -%}{{- '<|' + m.role + '|>' + m.content -}}{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|assistant|>' -}}{%- endif -%}"
)
# Hunyuan: bos opens every user turn.
HUNYUAN_LIKE = (
    "{%- for m in messages -%}{%- if m.role == 'user' -%}{{- bos_token + m.content + '<|extra_0|>' -}}"
    "{%- else -%}{{- m.content + eos_token -}}{%- endif -%}{%- endfor -%}"
)
# Apriel: the per-message terminator "<|end|>" is plain text, followed by eos on assistant turns.
APRIEL_LIKE = (
    "{{- bos_token + '<|system|>\\nBe brief.\\n<|end|>\\n' -}}{%- for m in messages -%}"
    "{%- if m.role == 'user' -%}{{- '<|user|>\\n' + m.content + '\\n<|end|>\\n' -}}"
    "{%- elif m.role == 'system' -%}{{- '<|system|>\\n' + m.content + '\\n<|end|>\\n' -}}"
    "{%- else -%}{{- '<|assistant|>\\n' + m.content + '\\n<|end|>\\n' + eos_token -}}{%- endif -%}{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|assistant|>\\n' -}}{%- endif -%}"
)
CHATML = (
    "{%- for m in messages -%}{{- '<|im_start|>' + m.role + '\n' + m.content + '<|im_end|>\n' -}}{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\n' -}}{%- endif -%}"
)

# family -> (repo, override template, extra special tokens, end-of-turn strings an assistant turn may end with)
FAMILIES = {
    "inkling2":      ("thinkingmachines/Inkling-2-Mini", None, None, ("<|end_message|><|content_model_end_sampling|>",)),
    "inkling2_genprompt": ("unsloth/Inkling-2-Mini@ae70da4f", None, None, ("<|end_message|><|content_model_end_sampling|>",)),
    "inkling_like_genprompt": ("hf-internal-testing/llama-tokenizer", INKLING_LIKE_GENPROMPT, INKLING_SPECIALS,
                      ("<|end_message|><|content_model_end_sampling|>",)),
    "inkling_like":  ("hf-internal-testing/llama-tokenizer", INKLING_LIKE, INKLING_SPECIALS,
                      ("<|end_message|><|content_model_end_sampling|>",)),
    "chatml":        ("hf-internal-testing/llama-tokenizer", CHATML, ["<|im_start|>", "<|im_end|>"], ("<|im_end|>",)),
    "glm_like":      ("hf-internal-testing/llama-tokenizer", GLM_LIKE,
                      ["[gMASK]", "<sop>", "<|system|>", "<|user|>", "<|assistant|>"], ("",)),
    "hunyuan_like":  ("hf-internal-testing/llama-tokenizer", HUNYUAN_LIKE, ["<|extra_0|>"], ("</s>",)),
    "apriel_like":   ("hf-internal-testing/llama-tokenizer", APRIEL_LIKE, ["<|system|>", "<|user|>", "<|assistant|>"],
                      ("<|end|>\n</s>",)),
    "glm47":         ("unsloth/GLM-4.7-Flash", None, None, ("",)),
    "qwen3":         ("unsloth/Qwen3-0.6B", None, None, ("<|im_end|>",)),
    "llama3":        ("unsloth/Llama-3.2-1B-Instruct", None, None, ("<|eot_id|>",)),
    "gemma3":        ("unsloth/gemma-3-270m-it", None, None, ("<end_of_turn>",)),
    "phi3":          ("microsoft/Phi-3-mini-4k-instruct", None, None, ("<|end|>", "<|end|><|endoftext|>")),
    "gpt_oss":       ("unsloth/gpt-oss-20b", None, None, ("<|end|>", "<|return|>")),
    "mistral":       ("unsloth/Mistral-7B-Instruct-v0.3", None, None, ("</s>",)),
}


@functools.lru_cache(maxsize = None)
def _base_tokenizer(repo):
    from transformers import AutoTokenizer
    repo, _, revision = repo.partition("@")
    try:
        return AutoTokenizer.from_pretrained(repo, revision = revision or None, local_files_only = True)
    except Exception:
        return None


def _tokenizer(family):
    repo, template, extra, _ = FAMILIES[family]
    tok = _base_tokenizer(repo)
    if tok is None:
        pytest.skip(f"{repo} tokenizer not cached offline")
    if template is not None:
        import copy
        tok = copy.deepcopy(tok)
        if extra: tok.add_special_tokens({"additional_special_tokens": extra})
        tok.chat_template = template
    elif not getattr(tok, "chat_template", None):
        pytest.skip(f"{repo} has no chat template")
    return tok


def _kept_segments(tok, labels_fn):
    text = tok.apply_chat_template(CONVO, tokenize = False)
    ids = tok(text, add_special_tokens = False)["input_ids"]
    labels = labels_fn({"input_ids": [ids]})["labels"][0]
    segments, run = [], []
    for i, l in zip(ids, labels):
        if l != -100:
            run.append(i)
        elif run:
            segments.append(run); run = []
    if run: segments.append(run)
    return [tok.decode(s) for s in segments], tok.decode([i for i, l in zip(ids, labels) if l != -100])


@pytest.mark.parametrize("family", list(FAMILIES))
def test_autodetect_keeps_assistant_end_of_turn(family):
    try:
        from unsloth_zoo.dataset_utils import train_on_responses_only
    except ImportError as e:
        pytest.skip(f"unsloth_zoo unavailable: {e}")
    tok = _tokenizer(family)
    ends = FAMILIES[family][3]
    fn = train_on_responses_only(None, tokenizer = tok, return_function = True)
    segments, kept = _kept_segments(tok, fn)

    # (b) no user / system text is trained on
    for t in USER_TEXTS + ("Thinking effort", "user", "system"):
        assert t not in kept, f"{family}: kept text leaks {t!r}: {kept!r}"
    # Each answer is kept exactly once, inside its own segment
    for a in ANSWERS:
        assert kept.count(a) == 1, f"{family}: {a!r} not kept once: {kept!r}"
    # (a) every assistant turn's kept text ends with the template's end-of-turn token(s)
    for a in ANSWERS:
        seg = next(s for s in segments if a in s)
        tail = seg[seg.index(a) + len(a):].strip()
        assert tail in ends, f"{family}: turn {a!r} kept tail {tail!r}, want one of {ends}; kept {kept!r}"


NEXT_SYS, NEXT_USER = "Ibex directive golf", "Ocelot question charlie"
NEXT = [{"role": "system", "content": NEXT_SYS}, {"role": "user", "content": NEXT_USER},
        {"role": "assistant", "content": "Macaw answer foxtrot"}]


@pytest.mark.parametrize("family", list(FAMILIES))
def test_autodetect_packed_row_stops_at_next_sample(family):
    """Two samples packed into one row: the first sample's last answer must not run into the next
    sample's system prompt or user turn (GLM-4.x has no assistant end token, only "[gMASK]<sop>")."""
    try:
        from unsloth_zoo.dataset_utils import train_on_responses_only
    except ImportError as e:
        pytest.skip(f"unsloth_zoo unavailable: {e}")
    tok = _tokenizer(family)
    try:
        tail = tok.apply_chat_template(NEXT, tokenize = False)
    except Exception:
        pytest.skip("template rejects a system message")
    text = tok.apply_chat_template(CONVO, tokenize = False) + tail
    ids = tok(text, add_special_tokens = False)["input_ids"]
    fn = train_on_responses_only(None, tokenizer = tok, return_function = True)
    labels = fn({"input_ids": [ids]})["labels"][0]
    kept = tok.decode([i for i, l in zip(ids, labels) if l != -100])
    for t in (NEXT_SYS, NEXT_USER) + USER_TEXTS:
        if t in text:
            assert t not in kept, f"{family}: packed row trains {t!r}: {kept!r}"
    assert "Macaw answer foxtrot" in kept, f"{family}: next sample's answer not trained: {kept!r}"
