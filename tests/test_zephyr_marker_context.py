"""Zephyr role markers are not special tokens, so get_chat_template_parts needs context-aware auto-detect: it probes
leading-newline candidates and prefers ones matching at least twice in a 3-turn probe; skips when no offline
tokenizer or unsloth_zoo is available."""
import pytest

# unsloth_zoo is imported lazily in _setup(): its __init__ raises ImportError without the separate unsloth package, which would break collection.


def _tokenizer():
    import json
    try:
        from transformers import AutoTokenizer
        from tokenizers import Tokenizer
    except ImportError:
        return None  # dependency-light run: skip instead of erroring
    # Uses a SentencePiece tokenizer (Llama/Mistral) so "<" tokenizes context-dependently; role tags must stay ordinary text, not added special tokens.
    try:
        tok = AutoTokenizer.from_pretrained(
            "hf-internal-testing/llama-tokenizer", local_files_only=True
        )
    except Exception:
        return None
    # Zephyr parses "</s>" mid-text as the eos token (normalized=False); the fixture tokenizer ships normalized=True and
    # would split it, so the flag is flipped to match Zephyr.
    d = json.loads(tok.backend_tokenizer.to_str())
    for t in d["added_tokens"]:
        if t["content"] == "</s>":
            t["normalized"] = False
    tok._tokenizer = Tokenizer.from_str(json.dumps(d))
    return tok


def _setup():
    tok = _tokenizer()
    if tok is None:
        pytest.skip("no SentencePiece tokenizer cached offline")
    try:
        from unsloth_zoo.dataset_utils import get_chat_template_parts, train_on_responses_only
    except ImportError as e:
        pytest.skip(f"unsloth_zoo unavailable: {e}")
    return tok, get_chat_template_parts, train_on_responses_only


# HuggingFaceH4/zephyr-7b-beta chat template, verbatim.
ZEPHYR = (
    "{% for message in messages %}\n"
    "{% if message['role'] == 'user' %}\n"
    "{{ '<|user|>\n' + message['content'] + eos_token }}\n"
    "{% elif message['role'] == 'system' %}\n"
    "{{ '<|system|>\n' + message['content'] + eos_token }}\n"
    "{% elif message['role'] == 'assistant' %}\n"
    "{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n"
    "{% endif %}\n"
    "{% if loop.last and add_generation_prompt %}\n"
    "{{ '<|assistant|>' }}\n"
    "{% endif %}\n"
    "{% endfor %}"
)

MSGS = [
    {"role": "system", "content": "SYSPROMPT terse"},
    {"role": "user", "content": "QONE alpha"},
    {"role": "assistant", "content": "ANSWERONE"},
    {"role": "user", "content": "QTWO bravo"},
    {"role": "assistant", "content": "ANSWERTWO"},
]


def _labels(tok, ins, res, tor):
    text = tok.apply_chat_template(MSGS, tokenize=False, add_generation_prompt=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    fn = tor(None, instruction_part=ins, response_part=res, tokenizer=tok, return_function=True)
    labels = fn({"input_ids": [ids]})["labels"][0]
    un = set(i for i in range(len(ids)) if labels[i] != -100)

    def is_trained(sub):
        i = text.index(sub); s, e = i, i + len(sub)
        return all(k in un for k in [j for j, (a, b) in enumerate(offs) if b > a and a < e and b > s])
    return ids, labels, is_trained


def test_zephyr_autodetects_context_dependent_markers():
    tok, get_chat_template_parts, train_on_responses_only = _setup()
    tok.chat_template = ZEPHYR
    ins, res = get_chat_template_parts(tok)  # raised ValueError before the fix
    # The leading newline anchors the mid-text tokenization ("<" not "▁<")
    assert ins == "\n<|user|>\n", ins
    assert res == "\n<|assistant|>\n", res

    ids, labels, tr = _labels(tok, ins, res, train_on_responses_only)
    assert tr("ANSWERONE") and tr("ANSWERTWO")
    assert not tr("SYSPROMPT terse")
    assert not tr("QONE alpha") and not tr("QTWO bravo")
    # Both assistant eos terminators are trained; the FINAL eos must never be -100 or the model never learns to stop.
    eos_id = tok.eos_token_id
    eos_positions = [i for i, t in enumerate(ids) if t == eos_id]
    assert labels[eos_positions[-1]] != -100
    assert labels[eos_positions[-3]] != -100  # first assistant turn's eos (-2 is user QTWO's)


def test_zephyr_first_user_turn_variant_not_picked():
    # The '<|user|>' plus newline marker matches ONLY at text start; a single text-start match must not beat 2+ mid-text
    # matches or later user turns get trained.
    tok, get_chat_template_parts, train_on_responses_only = _setup()
    tok.chat_template = ZEPHYR
    ins, _ = get_chat_template_parts(tok)
    assert ins != "<|user|>\n", "single text-start match must not win over 2+ mid-text matches"


def test_special_token_template_unchanged():
    # A ChatML template whose markers are special tokens still picks the plain first candidate (3/3 matches); the new preference must not alter it.
    tok, get_chat_template_parts, _ = _setup()
    tok.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]})
    tok.chat_template = (
        "{%- for m in messages %}"
        "{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>' + '\n' }}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{%- endif %}"
    )
    ins, res = get_chat_template_parts(tok)
    assert ins == "<|im_start|>user\n", ins
    assert res == "<|im_start|>assistant\n", res


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
