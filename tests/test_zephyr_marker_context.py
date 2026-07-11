"""Auto-detect must handle role markers whose tokenization depends on context (Zephyr).

Zephyr's role tags ("<|user|>", "<|assistant|>") are NOT special tokens: the underlying
SentencePiece tokenizer splits them into pieces, and the leading "<" becomes "▁<" at text
start (or standalone) but a bare "<" mid-text after "</s>\n". Two failures followed:

  response_part    - "<|assistant|>\n" tokenized standalone never matches the rendered
                     conversation, so get_chat_template_parts raised
                     "Could not reliably auto-detect response_part".
  instruction_part - "<|user|>\n" DID validate, but only via its text-start occurrence
                     (the very first user turn). At masking time that single-context match
                     misses every later user turn, so the span after an assistant header
                     would run to the end of the sample and train on user content.

get_chat_template_parts now also probes a leading-newline candidate ("\n<|assistant|>\n")
and prefers candidates whose token core matches at least twice in the 3-turn probe (i.e.
from the second turn on), falling back to the previous single-match acceptance for the
original candidates. Special-token templates match their first candidate 3x and are
unaffected.

Self-contained and hermetic: builds the Zephyr template on a tiny SentencePiece tokenizer
that is already cached locally, WITHOUT registering the role tags as special tokens (that
is the point). Skips (not fails) when no tokenizer can be constructed offline or when the
`unsloth`/`unsloth_zoo` package is not importable.
"""
import pytest

# unsloth_zoo is imported lazily in _setup(), not module scope: its __init__ raises
# ImportError when the separate `unsloth` package is absent, which would break pytest
# collection before the skip.


def _tokenizer():
    import json
    try:
        from transformers import AutoTokenizer
        from tokenizers import Tokenizer
    except ImportError:
        return None  # dependency-light run: skip instead of erroring
    # SentencePiece tokenizer (Llama/Mistral) so "<" tokenizes context-dependently;
    # role tags must stay ordinary text, NOT added special tokens.
    try:
        tok = AutoTokenizer.from_pretrained(
            "hf-internal-testing/llama-tokenizer", local_files_only=True
        )
    except Exception:
        return None
    # Zephyr parses "</s>" mid-text as the eos token (AddedToken normalized=False); the
    # testing tokenizer ships normalized=True and would split it into "</ s >". Flip the
    # flag so the fixture tokenizes like Zephyr (eos then prefix space "▁<0x0A><|...").
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
    # Both assistant eos terminators trained; the FINAL eos must never be -100 or the
    # model never learns to stop.
    eos_id = tok.eos_token_id
    eos_positions = [i for i, t in enumerate(ids) if t == eos_id]
    assert labels[eos_positions[-1]] != -100
    assert labels[eos_positions[-3]] != -100  # first assistant turn's eos (-2 is user QTWO's)


def test_zephyr_first_user_turn_variant_not_picked():
    # "<|user|>\n" matches ONLY at position 0 (text start); accepting that single-context
    # variant would leave later user turns trained.
    tok, get_chat_template_parts, train_on_responses_only = _setup()
    tok.chat_template = ZEPHYR
    ins, _ = get_chat_template_parts(tok)
    assert ins != "<|user|>\n", "single text-start match must not win over 2+ mid-text matches"


def test_special_token_template_unchanged():
    # ChatML template whose markers ARE special tokens still detects the plain first
    # candidate (3/3 matches): the new preference cannot alter it.
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
