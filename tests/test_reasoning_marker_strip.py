"""Must not bake reasoning scaffolding into the marker (it masks the final trained turn); shortened to bare header only
when generation-only, never unmasking user content; skips (not fails) if no offline tokenizer or unsloth_zoo is
unavailable, so pytest never reaches the network or breaks collection."""
import pytest

# unsloth_zoo/__init__.py raises ImportError without the separate `unsloth` package, so it is imported lazily in
# _setup(), not at module scope, to allow the skip.


def _tokenizer():
    from transformers import AutoTokenizer
    # local_files_only keeps this hermetic: an uncached tokenizer raises at once instead of downloading or timing out,
    # so the loop falls through to the next or a skip.
    for repo in ("hf-internal-testing/llama-tokenizer", "Qwen/Qwen2.5-0.5B-Instruct"):
        try:
            tok = AutoTokenizer.from_pretrained(repo, local_files_only=True)
        except Exception:
            continue
        tok.add_special_tokens({"additional_special_tokens": [
            "<|im_start|>", "<|im_end|>", "<|user|>", "<|assistant|>", "<think>", "</think>"]})
        return tok
    return None


def _setup():
    """Load an offline tokenizer and the zoo helpers; skips rather than fails when the tokenizer cache or unsloth_zoo (needs `unsloth`) is unavailable."""
    tok = _tokenizer()
    if tok is None:
        pytest.skip("no tokenizer cached offline")
    try:
        from unsloth_zoo.dataset_utils import get_chat_template_parts, train_on_responses_only
    except ImportError as e:
        pytest.skip(f"unsloth_zoo unavailable: {e}")
    return tok, get_chat_template_parts, train_on_responses_only


# History turns carry an empty <think></think> while the final turn keeps real reasoning; the generation prompt also
# emits the empty tag (Qwen3-Thinking / Nemotron Nano shape).
PAIRED = (
    "{%- for m in messages %}"
    "{%- if m['role'] == 'user' %}{{ '<|im_start|>user\n' + m['content'] + '<|im_end|>\n' }}"
    "{%- else %}"
    "{%- if loop.last %}{{ '<|im_start|>assistant\n' + m['content'] + '<|im_end|>\n' }}"
    "{%- else %}{{ '<|im_start|>assistant\n<think></think>' + m['content'].split('</think>')[-1] + '<|im_end|>\n' }}"
    "{%- endif %}{%- endif %}{%- endfor %}"
    "{%- if add_generation_prompt %}{{ '<|im_start|>assistant\n<think></think>' }}{%- endif %}"
)

# GLM-4.x shape: empty reasoning renders a lone closing tag <|assistant|></think>, while a real reasoning turn renders <|assistant|><think>reason</think>answer.
GLM = (
    "{%- set ns = namespace(lu=-1) %}"
    "{%- for m in messages %}{%- if m['role'] == 'user' %}{%- set ns.lu = loop.index0 %}{%- endif %}{%- endfor %}"
    "{%- for m in messages %}"
    "{%- if m['role'] == 'user' %}{{ '<|user|>' + m['content'] }}"
    "{%- else %}{{ '<|assistant|>' }}"
    "{%- set c = m['content'] %}{%- set rc = '' %}"
    "{%- if '</think>' in c %}{%- set rc = c.split('</think>')[0].split('<think>')[-1] %}{%- set c = c.split('</think>')[-1] %}{%- endif %}"
    "{%- if loop.index0 > ns.lu and rc %}{{ '<think>' + rc + '</think>' }}{%- else %}{{ '</think>' }}{%- endif %}"
    "{{ c }}{%- endif %}{%- endfor %}"
    "{%- if add_generation_prompt %}{{ '<|assistant|>' }}{%- endif %}"
)

# A template that always emits the empty tag, even on the trained turn: the strip is verified rather than assumed, so the tag must be kept.
ALWAYS = (
    "{%- for m in messages %}"
    "{%- if m['role'] == 'user' %}{{ '<|im_start|>user\n' + m['content'] + '<|im_end|>\n' }}"
    "{%- else %}{{ '<|im_start|>assistant\n<think></think>' + m['content'].split('</think>')[-1] + '<|im_end|>\n' }}"
    "{%- endif %}{%- endfor %}"
    "{%- if add_generation_prompt %}{{ '<|im_start|>assistant\n<think></think>' }}{%- endif %}"
)

REASON_MSGS = [
    {"role": "user", "content": "Q1 alpha"},
    {"role": "assistant", "content": "<think>reason one</think>ANSWERONE"},
    {"role": "user", "content": "Q2 bravo"},
    {"role": "assistant", "content": "<think>REASONTWO</think>ANSWERTWO"},
]
PLAIN_MSGS = [
    {"role": "user", "content": "Q1 alpha"}, {"role": "assistant", "content": "ANSWERONE"},
    {"role": "user", "content": "Q2 bravo"}, {"role": "assistant", "content": "ANSWERTWO"},
]


def _trained(tok, ins, res, msgs, tor):
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    fn = tor(None, instruction_part=ins, response_part=res, tokenizer=tok, return_function=True)
    labels = fn({"input_ids": [ids]})["labels"][0]
    un = set(i for i in range(len(ids)) if labels[i] != -100)

    def is_trained(sub):
        i = text.index(sub); s, e = i, i + len(sub)
        return all(k in un for k in [j for j, (a, b) in enumerate(offs) if b > a and a < e and b > s])
    return is_trained


def test_paired_empty_think_stripped_to_header():
    tok, get_chat_template_parts, train_on_responses_only = _setup()
    tok.chat_template = PAIRED
    ins, res = get_chat_template_parts(tok)
    assert "<think></think>" not in res, res
    assert res.strip().endswith("assistant"), res
    tr = _trained(tok, ins, res, REASON_MSGS, train_on_responses_only)
    assert tr("REASONTWO") and tr("ANSWERTWO") and tr("ANSWERONE")
    assert not tr("Q1 alpha") and not tr("Q2 bravo")


def test_lone_close_think_stripped_to_header():
    tok, get_chat_template_parts, train_on_responses_only = _setup()
    tok.chat_template = GLM
    ins, res = get_chat_template_parts(tok)
    assert res == "<|assistant|>", res
    tr = _trained(tok, ins, res, REASON_MSGS, train_on_responses_only)
    assert tr("REASONTWO") and tr("ANSWERTWO") and tr("ANSWERONE")
    assert not tr("Q1 alpha") and not tr("Q2 bravo")


def test_always_empty_think_is_kept():
    tok, get_chat_template_parts, train_on_responses_only = _setup()
    tok.chat_template = ALWAYS
    ins, res = get_chat_template_parts(tok)
    assert "<think></think>" in res, res
    tr = _trained(tok, ins, res, PLAIN_MSGS, train_on_responses_only)
    assert tr("ANSWERONE") and tr("ANSWERTWO")
    assert not tr("Q1 alpha") and not tr("Q2 bravo")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
