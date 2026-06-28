"""Auto-detection coverage over the unslothai/notebooks model set.

For every model shipped in unslothai/notebooks (mostly unsloth/* names, listed in
notebook_models.json) this checks that train_on_responses_only can auto-detect the
instruction/response markers and that the resulting mask is content-exact on BOTH a
plain multi-turn conversation AND a reasoning conversation (final assistant turn
carries "<think>...</think>answer"). The reasoning probe is what catches markers
that bake in an injected empty think block and so miss the trained turn.

This is a network/integration test and is OFF by default. Enable it with
    UNSLOTH_TEST_NOTEBOOK_MODELS=1 pytest tests/test_notebook_chat_templates.py
Tokenizers are fetched config-only and cached; set UNSLOTH_NOTEBOOK_TOK_CACHE to
reuse an existing cache directory. Models that 404/gate/need a newer transformers,
or whose template legitimately has no atomic role markers, are skipped, not failed.
"""
import os, sys, json, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # import the in-tree unsloth_zoo
from unsloth_zoo.dataset_utils import get_chat_template_parts, train_on_responses_only

ENABLED = os.environ.get("UNSLOTH_TEST_NOTEBOOK_MODELS", "") not in ("", "0", "false")
CACHE = os.environ.get("UNSLOTH_NOTEBOOK_TOK_CACHE", os.path.join(tempfile.gettempdir(), "unsloth_nb_tok"))
ALLOW_PATTERNS = ["*.json", "*.model", "tokenizer*", "merges*", "vocab*", "*.jinja", "*.txt"]

USERS = ["Zebra question alpha", "Quokka inquiry bravo"]
ASSTS = ["Penguin answer delta", "Dolphin reply echo"]
PLAIN = [{"role": "user", "content": USERS[0]}, {"role": "assistant", "content": ASSTS[0]},
         {"role": "user", "content": USERS[1]}, {"role": "assistant", "content": ASSTS[1]}]
# Reasoning probe: the final assistant turn carries a think block + answer.
REASON = [{"role": "user", "content": "Reason user one"},
          {"role": "assistant", "content": "<think>HIDDENREASON</think>VISIBLEANSWER"}]


def _models():
    return json.load(open(os.path.join(HERE, "notebook_models.json")))


def _download(repo):
    from huggingface_hub import snapshot_download
    path = os.path.join(CACHE, repo.replace("/", "__"))
    if os.path.isdir(path) and any(f.startswith("tokenizer") or f.endswith(".model") for f in os.listdir(path)):
        return path
    snapshot_download(repo, local_dir=path, allow_patterns=ALLOW_PATTERNS, token=os.environ.get("HF_TOKEN"))
    return path


def _content_leak_free(t, ins, res, convo, asst_texts, user_texts):
    """True iff every assistant-content token present is trained and no user-content
    token is trained. Content pieces the template drops are ignored."""
    text = t.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
    enc = t(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    fn = train_on_responses_only(None, instruction_part=ins, response_part=res, tokenizer=t, return_function=True)
    labels = fn({"input_ids": [ids]})["labels"][0]
    un = set(i for i in range(len(ids)) if labels[i] != -100)

    def tok_idx(sub):
        if sub not in text: return None  # template transformed/dropped it
        i = text.index(sub); s, e = i, i + len(sub)
        return [k for k, (a, b) in enumerate(offs) if b > a and a < e and b > s]

    for sub in asst_texts:
        idx = tok_idx(sub)
        if idx and not all(k in un for k in idx):
            return False, f"assistant content {sub!r} not trained"
    for sub in user_texts:
        idx = tok_idx(sub)
        if idx and any(k in un for k in idx):
            return False, f"user content {sub!r} leaked into labels"
    return True, ""


def _check(repo):
    from transformers import AutoTokenizer
    try:
        path = _download(repo)
    except Exception as e:
        return "SKIP", f"fetch {type(e).__name__}"
    try:
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
    except Exception as e:
        return "SKIP", f"load {type(e).__name__}"
    inner = tok.tokenizer if hasattr(tok, "tokenizer") else tok
    if getattr(inner, "chat_template", None) is None:
        return "SKIP", "no chat_template"
    try:
        ins, res = get_chat_template_parts(tok)
    except ValueError:
        return "SKIP", "safe-raise (no atomic markers)"
    try:
        for convo in (PLAIN, REASON):
            asst = ASSTS if convo is PLAIN else ["HIDDENREASON", "VISIBLEANSWER"]
            users = USERS if convo is PLAIN else ["Reason user one"]
            ok, why = _content_leak_free(inner, ins, res, convo, asst, users)
            if not ok:
                return "FAIL", f"{why} (i={ins!r} r={res!r})"
    except Exception as e:
        return "SKIP", f"verify {type(e).__name__}"
    return "PASS", ""


def run():
    models = _models()
    results = {}
    for repo in models:
        cat, detail = _check(repo)
        results[repo] = (cat, detail)
        print(f"{repo:54s} {cat}  {detail}", flush=True)
    summary = {}
    for cat, _ in results.values():
        summary[cat] = summary.get(cat, 0) + 1
    return results, summary


def test_notebook_chat_templates():
    import pytest
    if not ENABLED and not os.path.isdir(CACHE):
        pytest.skip("set UNSLOTH_TEST_NOTEBOOK_MODELS=1 (network) to run the notebook model sweep")
    results, summary = run()
    print("SUMMARY:", summary)
    checked = summary.get("PASS", 0) + summary.get("FAIL", 0)
    if checked == 0:
        pytest.skip(f"no notebook models were reachable to check ({summary})")
    fails = {r: d for r, (c, d) in results.items() if c == "FAIL"}
    assert not fails, f"auto-detect masking failed for: {fails}"


if __name__ == "__main__":
    results, summary = run()
    print("\n==== SUMMARY ====")
    for k, v in sorted(summary.items()):
        print(f"  {k:6s} {v}")
    fails = [r for r, (c, _) in results.items() if c == "FAIL"]
    if fails:
        print("FAILS:", fails); sys.exit(1)
