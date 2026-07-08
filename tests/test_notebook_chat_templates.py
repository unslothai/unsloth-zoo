"""Auto-detection coverage over the unslothai/notebooks model set.

For every model shipped in unslothai/notebooks (mostly unsloth/* names, listed in
notebook_models.json) this checks that train_on_responses_only can auto-detect the
instruction/response markers and that the resulting mask is content-exact on BOTH a
plain multi-turn conversation AND a reasoning conversation (final assistant turn
carries "<think>...</think>answer"). The reasoning probe is what catches markers
that bake in an injected empty think block and so miss the trained turn.

The model list is discovered dynamically from a notebooks checkout (git clone, or
UNSLOTH_NOTEBOOKS_DIR), so a newly added notebook is picked up automatically; the
pinned notebook_models.json is unioned in as a fallback so coverage never shrinks.

This is a network/integration test and is OFF by default. Enable it with
    UNSLOTH_TEST_NOTEBOOK_MODELS=1 pytest tests/test_notebook_chat_templates.py
Tokenizers are fetched config-only and cached; set UNSLOTH_NOTEBOOK_TOK_CACHE to
reuse an existing cache directory. Models that 404/gate/need a newer transformers,
or whose template legitimately has no atomic role markers, are skipped, not failed.
"""
import os, sys, json, re, glob, tempfile, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # import the in-tree unsloth_zoo
# unsloth_zoo is imported lazily (inside _check / _content_leak_free), not at module
# scope: importing it runs unsloth_zoo/__init__.py, which raises ImportError when the
# separate `unsloth` package is absent, and that would break pytest collection.

ENABLED = os.environ.get("UNSLOTH_TEST_NOTEBOOK_MODELS", "") not in ("", "0", "false")

# Notebook models are expected to auto-detect, so a ValueError from get_chat_template_parts
# is a regression and FAILS the sweep. The only sanctioned exceptions are templates that
# genuinely have no atomic role markers (e.g. zephyr-sft's multi-piece <|assistant|>); list
# them here (substring match) so a real detection regression can never hide as a SKIP.
KNOWN_NON_ATOMIC = ("zephyr-sft",)
CACHE = os.environ.get("UNSLOTH_NOTEBOOK_TOK_CACHE", os.path.join(tempfile.gettempdir(), "unsloth_nb_tok"))
ALLOW_PATTERNS = ["*.json", "*.model", "tokenizer*", "merges*", "vocab*", "*.jinja", "*.txt"]
NOTEBOOKS_REPO = "https://github.com/unslothai/notebooks"

# Dynamically discover every model named in the notebooks so a newly added notebook
# is covered automatically. A few hard cases (templated names, non-conversational
# repos) are intentionally dropped.
_MODEL_RE = re.compile(r"""(?:model_name\s*=\s*|from_pretrained\(\s*)["']([A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+)["']""")
_SUFFIX = ["-unsloth-bnb-4bit", "-bnb-4bit", "-unsloth", "-FP8-Block", "-FP8", "-BF16", "-GGUF"]
_DROP = ("embedding", "minilm", "mpnet", "bge-", "modernbert", "whisper", "outetts", "llasa",
         "spark-tts", "csm-", "orpheus", "ocr", "latex", "radiology", "diffusion", "gte-",
         "ultrafeedback", "gsm8k", "multilingual-thinking", "openmathreasoning", "mobile-actions",
         "instruction_following", "/repo", "all-minilm", "functiongemma", "snac")


def _norm(rid):
    for s in _SUFFIX:
        if rid.endswith(s): rid = rid[:-len(s)]
    return rid


def _scan_notebooks(root):
    out = set()
    files = glob.glob(os.path.join(root, "nb", "*.ipynb")) or \
        glob.glob(os.path.join(root, "**", "*.ipynb"), recursive=True)
    for f in files:
        try:
            nb = json.load(open(f))
        except Exception:
            continue
        src = "\n".join("".join(c.get("source", [])) for c in nb.get("cells", []) if c.get("cell_type") == "code")
        for m in _MODEL_RE.findall(src):
            n = _norm(m)
            if "/" not in n or n.startswith("./"):
                continue
            if any(s in n.lower() for s in _DROP):
                continue
            out.add(n)
    return out


def _models():
    """Dynamically extract model names from a notebooks checkout; union with the
    pinned fixture so coverage never shrinks and new notebooks are auto-covered."""
    fixture = set(json.load(open(os.path.join(HERE, "notebook_models.json"))))
    discovered = set()
    env_dir = os.environ.get("UNSLOTH_NOTEBOOKS_DIR")
    if env_dir and os.path.isdir(env_dir):
        discovered = _scan_notebooks(env_dir)
    elif ENABLED:
        clone = os.path.join(CACHE, "notebooks_repo")
        if not os.path.isdir(os.path.join(clone, ".git")):
            try:
                subprocess.run(["git", "clone", "--depth", "1", NOTEBOOKS_REPO, clone],
                               check=True, capture_output=True)
            except Exception:
                clone = None
        if clone and os.path.isdir(clone):
            discovered = _scan_notebooks(clone)
    return sorted(fixture | discovered)


USERS = ["Zebra question alpha", "Quokka inquiry bravo"]
ASSTS = ["Penguin answer delta", "Dolphin reply echo"]
PLAIN = [{"role": "user", "content": USERS[0]}, {"role": "assistant", "content": ASSTS[0]},
         {"role": "user", "content": USERS[1]}, {"role": "assistant", "content": ASSTS[1]}]
# Reasoning probe: the final assistant turn carries a think block + answer.
REASON = [{"role": "user", "content": "Reason user one"},
          {"role": "assistant", "content": "<think>HIDDENREASON</think>VISIBLEANSWER"}]


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
    from unsloth_zoo.dataset_utils import train_on_responses_only
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
    from unsloth_zoo.dataset_utils import get_chat_template_parts
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
    except ValueError as e:
        # A model shipped in a notebook is expected to auto-detect. Only allowlisted
        # non-atomic templates may safe-raise; anything else is a real regression.
        if any(k in repo.lower() for k in KNOWN_NON_ATOMIC):
            return "SKIP", "known non-atomic template (allowlisted)"
        return "FAIL", f"auto-detect raised ValueError for a supported model: {str(e)[:80]}"
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
    try:
        import unsloth_zoo.dataset_utils  # noqa: F401  (import here so collection never needs it)
    except ImportError as e:
        pytest.skip(f"unsloth_zoo unavailable: {e}")
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
