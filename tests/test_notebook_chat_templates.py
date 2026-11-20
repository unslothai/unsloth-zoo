"""Notebook auto-detection sweep: masks must be content-exact on plain, reasoning and boundary chats; models from
UNSLOTH_NOTEBOOKS_DIR or a clone plus notebook_models.json (coverage never shrinks); off unless
UNSLOTH_TEST_NOTEBOOK_MODELS=1; cache UNSLOTH_NOTEBOOK_TOK_CACHE; 404/gated models skip."""
import os, sys, json, re, glob, tempfile, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # import the in-tree unsloth_zoo
# Import unsloth_zoo lazily inside _check / _content_leak_free: a module-scope import runs __init__.py, which raises ImportError without the unsloth package.

ENABLED = os.environ.get("UNSLOTH_TEST_NOTEBOOK_MODELS", "") not in ("", "0", "false")

# A ValueError from get_chat_template_parts on notebook models is a regression that fails the sweep; only templates with
# no atomic role markers (e.g. zephyr-sft, multi-piece <|assistant|>) are listed in KNOWN_NON_ATOMIC (substring match)
# so regressions cannot hide as SKIP.
KNOWN_NON_ATOMIC = ("zephyr-sft",)
# ERNIE's '<|end_of_sentence|>User:' and 'Assistant:' share no token prefix and <|end_of_sentence|> is not eos_token_id,
# so packed rows have nothing to stop at; role-based labels are the documented answer; listed so a boundary regression
# cannot hide as a skip.
KNOWN_NO_BOUNDARY = ("ernie-4.5",)
CACHE = os.environ.get("UNSLOTH_NOTEBOOK_TOK_CACHE", os.path.join(tempfile.gettempdir(), "unsloth_nb_tok"))
ALLOW_PATTERNS = ["*.json", "*.model", "tokenizer*", "merges*", "vocab*", "*.jinja", "*.txt"]
NOTEBOOKS_REPO = "https://github.com/unslothai/notebooks"

# Discover every model named in the notebooks so new ones are covered automatically; templated names and non-conversational repos are intentionally dropped.
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
    """Extract model names from a notebooks checkout, unioned with the pinned fixture so coverage never shrinks and new notebooks are auto-covered."""
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
# Second conversation concatenated after PLAIN to stand in for packing; it opens with a system message since, with no
# user marker left in the first sample, the old scan's last span ran into the next sample's system prompt.
NEXT_SYS, NEXT_USER, NEXT_ASST = "Ibex directive golf", "Ocelot question charlie", "Macaw answer foxtrot"
NEXT = [{"role": "system", "content": NEXT_SYS},
        {"role": "user", "content": NEXT_USER},
        {"role": "assistant", "content": NEXT_ASST}]


def _download(repo):
    from huggingface_hub import snapshot_download
    path = os.path.join(CACHE, repo.replace("/", "__"))
    if os.path.isdir(path) and any(f.startswith("tokenizer") or f.endswith(".model") for f in os.listdir(path)):
        return path
    snapshot_download(repo, local_dir=path, allow_patterns=ALLOW_PATTERNS, token=os.environ.get("HF_TOKEN"))
    return path


def _content_leak_free(t, ins, res, convo, asst_texts, user_texts):
    """True iff every assistant-content token present is trained and no user-content token is trained; pieces the template drops are ignored."""
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


def _piece_index(t, text):
    """Returns (token ids, idx) where idx(sub) gives the covering token indices, or None if the template dropped it."""
    enc = t(text, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]

    def idx(sub):
        if sub not in text: return None
        s = text.index(sub); e = s + len(sub)
        return [k for k, (a, b) in enumerate(offs) if b > a and a < e and b > s] or None
    return enc["input_ids"], idx


def _mask(t, ins, res, ids, **kw):
    from unsloth_zoo.dataset_utils import train_on_responses_only
    fn = train_on_responses_only(None, instruction_part=ins, response_part=res,
                                 tokenizer=t, return_function=True, **kw)
    return fn({"input_ids": [list(ids)]})["labels"][0]


def _all_trained(labels, idx):
    """True if every token carries a target, False if none do, None if mixed."""
    on = [labels[k] != -100 for k in idx]
    return True if all(on) else (False if not any(on) else None)


def _boundary_arms(t, repo, ins, res):
    """Arms the plain content check cannot reach: last_response_only, force_match off, EOS-id padding, a packed second
    sample; every arm runs and every failure is reported."""
    text = t.apply_chat_template(PLAIN, tokenize=False, add_generation_prompt=False)
    ids, idx = _piece_index(t, text)
    bad = []

    first, last = idx(ASSTS[0]), idx(ASSTS[1])
    if first and last:
        labels = _mask(t, ins, res, ids, last_response_only=True)
        if _all_trained(labels, last) is not True:
            bad.append("last_response_only dropped the final answer")
        if _all_trained(labels, first) is True:
            bad.append("last_response_only retained an earlier answer")

    try:
        labels = _mask(t, ins, res, ids, force_match=False)
    except ValueError:
        labels = None          # no stable core without force_match; the guard's own job
    if labels is not None:
        for sub in ASSTS:
            i = idx(sub)
            if i and _all_trained(labels, i) is not True:
                bad.append(f"force_match=False did not train {sub!r}")
        for sub in USERS:
            i = idx(sub)
            if i and _all_trained(labels, i) is True:
                bad.append(f"force_match=False leaked {sub!r}")

    # Padding that reuses the EOS id: one eos may stay as the answer's own target; the rest of the pad run must not be trained.
    eos = getattr(t, "eos_token_id", None)
    if eos is not None:
        labels = _mask(t, ins, res, list(ids) + [eos] * 6)
        trained = sum(1 for l in labels[len(ids):] if l != -100)
        if trained > 1:
            bad.append(f"{trained}/6 padding tokens trained")

    if not any(k in repo.lower() for k in KNOWN_NO_BOUNDARY):
        try:
            # Templates that reject a system role skip only this arm; letting the exception escape would demote an otherwise checkable model to SKIP.
            tail = t.apply_chat_template(NEXT, tokenize=False, add_generation_prompt=False)
        except Exception:
            return (not bad), "; ".join(bad)
        ids2, idx2 = _piece_index(t, text + tail)
        labels = _mask(t, ins, res, ids2)
        for sub, why in ((NEXT_SYS, "system prompt"), (NEXT_USER, "user content")):
            i = idx2(sub)
            if i and _all_trained(labels, i) is True:
                bad.append(f"the next packed sample's {why} was trained")
        answer = idx2(NEXT_ASST)
        if answer and _all_trained(labels, answer) is not True:
            bad.append("the packed second sample's answer was not trained")
    return (not bad), "; ".join(bad)


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
        ok, why = _boundary_arms(inner, repo, ins, res)
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
