# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = [
    "train_on_responses_only",
    "get_chat_template_parts",
    "sft_prepare_dataset",
    "standardize_data_formats",
    "patch_torchcodec_audio_decoder",
]

from typing import Union, Callable, Optional, List, Dict
import itertools as _itertools
import torch

def _iterable_batch_size(dataset, default = 1000):
    """Batch size to re-use when mapping an IterableDataset.

    Only a dataset that has already been mapped carries one: a fresh streaming
    dataset holds an ArrowExamplesIterable, which has no `batch_size` at all, so
    reading it raised AttributeError before the first map could run. `default`
    matches datasets' own `map` default.
    """
    return getattr(getattr(dataset, "_ex_iterable", None), "batch_size", None) or default


# From https://www.geeksforgeeks.org/longest-common-substring-array-strings/
# Longest Common Substring in an Array of Strings
def _old_longest_common_substring(arr):
    n = len(arr)
    s = arr[0]
    l = len(s)
    res = ""
    for i in range(l):
        for j in range(i + 1, l + 1):
            stem = s[i:j]
            k = 1
            for k in range(1, n):
                if stem not in arr[k]:
                    break
            if (k + 1 == n and len(res) < len(stem)):
                res = stem
    return res
pass


def _longest_common_sublist(lists):
    """Longest common sublist among multiple lists (ties broken arbitrarily,
    empty list if none)."""
    if not lists: return []

    min_len = min(len(lst) for lst in lists)
    if min_len == 0: return []

    def has_common_sublist(length):
        """Return (exists, sublist) for a common sublist of `length`."""
        common = set()
        first = lists[0]
        # All sublists of `length` from the first list
        for i in range(len(first) - length + 1):
            sub = tuple(first[i:i + length])
            common.add(sub)
        pass

        # Keep only sublists also present in every remaining list
        for lst in lists[1:]:
            current = set()
            for i in range(len(lst) - length + 1):
                sub = tuple(lst[i:i + length])
                if sub in common:
                    current.add(sub)
            common = current
            if not common:
                return False, []
        pass
        return True, list(common.pop())
    pass

    # Binary search on length
    left, right = 1, min_len
    result = []

    while left <= right:
        mid = left + (right - left) // 2
        exists, sublist = has_common_sublist(mid)
        if exists:
            result = sublist
            left = mid + 1
        else:
            right = mid - 1
    pass

    return result
pass


def _find_common_token_ids(component, tokenizer, force_match = False):
    """Find the middle-most repeated token sequence for a chat component.

    Tokenizers may fold surrounding newlines/spaces into one token, so we probe
    variants (e.g. "\\n### User:\\n\\n") to find the stable common core.

    Returns (core, optional_left, optional_right). When no core can be located in
    the component's own tokenization the result is ([], [], []) - callers must
    treat an empty core as "no match" rather than as a matchable span.
    """
    right_text = ""
    if   component.endswith (" "): right_text = " "
    elif component.endswith("\n"): right_text = "\n"
    left_text = ""
    if   component.startswith (" "): left_text = " "
    elif component.startswith("\n"): left_text = "\n"
    stripped = component.strip()
    
    # Add current pieces and also newlines
    all_input_ids = []
    if not force_match:
        for left in range(3):
            for right in range(3):
                x = left*left_text + stripped + right*right_text
                x = tokenizer(x, add_special_tokens = False).input_ids
                all_input_ids.append(x)

                x = left*"\n" + stripped + right*"\n"
                x = tokenizer(x, add_special_tokens = False).input_ids
                all_input_ids.append(x)
            pass
        pass
    else:
        x = tokenizer(component, add_special_tokens = False).input_ids
        all_input_ids.append(x)
    pass

    # Old longest common substring is replaced with actual longest common list of numbers
    # substring = _old_longest_common_substring([str(x + [0]) for x in all_input_ids])
    # substring = substring.split(", ")[:-1]
    # substring = [int(x) for x in substring if x.isdigit()]
    substring = _longest_common_sublist([x + [0] for x in all_input_ids])

    # substring == [0] may just be the original single token.
    # Fixes https://github.com/unslothai/unsloth/issues/1290
    # Mistral [INST] [/INST] singular tokens break since we output [0] but need [3] [4].
    if substring == [0] and len(all_input_ids[0]) == 1:
        single_token = all_input_ids[0][0]
        if all(single_token in x for x in all_input_ids):
            substring = [single_token]
    pass

    # If substring is original input_ids + [0], keep the original. Happens when
    # the chat template uses no newlines/spaces (e.g. Phi-4).
    if (len(set(str(x) for x in all_input_ids)) == 1) and \
        (len(all_input_ids[0]) + 1 == len(substring)) and \
        (all_input_ids[0] == substring[:-1]):

        substring = all_input_ids[0]
    pass

    # Recover optional left/right tokens around the matched core. `substring` carries
    # an appended [0] sentinel, so it need not be a sublist of `original`; track the
    # match index explicitly rather than letting the loop fall through (which sliced
    # at the last index, or left `j` unbound on an empty tokenization).
    original = tokenizer(component, add_special_tokens = False).input_ids
    where = -1
    for j in range(len(original)):
        if original[j : j + len(substring)] == substring:
            where = j
            break
    if where == -1:
        # The core was never located in the component's own tokenization, so it was
        # never verified against real tokenizer output. It can be the bare [0]
        # sentinel, or a real token with that sentinel glued on (Phi-3 gives
        # [32010, 0] for a marker that tokenizes to [32010]). Returning it either
        # makes A_first == 0 and matches every <unk>/<pad>/"!" in the corpus, or
        # matches nothing and masks the whole dataset. Report no core instead, which
        # both empty-core guards downstream already handle. Only force_match = False
        # gets here: with force_match = True the core is the component's own
        # tokenization, so it is always located at index 0.
        return [], [], []
    optional_left  = original[:where]
    optional_right = original[where+len(substring):]
    return substring, optional_left, optional_right
pass


def get_chat_template_parts(tokenizer):
    """Auto-detect (instruction_part, response_part) from the tokenizer's chat
    template, so train_on_responses_only needs no manual markers."""
    # All Unsloth Zoo code licensed under LGPLv3
    import re
    from collections import Counter

    tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    # Render with the processor's template when it has one (that is what VLM batching renders
    # through, and it can differ from the inner tokenizer's text template); validate token ids
    # with the inner tokenizer. Fall back to the inner template for plain / processor-less cases.
    render_tok = tokenizer if getattr(tokenizer, "chat_template", None) is not None else tok
    if getattr(render_tok, "chat_template", None) is None:
        raise ValueError("Unsloth: No chat_template to auto-detect from - pass instruction_part and response_part.")

    # Sentinels survive Jinja |trim and never collide with real tokens
    U, A = "⁠USERPROBE7q⁠", "⁠ASSTPROBE4z⁠"
    render = lambda msgs, gen: render_tok.apply_chat_template(msgs, tokenize = False, add_generation_prompt = gen)
    starts = lambda text, n: [m.start() for m in re.finditer(re.escape(n), text)]
    ends   = lambda text, n: [m.end()   for m in re.finditer(re.escape(n), text)]
    eos = getattr(tok, "eos_token", "") or ""
    bos = getattr(tok, "bos_token", "") or ""
    try:    added = set(tok.get_added_vocab().keys())
    except Exception: added = set()
    # Keep only non-empty string tokens: "" would make strip_shared loop forever
    # and None would break sorting by len.
    specials = sorted({str(s) for s in (set(getattr(tok, "all_special_tokens", []) or []) | added) if s}, key = len, reverse = True)

    def strip_lead(s, *prefixes):
        # Remove any of the given leading strings, repeatedly
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if p and s.startswith(p): s, changed = s[len(p):], True
        return s

    def strip_shared(a, b):
        # Drop leading special-tokens/whitespace common to both until roles differ
        while True:
            wa, wb = re.match(r"\s+", a), re.match(r"\s+", b)
            if wa and wb and wa.group() == wb.group():
                a, b = a[wa.end():], b[wb.end():]
                continue
            t = next((s for s in specials if a.startswith(s)), None)
            if t and b.startswith(t):
                a, b = a[len(t):], b[len(t):]
                continue
            break
        return a, b

    def gap_mode(from_ends, to_starts):
        # Most common text between adjacent content blocks
        out = []
        for e in from_ends:
            nxt = [s for s in to_starts if s >= e]
            if nxt: out.append(full[e : min(nxt)])
        return Counter(out).most_common(1)[0][0] if out else ""

    # Render a 3-turn probe; read markers off the gaps between content blocks. VLM processors
    # differ on content shape: some want plain strings, some structured parts, and some want the
    # user as parts but the assistant collapsed to a string. Probe those shapes (per-role) and
    # keep whichever renders our sentinels, matching what VLM batching would render.
    _str, _lst = (lambda s: s), (lambda s: [{"type": "text", "text": s}])
    def _convo(uwrap, awrap):
        return [{"role": "user", "content": uwrap(U)}, {"role": "assistant", "content": awrap(A)}] * 3
    uwrap, awrap, full = None, None, ""
    for _uw, _aw in ((_str, _str), (_lst, _lst), (_lst, _str)):
        try: rendered = render(_convo(_uw, _aw), False)
        except Exception: continue
        if starts(rendered, U) and starts(rendered, A):
            uwrap, awrap, full = _uw, _aw, rendered; break
    if uwrap is None:
        raise ValueError("Unsloth: Could not auto-detect chat template structure - pass instruction_part and response_part.")
    convo = _convo(uwrap, awrap)
    instr_gap = gap_mode(ends(full, A), starts(full, U))
    resp_gap  = gap_mode(ends(full, U), starts(full, A))

    # Clean assistant header = generation prompt. Diff the tails after the last user
    # turn; shared leading special-tokens are the turn terminator, so drop them.
    # A headerless template (e.g. Mistral [INST]) leaves add_generation_prompt a no-op, so the
    # two renders match and there is no header: force it empty to reach the headerless fallback
    # (otherwise the shared tail like [/INST] is mistaken for a header and pulls eos into the marker).
    end_user = convo[:-1]
    tail = lambda s: s[s.rfind(U) + len(U):] if U in s else ""
    _gen_on, _gen_off = render(end_user, True), render(end_user, False)
    asst_header = "" if _gen_on == _gen_off else strip_shared(tail(_gen_on), tail(_gen_off))[0]

    if asst_header and resp_gap.endswith(asst_header) and len(asst_header) < len(resp_gap):
        # Header template (Llama/Gemma/Qwen/Phi-4): terminator is the resp_gap prefix
        response_part, instruction_part = asst_header, strip_lead(instr_gap, resp_gap[:-len(asst_header)])
    elif asst_header and asst_header == resp_gap:
        # Terminator leaked into the gen-diff (Phi-3): strip shared separators, then strip the
        # assistant turn terminator (eos/bos) off the instruction marker so a non-final assistant
        # turn's eos stays trainable, matching explicit markers.
        response_part, instruction_part = strip_shared(asst_header, instr_gap)
        instruction_part = strip_lead(instruction_part, " ", "\t", eos, bos)
    else:
        # Headerless template (Mistral [INST]/[/INST]): strip bos/eos separators here.
        # Strip eos alone, never eos+"\n": a turn-delimiting newline stays as the marker
        # anchor (e.g. "\n### Human:") instead of being glued to eos and dropped, which
        # left a bare marker that could match inside message content. strip_lead skips
        # empty prefixes, so an unset eos never strips a bare "\n".
        response_part    = strip_lead(resp_gap, " ", "\t", eos, bos)
        instruction_part = strip_lead(instr_gap, " ", "\t", eos, bos)

    # Reasoning templates inject thinking-block scaffolding into the generation prompt
    # that a real assistant turn ("<think>...</think>answer") does not carry right after
    # the header, so a marker holding it would miss the turn. Two shapes:
    #   paired empty tag - "<|im_start|>assistant\n<think></think>" (Qwen3-Thinking)
    #   lone close tag   - "<|assistant|></think>"                  (GLM-4.x)
    # Re-probe with a reasoning-filled turn and drop the scaffold only when confirmed gone
    # (templates that always emit it keep it). Dropping only shortens the marker to the
    # assistant header, so it can never unmask user content.
    mt = re.search(r"<([^\s/>]+)>\s*</\1>\s*$", response_part) or \
         re.search(r"</([^\s/>]+)>\s*$", response_part)
    if mt and mt.start() > 0:
        tag = mt.group(1)
        scaffold = response_part[mt.start():]
        header = response_part[:mt.start()]
        try:
            filled = render([{"role": "user", "content": uwrap(U)},
                             {"role": "assistant", "content": awrap(f"<{tag}>rZ9</{tag}>{A}")}], False)
            pos = filled.rfind(header)
            after = filled[pos + len(header):] if pos != -1 else ""
            if pos != -1 and not after.startswith(scaffold):
                response_part = header
        except Exception:
            pass

    # Only strip whitespace from header markers: do NOT strip bos here, since for some
    # tokenizers bos doubles as the turn opener (e.g. SmolLM2 bos == <|im_start|>) and
    # stripping it would leave an unanchored marker that matches inside user content.
    instruction_part = strip_lead(instruction_part, " ", "\t").rstrip(" \t")
    response_part    = strip_lead(response_part, " ", "\t").rstrip(" \t")
    if not instruction_part or not response_part:
        raise ValueError("Unsloth: Auto-detection produced an empty marker - pass instruction_part and response_part.")

    # Each marker must tokenize to a core present in a tokenized probe, else masking would
    # silently train on nothing (role tags that are not atomic tokens, or whose ids shift by
    # context). Some SentencePiece tokenizers need a leading space, so try that variant too.
    # Markers also tokenize differently at text start vs mid-text (context-dependent
    # SentencePiece): Zephyr's "<|assistant|>" is "▁<" standalone but bare "<" after "</s>\n",
    # so also probe a leading-newline variant.
    # 3 turns per role means a reliable marker matches >=2 times; a lone hit can be the
    # text-start tokenization only (Zephyr "<|user|>\n" at pos 0), which breaks multi-turn masking.
    probe_ids = tok(full, add_special_tokens = False).input_ids
    def count_matches(cand):
        core = _find_common_token_ids(cand, tok, True)[0]
        if not core: return 0
        return sum(probe_ids[i : i + len(core)] == core for i in range(len(probe_ids) - len(core) + 1))
    def validate(part, part_name):
        counts = [(cand, count_matches(cand)) for cand in (part, " " + part, "\n" + part)]
        for cand, n in counts:
            if n >= 2: return cand
        # Fall back to a single-match original variant (marker unique to the probe layout),
        # preserving prior behaviour when nothing matches twice.
        for cand, n in counts[:2]:
            if n >= 1: return cand
        raise ValueError(f"Unsloth: Could not reliably auto-detect {part_name} (detected {repr(part)}) - pass instruction_part and response_part.")
    return validate(instruction_part, "instruction_part"), validate(response_part, "response_part")
pass


def _model_forward_parameter_names(model):
    """Every named parameter of `model.forward`, unwrapping PEFT / compile layers.

    A wrapper's own forward hides the real signature, so walk down to the base
    model. Used to decide what a dataset column must survive for: anything the
    model is actually fed.
    """
    import inspect as _inspect
    names = set()
    for _ in range(6):
        if model is None: break
        forward = getattr(model, "forward", None)
        if forward is not None:
            try:
                names.update(
                    name for name, p in _inspect.signature(forward).parameters.items()
                    if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
                )
            except (TypeError, ValueError): pass
        unwrap = getattr(model, "get_base_model", None)
        nxt = None
        if callable(unwrap):
            try: nxt = unwrap()
            except Exception: nxt = None
        if nxt is None or nxt is model:
            nxt = getattr(model, "_orig_mod", None) or getattr(model, "base_model", None)
        model = None if nxt is model else nxt
    names.discard("self")
    return names


def _labels_are_token_level(dataset, sample):
    """Is a raw split's `labels` column token-level supervision? True / False /
    None for "cannot tell".

    Judging by the first row's value alone is wrong twice over: a nullable
    token-level column whose first row is null looks like no column at all, and
    dropping it makes the masking pass rebuild labels from `input_ids`, silently
    un-masking exactly what the caller masked. So ask the FEATURE first, then
    the first NON-null row, then give up.

    Everything here is duck-typed rather than version-gated: `Sequence`, `List`
    and `LargeList` all expose their element type as `.feature`, a bare `[...]`
    is the other list spelling, and `Value` / `ClassLabel` are the scalar ones.
    """
    features = getattr(dataset, "features", None)
    feature = None
    try:
        if features is not None: feature = features["labels"]
    except (KeyError, TypeError): feature = None
    if feature is not None:
        if isinstance(feature, (list, tuple)) or hasattr(feature, "feature"): return True
        if type(feature).__name__ in ("Value", "ClassLabel"): return False
        return None  # some nested/unknown feature: ambiguous, and the caller keeps
    # No usable features: an in-memory dict-of-lists, or an IterableDataset.
    value = sample.get("labels") if isinstance(sample, dict) else None
    if value is None:
        # Scan for the first non-null row, capped so a streaming or huge split
        # is not walked end to end for a column verdict.
        try:
            for i, row in enumerate(dataset):
                if i >= 100: break
                if not isinstance(row, dict): break
                value = row.get("labels")
                if value is not None: break
        except Exception: value = None
    if value is None: return None
    # A string/bytes `labels` is a class NAME, not tokens; it can never be
    # supervision, so it goes with the other raw columns.
    if isinstance(value, (str, bytes)): return False
    try:
        len(value)
        return True
    except TypeError:
        return False


def _case_variants(trainer, keys):
    """`keys` plus the spellings the trainer's own splits actually use.

    `_has_media` matches by lowercasing a column name, but this whitelist held
    only the lowercase spellings, so `remove_unused_columns` stripped an `Image`
    or `IMAGE_URL` column before the dispatcher could see it and the batch went
    to the text collator that cannot encode it. The observed names are read from
    the splits the trainer holds; the fixed variants below cover a later split
    that only reaches us after the signature is already cached.
    """
    found = set(keys)
    for key in keys:
        found.update((key.upper(), key.title(), key[:1].upper() + key[1:]))
    lowered = {k.lower() for k in keys}
    splits = [getattr(trainer, "train_dataset", None), getattr(trainer, "eval_dataset", None)]
    while splits:
        split = splits.pop()
        if isinstance(split, dict):
            splits.extend(split.values())
            continue
        try: names = getattr(split, "column_names", None) or ()
        except Exception: continue
        if isinstance(names, dict): names = [n for v in names.values() for n in v or ()]
        found.update(n for n in names if isinstance(n, str) and n.lower() in lowered)
    return found


def _keep_media_columns(trainer, keys):
    """Stop `remove_unused_columns` stripping media before the collator sees it.

    `_signature_columns` is cached lazily off the model's forward signature, so
    seeding it here is what a later split reads; the trainer only fills it when
    still None, and extending an existing list is additive.
    """
    try:
        existing = getattr(trainer, "_signature_columns", None)
        names = sorted(k for k in _case_variants(
            trainer, {k for k in keys if isinstance(k, str)}))
        if existing is None:
            # Seeding it stops the trainer deriving the signature later, so the
            # model's own forward parameters go in here: a declared
            # `position_ids` or a custom `sample_weight` outside a fixed list
            # would otherwise be removed before collation, silently changing
            # what the model is trained on.
            # `index`/`label`/`label_ids` are what HF always keeps; without them
            # a seeded list would drop the columns the loss itself needs.
            declared = _model_forward_parameter_names(getattr(trainer, "model", None))
            # Trainer's own derivation does `+= list(set(["label", "label_ids"]
            # + self.label_names))`, so a custom trainer whose supervision is
            # consumed by `compute_loss` rather than declared by `forward` had
            # it dropped here and on every later split.
            declared |= set(getattr(getattr(trainer, "args", None), "label_names", None) or ())
            trainer._signature_columns = sorted(set(names) | declared | {
                "label", "label_ids", "index", "input_ids", "attention_mask",
                "labels", "completion_mask", "assistant_masks", "token_type_ids",
            })
        else:
            trainer._signature_columns = list(existing) + [
                n for n in names if n not in existing]
    except Exception:
        pass


# Every suffix a missing entry costs is a VLM row silently trained as text,
# so cover the modern web formats too (`.avif` is what most image CDNs serve
# now) and the older spellings of the ones already here.
_MEDIA_SUFFIXES = (
    ".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".apng", ".gif", ".bmp", ".dib",
    ".webp", ".avif", ".tif", ".tiff", ".svg", ".heic", ".heif", ".heics",
    ".jp2", ".j2k", ".jxl", ".ppm", ".pgm", ".pnm", ".pbm",
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpg", ".mpeg", ".m4v",
    ".ogv", ".3gp", ".3g2", ".wmv", ".flv", ".m2ts", ".mts",
    ".wav", ".mp3", ".flac", ".ogg", ".oga", ".m4a", ".aac", ".opus",
    ".wma", ".aiff", ".aif", ".aifc", ".amr", ".mka", ".weba",
)

def _looks_like_media_value(value, _depth = 0):
    """A string naming an image/video/audio file, by extension or data URI.

    A plural ambiguous column (`urls`, `paths`, `files`) holds a *list* of
    those strings per row, so recurse rather than call the row safe.
    """
    if isinstance(value, (list, tuple)):
        return _depth < 4 and \
            any(_looks_like_media_value(v, _depth + 1) for v in value)
    if not isinstance(value, str): return False
    text = value.strip().lower()
    if text.startswith(("data:image/", "data:video/", "data:audio/")): return True
    # Drop a query string/fragment: `.../cat.jpg?width=64`.
    for sep in ("?", "#"):
        text = text.split(sep, 1)[0]
    return text.endswith(_MEDIA_SUFFIXES)


# The prose that goes WITH the media. Tokenizer outputs live in `_TEXT_COLUMNS`;
# these are the raw columns a vision collator still has to read, and a keep-list
# that held the image but not its prompt left that collator nothing to tokenize.
_RAW_TEXT_COMPANION_COLUMNS = frozenset((
    "text", "texts", "prompt", "prompts", "question", "questions",
    "caption", "captions", "instruction", "instructions", "input", "inputs",
    "query", "queries", "answer", "answers", "output", "outputs",
    "completion", "completions", "response", "responses", "content",
))


def _holds_raw_text(value, _depth = 0):
    """A string, or a nest of them, that no collator can stack into a tensor."""
    if isinstance(value, str): return True
    if isinstance(value, (list, tuple)):
        return _depth < 4 and any(_holds_raw_text(v, _depth + 1) for v in value)
    if isinstance(value, dict):
        return _depth < 4 and any(_holds_raw_text(v, _depth + 1) for v in value.values())
    return False


class _MediaAwareCollator:
    """The text collator, falling back to the caller's own when a batch has media.

    The bypass swaps the trainer-wide collator, and that swap outlives
    construction: `predict(test_dataset = ...)` and an `evaluate` override both
    build their dataloader from `trainer.data_collator`, so a multimodal split
    handed over later reached a collator that cannot process images and either
    failed or trained the modality away. Keeping the original and choosing per
    batch is the only thing that can answer for a split nobody has seen yet.

    Module level so a DataLoader worker under `spawn` can pickle it, which a
    closure or a `type()`-built class cannot.
    """
    def __init__(self, text, media, media_keys, ambiguous_keys = (),
                 companion_keys = ()):
        self.text = text
        self.media = media
        # Kept so the media collator can read the prompt beside its image, and
        # stripped again on the text path, which cannot tensorize a raw string.
        self.companion_keys = frozenset(companion_keys)
        # Split, not merged: an ambiguous name is decided by its value below,
        # and folding the two would match `url` on the name alone.
        self.ambiguous_keys = frozenset(ambiguous_keys)
        self.media_keys = frozenset(media_keys) - self.ambiguous_keys

    # Raw conversational columns. Their images sit INSIDE the value, as
    # `{"type": "image", ...}` parts, so no top-level key names them and a batch
    # of them went to the text collator that cannot read the conversation at all,
    # let alone its images. Routed to the media collator on the column's
    # presence: that collator handles this TRL format, and the text one cannot.
    _CONVERSATION_KEYS = frozenset((
        "messages", "conversations", "conversation", "chat",
    ))

    def _has_media(self, features):
        for feature in features or ():
            keys = feature.keys() if isinstance(feature, dict) else ()
            for k in keys:
                if not isinstance(k, str): continue
                lowered = k.lower()
                # An ambiguous name is weighed by its VALUE, the same trade the
                # initial-split guard makes: `path`/`url` is a media reference
                # or ordinary provenance, and matching the name alone would send
                # every row carrying a plain URL to the vision collator.
                if lowered in self.ambiguous_keys:
                    if _looks_like_media_value(feature.get(k)): return True
                    continue
                if lowered in self.media_keys: return True
                if lowered in self._CONVERSATION_KEYS and \
                    isinstance(feature.get(k), (list, tuple)): return True
        return False

    def __call__(self, features):
        if self._has_media(features):
            return self.media(features)
        # Strip what only the MEDIA path wanted kept. The signature whitelist is
        # global, so a benign `url`/`path`/`prompt`/`content` column now survives
        # unused-column removal for every split -- and the text collator
        # tensorizes every key it is handed, so a retained string kills the
        # batch. Removed here, not from the keep-list: the media path still
        # needs those columns, and this is the only point that knows which path
        # a batch took. Copies, so the caller's rows are left alone.
        # By VALUE for the ambiguous and companion names, not by name alone. Both
        # sets are ordinary English words, so under `remove_unused_columns=False`
        # a caller's tensorizable auxiliary input called `output` or `content`
        # was dropped from every text batch and its custom `compute_loss` never
        # saw it. What the text collator cannot stack is a raw STRING; a number
        # or a tensor under the same name is a model input and stays.
        # The media names go on sight: they are never model inputs, and a text
        # batch is one `_has_media` already declined to call multimodal.
        maybe = self.ambiguous_keys | self.companion_keys
        def _keep(key, value):
            if not isinstance(key, str): return True
            lower = key.lower()
            if lower in self.media_keys: return False
            return not (lower in maybe and _holds_raw_text(value))
        stripped = [
            {k: v for k, v in f.items() if _keep(k, v)}
            if isinstance(f, dict) else f
            for f in features or ()
        ]
        return self.text(stripped)

    def __getattr__(self, attribute):
        # Only for names this class does not define; `text` is set in __init__,
        # so this cannot recurse through it.
        if attribute.startswith("__"): raise AttributeError(attribute)
        return getattr(self.__dict__["text"], attribute)
pass


def train_on_responses_only(
    trainer,
    instruction_part  = None,
    response_part     = None,
    force_match       = True,  # Match newlines as well!
    tokenizer         = None,  # Optional
    return_function   = False, # Useful for iterating over lists
    num_proc          = None,
    last_response_only = False, # Train only on the last assistant turn
):
    """Train only on responses by masking instruction labels to -100.

    With last_response_only=True, only the final assistant turn is unmasked;
    earlier assistant turns stay at -100 (never written, never copied from
    old_labels).
    """
    # All Unsloth Zoo code licensed under LGPLv3
    if trainer is not None:
        try:
            from .mlx.trainer import (
                MLXTrainer,
                train_on_responses_only as _mlx_train_on_responses_only,
            )
        except ImportError:
            MLXTrainer = None
        if MLXTrainer is not None and isinstance(trainer, MLXTrainer):
            return _mlx_train_on_responses_only(
                trainer,
                instruction_part=instruction_part,
                response_part=response_part,
                force_match=force_match,
                tokenizer=tokenizer,
                return_function=return_function,
                num_proc=num_proc,
                last_response_only=last_response_only,
            )

    if tokenizer is None and trainer is not None:
        tokenizer = trainer.processing_class if hasattr(trainer, "processing_class") else trainer.tokenizer
    # Keep the original object (may be a VLM processor) so auto-detect can read a
    # chat template that lives only on the processor; the matcher uses the inner one.
    processor = tokenizer
    # Get non vision tokenizer
    if hasattr(tokenizer, "image_processor") or hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
    if  not hasattr(tokenizer, "_unsloth_input_part") or \
        not hasattr(tokenizer, "_unsloth_output_part"):

        if instruction_part is None and response_part is None:
            # Neither given: auto-detect both from the chat template
            instruction_part, response_part = get_chat_template_parts(processor)
            print(f"Unsloth: Auto-detected instruction_part = {repr(instruction_part)} and response_part = {repr(response_part)}")
        elif instruction_part is None or response_part is None:
            raise ValueError("Unsloth: Give both instruction_part and response_part, or neither to auto-detect!")
        pass
    elif (instruction_part is not None or response_part is not None) and \
        (hasattr(tokenizer, "_unsloth_input_part") or hasattr(tokenizer, "_unsloth_output_part")):

        raise ValueError("Unsloth: Your tokenizer already has instruction and response parts set - do not give custom ones!")
    else:
        instruction_part = tokenizer._unsloth_input_part
        response_part    = tokenizer._unsloth_output_part
    pass

    # Get most common tokens since tokenizers can tokenize stuff differently!
    Q_must, Q_left, Q_right = _find_common_token_ids(instruction_part, tokenizer, force_match)
    A_must, A_left, A_right = _find_common_token_ids(response_part,    tokenizer, force_match)

    # Empty core -> named error instead of IndexError on A_must[0]. Two ways in: an
    # explicitly-passed marker that tokenizes to nothing, and a marker whose core was
    # never located in its own tokenization (force_match = False on templates like
    # Phi-3, which used to mask the whole dataset instead of saying anything).
    if len(Q_must) == 0 or len(A_must) == 0:
        _empty = "instruction_part" if len(Q_must) == 0 else "response_part"
        # force_match = True already resolves the unlocated-core case, so only suggest
        # it to callers who are not using it.
        _retry = "" if force_match else ", or try force_match = True"
        raise ValueError(
            f"Unsloth: {_empty} could not be resolved to a stable token sequence, so it "
            "cannot be matched against your dataset - it tokenizes to nothing, or no "
            "stable core could be recovered from it. Pass a different marker, pass "
            f"neither to auto-detect both{_retry}."
        )
    pass

    # Store some temporary stuff
    A_first = A_must[0]
    len_A_must = len(A_must)
    A_left_reversed = A_left[::-1]
    A_right_forward = A_right

    Q_first = Q_must[0]
    len_Q_must = len(Q_must)
    Q_left_reversed = Q_left[::-1]
    Q_right_forward = Q_right
    torch_Tensor = torch.Tensor
    torch_int64  = torch.int64

    def _train_on_responses_only(examples):
        input_ids_ = examples["input_ids"]
        use_tensors = False
        if type(input_ids_) is torch_Tensor:
            use_tensors = True
            input_ids_ = input_ids_.tolist()
        elif not isinstance(input_ids_, list) and hasattr(input_ids_, "tolist"):
            # `with_format("numpy")` hands back an ndarray. Slicing one and
            # comparing it to the marker list gives an array, and the `if` on
            # that raised "truth value ... is ambiguous" before a single row was
            # masked. Not `use_tensors`: numpy is a read-side format that
            # `datasets` re-applies on the way out, so lists are what to return.
            input_ids_ = input_ids_.tolist()
        if "labels" in examples:
            # Type-check labels the same way input_ids is above: under
            # datasets.map(batched = True) a "labels" column arrives as a plain
            # list of lists, which has no .tolist() and raised AttributeError.
            labels_ = examples["labels"]
            if type(labels_) is torch_Tensor or (
                not isinstance(labels_, list) and hasattr(labels_, "tolist")):
                labels_ = labels_.tolist()
            assert(len(labels_) == len(input_ids_))
        else:
            labels_ = [None]*len(input_ids_)

        all_labels = []
        for input_ids, old_labels in zip(input_ids_, labels_):
            n = len(input_ids)
            labels = [-100] * n

            use_old_labels = False
            if old_labels is not None:
                use_old_labels = True
                assert(n == len(old_labels))
            n_minus_1 = n - 1
            j = 0

            # Collect all (assistant_k, user_j) spans for this sample
            spans = []
            while j < n:
                # Find <assistant>
                if (input_ids[j] == A_first) and \
                    (input_ids[j : (k := j + len_A_must)] == A_must):

                    # Extend over optional tokens, backward then forward
                    for optional_left in A_left_reversed:
                        if j < 1: break
                        if optional_left == input_ids[j-1]: j -= 1
                        else: break
                    pass
                    for optional_right in A_right_forward:
                        if k >= n_minus_1: break
                        if optional_right == input_ids[k+1]: k += 1
                        else: break
                    pass
                    # assistant_j = j
                    assistant_k = k

                    j = assistant_k
                    # Find the next <user> (or the final item if assistant is last)
                    while j < n:
                        if (j == n_minus_1) or \
                            ((input_ids[j] == Q_first) and \
                             (input_ids[j : (k := j + len_Q_must)] == Q_must)):

                            # Extend over optional tokens, backward then forward
                            for optional_left in Q_left_reversed:
                                if j < 1: break
                                if optional_left == input_ids[j-1]: j -= 1
                                else: break
                            pass
                            for optional_right in Q_right_forward:
                                if k >= n_minus_1: break
                                if optional_right == input_ids[k+1]: k += 1
                                else: break
                            pass
                            user_j = j
                            # Account for last item
                            if user_j != n_minus_1:
                                # user_k = k
                                # j = user_k
                                j = k
                            else:
                                user_j = n
                                k = n
                            pass

                            spans.append((assistant_k, user_j))
                            break
                        pass
                        j += 1
                    pass
                pass
                j += 1
            pass

            # Apply labels (last assistant turn only when last_response_only).
            # spans[-1:] is [] when no assistant turn was found, so such samples
            # stay fully masked at -100.
            apply_spans = spans[-1:] if last_response_only else spans
            for assistant_k, user_j in apply_spans:
                if not use_old_labels:
                    labels[assistant_k : user_j] = input_ids [assistant_k : user_j]
                else:
                    labels[assistant_k : user_j] = old_labels[assistant_k : user_j]

            all_labels.append(labels)
        pass
        return { "labels" : torch.tensor(all_labels, dtype = torch.int64) if use_tensors else all_labels }
    pass
    if return_function:
        return _train_on_responses_only

    import multiprocessing as _mp
    _num_proc_was_auto = num_proc is None or type(num_proc) is not int
    if _num_proc_was_auto:
        if _mp.get_start_method() != 'fork':
            num_proc = None
        else:
            import psutil
            num_proc = min(max((psutil.cpu_count() or 1)+4, 2), 64)
            # Cap by available memory to avoid OOM (1 proc per GB; 1 if <=2GB)
            memory_gb_left = psutil.virtual_memory().available / (1024**3)
            if memory_gb_left <= 2:
                num_proc = 1
            else:
                num_proc = min(num_proc, int(memory_gb_left))

    # Single-process small datasets (workers cost more than they save, and large auto
    # num_proc caused Windows spawn loops #3211/#3397); keep explicit user values.
    _MIN_ROWS_FOR_MULTIPROC = 5_000
    def _effective_num_proc(dataset):
        # `1` means "no multiprocessing" to everyone who passes it, but datasets
        # >= 4.1 pools for any num_proc >= 1, so returning it verbatim built a
        # Pool(1): one forked child holding a whole tokenizer, on a split over
        # _MIN_ROWS_FOR_MULTIPROC where the guard below no longer applies. That
        # left UNSLOTH_DATASET_NUM_PROC=0 -- the remedy the dead-worker message
        # recommends -- still forking. `None` is in-process on every supported
        # release, and is what datasets 3.x already did with `1`.
        if num_proc is None or num_proc == 1: return None
        if not _num_proc_was_auto: return num_proc  # honor explicit user value
        try:
            if len(dataset) < _MIN_ROWS_FOR_MULTIPROC: return None
        except TypeError:
            return None  # unknown length (e.g. IterableDataset)
        return num_proc
    pass

    # `remove_unused_columns = False` is the user asking for their columns to
    # survive, and HF's Trainer honours it: a custom `compute_loss` can pop a
    # `sample_weight` the model itself never declares, so the model-input
    # keep-lists below would delete the weighting the run depends on.
    _keep_every_column = not getattr(
        getattr(trainer, "args", None), "remove_unused_columns", True)

    # transformers 5.0+ VLMs skip dataset prep in SFTTrainer.__init__
    # (skip_prepare_dataset=True when _is_vlm), so tokenize before masking.
    def _maybe_tokenize_dataset(dataset):
        if dataset is None:
            return dataset
        # An empty split has no row to peek at and nothing to tokenize.
        sample = next(iter(dataset), None)
        if sample is None or "input_ids" in sample:
            return dataset  # Empty, or already tokenized
        # The already-unwrapped text tokenizer, and the `tokenizer =` override
        # when the caller gave one: the response markers were tokenized with it,
        # so encoding here with anything else yields IDs they can never match.
        _tok = tokenizer
        max_length = getattr(trainer.args, "max_length", None) or getattr(trainer.args, "max_seq_length", 2048)
        text_field = getattr(trainer.args, "dataset_text_field", "text")
        def _tokenize_fn(examples):
            texts = examples.get(text_field)
            if texts is None: texts = examples.get("text", [])
            # `or` boolean-tested the column, which under `with_format("numpy")`
            # is an ndarray and raises "truth value ... is ambiguous". Presence,
            # not truthiness, is the question, and the tokenizer wants a list.
            if not isinstance(texts, list) and hasattr(texts, "tolist"):
                texts = texts.tolist()
            return _tok(texts, truncation=True, max_length=max_length, padding=False)
        _map_kwargs = {"batched": True, "num_proc": _effective_num_proc(dataset)}
        if isinstance(dataset, IterableDataset):
            _map_kwargs = {"batched": True}
        # Drop the raw columns we just tokenized. Keeping them would hand the
        # collator a string column it cannot stack into a tensor. Two exceptions:
        # `labels`, which is already token-level and which the masking pass below
        # intersects with, so removing it would un-mask what the caller masked;
        # and anything `model.forward` declares, such as a per-row `sample_weight`
        # or a custom auxiliary target, which the tokenizer does not recreate and
        # which the later model-input keep-list never gets the chance to save.
        _raw_columns = getattr(dataset, "column_names", None) or list(sample.keys())
        if not isinstance(_raw_columns, dict):
            _keep = _model_forward_parameter_names(getattr(trainer, "model", None))
            # `labels` only when it really is token-level. A raw split can carry
            # a SCALAR `labels` (a class id), and keeping that as supervision
            # sent an int into `_train_on_responses_only`, which calls
            # `len(old_labels)` on it: `TypeError: object of type 'int' has no
            # len()`. A scalar is metadata for the tokenized split and goes with
            # the other raw columns. This has to DISCARD rather than decline to
            # add: `_keep` already holds every name `forward` declares, and every
            # causal LM declares `labels`, so the scalar was kept regardless.
            # Ambiguity resolves to keeping: a kept scalar raises loudly at the
            # masking pass, while a dropped sequence corrupts the run in silence.
            if _labels_are_token_level(dataset, sample) is False:
                _keep = _keep - {"labels"}
            else:
                _keep = _keep | {"labels"}
            # The raw text always goes, whatever else asks for it: it was just
            # turned into `input_ids`, and a `forward` that happens to declare a
            # `text` parameter would otherwise keep the string column for the
            # collator to fail on.
            _keep -= {text_field, "text"}
            # This strip runs before the keep-list at the end of the function, so
            # it has to honour the opt-out itself or the column is already gone.
            # Only the text just tokenized still goes: it is the string the
            # collator cannot stack, and its tokens replace it.
            if _keep_every_column:
                _keep |= set(_raw_columns) - {text_field, "text"}
            _map_kwargs["remove_columns"] = [c for c in _raw_columns if c not in _keep]
        import warnings as _w
        with _w.catch_warnings():
            _w.filterwarnings("ignore", message=".*couldn't be hashed properly.*")
            return dataset.map(_tokenize_fn, **_map_kwargs)
    pass

    # Drop samples with all labels -100 (no training signal). Happens when
    # truncation cuts off the response_part (e.g. long GPT-OSS reasoning
    # channels), which would give NaN loss from cross_entropy(mean)'s 0/0.
    def _has_valid_labels(example):
        labels = example.get("labels")
        if labels is None: return True
        if type(labels) is torch_Tensor:
            return (labels != -100).any().item()
        return any(l != -100 for l in labels)
    pass

    def _diagnose_truncation(dataset, dropped, fatal):
        # When (nearly) the whole dataset is masked away, the usual cause is
        # truncation: max_length cut off the response marker before masking found
        # it. Raise when nothing is left to train on, otherwise just warn so the
        # surviving rows still train (matching the old filter behaviour).
        if getattr(trainer.args, "packing", False): return
        max_length = getattr(trainer.args, "max_length", None) or getattr(trainer.args, "max_seq_length", None)
        # Truncation evidence is a row sitting at the length cap (max_length cut its
        # tail, including the response marker). Without a known cap, or for short rows,
        # a fully masked row is a wrong template / response_part, not truncation, so we
        # keep the generic error instead of telling users to raise max_length.
        if max_length is None: return
        n_sampled = 0; n_trunc = 0
        for i in dropped[:100]:
            input_ids = dataset[int(i)].get("input_ids")
            if input_ids is None: continue
            if getattr(input_ids, "tolist", None): input_ids = input_ids.tolist()
            n_sampled += 1
            if len(input_ids) >= max_length: n_trunc += 1
        if n_sampled == 0 or n_trunc / n_sampled < 0.9: return
        ml = max_length
        message = (
            "Unsloth: train_on_responses_only masked all/most labels to -100.\n"
            f"The most likely cause is truncation: max_length={ml} cut off the response marker "
            f"{repr(response_part)} before masking could find it.\n"
            "Increase max_length to fit your responses, for example SFTConfig(max_length = max_seq_length).\n"
            "If your sequences are genuinely longer, raise max_seq_length when loading the model."
        )
        if fatal: raise ValueError(message)
        print("Unsloth: Warning: " + message)
    pass

    def _no_training_signal_message(dataset_name, how_many):
        return (
            f"Unsloth: train_on_responses_only masked every label to -100 in {dataset_name}"
            f"{how_many}, so there is nothing to train on. The response marker "
            f"{repr(response_part)} was not found in any sample - check that "
            "instruction_part and response_part match your chat template."
        )
    pass

    def _no_training_signal(dataset_name, how_many):
        return ValueError(_no_training_signal_message(dataset_name, how_many))
    pass

    # Streaming rows cannot be counted or filtered, and fix_zero_training_loss
    # skips them too, so read a bounded prefix instead. Iterating restarts the
    # stream, so no rows are consumed.
    _STREAM_SCAN_ROWS = 16

    def _check_streaming_labels(dataset, dataset_name):
        rows = 0
        try:
            # One row past the bound, to tell "the whole stream" from "a prefix".
            for row in _itertools.islice(iter(dataset), _STREAM_SCAN_ROWS + 1):
                rows += 1
                if rows > _STREAM_SCAN_ROWS: break
                labels = row.get("labels") if isinstance(row, dict) else None
                if labels is None: return
                if getattr(labels, "tolist", None): labels = labels.tolist()
                if any(l != -100 for l in labels): return
        except Exception:
            return  # unreadable stream: leave it exactly as before
        if rows == 0: return
        # Only a stream that ENDED inside the prefix is provably all masked. A
        # longer one may be sorted or filtered so that responses start later, and
        # refusing it would block a run that trains fine, so only warn.
        if rows <= _STREAM_SCAN_ROWS:
            raise _no_training_signal(dataset_name, "")
        print("Unsloth: Warning: " + _no_training_signal_message(
            dataset_name, f" (first {_STREAM_SCAN_ROWS} samples)",
        ) + "\nLater samples may still carry responses, so training continues.")
    pass

    def _filter_fully_masked(dataset, dataset_name="dataset"):
        if isinstance(dataset, IterableDataset):
            # Cannot filter an IterableDataset efficiently, but a fully masked one
            # would otherwise train on no signal at all.
            _check_streaming_labels(dataset, dataset_name)
            return dataset
        if "labels" not in dataset.column_names:
            return dataset
        # filter rewrites the whole Arrow table even when it drops nothing, so scan the
        # labels column cheaply first; the common case (0 fully masked) returns as-is.
        n_before = len(dataset)
        # Track only the fully masked rows, so a huge clean corpus builds no per-row list.
        dropped = []
        try:
            idx = 0
            for batch in dataset.select_columns(["labels"]).iter(batch_size = 1000):
                for labels in batch["labels"]:
                    if labels is not None:
                        if getattr(labels, "tolist", None): labels = labels.tolist()
                        if not any(l != -100 for l in labels): dropped.append(idx)
                    idx += 1
        except Exception:
            # Datasets with a custom transform may need other columns; fall back.
            return dataset.filter(_has_valid_labels, num_proc = _effective_num_proc(dataset))
        if not dropped:
            return dataset  # nothing fully masked
        # Most rows masked away across the WHOLE dataset usually means truncation.
        # Only fatal when no rows survive; otherwise warn and keep the valid rows.
        if len(dropped) / n_before >= 0.9:
            _diagnose_truncation(dataset, dropped, fatal = len(dropped) == n_before)
        # Everything masked and not from truncation: the markers do not match the
        # template at all, so fail clearly instead of returning an empty dataset.
        if len(dropped) == n_before:
            raise _no_training_signal(dataset_name, "")
        # Drop via filter (Arrow mask), not select(keep_indices): a keep list would be one
        # Python int per surviving row (GBs on a large corpus). _has_valid_labels is the
        # exact inverse of `dropped`, so survivors are identical.
        dataset = dataset.filter(_has_valid_labels, num_proc = _effective_num_proc(dataset))
        n_removed = n_before - len(dataset)
        if n_removed > 0:
            print(
                f"Unsloth: Removed {n_removed} out of {n_before} samples from {dataset_name} "
                f"where all labels were -100 (no response marker found, usually truncation). "
                f"This prevents NaN loss during training."
            )
        return dataset
    pass

    # Vision/processor collators (e.g. UnslothVisionDataCollator) rebuild labels
    # from the processor at collate time, so dataset-level masking is ignored and
    # replacing the collator would break image handling. Enable response masking on
    # the collator itself and skip the text dataset path.
    def _is_vision_collator(collator):
        if collator is None: return False
        if any(b.__name__ == "UnslothVisionDataCollator" for b in type(collator).__mro__): return True
        # A vision collator may hold the processor under .processor or the common .tokenizer field
        # (e.g. DataCollatorForSeq2Seq(tokenizer=processor)); any multimodal half marks it.
        # Not `image_processor` alone: a Whisper/audio or video-only processor exposes
        # `feature_extractor`/`audio_processor`/`video_processor` and nothing else, so it
        # missed every guard below and went straight to the collator repair, which drops
        # the modality columns. Same list `_derive_multimodal_columns` asks for outputs.
        _halves = ("image_processor", "video_processor", "feature_extractor",
                   "audio_processor", "qformer_tokenizer")
        for attr in ("processor", "tokenizer"):
            obj = getattr(collator, attr, None)
            if obj is not None and any(hasattr(obj, h) for h in _halves): return True
        return any(hasattr(collator, h) for h in _halves)
    pass

    # Processor outputs beside `input_ids`; a row with any of these needs its collator.
    # Only the floor: the real set is derived below from the installed processor.
    _MULTIMODAL_COLUMNS = frozenset((
        "pixel_values", "pixel_values_videos", "pixel_attention_mask",
        "image_grid_thw", "video_grid_thw", "image_sizes", "image_sizes_videos",
        "images", "image", "videos", "video", "audio", "audios",
        "token_type_ids_images", "input_features", "input_features_mask",
        "audio_values", "audio_attention_mask", "input_audio_embeds",
        "aspect_ratio_ids", "aspect_ratio_mask", "cross_attention_mask",
        # Processor-specific spellings: phi4_multimodal, pix2struct, kosmos-2.5.
        "image_pixel_values", "audio_input_features", "audio_embed_sizes",
        "high_res_pixel_values", "flattened_patches",
        # Fuyu keeps its preprocessed images here, beside `input_ids`.
        "image_patches", "image_patches_indices",
        # Integer side-cars whose dtype reads as plain, so only the name refuses:
        # Gemma 3 declares `num_crops`, Llama 3.2 Vision `num_tiles`.
        "num_crops", "num_tiles",
    ))

    # Names the text tokenizer owns; a processor half repeating one of these must
    # not turn every text-only row into a refusal.
    _TEXT_COLUMNS = frozenset((
        "input_ids", "attention_mask", "token_type_ids", "position_ids",
        "labels", "label", "label_ids", "special_tokens_mask",
        "offset_mapping", "length",
    ))

    def _derive_multimodal_columns():
        """The non-text names this processor actually produces.

        A hand list rots: Fuyu emits `image_patches`/`image_patches_indices` and
        every new model spells its own. Each processor half declares its outputs
        in `model_input_names`, so ask them and subtract the text half.
        """
        _halves = ("image_processor", "video_processor", "feature_extractor",
                   "audio_processor", "qformer_tokenizer")
        names = set()
        holders = [getattr(processor, attr, None) for attr in _halves]
        # The processor's own list merges text and vision (that is the only place
        # FuyuProcessor names `image_patches_indices`), so take it too.
        if processor is not tokenizer: holders.append(processor)
        # With the `tokenizer =` override `processor` is the unwrapped text
        # tokenizer, and the real multimodal processor is only on the collator.
        _coll = getattr(trainer, "data_collator", None)
        for attr in ("processor", "tokenizer"):
            held = getattr(_coll, attr, None)
            if held is None or held is processor or held is tokenizer: continue
            holders.append(held)
            holders += [getattr(held, a, None) for a in _halves]
        # `_is_vision_collator` also accepts a half held straight on the collator
        # (`collator.image_processor`), and that half names its own outputs too.
        holders += [getattr(_coll, a, None) for a in _halves]
        for holder in holders:
            if holder is None: continue
            try: names.update(getattr(holder, "model_input_names", None) or ())
            except Exception: pass
        names -= set(getattr(tokenizer, "model_input_names", None) or ())
        return frozenset(names - _TEXT_COLUMNS)
    pass

    try:
        _MULTIMODAL_COLUMNS = _MULTIMODAL_COLUMNS | _derive_multimodal_columns()
    except Exception:
        pass

    # Keys and `type` tags a chat turn uses to point at an image/video/audio.
    # `input_image`/`input_video` are the responses-API spelling of the same
    # parts, and `mlx/loader.py` already renders them as media.
    _MEDIA_KEYS = frozenset((
        "image", "images", "image_url", "video", "videos", "video_url",
        "audio", "audios", "audio_url", "input_audio", "pixel_values",
        "input_image", "input_video",
        "bytes", "path", "url",
    ))

    # The subset of those keys that is media only half the time: `meta = {"url":
    # ..., "path": "corpus/shard.jsonl"}` is ordinary provenance on a text corpus,
    # so refusing on the key name alone refuses a healthy text-only run. Judged by
    # value instead, exactly like `_AMBIGUOUS_MEDIA_COLUMNS` at the top level.
    # `bytes` stays a hard reject: raw binary is never text.
    _AMBIGUOUS_MEDIA_KEYS = frozenset(("path", "url"))

    # Top-level column names that only ever point at media. A pretokenized VLM
    # set often keeps its media as a plain URL/path string beside `input_ids`,
    # and a string value alone looks like text, so the name has to say so.
    _MEDIA_COLUMNS = frozenset((
        "image_url", "image_urls", "img_url", "img_urls", "image_link",
        "video_url", "video_urls", "audio_url", "audio_urls",
        "input_image", "input_images", "input_video", "input_videos",
        "input_audio", "input_audios",
        "image_path", "image_paths", "img_path", "img_paths",
        "video_path", "video_paths", "audio_path", "audio_paths",
        "image_file", "image_files", "video_file", "video_files",
        "audio_file", "audio_files", "image_bytes",
        # `*_filename` is as common a spelling as `*_path`, and only `image_`
        # was here: a `video_filename`/`audio_filename` column matched neither
        # list, so its plain string schema called the split text-only.
        "image_filename", "image_filenames", "video_filename", "video_filenames",
        "audio_filename", "audio_filenames",
        # Bare names, which `_MEDIA_KEYS` already treats as unambiguous media one
        # level down. A pretokenized set storing "cat.jpg" in a plain `img` column
        # looked like text on schema alone, so the value was never examined and the
        # images were dropped before training.
        "img", "imgs", "image", "images", "video", "videos", "audio", "audios",
        # `picture`/`photo` name an image as unambiguously as `image` does, and
        # neither list had them: a pretokenized set keeping "cat.jpg" under
        # `picture` read as text on schema alone and lost its images.
        "picture", "pictures", "photo", "photos",
        # `bytes` is already an unambiguous media key one level down, and the
        # top-level list did not have it: a flattened base64 payload in a
        # `bytes` column has a string schema, so it read as text and the
        # replacement collator dropped it.
        "bytes",
    ))

    # A name that only ever points at media points at media nested too: a turn
    # storing `{"image_path": "cat.jpg"}` is exactly the top-level column moved
    # one level down, and a nested string is what `_is_plain_text` calls text.
    _MEDIA_KEYS = _MEDIA_KEYS | _MEDIA_COLUMNS

    # Names that are media half the time and an ordinary text field the other
    # half (`path` is a source file, `url` a citation), so refusing on the name
    # would refuse good text runs. Ask the value instead.
    _AMBIGUOUS_MEDIA_COLUMNS = frozenset((
        "path", "paths", "url", "urls", "uri", "file", "files",
        "file_path", "filepath", "file_name", "filename", "media", "source_url",
    ))

    # Those names are ambiguous nested too: `{"file_path": "images/cat.jpg"}` in a
    # turn or a `meta` struct is the top-level column moved one level down, and
    # only `path`/`url` were value-scanned there, so every other spelling was
    # called text and its media dropped.
    _AMBIGUOUS_MEDIA_KEYS = _AMBIGUOUS_MEDIA_KEYS | _AMBIGUOUS_MEDIA_COLUMNS

    # Keys whose dtype cannot settle the column. `type` joins the ambiguous
    # media names because the chat-part convention puts the modality in the
    # value: `{"type": "image", "content": "cat.jpg"}` is all `string`.
    _VALUE_SCAN_KEYS = _AMBIGUOUS_MEDIA_KEYS | frozenset(("type",))


    def _feature_holds_strings(feature, _depth = 0):
        """True when this column's values are strings, or lists of them."""
        if feature is None or _depth >= 8: return False
        if isinstance(feature, (list, tuple)):
            return bool(feature) and \
                all(_feature_holds_strings(f, _depth + 1) for f in feature)
        inner = getattr(feature, "feature", None)        # Sequence/List/LargeList
        if inner is not None: return _feature_holds_strings(inner, _depth + 1)
        dtype = getattr(feature, "dtype", None)          # Value/ClassLabel
        return isinstance(dtype, str) and dtype.endswith("string")
    pass

    def _string_column_batches(split, name):
        """The whole column in batches, or None when it must not be scanned.

        Reading a `datasets.Image`/audio column decodes every row (minutes and
        gigabytes on a real VLM set) to learn what its dtype already says, and
        `_columns_are_provably_text` refuses such a column anyway, so only ever
        scan a column the schema calls a string.
        """
        features = getattr(split, "features", None)
        if not features: return None                     # an unresolved stream
        try:
            if not _feature_holds_strings(features.get(name)): return None
            len(split)                                   # map-style only
            return split.select_columns([name]).iter(batch_size = 1000)
        except Exception:
            return None
    pass

    def _column_holds_strings(split, name):
        """Whether the schema says this column's values are strings."""
        features = getattr(split, "features", None)
        if not features: return False
        try:
            return _feature_holds_strings(features.get(name))
        except Exception:
            return False
    pass

    # Ambiguous string columns that could not be read past the sample. Named in
    # the refusal below, since dropping the column is the fix when it is text.
    _unscannable_media_columns = set()

    def _ambiguous_column_holds_media(split, name, rows):
        """Whether an ambiguous column points at media, over EVERY row when the
        split can be scanned: the 16-row sample misses a `cat.jpg` on row 5, and
        its dtype is `string` either way, so the schema cannot answer for it.
        Reading one string column is cheap - no image is ever decoded.
        """
        batches = _string_column_batches(split, name)
        if batches is None:
            # Not strings, so the name alone cannot make the column media.
            if not _column_holds_strings(split, name):
                return any(_looks_like_media_value(row.get(name)) for row in rows)
            # Strings this split will not hand over in full (streaming, or a
            # column a custom transform owns). The sample cannot speak for the
            # rows it never reads, and guessing "text" drops a media column
            # silently, so refuse instead.
            _unscannable_media_columns.add(name)
            return True
        try:
            for batch in batches:
                if any(_looks_like_media_value(v) for v in batch[name]): return True
            return False
        except Exception:
            _unscannable_media_columns.add(name)
            return True
    pass

    def _column_is_all_strings(split, name):
        """Whether EVERY row of a text column really holds a string.

        `Value('string')` is nullable, so a `None` past the sample keeps the
        dtype and would reach the plain tokenizer, crashing the run mid-`map`
        instead of being refused here.
        """
        batches = _string_column_batches(split, name)
        if batches is None: return True                  # the sample decided
        try:
            for batch in batches:
                if not all(isinstance(v, str) for v in batch[name]): return False
            return True
        except Exception:
            # The scan blew up partway (a custom transform needing the columns
            # `select_columns` just removed), so only the sampled rows were ever
            # checked and the rest are unproven. Refuse, exactly as the ambiguous
            # media scan does, rather than call the failure a proof.
            _unscannable_media_columns.add(name)
            return False
    pass

    def _has_media_column(names, rows, split = None):
        """True when a top-level column points at media the text path would drop."""
        names = [n for n in names if isinstance(n, str) and n not in _TEXT_COLUMNS]
        if any(n.lower() in _MEDIA_COLUMNS for n in names): return True
        # An ambiguous name costs a scan, so only reach it if no name settled it.
        return any(n.lower() in _AMBIGUOUS_MEDIA_COLUMNS and
                   _ambiguous_column_holds_media(split, n, rows) for n in names)
    pass

    def _is_plain_text(value, _depth = 0):
        """True only for text/scalars and nests of them.

        A column name says nothing about its contents: `messages` holds a list of
        turns whose content can be inline image parts, and dropping that column
        would drop the images. So anything the text path cannot encode -- a dict
        naming media, a PIL image, bytes -- is not plain text.
        """
        # A `data:` URI names its own media type, so unlike an extensionless
        # http URL there is nothing to weigh: it is an image/video/audio payload
        # wherever it turns up, including a column no name list covers.
        if isinstance(value, str):
            # Suffix as well as data URI: `attachments = ["cat.jpg"]` is a media
            # reference the text path cannot encode, and a scan that only weighed
            # `data:` called the column proven and dropped it.
            return not _looks_like_media_value(value)
        if value is None: return True
        if isinstance(value, (bool, int, float)): return True
        if _depth >= 6: return False
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str): return False
                lower = key.lower()
                if lower in _AMBIGUOUS_MEDIA_KEYS:
                    if _looks_like_media_value(item): return False
                elif lower in _MEDIA_KEYS: return False
                if not _is_plain_text(item, _depth + 1): return False
            kind = value.get("type")
            if isinstance(kind, str) and kind.lower() in _MEDIA_KEYS: return False
            return True
        if isinstance(value, (list, tuple)):
            return all(_is_plain_text(v, _depth + 1) for v in value)
        return False
    pass

    def _configured_text_field():
        return getattr(getattr(trainer, "args", None),
                       "dataset_text_field", None) or "text"

    def _row_is_plain_text(row, schema_proven = frozenset()):
        # Tokenizer/model columns are numeric by construction; the rest is what
        # can smuggle in images. A column the schema already proved needs no
        # second opinion: it judged EVERY row, where re-reading it as a value is
        # strictly weaker and misreads the tensor/ndarray `with_format("torch")`
        # and `with_format("numpy")` hand back for an ordinary numeric column.
        # The configured text field, when it really is a bare string, is prose by
        # definition and is exempt from the filename-suffix test. Otherwise an
        # assistant answer that happens to read `cat.jpg` classified the whole
        # column as a media reference and the text bypass was refused with the
        # vision-collator error, for a split the tokenizer handles fine. Only a
        # bare string: a `messages`-style field holding a list of turns can still
        # carry inline image parts, so that one is scanned as before.
        field = _configured_text_field()
        return all(_is_plain_text(v) for k, v in row.items()
                   if k not in _TEXT_COLUMNS and k not in schema_proven
                   and not (k == field and isinstance(v, str)))
    pass

    # A one-row peek calls a mixed split text-only: row 0 holds plain `messages`
    # while a later row hides an inline image. Sample a small fixed number of rows
    # instead - bounded, so a huge or streaming split stays cheap.
    _SCAN_ROWS = 16

    def _sample_rows(split):
        try: n = len(split)
        except Exception: n = None
        if n is None:
            # Streaming: a prefix is all that can be read without consuming it.
            return list(_itertools.islice(iter(split), _SCAN_ROWS))
        if n == 0: return []
        # Spread the sample over the whole split, first and last row included;
        # random access is cheap on Arrow.
        if n <= _SCAN_ROWS: idx = range(n)
        else: idx = sorted({i * (n - 1) // (_SCAN_ROWS - 1) for i in range(_SCAN_ROWS)})
        return [split[i] for i in idx]
    pass

    # A sample of any fixed size is guesswork: a 200-row split reads 16 positions,
    # so an image on row 5 goes unseen and the strip below drops it. Arrow types
    # are uniform down a column, so the schema answers for EVERY row at once, and
    # reading it is constant time - where scanning one `datasets.Image` column
    # exhaustively decodes each image (~1ms/row, ~94s per 100k rows).
    _PLAIN_DTYPES = ("string", "large_string", "bool", "null", "int", "uint",
                     "float", "double", "decimal", "date", "time", "duration",
                     "timestamp")

    def _leaf_dtype_is_float(feature, _depth = 0):
        """Does this feature bottom out in a float dtype?"""
        if feature is None or _depth >= 8: return False
        inner = getattr(feature, "feature", None)
        if inner is not None: return _leaf_dtype_is_float(inner, _depth + 1)
        dtype = getattr(feature, "dtype", None)
        return isinstance(dtype, str) and dtype.startswith(
            ("float", "double", "half", "bfloat"))

    def _leaf_dtype_is_narrow_int(feature, _depth = 0):
        """Does this feature bottom out in a sub-32-bit integer?

        That is a raw buffer, not tokens: int16 PCM, uint8 pixels. Token ids
        arrive as int32/int64 from every tokenizer, and the columns they land in
        are exempted by `_TEXT_COLUMNS` before this is ever asked.
        """
        if feature is None or _depth >= 8: return False
        inner = getattr(feature, "feature", None)
        if inner is not None: return _leaf_dtype_is_narrow_int(inner, _depth + 1)
        dtype = getattr(feature, "dtype", None)
        return isinstance(dtype, str) and dtype.startswith(
            ("int8", "int16", "uint8", "uint16"))

    # Column names that hold raw audio samples. 32-bit PCM is as much a waveform
    # as the 16-bit kind, so width cannot settle it and the name has to: the
    # wide-int allowance below exists for pretokenized token ids, and these are
    # not that.
    # Columns whose NAME says base64 media, whatever the value looks like. A bare
    # payload carries no `data:` prefix, so the value check cannot see it, and
    # base64-decoding to sniff magic bytes is neither cheap nor reliable. The
    # name is the evidence here, the same trade `_RAW_SAMPLE_COLUMNS` makes for
    # PCM.
    _BASE64_MEDIA_COLUMNS = frozenset((
        "image_base64", "img_base64", "image_b64", "img_b64", "image_data",
        "audio_base64", "audio_b64", "video_base64", "video_b64",
        "images_base64", "image_bytes", "audio_bytes", "video_bytes",
    ))

    _RAW_SAMPLE_COLUMNS = frozenset((
        "speech", "speeches", "waveform", "waveforms", "pcm",
        "raw_audio", "raw_speech", "audio_array", "audio_arrays",
        "audio_values", "speech_values",
    ))

    def _leaf_dtype_is_numeric(feature, _depth = 0):
        """Does this feature bottom out in any number at all?"""
        if feature is None or _depth >= 8: return False
        inner = getattr(feature, "feature", None)
        if inner is not None: return _leaf_dtype_is_numeric(inner, _depth + 1)
        dtype = getattr(feature, "dtype", None)
        return isinstance(dtype, str) and dtype.startswith(
            ("int", "uint", "float", "double", "half", "bfloat"))

    def _is_numeric_array_feature(feature):
        """`Array2D`/`Array3D`/`Array4D`/`Array5D`, by shape rather than name."""
        return (getattr(feature, "shape", None) is not None
                and isinstance(getattr(feature, "dtype", None), str))

    def _feature_is_plain_text(feature, _depth = 0, _seq_depth = 0):
        """True only when this column type cannot hold media in any row.

        `_seq_depth` counts sequence nesting only, so a struct holding a token
        column stays at the depth its sequence gives it.
        """
        if feature is None or _depth >= 8: return False
        if isinstance(feature, dict):                       # struct
            for key, value in feature.items():
                if not isinstance(key, str): return False
                lower = key.lower()
                # An ambiguous key is settled by the value scan below, not by name.
                if lower in _MEDIA_KEYS and lower not in _AMBIGUOUS_MEDIA_KEYS:
                    return False
                if not _feature_is_plain_text(value, _depth + 1, _seq_depth):
                    return False
            return True
        if isinstance(feature, (list, tuple)):
            return all(_feature_is_plain_text(f, _depth + 1, _seq_depth + 1)
                       for f in feature)
        # A multi-dimensional numeric block is a tensor, whatever the column is
        # called: `Array4D(float32)` under `frames` is video, and the leaf dtype
        # check below called it plain, so the column was marked proven, its
        # values never read, and the media dropped.
        if _is_numeric_array_feature(feature): return False
        inner = getattr(feature, "feature", None)           # Sequence/List/LargeList
        if inner is not None:
            # Same for a float sequence: `Sequence(float32)` under `speech` is a
            # waveform, and `Sequence(int16)` under the same name is the same
            # waveform stored as PCM. Wider integer sequences stay plain -- that
            # is the shape every pretokenized column has, and refusing them would
            # refuse the case this bypass exists for.
            if _leaf_dtype_is_float(inner): return False
            if _leaf_dtype_is_narrow_int(inner): return False
            return _feature_is_plain_text(inner, _depth + 1, _seq_depth + 1)
        # Value/ClassLabel name a primitive dtype. Image is "PIL.Image.Image" and
        # Audio/Video a dict, so anything unrecognised stays unsafe.
        dtype = getattr(feature, "dtype", None)
        if not isinstance(dtype, str): return False
        # A sequence OF sequences of integers is a numeric block, not tokens:
        # `Sequence(Sequence(int64))` under `frames` is quantised video, and the
        # wide-int allowance above waved it through on its leaf dtype alone,
        # marking the column proven so its values were never read. A pretokenized
        # column reaches its integers one sequence down; anything deeper is a
        # tensor by shape, exactly as `Array2D` is above.
        if _seq_depth >= 2 and dtype.startswith(("int", "uint")): return False
        return dtype.startswith(_PLAIN_DTYPES)
    pass

    def _feature_is_bare_string(feature, _depth = 0):
        """A `Value("string")`/`large_string`, bare or in a list of them.

        `Sequence(Value("string"))` too: `attachments = ["cat.jpg"]` advertises
        nothing in its dtype, so declaring it proven text skipped the value scan
        and the column was dropped, training silently without the media.
        """
        inner = getattr(feature, "feature", None)
        if inner is not None:
            return _depth < 3 and _feature_is_bare_string(inner, _depth + 1)
        if isinstance(feature, (list, tuple)):
            return (_depth < 3 and len(feature) == 1
                    and _feature_is_bare_string(feature[0], _depth + 1))
        dtype = getattr(feature, "dtype", None)
        return isinstance(dtype, str) and dtype.startswith(("string", "large_string"))

    def _feature_needs_a_value_scan(feature, _depth = 0):
        """Whether the dtypes alone cannot settle this column.

        Two all-string shapes decide at value level. A nested `path`/`url` is
        `string` for both a media reference and ordinary provenance. So is a
        `type` tag: `{"type": "image", "content": "cat.jpg"}` is an image part
        and `{"type": "text", ...}` is not, and calling the schema plain marked
        the column proven, which skips the tag check in `_row_is_plain_text` and
        drops the media silently.
        """
        if feature is None or _depth >= 8: return False
        if isinstance(feature, dict):
            return any(isinstance(k, str) and (k.lower() in _VALUE_SCAN_KEYS or
                                               _feature_needs_a_value_scan(v, _depth + 1))
                       for k, v in feature.items())
        if isinstance(feature, (list, tuple)):
            return any(_feature_needs_a_value_scan(f, _depth + 1) for f in feature)
        inner = getattr(feature, "feature", None)           # Sequence/List/LargeList
        if inner is not None: return _feature_needs_a_value_scan(inner, _depth + 1)
        return False
    pass

    def _column_values_are_plain_text(split, name):
        """Whether EVERY row of a struct the schema could not settle is text.

        The 16-row sample misses a `cat.jpg` on row 5000 and the schema says
        `string` either way, so read the whole column - the same trade the
        top-level ambiguous columns already make. Callers gate on
        `_feature_is_plain_text` first, so no Image/Audio column is ever decoded.
        """
        try:
            len(split)                                   # map-style only
            batches = split.select_columns([name]).iter(batch_size = 1000)
            for batch in batches:
                if not all(_is_plain_text(v) for v in batch[name]): return False
            return True
        except Exception:
            # A stream cannot speak for the rows it never hands over; guessing
            # "text" would drop a media column silently, so refuse instead.
            _unscannable_media_columns.add(name)
            return False
    pass

    def _is_map_style(split):
        """Can this split be indexed and scanned, or is it a one-pass stream?"""
        try:
            len(split)
            return True
        except Exception:
            return False
    pass

    def _columns_are_provably_text(split, names):
        """`(every column is text, the columns the schema itself proved)`.

        The second half is threaded down to `_row_is_plain_text`, which must not
        re-judge a column the schema has already settled for every row.
        """
        features = getattr(split, "features", None)
        if features:
            proven = set()
            try:
                for name, feature in features.items():
                    if name in _TEXT_COLUMNS: continue
                    # Name plus shape, because `_feature_is_plain_text` never sees
                    # the column name and int32 PCM is indistinguishable from token
                    # ids by dtype alone.
                    if str(name).lower() in _BASE64_MEDIA_COLUMNS:
                        return False, proven
                    if str(name).lower() in _RAW_SAMPLE_COLUMNS and \
                        getattr(feature, "feature", None) is not None and \
                        _leaf_dtype_is_numeric(feature): return False, proven
                    if not _feature_is_plain_text(feature): return False, proven
                    # Bare strings too, not just the structs the schema cannot
                    # settle. A `data:image/...` URI identifies itself, and an
                    # unlisted column holding one on row 5000 is invisible to the
                    # 16 sampled rows that were the only value check a bare
                    # string got. The scan is a `startswith` per row.
                    #
                    # Map-style only, unlike the struct case. A struct advertises
                    # its own ambiguity with a `path`/`url`/`type` key, so a
                    # stream that cannot be scanned is right to be refused; a
                    # bare `string` advertises nothing, and refusing it would
                    # refuse every streamed text column there is.
                    _scan = _feature_needs_a_value_scan(feature) or (
                        _feature_is_bare_string(feature) and _is_map_style(split))
                    # Never the configured text field, when its schema says bare
                    # string. Its contents are prose the tokenizer will encode,
                    # so weighing them for filename suffixes refused a whole
                    # split over an assistant answer that read `cat.jpg`. The
                    # caller still proves the column is all strings separately;
                    # a STRUCTURED text field is not exempt and is scanned here
                    # as before, because a list of turns can hold image parts.
                    if name == _configured_text_field() and _feature_is_bare_string(feature):
                        _scan = False
                    if _scan and not _column_values_are_plain_text(split, name):
                        return False, proven
                    # A plain string column is NOT added: `proven` exists so a
                    # numeric column that `with_format` hands back as a tensor is
                    # not re-judged by value, and a string stays a string under
                    # every format. Leaving it out costs nothing -- the sampled
                    # rows are read either way -- and is what lets a `data:` URI
                    # in an unlisted column be seen at all.
                    if not _feature_is_bare_string(feature): proven.add(name)
                return True, proven
            except Exception:
                return False, proven
        # No schema (an unresolved stream): a sample cannot prove what the rows it
        # never reads hold, so trust only the tokenizer's own columns.
        return not (set(names) - _TEXT_COLUMNS), set()
    pass

    def _has_custom_transform(split):
        """Does this split rewrite its rows on the way out?

        `with_format("torch")`/`("numpy")` re-type a column and keep its meaning,
        which is why the schema is allowed to speak for every row. `with_transform`
        does not: a `Value("string")` column can be decoded into a PIL image at
        read time, so the schema describes the storage and not what the collator
        will see. Drop the proof and let the rows be judged on their values.
        """
        try:
            fmt = getattr(split, "format", None)
            kind = fmt.get("type") if isinstance(fmt, dict) else None
            return (kind or getattr(split, "_format_type", None)) == "custom"
        except Exception:
            return False
    pass

    def _split_views(dataset):
        """`(column names, sampled rows, provably text, schema-proven, split)` per split.

        `iter()` restarts a datasets IterableDataset, so peeking rows does not
        consume the stream (this is how `_maybe_tokenize_dataset` peeks too).
        """
        splits = list(dataset.values()) if isinstance(dataset, dict) else [dataset]
        views = []
        for split in splits:
            if split is None: continue
            names = getattr(split, "column_names", None)
            if isinstance(names, dict): names = None
            rows = []
            try:
                rows = [r for r in _sample_rows(split) if isinstance(r, dict)]
            except Exception:
                if not names: raise
                rows = []
            if not names:
                if not rows: raise ValueError("Unsloth: cannot read the dataset columns")
                names = list(rows[0].keys())
            provable, proven = _columns_are_provably_text(split, names)
            # `provable` too, not just `proven`. Clearing the proof alone still
            # left the schema saying "no media in storage", and that is only
            # checked against the 16 sampled rows: a transform that decodes an
            # image at row 5000 passed both and the column was dropped. The
            # whole-column scan cannot settle it either, since it reads through
            # `select_columns`, which a transform needing the other columns
            # breaks. So refuse, the way an unscannable stream already is -- and
            # name the columns, which is what makes that refusal actionable.
            if _has_custom_transform(split):
                provable, proven = False, set()
                _unscannable_media_columns.update(
                    n for n in names
                    if isinstance(n, str) and n not in _TEXT_COLUMNS)
            views.append((set(names), rows, provable, proven, split))
        return views
    pass

    def _dataset_is_pretokenized(dataset):
        """True when rows carry `input_ids` and nothing but plain text beside them,
        i.e. text tokenized up front where dataset-level masking is correct.
        Multimodal or unreadable rows return False, keeping the caller's old refusal
        (the text path swaps in a text collator and would drop the image handling).
        """
        if dataset is None:
            return False
        try:
            views = _split_views(dataset)
        except Exception:
            return False
        if not views: return False
        for names, rows, provable, proven, split in views:
            if "input_ids" not in names: return False
            if not names.isdisjoint(_MULTIMODAL_COLUMNS): return False
            if _has_media_column(names, rows, split): return False
            # Columns look text-only, so check the values too: a `messages` column
            # can carry inline images that the strip below would throw away.
            if not provable: return False
            for row in rows:
                if not _row_is_plain_text(row, proven): return False
        return True
    pass

    def _split_is_raw_text_only(dataset):
        """True when a split carries no `input_ids` but a real string column.

        Applies to train and eval alike: proving every row of the text column is
        a string IS the evidence that the run is text-only, and it does not get
        stronger by the split's name. `_maybe_tokenize_dataset` below tokenizes
        such a split with the same text tokenizer. The column has to hold
        strings, not conversations: a list of turns needs a chat template, and
        its content can be inline images.
        """
        if dataset is None:
            return False
        text_field = getattr(getattr(trainer, "args", None), "dataset_text_field", None) or "text"
        try:
            views = _split_views(dataset)
        except Exception:
            return False
        if not views: return False
        for names, rows, provable, proven, split in views:
            if "input_ids" in names: return False
            if not names.isdisjoint(_MULTIMODAL_COLUMNS): return False
            if _has_media_column(names, rows, split): return False
            # The column `_maybe_tokenize_dataset` would actually read.
            field = text_field if text_field in names else "text"
            if field not in names: return False
            if not provable: return False
            for row in rows:
                if not isinstance(row.get(field), str): return False
                if not _row_is_plain_text(row, proven): return False
            # The sample says these rows are text; the tokenizer reads every row.
            if not _column_is_all_strings(split, field): return False
        return True
    pass

    # Classified up here rather than beside the swap below, so the packing
    # refusal can run before either split is tokenized: it is a deterministic
    # configuration error, and raising it after a full preprocessing pass makes
    # a large corpus pay for the answer twice and leaves the datasets mutated.
    from transformers import DataCollatorForSeq2Seq, DataCollatorWithPadding
    import transformers as _transformers
    _PAD_DELEGATING_COLLATORS = tuple(_cls for _cls in (
        DataCollatorForSeq2Seq, DataCollatorWithPadding,
        getattr(_transformers, "DataCollatorForTokenClassification", None),
        getattr(_transformers, "DataCollatorForLanguageModeling", None),
        getattr(_transformers, "DataCollatorForMultipleChoice", None),
    ) if isinstance(_cls, type))
    # Read defensively: this now runs ahead of the early returns below, and a
    # trainer without `.args` used to reach one of them.
    packing_enabled = getattr(getattr(trainer, "args", None), "packing", False)

    def _pads_through_a_processor(collator):
        """A pad-delegating collator holding a processor, which has no `.pad`."""
        source = getattr(collator, "tokenizer", None)
        if source is None: source = getattr(collator, "processor", None)
        return (source is not None and not hasattr(source, "pad")
                and isinstance(collator, _PAD_DELEGATING_COLLATORS))

    def _is_known_bypassed_collator(collator):
        if isinstance(collator, _PAD_DELEGATING_COLLATORS): return True
        # TRL's vision collator rebuilds labels through its processor. Matched by
        # name so this does not depend on which TRL version is installed.
        return any(b.__name__ == "DataCollatorForVisionLanguageModeling"
                   for b in type(collator).__mro__)

    def _refuse_packing_that_will_not_happen(collator, raw_splits):
        """Nothing that takes this bypass gets packed.

        A raw split is tokenized row by row, so nothing concatenates the
        examples. A pretokenized one is no better off: either the rows were
        never packed, or they were and the plain DataCollatorForSeq2Seq the swap
        installs drops the `seq_lengths` the packed batch needs to rebuild
        `position_ids`, silently letting the packed examples attend to each
        other. That holds for the collators exempted just above, and for a
        packing subclass of one, because the exemption is about who pads.
        """
        if not packing_enabled: return
        # Built here, not by the caller: classifying every split costs whole
        # column scans, and with packing off the return above threw all of it
        # away. `predict`/`evaluate` reach this guard on each call.
        if callable(raw_splits): raw_splits = raw_splits()
        _how = ("tokenizes each row on its own" if raw_splits else
                "cannot rebuild the packed batch's `position_ids`")
        raise ValueError(
            f"Unsloth: `packing = True` is not supported here. `{type(collator).__name__}` "
            "holds a processor, so response-only masking is applied at the dataset level, "
            f"and that path {_how} -- packing would be silently dropped. Turn packing off, or "
            "build UnslothVisionDataCollator(..., train_on_responses_only = True, "
            "instruction_part = ..., response_part = ...) so masking runs at collate time."
        )

    def _refuse_packing_with_a_foreign_collator(collator):
        """The case with no right answer: `_is_vision_collator` matches one
        merely holding a processor, and a custom self-packing collator does
        exactly that. Replacing it discards its packing, its `position_ids` and
        any block-attention inputs; keeping it risks its `__call__` rebuilding
        `labels` over the mask. Both are silently wrong, so say so."""
        if not packing_enabled or _is_known_bypassed_collator(collator): return
        raise ValueError(
            f"Unsloth: `{type(collator).__name__}` holds a processor and does not support "
            "response-only masking, and `packing = True` asks that same collator to build the "
            "packed batch. Both cannot be honoured: replacing it discards its packing, its "
            "`position_ids` and any block-attention inputs, while keeping it risks its "
            "`__call__` rebuilding `labels` over the mask just written. Turn packing off, or "
            "build UnslothVisionDataCollator(..., train_on_responses_only = True, "
            "instruction_part = ..., response_part = ...) so masking runs at collate time."
        )

    # Set when a foreign vision collator is let through to the text path below.
    _bypassed_vision_collator = False
    data_collator = getattr(trainer, "data_collator", None)
    if _is_vision_collator(data_collator):
        masking = getattr(data_collator, "train_on_responses_only", None)
        if callable(masking):
            return trainer  # collator already masks responses; nothing to do
        is_unsloth = any(b.__name__ == "UnslothVisionDataCollator" for b in type(data_collator).__mro__)
        if not is_unsloth:
            # A text-only run on a multimodal model still carries a processor as its
            # `tokenizer`, so a plain text collator trips `_is_vision_collator`.
            # Discriminate on the data instead, over every split that gets collated:
            # the swap below is trainer-wide, so a train-only check would strip a
            # multimodal eval set of its image handling.
            _eval = getattr(trainer, "eval_dataset", None)
            _eval_splits = list(_eval.values()) if isinstance(_eval, dict) else [_eval]
            _train = getattr(trainer, "train_dataset", None)
            # The train split gets the same allowance as an eval one. Requiring
            # it to be PRE-tokenized refused the common case: TRL 0.22.2 hands
            # a plain text SFT on a multimodal checkpoint its own
            # `DataCollatorForVisionLanguageModeling` and leaves the dataset at
            # `["text"]`, tokenizing inside the collator. Nothing there is a
            # vision run, and `_split_is_raw_text_only` proves it over every
            # row, so refusing lost Gemma3_(4B), Gemma3N_(4B)-Conversational,
            # Gemma3_(27B)_A100 and Qwen_3_5_27B for a masking pass that is
            # exactly what the dataset path below does correctly.
            def _collatable(d):
                return _dataset_is_pretokenized(d) or _split_is_raw_text_only(d)
            if not (
                (_train is None or _collatable(_train))
                and all(_collatable(d) for d in _eval_splits if d is not None)
            ):
                # Cannot configure this collator, so refuse rather than silently
                # return with responses left unmasked.
                _hint = ""
                if _unscannable_media_columns:
                    _hint = (
                        f" Column(s) {sorted(_unscannable_media_columns)} may point at "
                        "images/videos/audio and this split cannot be read past its first "
                        "rows to tell, so they are assumed to be media - drop them with "
                        "`dataset.remove_columns([...])` if they are ordinary text."
                    )
                raise ValueError(
                    "Unsloth: Detected a vision data collator that does not support response-only "
                    "masking. Build UnslothVisionDataCollator(..., train_on_responses_only = True, "
                    "instruction_part = ..., response_part = ...) so masking runs at collate time."
                    + _hint
                )
            # Fall through to the dataset-level text path below.
            _refuse_packing_with_a_foreign_collator(data_collator)
            _refuse_packing_that_will_not_happen(
                data_collator,
                lambda: [d for d in [_train] + _eval_splits
                         if d is not None and not _dataset_is_pretokenized(d)])
            _bypassed_vision_collator = True
            print(
                f"Unsloth: `{type(data_collator).__name__}` holds a processor but the "
                "dataset is already tokenized, so response-only masking is applied at "
                "the dataset level (image handling is untouched)."
            )
        else:
            # Parts already on the collator's tokenizer: let the nested call read
            # them, since passing them again hits the "already set" guard.
            coll_proc = getattr(data_collator, "processor", tokenizer)
            coll_tok = coll_proc.tokenizer if hasattr(coll_proc, "tokenizer") else coll_proc
            parts = {} if hasattr(coll_tok, "_unsloth_input_part") else \
                dict(instruction_part = instruction_part, response_part = response_part)
            data_collator.train_on_responses_only = train_on_responses_only(
                None,
                force_match        = force_match,
                tokenizer          = coll_proc,
                return_function    = True,
                last_response_only = last_response_only,
                **parts,
            )
            print(f"Unsloth: Enabled response-only masking on your {type(data_collator).__name__} (image handling kept intact).")
            return trainer
    pass

    # The other route to the same replacement: a pad-delegating collator holding
    # a processor is rebuilt around the text tokenizer below, which packs no more
    # than the swap above does. Refused here, before either split is mapped, and
    # after the block above so a collator that masks for itself is left alone.
    if _pads_through_a_processor(getattr(trainer, "data_collator", None)):
        _refuse_packing_that_will_not_happen(trainer.data_collator, None)

    if hasattr(trainer, "train_dataset") and trainer.train_dataset is not None:
        if not hasattr(trainer.train_dataset, "map"):
            raise TypeError("Unsloth: train_on_responses_only does not work on lists!")
        trainer.train_dataset = _maybe_tokenize_dataset(trainer.train_dataset)
        if isinstance(trainer.train_dataset, IterableDataset):
            trainer.train_dataset = trainer.train_dataset.map(_train_on_responses_only, batch_size = _iterable_batch_size(trainer.train_dataset), batched = True)
        else:
            trainer.train_dataset = trainer.train_dataset.map(_train_on_responses_only, batched = True, num_proc = _effective_num_proc(trainer.train_dataset))
        trainer.train_dataset = _filter_fully_masked(trainer.train_dataset, "train_dataset")
    pass

    if hasattr(trainer, "eval_dataset") and trainer.eval_dataset is not None:
        # Eval datasets could be a dict! DatasetDict subclasses dict, so match on
        # isinstance: `type(...) is dict` sent it down the single-dataset path,
        # where column_names is a dict of splits and every per-split step no-ops.
        if isinstance(trainer.eval_dataset, dict):
            for key, value in trainer.eval_dataset.items():
                if not hasattr(value, "map"):
                    raise TypeError("Unsloth: train_on_responses_only does not work on lists!")
                value = _maybe_tokenize_dataset(value)
                if isinstance(value, IterableDataset):
                    trainer.eval_dataset[key] = value.map(_train_on_responses_only, batch_size = _iterable_batch_size(value), batched = True)
                else:
                    trainer.eval_dataset[key] = value.map(_train_on_responses_only, batched = True, num_proc = _effective_num_proc(value))
                trainer.eval_dataset[key] = _filter_fully_masked(trainer.eval_dataset[key], f"eval_dataset[{key}]")
        else:
            if not hasattr(trainer.eval_dataset, "map"):
                raise TypeError("Unsloth: train_on_responses_only does not work on lists!")
            trainer.eval_dataset = _maybe_tokenize_dataset(trainer.eval_dataset)
            if isinstance(trainer.eval_dataset, IterableDataset):
                trainer.eval_dataset = trainer.eval_dataset.map(_train_on_responses_only, batch_size = _iterable_batch_size(trainer.eval_dataset), batched = True)
            else:
                trainer.eval_dataset = trainer.eval_dataset.map(_train_on_responses_only, batched = True, num_proc = _effective_num_proc(trainer.eval_dataset))
            trainer.eval_dataset = _filter_fully_masked(trainer.eval_dataset, "eval_dataset")
        pass
    pass

    # Edit data collator to DataCollatorForSeq2Seq. Collators that rebuild labels
    # from a processor already returned above, so what is left here only pads.
    _collator = getattr(trainer, "data_collator", None)
    # A collator holding a processor (DataCollatorForSeq2Seq/WithPadding) pads
    # through a `.pad` processors do not have, so it dies on the first batch;
    # rebuild it around the unwrapped text tokenizer. A collator holding no
    # padding object at all stays untouched (TRL's packing
    # DataCollatorForLanguageModeling takes a bare pad_token_id).
    # Holding the object is not proof it is padded through: a custom collator can
    # keep a processor for its own use and batch (and pack) everything itself, and
    # replacing that one throws its packing away. So only the classes that provably
    # delegate to `.pad` are repaired from the attribute; TRL's vision collator
    # calls its processor instead and is already covered by the bypass flag below.
    _processor_backed = _pads_through_a_processor(_collator)
    # That repair is not packing-gated like the swap: it fails either way. The
    # bypassed-collator case under packing already refused above, before
    # anything was mapped.
    if hasattr(trainer, "data_collator") and (
        _processor_backed or _bypassed_vision_collator
        or (not isinstance(_collator, DataCollatorForSeq2Seq) and not packing_enabled)
    ):
        # Keep the caller's settings when only swapping the tokenizer on a seq2seq
        # collator; for any other class this is a replacement, not a swap, and its
        # same-named attributes need not mean the same thing.
        _same_class = _processor_backed and isinstance(_collator, DataCollatorForSeq2Seq)
        # These are handed to `tokenizer.pad` with the same meaning by every
        # pad-delegating collator, not just DataCollatorWithPadding: a
        # DataCollatorForTokenClassification carrying `padding = "max_length"` is
        # a separate class, not a subclass, so an isinstance check on that one
        # class dropped its settings and silently turned the run into a
        # dynamically padded one. Ask whether the fields are there.
        _padding_class = _processor_backed and not _same_class and any(
            hasattr(_collator, _n)
            for _n in ("padding", "max_length", "pad_to_multiple_of")
        )
        # `label_pad_token_id` means the same thing to DataCollatorForSeq2Seq as
        # it does to a DataCollatorForTokenClassification, so a caller who chose
        # a non-default one keeps it; dropping it padded unequal-length batches
        # with -100 instead and can break a loss that reads the pad value.
        _names = ("model", "padding", "max_length", "pad_to_multiple_of",
                  "label_pad_token_id", "return_tensors") if _same_class else \
                 ("padding", "max_length", "pad_to_multiple_of",
                  "label_pad_token_id", "return_tensors")
        _kept = {
            name: getattr(_collator, name)
            for name in _names
            if (_same_class or _padding_class) and hasattr(_collator, name)
        }
        _text_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, **_kept)
        # Only when a media-capable collator was displaced. Everywhere else the
        # replacement is the whole point and there is nothing to fall back to.
        # Only when the displaced collator can actually take a media batch. The
        # `_processor_backed` repair replaces a collator whose "tokenizer" is a
        # processor with no `.pad`, and that object is broken for EVERY batch,
        # not just text: keeping it as the fallback would route a later media
        # batch straight back into the `processor.pad` AttributeError this
        # repair exists to remove. The bypass flag is the case where the
        # displaced collator is a working vision collator.
        _media_capable = _bypassed_vision_collator and not _processor_backed
        # The training text column is NOT a companion to strip: it is what the
        # text collator is there to read.
        _text_field = getattr(getattr(trainer, "args", None),
                              "dataset_text_field", None) or "text"
        # Every media form the initial-split guard recognises, not just the two
        # sets: a later split storing `image_base64`, a `speech` waveform or an
        # ambiguous `path` matched nothing and went to the text collator.
        _dispatch_keys = (_MEDIA_COLUMNS | _MULTIMODAL_COLUMNS
                          | _BASE64_MEDIA_COLUMNS | _AMBIGUOUS_MEDIA_COLUMNS
                          | _RAW_SAMPLE_COLUMNS)
        trainer.data_collator = _MediaAwareCollator(
            # Both sets. `_MEDIA_COLUMNS` names what a user hands in, and a split
            # that has already been through the processor carries `pixel_values`
            # / `image_grid_thw` / `input_features` instead: those live only in
            # `_MULTIMODAL_COLUMNS`, so a processed `predict` batch matched
            # nothing and went to the text collator that cannot tensorize it.
            _text_collator, _collator, _dispatch_keys, _AMBIGUOUS_MEDIA_COLUMNS,
            _RAW_TEXT_COMPANION_COLUMNS - {_text_field},
        ) if _media_capable else _text_collator
        if _media_capable:
            # And the keys have to survive `remove_unused_columns`, which is on
            # by default. The trainer caches its signature columns from the
            # model's forward, so a later `evaluate`/`predict` split had its
            # media stripped BEFORE the collator ever ran and the dispatcher saw
            # text-only keys: the fallback it advertises could not fire at all.
            # Conversation keys too, though they are deliberately NOT dispatch
            # keys: `_has_media` matches them only when the value is a real
            # message list, and widening the dispatch set would drop that check.
            # Media in a raw VLM split can live only inside `messages`, so
            # without them the conversation was stripped before the dispatcher
            # ever saw it and the batch went to the text collator.
            _keep_media_columns(
                trainer,
                _dispatch_keys
                | _MediaAwareCollator._CONVERSATION_KEYS
                # The prompt that goes WITH the media: a raw `{"text": ...,
                # "image": ...}` row reached the vision collator with its text
                # already stripped, so it had nothing to tokenize.
                | _RAW_TEXT_COMPANION_COLUMNS
                | {_text_field},
            )

        # `tokenizer.pad(..., return_tensors = "pt")` stacks every key it is
        # handed, and only pads the few it knows, so any leftover column kills
        # the first batch: a raw `text`/`messages` cannot be tensorized, a
        # ragged `prompt_ids` is stacked unpadded, and a scalar `label` is taken
        # for the labels themselves. The trainer normally strips them, but not
        # when unused-column removal is off (token-type-id models above turn it
        # off). So keep only what the model is actually fed.
        import inspect as _inspect
        def _model_input_columns():
            # token_type_ids is why unused-column removal is off; the rest are
            # asked of the processor and the model so this cannot rot.
            names = {"input_ids", "attention_mask", "token_type_ids", "labels"}
            holders = [processor, tokenizer]
            holders += [getattr(processor, attr, None) for attr in
                        ("tokenizer", "image_processor", "feature_extractor",
                         "video_processor", "audio_processor")]
            for holder in holders:
                try: names.update(getattr(holder, "model_input_names", None) or ())
                except Exception: pass
            # Unwrap PEFT/compile wrappers: their own forward hides pixel_values.
            names.update(_model_forward_parameter_names(getattr(trainer, "model", None)))
            # `args.label_names` too, exactly as `_keep_media_columns` does and
            # as Trainer's own signature derivation does. Supervision consumed
            # by a custom `compute_loss` is not declared by `forward`, so the
            # signature kept it while THIS list deleted it from the split.
            names.update(getattr(getattr(trainer, "args", None), "label_names", None) or ())
            names.discard("self")
            # Same reason as the tokenizing strip above: a `forward` declaring
            # `text` must not keep a raw string column no collator can tensorize.
            names -= {getattr(getattr(trainer, "args", None),
                              "dataset_text_field", None) or "text", "text"}
            return names
        _keep_columns = _model_input_columns()
        _text_columns = {getattr(getattr(trainer, "args", None),
                                 "dataset_text_field", None) or "text", "text"}

        import numbers as _numbers
        def _is_tensorizable(value, _depth = 0):
            """Can `tokenizer.pad(..., return_tensors = 'pt')` stack this?

            It tensorizes every key it is handed, so under the opt-out a kept
            `messages`/`source`/string id fails the first batch before any
            custom `compute_loss` sees it. Numbers and nests of numbers survive,
            which is what a per-row `sample_weight` or auxiliary target is.
            """
            if _depth >= 6: return False
            # str before Sequence: it is one, of one-character strings, forever.
            if isinstance(value, (str, bytes, bytearray, dict)) or value is None:
                return False
            # Tensors and arrays answer at once, by their own dtype: iterating a
            # long one element-wise to reach the same answer is pure cost, and
            # `numbers` misses `np.int64`, which is not an `int` subclass.
            kind = getattr(getattr(value, "dtype", None), "kind", None)
            if kind is not None: return kind in "biufc"
            if isinstance(value, torch.Tensor): return True
            if isinstance(value, _numbers.Number): return True
            try: items = list(value)
            except Exception: return False
            return all(_is_tensorizable(v, _depth + 1) for v in items)

        def _untensorizable_columns(dataset):
            """Named from a sampled row, so a column is judged by what it holds.

            Nothing the model is fed is ever named here, however odd it looks:
            those columns are the collator's job, not a guess from one row.
            """
            try: row = next(iter(dataset), None)
            except Exception: return set()
            if not isinstance(row, dict): return set()
            return {k for k, v in row.items()
                    if k not in _keep_columns and not _is_tensorizable(v)}

        def _drop_raw_columns(dataset):
            if dataset is None or not hasattr(dataset, "remove_columns"): return dataset
            try:
                names = getattr(dataset, "column_names", None)
                if names is None: names = list(next(iter(dataset)).keys())
                if isinstance(names, dict): return dataset
                # The opt-out keeps the caller's own columns, but only the ones
                # `tokenizer.pad(..., return_tensors = "pt")` can actually
                # stack. It tensorizes every key it is handed, so a surviving
                # `text`/`messages`/`source` dies on the first batch, before the
                # custom `compute_loss` the opt-out exists for ever runs. A
                # numeric `sample_weight` or auxiliary target still comes through.
                if _keep_every_column:
                    unusable = _text_columns | _untensorizable_columns(dataset)
                    # `label` is a reserved collator alias, not an extra column:
                    # DataCollatorForSeq2Seq reads `"label" if "label" in
                    # features[0] else "labels"`, so a numeric `label` beside the
                    # response-only `labels` we just built wins, and the masks are
                    # thrown away. Only when both are present: a set carrying
                    # `label` alone is supervising with it.
                    if "labels" in names and "label" in names:
                        unusable = unusable | {"label"}
                    drop = [c for c in names if c in unusable]
                else:
                    drop = [c for c in names if c not in _keep_columns]
            except Exception:
                return dataset
            if not drop: return dataset
            print(f"Unsloth: Dropping columns the model is not fed: {sorted(drop)}")
            return dataset.remove_columns(drop)
        if hasattr(trainer, "train_dataset"):
            trainer.train_dataset = _drop_raw_columns(trainer.train_dataset)
        _eval_now = getattr(trainer, "eval_dataset", None)
        if isinstance(_eval_now, dict):
            for _key in list(_eval_now.keys()):
                _eval_now[_key] = _drop_raw_columns(_eval_now[_key])
        elif _eval_now is not None:
            trainer.eval_dataset = _drop_raw_columns(_eval_now)
    pass

    # Check if all labels randomnly got masked to nothing - maybe wrong chat template?
    # Eval-only trainers have no train split, and this check calls len() on it.
    from .training_utils import fix_zero_training_loss
    if getattr(trainer, "train_dataset", None) is not None:
        fix_zero_training_loss(None, tokenizer, trainer.train_dataset)
    return trainer
pass


def standardize_data_formats(
    dataset,
    tokenizer             = None,
    aliases_for_system    = ["system",],
    aliases_for_user      = ["user", "human", "input",],
    aliases_for_assistant = ["gpt", "assistant", "output",],
    batch_size            = 1000,
    num_proc              = None,
):
    """Standardize ShareGPT and similar formats to user/assistant HF format.

    The alias lists map source role names onto "system"/"user"/"assistant".
    """
    import collections
    import itertools

    # VLMs need list-valued content ([{"type": "text", "text": ...}]); text
    # models use a plain string.
    is_vlm = False
    if tokenizer is not None:
        if hasattr(tokenizer, "image_processor") or hasattr(tokenizer, "tokenizer"):
            is_vlm = True

    column_names = set(next(iter(dataset)).keys())
    if "conversations" not in column_names:
        return dataset

    examples = itertools.islice(dataset, 10)
    uniques = collections.defaultdict(list)
    for example in examples:
        for message in example["conversations"]:
            for key, value in message.items():
                if type(value) is not str:
                    raise RuntimeError("Unsloth: Cannot standardize non text datasets!")
                uniques[key].append(value)
    pass

    # Must be only 2 entries
    assert(len(uniques.keys()) == 2)

    keys = list(uniques.keys())
    length_first  = len(set(uniques[keys[0]]))
    length_second = len(set(uniques[keys[1]]))

    if length_first < length_second:
        # Role is assigned to the first element
        role_key    = keys[0]
        content_key = keys[1]
    else:
        role_key    = keys[1]
        content_key = keys[0]
    pass

    # Check roles are in aliases
    all_aliases = set(aliases_for_system + aliases_for_user + aliases_for_assistant)
    roles = set(uniques[role_key])
    leftover_aliases = (all_aliases | roles) - all_aliases
    if len(leftover_aliases) != 0:
        raise TypeError(
            f"Unsloth: {list(leftover_aliases)} are not in aliases. Please update aliases."
        )
    pass

    # Mapping for aliases
    aliases_mapping = {}
    for x in aliases_for_system:    aliases_mapping[x] = "system"
    for x in aliases_for_user:      aliases_mapping[x] = "user"
    for x in aliases_for_assistant: aliases_mapping[x] = "assistant"

    def _standardize_dataset(examples):
        convos = examples["conversations"]
        all_convos = []
        for convo in convos:
            new_convo = []
            for message in convo:
                role = aliases_mapping[message[role_key]]
                text = message[content_key]
                if is_vlm: text = [ {"type" : "text", "text" : text} ]
                x = {"role" : role, "content" : text}
                new_convo.append(x)
            pass
            all_convos.append(new_convo)
        pass
        return { "conversations" : all_convos, }
    pass

    dataset_map_kwargs = {
        'batched': True,
        'batch_size': batch_size,
    }

    if not isinstance(dataset, IterableDataset):
        # One policy, one place. The copy that used to live here read stdlib
        # multiprocessing's start method while datasets uses multiprocess, and it
        # fell back to num_proc = 1 under memory pressure -- which is a Pool(1)
        # on datasets >= 4.1, the pool this exists to avoid.
        from .dataset_num_proc import get_dataset_num_proc
        dataset_map_kwargs['num_proc'] = get_dataset_num_proc(num_proc)
        dataset_map_kwargs['desc'] = "Unsloth: Standardizing formats"

    return dataset.map(
        _standardize_dataset,
        **dataset_map_kwargs
    )
pass


from datasets import (Dataset, IterableDataset,)
try:
    from trl.trainer.utils import ConstantLengthDataset
except:
    # TRL 0.20.0 removes ConstantLengthDataset
    ConstantLengthDataset = None

# Faster SFTTrainer prepare_dataset
def sft_prepare_dataset(
    self,
    dataset: Union[Dataset, IterableDataset],
    processing_class,
    args,
    packing: bool,
    formatting_func: Optional[Callable[[dict], str]],
    dataset_name: str,
) -> Union[Dataset, IterableDataset]:
    # All Unsloth Zoo code licensed under LGPLv3
    try:
        if isinstance(dataset, ConstantLengthDataset): return dataset
    except:
        pass

    map_kwargs = {}
    use_desc = isinstance(dataset, Dataset)
    is_vlm = hasattr(processing_class, "tokenizer")
    tokenizer = processing_class
    if is_vlm: tokenizer = processing_class.tokenizer

    # Detect whether the model's module needs token_type_ids when training
    import sys as _sys
    _needs_token_type_ids = False
    # Split to avoid compiler substring match on masking_utils names
    _ccm = 'create_' + 'causal_mask_mapping'
    _model = getattr(self, '_unsloth_model_ref', None) or getattr(self, 'model', None)
    if _model is not None:
        for _m in (_model, getattr(_model, 'model', None)):
            if _m is None: continue
            _mod = _sys.modules.get(type(_m).__module__)
            if _mod is not None and hasattr(_mod, _ccm):
                _needs_token_type_ids = True
                break

    if not _needs_token_type_ids:
        # Fallback: model not yet available, check processor class MRO
        for _base in type(processing_class).__mro__:
            _base_mod = getattr(_base, '__module__', '')
            if 'transformers.models.' in _base_mod:
                _modeling_mod = _base_mod.replace('.processing_', '.modeling_')
                _mod = _sys.modules.get(_modeling_mod)
                if _mod is not None and hasattr(_mod, _ccm):
                    _needs_token_type_ids = True
                    break
    if _needs_token_type_ids and hasattr(args, 'remove_unused_columns'):
        args.remove_unused_columns = False

    # Get max length
    max_seq_length = getattr(args, "max_length", 0)
    if max_seq_length == 0: max_seq_length = getattr(args, "max_seq_length", 0)
    if max_seq_length == 0: max_seq_length = getattr(self, "max_seq_length", 0)
    if max_seq_length == 0: max_seq_length = getattr(self, "max_seq", 0)
    if max_seq_length == 0: raise RuntimeError("Unsloth: max_seq_length is 0! Please specify one!")
    dataset_text_field = getattr(args, "dataset_text_field", "text")
    do_truncation = max_seq_length != 0
    do_formatting_func = False
    do_tokenize = True
    do_prompt_completion = False

    # Get correct column names
    column_names = set(next(iter(dataset)).keys())
    used_column_names = ["input_ids"]
    if "attention_mask" in column_names:
        used_column_names.append("attention_mask")
    if _needs_token_type_ids:
        used_column_names.append("token_type_ids")

    # Skip tokenization if already tokenized; just set the data collator
    from transformers import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
    if "labels" in column_names:
        # Most likely forgot data collator!
        if is_vlm and not hasattr(tokenizer, "pad"):
            raise RuntimeError(f"Unsloth: {processing_class.__class__} does not have .pad!")
        self.data_collator = DataCollatorForSeq2Seq(tokenizer)
        used_column_names.append("labels")
        do_tokenize = False
    elif "input_ids" in column_names:
        if is_vlm and not hasattr(tokenizer, "pad"):
            raise RuntimeError(f"Unsloth: {processing_class.__class__} does not have .pad!")
        self.data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = False)
        do_tokenize = False
    elif "prompt" in column_names and "completion" in column_names:
        # Prompt/completion dataset (used with completion_only_loss).
        # TRL's __init__ already set self.data_collator for completion_only_loss
        # before calling us -- we must NOT overwrite it here.
        do_prompt_completion = True
        used_column_names.append("completion_mask")
    elif dataset_text_field not in column_names:
        do_formatting_func = True
        if formatting_func is None:
            raise RuntimeError("Unsloth: You must specify a `formatting_func`")
    pass

    if do_tokenize:
        if do_formatting_func:
            test_text = formatting_func(next(iter(dataset)))
            if not isinstance(test_text, list):
                raise ValueError(
                    "Unsloth: The `formatting_func` should return a list of processed strings."
                )
            test_text = test_text[0]
        elif do_prompt_completion:
            _first_ex = next(iter(dataset))
            try:
                from trl import is_conversational as _sft_is_conversational
            except ImportError:
                def _sft_is_conversational(example):
                    for key in ("prompt", "completion", "messages"):
                        val = example.get(key)
                        if isinstance(val, list) and val and isinstance(val[0], dict):
                            if "role" in val[0] and "content" in val[0]:
                                return True
                    return False
            _is_conv = _sft_is_conversational(_first_ex)
            if not _is_conv:
                test_text = _first_ex["prompt"]
            else:
                test_text = None  # chat template handles BOS
        else:
            # No [0] on a str: that is the first CHARACTER, so startswith(bos_token)
            # below could never match. Only unwrap when the field really is a list.
            test_text = next(iter(dataset))[dataset_text_field]
            if isinstance(test_text, (list, tuple)):
                test_text = test_text[0] if len(test_text) != 0 else None

        chat_template = getattr(processing_class, 'chat_template', '')
        if chat_template == '' and is_vlm:
            chat_template = getattr(tokenizer, 'chat_template', '')
        if chat_template is None:
            chat_template = ''

        # Detect double BOS so we can drop the duplicate
        add_special_tokens = True
        bos_token_1 = getattr(processing_class, 'bos_token', None)
        bos_token_2 = getattr(tokenizer, 'bos_token', None)
        bos_token = bos_token_1 or bos_token_2

        if bos_token is not None:
            if (test_text is not None and test_text.startswith(bos_token)) or bos_token in chat_template:
                add_special_tokens = False
                print("Unsloth: We found double BOS tokens - we shall remove one automatically.")
        pass

        def _tokenize(example):
            return tokenizer(
                example[dataset_text_field] if not do_formatting_func else formatting_func(example),
                truncation = do_truncation,
                max_length = max_seq_length,
                return_token_type_ids = _needs_token_type_ids,
                add_special_tokens = add_special_tokens,
            )
        pass

        if not isinstance(dataset, IterableDataset):
            import multiprocessing as _mp
            dataset_num_proc = getattr(args, "dataset_num_proc", None)
            if dataset_num_proc is None:
                if _mp.get_start_method() != 'fork':
                    dataset_num_proc = None
                else:
                    import psutil
                    dataset_num_proc = min(max((psutil.cpu_count() or 1)+4, 2), 64)
                    memory_gb_left = psutil.virtual_memory().available / (1024**3)
                    if memory_gb_left <= 2:
                        dataset_num_proc = 1
                    else:
                        dataset_num_proc = min(dataset_num_proc, int(memory_gb_left))
            map_kwargs["num_proc"] = dataset_num_proc
        else:
            map_kwargs["batch_size"] = _iterable_batch_size(dataset)

        if do_prompt_completion:
            _eos_token = getattr(tokenizer, 'eos_token', None)

            def _tokenize_pc(example):
                if _is_conv:
                    prompt_ids = processing_class.apply_chat_template(
                        example["prompt"], tokenize=True,
                        add_generation_prompt=True, return_dict=False,
                        tools=example.get("tools"),
                        **(example.get("chat_template_kwargs") or {}),
                    )
                    if prompt_ids and isinstance(prompt_ids[0], list):
                        prompt_ids = prompt_ids[0]
                    pc_processed = processing_class.apply_chat_template(
                        example["prompt"] + example["completion"],
                        return_dict=True, tokenize=True,
                        tools=example.get("tools"),
                        **(example.get("chat_template_kwargs") or {}),
                    )
                    if isinstance(pc_processed.get("input_ids", [None])[0], list):
                        pc_processed = {k: v[0] for k, v in pc_processed.items()}
                    pc_ids = pc_processed["input_ids"]
                else:
                    _completion = example["completion"]
                    if _eos_token and not _completion.endswith(_eos_token):
                        _completion = _completion + _eos_token
                    prompt_ids = tokenizer(
                        example["prompt"], add_special_tokens=add_special_tokens,
                    )["input_ids"]
                    pc_ids = tokenizer(
                        example["prompt"] + _completion,
                        add_special_tokens=add_special_tokens,
                    )["input_ids"]
                if do_truncation and max_seq_length > 0:
                    pc_ids = pc_ids[:max_seq_length]
                n_prompt = min(len(prompt_ids), len(pc_ids))
                completion_mask = [0] * n_prompt + [1] * (len(pc_ids) - n_prompt)
                result = {"input_ids": pc_ids, "completion_mask": completion_mask}
                if _needs_token_type_ids:
                    result["token_type_ids"] = [0] * len(pc_ids)
                return result

            if use_desc:
                map_kwargs["desc"] = 'Unsloth: Tokenizing ["prompt"+"completion"]'
            import warnings as _w
            with _w.catch_warnings():
                _w.filterwarnings("ignore", message=".*couldn't be hashed properly.*")
                dataset = dataset.map(
                    _tokenize_pc, batched=False,
                    remove_columns=list(column_names), **map_kwargs,
                )
        else:
            if use_desc: map_kwargs["desc"] = f'Unsloth: Tokenizing ["{dataset_text_field}"]'
            import warnings as _w
            with _w.catch_warnings():
                _w.filterwarnings("ignore", message=".*couldn't be hashed properly.*")
                dataset = dataset.map(_tokenize, batched = True, remove_columns = list(column_names), **map_kwargs)

        # VLMs need .pad; switch the data collator
        if is_vlm and not hasattr(processing_class, "pad") and not do_prompt_completion:
            data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = False)
            self.data_collator = data_collator
        pass
    pass
    if packing:
        # Use TRL's pack_dataset if available
        try:
            pack_dataset
        except:
            print("Unsloth: Hugging Face's packing is currently buggy - we're disabling it for now!")
            return dataset

        if max_seq_length == 0:
            raise ValueError("When packing is enabled, `max_seq_length` can't be `None`.")

        if use_desc: map_kwargs["desc"] = f"Unsloth: Packing {dataset_name} dataset"
        dataset = pack_dataset(
            dataset.select_columns(used_column_names),
            max_seq_length,
            getattr(args, "packing_strategy", "bfd"),
            map_kwargs,
        )
    pass
    return dataset
pass


def patch_torchcodec_audio_decoder():
    """Make datasets AudioDecoder dict-compatible for backwards compat.

    The datasets library with torchcodec backend returns AudioDecoder objects
    that support __getitem__ but not __contains__, breaking code like
    '"array" in audio'. This adds dict-like protocol methods.
    """
    try:
        from datasets.features._torchcodec import AudioDecoder
        if hasattr(AudioDecoder, '__contains__'):
            return  # Already patched or newer version

        AudioDecoder.__contains__ = lambda self, key: key in ("array", "sampling_rate")
        AudioDecoder.__iter__ = lambda self: iter(("array", "sampling_rate"))
        AudioDecoder.keys = lambda self: ("array", "sampling_rate")
        AudioDecoder.get = lambda self, key, default=None: (
            self[key] if key in ("array", "sampling_rate") else default
        )
    except (ImportError, AttributeError, RuntimeError):
        pass  # torchcodec not available or different datasets version
pass


# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
