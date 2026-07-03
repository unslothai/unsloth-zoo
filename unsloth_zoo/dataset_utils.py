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
import torch

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

    # Recover optional left/right tokens around the matched core
    original = tokenizer(component, add_special_tokens = False).input_ids
    for j in range(len(original)):
        if original[j : j + len(substring)] == substring: break
    optional_left  = original[:j]
    optional_right = original[j+len(substring):]
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
    probe_ids = tok(full, add_special_tokens = False).input_ids
    def validate(part, part_name):
        for cand in (part, " " + part):
            core = _find_common_token_ids(cand, tok, True)[0]
            if core and any(probe_ids[i : i + len(core)] == core for i in range(len(probe_ids) - len(core) + 1)):
                return cand
        raise ValueError(f"Unsloth: Could not reliably auto-detect {part_name} (detected {repr(part)}) - pass instruction_part and response_part.")
    return validate(instruction_part, "instruction_part"), validate(response_part, "response_part")
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
        if "labels" in examples:
            labels_ = examples["labels"].tolist()
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
        if num_proc is None or num_proc == 1: return num_proc
        if not _num_proc_was_auto: return num_proc  # honor explicit user value
        try:
            if len(dataset) < _MIN_ROWS_FOR_MULTIPROC: return None
        except TypeError:
            return None  # unknown length (e.g. IterableDataset)
        return num_proc
    pass

    # transformers 5.0+ VLMs skip dataset prep in SFTTrainer.__init__
    # (skip_prepare_dataset=True when _is_vlm), so tokenize before masking.
    def _maybe_tokenize_dataset(dataset):
        if dataset is None:
            return dataset
        sample = next(iter(dataset))
        if "input_ids" in sample:
            return dataset  # Already tokenized
        _tokenizer = trainer.processing_class if hasattr(trainer, "processing_class") else trainer.tokenizer
        # Use the actual tokenizer, not the processor
        if hasattr(_tokenizer, "tokenizer"):
            _tok = _tokenizer.tokenizer
        else:
            _tok = _tokenizer
        max_length = getattr(trainer.args, "max_length", None) or getattr(trainer.args, "max_seq_length", 2048)
        text_field = getattr(trainer.args, "dataset_text_field", "text")
        def _tokenize_fn(examples):
            texts = examples.get(text_field) or examples.get("text", [])
            return _tok(texts, truncation=True, max_length=max_length, padding=False)
        _map_kwargs = {"batched": True, "num_proc": _effective_num_proc(dataset)}
        if isinstance(dataset, IterableDataset):
            _map_kwargs = {"batched": True}
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

    def _filter_fully_masked(dataset, dataset_name="dataset"):
        if isinstance(dataset, IterableDataset):
            return dataset  # Cannot filter IterableDataset efficiently
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
            raise ValueError(
                f"Unsloth: train_on_responses_only masked every label to -100 in {dataset_name}, "
                f"so there is nothing to train on. The response marker {repr(response_part)} was not "
                "found in any sample - check that instruction_part and response_part match your chat template."
            )
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
        # (e.g. DataCollatorForSeq2Seq(tokenizer=processor)); either exposing image_processor marks it.
        for attr in ("processor", "tokenizer"):
            obj = getattr(collator, attr, None)
            if obj is not None and hasattr(obj, "image_processor"): return True
        return hasattr(collator, "image_processor")
    pass

    data_collator = getattr(trainer, "data_collator", None)
    if _is_vision_collator(data_collator):
        masking = getattr(data_collator, "train_on_responses_only", None)
        if callable(masking):
            return trainer  # collator already masks responses; nothing to do
        is_unsloth = any(b.__name__ == "UnslothVisionDataCollator" for b in type(data_collator).__mro__)
        if not is_unsloth:
            # A processor-style collator we cannot reliably configure: do not return as
            # if masking were applied (it would leave responses unmasked silently).
            raise ValueError(
                "Unsloth: Detected a vision data collator that does not support response-only "
                "masking. Build UnslothVisionDataCollator(..., train_on_responses_only = True, "
                "instruction_part = ..., response_part = ...) so masking runs at collate time."
            )
        # If the collator's tokenizer already carries the parts, let the nested call
        # read them; passing them explicitly would hit the "already set" guard.
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

    if hasattr(trainer, "train_dataset") and trainer.train_dataset is not None:
        if not hasattr(trainer.train_dataset, "map"):
            raise TypeError("Unsloth: train_on_responses_only does not work on lists!")
        trainer.train_dataset = _maybe_tokenize_dataset(trainer.train_dataset)
        if isinstance(trainer.train_dataset, IterableDataset):
            trainer.train_dataset = trainer.train_dataset.map(_train_on_responses_only, batch_size = trainer.train_dataset._ex_iterable.batch_size, batched = True)
        else:
            trainer.train_dataset = trainer.train_dataset.map(_train_on_responses_only, batched = True, num_proc = _effective_num_proc(trainer.train_dataset))
        trainer.train_dataset = _filter_fully_masked(trainer.train_dataset, "train_dataset")
    pass

    if hasattr(trainer, "eval_dataset") and trainer.eval_dataset is not None:
        # Eval datasets could be a dict!
        if type(trainer.eval_dataset) is dict:
            for key, value in trainer.eval_dataset.items():
                if not hasattr(value, "map"):
                    raise TypeError("Unsloth: train_on_responses_only does not work on lists!")
                value = _maybe_tokenize_dataset(value)
                if isinstance(value, IterableDataset):
                    trainer.eval_dataset[key] = value.map(_train_on_responses_only, batch_size = value._ex_iterable.batch_size, batched = True)
                else:
                    trainer.eval_dataset[key] = value.map(_train_on_responses_only, batched = True, num_proc = _effective_num_proc(value))
                trainer.eval_dataset[key] = _filter_fully_masked(trainer.eval_dataset[key], f"eval_dataset[{key}]")
        else:
            if not hasattr(trainer.eval_dataset, "map"):
                raise TypeError("Unsloth: train_on_responses_only does not work on lists!")
            trainer.eval_dataset = _maybe_tokenize_dataset(trainer.eval_dataset)
            if isinstance(trainer.eval_dataset, IterableDataset):
                trainer.eval_dataset = trainer.eval_dataset.map(_train_on_responses_only, batch_size = trainer.eval_dataset._ex_iterable.batch_size, batched = True)
            else:
                trainer.eval_dataset = trainer.eval_dataset.map(_train_on_responses_only, batched = True, num_proc = _effective_num_proc(trainer.eval_dataset))
            trainer.eval_dataset = _filter_fully_masked(trainer.eval_dataset, "eval_dataset")
        pass
    pass

    # Edit data collator to DataCollatorForSeq2Seq (vision collators were handled
    # earlier and returned, so trainer.data_collator here is a text collator).
    from transformers import DataCollatorForSeq2Seq
    packing_enabled = getattr(trainer.args, "packing", False)
    if (
        hasattr(trainer, "data_collator")
        and not isinstance(trainer.data_collator, DataCollatorForSeq2Seq)
        and not packing_enabled
    ):
        trainer.data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer)

    # Check if all labels randomnly got masked to nothing - maybe wrong chat template?
    from .training_utils import fix_zero_training_loss
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
        import multiprocessing as _mp
        if num_proc is None or type(num_proc) is not int:
            if _mp.get_start_method() != 'fork':
                num_proc = None
            else:
                import psutil
                num_proc = min(max((psutil.cpu_count() or 1)+4, 2), 64)
                memory_gb_left = psutil.virtual_memory().available / (1024**3)
                if memory_gb_left <= 2:
                    num_proc = 1
                else:
                    num_proc = min(num_proc, int(memory_gb_left))
        dataset_map_kwargs['num_proc'] = num_proc
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
            test_text = next(iter(dataset))[dataset_text_field][0]

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
            map_kwargs["batch_size"] = dataset._ex_iterable.batch_size

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
