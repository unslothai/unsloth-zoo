# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This file is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0-only).
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU Affero General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

"""Generic pad_token repair shared by unsloth and unsloth-zoo.

A wrong/missing pad_token breaks training: pad == eos makes the loss ignore real
pad positions (the model never learns to stop). `fix_pad_token` heals such
tokenizers by picking a reserved pad-like token already in the vocab; if none
exists it reuses the model config's own declared pad id, and only if that is
unusable does it add a new token. It is a no-op for an already valid, distinct pad_token.

Any "*pad*"-named token (e.g. <|vision_pad|>, <|fim_pad|>, [PAD]) is a valid pad and
is kept as-is. Qwen3 text models share Qwen3-VL's vocab and ship
pad_token=<|vision_pad|>, which is fine. The old NaN reports (unsloth#3155, #4104)
were a masking artifact (collators set labels=-100 on pad), not a pad-identity bug.
"""

__all__ = [
    "POSSIBLE_RESERVED_TOKENS",
    "VISION_RESERVED_TOKENS",
    "fix_pad_token",
]

# Candidate pad tokens, highest priority first. Entries are matched against each
# added token by exact-equality OR prefix (families like "<|reserved" cover
# "<|reserved_special_token_0|>"). A closed candidate (ends with > |> ] )) wins
# over an open prefix match.
POSSIBLE_RESERVED_TOKENS = (
    "<|finetune_right_pad_id|>",  # Llama-3.1
    "<|finetune_right_pad|>",     # Llama-4
    "<pad>",                      # Mistral Nemo, Gemma
    "[PAD]",                      # Kimi K2
    "<PAD>",                      # Cohere
    "<|endoftext|>",              # Qwen / GPT text pad (upstream Qwen pads with this)
    "<fim_pad>",                  # Granite
    "<｜▁pad▁｜>",                # DeepSeek R1
    "[MASK]",                     # GLM
    "<|reserved",                 # Llama-3
    "<|placeholder",              # Phi-3
    "<|dummy_",                   # Phi-4
    "[control",                   # Mistral type models
    "|<EXTRA_TOKENS_",            # Molmo
    "<SPECIAL_",                  # Pixtral
    "<unused",                    # PaliGemma
    "<|EOT|>",                    # DeepSeek Prover
    "<|reject-unknown|>",         # Red Note
    # "<|endofprompt|>",          # Phi-4 mini; commented out, clashes with GPT-OSS
)

# Modality pad tokens: pad-named, so valid pads on any model. Appended as
# last-resort replacement candidates after the curated families above.
_MODALITY_PAD_TOKENS = (
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
)
# Re-exported for backwards compatibility; no longer used to reject pads.
VISION_RESERVED_TOKENS = frozenset(_MODALITY_PAD_TOKENS)

_CLOSERS = (">", "|>", "]", ")")
_MANUAL_PAD_TOKEN = "<｜PAD▁TOKEN｜>"


def _token_content(token):
    # AddedToken on transformers 5.x exposes .content; slow tokenizers use str.
    content = getattr(token, "content", None)
    return content if isinstance(content, str) else str(token)


def _is_pad_named(token):
    """True if the token is a dedicated pad sentinel: contains "pad" and is bracketed
    (e.g. <|vision_pad|>, <|fim_pad|>, <pad>, [PAD]), not an ordinary word like
    "keypad". The bracket form, not the unreliable `.special` flag (Qwen marks
    <|fim_pad|> non-special), is what distinguishes a sentinel from a real word."""
    content = _token_content(token)
    return ("pad" in content.lower()
            and content.startswith(("<", "[")) and content.endswith(_CLOSERS))


def _resolve_inner(tokenizer):
    # Processors wrap the real tokenizer under .tokenizer.
    inner = getattr(tokenizer, "tokenizer", None)
    return inner if inner is not None else tokenizer


def _display_name(model, model_config, inner):
    for source in (model_config, getattr(model, "config", None), inner):
        name = getattr(source, "_name_or_path", None) or getattr(source, "name_or_path", None)
        if name:
            return name
    return "Model"


def _eos_id_set(*values):
    """Collect EOS ids from scalar and list/tuple/set/frozenset forms into one flat set.

    `bool` is excluded (it subclasses int, but True/False are never real token ids).
    Used to reject replacement-pad candidates whose id is really an EOS. Callers pass the
    tokenizer's and the model config's eos, each of which may be a multi-EOS list. It does
    NOT take the generation_config stop list: a valid pad (e.g. Qwen's <|endoftext|>)
    legitimately appears there, so that list must not drive pad decisions.
    """
    ids = set()
    for value in values:
        if type(value) is int:
            ids.add(value)
        elif isinstance(value, (list, tuple, set, frozenset)):
            ids.update(v for v in value if type(v) is int)
    return ids


def _classify_bad_pad(inner, vocab_size):
    """Return a reason string if the current pad_token is bad, else None.

    A pad-named token (e.g. <|vision_pad|>) is valid; only missing, eos-collision
    and out-of-range pads are healed. Only the tokenizer's OWN eos defines a bad pad:
    a token that also appears in a model/generation multi-EOS stop list (e.g. Qwen pads
    with <|endoftext|>, which is also a secondary generation stop id) is still a valid,
    distinct pad and must be left alone.
    """
    if not hasattr(inner, "pad_token"):
        return None
    pad = inner.pad_token
    if pad is None:
        return "missing"
    eos = getattr(inner, "eos_token", None)
    if eos is not None and pad == eos:
        return "equals_eos"
    pad_id, eos_id = getattr(inner, "pad_token_id", None), getattr(inner, "eos_token_id", None)
    if pad_id is not None and eos_id is not None and pad_id == eos_id:
        return "equals_eos"
    # A pad id past the model's embeddings indexes out of range at runtime; only
    # checkable once a model/config gives us vocab_size.
    if vocab_size is not None and pad_id is not None and pad_id >= vocab_size:
        return "out_of_range"
    return None


def _config_declared_pad(inner, cfg, eos_token_ids, vocab_size):
    """Reuse the model config's own declared pad_token_id, as a last resort before
    synthesizing a brand-new token.

    Some shipped tokenizers alias pad to eos (or leave pad out of range) even though the
    model config already declares a valid, distinct pad id pointing at an existing token.
    Whisper-large-v3 is the canonical case (config pad 50256 -> the existing empty token,
    while the tokenizer reports pad == eos == 50257), but the pattern is not Whisper
    specific, so this stays model-type agnostic. Honoring that declared id avoids adding a
    token and resizing embeddings. Guards: the id must be a real int (not bool), in vocab,
    distinct from every known EOS id, and round-trip cleanly through the tokenizer. Runs
    only after the reserved-token search fails, so it never overrides a pad-named choice
    for an already-working model.
    """
    if cfg is None:
        return None
    pad_token_id = getattr(cfg, "pad_token_id", None)
    if type(pad_token_id) is not int or pad_token_id < 0:
        return None
    if vocab_size is not None and pad_token_id >= vocab_size:
        return None
    if pad_token_id in eos_token_ids:
        return None

    convert_ids_to_tokens = getattr(inner, "convert_ids_to_tokens", None)
    convert_tokens_to_ids = getattr(inner, "convert_tokens_to_ids", None)
    if not callable(convert_ids_to_tokens) or not callable(convert_tokens_to_ids):
        return None
    try:
        candidate = convert_ids_to_tokens(pad_token_id)
        # Require a clean round-trip so we never promote an unknown / unstable id.
        if candidate is None or convert_tokens_to_ids(candidate) != pad_token_id:
            return None
    except Exception:
        return None

    old_pad = getattr(inner, "pad_token", None)
    try:
        inner.pad_token = candidate
        if getattr(inner, "pad_token_id", None) != pad_token_id:
            inner.pad_token = old_pad
            return None
    except Exception:
        try:
            inner.pad_token = old_pad
        except Exception:
            pass
        return None
    return candidate


def _single_token_id(inner, token, vocab_size, eos_token_ids=frozenset()):
    """Return token's id if it encodes to exactly one in-vocab, non-EOS id, else None."""
    try:
        ids = inner(token, add_special_tokens=False).input_ids
    except Exception:
        return None
    if len(ids) != 1:
        return None
    token_id = ids[0]
    if vocab_size is not None and token_id >= vocab_size:
        return None
    if token_id in eos_token_ids:
        return None
    return token_id


def _find_reserved_pad(inner, eos_token, vocab_size, eos_token_ids=frozenset()):
    """Pick the best reserved/pad-named token already in the vocab, or None."""
    try:
        added = [_token_content(t) for t in inner.added_tokens_decoder.values()]
    except Exception:
        return None
    # Newest tokens last; reverse so reserved blocks are scanned high-id first.
    added = [a for a in added if a][::-1]

    # Modality pads are valid on any model; appended as last-resort candidates.
    families = POSSIBLE_RESERVED_TOKENS + _MODALITY_PAD_TOKENS
    for reserved in families:
        matches = [a for a in added if a == reserved or a.startswith(reserved)]
        if not matches:
            continue
        # Prefer a closed token (e.g. "<|reserved_special_token_0|>"); closed tokens
        # are a subset of matches, so order them first without duplicating.
        closed = [m for m in matches if m.endswith(_CLOSERS)]
        ordered = closed + [m for m in matches if m not in closed]
        for candidate in ordered:
            if candidate == eos_token:
                continue
            if _single_token_id(inner, candidate, vocab_size, eos_token_ids) is not None:
                return candidate

    # Last resort: any pad sentinel (a bracketed "*pad*" token such as <|fim_pad|>)
    # not in a curated family above. The bracket form in _is_pad_named excludes
    # ordinary added words like "keypad" / "padding" that would be masked out of
    # attention and loss if promoted.
    for candidate in added:
        if candidate == eos_token or not _is_pad_named(candidate):
            continue
        if _single_token_id(inner, candidate, vocab_size, eos_token_ids) is not None:
            return candidate
    return None


def _unk_fallback(inner, eos_token, vocab_size, eos_token_ids=frozenset()):
    """Last-resort existing-token pad: reuse unk_token (Llama-2 style), or None."""
    unk = getattr(inner, "unk_token", None)
    if not unk or unk == eos_token:
        return None
    if _single_token_id(inner, unk, vocab_size, eos_token_ids) is not None:
        return unk
    return None


def fix_pad_token(
    tokenizer,
    model=None,
    model_config=None,
    *,
    is_vision_model=None,
    allow_add=True,
):
    """Heal a bad/missing pad_token in place.

    Returns a result dict: {changed, reason, old_pad, new_pad, added}. No-op (and
    `changed=False`) when the pad_token is already valid and distinct from eos. Repair
    order: an existing reserved/pad-named token, then the tokenizer's unk, then the model
    config's own declared pad id (reused, no resize), and only if all fail a brand-new
    token. With `allow_add=False` (tokenizer-only callers with no model to resize) a brand
    new pad token is never added; repair is deferred to the model-aware call.
    `is_vision_model` is accepted for API compatibility but unused: any pad-named
    token is a valid pad regardless of modality.
    """
    result = {"changed": False, "reason": None, "old_pad": None, "new_pad": None, "added": False}
    if tokenizer is None:
        return result

    inner = _resolve_inner(tokenizer)
    if inner is None:
        return result

    cfg = model_config if model_config is not None else getattr(model, "config", None)
    vocab_size = getattr(cfg, "vocab_size", None)
    if type(vocab_size) is not int:
        # A malformed (e.g. string) vocab_size must not crash a later `id >= vocab_size`
        # comparison; treat it as unknown and skip range checks.
        vocab_size = None
    eos_token = getattr(inner, "eos_token", None)
    # EOS ids used to reject a replacement pad that would alias an EOS: the tokenizer's own
    # and the model config's (each a scalar or a multi-EOS list). Deliberately NOT the
    # generation_config stop list, which legitimately lists a valid pad (Qwen's
    # <|endoftext|>) and must not drive pad selection.
    eos_token_ids = _eos_id_set(
        getattr(inner, "eos_token_id", None),
        getattr(cfg, "eos_token_id", None),
    )

    reason = _classify_bad_pad(inner, vocab_size)
    if reason is None:
        return result
    result["reason"] = reason
    result["old_pad"] = getattr(inner, "pad_token", None)

    new_pad = _find_reserved_pad(inner, eos_token, vocab_size, eos_token_ids)
    if new_pad is None:
        new_pad = _unk_fallback(inner, eos_token, vocab_size, eos_token_ids)

    if new_pad is None:
        # Before synthesizing a token and resizing embeddings, honor the model config's
        # own declared pad id when it is safe (Whisper-large-v3 and any similarly
        # misconfigured tokenizer). This reuses an existing token, so no resize.
        config_pad = _config_declared_pad(inner, cfg, eos_token_ids, vocab_size)
        if config_pad is not None:
            result.update(changed=True, new_pad=config_pad, added=False)
            pad_token_id = getattr(inner, "pad_token_id", None)
            if model is not None:
                model_cfg = getattr(model, "config", None)
                if model_cfg is not None:
                    model_cfg.update({"pad_token_id": pad_token_id})
                if getattr(model, "generation_config", None) is not None:
                    model.generation_config.update(pad_token_id=pad_token_id)
            name = _display_name(model, model_config, inner)
            print(
                f"Unsloth: {name} had a bad pad_token ({result['old_pad']}). "
                f"Using model config pad_token_id = {pad_token_id} ({config_pad!r})."
            )
            return result

    added = False
    if new_pad is None:
        if not allow_add:
            # No model here to resize embeddings; let the model-aware call repair.
            return result
        new_pad = _MANUAL_PAD_TOKEN
        vocab = inner.get_vocab()
        while new_pad in vocab:
            new_pad = f"<｜{new_pad}｜>"
        added = True

    inner.add_special_tokens({"pad_token": new_pad})
    inner.pad_token = new_pad
    result.update(changed=True, new_pad=new_pad, added=added)

    if model is not None:
        if added and hasattr(model, "resize_token_embeddings"):
            try:
                if model.get_input_embeddings().weight.shape[0] < len(inner):
                    model.resize_token_embeddings(len(inner))
            except Exception:
                pass
        model_cfg = getattr(model, "config", None)
        if model_cfg is not None:
            model_cfg.update({"pad_token_id": inner.pad_token_id})
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.update(pad_token_id=inner.pad_token_id)

    name = _display_name(model, model_config, inner)
    verb = "has no pad_token" if reason == "missing" else f"had a bad pad_token ({result['old_pad']})"
    print(f"Unsloth: {name} {verb}. Using pad_token = {new_pad}.")

    if added:
        raise RuntimeError(
            f"Unsloth: Could not find a valid pad token for {name} - please inspect "
            f"the tokenizer. A temporary {new_pad!r} was added."
        )
    return result
