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
pad positions, and a vision pad_token baked into a text-only tokenizer produces
NaN losses (unsloth#3155, unsloth#4104). `fix_pad_token` heals such tokenizers by
picking a reserved pad-like token already in the vocab; if none exists it adds one.
It is a no-op for tokenizers that already have a valid, distinct pad_token.
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

# Vision/modality tokens that must never be a pad_token on a text-only model.
VISION_RESERVED_TOKENS = frozenset((
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<|audio_pad|>",
))

_CLOSERS = (">", "|>", "]", ")")
_MANUAL_PAD_TOKEN = "<｜PAD▁TOKEN｜>"


def _token_content(token):
    # AddedToken on transformers 5.x exposes .content; slow tokenizers use str.
    content = getattr(token, "content", None)
    return content if isinstance(content, str) else str(token)


def _resolve_inner(tokenizer):
    # Processors wrap the real tokenizer under .tokenizer.
    inner = getattr(tokenizer, "tokenizer", None)
    return inner if inner is not None else tokenizer


def _infer_vision(tokenizer, inner, model, model_config, is_vision_model):
    if is_vision_model is not None:
        return bool(is_vision_model)
    if hasattr(tokenizer, "image_processor"):
        return True
    cfg = model_config if model_config is not None else getattr(model, "config", None)
    model_type = (getattr(cfg, "model_type", "") or "") if cfg is not None else ""
    return "vl" in model_type.lower()


def _display_name(model, model_config, inner):
    for source in (model_config, getattr(model, "config", None), inner):
        name = getattr(source, "_name_or_path", None) or getattr(source, "name_or_path", None)
        if name:
            return name
    return "Model"


def _classify_bad_pad(inner, is_vision):
    """Return a reason string if the current pad_token is bad, else None."""
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
    if not is_vision and pad in VISION_RESERVED_TOKENS:
        return "vision_pad"
    return None


def _single_token_id(inner, token, vocab_size):
    """Return token's id if it encodes to exactly one in-vocab id, else None."""
    try:
        ids = inner(token, add_special_tokens=False).input_ids
    except Exception:
        return None
    if len(ids) != 1:
        return None
    token_id = ids[0]
    if vocab_size is not None and token_id >= vocab_size:
        return None
    return token_id


def _find_reserved_pad(inner, is_vision, eos_token, vocab_size):
    """Pick the best reserved pad-like token already in the vocab, or None."""
    try:
        added = [_token_content(t) for t in inner.added_tokens_decoder.values()]
    except Exception:
        return None
    # Newest tokens last; reverse so reserved blocks are scanned high-id first.
    added = [a for a in added if a][::-1]

    for reserved in POSSIBLE_RESERVED_TOKENS:
        if not is_vision and reserved in VISION_RESERVED_TOKENS:
            continue
        matches = [a for a in added if a == reserved or a.startswith(reserved)]
        if not matches:
            continue
        # Prefer a closed token (e.g. "<|reserved_special_token_0|>") that is
        # distinct from eos and encodes to a single in-vocab id.
        closed = [m for m in matches if m.endswith(_CLOSERS)]
        for candidate in closed + matches:
            if candidate == eos_token or candidate in VISION_RESERVED_TOKENS:
                continue
            if _single_token_id(inner, candidate, vocab_size) is not None:
                return candidate
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
    `changed=False`) when the pad_token is already valid and distinct from eos.
    With `allow_add=False` (tokenizer-only callers with no model to resize) a brand
    new pad token is never added; repair is deferred to the model-aware call.
    """
    result = {"changed": False, "reason": None, "old_pad": None, "new_pad": None, "added": False}
    if tokenizer is None:
        return result

    inner = _resolve_inner(tokenizer)
    if inner is None:
        return result

    is_vision = _infer_vision(tokenizer, inner, model, model_config, is_vision_model)
    reason = _classify_bad_pad(inner, is_vision)
    if reason is None:
        return result
    result["reason"] = reason
    result["old_pad"] = getattr(inner, "pad_token", None)

    cfg = model_config if model_config is not None else getattr(model, "config", None)
    vocab_size = getattr(cfg, "vocab_size", None)
    eos_token = getattr(inner, "eos_token", None)

    new_pad = _find_reserved_pad(inner, is_vision, eos_token, vocab_size)
    added = False

    if new_pad is None:
        if not allow_add:
            # No model here to resize embeddings; let the model-aware call repair.
            return result
        new_pad = _MANUAL_PAD_TOKEN
        while new_pad in inner.get_vocab():
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
        if hasattr(model, "config"):
            model.config.update({"pad_token_id": inner.pad_token_id})
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
