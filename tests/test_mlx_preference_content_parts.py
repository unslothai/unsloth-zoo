# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Regression tests: preference rows carrying OpenAI-style content PARTS.

``_render_preference_example`` did not flatten list-valued ``content`` the way
the SFT path does, and real templates diverged: CONCAT templates (Qwen2.5,
SmolLM) raise ``TypeError`` on ``str + list``, while STRING-GUARD templates
(Qwen3) render every turn EMPTY -- chosen and rejected come out identical, so
the odds-ratio term is a constant log 2 with zero gradient and the run silently
trains on nothing. The stubs below reproduce both behaviors.
"""

from __future__ import annotations

import math
import sys

import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_shim():
    shim_prefixes = ("mlx", "mlx_lm", "mlx_vlm")
    real_mlx_modules = {
        name: module
        for name, module in sys.modules.items()
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in shim_prefixes)
    }
    from mlx_simulation import simulate_mlx_on_torch
    from mlx_simulation.mlx_stub import _MLXFinder
    simulate_mlx_on_torch()
    for name in list(sys.modules):
        if name == "unsloth_zoo.mlx" or name.startswith("unsloth_zoo.mlx."):
            sys.modules.pop(name, None)
    yield
    for name in list(sys.modules):
        if (
            name == "unsloth_zoo.mlx" or name.startswith("unsloth_zoo.mlx.")
            or any(name == prefix or name.startswith(f"{prefix}.") for prefix in shim_prefixes)
        ):
            sys.modules.pop(name, None)
    sys.meta_path[:] = [
        finder for finder in sys.meta_path
        if not isinstance(finder, _MLXFinder)
    ]
    sys.modules.update(real_mlx_modules)


PROMPT = [{"role": "user", "content": [{"type": "text", "text": "What is 2+2?"}]}]
CHOSEN = [{"role": "assistant", "content": [{"type": "text", "text": "It is 4."}]}]
REJECTED = [{"role": "assistant", "content": [{"type": "text", "text": "It is 5."}]}]

PROMPT_STR = [{"role": "user", "content": "What is 2+2?"}]
CHOSEN_STR = [{"role": "assistant", "content": "It is 4."}]
REJECTED_STR = [{"role": "assistant", "content": "It is 5."}]


class _ConcatTokenizer:
    """Qwen2.5 / SmolLM semantics: ``'...' + message.content`` (str-only concat)."""

    name = "concat"

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, continue_final_message=False,
                            **kwargs):
        out = ""
        for message in messages:
            out += "<|im_start|>" + message["role"] + "\n" + message["content"] + "<|im_end|>\n"
        if add_generation_prompt:
            out += "<|im_start|>assistant\n"
        return out


class _StringGuardTokenizer:
    """Qwen3 semantics: ``{%- if message.content is string %}`` else empty body."""

    name = "string_guard"

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, continue_final_message=False,
                            **kwargs):
        out = ""
        for message in messages:
            raw = message["content"]
            content = raw if isinstance(raw, str) else ""  # the silent drop
            out += "<|im_start|>" + message["role"] + "\n" + content + "<|im_end|>\n"
        if add_generation_prompt:
            out += "<|im_start|>assistant\n"
        return out


def _identity_normalize(messages, **kwargs):
    """Reproduces the pre-fix renderer: no content-part flattening."""
    return messages


# 1. Document the bug: without normalization each family misbehaves.
def test_prefix_concat_template_raises_on_content_parts(monkeypatch):
    from unsloth_zoo.mlx import utils as U

    monkeypatch.setattr(U, "_normalize_mlx_messages", _identity_normalize)
    with pytest.raises(TypeError, match="can only concatenate str"):
        U._render_preference_example(_ConcatTokenizer(), PROMPT, CHOSEN, REJECTED)


def test_prefix_string_guard_template_silently_empties_turns(monkeypatch):
    from unsloth_zoo.mlx import utils as U

    monkeypatch.setattr(U, "_normalize_mlx_messages", _identity_normalize)
    _, chosen_str, rejected_str = U._render_preference_example(
        _StringGuardTokenizer(), PROMPT, CHOSEN, REJECTED,
    )
    # No exception -- and the preference signal is gone.
    assert "What is 2+2?" not in chosen_str
    assert "It is 4." not in chosen_str
    assert chosen_str == rejected_str


# 2. The fix: both families render content, chosen != rejected.
@pytest.mark.parametrize("tokenizer", [_ConcatTokenizer(), _StringGuardTokenizer()],
                         ids=["concat", "string_guard"])
def test_content_parts_render_like_plain_strings(tokenizer):
    from unsloth_zoo.mlx.utils import _render_preference_example

    prompt_str, chosen_str, rejected_str = _render_preference_example(
        tokenizer, PROMPT, CHOSEN, REJECTED,
    )
    assert "What is 2+2?" in prompt_str
    assert "It is 4." in chosen_str
    assert "It is 5." in rejected_str
    assert chosen_str != rejected_str

    # Byte-identical to the same row written with plain string content, which is
    # the SFT path's contract for a flattened text part-list.
    assert (prompt_str, chosen_str, rejected_str) == _render_preference_example(
        tokenizer, PROMPT_STR, CHOSEN_STR, REJECTED_STR,
    )


def test_plain_string_rows_are_unchanged():
    """Non-conversational (string) preference rows must stay byte-identical."""
    from unsloth_zoo.mlx.utils import _render_preference_example

    assert _render_preference_example(
        _ConcatTokenizer(), "Q: ", "A4", "A5",
    ) == ("Q: ", "Q: A4", "Q: A5")


# 3. The consequence: the odds-ratio term regains a gradient.
def test_identical_renders_make_odds_ratio_a_constant():
    """Pin WHY the silent case is harmful: equal logps => L_OR == log 2 exactly."""
    import mlx.core as mx
    from unsloth_zoo.mlx.utils import _orpo_odds_ratio_loss

    logp = mx.array([-1.3, -0.7], dtype=mx.float32)
    or_loss = float(_orpo_odds_ratio_loss(logp, logp).astype(mx.float32))
    assert abs(or_loss - math.log(2.0)) < 1e-6


def test_fixed_render_gives_distinct_preference_sequences():
    """Post-fix the two sides differ, so L_OR is no longer pinned at log 2."""
    import mlx.core as mx
    from unsloth_zoo.mlx.utils import _orpo_odds_ratio_loss, _render_preference_example

    _, chosen_str, rejected_str = _render_preference_example(
        _StringGuardTokenizer(), PROMPT, CHOSEN, REJECTED,
    )
    assert chosen_str != rejected_str

    # Distinct sequences => distinct per-side logps => the term moves off log 2.
    logp_c = mx.array([-1.30, -0.70], dtype=mx.float32)
    logp_r = mx.array([-1.55, -0.95], dtype=mx.float32)
    or_loss = float(_orpo_odds_ratio_loss(logp_c, logp_r).astype(mx.float32))
    assert abs(or_loss - math.log(2.0)) > 1e-3


# 4. Same assertions against the real shipped templates, when available.
REAL_TOKENIZERS = [
    "Qwen/Qwen3-0.6B",                        # string-guard: silently dropped
    "Qwen/Qwen2.5-0.5B-Instruct",             # concat: raised TypeError
    "mlx-community/SmolLM-135M-Instruct-4bit",  # concat: raised TypeError
]


@pytest.mark.parametrize("model_id", REAL_TOKENIZERS)
def test_real_tokenizers_render_content_parts(model_id):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("jinja2")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    except Exception as exc:  # not cached / no network
        pytest.skip(f"{model_id} unavailable: {exc}")
    if not getattr(tokenizer, "chat_template", None):
        pytest.skip(f"{model_id} has no chat template")

    from unsloth_zoo.mlx.utils import _render_preference_example

    prompt_str, chosen_str, rejected_str = _render_preference_example(
        tokenizer, PROMPT, CHOSEN, REJECTED,
    )
    assert "What is 2+2?" in prompt_str
    assert "It is 4." in chosen_str
    assert "It is 5." in rejected_str
    assert chosen_str != rejected_str
