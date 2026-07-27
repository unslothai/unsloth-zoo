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

"""Thought-channel splitting in the DiffusionGemma OpenAI shim.

The shim splits the model's thinking out of committed text into OpenAI-style
``reasoning_content``. DiffusionGemma emits two thinking dialects -- the chat
template's native ``<|channel>thought ... <channel|>`` and DeepSeek-style
``<think> ... </think>`` when thinking is not explicitly templated -- and
commits arrive in 256-token canvas blocks, so a marker can straddle a block
boundary mid-stream. These tests pin the splitter and the streaming holdback
that keeps marker fragments out of both channels.
"""

import pytest

# The shim imports fastapi/uvicorn at module level; neither ships in the
# [core] extras the CI runners install, so skip (not fail) collection there.
pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from unsloth_zoo.diffusion_studio.shim import (  # noqa: E402
    _marker_holdback,
    _split_thought_channels,
)


# ── _split_thought_channels ──────────────────────────────────────────


def test_no_markers_is_all_content():
    assert _split_thought_channels("Hello!") == ("", "Hello!")


def test_angle_brackets_without_markers_pass_through():
    text = "compare a < b | c > d and 1 << 2"
    assert _split_thought_channels(text) == ("", text)


def test_channel_dialect_splits():
    reasoning, content = _split_thought_channels(
        "<|channel>thought\nPlan: greet the user.\n<channel|>Hello! How can I help?"
    )
    assert reasoning == "Plan: greet the user."
    assert content == "Hello! How can I help?"


def test_think_dialect_splits():
    reasoning, content = _split_thought_channels(
        "<think>The user said hi.\nRespond politely.</think>Hi there!"
    )
    assert reasoning == "The user said hi.\nRespond politely."
    assert content == "Hi there!"


def test_unterminated_thought_is_all_reasoning():
    # Length-truncated generation: the end marker never arrives.
    reasoning, content = _split_thought_channels("<think>still thinking...")
    assert reasoning == "still thinking..."
    assert content == ""


def test_multiple_blocks_and_mixed_dialects():
    reasoning, content = _split_thought_channels(
        "<think>A</think>x<|channel>thought\nB<channel|>y"
    )
    assert reasoning == "A\nB"
    assert content == "xy"


def test_marker_text_never_reaches_either_channel():
    for text in (
        "<|channel>thought\nT\n<channel|>C",
        "<think>T</think>C",
        "lead<think>T</think>tail",
    ):
        reasoning, content = _split_thought_channels(text)
        for fragment in ("<|channel>", "<channel|>", "<think>", "</think>"):
            assert fragment not in reasoning
            assert fragment not in content


# ── _marker_holdback ─────────────────────────────────────────────────


def test_holdback_zero_when_no_partial_marker():
    assert _marker_holdback("plain text") == 0
    assert _marker_holdback("ends with full <think>") == 0  # complete marker, nothing partial


def test_holdback_covers_partial_markers():
    assert _marker_holdback("text<thi") == len("<thi")
    assert _marker_holdback("text<|channel>thoug") == len("<|channel>thoug")
    assert _marker_holdback("text</thin") == len("</thin")


# ── streaming reconstruction (mirrors the shim's gen() loop) ─────────


def _stream(full):
    """Replay *full* one character per commit through the shim's split + holdback
    + per-channel prefix-diff logic; return the concatenated channel outputs."""
    sent_reasoning = ""
    sent_content = ""
    out_reasoning = []
    out_content = []
    for cut in range(1, len(full) + 1):
        snapshot = full[:cut]
        done = cut == len(full)
        if done:
            reasoning, content = _split_thought_channels(snapshot)
        else:
            held = _marker_holdback(snapshot)
            safe = snapshot[: len(snapshot) - held] if held else snapshot
            reasoning, content = _split_thought_channels(safe)
        new_reasoning = reasoning[len(sent_reasoning):]
        if new_reasoning:
            out_reasoning.append(new_reasoning)
            sent_reasoning = reasoning
        new_content = content[len(sent_content):]
        if new_content:
            out_content.append(new_content)
            sent_content = content
    return "".join(out_reasoning), "".join(out_content)


@pytest.mark.parametrize(
    "full, expected_reasoning, expected_content",
    [
        (
            "<|channel>thought\nthink hard\n<channel|>The answer is 42.",
            "think hard",
            "The answer is 42.",
        ),
        ("<think>quick thought</think>Done.", "quick thought", "Done."),
        ("No thinking, straight answer.", "", "No thinking, straight answer."),
    ],
)
def test_streaming_every_chunk_boundary(full, expected_reasoning, expected_content):
    # One char per commit exercises every possible marker-straddling boundary.
    reasoning, content = _stream(full)
    assert reasoning == expected_reasoning
    assert content == expected_content
    for fragment in ("<|channel>", "<channel|>", "<think>", "</think>"):
        assert fragment not in reasoning
        assert fragment not in content


def test_streaming_preserves_literal_angle_text():
    full = "Use a < b and c | d; never a <thinko> typo."
    reasoning, content = _stream(full)
    assert reasoning == ""
    assert content == full
