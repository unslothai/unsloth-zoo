"""Real-renderer regression tests for ``encode_conversations_with_harmony``.

The companion test module ``test_encode_conversations_with_harmony.py`` stubs out the
entire ``openai_harmony`` library, so it only validates how the encoder *constructs*
Message objects, never what the real renderer puts into the returned token IDs. These
tests exercise the genuine ``openai_harmony`` renderer end to end and assert against the
decoded output.

They are skipped automatically when ``openai_harmony`` is not installed, or when the
gpt-oss harmony encoding cannot be loaded (e.g. no network to fetch the vocab), so the
suite stays runnable in hermetic CI while still covering the real behavior locally.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import pathlib
import sys
import types

import pytest


openai_harmony = pytest.importorskip("openai_harmony")


_MODULE_NAME = "unsloth_zoo.temporary_patches.gpt_oss"
_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1] / "unsloth_zoo"


def _package_shell(name, path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader = None, is_package = True)
    return module


@pytest.fixture()
def gpt_oss_module():
    # Mirror the import strategy used by the stubbed test module so the submodule can be
    # imported standalone (without a full unsloth install).
    if "unsloth_zoo" in sys.modules or importlib.util.find_spec("unsloth") is not None:
        try:
            module = importlib.import_module(_MODULE_NAME)
        except ImportError as error:
            if "Please install Unsloth" not in str(error):
                raise
        else:
            yield module
            return

    before_modules = set(sys.modules)
    sys.modules.pop("unsloth_zoo", None)
    sys.modules.pop("unsloth_zoo.temporary_patches", None)
    sys.modules.pop(_MODULE_NAME, None)
    sys.modules["unsloth_zoo"] = _package_shell("unsloth_zoo", _PACKAGE_ROOT)
    sys.modules["unsloth_zoo.temporary_patches"] = _package_shell(
        "unsloth_zoo.temporary_patches",
        _PACKAGE_ROOT / "temporary_patches",
    )
    try:
        yield importlib.import_module(_MODULE_NAME)
    finally:
        for module_name in list(sys.modules):
            if module_name == "unsloth_zoo" or module_name.startswith("unsloth_zoo."):
                if module_name not in before_modules:
                    sys.modules.pop(module_name, None)


@pytest.fixture(scope = "module")
def real_encoding():
    try:
        return openai_harmony.load_harmony_encoding(
            openai_harmony.HarmonyEncodingName.HARMONY_GPT_OSS
        )
    except Exception as error:  # pragma: no cover - environment dependent (e.g. no network)
        pytest.skip(f"gpt-oss harmony encoding unavailable: {error}")


def _decode(messages, gpt_oss_module, **kwargs):
    decoded_text, _ = gpt_oss_module.encode_conversations_with_harmony(messages, **kwargs)
    return decoded_text


def test_renderer_drops_analysis_before_final_by_default(real_encoding):
    # Characterization of the upstream renderer: with auto_drop_analysis (the default),
    # an analysis message that precedes a final message in the last assistant turn is
    # dropped, while disabling auto_drop_analysis keeps it. This is exactly why a
    # thinking+content turn does not surface reasoning in the default-rendered IDs.
    Role = openai_harmony.Role
    Message = openai_harmony.Message
    Conversation = openai_harmony.Conversation

    convo = Conversation.from_messages([
        Message.from_role_and_content(Role.USER, "user prompt"),
        Message.from_role_and_content(Role.ASSISTANT, "THINK_SENTINEL").with_channel("analysis"),
        Message.from_role_and_content(Role.ASSISTANT, "ANSWER_SENTINEL").with_channel("final"),
    ])

    default_text = real_encoding.decode(real_encoding.render_conversation(convo))
    assert "ANSWER_SENTINEL" in default_text
    assert "THINK_SENTINEL" not in default_text

    kept_text = real_encoding.decode(
        real_encoding.render_conversation(
            convo,
            openai_harmony.RenderConversationConfig(auto_drop_analysis = False),
        )
    )
    assert "ANSWER_SENTINEL" in kept_text
    assert "THINK_SENTINEL" in kept_text


def test_thinking_with_content_preserves_answer(real_encoding, gpt_oss_module):
    # The #246 contract: the visible answer (content) must survive into the encoded
    # output on the final channel. (Reasoning is dropped by Harmony's default
    # auto_drop_analysis when a final follows it; that is intended for this render path.)
    decoded = _decode(
        [{"role": "assistant", "thinking": "REASON_SENTINEL", "content": "ANSWER_SENTINEL"}],
        gpt_oss_module,
    )
    assert "ANSWER_SENTINEL" in decoded


def test_thinking_only_preserves_reasoning(real_encoding, gpt_oss_module):
    # A pure-reasoning turn ends in analysis (no trailing final), so the reasoning is
    # kept in the rendered output.
    decoded = _decode(
        [{"role": "assistant", "thinking": "REASON_SENTINEL"}],
        gpt_oss_module,
    )
    assert "REASON_SENTINEL" in decoded


def test_thinking_with_tool_calls_keeps_tool_call(real_encoding, gpt_oss_module):
    # Regression: a turn with both thinking and a tool call must keep the tool call in
    # the encoded output (the old unconditional continue dropped it).
    decoded = _decode(
        [{
            "role": "assistant",
            "thinking": "REASON_SENTINEL",
            "tool_calls": [{"name": "lookup_weather", "arguments": '{"city": "CITY_SENTINEL"}'}],
        }],
        gpt_oss_module,
    )
    assert "lookup_weather" in decoded
    assert "CITY_SENTINEL" in decoded


def test_thinking_tool_calls_and_content_keeps_tool_call(real_encoding, gpt_oss_module):
    decoded = _decode(
        [{
            "role": "assistant",
            "thinking": "REASON_SENTINEL",
            "content": "ANSWER_SENTINEL",
            "tool_calls": [{"name": "mixed_tool", "arguments": '{"value": "ARG_SENTINEL"}'}],
        }],
        gpt_oss_module,
    )
    assert "mixed_tool" in decoded
    assert "ARG_SENTINEL" in decoded


@pytest.mark.parametrize("thinking", [None, ""])
def test_nullable_thinking_does_not_crash(real_encoding, gpt_oss_module, thinking):
    # Nullable / empty thinking columns must not raise; the answer must still be encoded.
    decoded = _decode(
        [{"role": "assistant", "thinking": thinking, "content": "ANSWER_SENTINEL"}],
        gpt_oss_module,
    )
    assert "ANSWER_SENTINEL" in decoded


def test_plain_content_still_encodes(real_encoding, gpt_oss_module):
    decoded = _decode(
        [{"role": "assistant", "content": "PLAIN_SENTINEL"}],
        gpt_oss_module,
    )
    assert "PLAIN_SENTINEL" in decoded
