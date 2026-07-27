from __future__ import annotations

import builtins
import importlib
import importlib.machinery
import importlib.util
import pathlib
import sys
import types

import pytest


_MODULE_NAME = "unsloth_zoo.temporary_patches.gpt_oss"
_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1] / "unsloth_zoo"


def _package_shell(name, path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    module.__spec__ = importlib.machinery.ModuleSpec(
        name,
        loader = None,
        is_package = True,
    )
    return module


@pytest.fixture()
def gpt_oss_module():
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


class _Role:
    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class _ReasoningEffort:
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class _HarmonyEncodingName:
    HARMONY_GPT_OSS = "harmony_gpt_oss"


class _ChainContent:
    @classmethod
    def new(cls):
        return cls()

    def with_model_identity(self, *args, **kwargs):
        return self

    def with_reasoning_effort(self, *args, **kwargs):
        return self

    def with_conversation_start_date(self, *args, **kwargs):
        return self

    def with_knowledge_cutoff(self, *args, **kwargs):
        return self

    def with_required_channels(self, *args, **kwargs):
        return self

    def with_instructions(self, *args, **kwargs):
        return self

    def with_function_tools(self, *args, **kwargs):
        return self


class _ToolDescription:
    @classmethod
    def new(cls, name, description, parameters):
        return {
            "name": name,
            "description": description,
            "parameters": parameters,
        }


class _Author:
    @classmethod
    def new(cls, role, name):
        return {"role": role, "name": name}


class _Message:
    def __init__(self, role, content):
        self.role = role
        self.content = content
        self.channel = None
        self.recipient = None
        self.content_type = None

    @classmethod
    def from_role_and_content(cls, role, content):
        return cls(role, content)

    @classmethod
    def from_author_and_content(cls, author, content):
        return cls(author, content)

    def with_channel(self, channel):
        self.channel = channel
        return self

    def with_recipient(self, recipient):
        self.recipient = recipient
        return self

    def with_content_type(self, content_type):
        self.content_type = content_type
        return self


class _Conversation:
    @classmethod
    def from_messages(cls, messages):
        return tuple(messages)


class _Encoding:
    def __init__(self):
        self.calls = []

    def render_conversation_for_completion(self, conversation, role):
        self.calls.append(("completion", conversation, role))
        return [101]

    def render_conversation(self, conversation):
        self.calls.append(("conversation", conversation))
        return [202]

    def decode(self, input_ids):
        return f"decoded:{input_ids}"


def _install_fake_harmony(monkeypatch, gpt_oss, load_harmony_encoding):
    monkeypatch.setattr(gpt_oss, "Author", _Author, raising = False)
    monkeypatch.setattr(gpt_oss, "Conversation", _Conversation, raising = False)
    monkeypatch.setattr(gpt_oss, "DeveloperContent", _ChainContent, raising = False)
    monkeypatch.setattr(gpt_oss, "HarmonyEncodingName", _HarmonyEncodingName, raising = False)
    monkeypatch.setattr(gpt_oss, "Message", _Message, raising = False)
    monkeypatch.setattr(gpt_oss, "ReasoningEffort", _ReasoningEffort, raising = False)
    monkeypatch.setattr(gpt_oss, "Role", _Role, raising = False)
    monkeypatch.setattr(gpt_oss, "SystemContent", _ChainContent, raising = False)
    monkeypatch.setattr(gpt_oss, "ToolDescription", _ToolDescription, raising = False)
    monkeypatch.setattr(gpt_oss, "load_harmony_encoding", load_harmony_encoding, raising = False)
    monkeypatch.setattr(gpt_oss, "encoding", None, raising = False)


@pytest.mark.parametrize(
    ("add_generation_prompt", "expected_input_ids", "expected_call"),
    [
        (True, [101], "completion"),
        (False, [202], "conversation"),
    ],
)
def test_encode_conversations_with_harmony_loads_encoding_for_both_prompt_branches(
    monkeypatch,
    gpt_oss_module,
    add_generation_prompt,
    expected_input_ids,
    expected_call,
):
    encoding = _Encoding()

    def load_harmony_encoding(name):
        assert name == _HarmonyEncodingName.HARMONY_GPT_OSS
        return encoding

    _install_fake_harmony(monkeypatch, gpt_oss_module, load_harmony_encoding)

    decoded_text, input_ids = gpt_oss_module.encode_conversations_with_harmony(
        [{"role": "user", "content": "hello"}],
        add_generation_prompt = add_generation_prompt,
    )

    assert input_ids == expected_input_ids
    assert decoded_text == f"decoded:{expected_input_ids}"
    assert encoding.calls[0][0] == expected_call


def test_encode_conversations_with_harmony_load_failure_is_named(
    monkeypatch,
    gpt_oss_module,
):
    def load_harmony_encoding(name):
        raise ValueError("vocab unavailable")

    _install_fake_harmony(monkeypatch, gpt_oss_module, load_harmony_encoding)

    with pytest.raises(
        RuntimeError,
        match = "Unsloth: failed to load the gpt-oss harmony encoding: vocab unavailable",
    ) as error:
        gpt_oss_module.encode_conversations_with_harmony([])

    assert isinstance(error.value.__cause__, ValueError)


_UNSET = object()


@pytest.fixture()
def harmony_state(gpt_oss_module):
    # Snapshot and restore the harmony globals so lazy re-import (which rebinds them via
    # globals().update) cannot leak fake symbols into other tests.
    names = list(gpt_oss_module._HARMONY_SYMBOLS) + ["encoding"]
    saved = {name: getattr(gpt_oss_module, name, _UNSET) for name in names}
    try:
        yield gpt_oss_module
    finally:
        for name, value in saved.items():
            if value is _UNSET:
                if hasattr(gpt_oss_module, name):
                    delattr(gpt_oss_module, name)
            else:
                setattr(gpt_oss_module, name, value)


def _fake_harmony_module(load_harmony_encoding, omit = ()):
    module = types.ModuleType("openai_harmony")
    members = {
        "Author": _Author,
        "Conversation": _Conversation,
        "DeveloperContent": _ChainContent,
        "HarmonyEncodingName": _HarmonyEncodingName,
        "Message": _Message,
        "Role": _Role,
        "SystemContent": _ChainContent,
        "ToolDescription": _ToolDescription,
        "load_harmony_encoding": load_harmony_encoding,
        "ReasoningEffort": _ReasoningEffort,
    }
    for name, value in members.items():
        if name not in omit:
            setattr(module, name, value)
    return module


def _simulate_absent_at_import(gpt_oss):
    # Mimic openai_harmony not being importable when gpt_oss.py was first imported, so the
    # module-level symbols were never bound (the Colab "install after import" scenario).
    for name in gpt_oss._HARMONY_SYMBOLS:
        if hasattr(gpt_oss, name):
            delattr(gpt_oss, name)
    gpt_oss.encoding = None


def _block_harmony_import(monkeypatch, exc):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "openai_harmony":
            raise exc
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_encode_conversations_with_harmony_install_after_import_succeeds(
    monkeypatch,
    harmony_state,
):
    # Regression for unsloth#3361: openai_harmony installed after `import unsloth`.
    gpt_oss = harmony_state
    _simulate_absent_at_import(gpt_oss)

    encoding = _Encoding()
    monkeypatch.setitem(
        sys.modules, "openai_harmony", _fake_harmony_module(lambda name: encoding)
    )

    decoded_text, input_ids = gpt_oss.encode_conversations_with_harmony(
        [{"role": "user", "content": "hello"}],
    )

    assert input_ids == [101]
    assert decoded_text == "decoded:[101]"
    assert encoding.calls[0][0] == "completion"


def test_encode_conversations_with_harmony_missing_package_keeps_install_hint(
    monkeypatch,
    harmony_state,
):
    gpt_oss = harmony_state
    _simulate_absent_at_import(gpt_oss)
    _block_harmony_import(
        monkeypatch,
        ModuleNotFoundError("No module named 'openai_harmony'", name = "openai_harmony"),
    )

    with pytest.raises(ImportError, match = "pip install openai_harmony") as error:
        gpt_oss.encode_conversations_with_harmony([])

    assert isinstance(error.value.__cause__, ModuleNotFoundError)


def test_encode_conversations_with_harmony_transitive_dependency_named(
    monkeypatch,
    harmony_state,
):
    gpt_oss = harmony_state
    _simulate_absent_at_import(gpt_oss)
    _block_harmony_import(
        monkeypatch,
        ModuleNotFoundError("No module named 'tiktoken'", name = "tiktoken"),
    )

    with pytest.raises(ImportError, match = "dependency `tiktoken`") as error:
        gpt_oss.encode_conversations_with_harmony([])

    assert "pip install openai_harmony`" not in str(error.value)
    assert isinstance(error.value.__cause__, ModuleNotFoundError)


def test_encode_conversations_with_harmony_broken_install_named(
    monkeypatch,
    harmony_state,
):
    gpt_oss = harmony_state
    _simulate_absent_at_import(gpt_oss)
    _block_harmony_import(monkeypatch, ImportError("partially initialized module"))

    with pytest.raises(ImportError, match = "failed to import openai_harmony") as error:
        gpt_oss.encode_conversations_with_harmony([])

    assert isinstance(error.value.__cause__, ImportError)


def test_encode_conversations_with_harmony_internal_submodule_missing_is_broken_install(
    monkeypatch,
    harmony_state,
):
    gpt_oss = harmony_state
    _simulate_absent_at_import(gpt_oss)
    _block_harmony_import(
        monkeypatch,
        ModuleNotFoundError(
            "No module named 'openai_harmony._internal'",
            name = "openai_harmony._internal",
        ),
    )

    with pytest.raises(ImportError, match = "failed to import openai_harmony") as error:
        gpt_oss.encode_conversations_with_harmony([])

    assert "pip install openai_harmony" not in str(error.value)
    assert isinstance(error.value.__cause__, ModuleNotFoundError)


def test_encode_conversations_with_harmony_non_import_error_wrapped(
    monkeypatch,
    harmony_state,
):
    gpt_oss = harmony_state
    _simulate_absent_at_import(gpt_oss)
    _block_harmony_import(monkeypatch, RuntimeError("bad binary wheel"))

    with pytest.raises(ImportError, match = "failed to import openai_harmony") as error:
        gpt_oss.encode_conversations_with_harmony([])

    assert isinstance(error.value.__cause__, RuntimeError)


def test_encode_conversations_with_harmony_partial_build_lists_missing_symbol(
    monkeypatch,
    harmony_state,
):
    gpt_oss = harmony_state
    _simulate_absent_at_import(gpt_oss)
    monkeypatch.setitem(
        sys.modules,
        "openai_harmony",
        _fake_harmony_module(lambda name: _Encoding(), omit = ("ReasoningEffort",)),
    )

    with pytest.raises(ImportError, match = "ReasoningEffort"):
        gpt_oss.encode_conversations_with_harmony([])


def _encode_and_capture_messages(monkeypatch, gpt_oss, messages):
    # Run the encoder with the fake harmony stubs and return the per-channel assistant
    # Message objects that were appended to the conversation (the system message is index 0).
    encoding = _Encoding()
    _install_fake_harmony(monkeypatch, gpt_oss, lambda name: encoding)
    gpt_oss.encode_conversations_with_harmony(messages)
    conversation = encoding.calls[0][1]
    return [m for m in conversation if m.role == _Role.ASSISTANT]


def test_assistant_thinking_with_content_emits_analysis_and_final(
    monkeypatch,
    gpt_oss_module,
):
    # Core regression for unsloth-zoo#246: the analysis channel must carry the reasoning
    # from message["thinking"], and the answer in message["content"] must still be emitted
    # on the final channel rather than dropped.
    messages = [
        {"role": "assistant", "thinking": "let me reason", "content": "the answer"},
    ]
    assistant_messages = _encode_and_capture_messages(
        monkeypatch, gpt_oss_module, messages
    )

    analysis = [m for m in assistant_messages if m.channel == "analysis"]
    final = [m for m in assistant_messages if m.channel == "final"]

    assert len(analysis) == 1
    assert analysis[0].content == "let me reason"
    assert len(final) == 1
    assert final[0].content == "the answer"


def test_assistant_thinking_only_emits_single_analysis(
    monkeypatch,
    gpt_oss_module,
):
    # A pure-reasoning turn (thinking present, no/empty content) must produce only the
    # analysis message carrying the reasoning, and no final message.
    messages = [
        {"role": "assistant", "thinking": "just reasoning", "content": ""},
    ]
    assistant_messages = _encode_and_capture_messages(
        monkeypatch, gpt_oss_module, messages
    )

    assert len(assistant_messages) == 1
    assert assistant_messages[0].channel == "analysis"
    assert assistant_messages[0].content == "just reasoning"


def test_assistant_plain_content_unchanged_final_only(
    monkeypatch,
    gpt_oss_module,
):
    # No thinking and no tool_calls: exactly one final-channel message from content.
    messages = [
        {"role": "assistant", "content": "plain answer"},
    ]
    assistant_messages = _encode_and_capture_messages(
        monkeypatch, gpt_oss_module, messages
    )

    assert len(assistant_messages) == 1
    assert assistant_messages[0].channel == "final"
    assert assistant_messages[0].content == "plain answer"


def test_mixed_multi_turn_preserves_channels_and_branches(
    monkeypatch,
    gpt_oss_module,
):
    # user -> assistant(thinking+content) -> tool -> assistant(final): confirm channel
    # ordering and that the tool_calls / tool branches stay untouched.
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "thinking": "reason about it", "content": "first answer"},
        {"role": "tool", "name": "lookup", "content": "tool result"},
        {"role": "assistant", "content": "final answer"},
    ]
    encoding = _Encoding()
    _install_fake_harmony(monkeypatch, gpt_oss_module, lambda name: encoding)
    gpt_oss_module.encode_conversations_with_harmony(messages)
    conversation = encoding.calls[0][1]

    # Drop the leading system message, then assert the channel sequence.
    body = [m for m in conversation if m.role != _Role.SYSTEM]
    channels = [m.channel for m in body]
    assert channels == [None, "analysis", "final", "commentary", "final"]

    analysis = body[1]
    assert analysis.content == "reason about it"
    first_final = body[2]
    assert first_final.content == "first answer"

    tool_message = body[3]
    assert tool_message.channel == "commentary"
    assert tool_message.recipient == "assistant"
    assert tool_message.content == "tool result"


def test_assistant_thinking_with_tool_calls_keeps_both(
    monkeypatch,
    gpt_oss_module,
):
    # An assistant turn carrying BOTH reasoning and a tool call must emit the analysis
    # message AND the commentary tool call. The tool call must not be dropped.
    messages = [
        {
            "role": "assistant",
            "thinking": "decide to call the tool",
            "tool_calls": [{"name": "lookup", "arguments": "{\"city\": \"Paris\"}"}],
        },
    ]
    assistant_messages = _encode_and_capture_messages(
        monkeypatch, gpt_oss_module, messages
    )

    channels = [m.channel for m in assistant_messages]
    assert channels == ["analysis", "commentary"]
    assert assistant_messages[0].content == "decide to call the tool"
    tool_message = assistant_messages[1]
    assert tool_message.recipient == "functions.lookup"
    assert tool_message.content_type == "json"
    assert tool_message.content == "{\"city\": \"Paris\"}"


def test_assistant_thinking_none_falls_back_to_final(
    monkeypatch,
    gpt_oss_module,
):
    # A nullable "thinking" column materialized as None must not be passed into Harmony.
    # The turn should fall back to a single final-channel message from content.
    messages = [
        {"role": "assistant", "thinking": None, "content": "the answer"},
    ]
    assistant_messages = _encode_and_capture_messages(
        monkeypatch, gpt_oss_module, messages
    )

    assert len(assistant_messages) == 1
    assert assistant_messages[0].channel == "final"
    assert assistant_messages[0].content == "the answer"


def test_assistant_thinking_empty_string_falls_back_to_final(
    monkeypatch,
    gpt_oss_module,
):
    # Empty-string thinking is treated as absent (no empty analysis message); only the
    # final-channel message from content is emitted.
    messages = [
        {"role": "assistant", "thinking": "", "content": "the answer"},
    ]
    assistant_messages = _encode_and_capture_messages(
        monkeypatch, gpt_oss_module, messages
    )

    assert len(assistant_messages) == 1
    assert assistant_messages[0].channel == "final"
    assert assistant_messages[0].content == "the answer"


def test_assistant_thinking_tool_calls_and_content_keeps_analysis_and_tool_call(
    monkeypatch,
    gpt_oss_module,
):
    # When a turn has thinking, tool_calls and content, tool_calls and content stay
    # mutually exclusive: the tool call wins (the final answer comes after the tool
    # result), so we emit analysis + commentary and drop the content for this turn.
    messages = [
        {
            "role": "assistant",
            "thinking": "reason first",
            "content": "should not be emitted in a tool-call turn",
            "tool_calls": [{"name": "search", "arguments": "{\"q\": \"x\"}"}],
        },
    ]
    assistant_messages = _encode_and_capture_messages(
        monkeypatch, gpt_oss_module, messages
    )

    channels = [m.channel for m in assistant_messages]
    assert channels == ["analysis", "commentary"]
    assert "final" not in channels
    assert assistant_messages[1].recipient == "functions.search"


def test_encode_conversations_with_harmony_load_failure_retried_then_cached(
    monkeypatch,
    harmony_state,
):
    gpt_oss = harmony_state
    _simulate_absent_at_import(gpt_oss)

    calls = {"n": 0}
    encoding = _Encoding()

    def load_harmony_encoding(name):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise ValueError(f"transient failure {calls['n']}")
        return encoding

    monkeypatch.setitem(
        sys.modules, "openai_harmony", _fake_harmony_module(load_harmony_encoding)
    )

    messages = [{"role": "user", "content": "hi"}]
    for _ in range(2):
        with pytest.raises(RuntimeError, match = "failed to load the gpt-oss harmony encoding"):
            gpt_oss.encode_conversations_with_harmony(messages)

    decoded_text, input_ids = gpt_oss.encode_conversations_with_harmony(messages)
    assert input_ids == [101]
    assert calls["n"] == 3

    gpt_oss.encode_conversations_with_harmony(messages)
    assert calls["n"] == 3
