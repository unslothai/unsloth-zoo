from __future__ import annotations

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


def test_encode_conversations_with_harmony_missing_import_keeps_install_hint(
    monkeypatch,
    gpt_oss_module,
):
    monkeypatch.delattr(gpt_oss_module, "SystemContent", raising = False)

    with pytest.raises(ImportError, match = "pip install openai_harmony"):
        gpt_oss_module.encode_conversations_with_harmony([])


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
