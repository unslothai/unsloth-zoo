import pytest
import requests

from unsloth_zoo import llama_cpp

EXTRA = "EXTRA"
TOOLS = [{"type": "function", "function": {"name": "f"}}]


@pytest.fixture
def appended(monkeypatch):
    monkeypatch.setattr(llama_cpp, "_check_llama_cpp_appended_system_message", lambda: EXTRA)


def _messages():
    return [{"role": "system", "content": "SYS", "name": "n"}, {"role": "user", "content": "hi"}]


def test_copy_and_inplace_agree_with_a_system_message(appended):
    inplace = _messages()
    llama_cpp.add_llama_cpp_system_message(inplace, TOOLS, inplace = True)
    original = _messages()
    copied = llama_cpp.add_llama_cpp_system_message(original, TOOLS, inplace = False)
    assert copied == inplace
    assert copied[0] == {"role": "system", "content": "SYS\n\n" + EXTRA, "name": "n"}
    assert original == _messages()


def test_copy_without_system_message_prepends_one(appended):
    out = llama_cpp.add_llama_cpp_system_message([{"role": "user", "content": "hi"}], TOOLS)
    assert out[0] == {"role": "system", "content": EXTRA}


@pytest.mark.parametrize("messages", [None, []])
def test_empty_messages_returned_as_is(appended, messages):
    assert llama_cpp.add_llama_cpp_system_message(messages, TOOLS) is messages


def _no_network(monkeypatch, exc):
    calls = []

    def get(*args, **kwargs):
        calls.append(args)
        raise exc

    llama_cpp._check_llama_cpp_appended_system_message.cache_clear()
    monkeypatch.setattr(llama_cpp.requests, "get", get)
    yield calls
    llama_cpp._check_llama_cpp_appended_system_message.cache_clear()


@pytest.fixture
def offline(monkeypatch):
    yield from _no_network(monkeypatch, requests.exceptions.ConnectionError("offline"))


def test_offline_with_tools_leaves_messages_unchanged(offline):
    messages = _messages()
    assert llama_cpp.add_llama_cpp_system_message(messages, TOOLS) == _messages()
    assert len(offline) == 1


def test_no_tools_never_fetches(offline):
    messages = _messages()
    assert llama_cpp.add_llama_cpp_system_message(messages, None) is messages
    assert llama_cpp.add_llama_cpp_system_message(messages, []) is messages
    assert offline == []
