import pytest

from unsloth_zoo import llama_cpp

EXTRA = "EXTRA"
TOOLS = [{"type": "function", "function": {"name": "f"}}]


@pytest.fixture(autouse = True)
def _appended(monkeypatch):
    monkeypatch.setattr(llama_cpp, "_check_llama_cpp_appended_system_message", lambda: EXTRA)


def _messages():
    return [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}]


def test_copy_and_inplace_agree_with_a_system_message():
    inplace = _messages()
    llama_cpp.add_llama_cpp_system_message(inplace, TOOLS, inplace = True)
    original = _messages()
    copied = llama_cpp.add_llama_cpp_system_message(original, TOOLS, inplace = False)
    assert copied[0]["content"] == inplace[0]["content"] == "SYS\n\n" + EXTRA
    assert original == _messages()


def test_copy_without_system_message_prepends_one():
    out = llama_cpp.add_llama_cpp_system_message([{"role": "user", "content": "hi"}], TOOLS)
    assert out[0] == {"role": "system", "content": EXTRA}


def test_none_messages_returned_as_is():
    assert llama_cpp.add_llama_cpp_system_message(None, TOOLS) is None
