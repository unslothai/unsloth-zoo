# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
