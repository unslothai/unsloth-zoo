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

"""Apertus 1.5 rejects list assistant content in Jinja; the probe must fall back to a string."""

from __future__ import annotations

import pytest

jinja2 = pytest.importorskip("jinja2")

from unsloth_zoo.vision_utils import _probe_assistant_single_content


def _raise(message):
    raise jinja2.exceptions.TemplateError(message)


STRING_ONLY = (
    "{% for m in messages %}"
    "{% if m['role'] == 'user' %}<u>{% for p in m['content'] %}"
    "{% if p['type'] == 'text' %}{{ p['text'] }}{% else %}<img>{% endif %}{% endfor %}</u>"
    "{% elif m['content'] is string %}<a>{{ m['content'] }}</a>"
    "{% else %}{{ raise_exception('Invalid assistant content') }}{% endif %}{% endfor %}"
)
LIST_OK = (
    "{% for m in messages %}<{{ m['role'] }}>"
    "{% for p in m['content'] %}{% if p['type'] == 'text' %}{{ p['text'] }}{% endif %}{% endfor %}"
    "</{{ m['role'] }}>{% endfor %}"
)
REPR = "{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}{% endfor %}"
BROKEN = "{{ raise_exception('this template renders nothing') }}"


class _Processor:
    def __init__(self, template):
        env = jinja2.Environment()
        env.globals["raise_exception"] = _raise
        self.template = env.from_string(template)

    def apply_chat_template(self, messages, **kwargs):
        return self.template.render(messages = messages)


def test_jinja_rejection_of_list_content_falls_back_to_string():
    assert _probe_assistant_single_content(_Processor(STRING_ONLY)) is True


def test_list_content_template_keeps_list_form():
    assert _probe_assistant_single_content(_Processor(LIST_OK)) is False


def test_repr_rendering_still_switches_to_string():
    assert _probe_assistant_single_content(_Processor(REPR)) is True


def test_template_rendering_neither_form_still_raises():
    with pytest.raises(RuntimeError, match = "renders nothing"):
        _probe_assistant_single_content(_Processor(BROKEN))
