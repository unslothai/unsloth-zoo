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

"""Granite Speech chat templates drop an audio part (or print the content list); the audio token must go in the text."""

import pytest

pytest.importorskip("torch")


def _render(messages, keep_audio, as_repr):
    out = []
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            text = content
        elif as_repr:
            text = str(content)
        else:
            text = "".join(
                p["text"] if p["type"] == "text" else ("<|audio|>" if keep_audio else "") for p in content
            )
        out.append(f"<{m['role']}>{text}</{m['role']}>")
    return "".join(out)


class _Processor:
    audio_token = "<|audio|>"
    feature_extractor = object()

    def __init__(self, keep_audio = False, as_repr = False):
        self.keep_audio, self.as_repr = keep_audio, as_repr

    def apply_chat_template(self, conversation, tokenize = False, **kwargs):
        return _render(conversation, self.keep_audio, self.as_repr)


@pytest.mark.parametrize("as_repr", [False, True])
def test_template_dropping_audio_gets_the_token_in_text(as_repr):
    from unsloth_zoo.vision_utils import _patch_string_content_chat_template

    processor = _Processor(as_repr = as_repr)
    assert _patch_string_content_chat_template(processor)
    rendered = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": "Transcribe."}]}]
    )
    assert rendered == "<user><|audio|>\nTranscribe.</user>"
    assert not _patch_string_content_chat_template(processor)  # idempotent


def test_template_rendering_audio_is_untouched():
    from unsloth_zoo.vision_utils import _patch_string_content_chat_template

    processor = _Processor(keep_audio = True)
    original = processor.apply_chat_template
    assert not _patch_string_content_chat_template(processor)
    assert processor.apply_chat_template == original


def test_processor_without_audio_is_untouched():
    from unsloth_zoo.vision_utils import _template_drops_audio

    processor = _Processor()
    processor.feature_extractor = None
    assert not _template_drops_audio(processor)
