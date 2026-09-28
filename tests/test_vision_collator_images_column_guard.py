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

"""Images-column entries on the messages (non prompt-completion) path load through
fetch_image, so URLs hit Unsloth's private-address guard instead of a processor's fetcher."""

from __future__ import annotations

import io

import pytest
from PIL import Image

import unsloth_zoo.vision_utils as vu
from unsloth_zoo.vision_utils import UnslothVisionDataCollator


def _collator():
    collator = UnslothVisionDataCollator.__new__(UnslothVisionDataCollator)
    collator.patch_size = 14
    return collator


def test_pil_images_pass_through_untouched():
    image = Image.new("RGB", (32, 32))
    images, videos, _ = _collator()._extract_images_videos_for_example({"images": [image]}, [])
    assert images[0] is image and videos == []


def test_url_images_use_guarded_fetch(monkeypatch):
    buf = io.BytesIO()
    Image.new("RGB", (32, 32)).save(buf, format = "PNG")
    fetched = []
    def fake_fetch(url):
        fetched.append(url)
        return io.BytesIO(buf.getvalue())
    monkeypatch.setattr(vu, "fetch_remote_media_bytes", fake_fetch)

    images, _, _ = _collator()._extract_images_videos_for_example(
        {"images": ["https://example.com/a.png"]}, [],
    )
    assert fetched == ["https://example.com/a.png"]
    assert isinstance(images[0], Image.Image)


def test_link_local_url_is_refused(monkeypatch):
    monkeypatch.delenv("UNSLOTH_ALLOW_PRIVATE_URL_FETCH", raising = False)
    with pytest.raises(Exception, match = "Refusing to fetch"):
        _collator()._extract_images_videos_for_example(
            {"images": ["http://169.254.169.254/latest/meta-data/"]}, [],
        )
