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


def test_arrays_and_tensors_pass_through_untouched():
    import numpy as np
    import torch
    array = np.zeros((8, 8, 3), dtype = np.uint8)
    tensor = torch.zeros(3, 8, 8, dtype = torch.uint8)
    images, _, _ = _collator()._extract_images_videos_for_example({"images": [array, tensor]}, [])
    assert images[0] is array and images[1] is tensor


def test_datasets_url_path_dict_uses_guarded_fetch(monkeypatch):
    buf = io.BytesIO()
    Image.new("RGB", (32, 32)).save(buf, format = "PNG")
    fetched = []
    def fake_fetch(url):
        fetched.append(url)
        return io.BytesIO(buf.getvalue())
    monkeypatch.setattr(vu, "fetch_remote_media_bytes", fake_fetch)

    images, _, _ = _collator()._extract_images_videos_for_example(
        {"images": [{"bytes": None, "path": "https://example.com/a.png"}]}, [],
    )
    assert fetched == ["https://example.com/a.png"]
    assert isinstance(images[0], Image.Image)


def test_datasets_local_path_dict_loads(tmp_path):
    path = tmp_path / "a.png"
    Image.new("RGB", (32, 32)).save(path)
    images, _, _ = _collator()._extract_images_videos_for_example(
        {"images": [{"bytes": None, "path": str(path)}]}, [],
    )
    assert isinstance(images[0], Image.Image)


def test_decoded_entries_are_not_resized():
    source = Image.effect_noise((100, 100), 64).convert("RGB")
    buf = io.BytesIO()
    source.save(buf, format = "PNG")
    images, _, _ = _collator()._extract_images_videos_for_example(
        {"images": [{"bytes": buf.getvalue(), "path": None}]}, [],
    )
    assert images[0].size == (100, 100)
    assert images[0].tobytes() == source.tobytes()


def test_bare_base64_string_decodes():
    import base64
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (1, 2, 3)).save(buf, format = "PNG")
    encoded = base64.b64encode(buf.getvalue()).decode()
    images, _, _ = _collator()._extract_images_videos_for_example({"images": [encoded]}, [])
    assert images[0].size == (16, 16) and images[0].getpixel((0, 0)) == (1, 2, 3)


def test_missing_local_path_still_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        _collator()._extract_images_videos_for_example({"images": ["/no/such/image.png"]}, [])


def _rotated_jpeg():
    image = Image.new("RGB", (20, 10), (200, 10, 10))
    exif = image.getexif()
    exif[0x0112] = 6  # Orientation: rotate 90 CW on display
    buf = io.BytesIO()
    image.save(buf, format = "JPEG", exif = exif)
    return buf.getvalue()


def test_exif_orientation_matches_datasets_decode():
    datasets = pytest.importorskip("datasets")
    data = _rotated_jpeg()
    expected = datasets.Image().decode_example({"bytes": data, "path": None})
    images, _, _ = _collator()._extract_images_videos_for_example({"images": [{"bytes": data, "path": None}]}, [])
    assert images[0].size == expected.size == (10, 20)


def test_line_wrapped_base64_decodes():
    import base64
    buf = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buf, format = "PNG")
    wrapped = base64.encodebytes(buf.getvalue()).decode()
    assert "\n" in wrapped
    images, _, _ = _collator()._extract_images_videos_for_example({"images": [wrapped]}, [])
    assert images[0].size == (64, 64)


def test_none_entries_are_dropped():
    image = Image.new("RGB", (8, 8))
    images, _, _ = _collator()._extract_images_videos_for_example({"images": [None, image, None]}, [])
    assert images == [image]


def test_images_none_falls_back_to_embedded_messages():
    image = Image.new("RGB", (56, 56))
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "a"}]}]
    images, videos, _ = _collator()._extract_images_videos_for_example({"images": None, "messages": messages}, messages)
    assert len(images) == 1 and videos == []


def test_images_none_text_row_has_no_images():
    messages = [{"role": "user", "content": [{"type": "text", "text": "a"}]}]
    images, videos, _ = _collator()._extract_images_videos_for_example({"images": None, "messages": messages}, messages)
    assert images == [] and videos == []
