"""Tests for resolve_file_uri_to_path and file:// handling in media readers.

Naive `path[7:]` stripping broke file://localhost/... URIs (producing the
unopenable "localhost/...") and never decoded percent-escapes, so valid URIs
failed deep inside the video decoder with unrelated errors.
"""
import os

import pytest

from unsloth_zoo.vision_utils import resolve_file_uri_to_path


def test_plain_path_unchanged():
    assert resolve_file_uri_to_path("/tmp/clip.mp4") == "/tmp/clip.mp4"


def test_http_url_unchanged():
    url = "https://example.com/clip.mp4"
    assert resolve_file_uri_to_path(url) == url


def test_data_uri_unchanged():
    uri = "data:video/mp4;base64,AAAA"
    assert resolve_file_uri_to_path(uri) == uri


def test_non_string_unchanged():
    for value in (None, 42, ["frame.jpg"], {"url": "x"}):
        assert resolve_file_uri_to_path(value) is value


def test_file_uri_empty_authority():
    assert resolve_file_uri_to_path("file:///tmp/clip.mp4") == "/tmp/clip.mp4"


def test_file_uri_localhost_authority():
    assert resolve_file_uri_to_path("file://localhost/tmp/clip.mp4") == "/tmp/clip.mp4"


def test_file_uri_percent_encoded():
    assert resolve_file_uri_to_path("file:///tmp/my%20clip.mp4") == "/tmp/my clip.mp4"


def test_file_uri_non_local_authority_unchanged():
    uri = "file://nas-server/share/clip.mp4"
    assert resolve_file_uri_to_path(uri) == uri


def test_degenerate_file_uri_unchanged():
    assert resolve_file_uri_to_path("file://") == "file://"


def test_fetch_image_opens_localhost_file_uri(tmp_path):
    PIL = pytest.importorskip("PIL")
    from unsloth_zoo.vision_utils import fetch_image

    target = tmp_path / "img with space.png"
    PIL.Image.new("RGB", (32, 32), (255, 0, 0)).save(target)
    uri = "file://localhost" + str(target).replace(" ", "%20")
    image = fetch_image({"image": uri})
    assert image.size[0] >= 28 and image.size[1] >= 28


@pytest.fixture(scope="session")
def tiny_mp4(tmp_path_factory):
    av = pytest.importorskip("av", reason="PyAV required to synthesize a video")
    np = pytest.importorskip("numpy")

    path = tmp_path_factory.mktemp("vids") / "tiny clip.mp4"
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=4)
    stream.width = stream.height = 64
    stream.pix_fmt = "yuv420p"
    rng = np.random.default_rng(0)
    for _ in range(8):
        frame = av.VideoFrame.from_ndarray(
            rng.integers(0, 255, (64, 64, 3), dtype=np.uint8), format="rgb24"
        )
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return str(path)


@pytest.mark.parametrize("uri_form", ["plain", "file", "file_localhost", "file_encoded"])
def test_fetch_video_accepts_local_file_uri_forms(tiny_mp4, uri_form):
    from unsloth_zoo.vision_utils import fetch_video

    uri = {
        "plain": tiny_mp4,
        "file": "file://" + tiny_mp4,
        "file_localhost": "file://localhost" + tiny_mp4,
        "file_encoded": "file://" + tiny_mp4.replace(" ", "%20"),
    }[uri_form]
    video = fetch_video({"type": "video", "video": uri})
    assert video.ndim == 4 and video.shape[0] >= 2
