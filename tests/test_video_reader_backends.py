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

"""Video decoding without decord / torchcodec / torchvision.io.read_video (gone in recent torchvision, e.g. 0.29): an
Apple Silicon MLX install has neither torchvision nor PyAV, only mlx-vlm's opencv-python."""

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from unsloth_zoo import vision_utils  # noqa: E402

FRAMES = 30


def _frame(i):
    pixels = np.zeros((32, 32, 3), dtype = np.uint8)
    pixels[..., 0] = i * 8  # red ramps, so BGR/RGB order and frame order are both visible
    return pixels


@pytest.fixture
def clip(tmp_path):
    path = str(tmp_path / "clip.mp4")
    try:
        import av
    except ImportError:
        cv2 = pytest.importorskip("cv2")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 32))
        for i in range(FRAMES):
            writer.write(cv2.cvtColor(_frame(i), cv2.COLOR_RGB2BGR))
        writer.release()
        return path
    with av.open(path, "w") as container:
        stream = container.add_stream("mpeg4", rate = 10)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for i in range(FRAMES):
            for packet in stream.encode(av.VideoFrame.from_ndarray(_frame(i), format = "rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


@pytest.fixture
def backends(monkeypatch):
    """Pretend only the named modules are importable; decord and torchcodec never are."""
    def install(*present, read_video = False):
        monkeypatch.setattr(vision_utils, "FORCE_UNSLOTH_VIDEO_READER", None)
        monkeypatch.setattr(vision_utils, "is_decord_available", lambda: False)
        monkeypatch.setattr(vision_utils, "is_torchcodec_available", lambda: False)
        monkeypatch.setattr(vision_utils, "_module_available", lambda name: name in present)
        monkeypatch.setattr(vision_utils, "HAS_TORCHVISION", "torchvision" in present)
        monkeypatch.setattr(vision_utils, "_torchvision_read_video", lambda: (lambda *a, **k: None) if read_video else None)
        vision_utils.get_video_reader_backend.cache_clear()
    yield install
    vision_utils.get_video_reader_backend.cache_clear()


@pytest.mark.parametrize("present,read_video,expected", [
    (("torchvision", "av", "cv2"), True, "torchvision"),
    (("torchvision", "av", "cv2"), False, "pyav"),
    (("cv2",), False, "cv2"),
    (("av",), False, "pyav"),
    (("torchvision",), False, "torchvision"),
])
def test_backend_selection_only_widens(backends, present, read_video, expected):
    backends(*present, read_video = read_video)
    assert vision_utils.get_video_reader_backend() == expected


def test_no_backend_names_every_option(backends):
    backends()
    with pytest.raises(ValueError, match = "opencv-python"):
        vision_utils.get_video_reader_backend()


def test_torchvision_without_read_video_decodes_through_pyav(monkeypatch, clip):
    pytest.importorskip("av")
    monkeypatch.setattr(vision_utils, "_torchvision_read_video", lambda: None)
    video, sample_fps = vision_utils._read_video_torchvision({"video": clip, "nframes": 4})
    assert tuple(video.shape) == (4, 3, 32, 32) and sample_fps > 0


def test_cv2_reader_matches_pyav_frames(monkeypatch, clip):
    pytest.importorskip("cv2")
    pytest.importorskip("av")
    monkeypatch.setattr(vision_utils, "_torchvision_read_video", lambda: None)
    ele = {"video": clip, "nframes": 6}
    want, want_fps = vision_utils._read_video_torchvision(ele)
    got, got_fps = vision_utils._read_video_cv2(ele)
    assert got.dtype == torch.uint8 and got.shape == want.shape
    assert got_fps == pytest.approx(want_fps)
    # Same frames in the same order, RGB not BGR: red channel ramps, blue stays dark.
    assert (got[:, 0].float().mean((1, 2)) - want[:, 0].float().mean((1, 2))).abs().max() < 4
    assert got[:, 0].float().mean() > got[:, 2].float().mean() + 20
    assert torch.all(got[1:, 0].float().mean((1, 2)) > got[:-1, 0].float().mean((1, 2)))


def test_cv2_reader_rgb_frames_in_order(clip):
    pytest.importorskip("cv2")
    video, fps = vision_utils._read_video_cv2({"video": clip, "nframes": 6})
    red = video[:, 0].float().mean((1, 2))
    assert tuple(video.shape) == (6, 3, 32, 32) and video.dtype == torch.uint8 and fps > 0
    assert torch.all(red[1:] > red[:-1]) and red.mean() > video[:, 2].float().mean() + 20


def test_cv2_reader_honours_time_range(clip):
    pytest.importorskip("cv2")
    video, _ = vision_utils._read_video_cv2({"video": clip, "video_start": 1.0, "video_end": 2.0, "nframes": 4})
    red = video[:, 0].float().mean((1, 2))
    assert len(video) == 4 and red.min() >= 8 * 9 and red.max() <= 8 * 21


def test_cv2_reader_resamples_when_frame_count_overstates(monkeypatch, clip):
    cv2 = pytest.importorskip("cv2")
    real = cv2.VideoCapture

    class Overstated:
        def __init__(self, path):
            self._cap = real(path)

        def get(self, prop):
            return FRAMES * 3 if prop == cv2.CAP_PROP_FRAME_COUNT else self._cap.get(prop)

        def __getattr__(self, name):
            return getattr(self._cap, name)

    monkeypatch.setattr(cv2, "VideoCapture", Overstated)
    video, _ = vision_utils._read_video_cv2({"video": clip, "nframes": 8})
    assert len(video) == 8


@pytest.mark.parametrize("reported", [0, -1])
def test_cv2_reader_counts_frames_when_the_container_has_no_count(monkeypatch, clip, reported):
    cv2 = pytest.importorskip("cv2")
    real = cv2.VideoCapture

    class Unindexed:
        def __init__(self, path):
            self._cap = real(path)

        def get(self, prop):
            return reported if prop == cv2.CAP_PROP_FRAME_COUNT else self._cap.get(prop)

        def __getattr__(self, name):
            return getattr(self._cap, name)

    monkeypatch.setattr(cv2, "VideoCapture", Unindexed)
    video, _ = vision_utils._read_video_cv2({"video": clip, "nframes": 6})
    red = video[:, 0].float().mean((1, 2))
    assert len(video) == 6 and red[-1] > 8 * (FRAMES - 3)


def test_fetch_video_through_cv2_only(backends, monkeypatch, clip):
    pytest.importorskip("cv2")
    backends("cv2")
    # The Apple Silicon case: no torchvision module at all, so the resize must not touch it either.
    monkeypatch.delattr(vision_utils, "transforms")
    frames, fps = vision_utils.fetch_video({"video": clip, "nframes": 4}, return_video_sample_fps = True)
    assert frames.shape[0] == 4 and fps > 0


def test_failed_backend_without_a_fallback_raises_its_own_error(backends, monkeypatch, clip):
    backends("cv2")

    def broken(ele):
        raise RuntimeError("cv2 exploded")

    monkeypatch.setitem(vision_utils.VIDEO_READER_BACKENDS, "cv2", broken)
    with pytest.raises(RuntimeError, match = "cv2 exploded"):
        vision_utils.fetch_video({"video": clip})


def test_resize_without_torchvision_matches_torchvision(monkeypatch):
    pytest.importorskip("torchvision")
    video = torch.randint(0, 256, (3, 3, 37, 53), dtype = torch.uint8, generator = torch.Generator().manual_seed(0))
    want = vision_utils._resize_video(video, 28, 56)
    monkeypatch.setattr(vision_utils, "HAS_TORCHVISION", False)
    got = vision_utils._resize_video(video, 28, 56)
    assert got.dtype == want.dtype and got.shape == want.shape
    assert torch.equal(got, want)
