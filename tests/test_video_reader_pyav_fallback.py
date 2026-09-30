"""torchvision.io.read_video passes av.open(..., metadata_errors=...), which PyAV 19 removed.

_read_video_torchvision is both a primary backend and fetch_video's last resort, so on
PyAV 19 video input stopped working unless decord or torchcodec happened to succeed.
"""

import os

os.environ.setdefault("UNSLOTH_ZOO_DISABLE_GPU_INIT", "1")
os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

import pytest

av = pytest.importorskip("av")
np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from unsloth_zoo import vision_utils  # noqa: E402

PYAV_19_ERROR = "open() got an unexpected keyword argument 'metadata_errors'"


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate = 10)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for i in range(30):
            frame = av.VideoFrame.from_ndarray(
                np.full((32, 32, 3), i * 8, dtype = np.uint8), format = "rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return str(path)


def _torchvision_rejects_the_keyword(monkeypatch):
    def read_video(*args, **kwargs):
        raise TypeError(PYAV_19_ERROR)

    monkeypatch.setattr(vision_utils.io, "read_video", read_video)


def test_a_pyav_19_torchvision_read_falls_back_to_pyav(monkeypatch, clip):
    _torchvision_rejects_the_keyword(monkeypatch)
    video, sample_fps = vision_utils._read_video_torchvision({"video": clip})
    assert video.dim() == 4 and video.shape[1:] == (3, 32, 32)
    assert video.dtype == torch.uint8
    assert sample_fps > 0


def test_the_fallback_honours_the_requested_range(monkeypatch, clip):
    video, info = vision_utils._read_video_pyav(clip, 0.5, 1.5)
    assert video.shape[0] == 11
    assert info["video_fps"] == 10.0
    everything, _ = vision_utils._read_video_pyav(clip)
    assert everything.shape[0] == 30


def test_an_unrelated_type_error_still_propagates(monkeypatch, clip):
    def read_video(*args, **kwargs):
        raise TypeError("expected str, bytes or os.PathLike object")

    monkeypatch.setattr(vision_utils.io, "read_video", read_video)
    with pytest.raises(TypeError, match = "PathLike"):
        vision_utils._read_video_torchvision({"video": clip})
