# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""The PyAV fallback of _read_video_torchvision samples while reading: only the kept frames are converted to RGB, and
the result is bit for bit the full read indexed by the same frame indices (ranges, fps / nframes settings, seeks)."""

import tracemalloc

import pytest

av = pytest.importorskip("av")
np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from unsloth_zoo import vision_utils  # noqa: E402

PYAV_19_ERROR = "open() got an unexpected keyword argument 'metadata_errors'"
SIDE = 64
N_FRAMES = 60


@pytest.fixture(scope = "module")
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("video") / "clip.mp4"
    rng = np.random.default_rng(0)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate = 10)
        stream.width = stream.height = SIDE
        stream.pix_fmt = "yuv420p"
        for i in range(N_FRAMES):
            # Distinct content per frame, so a wrong frame index cannot compare equal.
            image = rng.integers(0, 256, (SIDE, SIDE, 3), dtype = np.uint8)
            image[: 8, : 8] = i * 4
            frame = av.VideoFrame.from_ndarray(image, format = "rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return str(path)


@pytest.fixture
def fallback(monkeypatch):
    def read_video(*args, **kwargs):
        raise TypeError(PYAV_19_ERROR)

    # raising = False: torchvision 0.26+ has no read_video at all.
    monkeypatch.setattr(vision_utils.io, "read_video", read_video, raising = False)


def _full_read_then_sample(ele):
    # The previous route: every frame in range as RGB, then the same sampling as _read_video_torchvision.
    video, info = vision_utils._read_video_pyav(ele["video"], ele.get("video_start", 0.0), ele.get("video_end", None))
    total = video.size(0)
    nframes = vision_utils.smart_nframes(ele, total_frames = total, video_fps = info["video_fps"])
    idx = torch.linspace(0, total - 1, nframes).round().long()
    return video[idx], nframes / max(total, 1e-6) * info["video_fps"]


@pytest.mark.parametrize("extra", [
    {},
    {"fps": 2.0},
    {"nframes": 8},
    {"nframes": N_FRAMES},
    {"video_start": 1.0, "video_end": 4.0},
    {"video_start": 2.5, "fps": 4.0},
    {"video_end": 2.0, "nframes": 6},
], ids = ["default", "fps", "nframes", "all_frames", "range", "late_start", "early_end"])
def test_sampled_read_equals_full_read_then_index(fallback, clip, extra):
    ele = {"video": clip, **extra}
    expected, expected_fps = _full_read_then_sample(ele)
    video, sample_fps = vision_utils._read_video_torchvision(ele)
    assert video.dtype == expected.dtype == torch.uint8
    assert video.shape == expected.shape and video.is_contiguous()
    assert torch.equal(video, expected)
    assert sample_fps == expected_fps


def test_an_empty_range_still_raises_like_the_full_read(fallback, clip):
    ele = {"video": clip, "video_start": 100.0}
    with pytest.raises(ValueError, match = "nframes should in interval"):
        _full_read_then_sample(ele)
    with pytest.raises(ValueError, match = "nframes should in interval"):
        vision_utils._read_video_torchvision(ele)


class _CountingFrame:
    def __init__(self, frame, counter):
        self._frame, self._counter = frame, counter

    @property
    def time(self):
        return self._frame.time

    def to_ndarray(self, *args, **kwargs):
        self._counter[0] += 1
        return self._frame.to_ndarray(*args, **kwargs)


class _CountingContainer:
    def __init__(self, container, counter):
        self._container, self._counter = container, counter

    def __getattr__(self, name):
        return getattr(self._container, name)

    def __enter__(self):
        self._container.__enter__()
        return self

    def __exit__(self, *exc):
        return self._container.__exit__(*exc)

    def decode(self, *args, **kwargs):
        for frame in self._container.decode(*args, **kwargs):
            yield _CountingFrame(frame, self._counter)


def test_only_the_sampled_frames_are_converted_to_rgb(fallback, clip, monkeypatch):
    counter = [0]
    real_open = av.open
    monkeypatch.setattr(av, "open", lambda *a, **k: _CountingContainer(real_open(*a, **k), counter))
    video, _ = vision_utils._read_video_torchvision({"video": clip, "nframes": 8})
    assert video.shape[0] == 8
    # One RGB conversion per kept frame; the full read converted all N_FRAMES.
    assert counter[0] == 8, counter[0]


def test_peak_rgb_host_memory_is_the_sampled_frames(fallback, clip):
    # numpy reports its buffers to tracemalloc: the RGB frame list and the stacked array of the full read.
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        video, _ = vision_utils._read_video_torchvision({"video": clip, "nframes": 8})
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    one_rgb_video = N_FRAMES * SIDE * SIDE * 3
    assert video.shape[0] == 8
    assert peak < one_rgb_video // 2, (peak, one_rgb_video)
