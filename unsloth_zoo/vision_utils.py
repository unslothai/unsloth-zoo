# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Copyright 2024-present QwenLM team https://github.com/QwenLM/Qwen2-VL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = [
    "process_vision_info",
    "UnslothVisionDataCollator",
]

global IMAGE_TOKENS
IMAGE_TOKENS = [
    "<|image|>",          # Llama 3.2 Vision, Phi 3.5
    "<|vision_start|>",   # Qwen
    "<|vision_end|>",     # Qwen
    "<|vision_pad|>",     # Qwen
    "<|image_pad|>",      # Qwen
    "<|video_pad|>",      # Qwen
    "<image>",            # PaliGemma / Llava
    "[IMG]",              # Mistral
    "[IMG_BREAK]",        # Mistral
    "[IMG_END]",          # Mistral
    "<image_soft_token>", # Gemma 3
    "<start_of_image>",   # Gemma 3
    "<end_of_image>",     # Gemma 3
    "<|START_OF_IMG|>",   # Cohere
    "<|END_OF_IMG|>",     # Cohere
    "<|IMG_LINE_BREAK|>", # Cohere
    "<|IMG_PATCH|>",      # Cohere
]

import torch
from PIL import Image
import base64
from io import BytesIO
import math
import time
import warnings
import os
from functools import lru_cache


import requests
import torchvision
from packaging import version
from typing import Union, Tuple, List, Dict, Sequence
from itertools import takewhile
try:
    from torchvision import io, transforms
    from torchvision.transforms import InterpolationMode
    HAS_TORCHVISION = True
except:
    HAS_TORCHVISION = False

from .log import logger

from .hf_utils import dtype_from_config, HAS_TORCH_DTYPE

UNSLOTH_ENABLE_LOGGING  = os.environ.get("UNSLOTH_ENABLE_LOGGING",  "0") == "1"

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = int(float(os.environ.get('VIDEO_MAX_PIXELS', 128000 * 28 * 28 * 0.9)))
if UNSLOTH_ENABLE_LOGGING:
    logger.info(f"Unsloth: set VIDEO_TOTAL_PIXELS: {VIDEO_TOTAL_PIXELS}")
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor
pass

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor
pass

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor
pass


def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar
pass


def fetch_image(
    ele: dict,
    size_factor: int = IMAGE_FACTOR,
) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
        if isinstance(image, dict) and "url" in image:
            image = image["url"]
    pass
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image_obj = Image.open(requests.get(image, stream=True).raw)
        elif image.startswith("file://"):
            image_obj = Image.open(image[7:])
        elif image.startswith("data:image"):
            if "base64," in image:
                _, base64_data = image.split("base64,", 1)
                data = base64.b64decode(base64_data)
                image_obj = Image.open(BytesIO(data))
        else:
            image_obj = Image.open(image)
    elif isinstance(image, bytes):
        image_obj = Image.open(BytesIO(image))
    elif isinstance(image, dict):
        if "bytes" in image and image["bytes"]:
            image_obj = Image.open(BytesIO(image["bytes"]))
        elif "path" in image and image["path"]:
            image_obj = Image.open(image["path"])
        elif "url" in image and image["url"]:
            image_obj = Image.open(requests.get(image["url"], stream=True).raw)

    if image_obj is None:
        raise ValueError(f"Unrecognized image input. We support local path, http url, base64 and PIL.Image, bytes and dict formats. Instead we got `{type(image).__name__}`")
    image = image_obj.convert("RGB")
    ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))

    return image
pass

def smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps: Union[int, float],
) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
        nframes = total_frames / video_fps * fps
        if nframes > total_frames:
            if UNSLOTH_ENABLE_LOGGING:
                logger.warning(f"Unsloth: smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]")
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
    return nframes


def _read_video_torchvision(
    ele: dict,
) -> tuple[torch.Tensor, float]:
    """read video using torchvision.io.read_video

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    video_path = ele["video"]
    if version.parse(torchvision.__version__) < version.parse("0.19.0"):
        if "http://" in video_path or "https://" in video_path:
            warnings.warn("torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0.")
        if "file://" in video_path:
            video_path = video_path[7:]
    st = time.time()
    video, audio, info = io.read_video(
        video_path,
        start_pts=ele.get("video_start", 0.0),
        end_pts=ele.get("video_end", None),
        pts_unit="sec",
        output_format="TCHW",
    )
    try:
        video_fps = info["video_fps"]
    except Exception as e:
        print('error getting video_fps there is probably a path issue', e)
        video_fps = 2.0
    total_frames, video_fps = video.size(0), video_fps
    if UNSLOTH_ENABLE_LOGGING:
        logger.info(f"Unsloth: torchvision:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long()
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    video = video[idx]
    return video, sample_fps


def is_decord_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("decord") is not None


def calculate_video_frame_range(
    ele: dict,
    total_frames: int,
    video_fps: float,
) -> tuple[int, int, int]:
    """
    Calculate the start and end frame indices based on the given time range.

    Args:
        ele (dict): A dictionary containing optional 'video_start' and 'video_end' keys (in seconds).
        total_frames (int): Total number of frames in the video.
        video_fps (float): Frames per second of the video.

    Returns:
        tuple: A tuple containing (start_frame, end_frame, frame_count).

    Raises:
        ValueError: If input parameters are invalid or the time range is inconsistent.
    """
    # Validate essential parameters
    if video_fps <= 0:
        raise ValueError("video_fps must be a positive number")
    if total_frames <= 0:
        raise ValueError("total_frames must be a positive integer")

    # Get start and end time in seconds
    video_start = ele.get("video_start", None)
    video_end = ele.get("video_end", None)
    if video_start is None and video_end is None:
        return 0, total_frames - 1, total_frames

    max_duration = total_frames / video_fps
    # Process start frame
    if video_start is not None:
        video_start_clamped = max(0.0, min(video_start, max_duration))
        start_frame = math.ceil(video_start_clamped * video_fps)
    else:
        start_frame = 0
    # Process end frame
    if video_end is not None:
        video_end_clamped = max(0.0, min(video_end, max_duration))
        end_frame = math.floor(video_end_clamped * video_fps)
        end_frame = min(end_frame, total_frames - 1)
    else:
        end_frame = total_frames - 1

    # Validate frame order
    if start_frame >= end_frame:
        raise ValueError(
            f"Invalid time range: Start frame {start_frame} (at {video_start_clamped if video_start is not None else 0}s) "
            f"exceeds end frame {end_frame} (at {video_end_clamped if video_end is not None else max_duration}s). "
            f"Video duration: {max_duration:.2f}s ({total_frames} frames @ {video_fps}fps)"
        )

    if UNSLOTH_ENABLE_LOGGING:
        logger.info(f"Unsloth: calculate video frame range: {start_frame=}, {end_frame=}, {total_frames=} from {video_start=}, {video_end=}, {video_fps=:.3f}")
    return start_frame, end_frame, end_frame - start_frame + 1


def _read_video_decord(
    ele: dict,
) -> tuple[torch.Tensor, float]:
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    import decord
    video_path = ele["video"]
    st = time.time()
    vr = decord.VideoReader(video_path)
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    start_frame, end_frame, total_frames = calculate_video_frame_range(
        ele,
        total_frames,
        video_fps,
    )
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
    video = vr.get_batch(idx).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    if UNSLOTH_ENABLE_LOGGING:
        logger.info(f"Unsloth: decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    return video, sample_fps


def is_torchcodec_available() -> bool:
    """Check if torchcodec is available and properly installed."""
    try:
        import importlib.util
        if importlib.util.find_spec("torchcodec") is None:
            return False
        from torchcodec.decoders import VideoDecoder
        return True
    except (ImportError, AttributeError, Exception):
        return False


def _read_video_torchcodec(
    ele: dict,
) -> tuple[torch.Tensor, float]:
    """read video using torchcodec.decoders.VideoDecoder

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    from torchcodec.decoders import VideoDecoder
    TORCHCODEC_NUM_THREADS = int(os.environ.get('TORCHCODEC_NUM_THREADS', 8))
    if UNSLOTH_ENABLE_LOGGING:
        logger.info(f"Unsloth: set TORCHCODEC_NUM_THREADS: {TORCHCODEC_NUM_THREADS}")
    video_path = ele["video"]
    # Support file URI scheme
    if isinstance(video_path, str) and video_path.startswith("file://"):
        video_path = video_path[7:]
    st = time.time()
    decoder = VideoDecoder(video_path, num_ffmpeg_threads=TORCHCODEC_NUM_THREADS)
    video_fps = decoder.metadata.average_fps
    total_frames = decoder.metadata.num_frames
    start_frame, end_frame, total_frames = calculate_video_frame_range(
        ele,
        total_frames,
        video_fps,
    )
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    video = decoder.get_frames_at(indices=idx).data
    # Ensure channel-first layout (T, C, H, W). torchcodec returns NHWC.
    if hasattr(video, "ndim") and video.ndim == 4:
        # If second dim looks like height and last dim like channels, permute to TCHW
        if video.shape[-1] in (1, 3, 4) and video.shape[1] not in (1, 3, 4):
            video = video.permute(0, 3, 1, 2).contiguous()
    if UNSLOTH_ENABLE_LOGGING:
        logger.info(f"Unsloth: torchcodec:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    return video, sample_fps


VIDEO_READER_BACKENDS = {
    "decord": _read_video_decord,
    "torchvision": _read_video_torchvision,
    "torchcodec": _read_video_torchcodec,
}

FORCE_UNSLOTH_VIDEO_READER = os.getenv("FORCE_UNSLOTH_VIDEO_READER", None)


@lru_cache(maxsize=1)
def get_video_reader_backend() -> str:
    if FORCE_UNSLOTH_VIDEO_READER is not None:
        video_reader_backend = FORCE_UNSLOTH_VIDEO_READER
    elif is_decord_available():
        video_reader_backend = "decord"
    elif is_torchcodec_available():
        video_reader_backend = "torchcodec"
    elif HAS_TORCHVISION:
        video_reader_backend = "torchvision"
    else:
        raise ValueError("Unsloth: No video reader backend available, please install decord or torchvision or torchcodec to process video inputs.")
    if UNSLOTH_ENABLE_LOGGING:
        logger.info(f"Unsloth: unsloth_zoo/vision_utils using {video_reader_backend} to read video.")
    return video_reader_backend


def fetch_video(ele: dict, image_factor: int = IMAGE_FACTOR, return_video_sample_fps: bool = False) -> Union[torch.Tensor, list[Image.Image]]:
    if isinstance(ele["video"], str):
        video_reader_backend = get_video_reader_backend()
        try:
            video, sample_fps = VIDEO_READER_BACKENDS[video_reader_backend](ele)
        except Exception as e:
            if UNSLOTH_ENABLE_LOGGING:
                logger.warning(f"Unsloth: video_reader_backend {video_reader_backend} error, use torchvision as default, msg: {e}")
            video, sample_fps = VIDEO_READER_BACKENDS["torchvision"](ele)

        nframes, _, height, width = video.shape
        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
        max_pixels_supposed = ele.get("max_pixels", max_pixels)

        if max_pixels_supposed > max_pixels:
            if UNSLOTH_ENABLE_LOGGING:
                logger.warning(f"Unsloth: The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")

        max_pixels = min(max_pixels_supposed, max_pixels)

        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        if return_video_sample_fps:
            return video, sample_fps
        return video
    else:
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        images = [
            fetch_image({"image": video_element, **process_info}, size_factor=image_factor)
            for video_element in ele["video"]
   ]
        nframes = ceil_by_factor(len(images), FRAME_FACTOR)
        if len(images) < nframes:
            images.extend([images[-1]] * (nframes - len(images)))
        if return_video_sample_fps:
            return images, process_info.pop("fps", 2.0)
        return images

def collapse_fps(fps, tol=1e-4):
    """Return a single float if all fps equal (within tol), else a list; pass None through."""
    if fps is None:
        return None
    if isinstance(fps, (int, float)):
        return float(fps)
    vals = [float(v) for v in fps]
    if not vals:
        return None
    f0 = vals[0]
    return float(f0) if all(math.isclose(v, f0, rel_tol=tol, abs_tol=tol) for v in vals[1:]) else vals


def extract_vision_info(conversations: Union[List[Dict], List[List[Dict]]]) -> List[Dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] in ("image", "image_url", "video"):
                        vision_infos.append(ele)
    return vision_infos
pass


def process_vision_info(
    conversations: Union[List[Dict], List[List[Dict]]],
    size_factor: int = IMAGE_FACTOR,
    return_video_kwargs: bool = False,
) -> Tuple[Union[List[Image.Image], None], Union[List[Union[torch.Tensor, List[Image.Image]]], None]]:
    
    vision_infos = extract_vision_info(conversations)
   
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []

    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info, size_factor=size_factor))
        elif "video" in vision_info:
            video_input, video_sample_fps = fetch_video(vision_info, image_factor=size_factor, return_video_sample_fps=True)
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    if return_video_kwargs:
        return image_inputs, video_inputs, {'fps': video_sample_fps_list}
    return image_inputs, video_inputs
pass


def get_padding_tokens_ids(tokenizer):
    global IMAGE_TOKENS

    tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    image_tokens = IMAGE_TOKENS
    if hasattr(tokenizer, "image_token"):
        image_tokens = IMAGE_TOKENS + [tokenizer.image_token]
    pass

    padding_token_ids = tokenizer.convert_tokens_to_ids(image_tokens)
    if hasattr(tokenizer, "pad_token_id"):
        padding_token_ids.append(tokenizer.pad_token_id)
    pass

    padding_token_ids = list(x for x in padding_token_ids if x is not None)
    padding_token_ids = list(set(padding_token_ids))
    padding_token_ids = torch.IntTensor(padding_token_ids)
    return padding_token_ids
pass


def _get_dtype(dtype):
    __DTYPE_MAP = {
        "float32": torch.float32,
        torch.float32: torch.float32,
        "float16": torch.float16,
        torch.float16: torch.float16,
        "bfloat16": torch.bfloat16,
        torch.bfloat16: torch.bfloat16,
    }
    if   dtype is None or dtype == None: return None
    elif dtype in __DTYPE_MAP: return __DTYPE_MAP[dtype]
    else:
        print(f"Unsloth: {dtype} is not recognized, so we'll default to None")
        return None
    pass
pass

import PIL.Image
LANCZOS = PIL.Image.Resampling.LANCZOS
from .dataset_utils import train_on_responses_only as _train_on_responses_only

class UnslothVisionDataCollator:
    # All Unsloth Zoo code licensed under LGPLv3
    __slots__ = (
        "padding_token_ids", "dtype", "ignore_index",
        "processor", "formatting_func", "image_size",
        "max_seq_length", "truncation", "train_on_responses_only",
        "num_proc", "assistant_single_content", "patch_size",
        "resize_dimension", "snap_to_patch_size",
        "completion_only_loss", "pad_to_multiple_of", "size_func",
    )

    def __init__(
        self,
        model,
        processor,
        max_seq_length  = None,
        formatting_func = None,
        resize = "min", # Can be (10, 10) or "min" to resize to fit
                        # the model's default image_size or "max"
                        # for no resizing and leave image intact
        ignore_index = -100,
        train_on_responses_only = False,
        instruction_part = None,
        response_part    = None,
        force_match      = True, # Match newlines as well!
        num_proc         = None,
        completion_only_loss = True,
        pad_to_multiple_of = None,
        resize_dimension = 0, # can be 0, 1, 'max' or 'min' (max resizes based on the max of height width, min the min size, 0 the first dim, etc)
        snap_to_patch_size = False,
    ):
        if not hasattr(processor, "image_processor"):
            raise TypeError("Unsloth: UnslothVisionDataCollator is only for image models!")

        self.padding_token_ids = get_padding_tokens_ids(processor)
        self.dtype = _get_dtype(
            dtype_from_config(model.config)
            if HAS_TORCH_DTYPE else
            model.get_input_embeddings().weight.dtype
        )
        self.ignore_index = ignore_index
        self.processor = processor
        self.formatting_func = formatting_func
        self.completion_only_loss = completion_only_loss
        self.pad_to_multiple_of = pad_to_multiple_of
        self.snap_to_patch_size = snap_to_patch_size
        try:
            self.patch_size = model.config.vision_config.patch_size
        except:
            if hasattr(model.config, 'vision_config') and hasattr(model.config.vision_config, 'model_type'):
                lower_name = model.config.vision_config.model_type.lower()
                if 'gemma3n' in lower_name or 'pixtral' in lower_name:
                    self.patch_size = 16 #  really gemma3n doesn't have a patch size but expects images in 256, 512, or 768
                else:
                    self.patch_size = IMAGE_FACTOR // 2
            else:
                self.patch_size = IMAGE_FACTOR // 2

        # Auto resize images to save VRAM!
        if resize == "min":
            try:
                self.image_size = model.config.vision_config.image_size
            except:
                print("Unsloth: Model does not have a default image size - using 512")
                self.image_size = 512

        elif resize == "max":
            self.image_size = None
        elif isinstance(resize, (tuple, list)):
            assert(len(resize) == 2)
            assert(isinstance(resize[0], int) and isinstance(resize[1], int))
            self.image_size = tuple(resize)
        elif type(resize) is int:
            self.image_size = resize
        else:
            raise TypeError(
                "Unsloth: resize accepts 'min', 'max', a tuple of 2 numbers or 1 number\n"\
                "For example (224, 224) or just 224. The default is 'min' which auto resizes images!"
            )
        pass
        if resize_dimension not in [0, 1, 'max', 'min']:
            raise TypeError(
                "Unsloth: resize_dimension accepts 0, 1, 'max' or 'min'\n"\
                "For example 0 resizes the first dimension, 1 the second, 'max' resizes based on the max of height width, 'min' the min size"
            )
        elif resize_dimension in [0, 1]:
            self.size_func = lambda x: x.size[resize_dimension]
        elif resize_dimension == 'max':
            self.size_func = lambda x: max(x.size[0], x.size[1])
        elif resize_dimension == 'min':
            self.size_func = lambda x: min(x.size[0], x.size[1])
        else:
            raise TypeError(
                "Unsloth: resize_dimension accepts 0, 1, 'max' or 'min'\n"\
                "For example 0 resizes the first dimension, 1 the second, 'max' resizes based on the max of height width, 'min' the min size"
            )
        self.resize_dimension = resize_dimension

        # Sequence lengths
        if max_seq_length is None:
            if hasattr(model, "max_seq_length"): max_seq_length = model.max_seq_length
        self.max_seq_length = max(max_seq_length, 0) if type(max_seq_length) is int else None
        self.truncation = self.max_seq_length is not None

        # Train on reponses if provided
        if train_on_responses_only:
            assert(isinstance(instruction_part, str) and isinstance(response_part, str))
            self.train_on_responses_only = _train_on_responses_only(
                None,
                instruction_part = instruction_part,
                response_part    = response_part,
                force_match      = force_match,
                tokenizer        = processor,
                return_function  = True,
                num_proc         = num_proc,
            )
        else:
            self.train_on_responses_only = None

        # Check what type for assistant VLM tokenizer allows!
        # Good for Mistral V3 and Pixtral I think
        try:
            processor.apply_chat_template([
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Hello!"}]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "How can I help you?"}]}
            ])
            self.assistant_single_content = False
        except TypeError:
            try:
                processor.apply_chat_template([
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Hello!"}]},
                    {"role": "assistant", "content": "How can I help you?"}
                ])
                self.assistant_single_content = True
                print(
                    f"Unsloth: {processor.__class__.__name__} only accepts 1 "\
                    "text field for assistant roles!\n"\
                    "We will auto fix the data collator to support it!"
                )
            except Exception as e:
                raise RuntimeError(e)
        except Exception as e:
            raise RuntimeError(e)
        return
    pass

    def __call__(self, examples):
        # [TODO] Support non image inputs as well
        # The issue is batch = self.processor( forces tensors to be returned and not None.

        if self.formatting_func is not None:
            examples = [self.formatting_func(example) for example in examples]
        
        if "prompt" in examples[0] and "completion" in examples[0]:
            return self._collate_prompt_completion(examples)

        texts  = []
        images = []
        videos = []
        video_kwargs = {'fps': []}
        for example in examples:
            messages = self._select_messages_or_raw(example)

            # Check if data format is correct for VLMs!
            if len(messages) != 0:
                messages = self._validate_and_normalize_first_message(messages)

                # Also fix the messages if assistant must only be 1 string!
                # Only affects Mistral V3 I think!
                if self.assistant_single_content:
                    messages = self._collapse_assistant_content(messages)
                messages = self._clean_none_keys(messages)
            pass

            message = self.processor.apply_chat_template(
                messages,
                tokenize = False,
                add_generation_prompt = False,
            )
            texts.append(message)
            # Dataset with 2 columns messages / images
            image, video, video_kwarg = self._extract_images_videos_for_example(example, messages)
            image = self._resize_images_inplace(image)
            if len(image) > 0:
                images.append(image)

            if len(video) > 0:  # Works for list, tuple or tensor
                videos.append(video)
                if video_kwarg is None:
                    video_kwarg = {"fps": []}
                video_kwargs['fps'].extend(video_kwarg['fps'])
        pass

        # Tokenize the texts and process the images
        proc_kwargs = dict(
            text=texts,
            padding=True,
            truncation=self.truncation,
            max_length=self.max_seq_length,
            return_tensors="pt",
            add_special_tokens=False,
        )
        if images and len(images) > 0:
            proc_kwargs["images"] = images
        if videos and len(videos) > 0:
            proc_kwargs["videos"] = videos
            video_kwargs["fps"] = collapse_fps(video_kwargs['fps'])
            for k, v in video_kwargs.items():
                proc_kwargs[k] = v
        if self.pad_to_multiple_of is not None:
            proc_kwargs["pad_to_multiple_of"] = self.pad_to_multiple_of
        batch = self.processor(**proc_kwargs)

        # Cannot remove due to bidirectional attention from Gemma 3!
        # batch.pop("token_type_ids", None)
        if 'pixel_values' in batch:
            batch = self._cast_pixel_values_dtype_inplace(batch)
        if 'pixel_values_videos' in batch:
            batch = self._cast_pixel_values_dtype_inplace(batch, 'pixel_values_videos')

        # Mask image tokens and pad tokens
        labels = batch["input_ids"].clone()
        labels[torch.isin(labels, self.padding_token_ids)] = self.ignore_index
        batch["labels"] = labels
        if self.train_on_responses_only:
            batch["labels"] = self.train_on_responses_only(batch)["labels"]
        return batch
    pass

    def _select_messages_or_raw(self, example):
        if "messages" in example:
            return example["messages"]
        elif "conversations" in example:
            return example["conversations"]
        else:
            # original behavior: allow the example itself to be the messages list
            return example

    def _validate_and_normalize_first_message(self, messages):
        if len(messages) == 0:
            return
        message = messages[0]
        assert isinstance(message, dict)
        if "role" not in message and "content" not in message:
            raise TypeError(
                "Unsloth: Failed to use vision data collator!\n"
                "Maybe use `standardize_data_formats` first!"
            )
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = [{"type": "text", "text": content}]
        elif isinstance(content, (list, tuple)):
            part = content[0]
            assert "type" in part
        else:
            raise TypeError(
                "Unsloth: Failed to use vision data collator!\n"
                "Your messages must be like:\n"
                "[{'role':'user', 'content':[{'type':'text', 'text':'Hello!'}]}]"
            )
        return messages

    def _collapse_assistant_content(self, messages):
        for message in messages:
            if message["role"] == "assistant":
                if isinstance(content := message["content"], list):
                    message["content"] = content[0]["text"]
        return messages

    def _render_chat(self, prompt_messages, completion_messages=None, add_generation_prompt=False, continue_final_message=False):
        return self.processor.apply_chat_template(
            prompt_messages + (completion_messages or []), tokenize=False, add_generation_prompt=add_generation_prompt, continue_final_message=continue_final_message
        )

    def _extract_images_videos_for_example(self, example, messages):
        if "images" in example:
            image = list(example["images"])
            video = []
            video_kwarg = None
        else:
            image, video, video_kwarg = process_vision_info(
                messages,
                size_factor=self.patch_size*2,
                return_video_kwargs=True,
            )
            if image is None: image = []
            if video is None: video = [] 
        pass
        return image, video, video_kwarg

    def _extract_images_for_pc(self, example, p_msgs, c_msgs):
        # PC: prefer embedded across prompt+completion; else top-level first image; else []
        imgs = None
        vids = None
        vids_kwarg = None
        try:
            msg_list = (p_msgs or []) + (c_msgs or [])
            if msg_list:
                imgs, vids, vids_kwarg = process_vision_info(
                    msg_list,
                    size_factor=self.patch_size*2,
                    return_video_kwargs=True,
                )
                if imgs is None: imgs = []
                if vids is None: vids = []
            else:
                if "images" in example:
                    vision_infos = [{'image': example['images'][i]} for i in range(len(example['images']))]
                    imgs, vids, vids_kwarg = process_vision_info(
                        vision_infos,
                        size_factor=self.patch_size*2,
                        return_video_kwargs=True,
                    )
                    if imgs is None: imgs = []
                    if vids is None: vids = []
        except Exception:
            imgs = []
            vids = []
        
        return imgs, vids, vids_kwarg

    def _resize_images_inplace(self, image):

        def quantize_to_factor(x):
            return max(factor, int(factor * (
                math.ceil(x/factor) if x >= image_size - 1e-6 else math.floor(x/factor+0.5)
            )))

        if image is None or self.image_size is None or (isinstance(self.image_size, (tuple, list)) and len(self.image_size) == 0):
            return image or []
        # Resize images
        image_size = self.image_size

        if image_size is not None:
            for i, img in enumerate(image):
                if type(image_size) is tuple:
                    image[i] = img.resize(image_size, LANCZOS)
                elif self.size_func(img) > image_size and hasattr(img, "resize"):
                    w, h = img.size
                    # integer math rounding
                    new_w = (w * image_size + self.size_func(img) // 2) // self.size_func(img)
                    new_h = (h * image_size + self.size_func(img) // 2) // self.size_func(img)
                    if self.snap_to_patch_size:
                        factor = self.patch_size * 2
                        new_w, new_h = quantize_to_factor(new_w), quantize_to_factor(new_h)

                    image[i] = img.resize((new_w, new_h), LANCZOS)

        return image

    def _cast_pixel_values_dtype_inplace(self, batch, key='pixel_values'):
        # Pixtral accepts multiple images, so we have to cast it individually
        pixel_values = batch[key]
        if type(pixel_values) is list:
            for j, pixel_value_j in enumerate(pixel_values):
                if type(pixel_value_j) is list:
                    for k, pixel_value_k in enumerate(pixel_value_j):
                        pixel_value_j[k] = pixel_value_k.to(self.dtype)
                else:
                    pixel_values[j] = pixel_value_j.to(self.dtype)
            pass
            batch[key] = pixel_values
        else:
            batch[key] = batch[key].to(self.dtype)
        pass
        return batch

    def _tokenizer_padding_side(self) -> str:
        tok = getattr(self.processor, "tokenizer", self.processor)
        side = getattr(tok, "padding_side", "right")
        return "left" if side == "left" else "right"

    def _pad_token_id_or_fail(self) -> int:
        tok = getattr(self.processor, "tokenizer", self.processor)
        pad_id = getattr(tok, "pad_token_id", None)
        if pad_id is None:
            raise ValueError("Tokenizer must define `pad_token_id` for prompt–completion collation.")
        return pad_id

    @torch.no_grad()
    def _flush_to_side(
        self,
        attention_mask: torch.Tensor,
        input_ids:     torch.Tensor,
        side:          str,
        pad_token_id:  int,
        extra_tensors: Sequence[torch.Tensor] | None = None,
        extra_pad_values: Sequence[int | float | bool] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        B, L = input_ids.shape
        device = input_ids.device

        keep = attention_mask.to(device=device, dtype=torch.bool)
        k    = keep.sum(dim=1)
        rank = keep.to(device=device, dtype=torch.int64).cumsum(dim=1) - 1

        if side == "left":
            dst = (L - k).unsqueeze(1) + rank
        elif side == "right":
            dst = rank
        else:
            raise ValueError("side must be 'left' or 'right'")

        new_ids  = input_ids.new_full((B, L), pad_token_id)
        new_attn = attention_mask.new_zeros((B, L))

        ridx, csrc = keep.nonzero(as_tuple=True)
        cdst = dst[ridx, csrc].to(torch.long)

        new_ids[ridx,  cdst] = input_ids[ridx, csrc]
        if new_attn.dtype == torch.bool:
            new_attn[ridx, cdst] = True
        else:
            new_attn[ridx, cdst] = 1

        # Extras move in lock-step; fill pads with provided pad values (default 0)
        new_extras: list[torch.Tensor] = []
        if extra_tensors is not None:
            if extra_pad_values is None:
                extra_pad_values = [0] * len(extra_tensors)
            assert len(extra_pad_values) == len(extra_tensors), "extra_pad_values must match extra_tensors"
            for m, padv in zip(extra_tensors, extra_pad_values):
                out = m.new_full(m.shape, padv)
                out[ridx, cdst] = m[ridx, csrc]
                new_extras.append(out)

        max_k = int(k.max().item())
        if 0 < max_k < L:
            if side == "left":
                sl = slice(L - max_k, L)
            else:
                sl = slice(0, max_k)
            new_ids  = new_ids[:,  sl]
            new_attn = new_attn[:, sl]
            if new_extras:
                new_extras = [e[:, sl] for e in new_extras]

        return new_attn, new_ids, tuple(new_extras)

    def _truncate_by_side(self, input_ids, attention_mask, completion_mask, side, max_len, token_type_ids=None):
        _, L = input_ids.shape
        if L <= max_len:
            return [input_ids, attention_mask, completion_mask] + ([token_type_ids] if token_type_ids is not None else [])
        sl = slice(-max_len, None) if side == "left" else slice(0, max_len)

        input_ids       = input_ids[:, sl]
        attention_mask  = attention_mask[:, sl]
        completion_mask = completion_mask[:, sl]

        if token_type_ids is not None:
            token_type_ids = token_type_ids[:, sl]
        return [input_ids, attention_mask, completion_mask] + ([token_type_ids] if token_type_ids is not None else [])

    def _pad_to_multiple(self, input_ids, attention_mask, completion_mask, side, pad_id, multiple, token_type_ids=None, token_type_pad_id=0):
        B, L = input_ids.shape
        L2 = ((L + multiple - 1) // multiple) * multiple
        if L2 == L:
            return [input_ids, attention_mask, completion_mask] + ([token_type_ids] if token_type_ids is not None else [])
        pad_len = L2 - L

        pad_ids = torch.full((B, pad_len), pad_id, dtype=input_ids.dtype, device=input_ids.device)
        zeros_att = torch.zeros((B, pad_len), dtype=attention_mask.dtype, device=attention_mask.device)
        zeros_comp = torch.zeros((B, pad_len), dtype=completion_mask.dtype, device=completion_mask.device)

        if token_type_ids is not None:
            pad_token_type_ids = torch.full((B, pad_len), token_type_pad_id, dtype=token_type_ids.dtype, device=token_type_ids.device)

        if side == "left":
            input_ids = torch.cat((pad_ids, input_ids), dim=1)
            attention_mask = torch.cat((zeros_att, attention_mask), dim=1)
            completion_mask = torch.cat((zeros_comp, completion_mask), dim=1)
            if token_type_ids is not None:
                token_type_ids = torch.cat((pad_token_type_ids, token_type_ids), dim=1)
        else:
            input_ids = torch.cat((input_ids, pad_ids), dim=1)
            attention_mask = torch.cat((attention_mask, zeros_att), dim=1)
            completion_mask = torch.cat((completion_mask, zeros_comp), dim=1)
            if token_type_ids is not None:
                token_type_ids = torch.cat((token_type_ids, pad_token_type_ids), dim=1)
        return [input_ids, attention_mask, completion_mask] + ([token_type_ids] if token_type_ids is not None else [])

    def _collate_prompt_completion(self, examples):
        prompt_texts, completion_texts, images, videos = [], [], [], []

        for ex in examples:
            p, c = ex["prompt"], ex["completion"]

            # Determine chat vs plain for each side
            is_p_msgs = isinstance(p, list) and (len(p) == 0 or isinstance(p[0], dict))
            is_c_msgs = isinstance(c, list) and (len(c) == 0 or isinstance(c[0], dict))

            if is_p_msgs:
                self._validate_and_normalize_first_message(p)
                if self.assistant_single_content:
                    self._collapse_assistant_content(p)
                p = self._clean_none_keys(p)
                p_txt = self._render_chat(p, add_generation_prompt=True, continue_final_message=False)
            else:
                p_txt = str(p)

            if is_c_msgs:
                self._validate_and_normalize_first_message(c)
                if self.assistant_single_content:
                    self._collapse_assistant_content(c)
                c = self._clean_none_keys(c)
                pc_txt = self._render_chat(prompt_messages=p, completion_messages=c)
                # some models append common template items so this removes them.
                # see trl/data_utils.py
                p_txt = "".join(x for x, _ in takewhile(lambda x: x[0] == x[1], zip(p_txt, pc_txt)))
                c_txt = pc_txt[len(p_txt):]
            else:
                c_txt = str(c)

            # Images: prefer embedded; else first top-level image; else []
            imgs, vids, vids_kwarg = self._extract_images_for_pc(ex, p if is_p_msgs else None, c if is_c_msgs else None)
            imgs = self._resize_images_inplace(imgs)

            prompt_texts.append(p_txt)
            completion_texts.append(c_txt)
            if imgs and len(imgs) > 0:
                images.append(imgs)

            if vids and len(vids) > 0:  # Works for list, tuple or tensor
                videos.append(vids)
                if vids_kwarg is None:
                    vids_kwarg = {"fps": []}
                vids_kwarg['fps'].extend(vids_kwarg['fps'])

        prompt_kwargs = dict(
            padding=True,
            padding_side="left",
            return_tensors="pt",
            add_special_tokens=False,
        )
        completion_kwargs = dict(
            padding=True,
            padding_side="right",
            return_tensors="pt",
            add_special_tokens=False,
        )
        if len(images) > 0:
            prompt_kwargs["images"] = images
        if len(videos) > 0:
            prompt_kwargs["videos"] = videos
            vids_kwarg["fps"] = collapse_fps(vids_kwarg['fps'])
            for k, v in vids_kwarg.items():
                prompt_kwargs[k] = v

        proc_prompts = self.processor(text=prompt_texts, **prompt_kwargs)
        # Encode completions (RIGHT pad) text-only
        proc_completions = self.processor(text=completion_texts, **completion_kwargs)

        p_ids, c_ids = proc_prompts["input_ids"], proc_completions["input_ids"]
        p_m, c_m = proc_prompts["attention_mask"], proc_completions["attention_mask"]
        p_tt, c_tt = proc_prompts.get("token_type_ids", None), proc_completions.get("token_type_ids", None)

        input_ids = torch.cat((p_ids, c_ids), dim=1)
        attention_mask = torch.cat((p_m, c_m), dim=1)
        completion_mask = torch.cat((torch.zeros_like(p_m), c_m), dim=1)
        if p_tt is not None and c_tt is not None:
            token_type_ids = torch.cat((p_tt, c_tt), dim=1)
        else:
            token_type_ids = None

        # Flush to tokenizer default padding side
        pad_id = self._pad_token_id_or_fail()
        flush_side = self._tokenizer_padding_side()
        if token_type_ids is not None:
            attention_mask, input_ids, (completion_mask, token_type_ids) = self._flush_to_side(
                attention_mask, input_ids, flush_side, pad_id, (completion_mask, token_type_ids)
            )

            # Truncate with side awareness
            if self.max_seq_length is not None:
                input_ids, attention_mask, completion_mask, token_type_ids = self._truncate_by_side(
                    input_ids, attention_mask, completion_mask, flush_side, self.max_seq_length, token_type_ids=token_type_ids
                )

            # Optional pad-to-multiple-of (manual in PC)
            if self.pad_to_multiple_of and self.pad_to_multiple_of > 1:
                input_ids, attention_mask, completion_mask, token_type_ids = self._pad_to_multiple(
                    input_ids, attention_mask, completion_mask, flush_side, pad_id, self.pad_to_multiple_of, token_type_ids=token_type_ids
                )
        else:
            attention_mask, input_ids, (completion_mask,) = self._flush_to_side(
                attention_mask, input_ids, flush_side, pad_id, (completion_mask,)
            )

            # Truncate with side awareness
            if self.max_seq_length is not None:
                input_ids, attention_mask, completion_mask = self._truncate_by_side(
                    input_ids, attention_mask, completion_mask, flush_side, self.max_seq_length
                )

            if self.pad_to_multiple_of and self.pad_to_multiple_of > 1:
                input_ids, attention_mask, completion_mask = self._pad_to_multiple(
                    input_ids, attention_mask, completion_mask, flush_side, pad_id, self.pad_to_multiple_of
                )

        # Labels: mask attention pads + image/pad tokens; completion-only if requested
        labels = input_ids.clone()
        labels[attention_mask == 0] = self.ignore_index
        labels[torch.isin(labels, self.padding_token_ids)] = self.ignore_index
        if self.completion_only_loss:
            labels[completion_mask == 0] = self.ignore_index

        # Build output (keep pixel_values from prompt batch) + cast dtype
        out = dict(proc_prompts)
        out["input_ids"] = input_ids
        out["attention_mask"] = attention_mask
        out["labels"] = labels
        if token_type_ids is not None:
            out["token_type_ids"] = token_type_ids
        if 'pixel_values' in out:
            out = self._cast_pixel_values_dtype_inplace(out)
        if 'pixel_values_videos' in out:
            out = self._cast_pixel_values_dtype_inplace(out, 'pixel_values_videos')

        return out

    def _clean_none_keys(self, messages):
        """Remove None-valued keys added by Arrow serialization"""
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        keys_to_remove = [k for k, v in item.items() if v is None]
                        for k in keys_to_remove:
                            del item[k]
        return messages
pass
