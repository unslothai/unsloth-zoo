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
import sys 
import warnings
import os
import logging
from functools import lru_cache
from typing import Optional

import requests
import torchvision
from packaging import version
from typing import Union, Tuple, List, Dict
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
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
) -> Tuple[int, int]:
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
    ele: Dict,
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
    elif image.startswith("http://") or image.startswith("https://"):
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
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
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
    video_fps: int | float,
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
            logger.warning(f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]")
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
    return nframes


def _read_video_torchvision(
    ele: dict,
) -> (torch.Tensor, float):
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
    total_frames, video_fps = video.size(0), info["video_fps"]
    logger.info(f"torchvision:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
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

    logger.info(f"calculate video frame range: {start_frame=}, {end_frame=}, {total_frames=} from {video_start=}, {video_end=}, {video_fps=:.3f}")
    return start_frame, end_frame, end_frame - start_frame + 1


def _read_video_decord(
    ele: dict,
) -> (torch.Tensor, float):
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
    logger.info(f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
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
) -> (torch.Tensor, float):
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
    logger.info(f"set TORCHCODEC_NUM_THREADS: {TORCHCODEC_NUM_THREADS}")
    video_path = ele["video"]
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
    logger.info(f"torchcodec:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    return video, sample_fps


VIDEO_READER_BACKENDS = {
    "decord": _read_video_decord,
    "torchvision": _read_video_torchvision,
    "torchcodec": _read_video_torchcodec,
}

FORCE_QWENVL_VIDEO_READER = os.getenv("FORCE_QWENVL_VIDEO_READER", None)


@lru_cache(maxsize=1)
def get_video_reader_backend() -> str:
    if FORCE_QWENVL_VIDEO_READER is not None:
        video_reader_backend = FORCE_QWENVL_VIDEO_READER
    elif is_torchcodec_available():
        video_reader_backend = "torchcodec"
    elif is_decord_available():
        video_reader_backend = "decord"
    else:
        video_reader_backend = "torchvision"
    print(f"qwen-vl-utils using {video_reader_backend} to read video.", file=sys.stderr)
    return video_reader_backend


def fetch_video(ele: dict, image_factor: int = IMAGE_FACTOR, return_video_sample_fps: bool = False) -> torch.Tensor | list[Image.Image]:
    if isinstance(ele["video"], str):
        video_reader_backend = get_video_reader_backend()
        try:
            video, sample_fps = VIDEO_READER_BACKENDS[video_reader_backend](ele)
        except Exception as e:
            logger.warning(f"video_reader_backend {video_reader_backend} error, use torchvision as default, msg: {e}")
            video, sample_fps = VIDEO_READER_BACKENDS["torchvision"](ele)

        nframes, _, height, width = video.shape
        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
        max_pixels_supposed = ele.get("max_pixels", max_pixels)
        
        if max_pixels_supposed > max_pixels:
            logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
        
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
        ).float()
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
    return_video_kwargs: bool = False,
) -> Tuple[Union[List[Image.Image], None], Union[List[Union[torch.Tensor, List[Image.Image]]], None]]:
    
    vision_infos = extract_vision_info(conversations)
   
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []

    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info:
            video_input, video_sample_fps = fetch_video(vision_info, return_video_sample_fps=True)
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
    __slots__ = \
        "padding_token_ids", "dtype", "ignore_index", \
        "processor", "formatting_func", "image_size", \
        "max_seq_length", "truncation", "train_on_responses_only", \
        "num_proc", "assistant_single_content",

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
    ):
        if not hasattr(processor, "image_processor"):
            raise TypeError("Unsloth: UnslothVisionDataCollator is only for image models!")

        self.padding_token_ids = get_padding_tokens_ids(processor)
        self.dtype = _get_dtype(
            model.config.torch_dtype \
            if hasattr(model.config, "torch_dtype") else \
            model.get_input_embeddings().weight.dtype
        )
        self.ignore_index = ignore_index
        self.processor = processor
        self.formatting_func = formatting_func

        # Auto resize images to save VRAM!
        if resize == "min":
            try:
                self.image_size = model.config.vision_config.image_size
            except:
                print("Unsloth: Model does not have a default image size - using 512")
                self.image_size = 512
        elif resize == "max":
            self.image_size = None
        elif type(resize) is tuple or type(resize) is list:
            assert(len(resize) == 2)
            assert(type(resize[0]) is int and type(resize[1]) is int)
            self.image_size = tuple(resize)
        elif type(resize) is int:
            self.image_size = resize
        else:
            raise TypeError(
                "Unsloth: resize accepts 'min', 'max', a tuple of 2 numbers or 1 number\n"\
                "For example (224, 224) or just 224. The default is 'min' which auto resizes images!"
            )
        pass

        # Sequence lengths
        if max_seq_length is None:
            if hasattr(model, "max_seq_length"): max_seq_length = model.max_seq_length
        self.max_seq_length = max(max_seq_length, 0) if type(max_seq_length) is int else None
        self.truncation = self.max_seq_length is not None

        # Train on reponses if provided
        if train_on_responses_only:
            assert(type(instruction_part) is str and type(response_part) is str)
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
        texts  = []
        images = []
        videos = []
        video_kwargs = None

        if self.formatting_func is not None:
            examples = [self.formatting_func(example) for example in examples]

        for example in examples:
            if "messages" in example:
                messages = example["messages"]
            elif "conversations" in example:
                messages = example["conversations"]
            else:
                messages = example

            # Check if data format is correct for VLMs!
            if len(messages) != 0:
                message = messages[0]
                assert(type(message) is dict)
                if "role" not in message and "content" not in message:
                    raise TypeError(
                        "Unsloth: Failed to use vision data collator!\n"\
                        "Maybe use `standardize_data_formats` first!"
                    )
                content = message["content"]
                if type(content) is str:
                    message["content"] = content = [{"type" : "text", "text" : content}]
                elif type(content) is list or type(content) is tuple:
                    part = content[0]
                    assert("type" in part)
                else:
                    raise TypeError(
                        "Unsloth: Failed to use vision data collator!\n"\
                        "Your messages must be a like:\n"\
                        "[{'role':'user', 'content':[{'type':'text', 'text':'Hello!'}]}]"
                    )
                pass

                # Also fix the messages if assistant must only be 1 string!
                # Only affects Mistral V3 I think!
                if self.assistant_single_content:
                    for message in messages:
                        if message["role"] == "assistant":
                            if type(content := message["content"]) is list:
                                message["content"] = content[0]["text"]
                pass
            pass
            message = self.processor.apply_chat_template(
                messages,
                tokenize = False,
                add_generation_prompt = False,
            )
            texts.append(message)
            # Dataset with 2 columns messages / images
            if "images" in example:
                image = [example["images"][0]]
            else:
                image, video, video_kwarg = process_vision_info(messages, return_video_kwargs=True)
                if image is None: image = []
                elif video is None: video = [] 
            pass
            # Resize images
            image_size = self.image_size

            if image_size is not None:
                for i, img in enumerate(image):
                    if type(image_size) is tuple:
                        image[i] = img.resize(image_size, LANCZOS)
                    elif img.size[0] > image_size:
                        if hasattr(img, "resize"):
                            wpercent = image_size / img.size[0]
                            hsize = int(img.size[1] * wpercent)
                            image[i] = img.resize((image_size, hsize), LANCZOS)
            pass
            images.append(image)

            if len(video) > 0: 
                videos.append(video)
                if video_kwarg is None: 
                    video_kwarg = {"fps": []}
                video_kwarg['fps'].extend(video_kwargs['fps'])
        pass

        # Tokenize the texts and process the images
        batch = self.processor(
            text    = texts,
            images  = images,
            videos  = videos,
            padding = True,
            truncation = self.truncation,
            max_length = self.max_seq_length,
            return_tensors = "pt",
            add_special_tokens = False, # Stop double BOS
            **video_kwargs
        )
        # Cannot remove due to bidirectional attention from Gemma 3!
        # batch.pop("token_type_ids", None)

        # Pixtral accepts multiple images, so we have to cast it individually
        pixel_values = batch["pixel_values"]
        if type(pixel_values) is list:
            for j, pixel_value_j in enumerate(pixel_values):
                if type(pixel_value_j) is list:
                    for k, pixel_value_k in enumerate(pixel_value_j):
                        pixel_value_j[k] = pixel_value_k.to(self.dtype)
                else:
                    pixel_values[j] = pixel_value_j.to(self.dtype)
            pass
            batch["pixel_values"] = pixel_values
        else:
            batch["pixel_values"] = batch["pixel_values"].to(self.dtype)
        pass

        # Mask image tokens and pad tokens
        labels = batch["input_ids"].clone()
        labels[torch.isin(labels, self.padding_token_ids)] = self.ignore_index
        batch["labels"] = labels
        if self.train_on_responses_only:
            batch["labels"] = self.train_on_responses_only(batch)["labels"]
        return batch
    pass
pass
