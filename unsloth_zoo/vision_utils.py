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
import requests
from typing import Union, Tuple, List, Dict
from .hf_utils import dtype_from_config, HAS_TORCH_DTYPE
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
) -> Tuple[Union[List[Image.Image], None], Union[List[Union[torch.Tensor, List[Image.Image]]], None]]:
    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info, size_factor=size_factor))
        elif "video" in vision_info:
            video_inputs.append(fetch_video(vision_info))
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
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
        completion_only_loss = False,
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
        for example in examples:
            messages = self._select_messages_or_raw(example)

            # Check if data format is correct for VLMs!
            if len(messages) != 0:
                messages = self._validate_and_normalize_first_message(messages)

                # Also fix the messages if assistant must only be 1 string!
                # Only affects Mistral V3 I think!
                if self.assistant_single_content:
                    messages = self._collapse_assistant_content(messages)
            pass

            message = self.processor.apply_chat_template(
                messages,
                tokenize = False,
                add_generation_prompt = False,
            )
            texts.append(message)
            # Dataset with 2 columns messages / images
            image = self._extract_images_for_example(example, messages)
            image = self._resize_images_inplace(image)
            images.append(image)
        pass

        # Tokenize the texts and process the images
        proc_kwargs = dict(
            text=texts,
            images=images,
            padding=True,
            truncation=self.truncation,
            max_length=self.max_seq_length,
            return_tensors="pt",
            add_special_tokens=False,
        )
        if self.pad_to_multiple_of is not None:
            proc_kwargs["pad_to_multiple_of"] = self.pad_to_multiple_of
        batch = self.processor(**proc_kwargs)

        # Cannot remove due to bidirectional attention from Gemma 3!
        # batch.pop("token_type_ids", None)
        batch = self._cast_pixel_values_dtype_inplace(batch)

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

    def _render_chat(self, messages):
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    def _extract_images_for_example(self, example, messages):
        if "images" in example:
            image = [example["images"][0]]
        else:
            image, video = process_vision_info(messages, size_factor=self.patch_size*2)
            if image is None: image = []
        return image

    def _extract_images_for_pc(self, example, p_msgs, c_msgs):
        # PC: prefer embedded across prompt+completion; else top-level first image; else []
        imgs = None
        try:
            msg_list = (p_msgs or []) + (c_msgs or [])
            if msg_list:
                imgs, _ = process_vision_info(msg_list, size_factor=self.patch_size*2)
        except Exception:
            imgs = None
        if imgs is None:
            if "images" in example:
                return [example["images"][0]]
            return []
        return imgs

    def _resize_images_inplace(self, image):

        def quantize_to_factor(x):
            return max(factor, int(factor * (
                math.ceil(x/factor) if x >= image_size - 1e-6 else math.floor(x/factor+0.5)
            )))

        if image is None or self.image_size is None:
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

    def _cast_pixel_values_dtype_inplace(self, batch):
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

    def _flush_to_side(self, attention_mask, input_ids, side, pad_token_id, extra_masks: tuple | list | None = None):
        """Compact non-pad tokens toward `side`. Returns (attn, ids, extras...)."""
        B, L = input_ids.shape
        new_ids = torch.full_like(input_ids, pad_token_id)
        new_attn = torch.zeros_like(attention_mask)
        new_extras = None
        if extra_masks is not None:
            new_extras = [torch.zeros_like(m) for m in extra_masks]
        keep = attention_mask.bool()
        for i in range(B):
            k = int(attention_mask[i].sum().item())
            if k == 0:
                continue
            src = input_ids[i][keep[i]]
            if side == "left":
                new_ids[i, L - k:] = src
                new_attn[i, L - k:] = 1
                if new_extras is not None:
                    for idx, m in enumerate(extra_masks):
                        new_extras[idx][i, L - k:] = m[i][keep[i]]
            else:
                new_ids[i, :k] = src
                new_attn[i, :k] = 1
                if new_extras is not None:
                    for idx, m in enumerate(extra_masks):
                        new_extras[idx][i, :k] = m[i][keep[i]]
        if new_extras is None:
            return new_attn, new_ids, ()
        return new_attn, new_ids, tuple(new_extras)

    def _truncate_by_side(self, input_ids, attention_mask, completion_mask, side, max_len):
        if side == "left":
            input_ids = input_ids[:, -max_len:]
            attention_mask = attention_mask[:, -max_len:]
            completion_mask = completion_mask[:, -max_len:]
        else:
            input_ids = input_ids[:, :max_len]
            attention_mask = attention_mask[:, :max_len]
            completion_mask = completion_mask[:, :max_len]
        return input_ids, attention_mask, completion_mask

    def _pad_to_multiple(self, input_ids, attention_mask, completion_mask, side, pad_id, multiple):
        B, L = input_ids.shape
        L2 = ((L + multiple - 1) // multiple) * multiple
        if L2 == L:
            return input_ids, attention_mask, completion_mask
        pad_len = L2 - L
        pad_ids = torch.full((B, pad_len), pad_id, dtype=input_ids.dtype, device=input_ids.device)
        zeros = torch.zeros_like(pad_ids)
        if side == "left":
            input_ids = torch.cat((pad_ids, input_ids), dim=1)
            attention_mask = torch.cat((zeros, attention_mask), dim=1)
            completion_mask = torch.cat((zeros, completion_mask), dim=1)
        else:
            input_ids = torch.cat((input_ids, pad_ids), dim=1)
            attention_mask = torch.cat((attention_mask, zeros), dim=1)
            completion_mask = torch.cat((completion_mask, zeros), dim=1)
        return input_ids, attention_mask, completion_mask

    def _collate_prompt_completion(self, examples):
        prompt_texts, completion_texts, images = [], [], []

        for ex in examples:
            p, c = ex["prompt"], ex["completion"]

            # Determine chat vs plain for each side
            is_p_msgs = isinstance(p, list) and (len(p) == 0 or isinstance(p[0], dict))
            is_c_msgs = isinstance(c, list) and (len(c) == 0 or isinstance(c[0], dict))

            if is_p_msgs:
                self._validate_and_normalize_first_message(p)
                if self.assistant_single_content:
                    self._collapse_assistant_content(p)
                p_txt = self._render_chat(p)
            else:
                p_txt = str(p)

            if is_c_msgs:
                self._validate_and_normalize_first_message(c)
                if self.assistant_single_content:
                    self._collapse_assistant_content(c)
                c_txt = self._render_chat(c)
            else:
                c_txt = str(c)

            # Images: prefer embedded; else first top-level image; else []
            imgs = self._extract_images_for_pc(ex, p if is_p_msgs else None, c if is_c_msgs else None)
            imgs = self._resize_images_inplace(imgs)

            prompt_texts.append(p_txt)
            completion_texts.append(c_txt)
            images.append(imgs)

        # Encode prompts (LEFT pad) with images
        proc_prompts = self.processor(
            images=images,
            text=prompt_texts,
            padding=True,
            padding_side="left",
            return_tensors="pt",
            add_special_tokens=False,
        )
        # Encode completions (RIGHT pad) text-only
        proc_completions = self.processor(
            text=completion_texts,
            padding=True,
            padding_side="right",
            return_tensors="pt",
            add_special_tokens=False,
        )

        p_ids, c_ids = proc_prompts["input_ids"], proc_completions["input_ids"]
        p_m, c_m = proc_prompts["attention_mask"], proc_completions["attention_mask"]
        input_ids = torch.cat((p_ids, c_ids), dim=1)
        attention_mask = torch.cat((p_m, c_m), dim=1)
        completion_mask = torch.cat((torch.zeros_like(p_m), c_m), dim=1)

        # Flush to tokenizer default padding side
        pad_id = self._pad_token_id_or_fail()
        flush_side = self._tokenizer_padding_side()
        attention_mask, input_ids, (completion_mask,) = self._flush_to_side(
            attention_mask, input_ids, flush_side, pad_id, (completion_mask,)
        )

        # Truncate with side awareness
        if self.max_seq_length is not None:
            input_ids, attention_mask, completion_mask = self._truncate_by_side(
                input_ids, attention_mask, completion_mask, flush_side, self.max_seq_length
            )

        # Optional pad-to-multiple-of (manual in PC)
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
        self._cast_pixel_values_dtype_inplace(out)
        return out
pass
