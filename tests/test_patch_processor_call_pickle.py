# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A patch_processor_call processor pickles (spawn DataLoader workers, TRL's AsyncGRPO rollout
worker): the patched subclass shares the stock class's name, so pickle could not look it up."""

import copy
import os
import pickle
import subprocess
import sys

import pytest

from unsloth_zoo.tokenizer_utils import patch_processor_call


class _Processor:
    image_processor = object()

    def apply_chat_template(
        self,
        conversation,
        tokenize = False,
        add_generation_prompt = False,
    ):
        return "|".join(m["content"] for m in conversation)

    def __call__(
        self,
        text = None,
        images = None,
        **kwargs,
    ):
        return {"text": text}


CONVERSATION = [{"role": "user", "content": "hi"}]


def test_pickle_round_trip_gives_the_stock_class():
    processor = patch_processor_call(_Processor())
    processor.extra = 3
    loaded = pickle.loads(pickle.dumps(processor))
    assert type(loaded) is _Processor
    assert loaded.extra == 3
    assert not hasattr(loaded, "_unsloth_patched_call")
    # The live processor keeps its patch.
    assert processor(text = CONVERSATION)["text"] == "hi"
    assert type(processor) is not _Processor


def test_copies_keep_the_patch():
    processor = patch_processor_call(_Processor())
    processor.items = [1]
    for clone in (copy.copy(processor), copy.deepcopy(processor)):
        assert type(clone) is type(processor)
        assert clone(text = CONVERSATION)["text"] == "hi"
    deep = copy.deepcopy(processor)
    assert deep.items == [1] and deep.items is not processor.items


def test_unpickled_processor_can_be_patched_again():
    loaded = pickle.loads(pickle.dumps(patch_processor_call(_Processor())))
    assert patch_processor_call(loaded)(text = CONVERSATION)["text"] == "hi"


def test_real_processor_loads_without_unsloth_zoo():
    transformers = pytest.importorskip("transformers")
    try:
        processor = transformers.AutoProcessor.from_pretrained(
            "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration"
        )
    except Exception as error:
        pytest.skip(f"tiny Qwen2.5-VL processor unavailable: {error}")
    payload = pickle.dumps(patch_processor_call(processor))
    code = (
        "import pickle, sys\n"
        "p = pickle.loads(sys.stdin.buffer.read())\n"
        "assert 'unsloth_zoo' not in sys.modules, 'unpickling imported unsloth_zoo'\n"
        "print(type(p).__name__, p(text = 'hello world')['input_ids'][0][:3])\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONSTARTUP"}
    result = subprocess.run(
        [sys.executable, "-c", code],
        input = payload,
        capture_output = True,
        env = env,
        timeout = 300,
    )
    assert result.returncode == 0, result.stderr.decode()[-2000:]
    assert result.stdout.decode().startswith(type(processor).__name__)
