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

"""Audio Flamingo 3 / Music Flamingo audio features keep their 30 s length through the collator.

Their encoders add a fixed 1500 positions, so the top-level padding = True that ProcessorMixin
also hands to the feature extractor (shortest-to-longest clip) made every training step raise.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from unsloth_zoo.vision_utils import UnslothVisionDataCollator

REPO = "nvidia/audio-flamingo-3-hf"


def _processor():
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "AudioFlamingo3Processor"):
        pytest.skip("transformers has no Audio Flamingo 3")
    try:
        return transformers.AutoProcessor.from_pretrained(REPO)
    except (OSError, ValueError) as error:
        pytest.skip(f"{REPO} processor not reachable: {error}")


def _model(model_type):
    emb = SimpleNamespace(weight = torch.zeros(1, dtype = torch.float32))
    return SimpleNamespace(
        config = SimpleNamespace(torch_dtype = "float32", model_type = model_type),
        get_input_embeddings = lambda: emb,
    )


def _rows():
    rng = np.random.default_rng(0)
    rows = []
    for seconds in (1.0, 2.0):
        clip = rng.standard_normal(int(16000 * seconds)).astype(np.float32) * 0.1
        rows.append({"messages": [
            {"role": "user", "content": [{"type": "audio", "audio": clip},
                                         {"type": "text", "text": "Transcribe the audio."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "hello world"}]},
        ]})
    return rows


@pytest.mark.parametrize("model_type", ["audioflamingo3", "musicflamingo"])
def test_flamingo_features_stay_full_length(model_type):
    processor = _processor()
    batch = UnslothVisionDataCollator(_model(model_type), processor)(_rows())
    assert tuple(batch["input_features"].shape) == (2, 128, 3000)
    assert tuple(batch["input_features_mask"].shape) == (2, 3000)
    # Text is still padded to the longest row, with labels masked on the padding.
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    assert (batch["attention_mask"] == 0).any()
    assert ((batch["labels"] != -100) & (batch["attention_mask"] == 0)).sum() == 0
    assert (batch["labels"] != -100).sum() > 0


def test_other_models_keep_top_level_padding():
    processor = _processor()
    calls = []
    original = type(processor).__call__

    def recording(self, *args, **kwargs):
        calls.append(dict(kwargs))
        return original(self, *args, **kwargs)

    collator = UnslothVisionDataCollator(_model("qwen2_audio"), processor)
    type(processor).__call__ = recording
    try:
        batch = collator(_rows())
    finally:
        type(processor).__call__ = original
    assert calls and all(c.get("padding") is True and "text_kwargs" not in c for c in calls)
    assert batch["input_features"].shape[-1] < 3000
