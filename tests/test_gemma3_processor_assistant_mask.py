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

"""Patched Gemma3Processor.__call__ must return text_replacement_offsets when asked; transformers >=
5.18 (issue 48793) needs them or assistant_only_loss trains on image tokens."""
import pytest

transformers = pytest.importorskip("transformers")
from packaging.version import Version
from PIL import Image

if Version(transformers.__version__.split("+")[0]) < Version("5.18.0.dev0"):
    pytest.skip("apply_chat_template realigns assistant masks from transformers 5.18", allow_module_level = True)

TINY = "trl-internal-testing/tiny-Gemma3ForConditionalGeneration"

# Gemma3's turn layout with generation markers around each assistant reply.
TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}"
    "{% set role = 'model' if message['role'] == 'assistant' else message['role'] %}"
    "{{ '<start_of_turn>' + role + '\n' }}"
    "{% if message['role'] == 'assistant' %}{% generation %}"
    "{% for item in message['content'] %}{{ item['text'] }}{% endfor %}{{ '<end_of_turn>\n' }}"
    "{% endgeneration %}{% else %}"
    "{% for item in message['content'] %}"
    "{% if item['type'] == 'image' %}{{ '<start_of_image>' }}{% else %}{{ item['text'] }}{% endif %}"
    "{% endfor %}{{ '<end_of_turn>\n' }}{% endif %}{% endfor %}"
)


def _processor():
    from transformers import AutoProcessor
    try:
        return AutoProcessor.from_pretrained(TINY)
    except OSError as e:  # Hub / network unavailable only; config or API breaks must fail
        pytest.skip(f"{TINY} not downloadable: {e}")


def _messages(n_images):
    img = Image.new("RGB", (32, 32), (255, 0, 0))
    first = [{"type": "image", "image": img} for _ in range(n_images)]
    return [
        {"role": "user", "content": first + [{"type": "text", "text": "What is in this picture?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "A red square."}]},
        {"role": "user", "content": [{"type": "text", "text": "What color is it?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "It is red."}]},
    ]


@pytest.mark.parametrize("n_images", [0, 1, 2])
@pytest.mark.parametrize("batch", [1, 2])
def test_assistant_mask_keeps_only_the_replies(n_images, batch):
    from unsloth_zoo.temporary_patches.gemma import patch_Gemma3Processor
    patch_Gemma3Processor()
    proc = _processor()
    out = proc.apply_chat_template(
        [_messages(n_images)] * batch,
        chat_template = TEMPLATE,
        tokenize = True,
        return_dict = True,
        return_tensors = "pt",
        return_assistant_tokens_mask = True,
        processor_kwargs = {"padding": True, "padding_side": "right", "add_special_tokens": False},
    )
    for row in range(batch):
        ids, mask = out["input_ids"][row], out["assistant_masks"][row].bool()
        assert proc.tokenizer.decode(ids[mask]) == "A red square.<end_of_turn>\nIt is red.<end_of_turn>\n"
        assert int((ids == proc.tokenizer.image_token_id).sum()) == n_images * proc.image_seq_length
    assert "text_replacement_offsets" not in out


@pytest.mark.parametrize("pan_and_scan", [False, True])
def test_replacement_offsets_point_at_the_expansion(pan_and_scan):
    from unsloth_zoo.temporary_patches.gemma import patch_Gemma3Processor
    patch_Gemma3Processor()
    proc = _processor()
    full = proc.full_image_sequence
    prompt = f"<start_of_turn>user\n{proc.boi_token} and {proc.boi_token}?<end_of_turn>\n"
    # A wide image is what Pan-and-Scan crops; the second stays uncropped.
    wide, small = Image.new("RGB", (1024, 256), (0, 0, 255)), Image.new("RGB", (32, 32), (0, 0, 255))
    kw = dict(add_special_tokens = False, do_pan_and_scan = pan_and_scan, pan_and_scan_min_crop_size = 16,
              pan_and_scan_max_num_crops = 2, pan_and_scan_min_ratio_to_activate = 1.2)
    out = proc(text = [prompt], images = [[wide, small]], return_text_replacement_offsets = True, **kw)
    offsets = out["text_replacement_offsets"][0]
    assert [o["span"] for o in offsets] == [
        (m, m + len(proc.boi_token)) for m in (prompt.index(proc.boi_token), prompt.rindex(proc.boi_token))
    ]
    expanded = prompt
    for o in reversed(offsets):
        expanded = expanded[: o["span"][0]] + o["replacement"] + expanded[o["span"][1]:]
    for o in offsets:
        assert expanded[o["new_span"][0]: o["new_span"][1]] == o["replacement"]
    cropped = [o["replacement"] != full for o in offsets]
    assert cropped == [pan_and_scan, False], "Pan-and-Scan did not engage as the case intends"
    if pan_and_scan:
        assert offsets[0]["replacement"].startswith(f"Here is the original image {full}")
    assert proc.tokenizer.decode(out["input_ids"][0]) == expanded
    # Not requested: not returned (it would reach the model as an unexpected kwarg).
    assert "text_replacement_offsets" not in proc(text = [prompt], images = [[wide, small]], **kw)
