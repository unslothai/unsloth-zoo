# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The CPU loader and plain PEFT wrap DeepSeek-V4.1's o_a_proj with this grouped LoRA; unsloth's tensor parallel
shards a grouped linear's LoRA only when the layer carries _unsloth_grouped_lora (same layout as unsloth's own
grouped LoRA). Without it every tensor parallel step raised. CPU."""

import pytest

pytest.importorskip("peft")
from unsloth_zoo.temporary_patches.deepseek_v41 import _grouped_lora_linear_cls  # noqa: E402


def test_grouped_lora_carries_the_tensor_parallel_layout_flag():
    assert _grouped_lora_linear_cls()._unsloth_grouped_lora is True
