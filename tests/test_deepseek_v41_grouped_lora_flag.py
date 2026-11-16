# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""DeepSeek-V4.1's grouped o_a_proj LoRA carries _unsloth_grouped_lora, which tensor parallel needs to shard it."""

import pytest

pytest.importorskip("peft")
from unsloth_zoo.temporary_patches.deepseek_v41 import _grouped_lora_linear_cls  # noqa: E402


def test_grouped_lora_carries_the_tensor_parallel_layout_flag():
    assert _grouped_lora_linear_cls()._unsloth_grouped_lora is True
