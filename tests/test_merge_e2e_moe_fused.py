# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""End-to-end LoRA merge correctness for fused-3D MoE models.

These pack all experts into single 3D params (experts.gate_up_proj/down_proj) with
grouped LoRA via PEFT target_parameters. The merge slices the adapter per expert and
applies the delta in the orientation that fits each expert (auto-detected from
distinct dims, so the #779 narrow-expert transpose bug cannot pass by luck).

gpt_oss runs on transformers 4.57.6 and 5.x. The 5.x-only worry arches (qwen3_5_moe
= Qwen3.5-35B-A3B, gemma4 = gemma-4-26B-A4B, lfm2_moe = LFM2.5-8B-A1B) skip cleanly
when a tiny config cannot exercise the fused path; their full-scale merge is covered
by the real-model sweep + #779.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch

import _merge_e2e_helpers as H

GATED = {"qwen3_5_moe", "gemma4", "lfm2_moe"}  # transformers 5.x only


def _gated_mark(family):
    return pytest.mark.skipif(
        not H.family_available(family),
        reason=f"{family} needs a transformers version that exposes it (5.x)")


FUSED = [
    "gpt_oss",
    pytest.param("qwen3_5_moe", marks=_gated_mark("qwen3_5_moe")),
    pytest.param("gemma4", marks=_gated_mark("gemma4")),
    pytest.param("lfm2_moe", marks=_gated_mark("lfm2_moe")),
]

# tiny-config / orchestration errors tolerated (skip) only for gated 5.x families;
# gpt_oss must never hit these.
_TOLERATED = ("validate_layer_type", "does not match # of saved modules",
              "could not be instantiated", "is not supported",
              "not found among base keys")


def _skip_or_raise(family, exc):
    msg = f"{type(exc).__name__}: {exc}"
    if family in GATED and any(s in msg for s in _TOLERATED):
        pytest.skip(f"{family}: tiny config cannot exercise the fused path "
                    f"({msg[:120]}); covered by the real-model sweep")
    raise exc


@pytest.mark.parametrize("family", FUSED)
def test_fused_moe_full(family, tmp_path):
    """Grouped expert LoRA + attention: every fused expert tensor matches the
    per-expert reference; merge does not fall back."""
    if not H.family_available(family):
        pytest.skip(f"{family} unavailable")
    H.set_offline_cpu_env()
    try:
        spec = H.make_spec(family)
        base_dir = os.path.join(str(tmp_path), "base")
        model = H.build_and_save_base(spec, base_dir)
        base_tensors = H.read_safetensors_dir(base_dir)
        pm = H.attach_lora(model, spec, "full")
    except Exception as e:
        _skip_or_raise(family, e)
    adapted = H.extract_adapted(pm)
    if not any(a.fused for a in adapted.values()):
        if family in GATED:
            pytest.skip(f"{family}: tiny config did not materialize fused experts "
                        f"(covered by the real-model sweep)")
        pytest.fail(f"{family}: expected fused expert adapters, found none")
    out_dir = os.path.join(str(tmp_path), "merged")
    try:
        H.run_merge(pm, base_dir, out_dir, save_dtype=torch.float32)
        H.assert_merge_correct(family=family, base_tensors=base_tensors,
                               out_dir=out_dir, save_dtype=torch.float32, adapted=adapted)
    except Exception as e:
        _skip_or_raise(family, e)
    assert sum(1 for a in adapted.values() if a.fused) >= 2


@pytest.mark.parametrize("family", FUSED)
def test_fused_moe_attention_only_keeps_experts_byte_identical(family, tmp_path):
    """No LoRA on the fused experts -> the fused expert tensors stay byte-identical
    to base (the merge must not rewrite/cast/transpose them)."""
    if not H.family_available(family):
        pytest.skip(f"{family} unavailable")
    H.set_offline_cpu_env()
    try:
        n_adapted, n_pass = H.run_case(family, "attn_only", str(tmp_path))
    except Exception as e:
        _skip_or_raise(family, e)
    assert n_adapted >= 1 and n_pass >= 1
