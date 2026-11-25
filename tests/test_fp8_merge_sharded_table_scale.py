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

"""16bit merge of a row-sharded fp8 table with one shared per-tensor scale (Qwen3.8-Flash-Next-FP8's n-gram
table: `ngram_embedding.shard_<i>.weight` F8_E4M3 over many files, one `ngram_embedding.weight_scale`)."""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass

import pytest
import torch

if not hasattr(torch, "float8_e4m3fn"):
    pytest.skip(reason = "this torch build has no float8_e4m3fn, so no FP8 shards can be written", allow_module_level = True)

from safetensors import safe_open
from safetensors.torch import save_file

TABLE = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
NAMES = [f"model-{i + 1:05d}-of-00003.safetensors" for i in range(3)]


@dataclass
class _FakeLoraStats:
    lora_A: torch.Tensor = None
    lora_B: torch.Tensor = None
    alpha: float = 1.0
    module: object = None


def _write_checkpoint(d, scale_numel = 1):
    """Table shards 0-1 in file 1, shards 2-3 and the shared scale in file 2, an fp8 linear pair and an
    unrelated bf16 `*.weight_scale` buffer (its base weight is bf16) in file 3."""
    torch.manual_seed(0)
    rows = [torch.randn(64, 8) for _ in range(4)]
    amax = max(float(r.abs().max()) for r in rows)
    scale = torch.full((scale_numel,), amax / 448.0, dtype = torch.bfloat16)
    q = [(r / scale.float()[0]).clamp(-448, 448).to(torch.float8_e4m3fn) for r in rows]
    files = [dict(), dict(), dict()]
    files[0][f"{TABLE}.shard_0.weight"] = q[0]
    files[0][f"{TABLE}.shard_1.weight"] = q[1]
    files[1][f"{TABLE}.shard_2.weight"] = q[2]
    files[1][f"{TABLE}.shard_3.weight"] = q[3]
    files[1][f"{TABLE}.weight_scale"] = scale
    w = torch.randn(16, 32)
    s = (w.abs().amax(dim = 1, keepdim = True) / 448.0).to(torch.bfloat16)
    files[2]["model.language_model.layers.0.self_attn.q_proj.weight"] = (w / s.float()).clamp(-448, 448).to(torch.float8_e4m3fn)
    files[2]["model.language_model.layers.0.self_attn.q_proj.weight_scale"] = s
    files[2]["model.language_model.layers.0.router.weight"] = torch.randn(4, 32, dtype = torch.bfloat16)
    files[2]["model.language_model.layers.0.router.weight_scale"] = torch.ones(4, dtype = torch.bfloat16)
    for name, tensors in zip(NAMES, files):
        save_file(tensors, os.path.join(d, name), metadata = {"format": "pt"})
    expected = {f"{TABLE}.shard_{i}.weight": (q[i].float() * scale.float()[0]).to(torch.bfloat16) for i in range(4)}
    return expected


def _export(d, order):
    import unsloth_zoo.saving_utils as sv

    prerewrite = sv._collect_fp8_weight_keys(d, NAMES)
    for fn in order:
        sv._merge_and_overwrite_lora(
            save_directory = d, filename = fn,
            lora_weights = defaultdict(_FakeLoraStats), output_dtype = torch.bfloat16,
            model_class_name = "Qwen4ExpForConditionalGeneration", base_model_is_quantized = True,
            quant_type = "fp8",
        )
    sv._drop_resolved_fp8_scales_after_rewrite(d, NAMES, prerewrite)
    out = {}
    for fn in NAMES:
        with safe_open(os.path.join(d, fn), framework = "pt", device = "cpu") as f:
            for k in f.keys():
                out[k] = f.get_tensor(k)
    return out


@pytest.mark.parametrize("order", [NAMES, NAMES[::-1]], ids = ["scale_file_after_shards", "scale_file_first"])
def test_sharded_table_decodes_with_shared_scale(tmp_path, order):
    expected = _write_checkpoint(str(tmp_path))
    out = _export(str(tmp_path), order)
    for key, want in expected.items():
        assert out[key].dtype == torch.bfloat16, key
        assert torch.equal(out[key], want), key
    assert f"{TABLE}.weight_scale" not in out
    assert out["model.language_model.layers.0.self_attn.q_proj.weight"].dtype == torch.bfloat16
    assert "model.language_model.layers.0.self_attn.q_proj.weight_scale" not in out
    assert torch.equal(out["model.language_model.layers.0.router.weight_scale"], torch.ones(4, dtype = torch.bfloat16))


def test_non_scalar_shared_scale_still_refused(tmp_path):
    _write_checkpoint(str(tmp_path), scale_numel = 2)
    with pytest.raises(RuntimeError, match = "has no companion weight_scale"):
        _export(str(tmp_path), NAMES)
