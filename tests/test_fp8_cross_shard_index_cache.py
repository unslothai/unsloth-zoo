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

"""FP8 cross-shard index header cache: byte-identical export, S header parses instead of S^2."""
from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass

import pytest
import torch

if not hasattr(torch, "float8_e4m3fn"):
    pytest.skip(reason = "this torch build has no float8_e4m3fn, so no FP8 shards can be written", allow_module_level = True)

from safetensors.torch import save_file


@dataclass
class _FakeLoraStats:
    lora_A: torch.Tensor = None
    lora_B: torch.Tensor = None
    alpha: float = 1.0
    module: object = None


def _quant(W):
    amax = W.abs().amax(dim = 1, keepdim = True).clamp_min(1e-12)
    scale = amax / 448.0
    return (W / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn), scale.to(torch.bfloat16)


def _write_checkpoint(d, n_shards = 5):
    """Per shard: a same-shard FP8 pair, a weight whose scale is in the next shard, a bf16 tensor."""
    torch.manual_seed(0)
    names = [f"model-{i + 1:05d}-of-{n_shards:05d}.safetensors" for i in range(n_shards)]
    shards = [dict() for _ in names]
    for i in range(n_shards):
        q, s = _quant(torch.randn(16, 32))
        shards[i][f"model.layers.{i}.self_attn.q_proj.weight"] = q
        shards[i][f"model.layers.{i}.self_attn.q_proj.weight_scale"] = s
        q, s = _quant(torch.randn(16, 32))
        shards[i][f"model.layers.{i}.mlp.down_proj.weight"] = q
        shards[(i + 1) % n_shards][f"model.layers.{i}.mlp.down_proj.weight_scale"] = s
        shards[i][f"model.layers.{i}.input_layernorm.weight"] = torch.randn(32, dtype = torch.bfloat16)
    for name, tensors in zip(names, shards):
        save_file(tensors, os.path.join(d, name), metadata = {"format": "pt"})
    return names


def _export(d, names, header_cache):
    import unsloth_zoo.saving_utils as sv

    prerewrite = sv._collect_fp8_weight_keys(d, names)
    for fn in names:
        kwargs = {} if header_cache is None else {"fp8_header_cache": header_cache}
        sv._merge_and_overwrite_lora(
            save_directory = d, filename = fn,
            lora_weights = defaultdict(_FakeLoraStats), output_dtype = torch.bfloat16,
            model_class_name = "Qwen2ForCausalLM", base_model_is_quantized = True,
            quant_type = "fp8", **kwargs,
        )
    sv._drop_resolved_fp8_scales_after_rewrite(d, names, prerewrite)
    out = {}
    for fn in names:
        with open(os.path.join(d, fn), "rb") as f:
            out[fn] = hashlib.sha256(f.read()).hexdigest()
    return out


def _count_header_parses(monkeypatch):
    import unsloth_zoo.saving_utils as sv

    calls = []
    real_build = sv._build_cross_shard_fp8_index

    class _Json:
        def __getattr__(self, name):
            return getattr(json, name)

        def loads(self, *args, **kwargs):
            calls.append(1)
            return json.loads(*args, **kwargs)

    def build(*args, **kwargs):
        monkeypatch.setattr(sv, "json", _Json())
        try:
            return real_build(*args, **kwargs)
        finally:
            monkeypatch.setattr(sv, "json", json)

    monkeypatch.setattr(sv, "_build_cross_shard_fp8_index", build)
    return calls


def test_cached_export_is_byte_identical_and_parses_each_header_once(tmp_path, monkeypatch):
    plain, cached = tmp_path / "plain", tmp_path / "cached"
    plain.mkdir()
    cached.mkdir()
    names = _write_checkpoint(str(plain))
    assert _write_checkpoint(str(cached)) == names

    calls = _count_header_parses(monkeypatch)
    expected = _export(str(plain), names, None)
    plain_parses = len(calls)
    del calls[:]
    got = _export(str(cached), names, {})
    cached_parses = len(calls)

    assert got == expected
    n = len(names)
    assert plain_parses == n * (n - 1)
    # Each shard is parsed before its rewrite and at most once after it.
    assert cached_parses <= 2 * n, (cached_parses, plain_parses)


def test_index_rereads_a_rewritten_sibling(tmp_path):
    from unsloth_zoo.saving_utils import _build_cross_shard_fp8_index

    d = str(tmp_path)
    save_file({"a.weight": torch.zeros(4, dtype = torch.float8_e4m3fn),
               "a.weight_scale": torch.ones(1)}, os.path.join(d, "s1.safetensors"))
    save_file({"b.weight": torch.zeros(4)}, os.path.join(d, "s2.safetensors"))
    cache = {}
    first = _build_cross_shard_fp8_index(d, "s2.safetensors", cache)
    assert first == _build_cross_shard_fp8_index(d, "s2.safetensors")
    assert "a.weight_scale" in first

    # Atomic replace (new inode), as the FP8 rewrite does.
    tmp = os.path.join(d, "tmp.part")
    save_file({"a.weight": torch.zeros(4, dtype = torch.bfloat16)}, tmp)
    os.replace(tmp, os.path.join(d, "s1.safetensors"))
    second = _build_cross_shard_fp8_index(d, "s2.safetensors", cache)
    assert second == _build_cross_shard_fp8_index(d, "s2.safetensors")
    assert "a.weight_scale" not in second and second["a.weight"][1] == "BF16"

    # In-place rewrite of the same inode with a different header.
    with open(os.path.join(d, "s1.safetensors"), "r+b") as f:
        save_file({"c.weight": torch.zeros(2, dtype = torch.float16)}, tmp)
        with open(tmp, "rb") as src:
            data = src.read()
        f.seek(0)
        f.write(data)
        f.truncate()
    os.remove(tmp)
    third = _build_cross_shard_fp8_index(d, "s2.safetensors", cache)
    assert third == _build_cross_shard_fp8_index(d, "s2.safetensors")
    assert set(third) == {"c.weight"}
