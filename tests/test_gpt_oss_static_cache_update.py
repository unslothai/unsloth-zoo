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

# gpt-oss decode: static_cache_update replays transformers' one-token StaticLayer / StaticSlidingWindowLayer
# update with fewer ops. Same cache contents, returned tensors and counters as Cache.update, step by step.
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest
import torch

cache_utils = pytest.importorskip("transformers.cache_utils")
from unsloth_zoo import gpt_oss_decode_fused as F  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
verified = pytest.mark.skipif(not F._static_classes(), reason = "transformers' static cache update is not a verified one")


def _cache(max_len = 12, window = 4):
    return cache_utils.Cache(layers = [cache_utils.StaticLayer(max_len), cache_utils.StaticSlidingWindowLayer(max_len, window)])


def _kv(g, n, device, dtype, B = 2, H = 3, D = 8):
    return (torch.randn(B, H, n, D, generator = g).to(device, dtype), torch.randn(B, H, n, D, generator = g).to(device, dtype))


@verified
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_decode_updates_match_transformers(device, dtype):
    g = torch.Generator().manual_seed(0)
    ref, fast = _cache(), _cache()
    hits = 0
    with torch.no_grad():
        k, v = _kv(g, 3, device, dtype)
        for cache in (ref, fast):
            for i in range(2):
                cache.update(k, v, i, {"cache_position": torch.arange(3, device = device)})
        for step in range(8):  # the 4-slot window fills on the first step, then wraps twice
            k, v = _kv(g, 1, device, dtype)
            position = torch.tensor([3 + step], device = device)
            for i in range(2):
                want = ref.update(k, v, i, {"cache_position": position})
                got = F.static_cache_update(fast, k, v, i)
                if got is None:
                    got = fast.update(k, v, i, {"cache_position": position})
                else:
                    hits += 1
                    assert got[0] is fast.layers[i].keys and got[1] is fast.layers[i].values
                assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])
            for a, b in zip(ref.layers, fast.layers):
                assert torch.equal(a.keys, b.keys) and torch.equal(a.values, b.values)
                assert int(a.cumulative_length) == int(b.cumulative_length)
                assert getattr(a, "cumulative_length_int", None) == getattr(b, "cumulative_length_int", None)
                assert a.get_seq_length() == b.get_seq_length() or torch.equal(a.get_seq_length(), b.get_seq_length())
                assert a.get_mask_sizes(1) == b.get_mask_sizes(1)
    # The full layer every step; the sliding one once full (3 prefilled of 4 slots: from the second step).
    assert hits == 8 + 7


@verified
def test_uncovered_calls_return_none(monkeypatch):
    g = torch.Generator().manual_seed(1)
    cache = _cache()
    with torch.no_grad():
        k, v = _kv(g, 3, "cpu", torch.float32)
        for i in range(2):
            cache.update(k, v, i, {"cache_position": torch.arange(3)})
        k1, v1 = _kv(g, 1, "cpu", torch.float32)
        assert F.static_cache_update(cache, k, v, 0) is None  # more than one token
        assert F.static_cache_update(cache, k1, v1, 1) is None  # sliding layer not full yet
        assert F.static_cache_update(cache, k1, v1, 2) is None  # no such layer
        assert F.static_cache_update(cache, k1.double(), v1.double(), 0) is None  # dtype differs from the cache
        assert F.static_cache_update(cache, k1[:1], v1[:1], 0) is None  # batch differs
        cache.offloading = True
        assert F.static_cache_update(cache, k1, v1, 0) is None
        cache.offloading = False

        class Custom(cache_utils.Cache):
            def update(self, *args, **kwargs):
                return super().update(*args, **kwargs)
        custom = Custom(layers = cache.layers)
        assert F.static_cache_update(custom, k1, v1, 0) is None
        # An update overridden on the instance (cache or layer) is the caller's own code: left to it.
        cache.update = lambda *args, **kwargs: None
        assert F.static_cache_update(cache, k1, v1, 0) is None
        del cache.update
        cache.layers[0].update = lambda *args, **kwargs: None
        assert F.static_cache_update(cache, k1, v1, 0) is None
        del cache.layers[0].update
        monkeypatch.setattr(F, "_CACHE_ENABLED", False)
        assert F.static_cache_update(cache, k1, v1, 0) is None
        monkeypatch.setattr(F, "_CACHE_ENABLED", True)
        monkeypatch.setattr(F, "_STATIC_CLASSES", ())  # a transformers whose update code is not the verified one
        assert F.static_cache_update(cache, k1, v1, 0) is None
    assert F.static_cache_update(cache, k1, v1, 0) is None  # autograd on


def test_cache_update_digests_cover_5_20_and_5_21():
    # transformers 5.20 / 5.21 change Cache.update only inside its offloading branch, which the fast path declines.
    assert "f5c3ba02bbf4c5f4" in F._VERIFIED_UPDATE["Cache"]


def test_unverified_source_is_rejected(monkeypatch):
    monkeypatch.setattr(F, "_STATIC_CLASSES", None)
    monkeypatch.setitem(F._VERIFIED_UPDATE, "StaticLayer", ("0" * 16,))
    assert F._static_classes() == ()
    monkeypatch.setattr(F, "_STATIC_CLASSES", None)


# The patched gpt-oss model generating with a static cache (CUDA: the attention patch needs flex attention):
# fast update on vs off gives the same tokens and bit-identical logits; the fast path runs on every decode
# step of every layer once the window is full.
_GEN = textwrap.dedent('''
    import json, os, sys
    os.environ.setdefault("UNSLOTH_MODEL_NAME", "unsloth/gpt-oss-test")
    import torch
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
    import transformers.models.gpt_oss.modeling_gpt_oss as M
    dev = sys.argv[1]
    from unsloth_zoo.temporary_patches import gpt_oss as G
    from unsloth_zoo import gpt_oss_decode_fused as F
    import unsloth_zoo.flex_attention as FA
    for name in ("is_flex_attention_decoding", "flex_attention_with_sink_decoding"):
        if not hasattr(FA, name):
            setattr(FA, name, None)
    G.patch_GptOssAttention(); G.patch_GptOssModel()
    L = 4
    torch.manual_seed(0)
    c = GptOssConfig(hidden_size = 128, intermediate_size = 128, num_hidden_layers = L, num_attention_heads = 8,
                     num_key_value_heads = 2, head_dim = 32, num_local_experts = 4, num_experts_per_tok = 2,
                     vocab_size = 256, max_position_embeddings = 512, sliding_window = 6,
                     layer_types = ["sliding_attention", "full_attention"] * (L // 2))
    c._attn_implementation = "eager"
    model = M.GptOssForCausalLM(c)
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.normal_(0, 0.5 if "sinks" in n else 0.06)
            if "norm" in n:
                p.add_(1.0)
    model = model.to(dev).eval()
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(1, 256, (2, 9), generator = g).to(dev)
    with torch.no_grad():
        out = model.generate(ids, attention_mask = torch.ones_like(ids), max_new_tokens = 20, do_sample = False,
                             pad_token_id = 0, output_scores = True, return_dict_in_generate = True,
                             cache_implementation = "static", disable_compile = True)  # eager decode loop
    res = {"tokens": out.sequences.tolist(), "scores": torch.stack(out.scores, 1).float().tolist(),
           "calls": F.FUSED_DECODE_CALLS["static_cache_update"]}
    print("RESULT " + json.dumps(res))
''')


def _gen(tmp_path, flag):
    script = tmp_path / "gen.py"
    script.write_text(_GEN)
    env = dict(os.environ, UNSLOTH_IS_PRESENT = "1", UNSLOTH_GPT_OSS_FAST_CACHE_UPDATE = flag,
               PYTHONPATH = _ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    proc = subprocess.run([sys.executable, str(script), "cuda"], capture_output = True, text = True, env = env,
                          timeout = 900)
    lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert lines, proc.stdout[-3000:] + proc.stderr[-6000:]
    return json.loads(lines[-1][len("RESULT "):])


@verified
@pytest.mark.skipif(not torch.cuda.is_available(), reason = "the gpt-oss attention patch needs flex attention (CUDA)")
def test_static_cache_generation_is_bit_identical(tmp_path):
    on, off = _gen(tmp_path, "1"), _gen(tmp_path, "0")
    assert off["calls"] == 0
    # 19 decode steps x 2 full layers, plus the sliding layers once their 6-slot window is full (9-token prompt).
    assert on["calls"] >= 19 * 2 + 18 * 2, on["calls"]
    assert on["tokens"] == off["tokens"]
    assert on["scores"] == off["scores"]
