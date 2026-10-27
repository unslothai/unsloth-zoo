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

# GPT-OSS eval / generate runs flex attention with sinks on a BlockMask built from absolute positions.
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "flex inference path is CUDA only")


def _oracle(q, k, v, allowed, sinks, scale):
    # fp32 dense reference: softmax over [visible keys, sink], rows with no visible key give 0.
    groups = q.shape[1] // k.shape[1]
    k = k.double().repeat_interleave(groups, 1)
    v = v.double().repeat_interleave(groups, 1)
    s = torch.einsum("bhqd,bhkd->bhqk", q.double(), k) * scale
    s = s.masked_fill(~allowed[:, None], float("-inf"))
    sink = sinks.double()[None, :, None, None].expand(*s.shape[:3], 1)
    p = torch.softmax(torch.cat([s, sink], -1), -1)[..., :-1]
    return torch.einsum("bhqk,bhkd->bhqd", p, v).transpose(1, 2)


class _Attn(torch.nn.Module):
    def __init__(self, heads, kv_heads, head_dim):
        super().__init__()
        self.num_key_value_groups = heads // kv_heads
        self.scaling = head_dim ** -0.5
        # Distinct sink per query head, so a sink broadcast over KV heads shows up.
        self.sinks = torch.nn.Parameter(torch.linspace(-2.0, 3.0, heads))


@pytest.mark.parametrize("window", [None, 5])
@pytest.mark.parametrize("q_len,q_offset,kv_offset,kv_len", [
    (13, 0, 0, 13),     # prefill
    (1, 12, 0, 16),     # decode into a static cache: slots 13..15 not written yet
    (1, 20, 16, 5),     # decode on a rolled sliding layer
    (4, 13, 0, 17),     # multi-token continuation
    (1, 0, 0, 1),       # fresh one-token prompt
])
def test_inference_matches_fp32_sink_oracle(window, q_len, q_offset, kv_offset, kv_len):
    from transformers.masking_utils import causal_mask_function, sliding_window_causal_mask_function
    from unsloth_zoo.flex_attention import flex_attention_with_sink_inference
    from unsloth_zoo.temporary_patches.gpt_oss import _gpt_oss_flex_mask

    torch.manual_seed(0)
    B, H, KVH, D = 3, 8, 2, 32
    mask_function = causal_mask_function if window is None else sliding_window_causal_mask_function(window)
    total = max(q_offset + q_len, kv_offset + kv_len)
    padding = torch.ones(B, total, dtype = torch.bool, device = "cuda")
    for b, n in enumerate([0, 4, 9]):
        padding[b, :n] = False
    q = torch.randn(B, H, q_len, D, device = "cuda")
    k = torch.randn(B, KVH, kv_len + 3, D, device = "cuda")
    v = torch.randn(B, KVH, kv_len + 3, D, device = "cuda")
    # Unwritten static slots and slots past the mask's KV length are loud but finite.
    unwritten = max(0, q_offset + q_len - kv_offset)
    k[:, :, unwritten:] = 1e4
    v[:, :, unwritten:] = 1e4
    k_before, v_before, q_before = k.clone(), v.clone(), q.clone()

    block_mask = _gpt_oss_flex_mask(
        batch_size = B, q_length = q_len, kv_length = kv_len, q_offset = torch.tensor(q_offset),
        kv_offset = kv_offset, mask_function = mask_function, attention_mask = padding.long(), device = "cuda",
    )
    attn = _Attn(H, KVH, D).cuda()
    out = flex_attention_with_sink_inference(attn, q, k, v, block_mask)

    qp = torch.arange(q_len, device = "cuda")[:, None] + q_offset
    kp = torch.arange(kv_len, device = "cuda")[None, :] + kv_offset
    allowed = (kp <= qp)
    if window is not None:
        allowed = allowed & (kp > qp - window)
    allowed = allowed[None] & padding[:, kp[0]][:, None, :]
    ref = _oracle(q, k[:, :, :kv_len], v[:, :, :kv_len], allowed, attn.sinks.detach(), attn.scaling)

    assert out.shape == (B, q_len, H, D)
    assert torch.isfinite(out).all()
    # q_len 1 runs flex_decoding (exact fp32). Longer queries run the forced flex kernel, which rounds fp32 like
    # TF32 once the mask_mod reads a captured tensor (stock torch.compile too): ~1e-3 on unit-scale values.
    atol = 2e-5 if q_len == 1 else 4e-3
    torch.testing.assert_close(out.double(), ref, atol = atol, rtol = 1e-4)
    empty = ~allowed.any(-1)
    if empty.any():
        assert (out.transpose(1, 2)[:, :, :][empty[:, None].expand(-1, H, -1)] == 0).all()
    assert torch.equal(q, q_before) and torch.equal(k, k_before) and torch.equal(v, v_before)


_SCRIPT = textwrap.dedent('''
    import os, sys, json
    os.environ.setdefault("UNSLOTH_MODEL_NAME", "unsloth/gpt-oss-test")
    import torch
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
    import transformers.models.gpt_oss.modeling_gpt_oss as M
    from transformers import DynamicCache, StaticCache

    mode, dtype, new_tokens = sys.argv[1], getattr(torch, sys.argv[2]), int(sys.argv[3])
    L, W = 4, 6
    torch.manual_seed(0)
    c = GptOssConfig(hidden_size = 128, intermediate_size = 128, num_hidden_layers = L, num_attention_heads = 8,
                     num_key_value_heads = 2, head_dim = 32, num_local_experts = 4, num_experts_per_tok = 2,
                     vocab_size = 256, max_position_embeddings = 512, sliding_window = W,
                     layer_types = ["sliding_attention", "full_attention"] * (L // 2))
    c._attn_implementation = "eager"
    model = M.GptOssForCausalLM(c)
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.normal_(0, 0.5 if "sinks" in n else 0.06)
    model = model.to("cuda", dtype).eval()

    g = torch.Generator().manual_seed(1)
    B, T = 3, 16
    ids = torch.randint(1, 256, (B, T), generator = g).cuda()
    pad = torch.ones(B, T, dtype = torch.long, device = "cuda")
    for b, n in enumerate([0, 4, 9]):
        pad[b, :n] = 0
        ids[b, :n] = 0

    CALLS = {"q1": 0, "qN": 0}
    if mode != "stock":
        import unsloth_zoo.flex_attention as FA
        _real = FA.flex_attention_with_sink_inference
        def _count(self_attn, q, *a, **k):
            CALLS["q1" if q.shape[-2] == 1 else "qN"] += 1
            return _real(self_attn, q, *a, **k)
        FA.flex_attention_with_sink_inference = _count
        from unsloth_zoo.temporary_patches import gpt_oss as G
        G.patch_GptOssAttention(); G.patch_GptOssModel()
        assert G._GPT_OSS_FLEX_SINK_ATTENTION_INSTALLED
        os.environ["UNSLOTH_GPT_OSS_EAGER_INFERENCE"] = "1" if mode == "eager" else "0"

    res = {}
    def gen(cache, input_ids, attention_mask, n):
        return model.generate(input_ids, attention_mask = attention_mask, max_new_tokens = n, do_sample = False,
                              pad_token_id = 0, cache_implementation = None if cache == "dynamic" else "static",
                              output_scores = True, return_dict_in_generate = True)
    with torch.no_grad():
        res["prefill"] = (model(input_ids = ids, attention_mask = pad, use_cache = False).logits.float()
                          * pad[:, :, None]).tolist()
        for cache in ("dynamic", "static"):
            out = gen(cache, ids, pad, new_tokens)
            res["tokens_" + cache] = out.sequences.tolist()
            res["scores_" + cache] = torch.stack(out.scores, 1).float().tolist()
            # Fresh one-token prompt right after a long generation: not a decode, no state carried over.
            one = gen(cache, ids[:1, -1:], None, 12)
            res["one_token_" + cache] = one.sequences.tolist()
        # Chunked prefill / multi-token continuation (rolls the sliding layers) vs one shot.
        full = model(input_ids = ids, attention_mask = pad).logits.float()
        for cache in ("dynamic", "static"):
            pkv = DynamicCache(config = c) if cache == "dynamic" else StaticCache(config = c, max_cache_len = 32)
            parts = []
            for s, e in ((0, 7), (7, 8), (8, 13), (13, 16)):
                parts.append(model(input_ids = ids[:, s:e], attention_mask = pad[:, :e], past_key_values = pkv,
                                   use_cache = True).logits.float())
            res["chunked_maxdiff_" + cache] = float(((torch.cat(parts, 1) - full) * pad[:, :, None]).abs().max())
        res["calls"] = dict(CALLS)

        if mode == "flex":
            # Recompiles: a second same-shape generation and longer decodes inside the same 128 block add no graphs.
            counters = torch._dynamo.utils.counters
            before = counters["stats"]["unique_graphs"]
            gen("dynamic", ids, pad, new_tokens)
            gen("static", ids, pad, new_tokens)
            res["graphs_same_shape"] = counters["stats"]["unique_graphs"] - before
            gen("dynamic", ids, pad, 100)
            mid = counters["stats"]["unique_graphs"]
            gen("dynamic", ids, pad, 110)
            res["graphs_longer_decode"] = counters["stats"]["unique_graphs"] - mid

    if mode == "flex":
        # train -> generate -> train: training keeps flex_attention_with_sink, nothing leaks from eval.
        labels = ids.masked_fill(pad == 0, -100)
        def train_loss():
            model.train()
            loss = model(input_ids = ids, attention_mask = pad, labels = labels).loss
            loss.backward()
            model.zero_grad(set_to_none = True)
            return float(loss)
        l1 = train_loss()
        model.eval()
        with torch.no_grad():
            gen("dynamic", ids, pad, 8)
        l2 = train_loss()
        model.train()
        with torch.no_grad():
            short = model(input_ids = ids[:, :3], attention_mask = pad[:, :3]).logits
        res["train_losses"] = [l1, l2]
        res["short_nograd_finite"] = bool(torch.isfinite(short).all())
        model.eval()
    print("RESULT " + json.dumps(res))
''')


def _run(tmp_path, mode, dtype = "float32", new_tokens = 64):
    script = tmp_path / "gpt_oss_flex_inference_case.py"
    script.write_text(_SCRIPT)
    env = dict(os.environ)
    env.pop("UNSLOTH_GPT_OSS_EAGER_INFERENCE", None)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join(x for x in (root, env.get("PYTHONPATH")) if x)
    env.setdefault("UNSLOTH_COMPILE_LOCATION", str(tmp_path / "unsloth_compiled_cache"))
    proc = subprocess.run([sys.executable, str(script), mode, dtype, str(new_tokens)], capture_output = True,
                          text = True, env = env, timeout = 1500)
    line = [x for x in proc.stdout.splitlines() if x.startswith("RESULT ")]
    assert proc.returncode == 0 and line, proc.stdout[-2000:] + proc.stderr[-3000:]
    return json.loads(line[-1][len("RESULT "):])


@pytest.fixture(scope = "module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("gpt_oss_flex")
    return {mode: _run(tmp, mode) for mode in ("stock", "eager", "flex")}


def test_generate_engages_flex_and_kill_switch_restores_eager(runs):
    assert runs["flex"]["calls"]["q1"] > 0 and runs["flex"]["calls"]["qN"] > 0
    assert runs["eager"]["calls"] == {"q1": 0, "qN": 0}


@pytest.mark.parametrize("cache", ["dynamic", "static"])
def test_greedy_tokens_match_eager_and_stock_hf(runs, cache):
    flex, eager, stock = runs["flex"], runs["eager"], runs["stock"]
    assert len(flex["tokens_" + cache][0]) >= 16 + 64
    assert flex["tokens_" + cache] == eager["tokens_" + cache] == stock["tokens_" + cache]
    assert flex["one_token_" + cache] == eager["one_token_" + cache] == stock["one_token_" + cache]
    diff = (torch.tensor(flex["scores_" + cache]) - torch.tensor(stock["scores_" + cache])).abs().max()
    assert diff < 1e-3, diff


def test_prefill_and_chunked_prefill_match(runs):
    flex, stock = runs["flex"], runs["stock"]
    assert (torch.tensor(flex["prefill"]) - torch.tensor(stock["prefill"])).abs().max() < 1e-3
    for cache in ("dynamic", "static"):
        assert flex["chunked_maxdiff_" + cache] < 1e-3, (cache, flex["chunked_maxdiff_" + cache])
        assert abs(flex["chunked_maxdiff_" + cache] - stock["chunked_maxdiff_" + cache]) < 1e-3


def test_decode_does_not_recompile_after_warmup(runs):
    assert runs["flex"]["graphs_same_shape"] == 0
    assert runs["flex"]["graphs_longer_decode"] == 0


def test_train_generate_train(runs):
    l1, l2 = runs["flex"]["train_losses"]
    assert l1 == l2 and l1 == l1
    assert runs["flex"]["short_nograd_finite"]
