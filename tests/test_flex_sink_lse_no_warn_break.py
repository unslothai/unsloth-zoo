"""flex_attention_with_sink must request the logsumexp without return_lse = True where torch has
AuxRequest (torch >= 2.9): the deprecation's warnings.warn is untraceable on torch 2.10, a graph
break in the gpt-oss attention under torch.compile. The output stays bit-equal."""

import pytest
import torch

S = pytest.importorskip("unsloth_zoo.flex_attention.attention_sink")
U = pytest.importorskip("unsloth_zoo.flex_attention.utils")
fa = pytest.importorskip("torch.nn.attention.flex_attention")


class _Attn(torch.nn.Module):
    def __init__(self, heads, device):
        super().__init__()
        self.sinks = torch.nn.Parameter(torch.linspace(-1, 1, heads, device = device))
        self.num_key_value_groups = 2
        self.scaling = 0.125
        self.sliding_window = None


def _inputs(device):
    g = torch.Generator(device = device).manual_seed(0)
    mk = lambda *s: torch.randn(*s, device = device, dtype = torch.bfloat16, generator = g)
    return mk(1, 4, 256, 64), mk(1, 2, 256, 64), mk(1, 2, 256, 64)


@pytest.mark.parametrize("decoding", [False, True])
def test_sink_attention_requests_lse_without_the_deprecated_flag(monkeypatch, decoding):
    if not hasattr(fa, "AuxRequest"):
        pytest.skip("torch < 2.9: return_lse is the only spelling")
    seen = []

    def fake_flex(q, k, v, **kwargs):
        seen.append(kwargs)
        lse = torch.zeros(q.shape[:-1], dtype = torch.float32)
        return torch.zeros_like(q), fa.AuxOutput(lse = lse)

    attn = _Attn(4, "cpu")
    q, k, v = _inputs("cpu")
    if decoding:
        monkeypatch.setattr(S, "flex_attention", fake_flex)
        attn._flex_attention_cache = lambda key: None
        out, lse = S.flex_attention_with_sink_decoding(attn, q, k, v)
    else:
        monkeypatch.setattr(S, "uncompiled_flex_attention", fake_flex)
        monkeypatch.setattr(S, "reused_compiled_create_block_mask", lambda *a, **kw: None)
        out = S.flex_attention_with_sink(attn, q, k, v, compile = False, has_static_cache = False)
    assert len(seen) == 1
    assert "return_lse" not in seen[0], seen[0]
    assert seen[0]["return_aux"].lse is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "flex_attention kernels need CUDA")
def test_sink_attention_output_unchanged():
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    attn = _Attn(4, "cuda").train()
    q, k, v = _inputs("cuda")
    out = S.flex_attention_with_sink(attn, q, k, v, compile = False, has_static_cache = False)
    bm = create_block_mask(lambda b, h, qi, ki: qi >= ki, 1, 4, 256, 256, device = "cuda")
    ref, lse = flex_attention(q, k, v, block_mask = bm, enable_gqa = True, scale = 0.125, return_lse = True)
    ref = ref * torch.sigmoid(lse - attn.sinks.unsqueeze(1)).unsqueeze(-1).to(ref.dtype)
    torch.testing.assert_close(out, ref.transpose(1, 2).contiguous(), atol = 0, rtol = 0)
