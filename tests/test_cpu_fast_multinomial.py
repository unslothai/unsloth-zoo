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

"""CPU torch.multinomial with one draw: same samples and RNG state as torch, no serial exponential_ fill."""

import pytest
import torch

from unsloth_zoo.cpu import sampling

ORIGINAL = getattr(torch.multinomial, "_unsloth_original", torch.multinomial)
FAST = sampling._wrap(ORIGINAL)
needs_fast = pytest.mark.skipif(not sampling.fast_multinomial_ok(),
                                reason = "this torch build's exponential_ is not the float64-uniform form")


def _probs(shape, trial):
    torch.manual_seed(100 + trial)
    p = torch.softmax(torch.randn(shape) * (1 + 3 * trial), -1)
    if trial == 1:
        p[..., ::3] = 0
    if trial == 2:
        p = p * 7.5  # unnormalised weights
    if trial == 3:
        p = torch.zeros(shape)
        p[..., 5] = 1.0  # one-hot
    return p.contiguous()


def _engaged(fn):
    before = sampling.ENGAGED[0]
    out = fn()
    return out, sampling.ENGAGED[0] - before


@needs_fast
@pytest.mark.parametrize("threads", [1, 2])
@pytest.mark.parametrize("shape", [(151936,), (2, 151936), (8, 32000), (3, 262144), (1, 4096)])
def test_same_samples_and_rng_state_as_torch(shape, threads):
    old = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        for trial in range(4):
            p = _probs(shape, trial)
            for replacement in (False, True):
                torch.manual_seed(trial)
                a = ORIGINAL(p, 1, replacement)
                state_a = torch.get_rng_state()
                torch.manual_seed(trial)
                b, n = _engaged(lambda: FAST(p, 1, replacement))
                assert n == 1
                assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
                assert torch.equal(state_a, torch.get_rng_state())
                g1, g2 = torch.Generator().manual_seed(9), torch.Generator().manual_seed(9)
                assert torch.equal(ORIGINAL(p, 1, generator = g1), FAST(p, 1, generator = g2))
                assert torch.equal(g1.get_state(), g2.get_state())
    finally:
        torch.set_num_threads(old)


@needs_fast
def test_many_draws_match():
    p = torch.softmax(torch.randn(2, 151936) * 2, -1)
    torch.manual_seed(0)
    a = torch.cat([ORIGINAL(p, 1) for _ in range(64)], 1)
    torch.manual_seed(0)
    b = torch.cat([FAST(p, 1) for _ in range(64)], 1)
    assert torch.equal(a, b)


@pytest.mark.parametrize("bad", ["nan", "negative", "zero_row", "inf"])
def test_invalid_probabilities_raise_torchs_error(bad):
    p = torch.full((2, 8192), 1.0)
    if bad == "nan":
        p[0, 3] = float("nan")
    elif bad == "negative":
        p[1, 7] = -1.0
    elif bad == "zero_row":
        p[1] = 0
    else:
        p[0, 0] = float("inf")
    with pytest.raises(RuntimeError) as stock:
        ORIGINAL(p, 1)
    torch.manual_seed(0)
    with pytest.raises(RuntimeError) as fast:
        FAST(p, 1)
    assert str(fast.value) == str(stock.value)


@pytest.mark.parametrize("case", ["float64", "bfloat16", "two_samples", "small", "strided", "out", "grad"])
def test_other_calls_keep_torchs_path(case):
    p = torch.softmax(torch.randn(4, 8192), -1)
    args, kwargs = (p, 1), {}
    if case == "float64":
        args = (p.double(), 1)
    elif case == "bfloat16":
        args = (p.bfloat16(), 1)
    elif case == "two_samples":
        args = (p, 2)
    elif case == "small":
        args = (p[:, :64].contiguous(), 1)
    elif case == "strided":
        args = (p.t().contiguous().t(), 1)
    elif case == "out":
        kwargs = {"out": torch.empty(4, 1, dtype = torch.long)}
    elif case == "grad":
        args = (p.clone().requires_grad_(), 1)
    torch.manual_seed(1)
    a = ORIGINAL(*args, **({"out": torch.empty(4, 1, dtype = torch.long)} if case == "out" else {}))
    torch.manual_seed(1)
    b, n = _engaged(lambda: FAST(*args, **kwargs))
    assert n == 0 and torch.equal(a, b)


@needs_fast
def test_draws_stay_on_cpu_under_another_default_device():
    p = torch.softmax(torch.randn(2, 8192), -1)
    torch.manual_seed(3)
    a = ORIGINAL(p, 1)
    with torch.device("meta"):  # any non-CPU default device: the draws must still be CPU tensors
        torch.manual_seed(3)
        b, n = _engaged(lambda: FAST(p, 1))
    assert n == 1 and b.device.type == "cpu" and torch.equal(a, b)


@needs_fast
def test_kill_switch(monkeypatch):
    monkeypatch.setenv("UNSLOTH_CPU_FAST_MULTINOMIAL", "0")
    p = torch.softmax(torch.randn(2, 8192), -1)
    _, n = _engaged(lambda: FAST(p, 1))
    assert n == 0


def test_build_check_matches_exponential():
    a, b = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
    q = torch.empty(2, 151936).exponential_(generator = a)
    same = torch.equal(q, sampling._exponential((2, 151936), torch.float32, b))
    assert sampling.fast_multinomial_ok() == (same and torch.equal(a.get_state(), b.get_state()))


@needs_fast
def test_cpu_model_attach_installs_once_and_generate_samples_match(monkeypatch):
    transformers = pytest.importorskip("transformers")
    from unsloth_zoo.cpu.loader import FastCPUModel

    torch.manual_seed(0)
    config = transformers.LlamaConfig(vocab_size = 8192, hidden_size = 32, intermediate_size = 64, num_hidden_layers = 1,
                                      num_attention_heads = 2, num_key_value_heads = 1, max_position_embeddings = 128)
    model = transformers.LlamaForCausalLM(config).eval()
    model.generation_config.pad_token_id = 0
    model.generation_config.eos_token_id = None
    saved = torch.multinomial
    try:
        FastCPUModel._attach(model)
        installed = torch.multinomial
        assert getattr(installed, "_unsloth_cpu_fast_multinomial", False)
        FastCPUModel._attach(model)
        assert torch.multinomial is installed  # never wrapped twice
        ids = torch.randint(3, 8192, (3, 6))

        def run():
            torch.manual_seed(4)
            with torch.no_grad():
                return model.generate(ids, attention_mask = torch.ones_like(ids), do_sample = True, max_new_tokens = 12,
                                      top_k = 0, top_p = 1.0, temperature = 1.3)

        fast, n = _engaged(run)
        assert n == 12
        monkeypatch.setenv("UNSLOTH_CPU_FAST_MULTINOMIAL", "0")
        stock, n = _engaged(run)
        assert n == 0 and torch.equal(fast, stock)
    finally:
        torch.multinomial = saved


def test_compiled_dynamic_sampling_adds_no_shape_guard():
    """Under torch.compile the wrapper is torch's call: no guard on the fast path's size threshold, so a marked
    dynamic range across it compiles once and lengths on both sides of it reuse one graph."""
    import torch._dynamo
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    def sample(x):
        return FAST(x, 1)

    torch._dynamo.reset()
    fn = torch.compile(sample, backend = backend, fullgraph = True, dynamic = True)
    for n in (2048, 8192, 1024):
        x = torch.softmax(torch.randn(n), -1)
        torch._dynamo.mark_dynamic(x, 0, min = 2, max = 10000)
        assert fn(x).shape == (1,)
    assert len(graphs) == 1, len(graphs)
    torch._dynamo.reset()


@pytest.mark.parametrize("args", [(1.0,), (1, "x"), (1, 1)])
def test_wrong_argument_types_raise_torchs_error(args):
    p = torch.softmax(torch.randn(8192), -1)
    with pytest.raises((TypeError, RuntimeError)):
        ORIGINAL(p, *args)
    with pytest.raises((TypeError, RuntimeError)):
        FAST(p, *args)
