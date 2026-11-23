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

"""ROCm: flex_attention calls whose backward runs are compiled with static shapes. transformers' default compile
recompiles with a symbolic sequence length at the second length, and that kernel returned inf / NaN gradients on
gfx1151 (tiny Gemma-3 trained at 24 tokens after a 32 token step). No-grad calls (decode) keep the stock compile.
Runs on CPU with HIP simulated: real Dynamo with a recording backend, flex replaced by a math attention (CPU flex has
no backward), the ROCm device check widened to CPU tensors where a test needs it."""

import inspect
import os
import subprocess
import sys

import pytest
import torch

flex_module = pytest.importorskip("transformers.integrations.flex_attention")
pytest.importorskip("torch.nn.attention.flex_attention")

# The rocm fixture records torch.compile; torch 2.13+ takes a per-call recompile_limit.
_REAL_TORCH_COMPILE = torch.compile


def _unwrap(fn):
    while getattr(fn, "_unsloth_rocm_static", False) and hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


# A ROCm host patches these at import: keep the stock objects as the baseline.
ORIGINAL_HELPER = _unwrap(flex_module.compile_friendly_flex_attention)
ORIGINAL_FLEX = _unwrap(flex_module.flex_attention)
ORIGINAL_CALL = _unwrap(flex_module.WrappedFlexAttention.__call__)


def math_flex(query, key, value, score_mod = None, block_mask = None, scale = None, enable_gqa = False, **kwargs):
    scores = (query @ key.transpose(-1, -2)) * (scale or query.shape[-1] ** -0.5)
    return scores.softmax(-1) @ value


@pytest.fixture
def rocm(monkeypatch):
    """Return (graphs, install). install(hip, cpu_is_rocm_gpu = True) installs the ROCm patch on fresh transformers
    globals; graphs lists (compiled function name, compile kwargs, symbolic placeholder names) per captured graph."""
    from unsloth_zoo.temporary_patches import misc
    try:
        from unsloth_zoo import rocm_flex_static
    except ImportError:  # older Zoo: the device check below does not exist
        rocm_flex_static = None

    graphs = []
    real_compile = torch.compile

    def recording_compile(fn, **kwargs):
        name = getattr(fn, "__name__", "?")
        recorded = dict(kwargs)

        def backend(gm, inputs):
            symbolic = [
                node.name for node in gm.graph.nodes
                if node.op == "placeholder" and isinstance(node.meta.get("example_value"), torch.SymInt)
            ]
            graphs.append((name, recorded, symbolic))
            return gm.forward
        kwargs.pop("mode", None)
        return real_compile(fn, backend = backend, **kwargs)

    def install(hip = "7.13.0", cpu_is_rocm_gpu = True):
        monkeypatch.setattr(torch.version, "hip", hip, raising = False)
        monkeypatch.setattr(flex_module, "compile_friendly_flex_attention", ORIGINAL_HELPER)
        monkeypatch.setattr(flex_module, "flex_attention", math_flex)
        monkeypatch.setattr(flex_module.WrappedFlexAttention, "__call__", ORIGINAL_CALL)
        monkeypatch.setattr(flex_module.WrappedFlexAttention, "_instance", None)
        monkeypatch.setattr(torch, "compile", recording_compile)
        if cpu_is_rocm_gpu and rocm_flex_static is not None:
            monkeypatch.setattr(rocm_flex_static, "_affected_device", lambda device: True)
        misc.patch_flex_attention_static_shapes_on_rocm()
        return flex_module.compile_friendly_flex_attention

    torch._dynamo.reset()
    yield graphs, install
    torch._dynamo.reset()


def _qkv(length, grad = True, heads = 2):
    return [torch.randn(1, heads, length, 8, requires_grad = grad) for _ in range(3)]


def _static(graphs):
    return [g for g in graphs if g[0] == "rocm_static_flex_attention"]


def _stock(graphs):
    return [g for g in graphs if g[0] != "rocm_static_flex_attention"]


def test_training_calls_compile_static_per_length(rocm):
    graphs, install = rocm
    fn = install()
    for length in (32, 24, 40, 24):
        fn(*_qkv(length), training = True, scale = 0.5)[0].sum().backward()
    assert len(_static(graphs)) == 3 and _stock(graphs) == []
    assert all(symbolic == [] and kwargs.get("dynamic") is False for _, kwargs, symbolic in _static(graphs))


def test_no_grad_decode_keeps_the_stock_compile(rocm):
    # Growing-KV decode: one query row against 5..12 keys must not compile a graph per length.
    graphs, install = rocm
    fn = install()
    with torch.no_grad():
        for length in range(5, 13):
            q = torch.randn(1, 2, 1, 8)
            fn(q, *_qkv(length, grad = False)[1:], training = True)
    assert _static(graphs) == []
    assert len(_stock(graphs)) <= 2


def test_enclosing_graph_specializes_gradient_lengths(rocm):
    graphs, install = rocm
    install()
    w = torch.randn(8, 8, requires_grad = True)

    def layer(x, training):
        q = (x @ w).unsqueeze(1)
        return flex_module.compile_friendly_flex_attention(q, q, q, training = training)

    compiled = torch.compile(layer, fullgraph = True)
    for length in (32, 24, 40):
        compiled(torch.randn(1, length, 8), True).sum().backward()
    assert len(graphs) == 3 and all(symbolic == [] for _, _, symbolic in graphs), graphs
    graphs.clear()
    with torch.no_grad():
        for length in (5, 6, 7):
            compiled(torch.randn(1, length, 8), False)
    assert len(graphs) <= 2, "no-grad calls inside a graph keep its dynamic shapes"


def test_enclosing_graph_with_a_strictly_dynamic_input_is_left_alone(rocm):
    graphs, install = rocm
    install()
    w = torch.randn(8, 8, requires_grad = True)

    def layer(x):
        q = (x @ w).unsqueeze(1)
        return flex_module.compile_friendly_flex_attention(q, q, q, training = True)

    compiled = torch.compile(layer, fullgraph = True)
    for length in (32, 24):
        x = torch.randn(1, length, 8)
        torch._dynamo.mark_dynamic(x, 1)
        compiled(x).sum().backward()  # specializing would raise ConstraintViolationError
    assert graphs and all(symbolic for _, _, symbolic in graphs)


def _batched_layer():
    w = torch.randn(8, 8, requires_grad = True)

    def layer(x):
        q = (x @ w).unsqueeze(1)
        return flex_module.compile_friendly_flex_attention(q, q, q, training = True)
    return torch.compile(layer, fullgraph = True)


def test_enclosing_graph_keeps_a_dynamic_batch_symbolic(rocm):
    # Block compile marks the block batch maybe_mark_dynamic (#15106): only the sequence length may be specialized,
    # else every batch size is a new block graph and eats the block's isolate_recompiles budget.
    graphs, install = rocm
    install()
    compiled = _batched_layer()
    for batch in (2, 3, 4, 5):
        x = torch.randn(batch, 32, 8)
        torch._dynamo.maybe_mark_dynamic(x, 0)
        compiled(x).sum().backward()
    assert len(graphs) == 1 and all(symbolic for _, _, symbolic in graphs), graphs


def test_enclosing_graph_compiles_once_per_length_not_per_batch_and_length(rocm):
    graphs, install = rocm
    install()
    compiled = _batched_layer()
    for length in (32, 24, 40):
        for batch in (2, 3):
            x = torch.randn(batch, length, 8)
            torch._dynamo.maybe_mark_dynamic(x, 0)
            compiled(x).sum().backward()
    assert len(graphs) == 3, graphs


def test_strictly_dynamic_batch_still_specializes_the_length(rocm):
    # mark_dynamic on the batch only: the length is not derived from it, so it is still made static.
    graphs, install = rocm
    install()
    compiled = _batched_layer()
    for length in (32, 24, 40):
        x = torch.randn(2, length, 8)
        torch._dynamo.mark_dynamic(x, 0)
        compiled(x).sum().backward()
    assert len(graphs) == 3 and all(symbolic for _, _, symbolic in graphs), graphs


@pytest.fixture
def stats(monkeypatch):
    from unsloth_zoo import rocm_flex_static
    monkeypatch.setattr(rocm_flex_static, "ROCM_FLEX_STATIC_STATS", dict.fromkeys(rocm_flex_static.ROCM_FLEX_STATIC_STATS, 0))
    monkeypatch.setattr(rocm_flex_static, "_WARNED", set())
    return rocm_flex_static


def test_engagement_counters(rocm, stats):
    graphs, install = rocm
    fn = install()
    for length in (32, 24):
        fn(*_qkv(length), training = True)[0].sum().backward()
    assert stats.ROCM_FLEX_STATIC_STATS["static_calls"] == 2
    compiled = _batched_layer()
    for length in (32, 24, 40):
        compiled(torch.randn(2, length, 8)).sum().backward()
    counts = stats.ROCM_FLEX_STATIC_STATS
    # The first trace is static already (nothing to specialize); the next two carry a symbolic length.
    assert counts["specialized"] == 2 and counts["failed"] == 0 and counts["left_dynamic"] == 0, counts


def test_strictly_dynamic_length_is_counted_and_warned(rocm, stats):
    graphs, install = rocm
    install()
    compiled = _batched_layer()
    with pytest.warns(RuntimeWarning, match = "stays symbolic"):
        for length in (32, 24):
            x = torch.randn(1, length, 8)
            torch._dynamo.mark_dynamic(x, 1)
            compiled(x).sum().backward()
    assert stats.ROCM_FLEX_STATIC_STATS["left_dynamic"] >= 1


def test_moved_dynamo_internals_warn_once_instead_of_failing_silently(rocm, stats, monkeypatch):
    graphs, install = rocm
    install()

    def moved(ctx):
        raise AttributeError("_ComptimeContext__tx")
    monkeypatch.setattr(stats, "_strict_symbols", moved)
    compiled = _batched_layer()
    with pytest.warns(RuntimeWarning, match = "could not make flex_attention sequence lengths static") as record:
        for length in (32, 24, 40):
            compiled(torch.randn(1, length, 8)).sum().backward()
    assert len([w for w in record if "could not make" in str(w.message)]) == 1
    # The symbolic length is kept (one graph serves 24 and 40), as on a torch whose internals moved.
    assert stats.ROCM_FLEX_STATIC_STATS["failed"] == 1 and stats.ROCM_FLEX_STATIC_STATS["specialized"] == 0
    assert len(graphs) == 2 and graphs[-1][2], graphs


def test_missing_comptime_warns_at_install(rocm, stats, monkeypatch):
    graphs, install = rocm
    monkeypatch.setattr(stats, "_comptime", None)
    with pytest.warns(RuntimeWarning, match = "comptime is unavailable"):
        install()
    assert stats.ROCM_FLEX_STATIC_STATS["failed"] == 1


def test_helper_imported_before_the_patch_is_covered(rocm):
    # modeling_doge imports compile_friendly_flex_attention by name.
    graphs, install = rocm
    imported_early = ORIGINAL_HELPER
    install()
    imported_early(*_qkv(16), training = True)
    assert len(_static(graphs)) == 1 and _stock(graphs) == []


def test_helper_object_and_its_wrappers_stay_installed(rocm, monkeypatch):
    # transformers 4.52 / 4.53 wrap the helper in a nonrecursive compiler.disable; saved attention wraps it too.
    from unsloth_zoo.saved_attention import _wrap_flex_kernel

    graphs, install = rocm
    wrapped = _wrap_flex_kernel(ORIGINAL_HELPER)
    monkeypatch.setattr(flex_module, "compile_friendly_flex_attention", wrapped)
    from unsloth_zoo.temporary_patches import misc
    install()
    monkeypatch.setattr(flex_module, "compile_friendly_flex_attention", wrapped)
    misc.patch_flex_attention_static_shapes_on_rocm()
    assert flex_module.compile_friendly_flex_attention is wrapped


def test_padding_mask_width_is_not_marked_dynamic_on_rocm(rocm, monkeypatch):
    masking_utils = pytest.importorskip("transformers.masking_utils")
    from unsloth_zoo.temporary_patches import misc

    marked = []
    original = _unwrap_mask(masking_utils.flex_attention_mask)
    import inspect
    if "q_length" not in inspect.signature(original).parameters:
        pytest.skip("the padding mask_mod patch covers the transformers 5.x mask API only")

    def run(hip):
        monkeypatch.setattr(torch.version, "hip", hip, raising = False)
        monkeypatch.setattr(masking_utils, "flex_attention_mask", original)
        misc.patch_flex_mask_keeps_padding_mask_mod()
        marked.clear()
        masking_utils.flex_attention_mask(
            batch_size = 1, q_length = 8, kv_length = 8, attention_mask = torch.ones(1, 8, dtype = torch.bool),
        )
        return list(marked)

    from unsloth_zoo import rocm_flex_static
    monkeypatch.setattr(rocm_flex_static, "_affected_device", lambda device: True)
    monkeypatch.setattr(torch._dynamo, "maybe_mark_dynamic", lambda tensor, dim: marked.append(dim))
    assert run(None) == [1], "CUDA / CPU keep the dynamic padding width"
    assert run("7.13.0") == []


def _unwrap_mask(fn):
    while getattr(fn, "_unsloth_keeps_padding", False) and hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def test_static_compile_has_room_for_many_lengths(rocm):
    from unsloth_zoo.rocm_flex_static import _raise_recompile_budget
    from unsloth_zoo.torch_config import set_torch_config

    graphs, install = rocm
    config = torch._dynamo.config
    name = "recompile_limit" if hasattr(config, "recompile_limit") else "cache_size_limit"
    before = getattr(config, name)
    try:
        set_torch_config(config, name, 8)
        fn = install()
        for length in range(16, 28):
            fn(*_qkv(length), training = True)
        assert len(_static(graphs)) == 12, "every length compiled, none fell back to eager"
        if "recompile_limit" in inspect.signature(_REAL_TORCH_COMPILE).parameters:
            # The static compile has its own limit and the process-wide one is left alone.
            assert getattr(config, name) == 8
            assert all(kwargs.get("recompile_limit") == 1024 for _, kwargs, _ in _static(graphs))
        else:
            assert getattr(config, name) >= 1024
        set_torch_config(config, name, 4096)
        _raise_recompile_budget()
        assert getattr(config, name) == 4096, "a larger limit is never lowered"
    finally:
        set_torch_config(config, name, before)


def test_static_entry_has_its_own_code_object(rocm):
    from unsloth_zoo import rocm_flex_static

    seen = []
    graphs, install = rocm
    real = rocm_flex_static.compile_static_flex

    def spy(entry, training = False):
        seen.append(entry)
        return real(entry, training)
    rocm_flex_static.compile_static_flex = spy
    try:
        install()(*_qkv(8), training = True)  # the installer imports compile_static_flex when it runs
    finally:
        rocm_flex_static.compile_static_flex = real
    assert seen and seen[0].__code__ is not math_flex.__code__
    assert seen[0].__code__ is not flex_module.flex_attention.__code__


def test_cpu_operands_on_a_hip_build_keep_the_stock_compile(rocm):
    graphs, install = rocm
    fn = install(cpu_is_rocm_gpu = False)
    fn(*_qkv(16), training = True)
    assert _static(graphs) == [] and len(_stock(graphs)) == 1


@pytest.mark.parametrize("hip", [None, ""])
def test_cuda_and_cpu_builds_are_untouched(rocm, hip):
    _, install = rocm
    assert install(hip) is ORIGINAL_HELPER
    assert flex_module.WrappedFlexAttention.__call__ is ORIGINAL_CALL
    assert flex_module.flex_attention is math_flex


def test_opt_out(rocm, monkeypatch):
    _, install = rocm
    monkeypatch.setenv("UNSLOTH_ROCM_FLEX_STATIC", "0")
    install()
    assert flex_module.WrappedFlexAttention.__call__ is ORIGINAL_CALL


def test_truthy_training_values_share_one_compile(rocm):
    graphs, install = rocm
    fn = install()
    fn(*_qkv(16), training = "yes")
    fn(*_qkv(16), training = True)
    assert len(_static(graphs)) == 1


def test_caller_dynamic_marks_do_not_make_the_static_graph_symbolic(rocm):
    graphs, install = rocm
    fn = install()
    for length in (32, 24):
        q, k, v = _qkv(length)
        for t in (q, k, v):
            torch._dynamo.mark_dynamic(t, 2)
        fn(q, k, v, training = True)
    assert len(_static(graphs)) == 2 and all(symbolic == [] for _, _, symbolic in _static(graphs))


def test_torch_2_6_training_keeps_the_max_autotune_mode(rocm, monkeypatch):
    graphs, install = rocm
    monkeypatch.setattr(torch, "__version__", "2.6.0+rocm6.2")
    fn = install()
    fn(*_qkv(8), training = True)
    fn(*_qkv(8), training = False)  # grad-bearing call with the flag off (Doge passes no flag)
    assert [kwargs.get("mode") for _, kwargs, _ in _static(graphs)] == ["max-autotune-no-cudagraphs", None]


def test_incompatible_helper_signature_skips_install(rocm, monkeypatch):
    from unsloth_zoo.temporary_patches import misc

    def incompatible(query, key, value, *, new_required_arg):
        raise AssertionError("the installer should not call this helper")

    _, install = rocm
    install(None)
    monkeypatch.setattr(torch.version, "hip", "7.13.0", raising = False)
    monkeypatch.setattr(flex_module, "compile_friendly_flex_attention", incompatible)
    misc.patch_flex_attention_static_shapes_on_rocm()
    assert flex_module.WrappedFlexAttention.__call__ is ORIGINAL_CALL
    assert flex_module.flex_attention is math_flex


def test_patch_is_idempotent(rocm):
    from unsloth_zoo.temporary_patches import misc

    _, install = rocm
    install()
    call, flex = flex_module.WrappedFlexAttention.__call__, flex_module.flex_attention
    misc.patch_flex_attention_static_shapes_on_rocm()
    assert flex_module.WrappedFlexAttention.__call__ is call and flex_module.flex_attention is flex


def test_zoo_flex_wrapper_dispatch(monkeypatch):
    # unsloth_zoo.flex_attention.utils, sparse_attention and deepseek_v4_sparse_flex wrap their compiles with this.
    from unsloth_zoo import rocm_flex_static

    calls, compiled = [], []
    monkeypatch.setattr(torch.version, "hip", "7.13.0", raising = False)
    monkeypatch.setattr(rocm_flex_static, "_affected_device", lambda device: True)
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: compiled.append(kwargs) or fn)

    def auto(q, k, v, **kwargs):
        calls.append("auto")
        return q

    def entry(q, k, v, **kwargs):
        calls.append("static")
        return q
    fn = rocm_flex_static.rocm_static_flex(auto, entry)
    fn(*_qkv(8))
    with torch.no_grad():
        fn(*_qkv(8))
    fn(*_qkv(8, grad = False))
    assert calls == ["static", "auto", "auto"] and [c["dynamic"] for c in compiled] == [False]
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    assert rocm_flex_static.rocm_static_flex(auto, entry) is auto


def test_sparse_attention_compile_gets_the_static_variant(monkeypatch):
    from unsloth_zoo import sparse_attention
    from unsloth_zoo.temporary_patches import deepseek_v4_sparse_flex

    monkeypatch.setattr(torch.version, "hip", "7.13.0", raising = False)
    for module in (sparse_attention, deepseek_v4_sparse_flex):
        monkeypatch.setattr(module, "_FLEX", None)
        assert getattr(module._compiled_flex(), "_unsloth_rocm_static", False), module.__name__


def test_shared_flex_compile_gets_the_static_variant():
    # unsloth_zoo.flex_attention.utils builds its compile at import, reading GPU properties.
    code = (
        "import types, torch, unsloth_zoo\n"
        "torch.version.hip = '7.13.0'\n"
        "torch.cuda.device_count = lambda: 1\n"
        "torch.cuda.get_device_properties = lambda i: types.SimpleNamespace(total_memory = 64 << 30, gcnArchName = 'gfx1151', name = 'AMD Radeon 8060S', major = 11, minor = 5)\n"
        "import unsloth_zoo.flex_attention.utils as u\n"
        "assert u.HAS_FLEX_ATTENTION\n"
        "print(bool(getattr(u.flex_attention, '_unsloth_rocm_static', False)))\n"
    )
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # No pytest-session switches (UNSLOTH_PYTEST_NO_GPU and friends would change the import path); the rest of the
    # environment stays (Windows needs SYSTEMROOT and friends to import torch).
    env = {name: value for name, value in os.environ.items() if not name.startswith(("UNSLOTH_", "PYTEST_"))}
    env.update(CUDA_VISIBLE_DEVICES = "", PYTHONPATH = root, UNSLOTH_IS_PRESENT = "1")
    out = subprocess.run([sys.executable, "-c", code], capture_output = True, text = True, env = env, cwd = root)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "True"


def test_sparse_backward_probe_runs_once_per_length(monkeypatch):
    from unsloth_zoo import rocm_flex_static
    from unsloth_zoo.flex_attention import sparse

    monkeypatch.setattr(sparse, "_ROCM_STATIC_FLEX", True)
    monkeypatch.setattr(rocm_flex_static, "_affected_device", lambda device: True)
    monkeypatch.setattr(sparse, "_BACKWARD_OK", set())
    runs = []

    def fn(q, k, v):
        runs.append(q.shape[-2])
        return (q * k * v).sum(-1)
    for length in (16, 16, 24):
        sparse.probe_flex_backward(("kind",), fn, _qkv(length))
    assert runs == [16, 24]
