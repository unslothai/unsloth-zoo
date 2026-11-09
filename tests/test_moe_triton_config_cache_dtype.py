# SPDX-License-Identifier: AGPL-3.0-only
import pytest
import torch
import torch.nn as nn

from unsloth_zoo.temporary_patches import moe_utils


class _Stop(Exception):
    pass


def _experts(E = 4, H = 32, I = 48):
    m = nn.Module()
    m.gate_up_proj = nn.Parameter(torch.zeros(E, 2 * I, H))
    m.down_proj = nn.Parameter(torch.zeros(E, H, I))
    m.num_experts = E
    return m


@pytest.fixture
def tuner_calls(monkeypatch):
    autotune_cache = pytest.importorskip("unsloth.kernels.moe.autotune_cache")
    pytest.importorskip("unsloth.kernels.moe.grouped_gemm.interface")
    calls = []

    def fake_autotune(*args, **kwargs):
        calls.append(kwargs["dtype"])
        return (object(), object(), object())

    def stop(*args, **kwargs):
        raise _Stop()

    monkeypatch.setattr(autotune_cache, "get_or_autotune_moe_kernels", fake_autotune)
    monkeypatch.setattr(moe_utils, "moe_routing_sort", stop)
    monkeypatch.setattr(moe_utils, "_should_use_separated_lora", lambda: False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    return calls


def _call(experts, dtype, H = 32):
    with pytest.raises(_Stop):
        moe_utils.forward_triton_grouped_gemm(
            experts, torch.zeros(3, H, dtype = dtype), torch.zeros(3, 1, dtype = torch.long), torch.ones(3, 1, dtype = dtype),
        )


@pytest.mark.parametrize("first, second", [(torch.float32, torch.bfloat16), (torch.bfloat16, torch.float32)])
def test_configs_reselected_after_module_dtype_change(monkeypatch, tuner_calls, first, second):
    monkeypatch.setenv("UNSLOTH_MOE_TRITON_FP32", "1")
    experts = _experts().to(first)
    _call(experts, first)
    _call(experts, first)
    assert tuner_calls == [first, first]  # gate_up and down tuned once
    experts.to(second)
    _call(experts, second)
    assert tuner_calls == [first, first, second, second]


def test_configs_reselected_when_only_activation_dtype_changes(tuner_calls):
    # float16 experts fed float16 then float32 activations (float32 is not cast down to float16): new operand widths.
    experts = _experts().to(torch.float16)
    _call(experts, torch.float16)
    _call(experts, torch.float32)
    _call(experts, torch.float32)
    assert tuner_calls == [torch.float16, torch.float16, torch.float32, torch.float32]


def test_configs_reselected_after_device_change(tuner_calls):
    experts = _experts()
    _call(experts, torch.float32)
    experts.to("meta")
    with pytest.raises(_Stop):
        moe_utils.forward_triton_grouped_gemm(
            experts,
            torch.zeros(3, 32, device = "meta"),
            torch.zeros(3, 1, dtype = torch.long, device = "meta"),
            torch.ones(3, 1, device = "meta"),
        )
    assert tuner_calls == [torch.float32, torch.float32] * 2


def test_preset_configs_without_key_are_reused(tuner_calls):
    experts = _experts()
    preset = (48, (None, None, None), (None, None, None))
    experts._unsloth_moe_configs = preset
    _call(experts, torch.float32)
    _call(experts, torch.float32)
    assert tuner_calls == []
    assert experts._unsloth_moe_configs is preset


def _selection_prefix():
    # The config-selection prefix of the real forward, up to routing; the GEMMs are replaced by x * 2.
    import ast, inspect, textwrap

    src = textwrap.dedent(inspect.getsource(moe_utils.forward_triton_grouped_gemm))
    fn = ast.parse(src).body[0]
    body = []
    for stmt in fn.body:
        if isinstance(stmt, ast.With):
            break
        if not isinstance(stmt, ast.ImportFrom):
            body.append(stmt)
    fn.body = body + ast.parse("return gemm_hidden_states * 2").body
    fn.decorator_list = []
    ast.fix_missing_locations(fn)
    calls = []

    @torch.compiler.disable
    def tuner(**kwargs):
        calls.append(kwargs["dtype"])
        return (None, None, None)

    ns = {
        "torch": torch,
        "get_or_autotune_moe_kernels": tuner,
        "_triton_grouped_gemm_dims_supported": lambda *a: True,
        "_should_use_separated_lora": lambda: False,
        "take_moe_lora_stash": lambda *a: None,
        "_triton_moe_compute_dtype": lambda m, x: x.dtype,
        "_experts_are_input_major": lambda *a: False,
        "_moe_weight_dtype_kwargs": lambda *a: {},
    }
    exec(compile(ast.Module(body = [fn], type_ignores = []), "<prefix>", "exec"), ns)
    return ns[fn.name], calls


def test_dtype_change_never_reselects_inside_compiled_forward(monkeypatch):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    forward, calls = _selection_prefix()
    experts = _experts(E = 2, H = 32, I = 32)
    idx = torch.zeros(4, 1, dtype = torch.long)
    forward(experts, torch.ones(4, 32), idx, torch.ones(4, 1))  # eager warmup selects fp32 configs
    assert calls == [torch.float32, torch.float32]
    compiled = torch.compile(forward, backend = "eager", fullgraph = True, dynamic = True)
    compiled(experts, torch.ones(4, 32), idx, torch.ones(4, 1))
    experts.to(torch.bfloat16)
    x = torch.ones(6, 32, dtype = torch.bfloat16, requires_grad = True)
    compiled(experts, x, torch.zeros(6, 1, dtype = torch.long), torch.ones(6, 1, dtype = torch.bfloat16)).sum().backward()
    assert dict(torch._dynamo.utils.counters["graph_break"]) == {}
    assert calls == [torch.float32, torch.float32]  # no tuner call while traced
    # The next eager call re-selects for the new dtype.
    forward(experts, torch.ones(4, 32, dtype = torch.bfloat16), idx, torch.ones(4, 1, dtype = torch.bfloat16))
    assert calls == [torch.float32, torch.float32, torch.bfloat16, torch.bfloat16]
    torch._dynamo.reset()
