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

"""GROUPED_MM_SYNC_CALLS counts the grouped GEMMs that read expert offsets on the host."""

import pytest
import torch

from unsloth_zoo.temporary_patches import moe_utils as M


def _case(requires_grad = False):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(13, 16, generator = g, requires_grad = requires_grad)
    w = torch.randn(4, 16, 8, generator = g, requires_grad = requires_grad)
    offsets = torch.tensor([3, 3, 9, 13], dtype = torch.int32)
    return x, w, offsets


def _delta(before):
    after = M.grouped_mm_sync_calls()
    return {k: after[k] - before[k] for k in after}


def test_loop_forward_counts_one_call():
    x, w, offsets = _case()
    before = M.grouped_mm_sync_calls()
    with torch.no_grad():
        M._manual_grouped_mm(x, w, offsets)
    assert _delta(before) == {"torch_fallback": 0, "loop": 1, "compiled_sites": 0}


def test_loop_forward_and_backward_count_two_calls():
    x, w, offsets = _case(requires_grad = True)
    before = M.grouped_mm_sync_calls()
    M._manual_grouped_mm(x, w, offsets).sum().backward()
    assert _delta(before) == {"torch_fallback": 0, "loop": 2, "compiled_sites": 0}


def test_wgrad_loop_counts():
    x, _, offsets = _case()
    grad = torch.randn(13, 8)
    # Force the loop branch, as on a card without torch._grouped_mm.
    saved = M._grouped_mm_supported_on
    M._grouped_mm_supported_on = lambda device: False
    try:
        before = M.grouped_mm_sync_calls()
        out = M._grouped_mm_fp16_wgrad_eager(x, grad, offsets)
    finally:
        M._grouped_mm_supported_on = saved
    assert _delta(before) == {"torch_fallback": 0, "loop": 1, "compiled_sites": 0}
    ref = torch.stack([x[s:e].T @ grad[s:e] for s, e in ((0, 3), (3, 3), (3, 9), (9, 13))])
    torch.testing.assert_close(out, ref)


def test_not_counted_while_compiling(monkeypatch):
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    before = M.grouped_mm_sync_calls()
    M._count_grouped_mm_sync("loop")
    assert _delta(before) == {"torch_fallback": 0, "loop": 0, "compiled_sites": 0}


def test_logs_once_with_logging(monkeypatch, capsys):
    monkeypatch.setenv("UNSLOTH_ENABLE_LOGGING", "1")
    monkeypatch.setattr(M, "_GROUPED_MM_SYNC_LOGGED", [False])
    M._count_grouped_mm_sync("loop")
    M._count_grouped_mm_sync("loop")
    out = capsys.readouterr().out
    assert out.count("one host sync per call") == 1


def test_silent_without_logging(monkeypatch, capsys):
    monkeypatch.delenv("UNSLOTH_ENABLE_LOGGING", raising = False)
    monkeypatch.setattr(M, "_GROUPED_MM_SYNC_LOGGED", [False])
    M._count_grouped_mm_sync("loop")
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "rocm,cap,cuda,dtype,groups,expected",
    [
        (False, (7, 5), "12.8", torch.float16, 64, True),    # T4
        (False, (8, 9), "13.0", torch.bfloat16, 64, True),   # L4
        (False, (12, 0), "13.0", torch.bfloat16, 64, True),  # G4
        (False, (9, 0), "12.8", torch.bfloat16, 64, False),  # H100 CUTLASS bf16
        (False, (10, 0), "13.0", torch.bfloat16, 64, False),  # B200 CUTLASS bf16
        (False, (9, 0), "12.8", torch.float16, 64, True),    # H100 fp16 before cuBLASLt grouped
        (False, (9, 0), "13.3", torch.float16, 64, False),   # cuBLASLt grouped fp16
        (False, (9, 0), "13.3", torch.float16, 2048, True),  # over cuBLASLt's 1024 groups
        (True, None, None, torch.bfloat16, 64, True),        # gfx1151
    ],
)
def test_fallback_predicate(monkeypatch, rocm, cap, cuda, dtype, groups, expected):
    monkeypatch.setattr(M, "_TORCH_GROUPED_MM_FALLBACK_BY_KEY", {})
    monkeypatch.setattr(M, "is_rocm_torch", lambda torch_module: rocm)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    if rocm:
        class Props:
            gcnArchName = "gfx1151"
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: Props())
    else:
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: cap)
        monkeypatch.setattr(torch.version, "cuda", cuda)
    assert M._torch_grouped_mm_takes_fallback(torch.device("cuda", 0), dtype, dtype, groups) is expected


def test_compiled_site_counted_once_per_trace_without_graph_break():
    # CPU stands in for a card torch serves with its fallback; fullgraph=True fails on any graph break.
    torch._dynamo.reset()
    x, w, _ = _case()

    @torch.compile(fullgraph = True, backend = "eager")
    def f(x, w):
        M._count_grouped_mm_sync("torch_fallback", x, w, 4)
        return x @ w[0]

    before = M.grouped_mm_sync_calls()
    for _ in range(3):
        f(x, w)
    assert _delta(before) == {"torch_fallback": 0, "loop": 0, "compiled_sites": 1}
    torch._dynamo.reset()


def test_eager_torch_fallback_counted_per_call():
    x, w, _ = _case()
    before = M.grouped_mm_sync_calls()
    for _ in range(3):
        M._count_grouped_mm_sync("torch_fallback", x, w, 4)
    assert _delta(before) == {"torch_fallback": 3, "loop": 0, "compiled_sites": 0}


def test_compiled_cache_copy_counts_into_package():
    # Unsloth execs a copy of moe_utils from its compiled cache; its calls must land in the package's counters.
    import types
    copy = types.ModuleType("unsloth_compiled_cache.moe_utils_copy")
    copy.__dict__.update({k: v for k, v in vars(M).items() if not k.startswith("__")})
    copy.GROUPED_MM_SYNC_CALLS = {"torch_fallback": 0, "loop": 0, "compiled_sites": 0}
    record = types.FunctionType(M._record_grouped_mm_sync.__code__, copy.__dict__)
    before = M.grouped_mm_sync_calls()
    record("loop")
    assert _delta(before) == {"torch_fallback": 0, "loop": 1, "compiled_sites": 0}
    assert copy.GROUPED_MM_SYNC_CALLS["loop"] == 0


def test_transformers_wrapper_classifies_at_weight_dtype(monkeypatch):
    # transformers casts input.to(weight.dtype) before torch._grouped_mm: fp32 input + bf16 weight is a bf16 call.
    seen = []
    monkeypatch.setattr(
        M, "_torch_grouped_mm_takes_fallback",
        lambda device, dtype, weight_dtype, groups: seen.append((dtype, weight_dtype)) or False,
    )
    import sys
    import types
    import transformers.integrations
    from unsloth_zoo.temporary_patches import moe_experts_interface as MEI
    tm = types.ModuleType("transformers.integrations.moe")  # stand-in: the real one moves across versions
    tm._grouped_mm = lambda i, w, o: None
    monkeypatch.setitem(sys.modules, "transformers.integrations.moe", tm)
    monkeypatch.setattr(transformers.integrations, "moe", tm, raising = False)
    monkeypatch.setattr(MEI, "_moe_utils_module", lambda: M)
    monkeypatch.setattr(M, "_triton_grouped_mm_wanted", lambda i, w: False)
    monkeypatch.setattr(M, "_GROUPED_MM_TRITON_OP", object())
    MEI._patch_transformers_grouped_mm()
    x = torch.randn(13, 16)
    w = torch.randn(4, 16, 8).to(torch.bfloat16)
    tm._grouped_mm(x, w, torch.tensor([3, 3, 9, 13], dtype = torch.int32))
    assert seen == [(torch.bfloat16, torch.bfloat16)]


def test_compiled_site_keeps_group_count(monkeypatch):
    # Over 1024 groups cuBLASLt declines, so the compiled classification must see the real count, as eager does.
    seen = []
    monkeypatch.setattr(
        M, "_torch_grouped_mm_takes_fallback",
        lambda device, dtype, weight_dtype, groups: seen.append(groups) or True,
    )
    torch._dynamo.reset()
    x, w, _ = _case()

    @torch.compile(fullgraph = True, backend = "eager")
    def f(x, w):
        M._count_grouped_mm_sync("torch_fallback", x, w, 2048)
        return x @ w[0]

    f(x, w)
    assert seen == [2048]
    torch._dynamo.reset()


def _patched_transformers_module(monkeypatch, can_use):
    import sys
    import types
    import transformers.integrations
    from unsloth_zoo.temporary_patches import moe_experts_interface as MEI
    tm = types.ModuleType("transformers.integrations.moe")
    tm._grouped_mm = lambda i, w, o: i @ w[0]
    tm._can_use_grouped_mm = can_use
    monkeypatch.setitem(sys.modules, "transformers.integrations.moe", tm)
    monkeypatch.setattr(transformers.integrations, "moe", tm, raising = False)
    monkeypatch.setattr(MEI, "_moe_utils_module", lambda: M)
    monkeypatch.setattr(M, "_triton_grouped_mm_wanted", lambda i, w: False)
    monkeypatch.setattr(M, "_GROUPED_MM_TRITON_OP", object())
    MEI._patch_transformers_grouped_mm()
    return tm


def test_transformers_own_fallback_counts_as_loop(monkeypatch):
    # transformers' grouped_mm_fallback reads offs.tolist(): a sync whatever the card.
    monkeypatch.setattr(M, "_torch_grouped_mm_takes_fallback", lambda *args: False)
    tm = _patched_transformers_module(monkeypatch, lambda i, w, o: False)
    x, w, offsets = _case()
    before = M.grouped_mm_sync_calls()
    tm._grouped_mm(x, w, offsets)
    assert _delta(before) == {"torch_fallback": 0, "loop": 1, "compiled_sites": 0}

    torch._dynamo.reset()
    f = torch.compile(lambda x, w: tm._grouped_mm(x, w, offsets), fullgraph = True, backend = "eager")
    before = M.grouped_mm_sync_calls()
    f(x, w)
    f(x, w)
    assert _delta(before) == {"torch_fallback": 0, "loop": 0, "compiled_sites": 1}
    torch._dynamo.reset()
