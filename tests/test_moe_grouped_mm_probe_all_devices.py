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

"""grouped_mm and ROCm crash probes answer per GPU model, probing only the cards experts actually run on."""
import subprocess
import sys
import types

import pytest
import torch

from unsloth_zoo.temporary_patches import moe_utils


@pytest.fixture(autouse = True)
def _fresh_dynamo_rules():
    # Torn down after monkeypatch restores torch: a compile under the mocks caches Dynamo's torch
    # rule map without the real _grouped_mm, which breaks later compiled MoE tests in this process.
    yield
    import torch._dynamo
    import torch._dynamo.trace_rules as trace_rules
    for name in dir(trace_rules):
        clear = getattr(getattr(trace_rules, name), "cache_clear", None)
        if callable(clear):
            clear()
    torch._dynamo.reset()


@pytest.fixture
def spoof(monkeypatch):
    def make(count, failing = (), hip = None, current = 0, models = None):
        models = models or [f"GPU{i}" for i in range(count)]
        def props(i): return types.SimpleNamespace(name = models[i], major = 9, minor = 0, gcnArchName = models[i])
        monkeypatch.setattr(torch.cuda, "get_device_properties", props)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: current)
        monkeypatch.setattr(torch.version, "hip", hip, raising = False)
        monkeypatch.setattr(moe_utils, "_TORCH_GROUPED_MM_AVAILABLE", True)
        monkeypatch.setattr(moe_utils, "_TORCH_GROUPED_MM_SUPPORTED", None)
        monkeypatch.setattr(moe_utils, "_GROUPED_MM_SURVIVES", None)
        monkeypatch.setattr(moe_utils, "_GROUPED_MM_HOME", None, raising = False)
        monkeypatch.setattr(moe_utils, "_GROUPED_MM_HOMOGENEOUS", False, raising = False)
        monkeypatch.setattr(moe_utils, "_TRANSPOSED_VIEW_GROUPED_MM_SAFE", None)
        for name in ("_GROUPED_MM_FOREIGN", "_TORCH_GROUPED_MM_SUPPORTED_BY_KIND", "_GROUPED_MM_SURVIVES_BY_KIND",
                     "_TRANSPOSED_VIEW_SAFE_BY_KIND"):
            monkeypatch.setattr(moe_utils, name, {}, raising = False)
        monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising = False)
        monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising = False)
        monkeypatch.delenv("GPU_DEVICE_ORDINAL", raising = False)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
        probed, last = [], {}
        real_ones, real_tensor = torch.ones, torch.tensor

        def ones(*a, device = None, **k):
            last["d"] = torch.device(device)
            return real_ones(*a, **k)

        def tensor(*a, device = None, **k):
            return real_tensor(*a, **k)

        def grouped_mm(x, w, offs = None):
            probed.append(last["d"].index)
            if last["d"].index in failing:
                raise RuntimeError("grouped_mm unsupported on this card")

        monkeypatch.setattr(torch, "ones", ones)
        monkeypatch.setattr(torch, "tensor", tensor)
        monkeypatch.setattr(torch, "_grouped_mm", grouped_mm, raising = False)
        return probed
    return make


def cuda(i): return torch.device("cuda", i)


@pytest.mark.parametrize("count, failing, expected", [
    (1, (), True), (1, (0,), False), (2, (), True), (2, (1,), True), (3, (2,), True), (3, (0,), False),
])
def test_default_probe_touches_only_the_current_card(spoof, count, failing, expected):
    seen = spoof(count, failing)
    assert moe_utils._probe_torch_grouped_mm_supported() is expected
    assert moe_utils._check_torch_grouped_mm_supported() is expected
    assert moe_utils._check_torch_grouped_mm_supported(cuda(0)) is expected
    assert seen == [0]
    assert moe_utils._TORCH_GROUPED_MM_SUPPORTED is expected


def test_mixed_host_each_card_answers_for_itself(spoof):
    seen = spoof(2, failing = (1,), models = ["H100", "RTX4090"])
    assert moe_utils._check_torch_grouped_mm_supported(cuda(0)) is True   # unused 4090 never disables the H100
    assert seen == [0]
    assert moe_utils._check_torch_grouped_mm_supported(cuda(1)) is False  # experts spilled onto the 4090
    assert moe_utils._check_torch_grouped_mm_supported(cuda(0)) is True
    assert moe_utils._check_torch_grouped_mm_supported() is True
    assert moe_utils._check_torch_grouped_mm_supported(cuda(1)) is False
    assert seen == [0, 1]


def test_mixed_host_current_card_unsupported(spoof):
    seen = spoof(2, failing = (0,), models = ["RTX4090", "H100"])
    assert moe_utils._check_torch_grouped_mm_supported(cuda(1)) is True
    assert seen == [1]
    assert moe_utils._check_torch_grouped_mm_supported() is False
    assert seen == [1, 0]


def test_a_current_device_switch_does_not_borrow_the_first_cards_answer(spoof, monkeypatch):
    # Model parallel moves the current device; cuda:1 must not reuse the answer probed on cuda:0.
    seen = spoof(2, failing = (1,), models = ["H100", "RTX4090"])
    assert moe_utils._grouped_mm_supported_on(cuda(0)) is True
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    assert moe_utils._grouped_mm_supported_on(cuda(1)) is False
    assert moe_utils._grouped_mm_supported_on(cuda(0)) is True
    assert seen == [0, 1]


def test_first_callers_on_different_cards_agree_on_the_home_card(spoof, monkeypatch):
    # Two first callers on different-model cards, neither probed yet: the second must not also claim "home".
    seen = spoof(2, failing = (1,), models = ["H100", "RTX4090"])
    assert moe_utils._grouped_mm_foreign_kind(cuda(0)) is None
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    assert moe_utils._grouped_mm_foreign_kind(cuda(1)) is not None
    assert moe_utils._check_torch_grouped_mm_supported() is True
    assert moe_utils._grouped_mm_supported_on(cuda(1)) is False
    assert seen == [0, 1]


def test_an_xpu_card_never_borrows_the_cuda_answer(spoof, monkeypatch):
    # CUDA and XPU on one host: an XPU tensor is a different backend, so it gets its own answer.
    spoof(1, models = ["H100"])
    monkeypatch.setattr(torch, "xpu", types.SimpleNamespace(
        get_device_properties = lambda i: types.SimpleNamespace(name = "Arc B580", major = None, minor = None),
    ), raising = False)
    assert moe_utils._grouped_mm_foreign_kind(cuda(0)) is None
    kind = moe_utils._grouped_mm_foreign_kind(torch.device("xpu", 0))
    assert kind is not None and kind[0] == "xpu"


@pytest.mark.parametrize("count", [2, 3, 8])
def test_homogeneous_host_keeps_one_probe(spoof, count):
    seen = spoof(count, models = ["MI300X"] * count)
    for i in range(count):
        assert moe_utils._check_torch_grouped_mm_supported(cuda(i)) is True
    assert seen == [0]


def test_same_model_cards_share_one_probe(spoof):
    seen = spoof(4, failing = (3,), models = ["H100", "H100", "A100", "A100"])
    assert [moe_utils._check_torch_grouped_mm_supported(cuda(i)) for i in range(4)] == [True] * 4
    assert seen == [0, 2]


def test_unreadable_properties_probe_the_card_when_used(spoof, monkeypatch):
    seen = spoof(2, failing = (1,), models = ["MI300X"] * 2)
    def props(i):
        if i == 1:
            raise RuntimeError("no properties")
        return types.SimpleNamespace(name = "MI300X", major = 9, minor = 4)
    monkeypatch.setattr(torch.cuda, "get_device_properties", props)
    assert moe_utils._check_torch_grouped_mm_supported() is True
    assert seen == [0]
    assert moe_utils._check_torch_grouped_mm_supported(cuda(1)) is False
    assert seen == [0, 1]


def test_call_site_gate_routes_same_model_cards_to_the_no_arg_check(spoof, monkeypatch):
    seen = spoof(3, failing = (2,), models = ["H100", "H100", "RTX4090"])
    assert [moe_utils._grouped_mm_supported_on(cuda(i)) for i in range(3)] == [True, True, False]
    assert seen == [0, 2]
    monkeypatch.setattr(moe_utils, "_check_torch_grouped_mm_supported", lambda: "patched")
    assert moe_utils._grouped_mm_supported_on(cuda(1)) == "patched"
    assert moe_utils._grouped_mm_supported_on(torch.device("cpu")) == "patched"


def test_cpu_or_indexless_device_uses_the_default_answer(spoof):
    seen = spoof(2, failing = (1,), models = ["H100", "RTX4090"])
    assert moe_utils._check_torch_grouped_mm_supported(torch.device("cpu")) is True
    assert moe_utils._check_torch_grouped_mm_supported(torch.device("cuda")) is True
    assert seen == [0]


def test_grouped_mm_eager_asks_for_the_input_device(spoof, monkeypatch):
    spoof(2)
    asked = []
    monkeypatch.setattr(moe_utils, "_grouped_mm_supported_on", lambda device: asked.append(device) or False)
    monkeypatch.setattr(moe_utils, "_manual_grouped_mm", lambda *a: "manual")
    x = torch.zeros(2, 4)
    assert moe_utils._grouped_mm_eager(x, torch.zeros(1, 4, 4), torch.tensor([2])) == "manual"
    assert asked == [x.device]


def _record_children(monkeypatch, crash_masks = ()):
    children = []

    def run(cmd, env = None, **k):
        children.append((cmd[-1], None if env is None else (env["HIP_VISIBLE_DEVICES"], env["CUDA_VISIBLE_DEVICES"])))
        crashed = env is not None and env["HIP_VISIBLE_DEVICES"] in crash_masks
        return subprocess.CompletedProcess(cmd, -11 if crashed else 0)

    monkeypatch.setattr(subprocess, "run", run)
    return children


def test_rocm_single_gpu_child_unchanged(spoof, monkeypatch):
    spoof(1, hip = "7.1.0")
    children = _record_children(monkeypatch)
    assert moe_utils._probe_torch_grouped_mm_supported() is True
    assert children == [("0", None)]


def test_rocm_homogeneous_host_one_child(spoof, monkeypatch):
    spoof(4, hip = "7.1.0", models = ["MI300X"] * 4)
    children = _record_children(monkeypatch)
    assert moe_utils._probe_torch_grouped_mm_supported() is True
    assert len(children) == 1


def test_rocm_crash_probe_one_child_per_used_model(spoof, monkeypatch):
    spoof(3, hip = "7.1.0")
    children = _record_children(monkeypatch)
    assert moe_utils._probe_torch_grouped_mm_supported() is True
    assert children == [("0", ("0", "0"))]
    assert moe_utils._check_torch_grouped_mm_supported(cuda(2)) is True
    assert children == [("0", ("0", "0")), ("0", ("2", "2"))]
    assert moe_utils._grouped_mm_survives_out_of_process(cuda(2)) is True
    assert len(children) == 2


def test_rocm_crash_on_unused_card_keeps_grouped_mm(spoof, monkeypatch):
    seen = spoof(2, hip = "7.1.0", models = ["gfx1100", "gfx1036"])
    children = _record_children(monkeypatch, crash_masks = ("1",))
    assert moe_utils._probe_torch_grouped_mm_supported() is True
    assert moe_utils._check_torch_grouped_mm_supported(cuda(1)) is False
    assert [c[1] for c in children] == [("0", "0"), ("1", "1")]
    assert seen == [0]   # the crashing card's kernel never ran in-process


@pytest.mark.parametrize("var", ["HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"])
def test_rocm_crash_probe_indexes_into_inherited_mask(spoof, monkeypatch, var):
    spoof(3, hip = "7.1.0")
    monkeypatch.setattr(sys, "platform", "linux")  # an unsorted mask is not mask order on Windows (below)
    monkeypatch.setenv(var, "5, 2,7")
    children = _record_children(monkeypatch)
    assert all(moe_utils._check_torch_grouped_mm_supported(cuda(i)) for i in range(3))
    assert [c[1] for c in children] == [("5", "5"), ("2", "2"), ("7", "7")]


@pytest.mark.parametrize("var", ["ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"])
def test_rocm_layered_selector_keeps_the_parents_env_and_index(spoof, monkeypatch, var):
    # A child HIP mask would compose with ROCr / GPU_DEVICE_ORDINAL and could name another card.
    spoof(3, hip = "7.1.0")
    monkeypatch.setenv(var, "1,3,4")
    children = _record_children(monkeypatch)
    assert all(moe_utils._check_torch_grouped_mm_supported(cuda(i)) for i in range(3))
    assert children == [("0", None), ("1", None), ("2", None)]


def test_rocm_repeated_hip_entries_follow_clr(spoof, monkeypatch):
    spoof(2, hip = "7.1.0")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "5,5,7")
    children = _record_children(monkeypatch)
    assert moe_utils._check_torch_grouped_mm_supported(cuda(1)) is True
    assert children[-1] == ("0", ("7", "7"))


def test_crash_probe_never_spawns_on_multi_gpu_nvidia(spoof, monkeypatch):
    seen = spoof(3)
    def boom(*a, **k): raise AssertionError("child probe launched off AMD")
    monkeypatch.setattr(subprocess, "run", boom)
    assert all(moe_utils._check_torch_grouped_mm_supported(cuda(i)) for i in range(3))
    assert seen == [0, 1, 2]


def test_real_child_sees_only_its_card(spoof, monkeypatch):
    spoof(3, hip = "7.1.0")
    monkeypatch.setattr(moe_utils, "_GROUPED_MM_CRASH_PROBE",
        "import os, signal, sys\n"
        "assert sys.argv[1] == '0' and os.environ['CUDA_VISIBLE_DEVICES'] == os.environ['HIP_VISIBLE_DEVICES']\n"
        "if os.environ['HIP_VISIBLE_DEVICES'] == '2': os.kill(os.getpid(), signal.SIGSEGV)\n")
    assert moe_utils._grouped_mm_survives_out_of_process(cuda(0)) is True
    assert moe_utils._grouped_mm_survives_out_of_process(cuda(2)) is False
    monkeypatch.setattr(moe_utils, "_GROUPED_MM_SURVIVES_BY_KIND", {})
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "4,5,6")
    assert moe_utils._grouped_mm_survives_out_of_process(cuda(2)) is True


def test_routed_grouped_gate_asks_for_the_hidden_states_device():
    import inspect
    from unsloth_zoo.temporary_patches import moe_routed
    assert "_grouped_mm_supported_on(hidden_states.device)" in inspect.getsource(moe_routed)


def _fullgraph_gate(monkeypatch, device):
    import torch._dynamo
    torch._dynamo.reset()
    # Dynamo initialises CUDA when told it exists; probes are done by now, so stop spoofing it.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    @torch.compile(fullgraph = True, backend = "eager")
    def f(x):
        return x + 1 if moe_utils._grouped_mm_supported_on(device) else x - 1
    return f(torch.zeros(2))


@pytest.mark.parametrize("count, index", [(1, 0), (2, 1), (4, 3)])
def test_warmed_gate_traces_without_a_break_on_every_same_model_card(spoof, monkeypatch, count, index):
    # The no-arg warmup leaves every same-model card resolved: a fullgraph trace must not reach the compiler-disabled resolver.
    seen = spoof(count, models = ["H100"] * count)
    assert moe_utils._check_torch_grouped_mm_supported() is True
    assert _fullgraph_gate(monkeypatch, cuda(index)).tolist() == [1.0, 1.0]
    assert seen == [0]


def test_absent_grouped_mm_symbol_gate_stays_traceable(spoof, monkeypatch):
    # torch 2.6 / 2.7: no torch._grouped_mm, the gate is a traceable definitive negative.
    spoof(1)
    monkeypatch.setattr(moe_utils, "_TORCH_GROUPED_MM_AVAILABLE", False)
    assert moe_utils._check_torch_grouped_mm_supported() is False
    assert _fullgraph_gate(monkeypatch, cuda(0)).tolist() == [-1.0, -1.0]


def test_view_probe_runs_on_the_input_card(spoof, monkeypatch):
    # cuda:0 (unsupported, unused) must not decide whether cuda:1 copies its weights.
    spoof(2, failing = (0,), models = ["RTX4090", "H100"])
    probed = []
    monkeypatch.setattr(moe_utils, "_transposed_view_probe", lambda d: probed.append(d.index) or d.index == 1)
    assert moe_utils._grouped_mm_supported_on(cuda(1)) is True
    # Probed with support, in the same eager region: the view check opens no second one.
    assert probed == [1]
    def no_second_region(fn): raise AssertionError("second compiler-disabled region")
    monkeypatch.setattr(moe_utils, "_run_probe_eagerly", no_second_region)
    assert moe_utils._transposed_view_safe_on(cuda(1)) is True
    assert moe_utils._transposed_view_safe_on(cuda(1)) is True
    assert probed == [1]
    assert moe_utils._TRANSPOSED_VIEW_GROUPED_MM_SAFE is None


def test_view_probe_same_model_keeps_the_process_wide_flag(spoof, monkeypatch):
    spoof(2, models = ["H100", "H100"])
    probed = []
    monkeypatch.setattr(moe_utils, "_transposed_view_probe", lambda d: probed.append(d.index) or True)
    assert moe_utils._transposed_view_safe_on(cuda(1)) is True
    assert moe_utils._transposed_view_grouped_mm_is_safe() is True
    assert probed == [0]


def test_rocm_quoted_mask_is_unquoted(spoof, monkeypatch):
    spoof(3, hip = "7.1.0")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", '"5,2,7"')
    children = _record_children(monkeypatch)
    assert all(moe_utils._check_torch_grouped_mm_supported(cuda(i)) for i in range(3))
    assert [c[1] for c in children] == [("5", "5"), ("2", "2"), ("7", "7")]


@pytest.mark.parametrize("platform, mask, rewritten", [
    ("linux", "0,GPU-4d2a6f1e8c3b,1", False),   # a UUID may alias an ordinal: keep the parent's view
    ("win32", "1,0", False),                    # PAL enumerates in physical order, not mask order
    ("win32", "0,1", True),
    ("linux", "1,0", True),
])
def test_rocm_mask_rewritten_only_when_token_order_is_logical_order(spoof, monkeypatch, platform, mask, rewritten):
    spoof(2, hip = "7.1.0")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", mask)
    monkeypatch.setattr(moe_utils.sys, "platform", platform)
    children = _record_children(monkeypatch)
    assert all(moe_utils._check_torch_grouped_mm_supported(cuda(i)) for i in range(2))
    if rewritten:
        tokens = mask.split(",")
        assert children == [("0", (tokens[0],) * 2), ("0", (tokens[1],) * 2)]
    else:
        assert children == [("0", None), ("1", None)]


def test_identical_cards_read_the_global_flags_without_classifying_each_call(spoof, monkeypatch):
    # One GPU model on the host: after the eager anchor, a GEMM's gates cost a flag read, as before #1928.
    seen = spoof(2, models = ["B200", "B200"])
    monkeypatch.setattr(moe_utils, "_TRANSPOSED_VIEW_GROUPED_MM_SAFE", True)
    assert moe_utils._grouped_mm_supported_on(cuda(0)) is True
    assert moe_utils._GROUPED_MM_HOMOGENEOUS is True
    lookups = []
    real = moe_utils._grouped_mm_foreign_kind
    monkeypatch.setattr(moe_utils, "_grouped_mm_foreign_kind", lambda d: lookups.append(d) or real(d))
    for _ in range(5):
        for i in (0, 1):
            assert moe_utils._grouped_mm_supported_on(cuda(i)) is True
            assert moe_utils._transposed_view_safe_on(cuda(i)) is True
    assert lookups == []
    assert seen == [0]


def test_a_mixed_host_keeps_per_card_answers(spoof):
    seen = spoof(2, failing = (1,), models = ["H100", "RTX4090"])
    assert moe_utils._grouped_mm_supported_on(cuda(0)) is True
    assert moe_utils._GROUPED_MM_HOMOGENEOUS is False
    assert moe_utils._grouped_mm_supported_on(cuda(1)) is False
    assert seen == [0, 1]


def test_tracing_an_unprobed_card_never_runs_a_probe(spoof, monkeypatch):
    # A block moved to an unwarmed card after enable: the trace must not reach the compiler-disabled probe.
    spoof(2, models = ["H100", "RTX4090"])
    monkeypatch.setattr(moe_utils, "_TRANSPOSED_VIEW_GROUPED_MM_SAFE", True)
    assert moe_utils._grouped_mm_supported_on(cuda(0)) is True
    def no_probe(_probe): raise AssertionError("probe ran while tracing")
    monkeypatch.setattr(moe_utils, "_run_probe_eagerly", no_probe)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    assert moe_utils._transposed_view_safe_on(cuda(1)) is True    # the home verdict, as before #1928
    assert moe_utils._grouped_mm_supported_on(cuda(1)) is True    # unprobed model: the home answer, as before #1928
    monkeypatch.setattr(moe_utils, "_TRANSPOSED_VIEW_GROUPED_MM_SAFE", None)
    assert moe_utils._transposed_view_safe_on(cuda(1)) is False   # nothing probed at all: the contiguous copy


def test_tracing_a_card_of_another_accelerator_type_never_resolves_it(spoof, monkeypatch):
    # Mixed CUDA/XPU: the eager anchor classifies only the home type, so an XPU card first seen in a
    # trace must not reach the compiler-disabled resolver; it takes the home answers.
    spoof(2, models = ["H100", "RTX4090"])
    monkeypatch.setattr(moe_utils, "_TRANSPOSED_VIEW_GROUPED_MM_SAFE", True)
    assert moe_utils._grouped_mm_supported_on(cuda(0)) is True
    def no_probe(_probe): raise AssertionError("resolver ran while tracing")
    monkeypatch.setattr(moe_utils, "_run_probe_eagerly", no_probe)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    xpu = types.SimpleNamespace(type = "xpu", index = 0)
    assert moe_utils._grouped_mm_supported_on(xpu) is True
    assert moe_utils._transposed_view_safe_on(xpu) is True
    assert ("xpu", 0) not in moe_utils._GROUPED_MM_FOREIGN


def test_a_fullgraph_compile_on_an_unprobed_card_traces(spoof, monkeypatch):
    spoof(2, models = ["H100", "RTX4090"])
    monkeypatch.setattr(moe_utils, "_TRANSPOSED_VIEW_GROUPED_MM_SAFE", True)
    assert moe_utils._grouped_mm_supported_on(cuda(0)) is True
    # Not a real torch.device: tracing one would initialise CUDA on a runner without a GPU.
    other = types.SimpleNamespace(type = "cuda", index = 1)

    def gate(x):
        view = moe_utils._transposed_view_safe_on(other)
        supported = moe_utils._grouped_mm_supported_on(other)
        return x + (1 if view else 0) + (10 if supported else 0)

    # The cards are spoofed: keep Dynamo from saving a CUDA RNG state no runner without a GPU has.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    out = torch.compile(gate, fullgraph = True, backend = "eager")(torch.zeros(2))
    assert out.tolist() == [11.0, 11.0]   # traced without a probe: the home card's verdicts


def test_the_modulelist_forward_declines_a_card_without_grouped_mm(monkeypatch):
    # A block moved after enable onto a GPU model without torch._grouped_mm runs its original loop.
    from types import SimpleNamespace
    from unsloth_zoo.temporary_patches import moe_grouped_modulelist as ml
    from unsloth_zoo.temporary_patches import moe_utils as mu

    monkeypatch.setattr(mu, "_grouped_mm_supported_on", lambda device: False)
    sentinel = object()
    block = SimpleNamespace(
        _unsloth_moe_spec = None, experts = [], _orig_moe_forward = lambda hs: sentinel
    )
    hidden = SimpleNamespace(device = torch.device("cuda", 1), dtype = torch.bfloat16)
    assert ml.grouped_moe_forward(block, hidden) is sentinel
