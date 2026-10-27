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

"""grouped_mm and ROCm crash probes answer per GPU model: one probe on a homogeneous host, one per model the experts use on a mixed one."""
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
    def make(count, failing = (), hip = None, current = 0, models = None, caps = None):
        models = models or [f"GPU{i}" for i in range(count)]
        caps = caps or [(9, 0)] * count
        def props(i): return types.SimpleNamespace(name = models[i], major = caps[i][0], minor = caps[i][1], gcnArchName = models[i])
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
        # A warm-up on the spoofed host selects grouped_mm; restored on teardown so later tests on this
        # CPU worker do not inherit a backend chosen for cards that do not exist.
        monkeypatch.setattr(moe_utils, "_SELECTED_MOE_BACKEND", None, raising = False)
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
        # The view probe builds real tensors (torch.Generator / randn on the card): on a GPU host that ran
        # on the real device and logged extra _grouped_mm calls. Answer from `failing` instead; tests of
        # the view probe override it.
        monkeypatch.setattr(moe_utils, "_transposed_view_probe", lambda d: d is not None and d.index not in failing)
        return probed
    return make


def cuda(i): return torch.device("cuda", i)


@pytest.mark.parametrize("count, failing, expected", [
    (1, (), True), (1, (0,), False), (2, (), True), (2, (1,), True), (3, (2,), True), (3, (0,), False),
])
def test_default_probe_answers_for_the_current_card(spoof, count, failing, expected):
    # Distinct models: the home probe runs on the home card only; another card is probed only once used.
    seen = spoof(count, failing)
    assert moe_utils._probe_torch_grouped_mm_supported() is expected
    assert moe_utils._check_torch_grouped_mm_supported() is expected
    assert moe_utils._check_torch_grouped_mm_supported(cuda(0)) is expected
    assert seen == [0]
    assert moe_utils._TORCH_GROUPED_MM_SUPPORTED is expected


def test_mixed_host_each_card_answers_for_itself(spoof):
    seen = spoof(2, failing = (1,), models = ["H100", "RTX4090"])
    assert moe_utils._check_torch_grouped_mm_supported(cuda(0)) is True   # unused 4090 never disables the H100
    assert seen == [0]   # an unused card is never probed
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
    assert moe_utils._check_torch_grouped_mm_supported(cuda(1)) is False   # unclassifiable: a model of its own
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
    assert seen == [0]   # the unused RTX4090 is never probed; its failure could not leak into the default answer


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


def test_rocm_crash_probe_one_child_per_model(spoof, monkeypatch):
    spoof(3, hip = "7.1.0")
    children = _record_children(monkeypatch)
    assert moe_utils._probe_torch_grouped_mm_supported() is True
    assert children == [("0", ("0", "0"))]
    moe_utils._warm_grouped_mm_devices([cuda(0), cuda(1), cuda(2)])   # experts on all three models
    assert children == [("0", ("0", "0")), ("0", ("1", "1")), ("0", ("2", "2"))]
    assert moe_utils._check_torch_grouped_mm_supported(cuda(2)) is True
    assert moe_utils._grouped_mm_survives_out_of_process(cuda(2)) is True
    assert len(children) == 3


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


def _no_probe_while_tracing(monkeypatch):
    def no_probe(_probe): raise AssertionError("probe ran while tracing")
    monkeypatch.setattr(moe_utils, "_run_probe_eagerly", no_probe)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)


@pytest.mark.parametrize("hip, crash_masks, failing", [("7.1.0", ("1",), ()), (None, (), (1,))])
def test_eager_warmup_lets_a_trace_on_the_other_model_read_its_own_answer(spoof, monkeypatch, hip, crash_masks, failing):
    # Mixed host: the second model cannot run grouped_mm (ROCm: the kernel crashes the process, unsloth#12391).
    seen = spoof(2, failing = failing, hip = hip, models = ["gfx942", "gfx1100"])
    children = _record_children(monkeypatch, crash_masks = crash_masks)
    views = []
    monkeypatch.setattr(moe_utils, "_transposed_view_probe", lambda d: views.append(d.index) or True)
    # Even where torch would fall back internally, a warmed card answers for itself.
    monkeypatch.setattr(moe_utils, "_TORCH_GROUPED_MM_FALLS_BACK", True, raising = False)
    # The eager warm-up with the experts split across both models (device_map).
    _warm_moe_backend(monkeypatch, [cuda(0), cuda(1)])
    assert moe_utils._TORCH_GROUPED_MM_SUPPORTED is True
    assert moe_utils._TRANSPOSED_VIEW_GROUPED_MM_SAFE is True
    if hip:
        assert [c[1] for c in children] == [("0", "0"), ("1", "1")]   # each model's crash probe, once
        assert seen == [0]                                             # the crashing kernel never ran in-process
    else:
        assert seen == [0, 1]
    assert views == [0]   # unsupported card: its view verdict is the copy, no probe
    _no_probe_while_tracing(monkeypatch)
    assert moe_utils._grouped_mm_supported_on(cuda(1)) is False
    assert moe_utils._transposed_view_safe_on(cuda(1)) is False
    assert moe_utils._grouped_mm_supported_on(cuda(0)) is True
    assert moe_utils._transposed_view_safe_on(cuda(0)) is True


def _warm_moe_backend(monkeypatch, expert_devices):
    # The real eager entry point on a model whose stacked expert weights sit on `expert_devices`.
    from unsloth_zoo.temporary_patches import moe_experts_interface as mei
    monkeypatch.setattr(moe_utils, "get_forward_moe_backend", lambda: moe_utils.forward_moe_backend)
    experts = [types.SimpleNamespace(_parameters = {"gate_up_proj": types.SimpleNamespace(device = d)}) for d in expert_devices]

    class Model:
        @classmethod
        def _can_set_experts_implementation(cls): return True
        def modules(self): return iter([types.SimpleNamespace(_parameters = {})] + experts)

    mei._warm_moe_backend(Model())


def _distributed(monkeypatch):
    # torchrun on one node: every rank sees every GPU, rank 0 uses cuda:0.
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "0")


def test_a_rank_never_probes_a_card_its_experts_do_not_use(spoof, monkeypatch):
    # Only the home card and the experts' cards are probed, never another rank's card.
    seen = spoof(2, failing = (1,), models = ["H100", "RTX4090"])
    _distributed(monkeypatch)
    views = []
    monkeypatch.setattr(moe_utils, "_transposed_view_probe", lambda d: views.append(d.index) or True)
    _warm_moe_backend(monkeypatch, [cuda(0)])
    assert seen == [0] and views == [0]
    assert moe_utils._TORCH_GROUPED_MM_SUPPORTED_BY_KIND == {} and moe_utils._TRANSPOSED_VIEW_SAFE_BY_KIND == {}


@pytest.mark.parametrize("hip", [None, "7.1.0"])
def test_a_rocm_rank_spawns_no_crash_probe_for_a_card_it_does_not_use(spoof, monkeypatch, hip):
    spoof(3, hip = hip, models = ["gfx942", "gfx1100", "gfx1100"])
    _distributed(monkeypatch)
    children = _record_children(monkeypatch)
    _warm_moe_backend(monkeypatch, [cuda(0)])
    assert children == ([] if hip is None else [("0", ("0", "0"))])


def test_a_device_map_split_across_two_models_warms_both(spoof, monkeypatch):
    seen = spoof(4, failing = (2,), models = ["H100", "H100", "RTX4090", "A100"])
    _warm_moe_backend(monkeypatch, [cuda(0), cuda(1), cuda(2)])   # cuda:3 holds no experts
    assert seen == [0, 2]   # one probe per model the experts use, the same-model cuda:1 shares the home's
    _no_probe_while_tracing(monkeypatch)
    assert [moe_utils._grouped_mm_supported_on(cuda(i)) for i in range(3)] == [True, True, False]
    assert ("cuda", 3) in moe_utils._GROUPED_MM_FOREIGN   # classified, never probed


def test_expert_devices_skip_meta_cpu_and_dense_weights():
    from unsloth_zoo.temporary_patches import moe_experts_interface as mei
    dense = torch.nn.Linear(2, 2)   # a dense MLP's down_proj is a submodule, not a stacked parameter
    stacks = [types.SimpleNamespace(_parameters = {"down_proj": types.SimpleNamespace(device = d)})
              for d in (torch.device("meta"), torch.device("cpu"), cuda(1), cuda(1), torch.device("xpu", 0))]
    model = types.SimpleNamespace(modules = lambda: iter([dense] + stacks))
    assert mei._expert_devices(model) == [cuda(1), torch.device("xpu", 0)]


def _never_warmed(spoof, monkeypatch, **kwargs):
    # Home probed and every card classified, no other model probed: experts moved after the warm-up.
    seen = spoof(**kwargs)
    monkeypatch.setattr(moe_utils, "_TORCH_GROUPED_MM_SUPPORTED", True)
    monkeypatch.setattr(moe_utils, "_TRANSPOSED_VIEW_GROUPED_MM_SAFE", True)
    moe_utils._grouped_mm_home()
    return seen


@pytest.mark.parametrize("falls_back, sm90_only, hip, expected", [
    (True, False, None, [True, True, True]),           # CUDA torch 2.9+: torch's fallback serves any arch
    (False, True, None, [False, True, False]),         # CUDA torch 2.8: sm90 only
    (False, False, "7.1.0", [False, False, False]),    # ROCm: no such promise, the manual loop
])
def test_tracing_a_never_warmed_card_takes_the_conservative_answers(spoof, monkeypatch, falls_back, sm90_only, hip, expected):
    seen = _never_warmed(spoof, monkeypatch, count = 4, hip = hip, models = ["H100", "A100", "H200", "B200"],
                         caps = [(9, 0), (8, 0), (9, 0), (10, 0)])
    monkeypatch.setattr(moe_utils, "_TORCH_GROUPED_MM_FALLS_BACK", falls_back, raising = False)
    monkeypatch.setattr(moe_utils, "_TORCH_GROUPED_MM_SM90_ONLY", sm90_only)
    _no_probe_while_tracing(monkeypatch)
    assert [moe_utils._grouped_mm_supported_on(cuda(i)) for i in (1, 2, 3)] == expected
    assert [moe_utils._transposed_view_safe_on(cuda(i)) for i in (1, 2, 3)] == [False] * 3   # the copy
    assert moe_utils._grouped_mm_supported_on(cuda(0)) is True
    assert moe_utils._transposed_view_safe_on(cuda(0)) is True
    assert seen == []


def test_tracing_a_card_of_another_accelerator_type_never_resolves_it(spoof, monkeypatch):
    # Mixed CUDA/XPU host whose XPU card no eager probe reached: the trace must not reach the
    # compiler-disabled resolver, and an XPU has no torch-side fallback, so the manual loop and the copy.
    _never_warmed(spoof, monkeypatch, count = 2, models = ["H100", "RTX4090"])
    monkeypatch.setattr(moe_utils, "_TORCH_GROUPED_MM_FALLS_BACK", True, raising = False)
    _no_probe_while_tracing(monkeypatch)
    xpu = types.SimpleNamespace(type = "xpu", index = 0)
    assert moe_utils._grouped_mm_supported_on(xpu) is False
    assert moe_utils._transposed_view_safe_on(xpu) is False
    assert ("xpu", 0) not in moe_utils._GROUPED_MM_FOREIGN


@pytest.mark.parametrize("warmed, index, expected", [
    (True, 0, 11.0),    # home card on a mixed host: its own verdicts
    (True, 1, 0.0),     # warmed RTX4090: its own "no" (support and view)
    (False, 1, 10.0),   # never warmed, CUDA 2.9+: torch's fallback, the contiguous copy
])
def test_a_fullgraph_compile_on_a_mixed_host_traces(spoof, monkeypatch, warmed, index, expected):
    if warmed:
        spoof(2, failing = (1,), models = ["H100", "RTX4090"])
        _warm_moe_backend(monkeypatch, [cuda(0), cuda(1)])
    else:
        _never_warmed(spoof, monkeypatch, count = 2, failing = (1,), models = ["H100", "RTX4090"])
    monkeypatch.setattr(moe_utils, "_TORCH_GROUPED_MM_FALLS_BACK", True, raising = False)
    # Not a real torch.device: tracing one would initialise CUDA on a runner without a GPU.
    card = types.SimpleNamespace(type = "cuda", index = index)

    def gate(x):
        view = moe_utils._transposed_view_safe_on(card)
        supported = moe_utils._grouped_mm_supported_on(card)
        return x + (1 if view else 0) + (10 if supported else 0)

    def no_probe(_probe): raise AssertionError("probe ran while tracing")
    monkeypatch.setattr(moe_utils, "_run_probe_eagerly", no_probe)
    # The cards are spoofed: keep Dynamo from saving a CUDA RNG state no runner without a GPU has.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    out = torch.compile(gate, fullgraph = True, backend = "eager")(torch.zeros(2))
    assert out.tolist() == [expected] * 2


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


class _OnCard(torch.Tensor):
    """CPU tensor reporting a CUDA device, so a device_map split is modelled without a GPU."""
    @property
    def device(self):
        return self._card


def _on(device, *shape):
    t = torch.Tensor._make_subclass(_OnCard, torch.zeros(*shape))
    t._card = device
    return t


class _FakeExperts(torch.nn.Module):
    """Stacked 3D expert weights on `device`."""
    def __init__(self, device):
        super().__init__()
        self.num_experts = 4
        self.gate_up_proj = _on(device, 4, 8, 4)
        self.down_proj = _on(device, 4, 4, 4)


class _FakeLinear(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.weight = _on(device, 4, 4)


def _split_model(*devices, others = ()):
    model = torch.nn.Module()
    model.num_experts = 4   # transformers wrappers (GptOssForCausalLM, OlmoeForCausalLM) expose it too
    model.layers = torch.nn.ModuleList(_FakeExperts(d) for d in devices)
    model.lm_head = torch.nn.ModuleList(_FakeLinear(d) for d in others)
    return model


def test_loader_warms_a_foreign_expert_card_before_compile(spoof, monkeypatch):
    # H100 home + A100 holding later layers: the loader resolves the A100 eagerly, so the compiled
    # forward's first A100 query traces under fullgraph=True instead of hitting the disabled probe.
    from unsloth_zoo.temporary_patches.moe_grouped_modulelist import auto_enable_grouped_moe
    seen = spoof(3, models = ["H100", "A100", "L4"])
    assert moe_utils._check_torch_grouped_mm_supported() is True
    auto_enable_grouped_moe(_split_model(cuda(0), cuda(1)))
    assert _fullgraph_gate(monkeypatch, cuda(1)).tolist() == [1.0, 1.0]
    assert 1 in seen and 2 not in seen  # the L4 holds no experts: never probed


@pytest.mark.parametrize("models", [["H100"], ["H100", "H100"]])
def test_loader_warm_adds_no_probe_on_one_gpu_or_one_model(spoof, models):
    from unsloth_zoo.temporary_patches.moe_grouped_modulelist import auto_enable_grouped_moe
    seen = spoof(len(models), models = models)
    assert moe_utils._check_torch_grouped_mm_supported() is True
    auto_enable_grouped_moe(_split_model(*(cuda(i) for i in range(len(models)))))
    assert seen == [0]


def test_loader_warm_skips_cards_holding_only_non_expert_weights(spoof, monkeypatch):
    # A wrapper exposing num_experts with its LM head on an A100: the experts all sit on the H100 home,
    # so the A100 must not be probed (nor the L4 that holds nothing).
    from unsloth_zoo.temporary_patches.moe_grouped_modulelist import auto_enable_grouped_moe
    from unsloth_zoo.temporary_patches.moe_experts_interface import _expert_devices
    seen = spoof(3, models = ["H100", "A100", "L4"])
    assert moe_utils._check_torch_grouped_mm_supported() is True
    model = _split_model(cuda(0), cuda(0), others = (cuda(1),))
    assert _expert_devices(model) == [cuda(0)]
    auto_enable_grouped_moe(model)
    assert seen == [0]


def test_loader_warms_an_xpu_card_beside_a_single_cuda_home(spoof, monkeypatch):
    # One CUDA card plus one XPU card: each backend counts 1 device, yet the XPU is foreign to the CUDA
    # home and its first traced query would reach the compiler-disabled resolver.
    from unsloth_zoo.temporary_patches.moe_grouped_modulelist import auto_enable_grouped_moe
    seen = spoof(1, models = ["H100"])
    real_xpu = getattr(torch, "xpu", None)
    monkeypatch.setattr(torch, "xpu", types.SimpleNamespace(
        is_available = lambda: True,
        device_count = lambda: 1,
        current_device = lambda: 0,
        get_device_properties = lambda i: types.SimpleNamespace(name = "Arc B580", major = None, minor = None),
    ), raising = False)
    # Warm only this module: a compiled-cache copy (if one exists here) would probe once more for itself.
    monkeypatch.setattr(moe_utils, "get_forward_moe_backend", lambda: moe_utils.forward_native_grouped_mm)
    xpu = torch.device("xpu", 0)
    assert moe_utils._check_torch_grouped_mm_supported() is True
    auto_enable_grouped_moe(_split_model(cuda(0), xpu))
    assert seen == [0, 0]   # the CUDA home, then the XPU card
    kind = moe_utils._grouped_mm_foreign_kind(xpu)
    assert kind is not None and moe_utils._TORCH_GROUPED_MM_SUPPORTED_BY_KIND.get(kind) is True
    monkeypatch.setattr(torch, "xpu", real_xpu, raising = False)   # Dynamo itself reads the real backend
    assert _fullgraph_gate(monkeypatch, xpu).tolist() == [1.0, 1.0]
