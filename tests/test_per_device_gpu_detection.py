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

"""Device gates must read every visible GPU (or the current one), never device 0 alone.

CPU-only: torch.cuda is spoofed per test. Covers bf16 for GC offload buffers, the
pre-Turing compile switch, tiled-MLP free memory, the per-device AMD arch / aiter
caches (set_device switch), the rocminfo mask fallback and the llama.cpp asset target.
"""
import ast
import importlib.machinery
import inspect
import re
import sys
import types
from unittest import mock

import pytest

torch = pytest.importorskip("torch")

from unsloth_zoo import device_type
from unsloth_zoo import gradient_checkpointing as gc
from unsloth_zoo import llama_cpp
from unsloth_zoo import tiled_mlp

_MISSING = object()
MIXED = [[(8, 0), (7, 5)], [(7, 5), (8, 0)]]


@pytest.fixture
def cuda_caps(monkeypatch):
    """Spoof N visible CUDA devices with the given capabilities; current device settable."""
    state = {"current": 0}

    def apply(caps, current = 0):
        state["current"] = current
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: len(caps))
        monkeypatch.setattr(torch.cuda, "current_device", lambda: state["current"])
        monkeypatch.setattr(
            torch.cuda, "get_device_capability",
            lambda device = None: caps[state["current"] if device is None else device],
        )
        return state

    return apply


# C1: bf16 offload buffers -------------------------------------------------------------

@pytest.fixture
def gc_dtype(monkeypatch, cuda_caps):
    tree = ast.parse(inspect.getsource(gc.initialize_unsloth_gradient_checkpointing).lstrip())
    names = {n for node in ast.walk(tree) if isinstance(node, ast.Global) for n in node.names}
    saved = {name: getattr(gc, name, _MISSING) for name in names}
    real_empty = torch.empty

    def empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        kwargs["device"] = "cpu"
        return real_empty(*args, **kwargs)

    def run(caps, current = 0):
        cuda_caps(caps, current)
        # The visible-device list comes from the spoof; the GC loop itself sees no device.
        listed = [tuple(c) for c in caps]
        monkeypatch.setattr(device_type, "_visible_cuda_capabilities", lambda: listed, raising = False)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
        # Not the env var: _double_buffer_disabled is functools.cached and would leak.
        monkeypatch.setattr(gc, "_double_buffer_disabled", lambda: True)
        monkeypatch.setattr(torch, "empty", empty)
        monkeypatch.setattr(gc, "DEVICE_TYPE", "cuda")
        monkeypatch.setattr(gc, "DEVICE_TYPE_TORCH", "cuda")
        gc.CPU_BUFFERS = []
        gc.initialize_unsloth_gradient_checkpointing()
        return gc.CPU_BUFFERS[0].dtype

    yield run
    for name, value in saved.items():
        if value is _MISSING:
            if hasattr(gc, name): delattr(gc, name)
        else:
            setattr(gc, name, value)


@pytest.mark.parametrize("caps", MIXED)
@pytest.mark.parametrize("current", [0, 1])
def test_gc_bf16_needs_every_visible_device(gc_dtype, caps, current):
    assert gc_dtype(caps, current) is torch.float16


def test_gc_bf16_homogeneous_ampere_unchanged(gc_dtype):
    assert gc_dtype([(8, 0), (8, 0)]) is torch.bfloat16


def test_visible_cuda_capabilities_lists_every_device(cuda_caps):
    cuda_caps([(8, 6), (6, 1), (9, 0)], current = 2)
    assert device_type._visible_cuda_capabilities() == [(8, 6), (6, 1), (9, 0)]


@pytest.mark.parametrize("current", [0, 2])
def test_visible_cuda_capabilities_partial_scan_keeps_readable_cards(monkeypatch, cuda_caps, current):
    # Matches core: the readable cards (current device included) are judged, so a known old card counts.
    expected = [(8, 0), (7, 5)]
    caps = [(8, 0), (8, 0), (7, 5)]
    cuda_caps(caps, current = current)

    def flaky(device = None):
        if device == 1:
            raise RuntimeError("unreadable")
        return caps[current if device is None else device]

    monkeypatch.setattr(torch.cuda, "get_device_capability", flaky)
    assert device_type._visible_cuda_capabilities() == expected


# C2: pre-Turing LM-head fusion switch --------------------------------------------------------

def _old_cuda_arch_version(caps, current, cuda_caps):
    """Run compiler.py's module-level capability block against spoofed devices."""
    import unsloth_zoo.compiler as compiler
    tree = ast.parse(inspect.getsource(compiler))
    block = next(
        node for node in tree.body
        if isinstance(node, ast.If) and "OLD_CUDA_ARCH_VERSION" in ast.unparse(node)
        and "get_device_capability" in ast.unparse(node)
    )
    cuda_caps(caps, current)
    namespace = {"__name__": "unsloth_zoo.compiler", "__package__": "unsloth_zoo",
                 "torch": torch, "DEVICE_TYPE": "cuda"}
    exec(compile(ast.Module(body = [block], type_ignores = []), "compiler_block", "exec"), namespace)
    return namespace["OLD_CUDA_ARCH_VERSION"]


def _old_cuda_arch_indices(caps, current, cuda_caps):
    import unsloth_zoo.compiler as compiler
    tree = ast.parse(inspect.getsource(compiler))
    block = next(
        node for node in tree.body
        if isinstance(node, ast.If) and "OLD_CUDA_ARCH_INDICES" in ast.unparse(node)
        and "get_device_capability" in ast.unparse(node)
    )
    cuda_caps(caps, current)
    namespace = {"__name__": "unsloth_zoo.compiler", "__package__": "unsloth_zoo",
                 "torch": torch, "DEVICE_TYPE": "cuda"}
    exec(compile(ast.Module(body = [block], type_ignores = []), "compiler_block", "exec"), namespace)
    return namespace["OLD_CUDA_ARCH_INDICES"]


@pytest.mark.parametrize("current", [0, 1])
@pytest.mark.parametrize("caps", [[(8, 0), (6, 1)], [(6, 1), (8, 0)], [(8, 9), (7, 0)]])
def test_lm_head_fusion_switch_follows_the_current_device(cuda_caps, caps, current):
    # An idle older card beside the current one must not turn the LM-head fusion off.
    major, minor = caps[current]
    assert _old_cuda_arch_version(caps, current, cuda_caps) is ((major <= 7) and (minor < 5))


def test_compile_switch_homogeneous_unchanged(cuda_caps):
    assert _old_cuda_arch_version([(8, 0), (8, 0)], 1, cuda_caps) is False
    assert _old_cuda_arch_version([(7, 0)], 0, cuda_caps) is True


def test_lm_head_fusion_records_every_old_card(cuda_caps):
    # sm89 current beside an sm61 card: the rewrite installs, and the sm61 index is recorded
    # so a forward whose LM head lands there skips Inductor and CCE.
    assert _old_cuda_arch_version([(8, 9), (6, 1)], 0, cuda_caps) is False
    assert _old_cuda_arch_indices([(8, 9), (6, 1)], 0, cuda_caps) == frozenset({1})
    assert _old_cuda_arch_indices([(8, 9)], 0, cuda_caps) == frozenset()
    assert _old_cuda_arch_indices([(8, 0), (8, 0)], 1, cuda_caps) == frozenset()


def test_an_unreadable_card_counts_as_old(cuda_caps, monkeypatch):
    cuda_caps([(8, 9), (8, 9)], 0)

    def capability(device = None):
        if device == 1:
            raise RuntimeError("unreadable")
        return (8, 9)

    import unsloth_zoo.compiler as compiler
    tree = ast.parse(inspect.getsource(compiler))
    block = next(
        node for node in tree.body
        if isinstance(node, ast.If) and "OLD_CUDA_ARCH_INDICES" in ast.unparse(node)
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    namespace = {"__name__": "unsloth_zoo.compiler", "__package__": "unsloth_zoo",
                 "torch": torch, "DEVICE_TYPE": "cuda"}
    exec(compile(ast.Module(body = [block], type_ignores = []), "compiler_block", "exec"), namespace)
    assert namespace["OLD_CUDA_ARCH_INDICES"] == frozenset({1})


def test_lm_head_on_old_cuda_arch_reads_the_weight_device(monkeypatch):
    import unsloth_zoo.compiler as compiler
    on = lambda device: types.SimpleNamespace(device = device)
    monkeypatch.setattr(compiler, "OLD_CUDA_ARCH_INDICES", frozenset({1}))
    assert compiler.lm_head_on_old_cuda_arch(on(torch.device("cuda", 1))) is True
    assert compiler.lm_head_on_old_cuda_arch(on(torch.device("cuda", 0))) is False
    assert compiler.lm_head_on_old_cuda_arch(on(torch.device("cpu"))) is False
    assert compiler.lm_head_on_old_cuda_arch(None) is False
    # No old card visible (one-GPU and homogeneous hosts): never true.
    monkeypatch.setattr(compiler, "OLD_CUDA_ARCH_INDICES", frozenset())
    assert compiler.lm_head_on_old_cuda_arch(on(torch.device("cuda", 1))) is False


def test_the_fused_lm_head_templates_consult_the_head_device():
    import unsloth_zoo.compiler as compiler
    templates = [compiler.cross_entropy_replacement_1, compiler.cross_entropy_replacement_2,
                 compiler.cross_entropy_replacement_3]
    for template in templates:
        calls = template.count("loss = unsloth_fused_ce_loss(")
        assert calls == template.count(
            "torch_compile        = not UNSLOTH_COMPILE_DISABLE and not lm_head_on_old_cuda_arch(lm_head_weight),"
        )
        assert template.count("(UNSLOTH_ENABLE_CCE and HAS_CUT_CROSS_ENTROPY)") == template.count(
            "(UNSLOTH_ENABLE_CCE and HAS_CUT_CROSS_ENTROPY) and not lm_head_on_old_cuda_arch(self.lm_head.weight)"
        )
        assert calls >= 1
    assert "from unsloth_zoo.compiler import lm_head_on_old_cuda_arch" in compiler._disabled_sdpa_code



def _adapter_dispatch(monkeypatch, device, **kwargs):
    """torch_compile the shared adapter hands the CE kernel for an LM head on `device`."""
    from unsloth_zoo.fused_losses import forward_adapter as adapter
    from unsloth_zoo.fused_losses import cross_entropy_loss as ce
    seen = []

    def spy(**call):
        bound = inspect.signature(ce.unsloth_fused_ce_loss).bind(**call)
        bound.apply_defaults()
        seen.append(bound.arguments["torch_compile"])
        return 1.0

    monkeypatch.setattr(adapter, "unsloth_fused_ce_loss", spy)
    monkeypatch.setattr(adapter, "unsloth_fsdp_unshard", lambda head: None)
    head = types.SimpleNamespace(weight = types.SimpleNamespace(device = device))
    adapter.unsloth_fused_lm_head_loss(None, head, None, **kwargs)
    return seen[0]


def test_the_shared_adapter_skips_inductor_for_a_head_on_an_old_card(monkeypatch):
    # AST-rewritten forwards (GPT-Neo, CLVP, ...) and the import hook call the adapter, not the
    # regex templates: it must ask the same per-head question, or a head a device_map put on a
    # Pascal card beside the current A100 still asks Inductor for the chunked CE.
    import unsloth_zoo.compiler as compiler
    monkeypatch.setattr(compiler, "OLD_CUDA_ARCH_INDICES", frozenset({1}))
    assert _adapter_dispatch(monkeypatch, torch.device("cuda", 1)) is False
    assert _adapter_dispatch(monkeypatch, torch.device("cuda", 1), torch_compile = True) is False
    assert _adapter_dispatch(monkeypatch, torch.device("cuda", 0)) is True
    # An explicit disable is never turned back on.
    assert _adapter_dispatch(monkeypatch, torch.device("cuda", 0), torch_compile = False) is False
    # One-GPU and homogeneous hosts: no old card, the default is untouched.
    monkeypatch.setattr(compiler, "OLD_CUDA_ARCH_INDICES", frozenset())
    assert _adapter_dispatch(monkeypatch, torch.device("cuda", 1)) is True


def test_ast_rewritten_lm_head_forwards_go_through_the_guarded_adapter():
    # The route the regex templates cannot take: real HF forwards whose fused call is spliced
    # by the AST rewriter. It must call the adapter (which holds the per-head gate) and not pass
    # a torch_compile of its own that would bypass it.
    import importlib
    import textwrap
    import unsloth_zoo.compiler as compiler
    rewritten = []
    for name, cls_name in [("opt", "OPTForCausalLM"), ("biogpt", "BioGptForCausalLM"),
                           ("xglm", "XGLMForCausalLM"), ("clvp", "ClvpForCausalLM"),
                           ("gpt_neo", "GPTNeoForCausalLM")]:
        try:
            module = importlib.import_module(f"transformers.models.{name}.modeling_{name}")
            cls = getattr(module, cls_name)
            source = textwrap.dedent(inspect.getsource(cls.forward))
        except Exception:
            continue
        new, route, _ = compiler.fused_lm_head_forward(cls_name, cls, cls.__module__, source)
        if route == "ast":
            rewritten.append(cls_name)
            assert "unsloth_fused_lm_head_loss(" in new
            assert "torch_compile" not in new
    if not rewritten:
        pytest.skip("no checked HF forward takes the AST route on this transformers")


# Z1: tiled-MLP free memory ------------------------------------------------------------

def _fake_cuda():
    fake = mock.MagicMock()
    fake.is_available.return_value = True
    fake.mem_get_info.return_value = (8 * 1024 ** 3, 16 * 1024 ** 3)
    return fake


def test_tiled_mlp_reads_current_device_not_gpu0():
    fake = _fake_cuda()
    with mock.patch.object(tiled_mlp, "DEVICE_TYPE", "cuda"), mock.patch.object(torch, "cuda", fake):
        assert tiled_mlp._default_target_gb() == 4.0
    assert fake.mem_get_info.call_args.args in ((), (None,)), fake.mem_get_info.call_args


def test_tiled_mlp_reads_the_activation_device():
    fake = _fake_cuda()
    device = torch.device("cuda", 1)
    with mock.patch.object(tiled_mlp, "DEVICE_TYPE", "cuda"), mock.patch.object(torch, "cuda", fake):
        tiled_mlp._default_target_gb(device)
    assert fake.mem_get_info.call_args.args in ((device,), (1,))  # the device or its index


def test_fused_ce_budget_reads_the_head_device_per_device():
    from unsloth_zoo.fused_losses import cross_entropy_loss as ce
    fake = _fake_cuda()
    free = {0: 40 * 1024 ** 3, 1: 2 * 1024 ** 3}
    fake.mem_get_info.side_effect = lambda device = None: (
        free[device if isinstance(device, int) else (getattr(device, "index", 0) or 0)], 0
    )
    ce._get_chunk_multiplier.cache_clear()
    try:
        with mock.patch.object(tiled_mlp, "DEVICE_TYPE", "cuda"), mock.patch.object(torch, "cuda", fake):
            big = ce.get_chunk_size(1, 16_384, 151_936, device = torch.device("cuda", 0))
            small = ce.get_chunk_size(1, 16_384, 151_936, device = torch.device("cuda", 1))
    finally:
        ce._get_chunk_multiplier.cache_clear()
    # The 2 GiB head GPU needs more chunks than the 40 GiB one, not its cached answer.
    assert small > big, (small, big)


# Z3: per-device AMD arch / aiter caches -----------------------------------------------

_AMD_CACHED = ("_detect_gfx_arch", "get_amd_attention_implementation", "get_amd_flash_attn_func")


def _clear_amd_caches():
    for name in _AMD_CACHED:
        getattr(device_type, name).cache_clear()


def _multi_device_caches(monkeypatch):
    # Import on a host without two HIP devices keeps the plain cache; rebuild the per-device one.
    for name in _AMD_CACHED:
        fn = inspect.unwrap(getattr(device_type, name))
        monkeypatch.setattr(device_type, name, device_type._per_device_cache(fn, multi_device = True))


@pytest.fixture
def amd_host(monkeypatch):
    """Two HIP devices: gfx942 at 0, gfx1100 at 1, ROCm 7, aiter importable."""
    state = {"current": 0}
    archs = ["gfx942:sramecc+:xnack-", "gfx1100"]
    monkeypatch.setattr(device_type, "is_hip", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: state["current"])
    monkeypatch.setattr(torch.cuda, "set_device", lambda i: state.__setitem__("current", i), raising = False)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda i: types.SimpleNamespace(gcnArchName = archs[i]))
    monkeypatch.setattr(torch.version, "hip", "7.0.1", raising = False)
    aiter = types.ModuleType("aiter")
    aiter.__spec__ = importlib.machinery.ModuleSpec("aiter", None)
    aiter.flash_attn_func = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "aiter", aiter)
    _multi_device_caches(monkeypatch)
    _clear_amd_caches()
    yield aiter
    _clear_amd_caches()


def test_gfx_arch_follows_set_device(amd_host):
    assert device_type._detect_gfx_arch() == "gfx942"
    torch.cuda.set_device(1)
    assert device_type._detect_gfx_arch() == "gfx1100"
    torch.cuda.set_device(0)
    assert device_type._detect_gfx_arch() == "gfx942"


def test_aiter_choice_follows_set_device(amd_host):
    assert device_type.get_amd_attention_implementation() == "amd_aiter"
    assert device_type.get_amd_flash_attn_func() is amd_host.flash_attn_func
    torch.cuda.set_device(1)
    assert device_type.get_amd_attention_implementation() == "sdpa"
    assert device_type.get_amd_flash_attn_func() is None


def test_single_hip_device_runs_aiter_discovery_once(monkeypatch):
    # One HIP device keeps the plain cache: the no-argument setup gate and the flash-attn lookup
    # must share one get_amd_attention_implementation entry.
    monkeypatch.setattr(device_type, "is_hip", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda i: types.SimpleNamespace(gcnArchName = "gfx942:sramecc+:xnack-"))
    monkeypatch.setattr(torch.version, "hip", "7.0.1", raising = False)
    aiter = types.ModuleType("aiter")
    aiter.__spec__ = importlib.machinery.ModuleSpec("aiter", None)
    aiter.flash_attn_func = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "aiter", aiter)
    for name in _AMD_CACHED:
        fn = inspect.unwrap(getattr(device_type, name))
        monkeypatch.setattr(device_type, name, device_type._per_device_cache(fn))
    _clear_amd_caches()
    assert device_type.get_amd_attention_implementation() == "amd_aiter"
    assert device_type.get_amd_flash_attn_func() is aiter.flash_attn_func
    assert device_type.get_amd_attention_implementation.cache_info().currsize == 1
    _clear_amd_caches()


def test_plain_cache_by_default():
    calls = []

    def probe(device = None):
        calls.append(device)
        return "gfx1151"

    cached = device_type._per_device_cache(probe)
    assert hasattr(cached, "cache_info") and inspect.unwrap(cached) is probe
    assert cached() == cached() == "gfx1151"
    assert calls == [None]


@pytest.mark.parametrize("dtype,count,multi", [("hip", 2, True), ("hip", 1, False), ("cuda", 2, False), ("cpu", 0, False)])
def test_per_device_caches_only_with_two_hip_devices(monkeypatch, dtype, count, multi):
    for name in _AMD_CACHED:
        monkeypatch.setattr(device_type, name, getattr(device_type, name))  # restored after the test
    monkeypatch.setattr(torch.cuda, "device_count", mock.Mock(side_effect = AssertionError("no probe")))
    monkeypatch.setattr(device_type, "AMD_PER_DEVICE_CACHES", False)
    assert device_type._use_per_device_caches(dtype, count) is multi
    assert device_type.AMD_PER_DEVICE_CACHES is multi
    for name in _AMD_CACHED:
        assert hasattr(getattr(device_type, name), "cache_info")
        assert (getattr(device_type, name).__class__.__name__ == "function") is multi


def test_plain_cache_resolves_a_device_argument(monkeypatch):
    monkeypatch.setattr(device_type, "is_hip", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda i: types.SimpleNamespace(gcnArchName = ["gfx1151"][i]))
    fn = device_type._per_device_cache(inspect.unwrap(device_type._detect_gfx_arch), multi_device = False)
    assert fn(torch.device("cuda", 0)) == fn("cuda:0") == fn(0) == "gfx1151"


def test_amd_caches_take_an_explicit_device(amd_host):
    assert device_type._detect_gfx_arch(1) == "gfx1100"
    assert device_type.get_amd_attention_implementation("cuda:1") == "sdpa"
    assert device_type.get_amd_attention_implementation(torch.device("cuda", 0)) == "amd_aiter"
    torch.cuda.set_device(1)
    # An index-less device means the current one, never a stale None-keyed entry.
    assert device_type._detect_gfx_arch("cuda") == "gfx1100"


def test_amd_caches_fall_back_when_current_device_raises(amd_host, monkeypatch):
    # A runtime that cannot answer (e.g. a forked worker) must reach the probes' own fallbacks.
    def boom():
        raise RuntimeError("Cannot re-initialize CUDA in forked subprocess")
    monkeypatch.setattr(torch.cuda, "current_device", boom)
    assert device_type.get_amd_attention_implementation() in ("sdpa", "amd_aiter")


@pytest.mark.parametrize("per_device", [True, False])
def test_registered_aiter_attention_asks_for_the_query_device(monkeypatch, per_device):
    from unsloth_zoo.temporary_patches import amd_aiter
    import transformers.integrations.sdpa_attention as sdpa
    seen = []
    def getter(*args):
        seen.append(args)
        return None
    monkeypatch.setattr(device_type, "get_amd_flash_attn_func", getter)
    monkeypatch.setattr(device_type, "AMD_PER_DEVICE_CACHES", per_device)
    monkeypatch.setattr(device_type, "is_hip", lambda: True)
    monkeypatch.setattr(sdpa, "sdpa_attention_forward", lambda *a, **k: "sdpa")
    query = types.SimpleNamespace(is_cuda = True, device = torch.device("cuda", 1))
    assert amd_aiter.aiter_attention_forward(None, query, None, None, None) == "sdpa"
    # One HIP device: the no-argument call, the cache entry the setup gate already filled.
    assert seen == ([(torch.device("cuda", 1),)] if per_device else [()])


def test_compiled_aiter_rewrite_asks_for_the_query_device(monkeypatch):
    from unsloth_zoo import compiler
    monkeypatch.setattr(device_type, "get_amd_attention_implementation", lambda device = None: "amd_aiter")
    source = (
        "def forward(self, q, k, v):\n"
        "    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)\n"
        "    return out\n"
    )
    rewritten = compiler.replace_sdpa_with_amd_aiter(source)
    assert compiler._AITER_MARKER in rewritten
    assert re.search(r"get_amd_flash_attn_func\(\w+\.device\)", rewritten), rewritten
    assert re.search(r"AMD_PER_DEVICE_CACHES else \w+\.get_amd_flash_attn_func\(\)", rewritten), rewritten
    compile(rewritten, "<rewritten>", "exec")


# Z2: rocminfo fallback honours HIP masks ----------------------------------------------

ROCMINFO = """
*******
Agent 1
*******
  Name:                    AMD Ryzen AI Max+ 395
  Device Type:             CPU
*******
Agent 2
*******
  Name:                    gfx1151
  Device Type:             GPU
  ISA 1: amdgcn-amd-amdhsa--gfx1151
*******
Agent 3
*******
  Name:                    gfx1100
  Device Type:             GPU
  ISA 1: amdgcn-amd-amdhsa--gfx1100
"""


@pytest.mark.parametrize("env, expected", [
    ({}, "gfx1151"),
    ({"HIP_VISIBLE_DEVICES": "1"}, "gfx1100"),
    ({"CUDA_VISIBLE_DEVICES": "1,0"}, "gfx1100"),
    ({"HIP_VISIBLE_DEVICES": ""}, None),
    ({"HIP_VISIBLE_DEVICES": "-1"}, None),
])
def test_rocminfo_fallback_honours_mask(monkeypatch, env, expected):
    for key in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(key, raising = False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    import subprocess
    def run(command, **kwargs):
        out = ROCMINFO if command[0] == "rocminfo" else ""
        return types.SimpleNamespace(stdout = out, returncode = 0)
    monkeypatch.setattr(subprocess, "run", run)
    device_type._detect_gfx_arch.cache_clear()
    try:
        assert device_type._detect_gfx_arch() == expected
    finally:
        device_type._detect_gfx_arch.cache_clear()


# Z4: llama.cpp asset target -----------------------------------------------------------

def _fake_torch(caps = None, archs = None, current = 0):
    cuda = types.SimpleNamespace(
        is_available = lambda: True,
        device_count = lambda: len(caps or archs),
        current_device = lambda: current,
        get_device_capability = lambda i = None: caps[current if i is None else i],
        get_device_properties = lambda i: types.SimpleNamespace(gcnArchName = archs[i]),
    )
    version = types.SimpleNamespace(hip = "6.4" if archs else None, cuda = None if archs else "12.8")
    return types.SimpleNamespace(cuda = cuda, version = version)


@pytest.mark.parametrize("caps", MIXED)
@pytest.mark.parametrize("current", [0, 1])
def test_cuda_target_covers_every_visible_sm(monkeypatch, caps, current):
    monkeypatch.setattr(llama_cpp, "torch", _fake_torch(caps = caps, current = current))
    assert llama_cpp._detect_gpu_target() == ("cuda", 75, "cuda12", (75, 80))


def test_cuda_target_homogeneous_unchanged(monkeypatch):
    monkeypatch.setattr(llama_cpp, "torch", _fake_torch(caps = [(8, 6), (8, 6)], current = 1))
    assert llama_cpp._detect_gpu_target() == ("cuda", 86, "cuda12")


def test_rocm_target_mixed_family_builds_from_source(monkeypatch):
    monkeypatch.setattr(llama_cpp, "IS_WINDOWS", False)  # Windows keeps the prebuilt (own test)
    monkeypatch.setattr(llama_cpp, "torch", _fake_torch(archs = ["gfx1100", "gfx1201"]))
    assert llama_cpp._detect_gpu_target() is None


def test_rocm_target_same_family_uses_current_arch(monkeypatch):
    monkeypatch.setattr(llama_cpp, "torch", _fake_torch(archs = ["gfx1100", "gfx1101:xnack-"], current = 1))
    assert llama_cpp._detect_gpu_target() == ("rocm", "gfx1101")


@pytest.mark.parametrize("caps,expected", [
    ([(8, 9), (8, 9), (8, 9)], ("cuda", 89, "cuda12")),
    ([(8, 9), (8, 9), (7, 5)], ("cuda", 75, "cuda12", (75, 89))),  # a readable old card still counts
])
def test_cuda_target_skips_an_unreadable_card(monkeypatch, caps, expected):
    fake = _fake_torch(caps = caps)

    def flaky(i = None):
        if i == 1:
            raise RuntimeError("unreadable")
        return caps[0 if i is None else i]

    fake.cuda.get_device_capability = flaky
    monkeypatch.setattr(llama_cpp, "torch", fake)
    assert llama_cpp._detect_gpu_target() == expected


@pytest.mark.parametrize("archs", [["gfx1100", "gfx1036"], ["gfx1201", "gfx1103:xnack-"]])
@pytest.mark.parametrize("current", [0, 1])
def test_rocm_target_ignores_a_shadowing_igpu(monkeypatch, archs, current):
    monkeypatch.setattr(llama_cpp, "torch", _fake_torch(archs = archs, current = current))
    assert llama_cpp._detect_gpu_target() == ("rocm", archs[0])


def test_rocm_target_lone_igpu_keeps_its_arch(monkeypatch):
    monkeypatch.setattr(llama_cpp, "torch", _fake_torch(archs = ["gfx1103"]))
    assert llama_cpp._detect_gpu_target() == ("rocm", "gfx1103")


def test_rocm_target_mixed_family_keeps_the_prebuilt_on_windows(monkeypatch):
    # Windows has no HIP source build, so a mixed host keeps the current card's bundle.
    monkeypatch.setattr(llama_cpp, "IS_WINDOWS", True)
    monkeypatch.setattr(llama_cpp, "torch", _fake_torch(archs = ["gfx1100", "gfx1201"]))
    assert llama_cpp._detect_gpu_target() == ("rocm", "gfx1100")


def test_rocm_target_skips_an_unreadable_card(monkeypatch):
    fake = _fake_torch(archs = ["gfx1100", "gfx1101", "gfx1100"])
    real = fake.cuda.get_device_properties

    def flaky(i):
        if i == 1:
            raise RuntimeError("unreadable")
        return real(i)

    fake.cuda.get_device_properties = flaky
    monkeypatch.setattr(llama_cpp, "torch", fake)
    assert llama_cpp._detect_gpu_target() == ("rocm", "gfx1100")


@pytest.mark.parametrize("archs,expected", [(["gfx1100", "gfx1101"], "gfx1101"), (["gfx1100", "gfx1036"], "gfx1036")])
def test_rocm_target_unreadable_current_card_uses_a_readable_one(monkeypatch, archs, expected):
    fake = _fake_torch(archs = archs)
    real = fake.cuda.get_device_properties

    def flaky(i):
        if i == 0:
            raise RuntimeError("unreadable")
        return real(i)

    fake.cuda.get_device_properties = flaky
    monkeypatch.setattr(llama_cpp, "torch", fake)
    assert llama_cpp._detect_gpu_target() == ("rocm", expected)


def _manifest():
    def art(name, sms, coverage = "narrow"):
        return {"asset_name": name, "install_kind": "linux-cuda", "runtime_line": "cuda12",
                "supported_sms": [str(s) for s in sms], "coverage_class": coverage,
                "min_sm": min(sms), "max_sm": max(sms), "rank": 0}
    artifacts = [art("sm80-only.tar.gz", [80, 86]), art("sm75-only.tar.gz", [75]),
                 art("turing-ampere.tar.gz", [75, 80, 86, 89]),
                 art("portable.tar.gz", [75, 80, 86, 89, 90, 100, 120], "portable")]
    return {"artifacts": artifacts}, {a["asset_name"]: f"https://x/{a['asset_name']}" for a in artifacts}


def test_asset_ranking_covers_every_visible_sm(monkeypatch):
    monkeypatch.setattr(llama_cpp.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(llama_cpp.platform, "system", lambda: "Linux")
    manifest, assets = _manifest()
    names = [n for n, _ in llama_cpp._select_gpu_assets(
        "b1", assets, manifest, target = ("cuda", 75, "cuda12", (75, 80)))]
    assert names == ["turing-ampere.tar.gz", "portable.tar.gz"]
    # Single device: narrowest covering bundle first, unchanged.
    names = [n for n, _ in llama_cpp._select_gpu_assets("b1", assets, manifest, target = ("cuda", 75, "cuda12"))]
    assert names == ["sm75-only.tar.gz", "portable.tar.gz"]


def test_cmake_targets_skip_an_unreadable_card(monkeypatch):
    fake = _fake_torch(archs = ["gfx1100", "gfx1030", "gfx1201"])
    real = fake.cuda.get_device_properties

    def flaky(i):
        if i == 1:
            raise RuntimeError("unreadable")
        return real(i)

    fake.cuda.get_device_properties = flaky
    monkeypatch.setattr(llama_cpp, "torch", fake)
    monkeypatch.setattr(llama_cpp, "IS_WINDOWS", False)
    flags = llama_cpp._gpu_cmake_flags("ON")
    assert "-DGPU_TARGETS=gfx1100;gfx1201" in flags


def test_compiled_aiter_rewrite_avoids_a_local_named_like_its_module_alias(monkeypatch):
    from unsloth_zoo import compiler
    monkeypatch.setattr(device_type, "get_amd_attention_implementation", lambda device = None: "amd_aiter")
    source = (
        "def forward(self, q, k, v):\n"
        "    _aiter_device_type = 1\n"
        "    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)\n"
        "    return out + _aiter_device_type\n"
    )
    rewritten = compiler.replace_sdpa_with_amd_aiter(source)
    assert "import unsloth_zoo.device_type as _aiter_device_type\n" not in rewritten, rewritten
    assert re.search(r"import unsloth_zoo\.device_type as _aiter_device_type_\d+", rewritten), rewritten
