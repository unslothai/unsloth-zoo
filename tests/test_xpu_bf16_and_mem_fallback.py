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

"""XPU mem fallback, XPU native bf16 and Apple MLX bf16, on CPU with spoofed torch.xpu / mlx.core."""
import ast
import inspect
import sys
import types

import pytest

torch = pytest.importorskip("torch")

from unsloth_zoo import device_type as dt

_GB = 1024 ** 3
_MISSING = object()


_PROVEN_NO_BF16 = "proven-no-bf16"


def _props(flag = _MISSING, total = 16 * _GB):
    props = types.SimpleNamespace(total_memory = total)
    if flag == _PROVEN_NO_BF16:
        props.has_bfloat16_conversions = False
        props.has_subgroup_2d_block_io = True
    elif flag is not _MISSING:
        props.has_bfloat16_conversions = flag
    return props


def _fake_xpu(flags = (True,), mem_get_info = None, is_bf16 = True, props_raise = False, current = None):
    devices = [_props(flag) for flag in flags]

    def get_device_properties(index = None):
        if props_raise:
            raise RuntimeError("unreadable")
        return devices[0 if index is None else index]

    xpu = types.SimpleNamespace(
        is_available = lambda: True,
        device_count = lambda: len(devices),
        current_device = lambda: len(devices) - 1 if current is None else current,
        get_device_properties = get_device_properties,
        memory_reserved = lambda index = None: 3 * _GB,
        mem_get_info = mem_get_info,
    )
    if is_bf16 is not None:
        xpu.is_bf16_supported = lambda including_emulation = True: is_bf16
    return xpu



not_on_macos = pytest.mark.skipif(sys.platform == "darwin", reason = "XPU / CUDA paths do not run on macOS")

@pytest.fixture
def on_xpu(monkeypatch):
    fake_cuda = types.SimpleNamespace(is_available = lambda: False)

    def install(xpu):
        monkeypatch.delitem(sys.modules, "unsloth.device_type", raising = False)
        monkeypatch.setattr(dt, "DEVICE_TYPE", "xpu")
        monkeypatch.setattr(torch, "cuda", fake_cuda)
        monkeypatch.setattr(torch, "xpu", xpu, raising = False)
        # load_vllm asks vLLM's own platform check on XPU: a stub that accepts bf16 keeps this hermetic.
        _fake_vllm_platforms(monkeypatch)
    return install


@not_on_macos
@pytest.mark.parametrize(
    "flags, expected",
    [
        ((True,), True),
        ((True, True), True),
        ((False,), True),                # no libocloc: unproven
        ((False, False), True),
        ((_PROVEN_NO_BF16,), False),     # emulation only
        ((_PROVEN_NO_BF16, _PROVEN_NO_BF16), False),
        ((_MISSING,), True),
        ((_MISSING, _MISSING), True),
    ],
)
def test_xpu_bf16_uses_native_conversions(on_xpu, flags, expected):
    on_xpu(_fake_xpu(flags))
    assert dt.device_is_bf16_supported() is expected


@not_on_macos
@pytest.mark.parametrize(
    "flags, current, expected",
    [
        ((True, _PROVEN_NO_BF16), 0, True),   # native current XPU: the unused sibling never narrows it
        ((True, _PROVEN_NO_BF16), 1, False),  # the current XPU itself emulates bf16
        ((_PROVEN_NO_BF16, True), 0, False),
        ((_PROVEN_NO_BF16, True), 1, True),
        ((True, False), 0, True),             # aspects unreadable on that device: unproven there
        ((True, False), 1, True),
        ((False, True), 0, True),
        ((_PROVEN_NO_BF16, False), 0, False),
        ((_PROVEN_NO_BF16, False), 1, True),
    ],
)
def test_xpu_bf16_judges_the_current_device_by_its_own_aspects(on_xpu, flags, current, expected):
    on_xpu(_fake_xpu(flags, current = current))
    assert dt.device_is_bf16_supported() is expected
    assert dt._xpu_bf16_supported() is expected


@not_on_macos
@pytest.mark.parametrize(
    "device, expected",
    [
        (0, True),
        (1, False),
        ("xpu:0", True),
        (torch.device("xpu", 1), False),
        ("xpu", True),          # no index: the current device (0 here)
        ([0, 1], False),        # a model split over both reaches the emulating one
        ([0, "xpu:0"], True),
        ("cpu", True),          # nothing on an XPU: nothing to narrow
    ],
)
def test_xpu_bf16_judges_the_devices_a_model_uses(on_xpu, device, expected):
    on_xpu(_fake_xpu((True, _PROVEN_NO_BF16), current = 0))
    assert dt._xpu_bf16_supported(device) is expected


@not_on_macos
def test_xpu_bf16_unreadable_current_device_keeps_bf16(on_xpu):
    xpu = _fake_xpu((_PROVEN_NO_BF16,))
    def current_device():
        raise RuntimeError("no context")
    xpu.current_device = current_device
    on_xpu(xpu)
    assert dt._xpu_bf16_supported() is True


@not_on_macos
def test_xpu_bf16_keeps_old_answer_beside_an_older_unsloth(on_xpu, monkeypatch):
    on_xpu(_fake_xpu((_PROVEN_NO_BF16,)))
    old_core = types.ModuleType("unsloth.device_type")
    monkeypatch.setitem(sys.modules, "unsloth.device_type", old_core)
    assert dt._xpu_bf16_supported() is True
    old_core.xpu_supports_native_bf16 = lambda: False
    assert dt._xpu_bf16_supported() is False


@not_on_macos
@pytest.mark.parametrize("core_answer", [True, False])
@pytest.mark.parametrize("current, expected", [(0, True), (1, False)])
def test_xpu_bf16_released_unsloth_never_narrows_the_current_device(on_xpu, monkeypatch, core_answer, current, expected):
    # Released unsloth judges every visible XPU: its False for an unused emulating sibling must not
    # turn bf16 off on a native current XPU. The current device's own answer decides.
    on_xpu(_fake_xpu((True, _PROVEN_NO_BF16), current = current))
    core = types.ModuleType("unsloth.device_type")
    core.xpu_supports_native_bf16 = lambda: core_answer
    monkeypatch.setitem(sys.modules, "unsloth.device_type", core)
    assert dt._xpu_bf16_supported() is expected


@not_on_macos
def test_xpu_bf16_unreadable_properties_keep_todays_answer(on_xpu):
    on_xpu(_fake_xpu((False,), props_raise = True))
    assert dt.device_is_bf16_supported() is True


@not_on_macos
def test_xpu_bf16_partial_build_still_false(on_xpu):
    on_xpu(_fake_xpu((True,), is_bf16 = None))
    assert dt.device_is_bf16_supported() is False


@not_on_macos
def test_cuda_bf16_still_asks_torch_cuda(monkeypatch):
    for answer in (True, False):
        fake_cuda = types.SimpleNamespace(is_available = lambda: True, is_bf16_supported = lambda a = answer: a)
        monkeypatch.setattr(dt, "DEVICE_TYPE", "cuda")
        monkeypatch.setattr(torch, "cuda", fake_cuda)
        assert dt.device_is_bf16_supported() is answer


@not_on_macos
def test_gradient_checkpointing_xpu_buffers_follow_native_bf16(monkeypatch):
    from unsloth_zoo import gradient_checkpointing as gc
    tree = ast.parse(inspect.getsource(gc.initialize_unsloth_gradient_checkpointing).lstrip())
    written = {n for node in ast.walk(tree) if isinstance(node, ast.Global) for n in node.names}
    saved = {name: getattr(gc, name, _MISSING) for name in written}
    real_empty = torch.empty

    def empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        kwargs["device"] = "cpu"
        return real_empty(*args, **kwargs)

    monkeypatch.delitem(sys.modules, "unsloth.device_type", raising = False)
    monkeypatch.setenv("UNSLOTH_DISABLE_DOUBLE_BUFFER", "1")
    # The answer is cached on the first init: an earlier test may have cached False.
    monkeypatch.setattr(gc, "_double_buffer_disabled", lambda: True)
    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(gc, "DEVICE_TYPE", "xpu")
    monkeypatch.setattr(gc, "DEVICE_TYPE_TORCH", "cuda")
    try:
        got = {}
        cases = (
            ((True,), None), ((False,), None), ((_PROVEN_NO_BF16,), None), ((_MISSING,), None),
            ((True, _PROVEN_NO_BF16), 0), ((True, _PROVEN_NO_BF16), 1),
            ((_PROVEN_NO_BF16, True), 0), ((_PROVEN_NO_BF16, True), 1), ((True, False), None),
        )
        for flags, current in cases:
            monkeypatch.setattr(torch, "xpu", _fake_xpu(flags, current = current), raising = False)
            gc.CPU_BUFFERS = []
            gc.initialize_unsloth_gradient_checkpointing()
            got[flags, current] = gc.CPU_BUFFERS[0].dtype
    finally:
        for name, value in saved.items():
            if value is _MISSING:
                if hasattr(gc, name):
                    delattr(gc, name)
            else:
                setattr(gc, name, value)
    assert got[(True,), None] is torch.bfloat16
    assert got[(_MISSING,), None] is torch.bfloat16
    assert got[(False,), None] is torch.bfloat16
    assert got[(_PROVEN_NO_BF16,), None] is torch.float16
    # The current XPU decides: an unused emulating sibling keeps a native current device bf16.
    assert got[(True, _PROVEN_NO_BF16), 0] is torch.bfloat16
    assert got[(True, _PROVEN_NO_BF16), 1] is torch.float16
    assert got[(_PROVEN_NO_BF16, True), 0] is torch.float16
    assert got[(_PROVEN_NO_BF16, True), 1] is torch.bfloat16
    assert got[(True, False), None] is torch.bfloat16


@pytest.fixture
def vu():
    import traceback

    try:
        from unsloth_zoo import vllm_utils
    except ImportError as e:
        pytest.skip(f"vllm_utils does not import here: {e}")
    except RuntimeError as e:
        # Only bitsandbytes' CUDA-mismatch error is an environment gap; anything else is a regression.
        if "bitsandbytes" not in traceback.format_exc().lower():
            raise
        pytest.skip(f"bitsandbytes does not load here: {e}")
    return vllm_utils


@not_on_macos
def test_get_mem_info_xpu_falls_back_when_mem_get_info_raises(monkeypatch, vu):
    def boom(*a, **k):
        raise RuntimeError("The device (Intel(R) Arc(TM) B580 Graphics) doesn't support querying the available free memory")
    monkeypatch.setattr(vu, "DEVICE_TYPE", "xpu")
    monkeypatch.setattr(torch, "xpu", _fake_xpu((True, True), mem_get_info = boom), raising = False)
    assert vu.get_mem_info() == (13 * _GB, 16 * _GB)


@not_on_macos
def test_get_mem_info_xpu_uses_mem_get_info_when_it_works(monkeypatch, vu):
    monkeypatch.setattr(vu, "DEVICE_TYPE", "xpu")
    monkeypatch.setattr(torch, "xpu", _fake_xpu(mem_get_info = lambda: (5, 7)), raising = False)
    assert vu.get_mem_info() == (5, 7)


@not_on_macos
def test_get_mem_info_cuda_unchanged(monkeypatch, vu):
    monkeypatch.setattr(vu, "DEVICE_TYPE", "cuda")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (11, 22))
    assert vu.get_mem_info() == (11, 22)


@not_on_macos
def test_vllm_xpu_dtype_asks_native_bf16_and_old_gpu_message_is_actionable(vu):
    source = inspect.getsource(vu.load_vllm)
    assert "_xpu_bf16_supported()" in source
    assert "fast_inference = False" in source
    assert "compute capability 7.0" in source
    assert "{major_version}.{minor_version}" in source


def _vllm_dtype_block(vu):
    source = inspect.getsource(vu.load_vllm)
    start = source.index("    # Get correct dtype")
    end = source.index("    free_memory, total_memory = get_mem_info()", start)
    lines = source[start:end].splitlines()
    return compile("\n".join(line[4:] for line in lines), "load_vllm_dtype", "exec")


@not_on_macos
@pytest.mark.parametrize(
    "flags, current, dtype, expected",
    [
        ((_PROVEN_NO_BF16,), None, None, "float16"),
        ((_PROVEN_NO_BF16,), None, "bfloat16", "bfloat16"),  # explicit bf16 still runs (emulated)
        ((_PROVEN_NO_BF16,), None, "float16", "float16"),
        ((True,), None, None, "bfloat16"),
        ((True, _PROVEN_NO_BF16), 0, None, "bfloat16"),  # native current XPU, unused emulating sibling
        ((True, _PROVEN_NO_BF16), 1, None, "float16"),   # the current XPU emulates bf16
        ((_PROVEN_NO_BF16, True), 0, None, "float16"),
        ((_PROVEN_NO_BF16, True), 1, None, "bfloat16"),
    ],
)
def test_vllm_xpu_dtype_keeps_explicit_bf16(on_xpu, vu, flags, current, dtype, expected):
    on_xpu(_fake_xpu(flags, current = current))
    scope = {
        "DEVICE_TYPE": "xpu", "torch": torch, "major_version": 0, "__name__": vu.__name__,
        "__package__": "unsloth_zoo", "dtype": None if dtype is None else getattr(torch, dtype),
    }
    exec(_vllm_dtype_block(vu), scope)
    assert scope["dtype"] is getattr(torch, expected)


@not_on_macos
@pytest.mark.parametrize("core_keeps, expected", [(True, "bfloat16"), (False, "float16")])
def test_vllm_xpu_explicit_bf16_follows_the_loading_unsloth(on_xpu, vu, monkeypatch, core_keeps, expected):
    # An unsloth that downgrades an explicit bf16 load to fp16 shares fp16 weights: vLLM must match.
    on_xpu(_fake_xpu((_PROVEN_NO_BF16,)))
    core = types.ModuleType("unsloth.device_type")
    core.xpu_supports_native_bf16 = lambda: False
    if core_keeps:
        core.xpu_keeps_explicit_bf16 = lambda: True
    monkeypatch.setitem(sys.modules, "unsloth.device_type", core)
    assert dt.XPU_KEEPS_EXPLICIT_BF16 is True
    scope = {
        "DEVICE_TYPE": "xpu", "torch": torch, "major_version": 0, "__name__": vu.__name__,
        "__package__": "unsloth_zoo", "dtype": torch.bfloat16,
    }
    exec(_vllm_dtype_block(vu), scope)
    assert scope["dtype"] is getattr(torch, expected)


def _fake_vllm_platforms(monkeypatch, refuses = False, has_check = True, missing = False):
    if missing:
        monkeypatch.setitem(sys.modules, "vllm.platforms", None)  # import raises ImportError
        return

    class _Platform:
        pass

    if has_check:
        def check_if_supports_dtype(dtype):
            if refuses and dtype is torch.bfloat16:
                raise ValueError("Intel Arc A770 have bfloat16 accuracy known issue.")
        _Platform.check_if_supports_dtype = staticmethod(check_if_supports_dtype)
    mod = types.ModuleType("vllm.platforms")
    mod.current_platform = _Platform()
    monkeypatch.setitem(sys.modules, "vllm.platforms", mod)


@not_on_macos
@pytest.mark.parametrize(
    "fake, expected",
    [
        ({"refuses": True}, True),
        ({"refuses": False}, False),
        ({"has_check": False}, False),   # older vLLM without the hook
        ({"missing": True}, False),      # vLLM not importable
    ],
)
def test_xpu_vllm_refuses_bf16_asks_vllms_own_check(monkeypatch, fake, expected):
    _fake_vllm_platforms(monkeypatch, **fake)
    assert dt._xpu_vllm_refuses_bf16() is expected


@not_on_macos
@pytest.mark.parametrize(
    "flags, dtype, refuses, expected",
    [
        ((_PROVEN_NO_BF16,), "bfloat16", True, "float16"),   # explicit bf16 on an A770: vLLM refuses
        ((True,), None, True, "float16"),                    # automatic bf16 on an A770 too
        ((_PROVEN_NO_BF16,), "bfloat16", False, "bfloat16"), # vLLM accepts it: kept
        ((True,), None, False, "bfloat16"),
        ((True,), "float16", True, "float16"),
    ],
)
def test_vllm_xpu_dtype_follows_vllms_bf16_check(on_xpu, vu, monkeypatch, flags, dtype, refuses, expected):
    on_xpu(_fake_xpu(flags))
    _fake_vllm_platforms(monkeypatch, refuses = refuses)
    scope = {
        "DEVICE_TYPE": "xpu", "torch": torch, "major_version": 0, "__name__": vu.__name__,
        "__package__": "unsloth_zoo", "dtype": None if dtype is None else getattr(torch, dtype),
    }
    exec(_vllm_dtype_block(vu), scope)
    assert scope["dtype"] is getattr(torch, expected)


@not_on_macos
@pytest.mark.parametrize("core_has_rule", [True, False])
def test_vllm_xpu_bf16_refusal_switches_beside_any_unsloth(on_xpu, vu, monkeypatch, core_has_rule):
    # Released unsloth has no xpu_vllm_refuses_bf16 and still asks for bf16. vLLM must switch anyway,
    # or its engine raises on an A770; convert_vllm_to_huggingface then follows the fp16 weights.
    expected = "float16"
    on_xpu(_fake_xpu((_PROVEN_NO_BF16,)))
    _fake_vllm_platforms(monkeypatch, refuses = True)
    core = types.ModuleType("unsloth.device_type")
    core.xpu_keeps_explicit_bf16 = lambda: True
    if core_has_rule:
        core.xpu_vllm_refuses_bf16 = lambda: True
    monkeypatch.setitem(sys.modules, "unsloth.device_type", core)
    scope = {
        "DEVICE_TYPE": "xpu", "torch": torch, "major_version": 0, "__name__": vu.__name__,
        "__package__": "unsloth_zoo", "dtype": torch.bfloat16,
    }
    exec(_vllm_dtype_block(vu), scope)
    assert scope["dtype"] is getattr(torch, expected)


@not_on_macos
@pytest.mark.parametrize("flags", [(True,), (_PROVEN_NO_BF16,)])
@pytest.mark.parametrize("dtype", [None, "bfloat16"])
def test_vllm_xpu_a770_refusal_beside_released_unsloth_main(on_xpu, vu, monkeypatch, flags, dtype):
    # zoo#1947 / zoo#1929 / unsloth#14493: unsloth main exports xpu_supports_native_bf16 (every
    # visible XPU) and xpu_keeps_explicit_bf16, but no xpu_vllm_refuses_bf16. Neither an explicit
    # nor an automatic bf16 may reach vLLM's A770 refusal beside it.
    on_xpu(_fake_xpu(flags))
    _fake_vllm_platforms(monkeypatch, refuses = True)
    core = types.ModuleType("unsloth.device_type")
    core.xpu_supports_native_bf16 = lambda: not any(f == _PROVEN_NO_BF16 for f in flags)
    core.xpu_keeps_explicit_bf16 = lambda: True
    core.zoo_shares_xpu_bf16 = lambda: True
    monkeypatch.setitem(sys.modules, "unsloth.device_type", core)
    scope = {
        "DEVICE_TYPE": "xpu", "torch": torch, "major_version": 0, "__name__": vu.__name__,
        "__package__": "unsloth_zoo", "dtype": None if dtype is None else getattr(torch, dtype),
    }
    exec(_vllm_dtype_block(vu), scope)
    assert scope["dtype"] is torch.float16


@not_on_macos
def test_vllm_cuda_dtype_never_asks_the_xpu_check(vu, monkeypatch):
    _fake_vllm_platforms(monkeypatch, refuses = True)
    scope = {
        "DEVICE_TYPE": "cuda", "torch": torch, "major_version": 9, "__name__": vu.__name__,
        "__package__": "unsloth_zoo", "dtype": torch.bfloat16,
    }
    exec(_vllm_dtype_block(vu), scope)
    assert scope["dtype"] is torch.bfloat16


@pytest.mark.parametrize(
    "info, expected",
    [
        ({"architecture": "applegpu_g13s", "device_name": "Apple M1 Max"}, False),
        ({"architecture": "applegpu_g14g", "device_name": "Apple M2"}, False),
        ({"architecture": "applegpu_g15p", "device_name": "Apple M3 Pro"}, True),
        ({"architecture": "applegpu_g16s", "device_name": "Apple M4 Max"}, True),
        ({"architecture": "applegpu_g17g", "device_name": "Apple M5"}, True),
        ({"architecture": "", "device_name": "Apple M2 Ultra"}, False),
        ({"architecture": "", "device_name": "Apple M4 Pro"}, True),
        ({}, True),
        (RuntimeError("no Metal"), True),
    ],
)
def test_mlx_bf16_matches_unsloth_apple_answer(monkeypatch, info, expected):
    def device_info():
        if isinstance(info, Exception):
            raise info
        return info
    core = types.ModuleType("mlx.core")
    core.device_info = device_info
    mlx = types.ModuleType("mlx")
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setattr(dt, "DEVICE_TYPE", "mlx")
    assert dt.device_is_bf16_supported() is expected
