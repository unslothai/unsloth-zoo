import os
import sys
import warnings
import torch
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from unsloth_zoo.temporary_patches.gpt_oss import patch_gpt_oss
import transformers
import transformers.integrations.mxfp4 as _mx_mod

patch_gpt_oss()
_QCLS = transformers.quantizers.quantizer_mxfp4.Mxfp4HfQuantizer


class Mxfp4GptOssExperts(torch.nn.Module):
    pass


class _Cfg:
    dequantize = False


class _Model:
    class config:
        pass

    is_quantized = True

    def __init__(self, mods=()):
        self._mods = mods

    def modules(self):
        return iter(self._mods)


def _make_quantizer(dequantize=False):
    cfg = _Cfg()
    cfg.dequantize = dequantize
    q = _QCLS.__new__(_QCLS)
    q.quantization_config = cfg
    q.pre_quantized = True
    q.modules_to_not_convert = []
    q.triton_kernels_hub = None
    return q


def _make_module(blocks_zero=True, scales_zero=True, scales_meta=False,
                 down_blocks_zero=True, down_scales_zero=True):
    m = Mxfp4GptOssExperts()
    m.gate_up_proj_blocks = torch.nn.Parameter(
        torch.zeros(2, 8, 4, 16, dtype=torch.uint8) if blocks_zero
        else torch.ones(2, 8, 4, 16, dtype=torch.uint8),
        requires_grad=False,
    )
    if scales_meta:
        m.gate_up_proj_scales = torch.nn.Parameter(
            torch.empty(2, 8, 4, dtype=torch.uint8, device="meta"),
            requires_grad=False,
        )
    else:
        m.gate_up_proj_scales = torch.nn.Parameter(
            torch.zeros(2, 8, 4, dtype=torch.uint8) if scales_zero
            else torch.ones(2, 8, 4, dtype=torch.uint8),
            requires_grad=False,
        )
    m.down_proj_blocks = torch.nn.Parameter(
        torch.zeros(2, 4, 4, 16, dtype=torch.uint8) if down_blocks_zero
        else torch.ones(2, 4, 4, 16, dtype=torch.uint8),
        requires_grad=False,
    )
    m.down_proj_scales = torch.nn.Parameter(
        torch.zeros(2, 4, 4, dtype=torch.uint8) if down_scales_zero
        else torch.ones(2, 4, 4, dtype=torch.uint8),
        requires_grad=False,
    )
    return m


@pytest.fixture
def hide_swizzle_fn():
    saved = getattr(_mx_mod, "swizzle_mxfp4_convertops", None)
    if saved is not None:
        delattr(_mx_mod, "swizzle_mxfp4_convertops")
    yield
    if saved is not None:
        _mx_mod.swizzle_mxfp4_convertops = saved


def test_zero_placeholders_do_not_raise_when_swizzle_fn_missing(hide_swizzle_fn):
    mod = _make_module(blocks_zero=True, scales_zero=True,
                       down_blocks_zero=True, down_scales_zero=True)
    q = _make_quantizer(dequantize=False)
    _QCLS._process_model_after_weight_loading(q, _Model([mod]))


def test_loaded_blocks_raise_when_swizzle_fn_missing(hide_swizzle_fn):
    mod = _make_module(blocks_zero=False, scales_zero=False)
    q = _make_quantizer(dequantize=False)
    with pytest.raises(RuntimeError, match="raw blocks/scales"):
        _QCLS._process_model_after_weight_loading(q, _Model([mod]))


def test_meta_scales_treated_as_not_loaded(hide_swizzle_fn):
    mod = _make_module(blocks_zero=False, scales_meta=True,
                       down_blocks_zero=True, down_scales_zero=True)
    q = _make_quantizer(dequantize=False)
    _QCLS._process_model_after_weight_loading(q, _Model([mod]))


def test_per_projection_skip_repairs_uncached_down_proj():
    mod = _make_module(blocks_zero=True, scales_zero=True,
                       down_blocks_zero=False, down_scales_zero=False)
    mod.__dict__["_gate_up_proj"] = torch.zeros(2, 4, 8)

    swizzle_calls = []

    def fake_swizzle(b, s, mod, proj, dev, tk):
        swizzle_calls.append(proj)
        if f"{proj}_blocks" in mod._parameters:
            del mod._parameters[f"{proj}_blocks"]
        if f"{proj}_scales" in mod._parameters:
            del mod._parameters[f"{proj}_scales"]

    saved = getattr(_mx_mod, "swizzle_mxfp4_convertops", None)
    _mx_mod.swizzle_mxfp4_convertops = fake_swizzle
    try:
        import types as _types
        sys.modules.setdefault("triton_kernels", _types.ModuleType("triton_kernels"))
        q = _make_quantizer(dequantize=False)
        _QCLS._process_model_after_weight_loading(q, _Model([mod]))
    finally:
        if saved is not None:
            _mx_mod.swizzle_mxfp4_convertops = saved

    assert swizzle_calls == ["down_proj"], swizzle_calls


def test_per_projection_skip_skips_both_when_both_cached():
    mod = _make_module(blocks_zero=False, scales_zero=False,
                       down_blocks_zero=False, down_scales_zero=False)
    mod.__dict__["_gate_up_proj"] = torch.zeros(2, 4, 8)
    mod.__dict__["_down_proj"] = torch.zeros(2, 4, 4)

    swizzle_calls = []

    def fake_swizzle(b, s, mod, proj, dev, tk):
        swizzle_calls.append(proj)

    saved = getattr(_mx_mod, "swizzle_mxfp4_convertops", None)
    _mx_mod.swizzle_mxfp4_convertops = fake_swizzle
    try:
        import types as _types
        sys.modules.setdefault("triton_kernels", _types.ModuleType("triton_kernels"))
        q = _make_quantizer(dequantize=False)
        _QCLS._process_model_after_weight_loading(q, _Model([mod]))
    finally:
        if saved is not None:
            _mx_mod.swizzle_mxfp4_convertops = saved

    assert swizzle_calls == []


def test_partial_load_one_projection_loaded(hide_swizzle_fn):
    mod = _make_module(blocks_zero=False, scales_zero=False,
                       down_blocks_zero=True, down_scales_zero=True)
    q = _make_quantizer(dequantize=False)
    with pytest.raises(RuntimeError, match="raw blocks/scales"):
        _QCLS._process_model_after_weight_loading(q, _Model([mod]))


def _stub_orig_noop():
    fn = _QCLS._process_model_before_weight_loading
    fn.__closure__[0].cell_contents = lambda self, model, **kwargs: None


def _patch_devices(monkeypatch, *, cuda=False, xpu=False):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    if not hasattr(torch, "xpu"):
        torch.xpu = type("xpu", (), {"is_available": staticmethod(lambda: xpu)})
    monkeypatch.setattr(torch.xpu, "is_available", lambda: xpu)


def test_cpu_use_kernels_true_keeps_dequantize_false(monkeypatch):
    _patch_devices(monkeypatch, cuda=False, xpu=False)
    _stub_orig_noop()
    q = _make_quantizer(dequantize=False)
    _QCLS._process_model_before_weight_loading(q, _Model(), use_kernels=True)
    assert q.quantization_config.dequantize is False


def test_cpu_use_kernels_false_forces_dequantize(monkeypatch):
    _patch_devices(monkeypatch, cuda=False, xpu=False)
    _stub_orig_noop()
    q = _make_quantizer(dequantize=False)
    _QCLS._process_model_before_weight_loading(q, _Model(), use_kernels=False)
    assert q.quantization_config.dequantize is True


def test_cpu_default_call_forces_dequantize(monkeypatch):
    _patch_devices(monkeypatch, cuda=False, xpu=False)
    _stub_orig_noop()
    q = _make_quantizer(dequantize=False)
    _QCLS._process_model_before_weight_loading(q, _Model())
    assert q.quantization_config.dequantize is True


def test_positional_use_kernels_does_not_raise(monkeypatch):
    _patch_devices(monkeypatch, cuda=False, xpu=False)
    _stub_orig_noop()
    q = _make_quantizer(dequantize=False)
    _QCLS._process_model_before_weight_loading(q, _Model(), True)
    assert q.quantization_config.dequantize is False


def test_use_kernels_forwarded_to_orig(monkeypatch):
    _patch_devices(monkeypatch, cuda=True, xpu=False)
    seen = {}

    def fake_orig(self, model, **kwargs):
        seen["use_kernels"] = kwargs.get("use_kernels")

    fn = _QCLS._process_model_before_weight_loading
    fn.__closure__[0].cell_contents = fake_orig
    q = _make_quantizer(dequantize=False)
    _QCLS._process_model_before_weight_loading(q, _Model(), use_kernels=True)
    assert seen.get("use_kernels") is True


def test_detection_failure_warns_and_proceeds(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: (_ for _ in ()).throw(RuntimeError("driver gone")))
    called = {"orig": False}

    def fake_orig(self, model, **kwargs):
        called["orig"] = True

    fn = _QCLS._process_model_before_weight_loading
    fn.__closure__[0].cell_contents = fake_orig
    q = _make_quantizer(dequantize=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _QCLS._process_model_before_weight_loading(q, _Model())
    assert called["orig"] is True
    assert any(
        "MXFP4 pre-load device detection failed" in str(w.message) for w in caught
    ), [str(w.message) for w in caught]
