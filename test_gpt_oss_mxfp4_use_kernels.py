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

patch_gpt_oss()
_QCLS = transformers.quantizers.quantizer_mxfp4.Mxfp4HfQuantizer


class _Cfg:
    dequantize = False


class _Model:
    class config:
        pass

    def modules(self):
        return iter(())


def _make_quantizer(dequantize=False):
    cfg = _Cfg()
    cfg.dequantize = dequantize
    q = _QCLS.__new__(_QCLS)
    q.quantization_config = cfg
    q.pre_quantized = True
    q.modules_to_not_convert = []
    q.triton_kernels_hub = None
    return q


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
    """If torch.cuda.is_available raises, the wrapper must warn and still
    call the original (not silently swallow)."""
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
