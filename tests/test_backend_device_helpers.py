from unittest import mock

import torch

from unsloth_zoo import device_type as dt


def test_device_synchronize_xpu_missing_attr_does_not_raise():
    class XpuStub:
        def is_available(self):
            return True

    fake_cuda = mock.MagicMock()
    fake_cuda.is_available.return_value = False
    with mock.patch.object(dt, "DEVICE_TYPE", "xpu"), \
         mock.patch.object(torch, "cuda", fake_cuda), \
         mock.patch.object(torch, "xpu", XpuStub(), create=True):
        dt.device_synchronize()


def test_device_synchronize_xpu_calls_synchronize_when_present():
    fake_cuda = mock.MagicMock()
    fake_cuda.is_available.return_value = False
    fake_xpu = mock.MagicMock()
    fake_xpu.is_available.return_value = True
    with mock.patch.object(dt, "DEVICE_TYPE", "xpu"), \
         mock.patch.object(torch, "cuda", fake_cuda), \
         mock.patch.object(torch, "xpu", fake_xpu, create=True):
        dt.device_synchronize()
    fake_xpu.synchronize.assert_called_once()


def test_all_xpu_helpers_share_partial_build_safety():
    class XpuStub:
        def is_available(self):
            return True

    fake_cuda = mock.MagicMock()
    fake_cuda.is_available.return_value = False
    with mock.patch.object(dt, "DEVICE_TYPE", "xpu"), \
         mock.patch.object(torch, "cuda", fake_cuda), \
         mock.patch.object(torch, "xpu", XpuStub(), create=True):
        dt.device_synchronize()
        dt.device_empty_cache()
        assert dt.device_is_bf16_supported() is False
