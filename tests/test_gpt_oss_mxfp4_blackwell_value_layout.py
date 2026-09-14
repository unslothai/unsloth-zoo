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

"""gpt-oss MXFP4 on Blackwell: value swizzling must be skipped on builds whose
matmul kernel refuses it, and only on those.

Measured motivation, B200 (sm_100) + triton_kernels @ 0add6826 + transformers
4.56.2: `unsloth/gpt-oss-20b` loads natively, then the first `generate` dies with
`CompileTimeAssertionFailure: Only Hopper swizzling is supported for values`,
because that build's layout selector hands back `BlackwellMXValueLayout` while
its `_matmul_ogs` asserts `SWIZZLE_MX_VALUE == "HOPPER_VALUE" or ... is None`.

CPU-only: everything here is decision logic, with `triton_kernels` faked. The
hardware claims live in the B200 and Kaggle T4x2 runs, not in this file.
"""
import sys
import types

import pytest
import torch

from unsloth_zoo.temporary_patches.gpt_oss import (
    _blackwell_value_swizzle_unsupported,
    _force_strided_mxfp4_values,
    _mxfp4_layout_arguments,
    _mxfp4_layout_selection_is_class_contract,
    _normalize_mxfp4_value_layout,
    _source_rejects_blackwell_value_swizzle,
)


# The assert as the affected build actually writes it (triton_kernels @ 0add6826,
# matmul_ogs_details/_matmul_ogs.py:107-114), kept verbatim so a rewrite upstream
# shows up here as a failing fixture rather than as silent under-detection.
HOPPER_ONLY_SOURCE = '''
@triton.jit
def _matmul_ogs(Y, X, W, stride_w_k, WMxScale, SWIZZLE_MX_VALUE: tl.constexpr,
                SWIZZLE_MX_SCALE: tl.constexpr, is_w_microscaled: tl.constexpr):
    if is_w_microscaled:
        w_type = W.dtype.element_ty
        tl.static_assert(WMxScale.dtype.element_ty == tl.uint8, "mx_scale_ptr must be uint8")
        tl.static_assert(SWIZZLE_MX_VALUE == "HOPPER_VALUE" or SWIZZLE_MX_VALUE is None, "Only Hopper swizzling is supported for values")
    else:
        tl.static_assert(SWIZZLE_MX_VALUE is None)
'''

# A build that takes Blackwell values: no such assert anywhere.
BLACKWELL_CAPABLE_SOURCE = '''
@triton.jit
def _matmul_ogs(Y, X, W, WMxScale, SWIZZLE_MX_VALUE: tl.constexpr, is_w_microscaled: tl.constexpr):
    if is_w_microscaled:
        tl.static_assert(WMxScale.dtype.element_ty == tl.uint8, "mx_scale_ptr must be uint8")
        if SWIZZLE_MX_VALUE == "BLACKWELL_VALUE":
            tl.static_assert(BLOCK_N % 128 == 0)
'''

# The same words, but only in a comment and a string. Must not count.
LOOKALIKE_SOURCE = '''
@triton.jit
def _matmul_ogs(Y, X, W):
    # tl.static_assert(SWIZZLE_MX_VALUE == "HOPPER_VALUE" or SWIZZLE_MX_VALUE is None)
    message = 'SWIZZLE_MX_VALUE == "HOPPER_VALUE" or SWIZZLE_MX_VALUE is None'
    return message
'''


class _FakeJITFunction:
    """Stands in for a triton JITFunction, which exposes its source as `.src`."""
    def __init__(self, src):
        self.src = src


class _SourcelessKernel:
    """No `.src`, no `.fn`: the probe must decline rather than raise."""


def _install_fake_kernel_module(monkeypatch, source, *, kernel_name = "_matmul_ogs"):
    for name in ("triton_kernels", "triton_kernels.matmul_ogs_details"):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    leaf = types.ModuleType("triton_kernels.matmul_ogs_details._matmul_ogs")
    if source is not None:
        setattr(leaf, kernel_name, _FakeJITFunction(source))
    else:
        setattr(leaf, kernel_name, _SourcelessKernel())
    monkeypatch.setitem(sys.modules, leaf.__name__, leaf)
    monkeypatch.setattr(
        sys.modules["triton_kernels.matmul_ogs_details"], "_matmul_ogs", leaf,
        raising = False,
    )
    return leaf


class BlackwellMXValueLayout:
    pass


class HopperMXValueLayout:
    pass


class StridedLayout:
    pass


class _LayoutModule:
    BlackwellMXValueLayout = BlackwellMXValueLayout
    HopperMXValueLayout = HopperMXValueLayout
    StridedLayout = StridedLayout


class _FakeWeight:
    """Minimal stand-in for a CUDA tensor: the guard only reads is_cuda/device."""
    def __init__(self, is_cuda = True, device = "cuda:0"):
        self.is_cuda = is_cuda
        self.device = device


@pytest.fixture
def sm100(monkeypatch):
    """Report sm_100 for any device. conftest pins the host to (8, 0)."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))


@pytest.fixture(autouse = True)
def clear_probe_cache():
    _source_rejects_blackwell_value_swizzle.cache_clear()
    yield
    _source_rejects_blackwell_value_swizzle.cache_clear()


# --- the source probe -------------------------------------------------------

def test_probe_detects_the_hopper_only_assert():
    assert _source_rejects_blackwell_value_swizzle(HOPPER_ONLY_SOURCE) is True


def test_probe_clears_a_blackwell_capable_kernel():
    assert _source_rejects_blackwell_value_swizzle(BLACKWELL_CAPABLE_SOURCE) is False


def test_probe_ignores_comments_and_strings():
    assert _source_rejects_blackwell_value_swizzle(LOOKALIKE_SOURCE) is False


def test_probe_survives_unparseable_source():
    assert _source_rejects_blackwell_value_swizzle("def (:\n") is False


def test_probe_reads_the_installed_module(monkeypatch):
    _install_fake_kernel_module(monkeypatch, HOPPER_ONLY_SOURCE)
    assert _blackwell_value_swizzle_unsupported() is True


def test_probe_scans_every_kernel_in_the_module(monkeypatch):
    """A persistent variant carrying the restriction counts too."""
    leaf = _install_fake_kernel_module(monkeypatch, BLACKWELL_CAPABLE_SOURCE)
    leaf._p_matmul_ogs = _FakeJITFunction(HOPPER_ONLY_SOURCE)
    assert _blackwell_value_swizzle_unsupported() is True


def test_probe_declines_when_source_is_unreadable(monkeypatch):
    _install_fake_kernel_module(monkeypatch, None)
    assert _blackwell_value_swizzle_unsupported() is False


def test_probe_declines_when_triton_kernels_is_absent(monkeypatch):
    """Blocking the package, not just the leaf: vLLM vendors triton_kernels under
    its real name, so a leaf-only block still resolves to a live module."""
    for name in ("triton_kernels", "triton_kernels.matmul_ogs_details"):
        monkeypatch.setitem(sys.modules, name, None)
    assert _blackwell_value_swizzle_unsupported() is False


# --- the two layout selector contracts --------------------------------------

def test_class_contract_is_passed_through():
    selection = (BlackwellMXValueLayout, {"mx_axis": 1})
    assert _mxfp4_layout_selection_is_class_contract(selection) is True
    assert _normalize_mxfp4_value_layout(selection) == (BlackwellMXValueLayout, {"mx_axis": 1})


def test_instance_contract_gets_empty_kwargs():
    """Newer triton_kernels returns a layout instance; unpacking it would raise."""
    instance = BlackwellMXValueLayout()
    assert _mxfp4_layout_selection_is_class_contract(instance) is False
    layout, opts = _normalize_mxfp4_value_layout(instance)
    assert layout is instance and opts == {}


def test_a_two_tuple_of_instances_is_not_the_class_contract():
    pair = (BlackwellMXValueLayout(), BlackwellMXValueLayout())
    assert _mxfp4_layout_selection_is_class_contract(pair) is False


# --- the guard --------------------------------------------------------------

def test_guard_fires_on_sm100_with_an_affected_build(monkeypatch, sm100):
    _install_fake_kernel_module(monkeypatch, HOPPER_ONLY_SOURCE)
    assert _force_strided_mxfp4_values(BlackwellMXValueLayout, _LayoutModule, _FakeWeight()) is True


def test_guard_fires_for_an_instance_selection(monkeypatch, sm100):
    _install_fake_kernel_module(monkeypatch, HOPPER_ONLY_SOURCE)
    assert _force_strided_mxfp4_values(BlackwellMXValueLayout(), _LayoutModule, _FakeWeight()) is True


@pytest.mark.parametrize("capability", [(7, 5), (8, 0), (9, 0)])
def test_guard_is_a_no_op_below_blackwell(monkeypatch, capability):
    """T4, A100 and H100 keep exactly the layout the build chose for them."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
    _install_fake_kernel_module(monkeypatch, HOPPER_ONLY_SOURCE)
    assert _force_strided_mxfp4_values(HopperMXValueLayout, _LayoutModule, _FakeWeight()) is False


def test_guard_is_a_no_op_on_a_blackwell_capable_build(monkeypatch, sm100):
    _install_fake_kernel_module(monkeypatch, BLACKWELL_CAPABLE_SOURCE)
    assert _force_strided_mxfp4_values(BlackwellMXValueLayout, _LayoutModule, _FakeWeight()) is False


def test_guard_is_a_no_op_when_the_build_did_not_choose_blackwell(monkeypatch, sm100):
    _install_fake_kernel_module(monkeypatch, HOPPER_ONLY_SOURCE)
    assert _force_strided_mxfp4_values(StridedLayout, _LayoutModule, _FakeWeight()) is False


def test_guard_is_a_no_op_without_cuda(monkeypatch, sm100):
    """macOS, Windows-without-CUDA and CPU boxes never reach the swizzle path."""
    _install_fake_kernel_module(monkeypatch, HOPPER_ONLY_SOURCE)
    assert _force_strided_mxfp4_values(
        BlackwellMXValueLayout, _LayoutModule, _FakeWeight(is_cuda = False),
    ) is False


def test_guard_is_a_no_op_when_the_layout_class_is_missing(monkeypatch, sm100):
    class _NoBlackwell:
        StridedLayout = StridedLayout
    _install_fake_kernel_module(monkeypatch, HOPPER_ONLY_SOURCE)
    assert _force_strided_mxfp4_values(BlackwellMXValueLayout, _NoBlackwell, _FakeWeight()) is False


def test_guard_survives_a_capability_lookup_that_raises(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("no driver")
    monkeypatch.setattr(torch.cuda, "get_device_capability", _boom)
    _install_fake_kernel_module(monkeypatch, HOPPER_ONLY_SOURCE)
    assert _force_strided_mxfp4_values(BlackwellMXValueLayout, _LayoutModule, _FakeWeight()) is False


# --- the arguments handed to convert_layout ---------------------------------

class _OldStyleLayoutModule:
    """triton_kernels @ 0add6826: selector returns (class, kwargs), convert_layout
    instantiates `layout_cls(shape, **kwargs)`."""
    BlackwellMXValueLayout = BlackwellMXValueLayout
    HopperMXValueLayout = HopperMXValueLayout
    StridedLayout = StridedLayout

    @staticmethod
    def make_default_matmul_mxfp4_w_layout(mx_axis):
        return BlackwellMXValueLayout, {}


class _NewStyleLayoutModule:
    """Current triton_kernels: selector returns an instance and convert_layout
    takes it as is, so unpacking the result would raise TypeError."""
    BlackwellMXValueLayout = BlackwellMXValueLayout
    HopperMXValueLayout = HopperMXValueLayout
    StridedLayout = StridedLayout

    @staticmethod
    def make_default_matmul_mxfp4_w_layout(mx_axis):
        return BlackwellMXValueLayout()


def test_old_contract_passes_the_class_and_kwargs(monkeypatch, sm100):
    _install_fake_kernel_module(monkeypatch, BLACKWELL_CAPABLE_SOURCE)
    value, opts, strided = _mxfp4_layout_arguments(_OldStyleLayoutModule, _FakeWeight())
    assert value is BlackwellMXValueLayout and opts == {}
    # Scales keep zoo's long-standing strided handling, in class form here.
    assert strided is StridedLayout


def test_new_contract_passes_instances(monkeypatch, sm100):
    _install_fake_kernel_module(monkeypatch, BLACKWELL_CAPABLE_SOURCE)
    value, opts, strided = _mxfp4_layout_arguments(_NewStyleLayoutModule, _FakeWeight())
    assert isinstance(value, BlackwellMXValueLayout) and opts == {}
    assert isinstance(strided, StridedLayout)


def test_guard_swaps_in_strided_in_the_form_each_contract_wants(monkeypatch, sm100):
    _install_fake_kernel_module(monkeypatch, HOPPER_ONLY_SOURCE)

    value, opts, strided = _mxfp4_layout_arguments(_OldStyleLayoutModule, _FakeWeight())
    assert value is StridedLayout and opts == {} and strided is StridedLayout

    value, opts, strided = _mxfp4_layout_arguments(_NewStyleLayoutModule, _FakeWeight())
    assert isinstance(value, StridedLayout) and opts == {}
    assert value is strided


def test_cpu_weight_keeps_the_builds_own_choice(monkeypatch, sm100):
    """No CUDA weight means no decision to make, under either contract."""
    _install_fake_kernel_module(monkeypatch, HOPPER_ONLY_SOURCE)
    value, _, _ = _mxfp4_layout_arguments(_NewStyleLayoutModule, _FakeWeight(is_cuda = False))
    assert isinstance(value, BlackwellMXValueLayout)


# --- the escape hatch -------------------------------------------------------

def test_env_default_opts_out_even_on_an_affected_build(monkeypatch, sm100):
    monkeypatch.setenv("UNSLOTH_MXFP4_VALUE_LAYOUT", "default")
    _install_fake_kernel_module(monkeypatch, HOPPER_ONLY_SOURCE)
    assert _force_strided_mxfp4_values(BlackwellMXValueLayout, _LayoutModule, _FakeWeight()) is False


def test_env_strided_opts_in_without_the_probe(monkeypatch, sm100):
    """For a build we misjudge: force it, no kernel inspection at all."""
    monkeypatch.setenv("UNSLOTH_MXFP4_VALUE_LAYOUT", "strided")
    _install_fake_kernel_module(monkeypatch, BLACKWELL_CAPABLE_SOURCE)
    assert _force_strided_mxfp4_values(BlackwellMXValueLayout, _LayoutModule, _FakeWeight()) is True


def test_env_strided_still_needs_a_cuda_weight(monkeypatch, sm100):
    monkeypatch.setenv("UNSLOTH_MXFP4_VALUE_LAYOUT", "strided")
    _install_fake_kernel_module(monkeypatch, BLACKWELL_CAPABLE_SOURCE)
    assert _force_strided_mxfp4_values(
        BlackwellMXValueLayout, _LayoutModule, _FakeWeight(is_cuda = False),
    ) is False


def test_env_is_read_per_call_not_at_import(monkeypatch, sm100):
    """Setting it after `import unsloth` must still take effect."""
    _install_fake_kernel_module(monkeypatch, HOPPER_ONLY_SOURCE)
    assert _force_strided_mxfp4_values(BlackwellMXValueLayout, _LayoutModule, _FakeWeight()) is True
    monkeypatch.setenv("UNSLOTH_MXFP4_VALUE_LAYOUT", "DEFAULT")
    assert _force_strided_mxfp4_values(BlackwellMXValueLayout, _LayoutModule, _FakeWeight()) is False
