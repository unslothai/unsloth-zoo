import os
import sys
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

    def __init__(self, mods):
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
    """Freshly-init Mxfp4GptOssExperts with zero placeholders should be
    treated as not-loaded and skipped, not raised on."""
    mod = _make_module(blocks_zero=True, scales_zero=True,
                       down_blocks_zero=True, down_scales_zero=True)
    q = _make_quantizer(dequantize=False)
    _QCLS._process_model_after_weight_loading(q, _Model([mod]))


def test_loaded_blocks_raise_when_swizzle_fn_missing(hide_swizzle_fn):
    """When real blocks are present, the missing-swizzle branch must still raise."""
    mod = _make_module(blocks_zero=False, scales_zero=False)
    q = _make_quantizer(dequantize=False)
    with pytest.raises(RuntimeError, match="raw blocks/scales"):
        _QCLS._process_model_after_weight_loading(q, _Model([mod]))


def test_meta_scales_treated_as_not_loaded(hide_swizzle_fn):
    """Real blocks + meta scales should be skipped, not crash."""
    mod = _make_module(blocks_zero=False, scales_meta=True,
                       down_blocks_zero=True, down_scales_zero=True)
    q = _make_quantizer(dequantize=False)
    _QCLS._process_model_after_weight_loading(q, _Model([mod]))


def test_per_projection_skip_repairs_uncached_down_proj():
    """If _gate_up_proj is cached early (e.g. property accessed) but
    down_proj is still raw and loaded, the active swizzle path must
    repair down_proj instead of skipping the whole module."""
    mod = _make_module(blocks_zero=True, scales_zero=True,
                       down_blocks_zero=False, down_scales_zero=False)
    # Simulate gate_up_proj cached
    mod.__dict__["_gate_up_proj"] = torch.zeros(2, 4, 8)

    swizzle_calls = []

    def fake_swizzle(b, s, mod, proj, dev, tk):
        swizzle_calls.append(proj)
        # mimic upstream cleanup: remove the raw params
        if f"{proj}_blocks" in mod._parameters:
            del mod._parameters[f"{proj}_blocks"]
        if f"{proj}_scales" in mod._parameters:
            del mod._parameters[f"{proj}_scales"]

    saved = getattr(_mx_mod, "swizzle_mxfp4_convertops", None)
    _mx_mod.swizzle_mxfp4_convertops = fake_swizzle
    try:
        # provide a fake top-level triton_kernels module so the import succeeds
        import types as _types
        sys.modules.setdefault("triton_kernels", _types.ModuleType("triton_kernels"))
        q = _make_quantizer(dequantize=False)
        _QCLS._process_model_after_weight_loading(q, _Model([mod]))
    finally:
        if saved is not None:
            _mx_mod.swizzle_mxfp4_convertops = saved

    assert swizzle_calls == ["down_proj"], swizzle_calls


def test_per_projection_skip_skips_both_when_both_cached():
    """If both projections are cached (post-load already ran), the
    active swizzle path must be a no-op."""
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
    """Module has gate_up_proj loaded but down_proj still zero. Missing
    swizzle_fn branch must raise (since some weights are loaded)."""
    mod = _make_module(blocks_zero=False, scales_zero=False,
                       down_blocks_zero=True, down_scales_zero=True)
    q = _make_quantizer(dequantize=False)
    with pytest.raises(RuntimeError, match="raw blocks/scales"):
        _QCLS._process_model_after_weight_loading(q, _Model([mod]))
