# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

"""A packed 4-bit weight with no quant_state must be named, never fed to F.linear.

unsloth #9867, #10010, #10017, #10276. transformers 5.4.0 and 5.5.x discard the
bitsandbytes quant_state sidecars of pre-quantized composite checkpoints. The
patched Linear4bit.forward used to treat the resulting ``quant_state is None`` as
"this layer is simply not quantized" and call ``F.linear`` on the packed [N, 1]
uint8 buffer, producing

    RuntimeError: mat1 and mat2 shapes cannot be multiplied (8x5120 and 1x15728640)

which reads like a corrupt checkpoint. Reproduced end to end on one B200 with
``unsloth/qwen3.8-27b-unsloth-bnb-4bit`` and transformers 5.5.4.

CPU only, no GPU, no bitsandbytes CUDA kernels, no network.
"""

from __future__ import annotations

import torch

import pytest

from unsloth_zoo.temporary_patches import bitsandbytes as bnb_patch


class _FakeLinear4bit(torch.nn.Module):
    """The shape of a Linear4bit after a load that dropped the quant_state.

    ``out_features`` and ``in_features`` are always set, because the real class
    subclasses ``nn.Linear`` and can never be missing them. Leaving them off made
    the guard fire through its ``getattr(self, "out_features", -1)`` default, which
    is a state no real module reaches, so the shape comparison went untested.
    """

    def __init__(self, weight, bias = None, out_features = 5120, in_features = 5120):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad = False)
        self.bias = bias
        self.out_features = out_features
        self.in_features = in_features
        self.quant_state = None
        self.quant_storage = torch.uint8
        self.compute_type_is_set = True
        self.compute_dtype = None


def _packed_weight(n_bytes = 8192):
    return torch.zeros((n_bytes, 1), dtype = torch.uint8)


def _patched_forward():
    """The real patched Linear4bit.forward, as installed by the temporary patch."""
    bitsandbytes = pytest.importorskip("bitsandbytes")
    bnb_patch.patch_bitsandbytes_linear4bit_forward()
    forward = bitsandbytes.nn.modules.Linear4bit.forward
    forward = getattr(forward, "__wrapped__", forward)
    # patch_function can decline and return False, which this patch discards. Without
    # this assert a test would then silently exercise upstream bitsandbytes and pass.
    assert forward.__module__ == "unsloth_zoo.temporary_patches.bitsandbytes"
    return forward


def test_packed_weight_without_quant_state_raises_instead_of_a_shape_error():
    """This is the whole point: the reported RuntimeError must not happen again."""
    forward = _patched_forward()
    module = _FakeLinear4bit(_packed_weight(15728640), out_features = 6144, in_features = 5120)
    x = torch.zeros(8, 5120, dtype = torch.float16)
    with pytest.raises(RuntimeError) as excinfo:
        forward(module, x)
    message = str(excinfo.value)
    assert "quant_state" in message
    assert "PACKED" in message
    # The old behaviour, verbatim from unsloth #10010, which must never be what a user sees.
    assert "mat1 and mat2 shapes cannot be multiplied" not in message


def test_error_message_names_the_transformers_window_when_installed(monkeypatch):
    monkeypatch.setattr(
        bnb_patch, "_transformers_drops_prequantized_quant_state", lambda: True
    )
    message = bnb_patch._packed_weight_without_quant_state_error(_FakeLinear4bit(_packed_weight()))
    assert "5.4.0" in message and "5.5.4" in message
    assert "#45567" in message
    # It must not repeat the false advice that the checkpoint needs regenerating.
    # The advice is conditional on trying a supported transformers first, because the
    # guard reads the installed version and never inspects the checkpoint: a file whose
    # sidecars really are absent reaches this same branch.
    assert "before regenerating anything" in message
    assert "does need rebuilding" in message
    # transformers 4.57.6 ships no qwen3_5 model at all (the first release carrying it
    # is 5.3.0), so recommending it to a Qwen3.5 reporter swaps the shape error for an
    # unrecognised-architecture error. Never offer it as the fallback.
    assert "4.57" not in message
    # The claim about where the state was lost is a version inference, not an
    # observation of the checkpoint, and must stay hedged.
    assert "most likely" in message


def test_error_message_stays_generic_outside_the_window(monkeypatch):
    monkeypatch.setattr(
        bnb_patch, "_transformers_drops_prequantized_quant_state", lambda: False
    )
    message = bnb_patch._packed_weight_without_quant_state_error(_FakeLinear4bit(_packed_weight()))
    assert "absmax" in message
    assert "#45567" not in message


@pytest.mark.parametrize(
    "version,expected",
    [
        ("5.3.0", False),
        ("5.4.0", True),
        ("5.5.0", True),
        ("5.5.4", True),
        ("5.6.0", False),
        ("5.17.0", False),
        ("4.57.6", False),
    ],
)
def test_defect_window_predicate(monkeypatch, version, expected):
    import importlib.metadata as _md

    real = _md.version
    monkeypatch.setattr(
        _md, "version", lambda name: version if name == "transformers" else real(name)
    )
    assert bnb_patch._transformers_drops_prequantized_quant_state() is expected


def test_predicate_never_raises_without_transformers(monkeypatch):
    import importlib.metadata as _md

    def _boom(name):
        raise _md.PackageNotFoundError(name)

    monkeypatch.setattr(_md, "version", _boom)
    assert bnb_patch._transformers_drops_prequantized_quant_state() is False


def test_unquantized_layer_still_falls_through():
    """A genuinely unquantized [out, in] weight must keep working, not raise."""
    forward = _patched_forward()
    module = _FakeLinear4bit(torch.randn(16, 8), out_features = 16, in_features = 8)
    out = forward(module, torch.randn(3, 8))
    assert tuple(out.shape) == (3, 16)


def test_packed_blob_whose_rows_equal_out_features_keeps_the_old_behaviour():
    """The one documented corner where the guard deliberately declines.

    A packed blob has ``(out_features * in_features) // (2 * quant_storage.itemsize)``
    rows, which equals ``out_features`` exactly when ``in_features`` is twice the
    storage itemsize: 2 for uint8, 4 for float16/bfloat16, 8 for float32. The guard
    cannot tell that apart from a legitimate ``in_features == 1`` layer, so it
    declines and the pre-existing F.linear path runs, unchanged. That is a known
    false negative, never a false accusation, and this pins it so a future change
    to the predicate has to decide about it on purpose.
    """
    forward = _patched_forward()
    module = _FakeLinear4bit(_packed_weight(16), out_features = 16, in_features = 2)
    with pytest.raises(RuntimeError) as excinfo:
        forward(module, torch.zeros(3, 2, dtype = torch.float16))
    # Still the old, unhelpful message: the guard did not fire.
    assert "mat1 and mat2 shapes cannot be multiplied" in str(excinfo.value)


def test_unquantized_linear4bit_with_one_input_feature_is_not_accused():
    """A legitimate Linear4bit with in_features == 1 has a (out_features, 1) weight.

    That matches the packed-blob shape test on its own, so the guard also compares
    against out_features. Without that clause this forward raises instead of
    returning, which turns a working model into a hard error and tells the user to
    reinstall transformers over a checkpoint that was never involved.
    """
    bnb = pytest.importorskip("bitsandbytes")
    bnb_patch.patch_bitsandbytes_linear4bit_forward()
    module = bnb.nn.Linear4bit(1, 16)
    # On CPU bitsandbytes leaves the layer unquantized, which is the state under test.
    # Pin it, so a future bnb that quantizes here fails loudly instead of quietly
    # exercising the other branch and passing for the wrong reason.
    assert getattr(module.weight, "quant_state", None) is None
    out = module(torch.randn(4, 1))
    assert tuple(out.shape) == (4, 16)
