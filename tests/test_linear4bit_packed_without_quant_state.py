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

import pytest

# `importorskip`, not a bare import. Every test here needs torch, and the macOS staging
# runner does not ship it: a bare `import torch` turns the whole module into a COLLECTION
# ERROR, which is not the same thing as a skip. The step that runs this suite has an escape
# for a missing dependency, but it counts `^(FAILED|ERROR) ` lines against the ones that name
# the module, and a collection error prints only the file path with no message -- so the
# escape could not fire and the leg went red. Measured on staging-1313 macos-15: "1 skipped,
# 1 warning, 1 error", exit 2, while windows-latest passed because its install brings torch.
torch = pytest.importorskip("torch")

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


def _real_bitsandbytes():
    """The installed bitsandbytes, or skip.

    ``importorskip`` is not enough. On a host with no real bitsandbytes, zoo installs
    `stubs/bitsandbytes_stub.py` under that name, so the import succeeds and every
    attribute resolves to a permissive no-op: the patch has nothing to patch,
    `Linear4bit(...)` raises NotImplementedError, and these tests either fail for a
    reason that has nothing to do with the guard or pass without touching it. Ask the
    canonical helper whether the real package is there.
    """
    bitsandbytes = pytest.importorskip("bitsandbytes")
    from unsloth_zoo.stubs.bitsandbytes_stub import real_bitsandbytes_available
    if not real_bitsandbytes_available():
        pytest.skip("bitsandbytes is the unsloth_zoo stub on this host")
    return bitsandbytes


def _patched_forward():
    """The real patched Linear4bit.forward, as installed by the temporary patch."""
    bitsandbytes = _real_bitsandbytes()
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
    # Explicitly the no-repair case, so this cannot depend on whether something else in the
    # session installed the runtime repair. The repair branch is asserted in
    # tests/test_composite_conversion_rescope.py.
    monkeypatch.setattr(
        bnb_patch, "_composite_renaming_repair_installed", lambda: False
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
    #
    # Asserted on the ADVICE, not on the whole message. A bare `"4.57" not in message`
    # also matches the installed version this message interpolates, so it failed on a
    # host that really is running 4.57.6 -- where the predicate is mocked True here --
    # for a reason that has nothing to do with what the advice says.
    _, _, advice = message.partition("fixed by PR #45567 in 5.6.0.")
    assert advice, "the advice section moved; this assertion no longer reads it"
    assert "4.57" not in advice
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
    import sys

    def _boom(name):
        raise _md.PackageNotFoundError(name)

    monkeypatch.setattr(_md, "version", _boom)
    monkeypatch.delitem(sys.modules, "transformers", raising = False)
    assert bnb_patch._installed_transformers_version() == "unknown"
    assert bnb_patch._transformers_drops_prequantized_quant_state() is False


def test_version_falls_back_to_an_already_imported_transformers(monkeypatch):
    """A checkout with no .dist-info still gets the defect-window explanation."""
    import importlib.metadata as _md
    import sys
    import types

    def _boom(name):
        raise _md.PackageNotFoundError(name)

    monkeypatch.setattr(_md, "version", _boom)
    fake = types.ModuleType("transformers")
    fake.__version__ = "5.5.4"
    monkeypatch.setitem(sys.modules, "transformers", fake)
    assert bnb_patch._installed_transformers_version() == "5.5.4"
    assert bnb_patch._transformers_drops_prequantized_quant_state() is True


def test_unquantized_layer_still_falls_through():
    """A genuinely unquantized [out, in] weight must keep working, not raise."""
    forward = _patched_forward()
    module = _FakeLinear4bit(torch.randn(16, 8), out_features = 16, in_features = 8)
    out = forward(module, torch.randn(3, 8))
    assert tuple(out.shape) == (3, 16)


def test_a_packed_one_by_one_weight_is_not_mistaken_for_a_scalar_layer():
    """in_features == out_features == 1 packs to (1, 1), the same shape as unquantized.

    The one-input exemption is what makes this ambiguous, so it also requires a float
    weight. Measured on bitsandbytes 0.50.2: a quantized Linear4bit(1, 1) has weight
    (1, 1) uint8, and once the sidecars are lost it is a plain Parameter, so
    `isinstance(weight, Params4bit)` and `bnb_quantized` are both gone and cannot be the
    discriminator. The dtype survives, and an unquantized weight is never uint8.
    """
    forward = _patched_forward()
    packed = _FakeLinear4bit(
        torch.zeros((1, 1), dtype = torch.uint8), out_features = 1, in_features = 1,
    )
    with pytest.raises(RuntimeError) as excinfo:
        forward(packed, torch.zeros(4, 1, dtype = torch.float16))
    assert "quant_state" in str(excinfo.value)

    # The real scalar layer this exemption exists for still works.
    scalar = _FakeLinear4bit(torch.randn(1, 1), out_features = 1, in_features = 1)
    assert tuple(forward(scalar, torch.randn(4, 1)).shape) == (4, 1)


@pytest.mark.parametrize("in_features,itemsize", [(2, 1), (4, 2), (8, 4)])
def test_packed_blob_whose_rows_equal_out_features_is_still_caught(in_features, itemsize):
    """The arithmetic coincidence that a row-count comparison alone misses.

    A packed blob has ``out_features * in_features // (2 * quant_storage.itemsize)``
    rows, which equals ``out_features`` exactly when ``in_features`` is twice the
    storage itemsize: 2 for uint8, 4 for float16/bfloat16, 8 for float32. Comparing
    only ``shape[0]`` against ``out_features`` excused those layers and handed the
    user the shape error this PR exists to remove. Only a real one-input layer may
    be excused, so the guard asks about ``in_features`` instead.
    """
    forward = _patched_forward()
    out_features = 16
    packed = torch.zeros((out_features * in_features // (2 * itemsize), 1), dtype = torch.uint8)
    assert packed.shape[0] == out_features, "this test is pointless unless they coincide"
    module = _FakeLinear4bit(packed, out_features = out_features, in_features = in_features)
    with pytest.raises(RuntimeError) as excinfo:
        forward(module, torch.zeros(3, in_features, dtype = torch.float16))
    assert "mat1 and mat2 shapes cannot be multiplied" not in str(excinfo.value)
    assert "quant_state" in str(excinfo.value)


def test_unquantized_linear4bit_with_one_input_feature_is_not_accused():
    """A legitimate Linear4bit with in_features == 1 has a (out_features, 1) weight.

    That matches the packed-blob shape test on its own, so the guard also compares
    against out_features. Without that clause this forward raises instead of
    returning, which turns a working model into a hard error and tells the user to
    reinstall transformers over a checkpoint that was never involved.
    """
    bnb = _real_bitsandbytes()
    bnb_patch.patch_bitsandbytes_linear4bit_forward()
    module = bnb.nn.Linear4bit(1, 16)
    # On CPU bitsandbytes leaves the layer unquantized, which is the state under test.
    # Pin it, so a future bnb that quantizes here fails loudly instead of quietly
    # exercising the other branch and passing for the wrong reason.
    assert getattr(module.weight, "quant_state", None) is None
    out = module(torch.randn(4, 1))
    assert tuple(out.shape) == (4, 16)
