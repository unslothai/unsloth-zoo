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

"""trl logs a token-entropy metric from logits Unsloth deliberately does not have.

`SFTTrainer.compute_loss` does this, with no way to turn it off short of
`use_liger_kernel`:

    if not self.args.use_liger_kernel:
        with torch.no_grad():
            per_token_entropy = entropy_from_logits(outputs.logits)

Unsloth's entire point there is that a [batch, seq, vocab] float32 tensor -- the
largest allocation in an SFT step -- is never materialised, and `EMPTY_LOGITS`
stands in for it. So a diagnostic metric became a hard training failure, in two
different shapes, both observed live in the sweep:

    Qwen3_(32B)_A100   NotImplementedError: Unsloth: Logits are empty from
                       2024.11 onwards ... set UNSLOTH_RETURN_LOGITS=1
    Spark_TTS_(0_5B)   TypeError: iteration over a 0-d tensor
                       (trl/trainer/utils.py, per_token_entropies.extend)

on Colab AND on Kaggle, on notebooks that trained fine before trl added the
metric. The advice in the first cannot be taken: UNSLOTH_RETURN_LOGITS=1 buys
the metric back by giving up the memory saving the user came to Unsloth for.

The detection is on the OBJECT, not the exception text, and that matters.
`EmptyLogits.__getattr__` returns the `raise_logits_error` FUNCTION for every
attribute, so what actually blows up depends on which attribute the installed
trl touches first -- `.split(...)` raises NotImplementedError, while
`logits.shape[:-1]` subscripts a function object and raises TypeError. A first
draft of this patch keyed on the two messages and would have fixed one trl
version while missing the other.
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---- the real sentinel, reproduced (unsloth_zoo must not import unsloth) ----

def _raise_logits_error(*a, **k):
    raise NotImplementedError("Unsloth: Logits are empty from 2024.11 onwards.")


def _return_none(*a, **k):
    return None


class EmptyLogits:
    """Byte-for-byte the shape of unsloth/models/_utils.py's EmptyLogits."""

    def raise_getattr_error(self, attr):
        return _return_none if attr == "to" else _raise_logits_error

    __getitem__ = _raise_logits_error
    __getattr__ = raise_getattr_error


@pytest.fixture(autouse = True)
def _pin_the_accelerate_device(monkeypatch):
    """`accelerate.PartialState` is a process-global borg. Anything earlier in
    the session that builds an Accelerator (many tests here do) leaves
    `_shared_state["device"] = cuda`, and the fallback below then allocates its
    scalar there, where it meets the CPU tensors in these tests and raises
    "Expected all tensors to be on the same device". Pinned so the result of
    this module does not depend on what ran before it in the same process; the
    two device tests set their own value on top of it.
    """
    state = pytest.importorskip("accelerate.state")
    monkeypatch.setitem(state.PartialState._shared_state, "device", torch.device("cpu"))


@pytest.fixture(scope="module")
def patched():
    trl_utils = pytest.importorskip("trl.trainer.utils")
    import trl.trainer.sft_trainer as sft
    from unsloth_zoo.temporary_patches.misc import patch_trl_entropy_from_logits

    # entropy_from_logits landed in trl 0.20.0 and pyproject allows >= 0.18.2,
    # so an allowed build can lack it. Production no-ops there, so skip rather
    # than error out of the fixture.
    original = getattr(trl_utils, "entropy_from_logits", None)
    if original is None:
        pytest.skip("this trl predates entropy_from_logits (added in 0.20.0)")
    # Importing unsloth_zoo applies TEMPORARY_PATCHES, so by the time this
    # fixture runs the name may already be the wrapper. functools.wraps records
    # the real one on __wrapped__; without this the "unpatched control" test
    # would call the wrapper and conclude the bug had never existed.
    while getattr(original, "_unsloth_patched", False) and \
            getattr(original, "__wrapped__", None) is not None:
        original = original.__wrapped__
    patch_trl_entropy_from_logits()
    yield trl_utils, sft, original
    trl_utils.entropy_from_logits = original
    sft.entropy_from_logits = original


# ---- it degrades instead of failing --------------------------------------

def test_the_unsloth_sentinel_no_longer_raises(patched):
    trl_utils, _, _ = patched
    assert float(trl_utils.entropy_from_logits(EmptyLogits())) == 0.0


def test_the_unpatched_call_really_does_raise(patched):
    """Without this the test above could be passing against a bug that no
    longer exists."""
    _, _, original = patched
    with pytest.raises(Exception):
        original(EmptyLogits())


@pytest.mark.parametrize("logits", [None, torch.randn(0, 7), torch.randn(5)])
def test_other_unusable_logits_degrade_too(patched, logits):
    trl_utils, _, _ = patched
    assert float(trl_utils.entropy_from_logits(logits)) == 0.0


def test_the_result_broadcasts_the_way_the_caller_uses_it(patched):
    """The caller does `sum(entropy * attention_mask) / sum(mask)` in one
    branch and `mean(entropy)` in the other. A shape that raises in either is
    no better than the original failure."""
    trl_utils, _, _ = patched
    e = trl_utils.entropy_from_logits(EmptyLogits())
    mask = torch.ones(2, 5)
    assert float(torch.sum(e * mask) / mask.sum()) == 0.0
    assert float(torch.mean(e)) == 0.0


# ---- and survives the gather trl does next -------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason = "no accelerator to land on")
def test_the_fallback_lands_on_the_accelerate_device(patched, monkeypatch):
    """trl passes this straight to `accelerator.gather_for_metrics`, whose
    distributed path allocates on PartialState().device and calls NCCL, which
    rejects a CPU tensor with "Tensors must be CUDA and dense". The masked
    branch is rescued by multiplying with a device-resident mask, but the
    padding-free branch (trl's default packing_strategy is "bfd", so plain
    packing=True reaches it) is a bare mean(), so a CPU scalar would swap one
    multi-GPU crash for another."""
    trl_utils, _, _ = patched
    state = pytest.importorskip("accelerate.state")
    monkeypatch.setitem(state.PartialState._shared_state, "device", torch.device("cuda", 0))
    e = trl_utils.entropy_from_logits(EmptyLogits())
    assert e.device.type == "cuda"
    assert torch.mean(e).device.type == "cuda"
    mask = torch.ones(2, 5, device = "cuda")
    assert (torch.sum(e * mask) / mask.sum()).device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "no accelerator to land on")
def test_a_cpu_sentinel_tensor_does_not_drag_the_scalar_to_cpu(patched, monkeypatch):
    """The fused-loss path returns a MODULE-LEVEL `EMPTY_LOGITS = torch.empty(0)`
    (unsloth_zoo/fused_losses/forward_adapter.py), built at import time and so
    always on cpu regardless of the run's device. Following the tensor's own
    device there would hand NCCL a CPU scalar on exactly the distributed run
    this fallback exists for, and the padding-free branch is a bare mean() with
    no device-resident mask to rescue it."""
    trl_utils, _, _ = patched
    state = pytest.importorskip("accelerate.state")
    monkeypatch.setitem(state.PartialState._shared_state, "device", torch.device("cuda", 0))
    assert trl_utils.entropy_from_logits(torch.empty(0)).device.type == "cuda"


def test_a_cpu_sentinel_tensor_stays_on_cpu_for_a_cpu_run(patched, monkeypatch):
    trl_utils, _, _ = patched
    state = pytest.importorskip("accelerate.state")
    monkeypatch.setitem(state.PartialState._shared_state, "device", torch.device("cpu"))
    assert trl_utils.entropy_from_logits(torch.empty(0)).device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "no accelerator to land on")
def test_the_fallback_follows_the_logits_device(patched):
    trl_utils, _, _ = patched
    empty = torch.empty(0, 7, device = "cuda")
    assert trl_utils.entropy_from_logits(empty).device == empty.device


def test_the_fallback_stays_on_cpu_for_a_cpu_run(patched, monkeypatch):
    """A CPU-only run must not be handed a device it never allocates on, so the
    device is read off accelerate rather than guessed from torch.cuda."""
    trl_utils, _, _ = patched
    state = pytest.importorskip("accelerate.state")
    monkeypatch.setitem(state.PartialState._shared_state, "device", torch.device("cpu"))
    assert trl_utils.entropy_from_logits(EmptyLogits()).device.type == "cpu"


# ---- and does not quietly break the metric for everyone else -------------

def test_real_logits_are_untouched(patched):
    """The patch must be invisible when logits do exist -- otherwise it turns
    a working metric into a zero for every non-Unsloth user of trl."""
    trl_utils, _, original = patched
    logits = torch.randn(2, 5, 11)
    assert torch.allclose(trl_utils.entropy_from_logits(logits), original(logits))


def test_a_genuine_error_still_raises(patched):
    """A bug inside entropy_from_logits must not be swallowed on every step."""
    trl_utils, _, _ = patched
    with pytest.raises(Exception):
        trl_utils.entropy_from_logits("not logits at all")


# ---- where it is installed ------------------------------------------------

def test_it_patches_the_module_the_caller_actually_reads(patched):
    """sft_trainer.py does `from ..trainer.utils import entropy_from_logits`,
    which binds the name at import time. Patching only trl.trainer.utils would
    be a no-op for the one caller that matters."""
    trl_utils, sft, _ = patched
    assert getattr(trl_utils.entropy_from_logits, "_unsloth_patched", False)
    assert getattr(sft.entropy_from_logits, "_unsloth_patched", False)


def test_applying_it_twice_does_not_stack(patched):
    from unsloth_zoo.temporary_patches.misc import patch_trl_entropy_from_logits
    trl_utils, _, _ = patched
    once = trl_utils.entropy_from_logits
    patch_trl_entropy_from_logits()
    assert trl_utils.entropy_from_logits is once


def test_it_is_registered(patched):
    from unsloth_zoo.temporary_patches.common import TEMPORARY_PATCHES
    assert any(getattr(f, "__name__", "") == "patch_trl_entropy_from_logits"
               for f in TEMPORARY_PATCHES)


# ---- the source ----------------------------------------------------------

def _src():
    return (ROOT / "unsloth_zoo" / "temporary_patches" / "misc.py").read_text(encoding="utf-8")


def test_detection_is_on_the_object_not_the_message():
    """The point of the rewrite. If someone reintroduces a message match as
    the primary discriminator, it will pass on one trl and fail on another."""
    src = _src()
    i = src.index("def patch_trl_entropy_from_logits")
    body = src[i:src.index('TEMPORARY_PATCHES.append(patch_trl_entropy_from_logits)', i)]
    assert 'type(logits).__name__ == "EmptyLogits"' in body
    assert body.index("def _unusable") < body.index("except TypeError")


def test_the_backstop_stays_narrow():
    src = _src()
    i = src.index("def patch_trl_entropy_from_logits")
    body = src[i:src.index('TEMPORARY_PATCHES.append(patch_trl_entropy_from_logits)', i)]
    assert '"iteration over a 0-d tensor" not in str(e)' in body
    assert "raise" in body


def test_the_logger_is_imported():
    """The warning is not inside a bare `except Exception: pass`, so a missing
    `logger` would be a NameError on the first step rather than a silent no-op."""
    import unsloth_zoo.temporary_patches.misc as misc
    assert hasattr(misc, "logger")
    assert hasattr(misc, "functools")


def test_the_warning_says_how_to_get_the_real_number():
    src = _src()
    i = src.index("def patch_trl_entropy_from_logits")
    assert "UNSLOTH_RETURN_LOGITS=1" in src[i:src.index('TEMPORARY_PATCHES.append(patch_trl_entropy_from_logits)', i)]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
