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

"""trl's token-entropy metric must not break SFT on Unsloth's EmptyLogits stand-in for logits; detect the object, not
the message, since EmptyLogits.__getattr__ makes the error depend on the attribute touched; UNSLOTH_RETURN_LOGITS=1
gives up the memory saving."""

import inspect
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
    """accelerate.PartialState is a process-global borg: an earlier Accelerator leaves the shared device set to cuda, so
    the fallback scalar meets CPU tensors; pin it to CPU here."""
    state = pytest.importorskip("accelerate.state")
    monkeypatch.setitem(state.PartialState._shared_state, "device", torch.device("cpu"))


@pytest.fixture(scope="module")
def patched():
    trl_utils = pytest.importorskip("trl.trainer.utils")
    import trl.trainer.sft_trainer as sft
    from unsloth_zoo.temporary_patches.misc import patch_trl_entropy_from_logits

    # entropy_from_logits landed in trl 0.20.0 and pyproject allows >= 0.18.2, so an allowed build can lack it;
    # production no-ops there, so skip rather than error.
    original = getattr(trl_utils, "entropy_from_logits", None)
    if original is None:
        pytest.skip("this trl predates entropy_from_logits (added in 0.20.0)")
    # Importing unsloth_zoo applies TEMPORARY_PATCHES, so the name may already be the wrapper; functools.wraps records
    # the real one on __wrapped__, and the unpatched control must unwrap it or it would call the wrapper and conclude
    # the bug never existed.
    while getattr(original, "_unsloth_patched", False) and \
            getattr(original, "__wrapped__", None) is not None:
        original = original.__wrapped__
    patch_trl_entropy_from_logits()
    yield trl_utils, sft, original
    trl_utils.entropy_from_logits = original
    sft.entropy_from_logits = original


def test_the_unsloth_sentinel_no_longer_raises(patched):
    trl_utils, _, _ = patched
    assert float(trl_utils.entropy_from_logits(EmptyLogits())) == 0.0


def test_the_unpatched_call_really_does_raise(patched):
    """Guards that the unpatched call really raises, so the test above cannot pass against a bug that no longer exists."""
    _, _, original = patched
    with pytest.raises(Exception):
        original(EmptyLogits())


@pytest.mark.parametrize("logits", [None, torch.randn(0, 7), torch.randn(5)])
def test_other_unusable_logits_degrade_too(patched, logits):
    trl_utils, _, _ = patched
    assert float(trl_utils.entropy_from_logits(logits)) == 0.0


def test_the_result_broadcasts_the_way_the_caller_uses_it(patched):
    """The caller uses sum(entropy * attention_mask) / sum(mask) in one branch and mean(entropy) in the other, so the
    result must work with both; a shape that raises in either is no better than the original failure."""
    trl_utils, _, _ = patched
    e = trl_utils.entropy_from_logits(EmptyLogits())
    mask = torch.ones(2, 5)
    assert float(torch.sum(e * mask) / mask.sum()) == 0.0
    assert float(torch.mean(e)) == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "no accelerator to land on")
def test_the_fallback_lands_on_the_accelerate_device(patched, monkeypatch):
    """trl passes the scalar to accelerator.gather_for_metrics, which NCCL rejects on CPU; the masked branch is saved by
    a device-resident mask, but the padding-free bare mean() (plain packing=True reaches it) is not, so the scalar
    must be on the accelerate device."""
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
    """EMPTY_LOGITS (unsloth_zoo/fused_losses/forward_adapter.py) is built on cpu at import, so the fallback must not
    follow the sentinel's device: NCCL would get a CPU scalar."""
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
    """A CPU-only run must not get a device it never allocates on, so the device is read from accelerate, not guessed from torch.cuda."""
    trl_utils, _, _ = patched
    state = pytest.importorskip("accelerate.state")
    monkeypatch.setitem(state.PartialState._shared_state, "device", torch.device("cpu"))
    assert trl_utils.entropy_from_logits(EmptyLogits()).device.type == "cpu"


def test_real_logits_are_untouched(patched):
    """The patch must leave real logits untouched; otherwise a working metric becomes zero for every non-Unsloth user of trl."""
    trl_utils, _, original = patched
    logits = torch.randn(2, 5, 11)
    assert torch.allclose(trl_utils.entropy_from_logits(logits), original(logits))


def test_a_genuine_error_still_raises(patched):
    """A bug inside entropy_from_logits must not be swallowed on every step."""
    trl_utils, _, _ = patched
    with pytest.raises(Exception):
        trl_utils.entropy_from_logits("not logits at all")


def test_it_patches_the_module_the_caller_actually_reads(patched):
    """sft_trainer.py binds entropy_from_logits at import time, so patching only trl.trainer.utils is a no-op for the one caller that matters."""
    trl_utils, sft, _ = patched
    assert getattr(trl_utils.entropy_from_logits, "_unsloth_patched", False)
    if not hasattr(sft, "entropy_from_logits"):
        # trl 1.15 takes the entropy from its fused LM head and no longer imports the helper.
        assert "entropy_from_logits" not in inspect.getsource(sft.SFTTrainer.compute_loss)
        pytest.skip("this trl's SFTTrainer does not call entropy_from_logits")
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


def _src():
    return (ROOT / "unsloth_zoo" / "temporary_patches" / "misc.py").read_text(encoding="utf-8")


def test_detection_is_on_the_object_not_the_message():
    """Detection must key on the object, not the message: a message match as the primary discriminator passes on one trl and fails on another."""
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
    """The warning is not inside a bare except Exception: pass, so a missing logger raises NameError on the first step, not a silent no-op."""
    import unsloth_zoo.temporary_patches.misc as misc
    assert hasattr(misc, "logger")
    assert hasattr(misc, "functools")


def test_the_warning_says_how_to_get_the_real_number():
    src = _src()
    i = src.index("def patch_trl_entropy_from_logits")
    assert "UNSLOTH_RETURN_LOGITS=1" in src[i:src.index('TEMPORARY_PATCHES.append(patch_trl_entropy_from_logits)', i)]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
