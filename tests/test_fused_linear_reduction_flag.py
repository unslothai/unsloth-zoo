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

"""_mm_into turns the process-wide reduced-precision reduction switch off around its GEMM. Overlapping threads must
all see it off and the caller's setting (split-K included) must come back. The switch is settable without a GPU, so
the real _mm_into runs on CPU with stand-in operands (only .is_cuda / .dtype are read before the patched torch.mm)."""

import threading
import types

import pytest
import torch

import unsloth_zoo.fused_losses.fused_linear as fl

FLAGS = torch.backends.cuda.matmul
DTYPES = [torch.float16, torch.bfloat16]


def _name(dtype):
    return "allow_bf16_reduced_precision_reduction" if dtype == torch.bfloat16 else "allow_fp16_reduced_precision_reduction"


def _half(dtype):
    return types.SimpleNamespace(is_cuda = True, dtype = dtype)


def _call(dtype):
    fl._mm_into(_half(dtype), _half(dtype), _half(dtype))


@pytest.fixture
def restore_flag():
    saved = {n: getattr(FLAGS, n) for n in map(_name, DTYPES)}
    yield
    for n, v in saved.items():
        setattr(FLAGS, n, v)


@pytest.mark.parametrize("dtype", DTYPES)
def test_overlapping_threads_see_the_switch_off_and_restore_it(dtype, monkeypatch, restore_flag):
    name = _name(dtype)
    b_started, b_in, a_done = threading.Event(), threading.Event(), threading.Event()
    seen, errors = {}, []

    def held_mm(a, b, out = None):
        if threading.current_thread().name == "A":
            assert b_started.wait(15)
            b_in.wait(0.5)  # B only gets inside first when nothing serialises the switch
        else:
            b_in.set()
            assert a_done.wait(15)
        seen[threading.current_thread().name] = getattr(FLAGS, name)

    def work(tag):
        try:
            if tag == "B":
                b_started.set()
            _call(dtype)
        except Exception as e:
            errors.append(repr(e))
        finally:
            if tag == "A":
                a_done.set()

    setattr(FLAGS, name, True)
    monkeypatch.setattr(torch, "mm", held_mm)
    threads = [threading.Thread(target = work, args = (t,), name = t) for t in ("A", "B")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert not errors, errors
    assert seen == {"A": False, "B": False}
    assert getattr(FLAGS, name) is True


@pytest.mark.parametrize("dtype", DTYPES)
def test_switch_restored_when_the_gemm_raises(dtype, monkeypatch, restore_flag):
    def boom(a, b, out = None):
        raise RuntimeError("boom")

    setattr(FLAGS, _name(dtype), True)
    monkeypatch.setattr(torch, "mm", boom)
    with pytest.raises(RuntimeError, match = "boom"):
        _call(dtype)
    assert getattr(FLAGS, _name(dtype)) is True


@pytest.mark.parametrize("dtype", DTYPES)
def test_caller_split_k_setting_survives(dtype, monkeypatch, restore_flag):
    name = _name(dtype)
    if not hasattr(FLAGS, name + "_split_k"):
        pytest.skip("this torch has no split-K reduction setting (added in torch 2.10)")
    state = lambda: (getattr(FLAGS, name), getattr(FLAGS, name + "_split_k"))
    inside = []
    monkeypatch.setattr(torch, "mm", lambda a, b, out = None: inside.append(state()))
    setattr(FLAGS, name, (False, False))
    _call(dtype)
    assert inside == [(False, False)]
    assert state() == (False, False)
