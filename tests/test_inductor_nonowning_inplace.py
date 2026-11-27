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

"""A cat input that is a NonOwningLayout view of the cat buffer must not be reused in place (wrong output on
torch < 2.16 without the guard). CPU Inductor shows it, so no GPU is needed."""

import shutil

import pytest

torch = pytest.importorskip("torch")

from torch._inductor import scheduler  # noqa: E402

from unsloth_zoo.temporary_patches import inductor_nonowning_inplace as guard  # noqa: E402

needs_cpp = pytest.mark.skipif(shutil.which("g++") is None and shutil.which("c++") is None, reason = "no C++ compiler")


def _graph(a, b, w1, w2, w3):
    x = a + torch.mm(a.view(1, 2048), w1).view(1, 1, 2048)
    cat_buf = torch.cat([x, b], dim = -1)
    y = torch.mm(cat_buf.view(1, 4096), w2)
    z = x + y[:, :2048].view(1, 1, 2048)
    h = torch.mm(z.view(1, 2048), w3)
    clobber = torch.relu(h + 10.0)
    return z + torch.mm(clobber, w2).view(1, 1, 2048)


def _compiled_error(decide_inplace_update):
    import torch._inductor.config as inductor_config
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(1, 1, 2048, generator = g), torch.randn(1, 1, 2048, generator = g)
    w1 = torch.randn(2048, 2048, generator = g) * 0.02
    w2 = torch.randn(4096, 2048, generator = g) * 0.02
    w3 = torch.randn(2048, 4096, generator = g) * 0.02
    expected = _graph(a, b, w1, w2, w3)
    node = scheduler.BaseSchedulerNode
    current = node.decide_inplace_update
    node.decide_inplace_update = decide_inplace_update
    try:
        torch._dynamo.reset()
        with inductor_config.patch({"force_disable_caches": True, "cpp_wrapper": False, "inplace_buffers": True}):
            got = torch.compile(_graph, backend = "inductor")(a, b, w1, w2, w3)
    finally:
        node.decide_inplace_update = current
        torch._dynamo.reset()
    return (expected - got).abs().max().item()


def test_guard_installed_exactly_below_2_16():
    installed = guard.patch_inductor_nonowning_inplace()
    method = scheduler.BaseSchedulerNode.decide_inplace_update
    if guard._torch_below_2_16():
        assert installed and getattr(method, guard._FLAG, False)
        assert guard.patch_inductor_nonowning_inplace() is True
        assert scheduler.BaseSchedulerNode.decide_inplace_update is method
    else:
        assert installed is False and not getattr(method, guard._FLAG, False)


def test_patched_source_adds_only_nonowning_layout():
    if not guard.patch_inductor_nonowning_inplace():
        pytest.skip("torch already excludes NonOwningLayout")
    import inspect
    original = inspect.getsource(scheduler.BaseSchedulerNode.decide_inplace_update._unsloth_original).splitlines()
    assert "NonOwningLayout" not in "\n".join(original)
    i = next(k for k, line in enumerate(original) if guard._ANCHOR in line)
    assert original[i - 1].strip() == "ir.MultiOutputLayout,"


@needs_cpp
def test_compiled_cat_matches_eager():
    guard.patch_inductor_nonowning_inplace()
    assert _compiled_error(scheduler.BaseSchedulerNode.decide_inplace_update) < 1e-3


@needs_cpp
def test_unguarded_inductor_reuses_the_cat_slice():
    # Proves the test above can fail: the same graph without the guard is wrong.
    if not guard.patch_inductor_nonowning_inplace():
        pytest.skip("torch already excludes NonOwningLayout")
    original = scheduler.BaseSchedulerNode.decide_inplace_update._unsloth_original
    assert _compiled_error(original) > 1.0


def test_fixed_or_moved_source_is_left_alone(monkeypatch):
    node = scheduler.BaseSchedulerNode
    current = node.decide_inplace_update
    plain = getattr(current, "_unsloth_original", current)
    monkeypatch.setattr(node, "decide_inplace_update", plain)
    fixed = "def decide_inplace_update(self):\n    return (ir.MutationLayoutSHOULDREMOVE,\n        ir.NonOwningLayout,)\n"
    monkeypatch.setattr(guard.inspect, "getsource", lambda fn: fixed)
    assert guard.patch_inductor_nonowning_inplace() is False
    monkeypatch.setattr(guard.inspect, "getsource", lambda fn: "def decide_inplace_update(self):\n    pass\n")
    assert guard.patch_inductor_nonowning_inplace() is False
    monkeypatch.undo()
    assert node.decide_inplace_update is current


def test_torch_2_16_is_never_patched(monkeypatch):
    node = scheduler.BaseSchedulerNode
    current = node.decide_inplace_update
    monkeypatch.setattr(node, "decide_inplace_update", getattr(current, "_unsloth_original", current))
    monkeypatch.setattr(torch, "__version__", "2.16.0")
    assert guard.patch_inductor_nonowning_inplace() is False
    assert not getattr(node.decide_inplace_update, guard._FLAG, False)
