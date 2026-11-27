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

import os
import sys

import pytest

torch = pytest.importorskip("torch")

from torch._inductor import scheduler  # noqa: E402

from unsloth_zoo import inductor_nonowning_inplace as guard  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from _cpu_compiler_skip import skip_without_cpu_inductor  # noqa: E402


@pytest.fixture(autouse = True)
def _guard_enabled(monkeypatch):
    # A caller's opt-out must not change what these tests check.
    monkeypatch.delenv("UNSLOTH_INDUCTOR_NONOWNING_GUARD", raising = False)


def _upstream_fixed():
    import inspect
    method = scheduler.BaseSchedulerNode.decide_inplace_update
    return "NonOwningLayout" in inspect.getsource(getattr(method, "_unsloth_original", method))


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
    if guard._torch_below_2_16() and not _upstream_fixed():
        assert installed and getattr(method, guard._FLAG, False)
        assert guard.patch_inductor_nonowning_inplace() is True
        assert scheduler.BaseSchedulerNode.decide_inplace_update is method
    else:
        assert installed is False and not getattr(method, guard._FLAG, False)


def test_patched_source_adds_only_nonowning_layout():
    import difflib
    import inspect
    import textwrap
    node = scheduler.BaseSchedulerNode
    plain = getattr(node.decide_inplace_update, "_unsloth_original", node.decide_inplace_update)
    source = textwrap.dedent(inspect.getsource(plain))
    if "NonOwningLayout" in source:
        pytest.skip("torch already excludes NonOwningLayout")
    patched = guard._insert_nonowning(source)
    changed = [line for line in difflib.ndiff(source.splitlines(), patched.splitlines()) if line[:1] in "+-"]
    assert [(line[0], line[2:].strip()) for line in changed] == [("+", "ir.NonOwningLayout,")]


def test_inline_anchor_is_refused():
    inline = "def f(self):\n    return (ir.NoneLayout, ir.MutationLayoutSHOULDREMOVE,)\n"
    assert guard._insert_nonowning(inline) is None


def test_version_gate(monkeypatch):
    for version, below in (("2.6.0", True), ("2.15.0+cu132", True), ("2.16.0.dev20261122+cpu", True),
                           ("2.16.0", False), ("2.17.0.dev20270101", False)):
        monkeypatch.setattr(torch, "__version__", version)
        assert guard._torch_below_2_16() is below, version


def test_kill_switch_and_installed_state(monkeypatch):
    node = scheduler.BaseSchedulerNode
    current = node.decide_inplace_update
    installed = getattr(current, guard._FLAG, False)
    monkeypatch.setenv("UNSLOTH_INDUCTOR_NONOWNING_GUARD", "0")
    # An installed guard still reports itself; a fresh install is refused.
    assert guard.patch_inductor_nonowning_inplace() is installed
    monkeypatch.setattr(node, "decide_inplace_update", getattr(current, "_unsloth_original", current))
    assert guard.patch_inductor_nonowning_inplace() is False
    assert not getattr(node.decide_inplace_update, guard._FLAG, False)


def test_cache_tag_is_set_once_and_reaches_workers():
    import threading
    if not guard.patch_inductor_nonowning_inplace():
        pytest.skip("guard not installed on this torch")
    config = torch.compiler.config
    guard._tag_inductor_cache()
    tags = config.cache_key_tag.split(",")
    assert tags.count(guard.NONOWNING_INPLACE_CACHE_TAG) == 1
    assert guard.NONOWNING_INPLACE_CACHE_TAG in os.environ.get("TORCH_COMPILE_CACHE_KEY_TAG", "").split(",")
    seen = []
    thread = threading.Thread(target = lambda: seen.append(config.cache_key_tag))
    thread.start()
    thread.join()
    assert guard.NONOWNING_INPLACE_CACHE_TAG in seen[0].split(",")


def test_compiled_cat_matches_eager():
    skip_without_cpu_inductor()
    guard.patch_inductor_nonowning_inplace()
    assert _compiled_error(scheduler.BaseSchedulerNode.decide_inplace_update) < 1e-3


def test_unguarded_inductor_reuses_the_cat_slice():
    skip_without_cpu_inductor()
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


def test_cpu_compile_path_installs_the_guard():
    import inspect
    from unsloth_zoo.cpu import loader
    source = inspect.getsource(loader.compile_cpu_layers)
    assert "patch_inductor_nonowning_inplace()" in source
    assert source.index("patch_inductor_nonowning_inplace()") < source.index("_compile_layers(model)")


def test_temporary_patches_reexports_the_installer():
    from unsloth_zoo.temporary_patches import inductor_nonowning_inplace as shim
    assert shim.patch_inductor_nonowning_inplace is guard.patch_inductor_nonowning_inplace
