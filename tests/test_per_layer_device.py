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

"""verify_and_set_device must publish a usable device for every layer (#3538).

CPU and meta both have index None, which `move_to_device` rejects with "Invalid target
device: None" the moment a pipeline-parallel reader in unsloth reaches the layer.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from unsloth_zoo.patching_utils import verify_and_set_device


class _FakeLayer:
    def __init__(self, *devices):
        self._devices = devices

    def parameters(self):
        for device in self._devices:
            yield SimpleNamespace(device=torch.device(device))


def _move_to_device_accepts(target_device) -> bool:
    """move_to_device's type contract, copied so this does not need `unsloth` installed."""
    return isinstance(target_device, (int, str, torch.device))


CASES = {
    "cuda indexed": ("cuda:3", 3),
    "cuda zero": ("cuda:0", 0),
    "xpu indexed": ("xpu:1", 1),
    "cpu offloaded": ("cpu", "cpu"),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_both_attributes_are_usable(case):
    device_string, expected_index = CASES[case]
    layer = _FakeLayer(device_string)

    verify_and_set_device(layer)

    assert layer._per_layer_device == torch.device(device_string)
    assert layer._per_layer_device_index == expected_index
    assert layer._per_layer_device_index is not None, (
        "a None index is exactly the #3538 crash: move_to_device raises "
        "ValueError: Invalid target device: None"
    )
    assert _move_to_device_accepts(layer._per_layer_device)
    assert _move_to_device_accepts(layer._per_layer_device_index)


@pytest.mark.parametrize("device_string", ["cuda:0", "cuda:2", "xpu:1"])
def test_indexed_accelerators_keep_the_integer_index(device_string):
    """Still an int: readers subscript a per-device tuple of layernorm buffers with it."""
    layer = _FakeLayer(device_string)
    verify_and_set_device(layer)

    index = layer._per_layer_device_index
    assert isinstance(index, int) and not isinstance(index, bool)
    assert index == torch.device(device_string).index

    per_device_buffers = tuple(range(8))
    assert per_device_buffers[index] == index


def test_cpu_index_does_not_silently_become_cuda_zero():
    layer = _FakeLayer("cpu")
    verify_and_set_device(layer)

    assert layer._per_layer_device_index != 0
    assert torch.device(layer._per_layer_device_index) == torch.device("cpu")


def test_a_layer_on_two_devices_still_raises():
    layer = _FakeLayer("cpu", "cuda:0")
    with pytest.raises(ValueError, match="should be on the same device"):
        verify_and_set_device(layer)


def test_real_module_on_cpu():
    layer = torch.nn.Linear(4, 4)
    verify_and_set_device(layer)

    assert layer._per_layer_device == torch.device("cpu")
    assert layer._per_layer_device_index == "cpu"
    moved = torch.zeros(2, 4).to(torch.device(layer._per_layer_device_index))
    assert moved.device == torch.device("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a real GPU")
def test_real_module_on_cuda():
    layer = torch.nn.Linear(4, 4).cuda()
    verify_and_set_device(layer)

    assert layer._per_layer_device.type == "cuda"
    assert layer._per_layer_device_index == torch.cuda.current_device()



def test_a_meta_layer_publishes_nothing_rather_than_meta():
    """meta passes every reader's type check and still propagates through matmul rather
    than raising, so publishing it turns the #3538 ValueError into a silent empty decode."""
    activation = torch.ones(2, 4)
    assert activation.to("meta").device.type == "meta"
    assert torch.matmul(activation.to("meta"), torch.ones(4, 4)).device.type == "meta"

    layer = _FakeLayer("meta")
    verify_and_set_device(layer)

    assert not hasattr(layer, "_per_layer_device")
    assert not hasattr(layer, "_per_layer_device_index")


def test_a_meta_layer_publishes_the_accelerate_execution_device():
    """AlignDevicesHook sends the layer's inputs there, so that is where they belong."""
    layer = _FakeLayer("meta")
    layer._hf_hook = SimpleNamespace(execution_device = "cpu")
    verify_and_set_device(layer)

    assert layer._per_layer_device == torch.device("cpu")
    assert layer._per_layer_device_index == "cpu"

    layer = _FakeLayer("meta")
    layer._hf_hook = SimpleNamespace(execution_device = 2)
    verify_and_set_device(layer)

    assert layer._per_layer_device == torch.device(2)
    assert layer._per_layer_device_index == 2


@pytest.mark.parametrize("execution_device", [None, "meta", "not-a-device", object()])
def test_a_hook_that_cannot_name_a_device_publishes_nothing(execution_device):
    """execution_device is optional and is itself "meta" mid-build."""
    layer = _FakeLayer("meta")
    layer._hf_hook = SimpleNamespace(execution_device = execution_device)
    verify_and_set_device(layer)

    assert not hasattr(layer, "_per_layer_device")


def test_a_stale_pair_is_cleared_when_the_layer_goes_back_to_meta():
    layer = _FakeLayer("cpu")
    verify_and_set_device(layer)
    assert layer._per_layer_device == torch.device("cpu")

    layer._devices = ("meta",)
    verify_and_set_device(layer)
    assert not hasattr(layer, "_per_layer_device")
    assert not hasattr(layer, "_per_layer_device_index")


def test_the_non_meta_placements_are_untouched():
    for device_string, expected in (("cuda:3", 3), ("cpu", "cpu"), ("xpu:1", 1)):
        layer = _FakeLayer(device_string)
        verify_and_set_device(layer)
        assert layer._per_layer_device == torch.device(device_string)
        assert layer._per_layer_device_index == expected


def test_a_chained_hook_is_unwrapped_to_the_alignment_device():
    """`append=True` stores `SequentialHook(old, new)`, which defines no
    `execution_device`, so the outer hook alone answered None and readers fell to device 0."""
    outer = SimpleNamespace(
        hooks = (
            SimpleNamespace(),
            SimpleNamespace(execution_device = 1),
        )
    )
    layer = _FakeLayer("meta")
    layer._hf_hook = outer
    verify_and_set_device(layer)

    assert layer._per_layer_device == torch.device(1)
    assert layer._per_layer_device_index == 1


def test_a_nested_chain_is_followed_and_the_outermost_real_device_wins():
    """The first hook naming a REAL device wins; a meta one does not end the search."""
    layer = _FakeLayer("meta")
    layer._hf_hook = SimpleNamespace(
        hooks = (
            SimpleNamespace(
                hooks = (
                    SimpleNamespace(execution_device = "meta"),
                    SimpleNamespace(execution_device = "cpu"),
                )
            ),
            SimpleNamespace(execution_device = 3),
        )
    )
    verify_and_set_device(layer)

    assert layer._per_layer_device == torch.device("cpu")


def test_a_chain_that_names_no_device_still_publishes_nothing():
    layer = _FakeLayer("meta")
    layer._hf_hook = SimpleNamespace(
        hooks = (
            SimpleNamespace(execution_device = "meta"),
            SimpleNamespace(execution_device = None),
            SimpleNamespace(),
        )
    )
    verify_and_set_device(layer)

    assert not hasattr(layer, "_per_layer_device")
    assert not hasattr(layer, "_per_layer_device_index")


def test_a_self_referential_hook_chain_terminates():
    loop = SimpleNamespace()
    loop.hooks = (loop,)
    layer = _FakeLayer("meta")
    layer._hf_hook = loop

    verify_and_set_device(layer)
    assert not hasattr(layer, "_per_layer_device")
