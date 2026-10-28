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

"""The bnb compute-dtype override must reach vLLM's quant registry: in-tree vLLM <= 0.27.1 re-imports BitsAndBytesConfig
per call, so a swap works; a plugin's class object in _CUSTOMIZED_METHOD_TO_QUANT_CONFIG hides it and
UNSLOTH_bnb_4bit_compute_dtype is ignored."""

import importlib.machinery
import sys
import types

# Import before any stub lands in sys.modules: unsloth_zoo.vllm_utils calls importlib.util.find_spec for vllm at import time, which raises on a stub.
from unsloth_zoo.vllm_utils import _set_registered_quant_config


def _stub(name):
    module = types.ModuleType(name)
    # find_spec() raises ValueError on a module whose __spec__ is None.
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader = None)
    return module


def _helper(monkeypatch, registry):
    """Stub `vllm.model_executor.layers.quantization` holding `registry`."""
    quant = _stub("vllm.model_executor.layers.quantization")
    quant._CUSTOMIZED_METHOD_TO_QUANT_CONFIG = registry
    for name in (
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.quantization",
    ):
        monkeypatch.setitem(
            sys.modules, name, quant if name.endswith("quantization") else _stub(name)
        )
    return _set_registered_quant_config


def test_registered_config_is_swapped_and_restored(monkeypatch):
    class _Original: pass
    class _Patched: pass

    registry = {"bitsandbytes": _Original}
    set_config = _helper(monkeypatch, registry)

    set_config("bitsandbytes", _Patched)
    assert registry["bitsandbytes"] is _Patched, (
        "the plugin's registered class must be replaced, or vLLM keeps building "
        "the unpatched config and ignores UNSLOTH_bnb_4bit_compute_dtype"
    )

    set_config("bitsandbytes", _Original)
    assert registry["bitsandbytes"] is _Original, "restore must put the original class back"


def test_an_unregistered_method_is_left_alone(monkeypatch):
    # In-tree vLLM registers nothing here; inventing a registry key would shadow the call-time import that path relies on.
    registry = {}
    set_config = _helper(monkeypatch, registry)

    set_config("bitsandbytes", object)
    assert registry == {}, "must not create an entry vLLM never registered"


def test_the_missing_plugin_error_is_raised_where_4bit_is_decided():
    """The missing-plugin install hint must be raised where 4-bit is decided: patch_vllm_bitsandbytes never installs the
    no-bnb `_apply_4bit_weight`, so vLLM >= 0.28 gave a generic error."""
    import inspect

    from unsloth_zoo import vllm_utils

    source = inspect.getsource(vllm_utils.load_vllm)
    assert "vllm-bnb-plugin" in source, (
        "load_vllm must raise the install hint once it knows bitsandbytes is "
        "wanted; an unreachable stub leaves users with vLLM's generic error"
    )
    assert "_vllm_bnb is None" in source, "the hint must be gated on the resolution failing"
