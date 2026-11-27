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

"""DeepSeek-V4.1 test implementations, loaded without unsloth: no GPU init, no global patches."""

import importlib
import importlib.util
import os
import sys
import types

NATIVE_ALIAS = "_unsloth_native_test.deepseek_v41"


def port():
    """transformers' (community port) deepseek_v41, or None."""
    try:
        if importlib.util.find_spec("transformers.models.deepseek_v41") is None:
            return None
        from transformers.models.deepseek_v41 import configuration_deepseek_v41 as configuration
        from transformers.models.deepseek_v41 import modeling_deepseek_v41 as modeling
    except Exception:
        return None
    return types.SimpleNamespace(
        name = "port", modeling = modeling, configuration = configuration,
        TextConfig = configuration.DeepseekV41TextConfig, register = lambda: None,
    )


def native():
    """Unsloth's native deepseek_v41, or None. Only used when transformers has no deepseek_v41 of its own."""
    if port() is not None:
        return None
    package = sys.modules.get("unsloth.models.deepseek_v41") or sys.modules.get(NATIVE_ALIAS)
    if package is None:
        try:
            spec = importlib.util.find_spec("unsloth")
        except Exception:
            spec = None
        locations = list(getattr(spec, "submodule_search_locations", None) or [])
        directory = os.path.join(locations[0], "models", "deepseek_v41") if locations else ""
        if not os.path.isfile(os.path.join(directory, "modeling.py")):
            return None
        root = NATIVE_ALIAS.split(".")[0]
        if root not in sys.modules:
            sys.modules[root] = types.ModuleType(root)
        spec = importlib.util.spec_from_file_location(
            NATIVE_ALIAS, os.path.join(directory, "__init__.py"), submodule_search_locations = [directory]
        )
        package = importlib.util.module_from_spec(spec)
        sys.modules[NATIVE_ALIAS] = package
        try:
            spec.loader.exec_module(package)
        except Exception:
            sys.modules.pop(NATIVE_ALIAS, None)
            return None
        setattr(sys.modules[root], NATIVE_ALIAS.split(".")[1], package)
    modeling = importlib.import_module(package.__name__ + ".modeling")
    configuration = importlib.import_module(package.__name__ + ".configuration")
    return types.SimpleNamespace(
        name = "native", modeling = modeling, configuration = configuration,
        TextConfig = configuration.DeepseekV41TextConfig, register = package.register_deepseek_v41,
    )


def available():
    """[implementation, ...] to run the shared tests against."""
    return [impl for impl in (port(), native()) if impl is not None]
