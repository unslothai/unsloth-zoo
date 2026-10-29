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

"""torch >= 2.12 keeps torch config assignments per thread (ContextVar), so values set at import are
lost on autograd's device thread (reentrant checkpoint recompute + its compiles). Publish them to all threads."""

__all__ = [
    "set_torch_config",
    "publish_torch_config",
    "torch_config_is_per_thread",
]

import sys
from contextvars import ContextVar

_CACHE_VARS = (
    ("_hash_dirty_var", True),
    ("_hash_cache_var", None),
    ("_get_dict_dirty_keys_var", None),
    ("_get_dict_cache_var", None),
)


def _entry(config_module, name):
    config = getattr(config_module, "_config", None)
    if not isinstance(config, dict):
        return None
    return config.get(name)


def torch_config_is_per_thread(config_module, name = None):
    """True when assignments to `config_module` are stored per thread (torch >= 2.12)."""
    config = getattr(config_module, "_config", None)
    if not isinstance(config, dict) or not config:
        return False
    entry = config.get(name) if name is not None else next(iter(config.values()))
    return isinstance(getattr(entry, "user_override", None), ContextVar)


def _invalidate_caches(config_module):
    if hasattr(config_module, "_is_dirty"):
        config_module._is_dirty = True  # torch 2.12: one shared hash cache
    module_name = getattr(config_module, "__name__", "torch_config")
    for attr, default in _CACHE_VARS:
        if isinstance(getattr(config_module, attr, None), ContextVar):
            setattr(config_module, attr, ContextVar(f"{module_name}.{attr}", default = default))


def _publish(config_module, name, value, force, depth = 0):
    entry = _entry(config_module, name)
    if entry is None or not isinstance(getattr(entry, "user_override", None), ContextVar):
        return False
    alias = getattr(entry, "alias", None)
    if alias is not None:
        # e.g. dynamo cache_size_limit -> recompile_limit: publish on the target entry.
        module_name, _, target = alias.rpartition(".")
        target_module = sys.modules.get(module_name)
        if target_module is None or depth > 4:
            return False
        return _publish(target_module, target, value, force, depth + 1)
    if force:
        entry.env_value_force = value
    else:
        entry.env_value_default = value
    _invalidate_caches(config_module)
    return True


def publish_torch_config(config_module, name, force = False):
    """Publish this thread's `config_module.name` to every thread; False if already shared or missing.

    force = True also overrides other threads' assignments and `config.patch`."""
    if _entry(config_module, name) is None:
        return False
    return _publish(config_module, name, getattr(config_module, name), force)


def set_torch_config(config_module, name, value, force = False):
    """`setattr(config_module, name, value)` for every thread."""
    setattr(config_module, name, value)
    _publish(config_module, name, value, force)
    return value
