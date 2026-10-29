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

"""Torch config writes that every thread sees.

torch >= 2.12 keeps `torch._inductor.config.X = Y` (and the dynamo / functorch /
compiler configs) in a per-thread ContextVar, so a value set at import is lost on
autograd's device thread. A reentrant checkpoint's recompute runs there, and so do
the compiles it triggers (the grad-mode graph and its lazily compiled backward).
`env_value_default` is a plain attribute read by every thread that has no
assignment of its own, so it is published there too.
"""

__all__ = [
    "set_torch_config",
    "publish_torch_config",
    "torch_config_is_per_thread",
]

from contextvars import ContextVar

# ContextVars that cache the config hash / dict per thread (torch >= 2.13).
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
    # Every thread must rehash: fresh ContextVars start dirty in all of them.
    if hasattr(config_module, "_is_dirty"):
        config_module._is_dirty = True  # torch 2.12: one shared hash cache
    module_name = getattr(config_module, "__name__", "torch_config")
    for attr, default in _CACHE_VARS:
        if isinstance(getattr(config_module, attr, None), ContextVar):
            setattr(config_module, attr, ContextVar(f"{module_name}.{attr}", default = default))


def _publish(config_module, name, value, force):
    entry = _entry(config_module, name)
    if entry is None or not isinstance(getattr(entry, "user_override", None), ContextVar):
        return False
    if getattr(entry, "alias", None) is not None:
        return False  # aliases write a module attribute, shared by all threads
    if force:
        entry.env_value_force = value
    else:
        entry.env_value_default = value
    _invalidate_caches(config_module)
    return True


def publish_torch_config(config_module, name, force = False):
    """Make the current thread's value of `config_module.name` the value of every thread.

    force = False writes `env_value_default`: threads without an assignment of their own
    (autograd device threads, new threads) read it, and `config.patch` still wins.
    force = True writes `env_value_force`, which wins over every thread's assignment and
    over `config.patch`. Returns True when the value was published, False on torch builds
    whose config is shared by all threads already (nothing to do) or that lack the entry.
    """
    if _entry(config_module, name) is None:
        return False
    return _publish(config_module, name, getattr(config_module, name), force)


def set_torch_config(config_module, name, value, force = False):
    """`setattr(config_module, name, value)` for every thread, not only the current one.

    No-op beyond the plain assignment on torch < 2.12, where it is already global.
    """
    setattr(config_module, name, value)
    _publish(config_module, name, value, force)
    return value
