# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = [
    "is_mlx_available",
]

import functools
import importlib.util
import os
import platform


@functools.cache
def is_mlx_available() -> bool:
    return (
        os.environ.get("UNSLOTH_FORCE_GPU_PATH", "0") != "1"
        and platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and importlib.util.find_spec("mlx") is not None
    )


_FOLDER_BUILDER = "datasets.packaged_modules.folder_based_builder.folder_based_builder"


def _load_folder_metadata_patch():
    """Loads the folder metadata patch alone: __init__ imports torch and GPU kernels MLX lacks."""
    import sys
    import types

    module_name = "unsloth_zoo.mlx._guard_datasets_folder_metadata"
    module = sys.modules.get(module_name)
    if module is None:
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temporary_patches")
        spec = importlib.util.spec_from_file_location(module_name, os.path.join(root, "datasets_folder_metadata.py"))
        if spec is None:
            return None
        common_name = "unsloth_zoo.mlx.common"
        common = types.ModuleType(common_name)
        common.TEMPORARY_PATCHES = []
        previous = sys.modules.get(common_name)
        sys.modules[common_name] = common
        try:
            module = importlib.util.module_from_spec(spec)
            module.__package__ = "unsloth_zoo.mlx"
            spec.loader.exec_module(module)
        except Exception:
            return None
        finally:
            if previous is None:
                sys.modules.pop(common_name, None)
            else:
                sys.modules[common_name] = previous
        sys.modules[module_name] = module
    return getattr(module, "patch_datasets_folder_metadata_file_name", None)


class _FolderBuilderGuardFinder:
    """Installed right after datasets loads its folder builder, so MLX startup skips datasets."""

    _unsloth_folder_guard = True

    def __init__(self, patch):
        self.patch = patch

    def find_spec(self, fullname, path = None, target = None):
        if fullname != _FOLDER_BUILDER:
            return None
        import sys

        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                break
        else:
            return None
        loader = getattr(spec, "loader", None)
        exec_module = getattr(loader, "exec_module", None)
        if exec_module is None:
            return spec
        patch = self.patch

        class _Loader:
            def __getattr__(self, name):
                return getattr(loader, name)

            def create_module(self, spec):
                return loader.create_module(spec)

            def exec_module(self, module):
                exec_module(module)
                try:
                    patch()
                except Exception:
                    pass

        spec.loader = _Loader()
        return spec


def install_mlx_security_patches():
    """Device-independent guards the GPU and CPU paths get from temporary_patches; idempotent."""
    import sys

    patch = _load_folder_metadata_patch()
    if patch is None:
        return
    if _FOLDER_BUILDER in sys.modules:
        try:
            patch()
        except Exception:
            pass
        return
    if not any(getattr(finder, "_unsloth_folder_guard", False) for finder in sys.meta_path):
        sys.meta_path.insert(0, _FolderBuilderGuardFinder(patch))
