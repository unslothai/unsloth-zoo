# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
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
"""A removed PARENT package is upstream drift: ModuleNotFoundError.name is the deepest missing
package, so matching only the full path or top-level name skipped it and let a hard gate pass."""

import importlib
import pathlib
import sys

import pytest

# tests/ is not a package; unsloth's tests/__init__.py made CI resolve a sibling's package.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from test_temporary_patches_exhaustive import (  # noqa: E402
    _names_the_target as _names_the_target_patches,
)
from test_upstream_signatures import (  # noqa: E402
    _names_the_target as _names_the_target_signatures,
)

PREDICATES = pytest.mark.parametrize(
    "predicate",
    [_names_the_target_patches, _names_the_target_signatures],
    ids = ["temporary_patches_exhaustive", "upstream_signatures"],
)


def _module_not_found(name):
    """A ModuleNotFoundError carrying `name`, built the way Python builds it."""
    return ModuleNotFoundError(f"No module named {name!r}", name = name)


def test_a_missing_parent_package_really_is_reported_as_the_parent():
    """Pins the premise on the live interpreter: ModuleNotFoundError.name is the parent package; if CPython reported the full path, the fix would be moot."""
    with pytest.raises(ModuleNotFoundError) as excinfo:
        importlib.import_module("transformers.models.unsloth_no_such_pkg.modeling_x")
    assert excinfo.value.name == "transformers.models.unsloth_no_such_pkg"


@PREDICATES
def test_removed_parent_package_counts_as_the_target(predicate):
    """The reported bug: this was False, so the caller skipped instead of failing."""
    exc = _module_not_found("transformers.models.siglip")
    assert predicate(exc, "transformers.models.siglip.modeling_siglip"), (
        "a removed parent package must be reported as the target going away, "
        "not as a broken dependency"
    )


@PREDICATES
@pytest.mark.parametrize(
    "missing",
    ["transformers.models.siglip.modeling_siglip", "transformers.models", "transformers"],
)
def test_every_package_prefix_counts_as_the_target(predicate, missing):
    assert predicate(_module_not_found(missing), "transformers.models.siglip.modeling_siglip")


@PREDICATES
@pytest.mark.parametrize("missing", ["timm", "timm.data", "torchvision"])
def test_a_broken_dependency_still_does_not_count(predicate, missing):
    """Broken deps are not drift: gemma3n imports ImageNetInfo, which newer timm dropped."""
    assert not predicate(_module_not_found(missing), "transformers.models.gemma3n.configuration_gemma3n")


@PREDICATES
def test_a_prefix_that_is_not_a_package_boundary_does_not_count(predicate):
    """`startswith` without the trailing dot would claim this one."""
    assert not predicate(
        _module_not_found("transformers.models.siglip"),
        "transformers.models.siglipx.modeling_x",
    )


@PREDICATES
def test_an_unnamed_module_not_found_does_not_count(predicate):
    """`exc.name` is Optional; an empty name must not prefix-match everything."""
    assert not predicate(_module_not_found(None), "transformers.models.siglip.modeling_siglip")
