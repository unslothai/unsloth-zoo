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

"""requests < 2.32.4 can send ~/.netrc credentials to another host for a crafted URL (CVE-2024-47081), and media
fetches take URLs from datasets: every dependency list that pulls requests must floor it."""
import pytest

try:
    import tomllib
except ImportError:  # Python < 3.11
    tomllib = pytest.importorskip("tomli")
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _requests_specs():
    data = tomllib.loads(PYPROJECT.read_text())
    project = data["project"]
    lists = [project.get("dependencies", [])] + list(project.get("optional-dependencies", {}).values())
    return [Requirement(r) for deps in lists for r in deps if Requirement(r).name == "requests"]


def test_requests_floor_excludes_netrc_leak():
    specs = _requests_specs()
    assert specs
    for req in specs:
        assert not req.specifier.contains(Version("2.32.3")), str(req)
        assert req.specifier.contains(Version("2.32.4")), str(req)
