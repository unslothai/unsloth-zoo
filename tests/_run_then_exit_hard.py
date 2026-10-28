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

"""Runs pytest then os._exit to skip finalization: mlx 0.32.1 segfaults at shutdown when a fused Metal kernel was the
last thing touched (0.32.0 is clean), not fixable in Python; pytest's exit code is kept so failures still fail; drop
once mlx ships a fix."""

import os
import sys

import pytest


def main():
    code = pytest.main(sys.argv[1:])
    code = int(getattr(code, "value", code))
    sys.stdout.flush()
    sys.stderr.flush()
    # Not sys.exit: that unwinds and finalizes, which is the part that crashes.
    os._exit(code)


if __name__ == "__main__":
    main()
