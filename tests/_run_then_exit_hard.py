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

"""Run pytest, then exit without interpreter finalization.

mlx 0.32.1 segfaults at interpreter shutdown when a fused Metal custom kernel is
the last thing a process touched, so a fully green suite dies with exit 139.

Measured on an Apple Silicon runner, three samples each, using the shape from
tests/test_qwen35_vjp_metal.py::test_disable_fused_mrope_fixes_rotary_grad:

    build model + fused rotary forward                exit 139, 139, 139
      + del refs, gc.collect(), mx.synchronize()      exit 139, 139, 139
      + gc.collect() registered at exit               exit 139, 139, 139
      + os._exit() after the work                     exit   0,   0,   0

Not fixable from Python: the fault is in mlx's own static destruction, after any
cleanup we can do. mlx 0.32.0 is clean on the same tree, dating it to 0.32.1.

Exits with pytest's own return code, so a failing test still fails the job; only
finalization is skipped. Drop this wrapper once mlx ships a fix.
"""

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
