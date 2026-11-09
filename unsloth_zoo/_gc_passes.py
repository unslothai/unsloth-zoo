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

import gc

__all__ = ["collect_garbage"]


def collect_garbage(passes = 3, after = None):
    """Up to `passes` gc.collect() passes, calling `after` after each; stops after a pass returning 0 unless
    gc.callbacks are registered, gc.collect returned None (vLLM stub), or gen-2 stats did not move by exactly one
    quiet collection; resurrecting finalizers return 0, so garbage waits for the next pass."""
    for _ in range(passes):
        seen = gc.get_stats()[2]
        found = gc.collect()
        ran = gc.get_stats()[2]
        if after is not None:
            after()
        if (
            found == 0
            and not gc.callbacks
            and ran["collections"] == seen["collections"] + 1
            and ran["collected"] == seen["collected"]
            and ran["uncollectable"] == seen["uncollectable"]
        ):
            break
