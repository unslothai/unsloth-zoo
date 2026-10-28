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

"""Full gc.collect() passes that stop once a pass that really ran finds nothing. No torch import."""

import gc

__all__ = ["collect_garbage"]


def collect_garbage(passes = 3, after = None):
    """Up to `passes` full gc.collect() passes, `after` run after each.

    Stops after a pass that ran and found nothing. Keeps going while gc.callbacks are registered (they
    may drop references) or when the gen 2 stats do not show one more empty collection (CPython returns 0
    without collecting while another thread is collecting).
    """
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
