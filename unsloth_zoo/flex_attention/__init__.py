# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from .utils import (
    HAS_FLEX_ATTENTION,
    FLEX_ATTENTION_BLOCK_SIZE,
    flex_attention,
    create_block_mask_cached,
    causal_mask,
    generate_sliding_window_mask,
)

if HAS_FLEX_ATTENTION:
    from .attention_sink import (
        old_flex_attention_with_sink,
        flex_attention_with_sink,
        is_flex_attention_decoding,
        flex_attention_with_sink_decoding,
        flex_attention_add_sinks,
        flash_attention_left_padded,
    )
else:
    old_flex_attention_with_sink = None
    flex_attention_with_sink = None
