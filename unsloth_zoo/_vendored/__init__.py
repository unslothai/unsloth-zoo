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

"""Vendored third-party components bundled with Unsloth Zoo.

Each subpackage keeps its own upstream LICENSE and a MANIFEST recording the
source project, pinned version, tag, upstream URL and the exported API. See the
repo-root NOTICE for the summary list. These components are shipped verbatim
(only their package __init__ exports are narrowed) so Unsloth can offer the fast
kernel paths without requiring an extra `pip install`.

Currently vendored:
  * fla  -  flash-linear-attention (fla-core) 0.5.1, MIT. The minimal
            gated-delta-rule kernel closure used by the Qwen3.5 / Qwen3.6 /
            Qwen3-Next gated-deltanet models.
"""
