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

"""Hugging Face Hub download helpers that import without ``unsloth_zoo``'s package init.

``unsloth_zoo/__init__`` imports torch, transformers and the temporary patches, which costs
seconds and hundreds of megabytes that a download-only caller never uses. Every module here
is stdlib-only at import time, so a backend can take the Xet health verdict, the RAM-derived
buffer caps and the stall watchdog without that init, and so can the download child it spawns.
The ``unsloth_zoo.<module>`` names stay importable and resolve to these same module objects.
Importing from here also skips the init's environment steps (the read-only cache redirect,
``apply_xet_env``, the offline and stable-download flags), so a process that never imports
``unsloth_zoo`` applies the ones it needs itself.
"""
