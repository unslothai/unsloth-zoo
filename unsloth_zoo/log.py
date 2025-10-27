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

__all__ = [
    "UNSLOTH_ENABLE_LOGGING",
    "logger",
]

import logging
import os

UNSLOTH_ENABLE_LOGGING = os.environ.get("UNSLOTH_ENABLE_LOGGING",  "0") in ("1", "True", "true",)

logger = logging.getLogger(__name__)
if UNSLOTH_ENABLE_LOGGING:
    logging.basicConfig(level = logging.INFO, format = '[%(name)s|%(levelname)s]%(message)s')
    logger.setLevel(logging.INFO)
else:
    logging.basicConfig(level = logging.WARNING, format = '[%(name)s|%(levelname)s]%(message)s')
    logger.setLevel(logging.WARNING) 
pass
