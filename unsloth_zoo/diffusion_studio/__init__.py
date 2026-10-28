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
"""Runs DiffusionGemma in Unsloth Studio over GGUF via llama-diffusion-gemma-visual-server behind an OpenAI-compatible
shim; no tokenizer files are needed, fastapi/uvicorn load only when the shim runs; autoregressive flows are
untouched."""
import os

from .visual_engine import VisualServer, generate_visual, CANVAS


def canvas_player_path():
    """Absolute path to the self-contained denoising canvas player template (bundled with this package)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "canvas_player.html")


__all__ = ["VisualServer", "generate_visual", "CANVAS", "canvas_player_path"]
