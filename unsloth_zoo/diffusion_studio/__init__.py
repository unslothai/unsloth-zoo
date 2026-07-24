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
"""Run DiffusionGemma in Unsloth Studio over the GGUF path.

DiffusionGemma is a block-diffusion model, so llama.cpp serves it through a dedicated diffusion runner
rather than the autoregressive server. This package drives the optimized on-device visual decoder
(llama-diffusion-gemma-visual-server) and wraps it in an OpenAI-compatible shim so Unsloth can serve it
as an ordinary llama.cpp / OpenAI-compatible model. The shim streams the committed answer text and a
self-contained ```html artifact that replays the per-step denoising canvas, which Unsloth auto-renders for
DiffusionGemma. Autoregressive flows are untouched.

The visual server tokenizes, applies the chat template and detokenizes from the GGUF's own embedded
tokenizer, so no tokenizer files are needed here. visual_engine imports cleanly; the HTTP shim pulls in
fastapi/uvicorn only when run, so importing this package stays dependency-light.
"""
import os

from .visual_engine import VisualServer, generate_visual, CANVAS


def canvas_player_path():
    """Absolute path to the self-contained denoising canvas player template (bundled with this package)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "canvas_player.html")


__all__ = ["VisualServer", "generate_visual", "CANVAS", "canvas_player_path"]
