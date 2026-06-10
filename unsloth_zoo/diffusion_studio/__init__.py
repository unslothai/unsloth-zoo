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
rather than the autoregressive server. This package wraps that runner in an OpenAI-compatible shim so
Studio can talk to it as an ordinary llama.cpp / OpenAI-compatible Connection - an additive path that
leaves the autoregressive flows untouched. See the README for the end-to-end setup.

The engine (LlamaServer / Tok / generate) imports cleanly; the HTTP shim pulls in fastapi/uvicorn only
when you build or run it, so importing this package stays dependency-light.
"""
from .engine import LlamaServer, Tok, generate, VOCAB, CANVAS, EOS_IDS, DEFAULTS

__all__ = ["LlamaServer", "Tok", "generate", "VOCAB", "CANVAS", "EOS_IDS", "DEFAULTS"]
