# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""mlx_lm.stream_generate / mlx_vlm.stream_generate — stub yields token chunks.

Tier-1 stub: yields N synthetic token objects with `.token`, `.text`,
`.finish_reason` attributes.  Sufficient for testing PR-B
`_run_inference_request` plumbing without a real model.

Tier-2 (real torch generation) gated behind UNSLOTH_MLX_SIM_REAL_GENERATE=1
delegates to transformers.generate.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class _GenerationResponse:
    token: int
    text: str
    finish_reason: str | None = None


def stream_generate(model, tokenizer, prompt, *,
                    max_tokens: int = 100,
                    sampler=None,
                    logits_processors=None,
                    **kwargs):
    """Yield generation chunks one token at a time."""
    if os.environ.get("UNSLOTH_MLX_SIM_REAL_GENERATE", "0") == "1":
        yield from _real_stream_generate(model, tokenizer, prompt,
                                         max_tokens=max_tokens, **kwargs)
        return

    # Stub: yield N short tokens then stop
    for i in range(min(max_tokens, 8)):
        yield _GenerationResponse(token=100 + i, text=f" tok{i}")
    yield _GenerationResponse(token=2, text="", finish_reason="stop")


def vlm_stream_generate(model, processor, prompt, image=None, *,
                        max_tokens: int = 100,
                        **kwargs):
    """VLM stream — same stub shape."""
    if os.environ.get("UNSLOTH_MLX_SIM_REAL_GENERATE", "0") == "1":
        yield from _real_stream_generate(model, processor, prompt,
                                         max_tokens=max_tokens, **kwargs)
        return
    for i in range(min(max_tokens, 8)):
        yield _GenerationResponse(token=200 + i, text=f" img-tok{i}")
    yield _GenerationResponse(token=2, text="", finish_reason="stop")


def _real_stream_generate(model, tokenizer, prompt, *, max_tokens, **kwargs):
    """Tier-2 real generation via transformers.generate."""
    raise NotImplementedError(
        "mlx-shim: UNSLOTH_MLX_SIM_REAL_GENERATE=1 path not implemented yet "
        "(Phase 5)."
    )
