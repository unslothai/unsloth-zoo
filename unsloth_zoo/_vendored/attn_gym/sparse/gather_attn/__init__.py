# Modified by Unsloth: package-absolute import of the vendored attn_gym.types.
"""Gather attention over explicit KV indices and an optional causal local window."""

from unsloth_zoo._vendored.attn_gym.types import Impl

from .api import AuxRequest, GatherAttnAux, RopePair, gather_attn

__all__ = ["AuxRequest", "GatherAttnAux", "Impl", "RopePair", "gather_attn"]
