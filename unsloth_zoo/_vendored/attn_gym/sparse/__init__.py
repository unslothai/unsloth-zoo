# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Modified by Unsloth: narrowed to gather_attn (the lightning indexer is not vendored).
"""Sparse attention primitives."""

# Modified by Unsloth: absolute import (transformers' custom_object_save opens relative targets as files).
from unsloth_zoo._vendored.attn_gym.sparse.gather_attn import AuxRequest, GatherAttnAux, Impl, RopePair, gather_attn

__all__ = ["AuxRequest", "GatherAttnAux", "Impl", "RopePair", "gather_attn"]
