# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Decoders with no language / text tag (CSM's backbone_model / depth_decoder) still get LoRA targets PEFT accepts."""
import re

import torch.nn as nn

from unsloth_zoo.peft_utils import get_peft_regex


class _Attn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)


class _Mlp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up_proj = nn.Linear(dim, dim)
        self.down_proj = nn.Linear(dim, dim)


class _Layer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.self_attn = _Attn(dim)
        self.mlp = _Mlp(dim)


class _Stack(nn.Module):
    def __init__(self, dim, n = 2):
        super().__init__()
        self.layers = nn.ModuleList(_Layer(dim) for _ in range(n))


class _CsmLike(nn.Module):
    """CsmForConditionalGeneration's layout: backbone_model.layers.N and depth_decoder.model.layers.N."""
    def __init__(self, dim = 8):
        super().__init__()
        self.backbone_model = _Stack(dim)
        self.depth_decoder = nn.Module()
        self.depth_decoder.model = _Stack(dim)
        self.lm_head = nn.Linear(dim, 4)

    class config:
        _name_or_path = "test/csm-like"
        model_type = "csm"


def test_untagged_decoders_get_peft_targets():
    model = _CsmLike()
    regex = get_peft_regex(model)
    linears = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    # PEFT matches a string target_modules with re.fullmatch.
    targets = {n for n in linears if re.fullmatch(regex, n)}
    want = {n for n in linears if n != "lm_head"}
    assert targets == want


def test_frozen_codec_is_not_a_target():
    model = _CsmLike()
    model.codec_model = nn.Module()
    model.codec_model.encoder = _Stack(8)
    model.codec_model.encoder.layers[0].mlp.fc1 = nn.Linear(8, 8)
    regex = get_peft_regex(model)
    targets = {n for n, m in model.named_modules() if isinstance(m, nn.Linear) and re.fullmatch(regex, n)}
    assert targets and not any(n.startswith("codec_model.") for n in targets)
    assert any(n.startswith("backbone_model.") for n in targets)
    assert any(n.startswith("depth_decoder.") for n in targets)
