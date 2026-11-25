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

"""transformers 5.24 generate hands the CSM depth decoder its cache as `cache=` (renamed from past_key_values): the
patched depth decoder forwards must read it there, or codebook positions restart at 0 every step; and the
CsmOutputWithPast cache fields (depth_decoder_cache on 5.24) must keep their values."""

import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("transformers.models.csm.modeling_csm")

_RUNTIME = r"""
import json, os
os.environ["UNSLOTH_IS_PRESENT"] = "1"
import torch
import transformers.models.csm.modeling_csm as M
from transformers import CsmDepthDecoderConfig, DynamicCache

config = CsmDepthDecoderConfig(
    hidden_size = 32, intermediate_size = 64, num_hidden_layers = 2, num_attention_heads = 2,
    num_key_value_heads = 1, head_dim = 16, num_codebooks = 6, vocab_size = 64, backbone_hidden_size = 32,
    max_position_embeddings = 8,
)
torch.manual_seed(0)
model = M.CsmDepthDecoderForCausalLM._from_config(config, dtype = torch.float32).eval()
backbone = torch.randn(2, 32, generator = torch.Generator().manual_seed(1))
ids = torch.randint(1, 64, (2, 5), generator = torch.Generator().manual_seed(2))
ids[:, 0] = 0

import inspect
# transformers 4.x generate passes cache_position; 5.x derives positions from the cache.
legacy = "cache_position" in inspect.signature(M.CsmDepthDecoderModel.forward).parameters

def positions(start, end):
    return {"cache_position": torch.arange(start, end)} if legacy else {}

def decode(keyword):
    cache = DynamicCache(config = config)
    with torch.no_grad():
        out = model(input_ids = ids[:, :2], backbone_last_hidden_state = backbone, use_cache = True,
                    **positions(0, 2), **{keyword: cache})
        logits = [out.logits[:, -1]]
        for t in range(2, ids.shape[1]):
            out = model(input_ids = ids[:, t : t + 1], use_cache = True, logits_to_keep = 1,
                        **positions(t, t + 1), **{keyword: cache})
            logits.append(out.logits[:, -1])
    return torch.stack(logits, 1)

keyword = "cache" if "cache" in inspect.signature(M.CsmDepthDecoderForCausalLM.forward).parameters else "past_key_values"
reference = decode(keyword)
from unsloth_zoo.temporary_patches import misc
misc.patch_CsmDepthDecoderForCausalLM_forward()
misc.patch_CsmDepthDecoderModel_forward()
res = {
    "keyword": keyword,
    "patched": getattr(M.CsmDepthDecoderForCausalLM.forward, "__module__", "") == misc.__name__,
    "stock_keyword": float((decode(keyword) - reference).abs().max()),
    "past_key_values": float((decode("past_key_values") - reference).abs().max()),
}
from unsloth_zoo.temporary_patches.utils import process_return
from transformers.models.csm.modeling_csm import CsmOutputWithPast
backbone_cache, depth_cache = DynamicCache(config = config), DynamicCache(config = config)
out = process_return(CsmOutputWithPast, {
    "logits": torch.zeros(1), "past_key_values": backbone_cache, "depth_decoder_past_key_values": depth_cache,
})
fields = CsmOutputWithPast.__dataclass_fields__
res["backbone_cache_kept"] = out["cache" if "cache" in fields else "past_key_values"] is backbone_cache
res["depth_cache_kept"] = out[
    "depth_decoder_cache" if "depth_decoder_cache" in fields else "depth_decoder_past_key_values"
] is depth_cache
print("RESULT " + json.dumps(res))
"""


def test_patched_depth_decoder_reads_the_cache_under_either_name():
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join(p for p in (env.get("PYTHONPATH"), root) if p)
    proc = subprocess.run([sys.executable, "-c", _RUNTIME], capture_output = True, text = True, timeout = 600, env = env)
    lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, f"exit={proc.returncode}\n{proc.stdout[-3000:]}\n{proc.stderr[-6000:]}"
    res = json.loads(lines[-1][len("RESULT "):])
    assert res["patched"], res
    assert res["stock_keyword"] < 1e-5, res
    assert res["past_key_values"] < 1e-5, res
    assert res["backbone_cache_kept"] and res["depth_cache_kept"], res
