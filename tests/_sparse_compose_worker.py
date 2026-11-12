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

"""python _sparse_compose_worker.py {skip_first|flex_first} {model_type}: checks skip and flex patches compose."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_sparse_attention_flex as T  # noqa: E402
from unsloth_zoo.flex_attention import sparse as gates  # noqa: E402
from unsloth_zoo.temporary_patches import sparse_attention as skip  # noqa: E402
from unsloth_zoo.temporary_patches import sparse_attention_flex as flex  # noqa: E402

order, family = sys.argv[1], sys.argv[2]
install_skip = {
    "deepseek_v32": skip.patch_deepseek_v32_sparse_skip,
    "glm_moe_dsa": skip.patch_glm_moe_dsa_sparse_skip,
    "minimax_m3_vl": skip.patch_minimax_m3_sparse_skip,
}[family]
install_flex = flex.patch_minimax_m3_sparse_flex if family == "minimax_m3_vl" else flex.patch_dsa_sparse_flex
steps = [install_skip, install_flex] if order == "skip_first" else [install_flex, install_skip]
for fn in steps + steps:
    fn()

modeling = __import__(f"transformers.models.{family}.modeling_{family}", fromlist = ["x"])
indexer_cls = getattr(modeling, {"deepseek_v32": "DeepseekV32Indexer", "glm_moe_dsa": "GlmMoeDsaIndexer",
                                 "minimax_m3_vl": "MiniMaxM3VLIndexer"}[family])
assert getattr(indexer_cls.forward, "_unsloth_sparse_flex", False), "sparse flex not installed"
key = f"{family}.indexer"
assert skip._is_transformers_function(skip._REFERENCE[key], family), "_REFERENCE is not upstream"
assert indexer_cls._unsloth_original_forward is skip._REFERENCE[key]
assert indexer_cls._unsloth_inner_forward is not skip._REFERENCE[key], "skip not composed"

cuda = torch.cuda.is_available() and torch.version.hip is None and torch.cuda.get_device_capability() >= (8, 0)
device = "cuda" if cuda else "cpu"
if family == "minimax_m3_vl":
    config = T._minimax_config(num_attention_heads = 8, num_key_value_heads = 2, head_dim = 64, hidden_size = 128,
                               num_hidden_layers = 1, layer_types = ["minimax_m3_sparse"], mlp_layer_types = ["dense"])
    budget = config.index_topk_blocks * config.index_block_size
else:
    config = T._dsa_config(family, qk_nope_head_dim = 32, qk_rope_head_dim = 32, v_head_dim = 64)
    budget = config.index_topk
torch.manual_seed(0)
model = {"minimax_m3_vl": "MiniMaxM3VLForCausalLM", "deepseek_v32": "DeepseekV32ForCausalLM",
         "glm_moe_dsa": "GlmMoeDsaForCausalLM"}[family]
model = getattr(modeling, model)(config).to(device).eval()


def stat(kind):
    return gates.SPARSE_ATTN_STATS.get(kind, {}).get(family, 0)


def logits(seq, dtype = torch.float32):
    torch.manual_seed(1)
    ids = torch.randint(0, config.vocab_size, (2, seq), device = device)
    with torch.no_grad():
        return model.to(dtype)(input_ids = ids).logits.float()


# The first CPU forward can differ from later ones in the last bits (about 6e-5 on MiniMax with 4 threads, even
# unpatched); warm it up here so the bitwise check compares the skip path with the reference.
logits(budget)
# Within the budget: skip engaged, bitwise equal to the skip-off (upstream indexer) run.
before = stat("skip_full")
on = logits(budget)
assert stat("skip_full") > before, gates.SPARSE_ATTN_STATS
os.environ["UNSLOTH_SPARSE_ATTN_SKIP_FULL"] = "0"
off = logits(budget)
assert torch.equal(on, off), (on - off).abs().max()
# UNSLOTH_SPARSE_ATTN_FLEX=0 keeps the skip.
os.environ["UNSLOTH_SPARSE_ATTN_SKIP_FULL"] = "1"
os.environ["UNSLOTH_SPARSE_ATTN_FLEX"] = "0"
before = stat("skip_full")
logits(budget)
assert stat("skip_full") > before
os.environ["UNSLOTH_SPARSE_ATTN_FLEX"] = "1"

# Above the budget: chunked indexer, and flex where the device allows.
os.environ["UNSLOTH_SPARSE_INDEXER_Q_CHUNK"] = "64"
os.environ["UNSLOTH_SPARSE_ATTN_DSA_MIN_KEYS"] = str(budget + 1)
before_chunk = stat("indexer_chunked")
before_skip = stat("skip_full")
logits(2 * budget)
assert stat("indexer_chunked") > before_chunk, gates.SPARSE_ATTN_STATS
assert stat("skip_full") == before_skip
if cuda:
    # UNSLOTH_SPARSE_ATTN_SKIP_FULL=0 keeps flex.
    os.environ["UNSLOTH_SPARSE_ATTN_SKIP_FULL"] = "0"
    before = stat("flex")
    logits(2 * budget, torch.bfloat16)
    assert stat("flex") > before, gates.SPARSE_ATTN_STATS
print("COMPOSE_OK", order, family, "cuda" if cuda else "cpu", dict(gates.SPARSE_ATTN_STATS))
