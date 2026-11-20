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

"""Checks skip and flex patches compose per install order, one process per order; each family is
checked alone, its failure reported for that family only, env restored."""

import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_sparse_attention_flex as T  # noqa: E402
from unsloth_zoo.flex_attention import sparse as gates  # noqa: E402
from unsloth_zoo.temporary_patches import sparse_attention as skip  # noqa: E402
from unsloth_zoo.temporary_patches import sparse_attention_flex as flex  # noqa: E402

FAMILIES = ("deepseek_v32", "glm_moe_dsa", "minimax_m3_vl")
order, families = sys.argv[1], tuple(sys.argv[2:]) or FAMILIES
install_skip = {
    "deepseek_v32": skip.patch_deepseek_v32_sparse_skip,
    "glm_moe_dsa": skip.patch_glm_moe_dsa_sparse_skip,
    "minimax_m3_vl": skip.patch_minimax_m3_sparse_skip,
}
install_flex = {"minimax_m3_vl": flex.patch_minimax_m3_sparse_flex, "deepseek_v32": flex.patch_dsa_sparse_flex,
                "glm_moe_dsa": flex.patch_dsa_sparse_flex}
skips = [install_skip[f] for f in families]
flexes = list(dict.fromkeys(install_flex[f] for f in families))
steps = skips + flexes if order == "skip_first" else flexes + skips
for fn in steps + steps:
    fn()

cuda = torch.cuda.is_available() and torch.version.hip is None and torch.cuda.get_device_capability() >= (8, 0)
device = "cuda" if cuda else "cpu"
if not cuda:
    # CPU: eager masks and two threads (bitwise checks compare within one setting).
    import torch._dynamo
    torch._dynamo.config.disable = True
    torch.set_num_threads(min(torch.get_num_threads(), 2))


def check(family):
    modeling = __import__(f"transformers.models.{family}.modeling_{family}", fromlist = ["x"])
    indexer_cls = getattr(modeling, {"deepseek_v32": "DeepseekV32Indexer", "glm_moe_dsa": "GlmMoeDsaIndexer",
                                     "minimax_m3_vl": "MiniMaxM3VLIndexer"}[family])
    assert getattr(indexer_cls.forward, "_unsloth_sparse_flex", False), "sparse flex not installed"
    key = f"{family}.indexer"
    assert skip._is_transformers_function(skip._REFERENCE[key], family), "_REFERENCE is not upstream"
    assert indexer_cls._unsloth_original_forward is skip._REFERENCE[key]
    assert indexer_cls._unsloth_inner_forward is not skip._REFERENCE[key], "skip not composed"

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

    # Warm up: the first CPU forward can differ in the last bits even unpatched.
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


start_env = {k: v for k, v in os.environ.items() if k.startswith("UNSLOTH_SPARSE_")}
for family in families:
    for k in [k for k in os.environ if k.startswith("UNSLOTH_SPARSE_")]:
        del os.environ[k]
    os.environ.update(start_env)
    try:
        check(family)
    except BaseException:
        print("COMPOSE_FAIL", order, family, flush = True)
        traceback.print_exc(file = sys.stdout)
    else:
        print("COMPOSE_OK", order, family, device, flush = True)
