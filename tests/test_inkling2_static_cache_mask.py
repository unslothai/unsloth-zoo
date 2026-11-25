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

"""generate with a StaticCache on hybrid_sliding models past the sliding window, on a tiny Inkling 2 Loop CPU fake,
after the real startup (import unsloth, or every registered temporary patch in order) in a fresh process."""

import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("transformers.models.inkling2_loop.modeling_inkling2_loop")
import transformers.masking_utils as masking_utils  # noqa: E402

SKIP = 77

_CHILD = textwrap.dedent("""
    import sys
    startup = sys.argv[1]
    if startup.startswith("unsloth"):
        try:
            import unsloth
        except ImportError:
            sys.exit(%d)
        if getattr(unsloth, "__file__", None) is None:  # a bare source directory on sys.path, not the package
            sys.exit(%d)
        # The startup under test really ran: the CPU backend imports unsloth_zoo.cpu, the accelerator one does not.
        assert ("unsloth_zoo.cpu" in sys.modules) == (startup == "unsloth_cpu"), startup
    else:
        # The GPU startup order without a GPU: every registered temporary patch, as unsloth runs them.
        from unsloth_zoo.temporary_patches.common import TEMPORARY_PATCHES
        import unsloth_zoo.temporary_patches  # noqa: F401
        for patch in TEMPORARY_PATCHES:
            try:
                patch()
            except Exception:
                pass
    import torch
    import transformers.masking_utils as masking_utils
    from transformers.cache_utils import DynamicCache, StaticCache
    from transformers.models.inkling2_loop import modeling_inkling2_loop as modeling
    from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig

    torch.set_num_threads(1)
    config = Inkling2LoopTextConfig(
        vocab_size = 64, unpadded_vocab_size = 64, hidden_size = 32, num_hidden_layers = 6, num_attention_heads = 2,
        num_key_value_heads = 1, head_dim = 16, swa_num_attention_heads = 2, swa_num_key_value_heads = 1,
        swa_head_dim = 16, sliding_window_size = 4, rel_extent = 16, d_rel = 4, moe_intermediate_size = 16,
        n_routed_experts = 4, num_experts_per_tok = 2, n_layers_per_attn_res_block = 2, loop_layer_index_start = 2,
        loop_layer_index_end = 4, loop_num_passes = 3, loop_gate_rank = 4, eos_token_id = None, pad_token_id = 0,
        layer_types = ["hybrid_sliding", "hybrid"] * 3,
    )
    torch.manual_seed(0)
    model = modeling.Inkling2LoopForCausalLM(config).eval()
    input_ids = torch.randint(1, 64, (1, 12), generator = torch.Generator().manual_seed(1))  # 12 > window 4
    mapping = masking_utils.LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING
    assert mapping["hybrid_sliding"]["sliding_attention"] is mapping["sliding_attention"], mapping["hybrid_sliding"]
    with torch.no_grad():
        dynamic = model.generate(input_ids, max_new_tokens = 4, do_sample = False, cache = DynamicCache(config = config))
        static = model.generate(
            input_ids, max_new_tokens = 4, do_sample = False, cache = StaticCache(config = config, max_cache_len = 20),
        )
    assert torch.equal(static, dynamic), (static, dynamic)
    print("OK")
""" % (SKIP, SKIP))


# unsloth_cpu: the native CPU backend (unsloth_zoo.cpu startup). unsloth_cuda: the accelerator startup, which on a
# driverless host is the UNSLOTH_ALLOW_CPU spoof. temporary_patches: the registered patch order without unsloth.
@pytest.mark.parametrize("startup", ["unsloth_cpu", "unsloth_cuda", "temporary_patches"])
def test_static_cache_past_window_matches_dynamic_after_startup(startup):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Only this checkout on PYTHONPATH: an inherited entry holding an unsloth source tree would shadow the package.
    env = dict(os.environ, PYTHONPATH = root)
    env.pop("UNSLOTH_ALLOW_CPU", None)
    env.pop("UNSLOTH_FORCE_CPU", None)
    if startup == "unsloth_cpu":
        env["UNSLOTH_FORCE_CPU"] = "1"
    elif startup == "unsloth_cuda":
        env["UNSLOTH_ALLOW_CPU"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", _CHILD, startup], cwd = root, env = env, capture_output = True, text = True,
        timeout = 900,
    )
    if result.returncode == SKIP:
        pytest.skip("unsloth is not installed")
    assert result.returncode == 0 and "OK" in result.stdout, result.stdout[-3000:] + result.stderr[-6000:]


def _patch():
    from unsloth_zoo.temporary_patches.hybrid_sliding_masks import patch_hybrid_sliding_generate_masks
    patch_hybrid_sliding_generate_masks()


def test_wrapped_public_builders_map_to_the_original_sliding_builder(monkeypatch):
    mapping = masking_utils.LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING
    original_causal, original_sliding = mapping["full_attention"], mapping["sliding_attention"]
    # What Unsloth's mask patches leave behind: public names wrapped, the table holding the originals.
    monkeypatch.setattr(masking_utils, "create_causal_mask", lambda *args, **kwargs: original_causal(*args, **kwargs))
    monkeypatch.setattr(
        masking_utils, "create_sliding_window_causal_mask", lambda *args, **kwargs: original_sliding(*args, **kwargs),
    )
    monkeypatch.setitem(mapping["hybrid_sliding"], "sliding_attention", original_causal)
    _patch()
    assert mapping["hybrid_sliding"]["sliding_attention"] is original_sliding


def test_noop_when_transformers_maps_it(monkeypatch):
    entry = masking_utils.LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING["hybrid_sliding"]
    sentinel = lambda **kwargs: None  # noqa: E731
    monkeypatch.setitem(entry, "sliding_attention", sentinel)
    _patch()
    assert entry["sliding_attention"] is sentinel
    sliding = masking_utils.LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING["sliding_attention"]
    monkeypatch.setitem(entry, "sliding_attention", sliding)
    _patch()
    assert entry["sliding_attention"] is sliding
