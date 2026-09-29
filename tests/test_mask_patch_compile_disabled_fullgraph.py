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

import os
import subprocess
import sys
import textwrap


def test_mask_builder_stays_traceable_under_compile_disable():
    """UNSLOTH_COMPILE_DISABLE=1 must not wrap the builders in torch.compiler.disable (fullgraph user compiles break)."""
    code = textwrap.dedent(
        """
        import os, torch
        os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
        from unsloth_zoo.temporary_patches.misc import patch_transformers_masks
        patch_transformers_masks()
        import transformers.masking_utils as mu
        from transformers import LlamaConfig
        config = LlamaConfig(hidden_size = 16, num_attention_heads = 2, num_key_value_heads = 2, num_hidden_layers = 1)
        config._attn_implementation = "sdpa"
        embeds = torch.zeros(1, 4, 16)
        position_ids = torch.arange(4)[None]

        import inspect
        original = getattr(mu, "_unsloth_original_create_causal_mask", mu.create_causal_mask)
        takes = inspect.signature(original).parameters
        extra = {"cache_position": torch.arange(4)} if "cache_position" in takes else {}
        embeds_name = "inputs_embeds" if "inputs_embeds" in takes else "input_embeds"

        def build(embeds, position_ids):
            return mu.create_causal_mask(
                config = config, attention_mask = torch.tensor([[0, 1, 1, 1]]),
                past_key_values = None, position_ids = position_ids, **{embeds_name: embeds}, **extra,
            )

        eager = build(embeds, position_ids)
        compiled = torch.compile(build, fullgraph = True, backend = "eager")(embeds, position_ids)
        assert torch.equal(eager, compiled)
        print("OK")
        """
    )
    env = dict(os.environ, UNSLOTH_COMPILE_DISABLE = "1", UNSLOTH_IS_PRESENT = "1")
    result = subprocess.run([sys.executable, "-c", code], capture_output = True, text = True, env = env, timeout = 600)
    assert result.returncode == 0 and "OK" in result.stdout, result.stderr[-2000:]
