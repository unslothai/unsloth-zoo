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

"""patch_transformers_masks replaces masking_utils.create_causal_mask for the whole process. With a cache
(every eval forward and generate) it materialised a boolean mask where transformers returns None, so a model
loaded with plain transformers lost maskless causal SDPA (flash) and its logits moved by a bf16 rounding step.
Once require_model_marker() is on, a model Unsloth neither loaded nor trains gets transformers' own mask."""

import json
import os
import subprocess
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCRIPT = textwrap.dedent(
    """
    import json, sys
    import torch
    import torch.nn.functional as F
    from transformers import LlamaConfig, LlamaForCausalLM

    def build():
        torch.manual_seed(0)
        config = LlamaConfig(vocab_size = 128, hidden_size = 32, intermediate_size = 64, num_hidden_layers = 2,
                             num_attention_heads = 4, num_key_value_heads = 2, max_position_embeddings = 64,
                             attn_implementation = "sdpa")
        return LlamaForCausalLM(config).eval()

    seen = []
    inner = F.scaled_dot_product_attention
    def spy(*args, **kwargs):
        mask = kwargs.get("attn_mask", args[3] if len(args) > 3 else None)
        seen.append(mask is None)
        return inner(*args, **kwargs)
    F.scaled_dot_product_attention = spy

    ids = torch.randint(0, 128, (2, 12), generator = torch.Generator().manual_seed(1))
    def run(model):
        seen.clear()
        with torch.no_grad():
            logits = model(input_ids = ids, use_cache = True).logits
        return logits, all(seen)

    ref, ref_maskless = run(build())

    import transformers.masking_utils as masking_utils
    import unsloth_zoo.temporary_patches.misc as misc
    from unsloth_zoo import model_marker

    # Stands in for the compiled builder on any device: transformers drops the causal mask only when
    # not tracing, so the compiled create_causal_mask always materialises it.
    def compiled_like(function):
        def builder(*args, **kwargs):
            skip = masking_utils._ignore_causal_mask_sdpa
            masking_utils._ignore_causal_mask_sdpa = lambda *a, **k: False
            try:
                return function(*args, **kwargs)
            finally:
                masking_utils._ignore_causal_mask_sdpa = skip
        return builder
    misc.compile_mask_builder = compiled_like
    patch_transformers_masks = misc.patch_transformers_masks
    patch_transformers_masks()
    # A modeling module imported after the patch (the order under `import unsloth`) binds the wrapper.
    import transformers.models.llama.modeling_llama as modeling_llama
    modeling_llama.create_causal_mask = masking_utils.create_causal_mask
    if sys.argv[1] == "marker":
        model_marker.require_model_marker()
    plain, marked = build(), build()
    model_marker.mark_model(marked)
    out = {"ref_maskless": ref_maskless}
    for name, model in (("plain", plain), ("marked", marked)):
        logits, maskless = run(model)
        out[name] = dict(maskless = maskless, equal = bool(torch.equal(logits, ref)), diff = (logits - ref).abs().max().item())
    print("RESULT " + json.dumps(out))
    """
)


def _run(arm, compile_disable = None):
    env = dict(os.environ, PYTHONPATH = ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""), CUDA_VISIBLE_DEVICES = "")
    # CI sets UNSLOTH_COMPILE_DISABLE=1, under which the patch keeps transformers' own builder and never
    # reaches the compiled one stubbed above: state it per test instead of inheriting it.
    env.pop("UNSLOTH_COMPILE_DISABLE", None)
    if compile_disable is not None:
        env["UNSLOTH_COMPILE_DISABLE"] = compile_disable
    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT, arm], capture_output = True, text = True, env = env, cwd = ROOT, timeout = 600,
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    assert lines, proc.stdout[-2000:] + proc.stderr[-4000:]
    return json.loads(lines[-1][len("RESULT "):])


def test_plain_model_keeps_transformers_mask():
    out = _run("marker")
    assert out["ref_maskless"]
    # The plain model runs exactly as without the patch.
    assert out["plain"]["maskless"]
    assert out["plain"]["equal"], out
    # A marked model keeps the patched mask, as every model did before.
    assert not out["marked"]["maskless"]


def test_without_marker_requirement_every_model_gets_the_patched_mask():
    out = _run("legacy")
    assert not out["plain"]["maskless"]
    assert not out["marked"]["maskless"]


def test_compile_disabled_keeps_transformers_mask_for_every_model():
    out = _run("legacy", compile_disable = "1")
    for name in ("plain", "marked"):
        assert out[name]["maskless"], out
        assert out[name]["equal"], out
