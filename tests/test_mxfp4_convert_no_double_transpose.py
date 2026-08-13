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

"""Regression test for PR #826.

Making convert_moe_packed_tensors survive a missing _cpu variant is not enough on its
own: when convert_moe_packed_tensors_cpu is absent, the base bound in saving_utils is
stock transformers' convert_moe_packed_tensors, which already returns the transposed
GPT-OSS layout (`return out.transpose(1, 2).contiguous()`). saving_utils then applies its
OWN .transpose(1, 2), so with the stock base the exported mxfp4 weight is DOUBLE
transposed (dims 1 and 2 swapped) and silently corrupted.

This test runs the real _merge_and_overwrite_lora_mxfp4 converter on a tiny fake mxfp4
file in two import orderings and asserts they produce byte-identical output:

  * without the Unsloth mxfp4 patch  -> _cpu absent, base is stock (self-transposing)
  * with the Unsloth mxfp4 patch     -> _cpu present, base is no-self-transpose

Before the fix the first ordering yields shape (E, D, G*B*2) and the second (E, G*B*2, D):
they differ by a transpose. After the fix both yield the correct (E, G*B*2, D) layout.
"""
import os
import subprocess
import sys
import tempfile
import textwrap

import pytest


_RUNNER = textwrap.dedent(
    """
    import os, sys, tempfile
    from collections import defaultdict
    import torch
    from safetensors.torch import save_file, safe_open

    mode, outfile = sys.argv[1], sys.argv[2]

    torch.manual_seed(0)
    E, D, G, B = 2, 6, 4, 8            # dequant last dim = G*B*2 = 64
    blocks = torch.randint(0, 256, (E, D, G, B), dtype=torch.uint8)
    scales = torch.randint(120, 135, (E, D, G), dtype=torch.uint8)

    save_dir = tempfile.mkdtemp(prefix="mxfp4regr_")
    fname = "model-00001-of-00001.safetensors"
    kb = "model.layers.0.mlp.experts.gate_up_proj"
    save_file({kb + "_blocks": blocks, kb + "_scales": scales},
              os.path.join(save_dir, fname), metadata={"format": "pt"})

    if mode == "patched":
        from unsloth_zoo.temporary_patches.mxfp4 import patch_convert_moe_packed_tensors
        patch_convert_moe_packed_tensors()

    import unsloth_zoo.saving_utils as sv
    sv._merge_and_overwrite_lora_mxfp4(
        save_directory=save_dir,
        filename=fname,
        lora_weights=defaultdict(lambda: None),
        output_dtype=torch.bfloat16,
        model_class_name="GptOssForCausalLM",
    )
    with safe_open(os.path.join(save_dir, fname), framework="pt", device="cpu") as f:
        W = f.get_tensor(kb)
    torch.save(W.float().cpu(), outfile)
    print("SHAPE", tuple(W.shape))
    """
)


def _run(mode, outfile):
    env = dict(os.environ)
    # Force CPU-friendly behaviour; the correctness check is device-independent.
    r = subprocess.run(
        [sys.executable, "-c", _RUNNER, mode, outfile],
        capture_output=True, text=True, env=env,
    )
    assert r.returncode == 0, f"mode={mode} failed:\nstdout={r.stdout}\nstderr={r.stderr}"
    return r.stdout


def test_mxfp4_export_layout_matches_with_and_without_cpu_variant():
    # transformers.integrations.mxfp4 only exists from ~4.55 (GPT-OSS); on the older
    # transformers unsloth_zoo still supports (pin floor 4.51.3) the subprocess converter
    # below cannot run, so skip cleanly rather than error.
    pytest.importorskip("transformers.integrations.mxfp4")
    import torch
    with tempfile.TemporaryDirectory() as d:
        stock = os.path.join(d, "stock.pt")
        patched = os.path.join(d, "patched.pt")
        out_stock = _run("stock", stock)      # _cpu absent -> stock self-transposing base
        out_patched = _run("patched", patched)  # _cpu present -> Unsloth no-self-transpose base
        Ws = torch.load(stock)
        Wp = torch.load(patched)
        # The correct GPT-OSS layout is [E, G*B*2, D] == (2, 64, 6).
        assert tuple(Wp.shape) == (2, 64, 6), (out_patched, Wp.shape)
        assert tuple(Ws.shape) == (2, 64, 6), (
            "missing-_cpu export is double-transposed (wrong layout)", out_stock, Ws.shape,
        )
        assert torch.equal(Ws, Wp), "missing-_cpu export does not match the patched export"
