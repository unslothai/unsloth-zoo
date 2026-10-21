# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""UMT5 on transformers 5: untied logits unscaled, model_type-less configs load, attention layers compile."""

import ast
import json
import os
import subprocess
import sys

import pytest
import torch

from transformers import AutoConfig, UMT5Config, UMT5ForConditionalGeneration

from unsloth_zoo.temporary_patches.t5 import (
    patch_config_model_type_from_architectures,
    patch_UMT5_scale_decoder_outputs,
)

for _ in range(2):
    patch_UMT5_scale_decoder_outputs()
    patch_config_model_type_from_architectures()

TINY = dict(vocab_size = 128, d_model = 32, d_kv = 8, d_ff = 64, num_layers = 2, num_heads = 4, decoder_start_token_id = 0)


def _logit_ratio(model):
    ids = torch.randint(2, 128, (2, 8), generator = torch.Generator().manual_seed(0))
    with torch.no_grad():
        out = model(input_ids = ids, decoder_input_ids = ids, output_hidden_states = True)
        unscaled = model.lm_head(out.decoder_hidden_states[-1])
    return (out.logits / unscaled).median().item()


@pytest.mark.parametrize("tie, expected", [(False, 1.0), (None, 32**-0.5)])
def test_decoder_output_scaling_follows_checkpoint_tie_flag(tie, expected):
    kwargs = {} if tie is None else {"tie_word_embeddings": tie}
    torch.manual_seed(0)
    model = UMT5ForConditionalGeneration(UMT5Config(**TINY, **kwargs)).eval()
    assert _logit_ratio(model) == pytest.approx(expected, rel = 1e-4)


def test_untied_flag_survives_save_and_reload(tmp_path):
    torch.manual_seed(0)
    UMT5ForConditionalGeneration(UMT5Config(**TINY, tie_word_embeddings = False)).save_pretrained(tmp_path)
    model = UMT5ForConditionalGeneration.from_pretrained(tmp_path).eval()
    assert getattr(model.config, "scale_decoder_outputs", not model.config.tie_word_embeddings) is False
    assert _logit_ratio(model) == pytest.approx(1.0, rel = 1e-4)


def test_model_type_inferred_from_architectures(tmp_path):
    config = UMT5Config(**TINY).to_dict()
    config.pop("model_type", None)
    config["architectures"] = ["UMT5ForConditionalGeneration"]
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert type(AutoConfig.from_pretrained(tmp_path)).__name__ == "UMT5Config"

    config["architectures"] = ["NotARealModelForAnything"]
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match = "model_type"):
        AutoConfig.from_pretrained(tmp_path)


_COMPILE_CHILD = r"""
import contextlib, io, json, os
os.environ["UNSLOTH_COMPILE_LOCATION"] = "unsloth_compiled_cache"
from unsloth_zoo.temporary_patches.t5 import patch_UMT5_scale_decoder_outputs
patch_UMT5_scale_decoder_outputs()
from unsloth_zoo.compiler import unsloth_compile_transformers
with contextlib.redirect_stdout(io.StringIO()):
    unsloth_compile_transformers(
        model_type = "umt5", fast_lora_forwards = False, fullgraph = True,
        import_from_cache = False, disable = False, supports_sdpa = [None],
    )
print("@@@" + json.dumps(open("unsloth_compiled_cache/unsloth_compiled_module_umt5.py", encoding = "utf-8").read()))
"""


def test_compiled_umt5_does_not_inline_disabled_attention(tmp_path):
    env = dict(os.environ, UNSLOTH_ALLOW_CPU = "1", UNSLOTH_ZOO_DISABLE_GPU_INIT = "1", CUDA_VISIBLE_DEVICES = "")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join([repo_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.pop("UNSLOTH_COMPILE_DISABLE", None)
    env.pop("UNSLOTH_FULLGRAPH", None)
    result = subprocess.run(
        [sys.executable, "-c", _COMPILE_CHILD], cwd = tmp_path, env = env, capture_output = True, text = True, timeout = 1800
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    generated = json.loads(next(line[3:] for line in result.stdout.splitlines() if line.startswith("@@@")))
    tree = ast.parse(generated)
    decorators = {
        node.name: " ".join(ast.get_source_segment(generated, d) for d in node.decorator_list)
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    assert "torch_compiler_disable_unless_decode" in decorators["UMT5Attention_forward"]
    for layer in ("UMT5LayerSelfAttention_forward", "UMT5LayerCrossAttention_forward"):
        assert "fullgraph = True" not in decorators.get(layer, ""), decorators.get(layer)
    if UMT5Config(tie_word_embeddings = False).tie_word_embeddings:
        assert 'getattr(self.config, "scale_decoder_outputs"' in generated
